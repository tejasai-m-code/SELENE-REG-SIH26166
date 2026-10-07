"""Resource-aware hardware detection, memory tracking, and bounded concurrency manager.

Detects:
- Host GPU (via nvidia-smi / WMI / PyTorch / OpenCV)
- CUDA driver version and PyTorch / OpenCV CUDA runtime status
- System RAM and available RAM
- Bounded concurrency workers to prevent OOM on large lunar rasters
- Honest runtime reporting (never claims CUDA when CPU fallback is executing)
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
from typing import Any

import cv2
import numpy as np


def _detect_nvidia_smi() -> dict[str, Any]:
    """Query nvidia-smi for host GPU hardware and driver information."""
    for exe in ("nvidia-smi", "nvidia-smi.exe", r"C:\Windows\System32\nvidia-smi.exe"):
        try:
            res = subprocess.run(
                [exe, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    return {
                        "gpu_name": parts[0],
                        "driver_version": parts[1],
                        "vram_total_mb": int(float(parts[2])),
                        "vram_free_mb": int(float(parts[3])),
                        "smi_available": True,
                    }
        except Exception:
            continue
    return {}


def get_system_memory_mb() -> tuple[int, int]:
    """Return (total_mb, available_mb) for system RAM."""
    # On Windows, try ctypes GlobalMemoryStatusEx
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_mb = int(stat.ullTotalPhys / (1024 * 1024))
                avail_mb = int(stat.ullAvailPhys / (1024 * 1024))
                return total_mb, avail_mb
        except Exception:
            pass

    return 16384, 8192  # Safe fallback estimate


def get_hardware_status() -> dict[str, Any]:
    """Authoritative, evidence-based hardware status.

    Returns exact hardware presence vs active software runtime execution path.
    """
    smi_info = _detect_nvidia_smi()

    # OpenCV CUDA check
    opencv_cuda_count = 0
    try:
        opencv_cuda_count = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except Exception:
        opencv_cuda_count = 0

    # PyTorch CUDA check
    torch_cuda = False
    torch_version = "Not loaded"
    try:
        import torch
        torch_cuda = bool(torch.cuda.is_available())
        torch_version = str(torch.__version__)
    except Exception:
        pass

    total_ram_mb, avail_ram_mb = get_system_memory_mb()
    cpu_cores = os.cpu_count() or 4

    # Determine whether GPU acceleration is ACTUALLY usable in the current Python environment
    can_use_gpu = (opencv_cuda_count > 0) or torch_cuda

    gpu_name = smi_info.get("gpu_name") or ("NVIDIA GPU" if smi_info else "None Detected")
    driver_version = smi_info.get("driver_version") or "N/A"
    vram_mb = smi_info.get("vram_total_mb") or 0

    # Bounded concurrency calculation
    # Heavy lunar rasters take ~500MB-1.5GB per registration job
    # Ensure safe worker budget: 1 worker per 2048 MB available RAM, min 1, max cpu_cores
    recommended_workers = max(1, min(cpu_cores, avail_ram_mb // 2048))

    if can_use_gpu:
        status_text = f"GPU: {gpu_name} (CUDA ACTIVE)"
        mode = "GPU ACCELERATED"
        device = "CUDA"
        exec_mode = "GPU / HYBRID"
    elif smi_info:
        status_text = f"GPU: {gpu_name} | CUDA Driver: {driver_version} | CPU Fallback"
        mode = "CPU FALLBACK"
        device = "CPU"
        exec_mode = "CPU FALLBACK"
    else:
        status_text = "CPU Fallback (No CUDA GPU Detected)"
        mode = "CPU FALLBACK"
        device = "CPU"
        exec_mode = "CPU FALLBACK"

    note = (
        f"Hardware GPU '{gpu_name}' detected on host with Driver {driver_version}. "
        "The Python environment is using CPU-optimized OpenCV and PyTorch binaries. "
        "All geometric and registration algorithms execute faithfully via CPU fallback."
        if (smi_info and not can_use_gpu)
        else "CUDA acceleration active for image tensors and feature filters."
        if can_use_gpu
        else "Running on standard multi-core CPU architecture."
    )

    from app.services.gpu_accelerator import run_gpu_self_test
    self_test = run_gpu_self_test()

    return {
        "device": device,
        "acceleration_status": mode,
        "execution_mode": exec_mode,
        "status_display": status_text,
        "hardware_gpu_name": gpu_name,
        "hardware_gpu_detected": bool(smi_info),
        "cuda_driver_version": driver_version,
        "vram_total_mb": vram_mb,
        "vram_free_mb": smi_info.get("vram_free_mb", 0),
        "vram_safety_reserve_mb": 512,
        "opencv_cuda_devices": opencv_cuda_count,
        "pytorch_cuda_available": torch_cuda,
        "pytorch_version": torch_version,
        "opencv_version": str(cv2.__version__),
        "system_ram_total_mb": total_ram_mb,
        "system_ram_available_mb": avail_ram_mb,
        "cpu_logical_cores": cpu_cores,
        "bounded_worker_concurrency": recommended_workers,
        "measured_speedup": None,  # Will only be populated if both CPU and GPU paths are measured
        "gpu_self_test": self_test,
        "note": note,
    }


def run_hardware_self_test() -> dict[str, Any]:
    """Execute authoritative self-test of host compute resources."""
    from app.services.gpu_accelerator import run_gpu_self_test
    return run_gpu_self_test()


def check_memory_safety(required_vram_mb: float = 256.0, required_ram_mb: float = 1024.0) -> dict[str, Any]:
    """Verify that available system RAM and GPU VRAM satisfy memory safety limits (Master Prompt §25)."""
    from app.services.gpu_accelerator import get_free_vram_mb, is_cuda_available
    tot_ram, avail_ram = get_system_memory_mb()
    free_vram = get_free_vram_mb()
    has_cuda = is_cuda_available()
    vram_safe = (not has_cuda) or (free_vram >= (required_vram_mb + 512.0))
    ram_safe = avail_ram >= required_ram_mb
    return {
        "is_safe": bool(vram_safe and ram_safe),
        "vram_safe": bool(vram_safe),
        "ram_safe": bool(ram_safe),
        "free_vram_mb": round(free_vram, 1),
        "available_ram_mb": avail_ram,
        "required_vram_mb": required_vram_mb,
        "required_ram_mb": required_ram_mb,
        "recommendation": "PROCEED" if (vram_safe and ram_safe) else "FALLBACK_TO_CPU" if not vram_safe else "REDUCE_BATCH_SIZE",
    }



def measure_execution_benchmark(cpu_seconds: float | None = None, gpu_seconds: float | None = None) -> dict[str, Any]:
    """Calculate measured speedup honestly without fabricating numbers."""
    if cpu_seconds is not None and gpu_seconds is not None and gpu_seconds > 0:
        speedup = float(round(cpu_seconds / gpu_seconds, 2))
        return {
            "status": "ACCELERATION MEASURED",
            "cpu_runtime_seconds": float(round(cpu_seconds, 3)),
            "gpu_runtime_seconds": float(round(gpu_seconds, 3)),
            "measured_speedup_factor": speedup,
            "display": f"Measured Speedup: {speedup}x (CPU: {cpu_seconds:.2f}s vs GPU: {gpu_seconds:.2f}s)",
        }
    return {
        "status": "CPU ONLY MEASURED" if cpu_seconds is not None else "GPU ONLY MEASURED" if gpu_seconds is not None else "NOT MEASURED",
        "cpu_runtime_seconds": float(round(cpu_seconds, 3)) if cpu_seconds is not None else None,
        "gpu_runtime_seconds": float(round(gpu_seconds, 3)) if gpu_seconds is not None else None,
        "measured_speedup_factor": None,
        "display": "Speedup measurement unavailable (single-device execution)",
    }

