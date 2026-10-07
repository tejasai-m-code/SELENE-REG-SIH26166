"""Hybrid CPU-GPU Workload Scheduler & 4 GB VRAM Memory Safety Engine.

Implements SIH 26166 Master Engineering Prompt §23 & §25:
- Intelligent task routing: CPU for I/O, decoding, metadata, RANSAC, graph;
  GPU for tensor convolutions, descriptor matching, 2D FFT phase correlation,
  and grid sampling perspective warping.
- Strict VRAM safety envelope for 4 GB GPUs (NVIDIA RTX 3050):
  - Dynamic free VRAM headroom check (minimum 512 MB safety margin).
  - Batch chunking of large descriptor distance matrices.
  - Transparent, non-crashing CPU fallback on VRAM exhaustion.
  - Explicit CUDA stream synchronization and memory allocator cleanup.
- Verifiable, non-fabricated telemetry:
  - Exact counts of CPU vs GPU tasks executed.
  - Real execution percentages and timings.
"""

from __future__ import annotations

import gc
import logging
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from app.services.gpu_accelerator import (
    clear_gpu_cache,
    get_free_vram_mb,
    get_gpu_info,
    is_cuda_available,
    match_descriptors_gpu,
    phase_correlation_gpu,
    warp_perspective_gpu,
)

logger = logging.getLogger(__name__)

# Minimum free VRAM headroom in MB required before dispatching heavy GPU jobs
DEFAULT_VRAM_SAFETY_MARGIN_MB = 512.0

# Threshold above which descriptor distance matrices are chunked to prevent VRAM spikes
DEFAULT_MAX_DESCRIPTOR_PRODUCT = 8_000_000  # e.g., 2000 x 4000 pairs


class TaskCategory:
    """Task category definitions with hardware affinity."""

    # Inherently CPU-bound tasks
    FILE_IO = "FILE_IO"
    DECODING = "DECODING"
    METADATA_PARSING = "METADATA_PARSING"
    RANSAC_ESTIMATION = "RANSAC_ESTIMATION"
    GRAPH_ORCHESTRATION = "GRAPH_ORCHESTRATION"
    SPATIAL_METRICS = "SPATIAL_METRICS"

    # GPU-accelerated tasks with CPU fallback
    DESCRIPTOR_MATCHING = "DESCRIPTOR_MATCHING"
    PHASE_CORRELATION = "PHASE_CORRELATION"
    PERSPECTIVE_WARPING = "PERSPECTIVE_WARPING"
    SPATIAL_FILTERING = "SPATIAL_FILTERING"
    TENSOR_PREPROCESSING = "TENSOR_PREPROCESSING"


class HybridScheduler:
    """Resource-aware workload scheduler balancing CPU multi-core and GPU CUDA execution."""

    def __init__(
        self,
        vram_safety_margin_mb: float = DEFAULT_VRAM_SAFETY_MARGIN_MB,
        max_descriptor_product: int = DEFAULT_MAX_DESCRIPTOR_PRODUCT,
    ) -> None:
        self.vram_safety_margin_mb = vram_safety_margin_mb
        self.max_descriptor_product = max_descriptor_product
        self.task_history: list[dict[str, Any]] = []
        self.gpu_task_count: int = 0
        self.cpu_task_count: int = 0
        self.fallback_count: int = 0

    def should_use_gpu(
        self,
        task_category: str,
        estimated_vram_mb: float = 50.0,
    ) -> tuple[bool, str]:
        """Determine if a task should execute on GPU based on category, CUDA, and free VRAM."""
        if not is_cuda_available():
            return False, "CUDA_UNAVAILABLE"

        cpu_affinity_categories = {
            TaskCategory.FILE_IO,
            TaskCategory.DECODING,
            TaskCategory.METADATA_PARSING,
            TaskCategory.RANSAC_ESTIMATION,
            TaskCategory.GRAPH_ORCHESTRATION,
            TaskCategory.SPATIAL_METRICS,
        }
        if task_category in cpu_affinity_categories:
            return False, "CPU_AFFINITY_TASK"

        free_vram = get_free_vram_mb()
        needed_vram = estimated_vram_mb + self.vram_safety_margin_mb
        if free_vram < needed_vram:
            logger.info(
                "Task %s routed to CPU: free VRAM %.1f MB < required %.1f MB (headroom: %.1f MB)",
                task_category, free_vram, needed_vram, self.vram_safety_margin_mb,
            )
            return False, f"INSUFFICIENT_VRAM_FREE_{free_vram:.0f}MB"

        return True, "GPU_ACCOMMODATED"

    def dispatch(
        self,
        task_category: str,
        task_name: str,
        fn_gpu: Callable[..., Any],
        fn_cpu: Callable[..., Any],
        estimated_vram_mb: float = 50.0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intelligently dispatch a unit of work to GPU or CPU with automatic fallback."""
        use_gpu, routing_reason = self.should_use_gpu(task_category, estimated_vram_mb)

        if use_gpu:
            t0 = perf_counter()
            try:
                result = fn_gpu(*args, **kwargs)
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (perf_counter() - t0) * 1000.0

                self.gpu_task_count += 1
                self._record_task(task_name, task_category, "GPU_CUDA", elapsed_ms, "NORMAL_DISPATCH")
                return result

            except Exception as exc:
                elapsed_ms = (perf_counter() - t0) * 1000.0
                logger.warning("GPU execution failed for %s (%s); triggering CPU fallback: %s", task_name, task_category, exc)
                clear_gpu_cache()
                gc.collect()

                # Fallback to CPU execution faithfully
                t_cpu = perf_counter()
                result = fn_cpu(*args, **kwargs)
                cpu_elapsed_ms = (perf_counter() - t_cpu) * 1000.0

                self.cpu_task_count += 1
                self.fallback_count += 1
                self._record_task(task_name, task_category, "CPU_FALLBACK", cpu_elapsed_ms, f"FALLBACK_AFTER_ERROR: {exc}")
                return result

        else:
            t0 = perf_counter()
            result = fn_cpu(*args, **kwargs)
            elapsed_ms = (perf_counter() - t0) * 1000.0

            self.cpu_task_count += 1
            if routing_reason.startswith("INSUFFICIENT_VRAM"):
                self.fallback_count += 1
                self._record_task(task_name, task_category, "CPU_FALLBACK", elapsed_ms, routing_reason)
            else:
                self._record_task(task_name, task_category, "CPU", elapsed_ms, routing_reason)
            return result

    def match_descriptors(
        self,
        descriptors_s: np.ndarray,
        descriptors_r: np.ndarray,
        ratio: float = 0.72,
        mutual_check: bool = True,
        metric: str = "auto",
    ) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
        """Schedule descriptor matching with batch chunking if descriptor sizes exceed VRAM safety threshold."""
        n_s = len(descriptors_s)
        n_r = len(descriptors_r)
        product = n_s * n_r

        # Estimate required VRAM for distance matrix in MB
        # Float32 matrix: 4 bytes per pair. Uint8 Hamming: 1 byte per pair.
        bytes_per_pair = 1 if (metric == "hamming" or (metric == "auto" and descriptors_s.dtype in (np.uint8, np.int8))) else 4
        estimated_mb = (product * bytes_per_pair) / (1024 * 1024)

        # If product exceeds chunking threshold and CUDA is available, execute chunked matching
        if product > self.max_descriptor_product and is_cuda_available():
            return self._chunked_gpu_matching(descriptors_s, descriptors_r, ratio, mutual_check, metric)

        return self.dispatch(
            TaskCategory.DESCRIPTOR_MATCHING,
            "descriptor_matching",
            fn_gpu=lambda: match_descriptors_gpu(descriptors_s, descriptors_r, ratio, mutual_check, metric),
            fn_cpu=lambda: self._cpu_descriptor_matching(descriptors_s, descriptors_r, ratio, mutual_check, metric),
            estimated_vram_mb=estimated_mb,
        )

    def phase_correlation(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> tuple[float, float, float]:
        """Schedule 2D Fourier phase correlation on GPU or CPU."""
        h, w = img1.shape[:2]
        # 2D FFT tensor allocations for 2 float32 complex images: ~ 4 * (h * w * 8) bytes
        estimated_mb = (h * w * 32) / (1024 * 1024)

        return self.dispatch(
            TaskCategory.PHASE_CORRELATION,
            "phase_correlation",
            fn_gpu=lambda: phase_correlation_gpu(img1, img2),
            fn_cpu=lambda: self._cpu_phase_correlation(img1, img2),
            estimated_vram_mb=estimated_mb,
        )

    def warp_perspective(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Schedule perspective warping on GPU grid sampler or CPU."""
        out_w, out_h = output_shape
        estimated_mb = (out_w * out_h * 16) / (1024 * 1024)

        return self.dispatch(
            TaskCategory.PERSPECTIVE_WARPING,
            "perspective_warping",
            fn_gpu=lambda: warp_perspective_gpu(image, homography, output_shape),
            fn_cpu=lambda: cv2.warpPerspective(image, homography, (out_w, out_h)),
            estimated_vram_mb=estimated_mb,
        )

    def _chunked_gpu_matching(
        self,
        descriptors_s: np.ndarray,
        descriptors_r: np.ndarray,
        ratio: float,
        mutual_check: bool,
        metric: str,
        chunk_size: int = 3000,
    ) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
        """Memory-safe chunked descriptor matching preventing VRAM exhaustion on 4 GB GPUs."""
        t0 = perf_counter()
        all_matches: list[tuple[int, int, float]] = []
        n_s = len(descriptors_s)

        for i in range(0, n_s, chunk_size):
            chunk_s = descriptors_s[i : i + chunk_size]
            matches_chunk, _ = match_descriptors_gpu(
                chunk_s, descriptors_r, ratio=ratio, mutual_check=mutual_check, metric=metric
            )
            # Offset query index by chunk start
            for q_idx, t_idx, d in matches_chunk:
                all_matches.append((q_idx + i, t_idx, d))

            clear_gpu_cache()

        # Deduplicate by reference index keeping lowest distance
        unique_by_ref: dict[int, tuple[int, int, float]] = {}
        for s_idx, r_idx, d in all_matches:
            if r_idx not in unique_by_ref or d < unique_by_ref[r_idx][2]:
                unique_by_ref[r_idx] = (s_idx, r_idx, d)

        final_matches = list(unique_by_ref.values())
        elapsed_ms = (perf_counter() - t0) * 1000.0

        self.gpu_task_count += 1
        self._record_task(
            "chunked_descriptor_matching",
            TaskCategory.DESCRIPTOR_MATCHING,
            "GPU_CUDA_CHUNKED",
            elapsed_ms,
            f"Chunked {n_s} features into {int(np.ceil(n_s / chunk_size))} passes",
        )

        return final_matches, {
            "acceleration": "GPU_CUDA_CHUNKED",
            "execution_time_ms": round(elapsed_ms, 2),
            "chunks": int(np.ceil(n_s / chunk_size)),
            "matches_retained": len(final_matches),
        }

    @staticmethod
    def _cpu_descriptor_matching(
        descriptors_s: np.ndarray,
        descriptors_r: np.ndarray,
        ratio: float,
        mutual_check: bool,
        metric: str,
    ) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
        from app.services.gpu_accelerator import _match_descriptors_cpu_fallback
        return _match_descriptors_cpu_fallback(descriptors_s, descriptors_r, ratio, mutual_check, metric)

    @staticmethod
    def _cpu_phase_correlation(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float, float]:
        shift, resp = cv2.phaseCorrelate(img1.astype(np.float32), img2.astype(np.float32))
        return float(shift[0]), float(shift[1]), float(resp)

    def _record_task(
        self,
        task_name: str,
        category: str,
        target: str,
        runtime_ms: float,
        notes: str,
    ) -> None:
        record = {
            "task_name": task_name,
            "category": category,
            "target": target,
            "runtime_ms": round(runtime_ms, 2),
            "notes": notes,
        }
        self.task_history.append(record)
        # Keep recent 100 records
        if len(self.task_history) > 100:
            self.task_history.pop(0)

    def get_telemetry(self) -> dict[str, Any]:
        """Return truthful execution breakdown between GPU and CPU."""
        total = self.gpu_task_count + self.cpu_task_count
        gpu_pct = round((self.gpu_task_count / total * 100.0), 1) if total > 0 else 0.0
        cpu_pct = round((self.cpu_task_count / total * 100.0), 1) if total > 0 else 100.0
        gpu_info = get_gpu_info()

        return {
            "total_tasks_scheduled": total,
            "gpu_tasks_executed": self.gpu_task_count,
            "cpu_tasks_executed": self.cpu_task_count,
            "fallbacks_count": self.fallback_count,
            "gpu_execution_percentage": gpu_pct,
            "cpu_execution_percentage": cpu_pct,
            "active_mode": "HYBRID (GPU + CPU)" if (self.gpu_task_count > 0 and self.cpu_task_count > 0) else "GPU ONLY" if self.gpu_task_count > 0 else "CPU FALLBACK",
            "cuda_active": is_cuda_available(),
            "gpu_device_name": gpu_info.get("device_name", "None"),
            "vram_allocated_mb": gpu_info.get("allocated_memory_mb", 0.0),
            "vram_reserved_mb": gpu_info.get("reserved_memory_mb", 0.0),
            "vram_free_mb": round(get_free_vram_mb(), 1),
            "vram_safety_margin_mb": self.vram_safety_margin_mb,
            "recent_tasks": self.task_history[-10:],
        }


# Global singleton scheduler instance
_GLOBAL_SCHEDULER: HybridScheduler | None = None


def get_global_scheduler() -> HybridScheduler:
    """Return the global HybridScheduler singleton."""
    global _GLOBAL_SCHEDULER
    if _GLOBAL_SCHEDULER is None:
        _GLOBAL_SCHEDULER = HybridScheduler()
    return _GLOBAL_SCHEDULER
