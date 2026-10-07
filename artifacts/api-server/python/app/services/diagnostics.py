"""System diagnostics and honest processing device capability reporting."""

from app.services.hardware_manager import get_hardware_status


def processing_device_diagnostics() -> dict:
    """Report hardware and acceleration honestly.

    Preserves backward-compatible keys while adding full host hardware inspection.
    """
    status = get_hardware_status()
    return {
        "device": status["device"],
        "acceleration_status": status["acceleration_status"],
        "hardware_gpu_available": status["hardware_gpu_detected"],
        "opencv_cuda_devices": status["opencv_cuda_devices"],
        "gpu_name": status["hardware_gpu_name"],
        "cuda_version": status["cuda_driver_version"],
        "vram_total_mb": status["vram_total_mb"],
        "vram_free_mb": status["vram_free_mb"],
        "pytorch_cuda_available": status["pytorch_cuda_available"],
        "status_display": status["status_display"],
        "bounded_workers": status["bounded_worker_concurrency"],
        "system_ram_mb": status["system_ram_total_mb"],
        "measured_speedup": status["measured_speedup"],
        "note": status["note"],
    }
