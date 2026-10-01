"""Small, dependency-free processing diagnostics for the local prototype."""

import cv2


def processing_device_diagnostics() -> dict:
    """Report only acceleration that OpenCV can actually use at runtime."""
    try:
        cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except cv2.error:
        cuda_devices = 0
    if cuda_devices > 0:
        return {"device": "CPU", "opencv_cuda_devices": cuda_devices, "note": "CUDA hardware is visible, but this OpenCV classical registration pipeline currently uses its CPU implementation."}
    return {"device": "CPU", "opencv_cuda_devices": 0, "note": "OpenCV CUDA is not available to this process; CPU fallback is in use."}
