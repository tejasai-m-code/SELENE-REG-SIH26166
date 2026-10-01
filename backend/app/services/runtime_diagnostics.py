import platform
import os
import psutil
import cv2
import numpy as np
import scipy
from fastapi import __version__ as fastapi_version

def get_runtime_diagnostics():
    # Detect CPU
    cpu_info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True)
    }

    # Detect RAM safely
    mem = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2)
    }

    # OpenCV Build / GPU capabilities
    build_info = cv2.getBuildInformation()
    cuda_available = "CUDA" in build_info and "NVIDIA" in build_info
    opencl_available = "OpenCL" in build_info

    # Attempt to query OpenCL device directly
    gpu_devices = []
    execution_device = "CPU"
    acceleration_available = False
    
    try:
        if cv2.ocl.haveOpenCL():
            opencl_available = True
            cv2.ocl.setUseOpenCL(True)
            execution_device = "CPU" # default fallback
            # But wait, OpenCV usually executes mostly on CPU unless UMat is explicitly used.
            # We don't use UMat in our pipeline. So actual execution is CPU.
    except Exception:
        pass

    gpu_info = {
        "cuda_available": cuda_available,
        "opencl_available": opencl_available,
        "detected_devices": gpu_devices
    }

    # Check process info
    proc = psutil.Process()
    process_info = {
        "pid": proc.pid,
        "memory_rss_mb": round(proc.memory_info().rss / (1024**2), 2),
        "threads": proc.num_threads()
    }

    # Software Versions
    libraries = {
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "fastapi": fastapi_version
    }

    return {
        "execution_device": execution_device,
        "acceleration_available": acceleration_available,
        "acceleration_used": False,
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": gpu_info,
        "process": process_info,
        "libraries": libraries
    }

_akaze_available = None
def _check_akaze():
    global _akaze_available
    if _akaze_available is not None:
        return _akaze_available
    _akaze_available = hasattr(cv2, 'AKAZE_create') or (hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'AKAZE_create'))
    if _akaze_available:
        try:
            # Minimal sanity check
            if hasattr(cv2, 'AKAZE_create'):
                akaze = cv2.AKAZE_create()
            else:
                akaze = cv2.xfeatures2d.AKAZE_create()
            dummy = np.zeros((32, 32), dtype=np.uint8)
            akaze.detectAndCompute(dummy, None)
        except Exception:
            _akaze_available = False
    return _akaze_available

_superpoint_available = None
def _check_superpoint():
    global _superpoint_available
    if _superpoint_available is not None:
        return _superpoint_available
    try:
        from app.services.feature_matching import _compute_superpoint
        dummy = np.zeros((32, 32), dtype=np.uint8)
        _compute_superpoint(dummy, max_features=10)
        _superpoint_available = True
    except Exception as e:
        if "no features" in str(e).lower() or "insufficient" in str(e).lower():
            _superpoint_available = True
        else:
            _superpoint_available = False
    return _superpoint_available

_loftr_available = None
def _check_loftr():
    global _loftr_available
    if _loftr_available is not None:
        return _loftr_available
    try:
        from app.services.feature_matching import _match_loftr
        dummy = np.zeros((32, 32), dtype=np.uint8)
        _match_loftr(dummy, dummy)
        _loftr_available = True
    except Exception as e:
        if "insufficient" in str(e).lower() or "no matches" in str(e).lower():
            _loftr_available = True
        else:
            _loftr_available = False
    return _loftr_available

def get_capability_matrix():
    return [
        {"name": "SIFT", "implemented": True, "available": True, "fallback": None},
        {"name": "ORB", "implemented": True, "available": True, "fallback": None},
        {"name": "AKAZE", "implemented": True, "available": _check_akaze(), "fallback": "Unavailable in current OpenCV build"},
        {"name": "SuperPoint", "implemented": True, "available": _check_superpoint(), "fallback": "SIFT/ORB"},
        {"name": "LoFTR", "implemented": True, "available": _check_loftr(), "fallback": "SIFT/ORB"},
        {"name": "Taylor", "implemented": True, "available": True, "fallback": "Diverges on high-freq without pyramid"},
        {"name": "Lucas-Kanade", "implemented": True, "available": True, "fallback": None},
        {"name": "ECC", "implemented": True, "available": True, "fallback": None},
        {"name": "Phase Correlation", "implemented": True, "available": True, "fallback": None},
        {"name": "Local Peak", "implemented": True, "available": True, "fallback": None},
        {"name": "MAGSAC++", "implemented": True, "available": hasattr(cv2, 'USAC_MAGSAC'), "fallback": "RANSAC"},
        {"name": "RANSAC", "implemented": True, "available": True, "fallback": None},
        {"name": "Translation", "implemented": True, "available": True, "fallback": None},
        {"name": "Similarity", "implemented": True, "available": True, "fallback": None},
        {"name": "Affine", "implemented": True, "available": True, "fallback": None},
        {"name": "Homography", "implemented": True, "available": True, "fallback": None},
        {"name": "Weighted Pose Graph", "implemented": True, "available": True, "fallback": None}
    ]
