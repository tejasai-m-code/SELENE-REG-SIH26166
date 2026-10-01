import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.services.multi_registration import run_multi_registration
from app.services.registration_cache import RegistrationCache

def create_synthetic_image():
    np.random.seed(42)
    img = np.zeros((300, 300), dtype=np.float32)
    y, x = np.mgrid[0:300, 0:300]
    img += np.sin(x / 20.0) * 50
    img += np.cos(y / 15.0) * 50
    cv2.circle(img, (150, 150), 40, 200, -1)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def benchmark():
    cache_dir = Path("cache_benchmark")
    
    img1 = create_synthetic_image()
    H = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32)
    img2 = cv2.warpPerspective(img1, H, (300, 300), flags=cv2.INTER_LINEAR)
    images = [img1, img2]
    
    settings = {
        "detector": "sift",
        "max_features": 500,
        "ratio": 0.72,
        "ransac_threshold": 3.0,
        "ecc_refinement": True
    }
    
    print("--- COLD RUN ---")
    t0 = time.perf_counter()
    res_cold = run_multi_registration(images, settings=settings, cache_dir=cache_dir)
    t_cold = time.perf_counter() - t0
    
    print(f"Total multi-registration runtime: {t_cold * 1000:.2f} ms")
    print(f"Execution Device: {res_cold.summary['processing_device']}")
    print(f"Cache Hits (Pairs): {res_cold.summary['cache_hits']}")
    print(f"Cache Misses (Pairs): {res_cold.summary['cache_misses']}")
    print("Stage timings (ms):")
    for stage, sec in res_cold.summary['stage_timings_seconds'].items():
        print(f"  {stage}: {sec * 1000:.2f} ms")
        
    print("\n--- WARM RUN (CACHE HIT) ---")
    t1 = time.perf_counter()
    res_warm = run_multi_registration(images, settings=settings, cache_dir=cache_dir)
    t_warm = time.perf_counter() - t1
    
    print(f"Total multi-registration runtime: {t_warm * 1000:.2f} ms")
    print(f"Execution Device: {res_warm.summary['processing_device']}")
    print(f"Cache Hits (Pairs): {res_warm.summary['cache_hits']}")
    print(f"Cache Misses (Pairs): {res_warm.summary['cache_misses']}")
    print("Stage timings (ms):")
    for stage, sec in res_warm.summary['stage_timings_seconds'].items():
        print(f"  {stage}: {sec * 1000:.2f} ms")
        
    print("\n--- PROVENANCE RECORD ---")
    print(f"Job ID: JOB_{int(time.time())}")
    print(f"Image Dimensions: {img1.shape}")
    print(f"DType: {img1.dtype}")
    print(f"Detector: {settings['detector']}")
    print(f"Geometric Model: Homography (via RANSAC/MAGSAC++)")
    print(f"Subpixel Refinement: {'ECC' if settings['ecc_refinement'] else 'None'}")
    
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    benchmark()
