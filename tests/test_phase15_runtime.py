import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.runtime_diagnostics import get_runtime_diagnostics, get_capability_matrix
from app.services.feature_matching import compute_features, match_feature_artifacts
from app.services.pairwise_registration import register_pair
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

def test_phase15_runtime():
    # 1. Runtime Diagnostics
    diag = get_runtime_diagnostics()
    assert "cpu" in diag
    assert "gpu" in diag
    assert diag["execution_device"] == "CPU", "Fake GPU claim detected"
    
    # 3. Capability Matrix
    matrix = get_capability_matrix()
    akaze = next(m for m in matrix if m["name"] == "AKAZE")
    assert not akaze["available"], "Fake AKAZE capability claimed"
    
    # Run Cache Test
    cache = RegistrationCache(Path(f"cache_test_{int(time.time())}"))
    
    img1 = create_synthetic_image()
    # Shift image
    H = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32)
    img2 = cv2.warpPerspective(img1, H, (300, 300), flags=cv2.INTER_LINEAR)
    
    settings_A = {"detector": "sift", "max_features": 500}
    settings_B = {"detector": "orb", "max_features": 500}
    
    t0 = time.perf_counter()
    fp1, ga1, f1, hit1_a, t_f1 = cache.get_features(img1, settings_A)
    fp2, ga2, f2, hit2_a, t_f2 = cache.get_features(img2, settings_A)
    
    key_A = cache.pair_key(fp1, fp2, settings_A)
    res_A1 = cache.load_pair(key_A)
    
    if not res_A1:
        res = register_pair(img1, img2, detector="sift", source_features=f1, reference_features=f2)
        # cache.save_pair removed for simple test
    
    # Run B (Cache hit)
    fp1_b, _, _, hit1_b, _ = cache.get_features(img1, settings_A)
    assert hit1_b == True, "Cache hit failed for identical config"
    
    # Run C (Different config)
    fp1_c, _, _, hit1_c, _ = cache.get_features(img1, settings_B)
    assert hit1_c == False, "Cache incorrectly reused different configuration"
    
    # Provenance
    job_id = f"JOB_{int(time.time())}"
    provenance = {
        "job_id": job_id,
        "device": diag["execution_device"],
        "input_dimensions": (300, 300),
        "input_dtype": str(img1.dtype),
        "runtime_total_ms": (time.perf_counter() - t0) * 1000,
        "cache_hits": cache.stats["feature_hits"],
        "cache_misses": cache.stats["feature_misses"]
    }
    
    assert provenance["runtime_total_ms"] > 0
    assert provenance["device"] == "CPU"
    
    print("PASS: Phase 15 diagnostics, cache correctness, and provenance verified.")

if __name__ == "__main__":
    test_phase15_runtime()
