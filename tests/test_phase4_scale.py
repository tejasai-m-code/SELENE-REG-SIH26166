import numpy as np
import cv2
from app.services.scale import (
    build_image_pyramid,
    propagate_homography,
    estimate_relative_scale,
    compute_expected_scale_ratio,
    coarse_to_fine_register_pair
)

def test_phase4():
    print("Testing Phase 4 Scale/GSD Framework...")
    
    # A. GSD Ratio
    r1 = compute_expected_scale_ratio(2.0, 1.0)
    assert r1 == 2.0
    
    # B. Missing GSD
    r2 = compute_expected_scale_ratio(None, 1.0)
    assert r2 is None
    
    # E. Image-derived scale estimator (known synthetic scaling)
    # create synthetic points
    src_pts = np.array([[[0, 0]], [[10, 0]], [[0, 10]], [[10, 10]]], dtype=np.float32)
    # reference is scaled by 3.0
    ref_pts = src_pts * 3.0
    est, count, rot = estimate_relative_scale(src_pts, ref_pts)
    assert count == 4
    assert np.isclose(est, 3.0), f"Expected 3.0, got {est}"
    
    # F. Outlier-resistant
    ref_pts_outlier = np.vstack([ref_pts, [[[1000, 1000]]]])
    src_pts_outlier = np.vstack([src_pts, [[[20, 20]]]])
    est_out, count_out, rot_out = estimate_relative_scale(src_pts_outlier, ref_pts_outlier)
    assert count_out >= 4
    assert np.isclose(est_out, 3.0)
    
    # G. Pyramid dimensions
    img = np.zeros((100, 100), dtype=np.uint8)
    pyr = build_image_pyramid(img, levels=3, scale_factor=0.5, min_size=10)
    assert len(pyr) == 3
    assert pyr[0].shape == (100, 100)
    assert pyr[1].shape == (50, 50)
    assert pyr[2].shape == (25, 25)
    
    # I. Transform propagation
    H_coarse = np.array([
        [1, 0, 10],
        [0, 1, 5],
        [0, 0, 1]
    ], dtype=np.float32)
    # Propagate from 0.5x to 1.0x (scale_ratio = 2.0)
    H_fine = propagate_homography(H_coarse, 2.0)
    assert H_fine[0, 2] == 20.0
    assert H_fine[1, 2] == 10.0
    
    # J. Known synthetic coarse-to-fine registration
    src_img = np.zeros((200, 200), dtype=np.uint8)
    # draw a strong feature
    cv2.rectangle(src_img, (50, 50), (150, 150), 255, -1)
    
    # Create reference translated by 20, 20
    ref_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(ref_img, (70, 70), (170, 170), 255, -1)
    
    res, diag, prov = coarse_to_fine_register_pair(src_img, ref_img, levels=2)
    
    assert prov.coarse_to_fine_enabled == True
    assert diag.expected_scale_ratio is None
    assert diag.estimated_image_scale_ratio is not None
    assert prov.scale_evidence_type == "ESTIMATED_FROM_IMAGES"
    
    print("All Phase 4 tests passed.")

if __name__ == "__main__":
    test_phase4()
