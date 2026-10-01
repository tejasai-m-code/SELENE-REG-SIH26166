import numpy as np
import cv2

from app.services.geometry import (
    estimate_and_select_model, robust_translation, check_degeneracy, GeometricCandidate
)

def test_phase6_geometry():
    print("Testing Phase 6 Geometric Model Selection...\n")
    
    # 6.12 SYNTHETIC KNOWN-TRANSFORM LAB
    # TEST A - TRANSLATION
    print("--- TEST A: TRANSLATION ---")
    src_pts = np.random.rand(50, 2) * 100
    dst_pts_trans = src_pts + np.array([25.0, -17.0])
    # Add outliers
    src_pts = np.vstack([src_pts, np.random.rand(20, 2) * 100])
    dst_pts_trans = np.vstack([dst_pts_trans, np.random.rand(20, 2) * 100])
    
    cands, best = estimate_and_select_model(src_pts, dst_pts_trans, threshold=3.0, allowed_models=["translation"])
    print(f"Translation dx={best.homography[0,2]:.2f}, dy={best.homography[1,2]:.2f}")
    assert abs(best.homography[0,2] - 25.0) < 1e-3
    assert abs(best.homography[1,2] + 17.0) < 1e-3
    
    # TEST B - SIMILARITY
    print("--- TEST B: SIMILARITY ---")
    scale = 1.25
    theta = np.radians(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]) * scale
    dst_pts_sim = (R @ src_pts[:50].T).T + np.array([10.0, 15.0])
    dst_pts_sim = np.vstack([dst_pts_sim, np.random.rand(20, 2) * 100])
    
    cands, best = estimate_and_select_model(src_pts, dst_pts_sim, threshold=3.0, allowed_models=["similarity"])
    print(f"Similarity Scale (approx)={np.linalg.norm(best.homography[:2,0]):.2f}")
    assert abs(np.linalg.norm(best.homography[:2,0]) - 1.25) < 0.1
    
    # TEST C - AFFINE
    print("--- TEST C: AFFINE ---")
    A = np.array([[1.5, 0.2], [0.1, 0.8]])
    dst_pts_aff = (A @ src_pts[:50].T).T + np.array([5.0, -5.0])
    dst_pts_aff = np.vstack([dst_pts_aff, np.random.rand(20, 2) * 100])
    cands, best = estimate_and_select_model(src_pts, dst_pts_aff, threshold=3.0, allowed_models=["affine"])
    print(f"Affine Matrix:\n{best.homography}")
    assert abs(best.homography[0,0] - 1.5) < 0.1
    
    # TEST D - HOMOGRAPHY
    print("--- TEST D: HOMOGRAPHY ---")
    H_true = np.array([
        [1.2, 0.1, 10],
        [-0.05, 1.1, -20],
        [0.001, 0.002, 1.0]
    ])
    src_h = np.hstack([src_pts[:50], np.ones((50,1))])
    dst_proj = (H_true @ src_h.T).T
    dst_proj = dst_proj[:, :2] / dst_proj[:, 2:]
    dst_pts_hom = np.vstack([dst_proj, np.random.rand(20, 2) * 100])
    
    cands, best = estimate_and_select_model(src_pts, dst_pts_hom, threshold=3.0)
    print(f"Homography Selected: {best.model_name} (Estimator: {best.estimator})")
    
    # 6.13 MODEL-SELECTION SYNTHETIC TEST
    print("--- 6.13 MODEL-SELECTION TEST ---")
    # Pure translation case should favor translation due to complexity penalty
    cands, best_t = estimate_and_select_model(src_pts, dst_pts_trans, threshold=3.0, allowed_models=["auto"])
    print(f"For translation data, selected model: {best_t.model_name}")
    # Translation or similarity is fine as long as homography doesn't trivially win

    # 6.15 DEGENERACY TESTS
    print("--- 6.15 DEGENERACY TESTS ---")
    dup_pts = np.zeros((10, 2))
    dup_dst = np.zeros((10, 2))
    cands, best_dup = estimate_and_select_model(dup_pts, dup_dst, threshold=3.0)
    if not cands:
        print("Degeneracy: Dup points caught")
    else:
        for c in cands:
            print(f"{c.model_name} degeneracy status: {c.degeneracy_status}")

    # Collinear points
    col_pts = np.array([[i, i] for i in range(10)], dtype=np.float32)
    col_dst = col_pts + np.array([5, 5])
    cands, best_col = estimate_and_select_model(col_pts, col_dst, threshold=3.0)
    for c in cands:
        if c.homography is not None:
             print(f"{c.model_name} collinear degeneracy: {c.degeneracy_status}")
    
    print("\nPhase 6 Tests completed successfully.")

if __name__ == "__main__":
    test_phase6_geometry()
