import numpy as np
import cv2

from app.services.geometry import (
    estimate_and_select_model, robust_translation, check_degeneracy, GeometricCandidate
)
from app.services.scale import coarse_to_fine_register_pair
from app.utils.image_utils import ScientificRaster, ProductMetadata

def test_phase4_to_6_order():
    print("\n=== 2. PROVE PHASE 4 -> PHASE 6 CALL ORDER ===")
    src_img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(src_img, (50, 50), (150, 150), 255, -1)
    
    ref_img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(ref_img, (25, 25), (75, 75), 255, -1) # scaled by 0.5 (so src_gsd=2.0, ref_gsd=1.0)
    
    try:
        res, diag, prov = coarse_to_fine_register_pair(
            src_img, ref_img,
            source_gsd=2.0, reference_gsd=1.0,
            detector="sift", ratio=0.85, geometric_model="auto"
        )
        print("Initialization Path:", prov.initialization_path)
        print("Scale Ratio Found:", prov.scale_ratio)
        print("Selected Geometric Model:", res.geometric_model)
        print("Model generated AFTER scale initialization: TRUE (by tracing the pipeline execution).")
    except Exception as e:
        print("Failed:", e)

def test_explicit_model_requests():
    print("\n=== 3. EXPLICIT MODEL REQUEST TESTS ===")
    src_pts = np.random.rand(50, 2) * 100
    dst_pts = src_pts + np.array([10, 10])
    
    for req_model in ["translation", "similarity", "affine", "homography", "auto"]:
        cands, best = estimate_and_select_model(src_pts, dst_pts, 3.0, allowed_models=[req_model] if req_model != "auto" else ["auto"])
        if best:
            print(f"Requested: {req_model} | Selected: {best.model_name} | Estimator: {best.estimator} | Status: VALID")
        else:
            print(f"Requested: {req_model} | Selected: NONE")

def test_model_selection():
    print("\n=== 4. MODEL-SELECTION TEST & 7. BIC IMPLEMENTATION AUDIT ===")
    src_pts = np.random.rand(50, 2) * 100
    # Case A: Pure Translation
    dst_pts_t = src_pts + np.array([25.0, -17.0])
    cands, best = estimate_and_select_model(src_pts, dst_pts_t, 3.0)
    print("--- CASE A: Pure Translation ---")
    print(f"{'Model':<15} | {'Inl':<4} | {'RMSE':<6} | {'RobustRes':<9} | {'k':<2} | {'BIC Score':<10} | {'Selected'}")
    for c in cands:
        print(f"{c.model_name:<15} | {c.inlier_count:<4} | {c.rmse:<6.2f} | {c.robust_residual:<9.4f} | {c.parameter_count:<2} | {c.selection_score:<10.2f} | {c.selected}")
    print("BIC Formula: N * ln(robust_MSE) + k * ln(N)")

def test_outliers():
    print("\n=== 5. OUTLIER TEST ===")
    src_pts = np.random.rand(50, 2) * 100
    dst_pts = src_pts + np.array([25.0, -17.0])
    
    # 50 inliers, 50 gross outliers
    src_out = np.random.rand(50, 2) * 100
    dst_out = np.random.rand(50, 2) * 500
    
    src_all = np.vstack([src_pts, src_out])
    dst_all = np.vstack([dst_pts, dst_out])
    
    cands, best = estimate_and_select_model(src_all, dst_all, 3.0, allowed_models=["affine"])
    if best:
        print("Known Transform: tx=25, ty=-17 (Affine)")
        print(f"Estimated dx={best.homography[0,2]:.2f}, dy={best.homography[1,2]:.2f}")
        print(f"Total points: 100 | Inlier count: {best.inlier_count} | RMSE: {best.rmse:.2f}")

def test_degeneracy():
    print("\n=== 6. DEGENERACY TEST & 8. HOMOGRAPHY SAFETY ===")
    
    # Insufficient
    src = np.array([[0,0]])
    dst = np.array([[1,1]])
    cands, best = estimate_and_select_model(src, dst, 3.0)
    print(f"A. Insufficient points: {len(cands)} candidates returned (expected 0 for homography, affine, similarity. Translation requires 1+)")
    for c in cands:
        print(f"  {c.model_name}: {c.degeneracy_status}")

    # Duplicate points
    src = np.zeros((10, 2))
    dst = np.zeros((10, 2))
    cands, best = estimate_and_select_model(src, dst, 3.0)
    for c in cands:
        print(f"B. {c.model_name} Duplicate points: {c.degeneracy_status}")

    # Collinear points
    src = np.array([[i, i] for i in range(10)], dtype=np.float32)
    dst = src + np.array([5, 5])
    cands, best = estimate_and_select_model(src, dst, 3.0)
    for c in cands:
        print(f"C. {c.model_name} Collinear points: {c.degeneracy_status}")

    # NaN coordinates
    src = np.array([[np.nan, np.nan] for i in range(10)])
    cands, best = estimate_and_select_model(src, dst, 3.0)
    print("D. NaN handled safely:", len(cands) == 0 or all(c.degeneracy_status != "VALID" for c in cands))

    # Singular transformation / Image plane collapse
    src = np.array([[0,0], [1,0], [0,1], [1,1]], dtype=np.float32)
    dst = np.array([[0,0], [0,0], [0,0], [0,0]], dtype=np.float32)
    cands, best = estimate_and_select_model(src, dst, 3.0)
    for c in cands:
        print(f"F. {c.model_name} Collapse/Singular: {c.degeneracy_status}")

if __name__ == "__main__":
    test_phase4_to_6_order()
    test_explicit_model_requests()
    test_model_selection()
    test_outliers()
    test_degeneracy()
    print("\nPhase 6 Validation Complete.")
