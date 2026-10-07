import sys
import os
from pathlib import Path

# Add python backend to path
sys.path.insert(0, os.path.abspath("artifacts/api-server/python"))

import cv2
import numpy as np
import json
from app.services.pairwise_registration import register_pair, serialize_correspondences
from app.services.subpixel import compare_and_select_refinement

src_path = "demo_data/synthetic_source.png"
ref_path = "demo_data/synthetic_reference.png"

src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
print(f"Images loaded successfully: src={src.shape}, ref={ref.shape}")

print("\n" + "=" * 60)
print("TEST 1: AUTO SELECT REFINEMENT EXECUTION")
print("=" * 60)

res_auto = register_pair(src, ref, refinement_methods=["auto"])
subpixel_auto = res_auto.subpixel or {}
print(f"Registration status: {res_auto.registration_status} (Success: {res_auto.success})")
print(f"Selected method: {subpixel_auto.get('selected_method')}")
print(f"Validation status: {subpixel_auto.get('subpixel_validation_status')}")
print(f"Validation reason: {subpixel_auto.get('subpixel_validation_reason')}")
raw_rmse = res_auto.raw_rmse
refined_rmse = res_auto.refined_rmse
improvement = res_auto.rmse_improvement
print(f"Raw RMSE: {raw_rmse:.4f} px" if raw_rmse is not None else "Raw RMSE: None")
print(f"Refined RMSE: {refined_rmse:.4f} px" if refined_rmse is not None else "Refined RMSE: None")
print(f"Improvement: {improvement:.4f} px" if improvement is not None else "Improvement: None")

print("\nComparison Ledger:")
ledger = subpixel_auto.get("comparison_ledger", [])
for item in ledger:
    print(f"  - {item.get('display_name')} ({item.get('method')}):")
    print(f"      succeeded: {item.get('succeeded')}")
    print(f"      refined_rmse: {item.get('refined_rmse_pixels')}")
    print(f"      reduction: {item.get('rmse_reduction_pixels')}")
    print(f"      converged_pts: {item.get('converged_points')}")
    print(f"      mean_shift: {item.get('mean_shift_pixels')}")
    print(f"      hardware: {item.get('hardware')}")
    print(f"      reason: {item.get('reason')}")

corrs = serialize_correspondences(res_auto)
print(f"\nTotal correspondences: {len(corrs)}")
inliers = [c for c in corrs if c.get("is_inlier")]
print(f"Inlier correspondences: {len(inliers)}")
if inliers:
    print("\nPoint-Level Inspection Sample (Inlier Match 1):")
    sample = inliers[0]
    for k in [
        "index",
        "source",
        "reference",
        "initial_reference",
        "refined_reference",
        "integer_coordinate",
        "refinement_delta",
        "shift_magnitude",
        "residual",
        "raw_residual",
        "refined_residual",
        "residual_improvement",
        "refinement_method",
        "iterations",
        "converged",
        "convergence_status",
        "local_correlation",
        "status",
        "is_inlier",
    ]:
        print(f"    {k}: {sample.get(k)}")

print("\n" + "=" * 60)
print("TEST 2: INDIVIDUAL REFINEMENT METHODS ON REAL SYNTHETIC PAIR")
print("=" * 60)
methods = ["taylor", "lucas_kanade", "ecc", "phase", "quadratic"]
for m in methods:
    res = register_pair(src, ref, refinement_methods=[m])
    sp = res.subpixel or {}
    app = sp.get("applied_methods", [])
    m_stat = sp.get("method_status", [{}])[0] if sp.get("method_status") else {}
    print(f"\nMethod: {m.upper()}")
    print(f"  Applied methods: {app}")
    print(f"  Succeeded: {m_stat.get('succeeded')}")
    print(f"  Reason: {m_stat.get('reason')}")
    print(f"  Raw RMSE: {res.raw_rmse:.4f} px" if res.raw_rmse else "  Raw RMSE: None")
    print(f"  Refined RMSE: {res.refined_rmse:.4f} px" if res.refined_rmse else "  Refined RMSE: None")
    print(f"  Improvement: {res.rmse_improvement:.4f} px" if res.rmse_improvement else "  Improvement: None")
    print(f"  Subpixel validation: {sp.get('subpixel_validation_status')}")

print("\n" + "=" * 60)
print("TEST 3: KNOWN FRACTIONAL TRANSLATION TEST (Controlled Ground Truth)")
print("=" * 60)
# Known synthetic subpixel displacement: dx = +0.35, dy = -0.45
M_sub = np.array([[1.0, 0.0, 0.35], [0.0, 1.0, -0.45]], dtype=np.float32)
ref_sub = cv2.warpAffine(src, M_sub, (src.shape[1], src.shape[0]), flags=cv2.INTER_CUBIC)

res_gt = register_pair(src, ref_sub, refinement_methods=["auto"])
sp_gt = res_gt.subpixel or {}
print(f"Ground Truth Shift: dx=+0.35 px, dy=-0.45 px")
print(f"Auto Selected Method: {sp_gt.get('selected_method')}")
print(f"Validation Status: {sp_gt.get('subpixel_validation_status')}")
print(f"Validation Reason: {sp_gt.get('subpixel_validation_reason')}")
print(f"Raw RMSE: {res_gt.raw_rmse:.4f} px" if res_gt.raw_rmse else "Raw RMSE: None")
print(f"Refined RMSE: {res_gt.refined_rmse:.4f} px" if res_gt.refined_rmse else "Refined RMSE: None")
print(f"Improvement: {res_gt.rmse_improvement:.4f} px" if res_gt.rmse_improvement else "Improvement: None")

for item in sp_gt.get("comparison_ledger", []):
    print(f"  Candidate: {item.get('display_name')} | Succeeded: {item.get('succeeded')} | Refined RMSE: {item.get('refined_rmse_pixels')} | Reduction: {item.get('rmse_reduction_pixels')}")
