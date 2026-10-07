import sys
sys.path.insert(0, 'artifacts/api-server/python')
import cv2
import numpy as np
from app.services.subpixel import compare_and_select_refinement

rng = np.random.default_rng(2026)
base = rng.normal(128, 25, (300, 300)).clip(20, 240).astype(np.float32)
base = cv2.GaussianBlur(base, (3, 3), 1.0)
for _ in range(35):
    cx = rng.integers(30, 270)
    cy = rng.integers(30, 270)
    r = rng.integers(8, 25)
    cv2.circle(base, (cx, cy), r, float(rng.integers(190, 255)), 2)
    cv2.circle(base, (cx - 1, cy - 1), r - 3, float(rng.integers(30, 80)), -1)
src = cv2.GaussianBlur(base, (3, 3), 0.8).astype(np.uint8)

# Synthetic transform: dx = +0.40, dy = -0.30
M = np.array([[1.0, 0.0, 0.40], [0.0, 1.0, -0.30]], dtype=np.float32)
ref = cv2.warpAffine(src, M, (300, 300), flags=cv2.INTER_LINEAR)

grid_x, grid_y = np.meshgrid(np.linspace(60, 240, 5), np.linspace(60, 240, 5))
pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

# In real registration, initial correspondence points have some integer/coarse error
# True ref positions:
pts_r_true = pts_s + np.array([0.40, -0.30], dtype=np.float32)
# Perturbed initial correspondences with ~0.25 px noise
pts_r_init = pts_r_true + rng.normal(0, 0.25, pts_r_true.shape).astype(np.float32)
# Initial homography fitted to noisy correspondences
H_init = np.array([[1.0, 0.0, 0.40], [0.0, 1.0, -0.30], [0.0, 0.0, 1.0]], dtype=np.float64)

res = compare_and_select_refinement(
    source_gray=src,
    reference_gray=ref,
    source_points=pts_s,
    reference_points=pts_r_init,
    homography=H_init,
    candidate_methods=["auto"],
)

print("Selected:", res["selected_method"])
print("Reason:", res["reason"])
print("Raw RMSE:", res["raw_rmse_pixels"])
print("Refined RMSE:", res["refined_rmse_pixels"])
print("Improvement:", res["rmse_improvement_pixels"])
for item in res["comparison_ledger"]:
    print(f"  Ledger: {item['method']} -> succ={item['succeeded']} rmse={item['refined_rmse_pixels']} red={item['rmse_reduction_pixels']} shift={item['mean_shift_pixels']} reason={item['reason']}")
