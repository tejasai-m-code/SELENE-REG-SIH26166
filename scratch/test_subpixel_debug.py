import sys
sys.path.insert(0, 'artifacts/api-server/python')
import cv2
import numpy as np
from app.services.subpixel import (
    lucas_kanade,
    taylor_expansion,
    dft_upsample_registration,
    compare_and_select_refinement,
)

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

# 1. LK test
M = np.array([[1.0, 0.0, -0.40], [0.0, 1.0, 0.35]], dtype=np.float32)
ref = cv2.warpAffine(src, M, (300, 300), flags=cv2.INTER_LINEAR)
grid_x, grid_y = np.meshgrid(np.linspace(70, 230, 4), np.linspace(70, 230, 4))
pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
pts_r_initial = pts_s.copy()

refined_pts, valid, details = lucas_kanade(src, ref, pts_s, pts_r_initial, return_details=True)
print("LK valid count:", np.sum(valid))
print("LK refined - initial (mean):", np.mean(refined_pts[valid] - pts_r_initial[valid], axis=0))
for d in details[:3]:
    print("LK detail:", d)

# 2. DFT upsample test
target_shift = (0.35, -0.45)
M_dft = np.array([[1.0, 0.0, target_shift[0]], [0.0, 1.0, target_shift[1]]], dtype=np.float32)
ref_dft = cv2.warpAffine(src.astype(np.float32), M_dft, (300, 300), flags=cv2.INTER_CUBIC)
(rec_dx, rec_dy), resp = dft_upsample_registration(src.astype(np.float32), ref_dft, upsample_factor=20)
print(f"DFT upsample from src to ref: rec_dx={rec_dx}, rec_dy={rec_dy}, resp={resp}")
(rec_dx2, rec_dy2), resp2 = dft_upsample_registration(ref_dft, src.astype(np.float32), upsample_factor=20)
print(f"DFT upsample from ref to src: rec_dx2={rec_dx2}, rec_dy2={rec_dy2}, resp2={resp2}")

# 3. Taylor test
refined_t, valid_t, details_t = taylor_expansion(src, ref_dft, pts_s, pts_r_initial, return_details=True)
print("Taylor valid count:", np.sum(valid_t))
print("Taylor refined - initial (mean):", np.mean(refined_t[valid_t] - pts_r_initial[valid_t], axis=0))
