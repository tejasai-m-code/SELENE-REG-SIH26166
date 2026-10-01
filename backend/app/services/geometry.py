import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

@dataclass
class GeometricCandidate:
    model_name: str
    estimator: str
    parameter_count: int
    minimum_points: int
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_reprojection_error: float
    mean_reprojection_error: float
    rmse: float
    robust_residual: float
    spatial_coverage: float
    degeneracy_status: str
    conditioning: float
    selection_score: float
    selected: bool
    failure_reason: Optional[str]
    homography: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]

def robust_translation(src, dst, threshold):
    deltas = dst - src
    if len(deltas) == 0:
        return None, None
    deltas = deltas.reshape(-1, 2)
    dx_med = np.median(deltas[:, 0])
    dy_med = np.median(deltas[:, 1])
    res = np.linalg.norm(deltas[:, :] - np.array([dx_med, dy_med]), axis=1)
    mask = res <= threshold
    if np.sum(mask) == 0:
        return None, None
    dx_mean = np.mean(deltas[mask, 0])
    dy_mean = np.mean(deltas[mask, 1])
    H = np.array([[1.0, 0.0, dx_mean], [0.0, 1.0, dy_mean], [0.0, 0.0, 1.0]], dtype=np.float64)
    res = np.linalg.norm(deltas[:, :] - np.array([dx_mean, dy_mean]), axis=1)
    return H, (res <= threshold).astype(np.uint8).reshape(-1, 1)

def check_degeneracy(src, dst, H, mask):
    if H is None or mask is None:
        return "FAILED_ESTIMATION", 0.0
    inliers = np.sum(mask)
    if inliers == 0:
        return "ZERO_INLIERS", 0.0
        
    # Check H validity
    if not np.all(np.isfinite(H)):
        return "NON_FINITE_PARAMETERS", 0.0
        
    if abs(H[2, 2]) < 1e-6:
        return "INVALID_H33_NORMALIZATION", 0.0
        
    # Condition number of top-left 2x2
    A = H[:2, :2]
    det = np.linalg.det(A)
    if det <= 0:
        return "CATASTROPHIC_COLLAPSE_OR_REFLECTION", det
        
    cond = np.linalg.cond(A)
    if cond > 1e4:
        return "ILL_CONDITIONED_TRANSFORMATION", cond
        
    # Check extreme scale
    scale_x, scale_y = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    if scale_x < 0.01 or scale_y < 0.01 or scale_x > 100 or scale_y > 100:
        return "EXTREME_SCALE", cond
        
    # Check extreme projective distortion
    if abs(H[2, 0]) > 0.01 or abs(H[2, 1]) > 0.01:
        return "EXTREME_PROJECTIVE_DISTORTION", cond
        
    return "VALID", cond

def evaluate_model(model_name: str, k: int, min_pts: int, src, dst, H, mask, threshold, image_shape, spatial_grid):
    N = len(src)
    if H is None or mask is None or np.sum(mask) < min_pts:
        return GeometricCandidate(
            model_name=model_name, estimator="FAILED", parameter_count=k, minimum_points=min_pts,
            candidate_count=N, inlier_count=0, inlier_ratio=0.0, median_reprojection_error=999.0,
            mean_reprojection_error=999.0, rmse=999.0, robust_residual=999.0, spatial_coverage=0.0,
            degeneracy_status="INSUFFICIENT_CORRESPONDENCES", conditioning=0.0, selection_score=99999.0,
            selected=False, failure_reason="Insufficient valid inliers", homography=None, inlier_mask=None
        )
        
    deg_status, cond = check_degeneracy(src, dst, H, mask)
    if deg_status != "VALID":
        return GeometricCandidate(
            model_name=model_name, estimator="REJECTED", parameter_count=k, minimum_points=min_pts,
            candidate_count=N, inlier_count=int(np.sum(mask)), inlier_ratio=float(np.sum(mask))/max(N, 1),
            median_reprojection_error=999.0, mean_reprojection_error=999.0, rmse=999.0, robust_residual=999.0,
            spatial_coverage=0.0, degeneracy_status=deg_status, conditioning=cond, selection_score=99999.0,
            selected=False, failure_reason=deg_status, homography=None, inlier_mask=None
        )
        
    # Normalize H33
    H = H / H[2, 2]
    
    # Calculate residuals for all points
    src_h = np.concatenate([src.reshape(-1, 2), np.ones((N, 1))], axis=1)
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / np.maximum(np.abs(proj[:, 2:]), 1e-8)
    sq_errs = np.sum((dst.reshape(-1, 2) - proj)**2, axis=1)
    errs = np.sqrt(sq_errs)
    
    inlier_idx = mask.ravel() > 0
    inlier_errs = errs[inlier_idx]
    
    if len(inlier_errs) == 0:
        mean_err, med_err, rmse_val = 999.0, 999.0, 999.0
    else:
        mean_err = float(np.mean(inlier_errs))
        med_err = float(np.median(inlier_errs))
        rmse_val = float(np.sqrt(np.mean(inlier_errs**2)))
        
    # Robust MSE for Model Selection (cap squared errors at threshold^2)
    robust_sq_errs = np.minimum(sq_errs, threshold**2)
    robust_mse = max(float(np.mean(robust_sq_errs)), 1e-6)
    
    # BIC / Robust Selection Criterion: N * ln(robust_MSE) + k * ln(N)
    # The lower the better
    score = N * np.log(robust_mse) + k * np.log(max(N, 1))
    
    # Spatial coverage
    h, w = image_shape[:2] if image_shape else (1000, 1000)
    grid_rows, grid_cols = spatial_grid if spatial_grid else (4, 4)
    cells = set()
    for pt in src.reshape(-1, 2)[inlier_idx]:
        r = min(grid_rows - 1, max(0, int(pt[1] / max(h, 1) * grid_rows)))
        c = min(grid_cols - 1, max(0, int(pt[0] / max(w, 1) * grid_cols)))
        cells.add((r, c))
    coverage = len(cells) / max(1, grid_rows * grid_cols)
    
    return GeometricCandidate(
        model_name=model_name, estimator="RANSAClike", parameter_count=k, minimum_points=min_pts,
        candidate_count=N, inlier_count=int(np.sum(mask)), inlier_ratio=float(np.sum(mask))/max(N, 1),
        median_reprojection_error=med_err, mean_reprojection_error=mean_err, rmse=rmse_val,
        robust_residual=robust_mse, spatial_coverage=coverage, degeneracy_status="VALID",
        conditioning=cond, selection_score=score, selected=False, failure_reason=None,
        homography=H, inlier_mask=mask
    )

def estimate_and_select_model(src, dst, threshold, image_shape=None, spatial_grid=(4,4), allowed_models=None):
    if allowed_models is None or "auto" in allowed_models:
        allowed_models = ["translation", "similarity", "affine", "homography"]
        
    candidates = []
    
    # Check duplicates/collinear basically
    if len(src) < 1:
        return candidates, None
        
    # 1. Translation (k=2)
    if "translation" in allowed_models:
        H_t, mask_t = robust_translation(src, dst, threshold)
        cand_t = evaluate_model("translation", 2, 1, src, dst, H_t, mask_t, threshold, image_shape, spatial_grid)
        cand_t.estimator = "MEDIAN_RANSAC"
        candidates.append(cand_t)
        
    # 2. Similarity (k=4)
    if "similarity" in allowed_models and len(src) >= 2:
        try:
            H_s_2x3, mask_s = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold)
            if H_s_2x3 is not None:
                H_s = np.vstack([H_s_2x3, [0, 0, 1]])
            else:
                H_s = None
        except Exception:
            H_s, mask_s = None, None
        cand_s = evaluate_model("similarity", 4, 2, src, dst, H_s, mask_s, threshold, image_shape, spatial_grid)
        cand_s.estimator = "cv2.RANSAC"
        candidates.append(cand_s)
        
    # 3. Affine (k=6)
    if "affine" in allowed_models and len(src) >= 3:
        try:
            meth = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
            H_a_2x3, mask_a = cv2.estimateAffine2D(src, dst, method=meth, ransacReprojThreshold=threshold)
            if H_a_2x3 is not None:
                H_a = np.vstack([H_a_2x3, [0, 0, 1]])
            else:
                H_a = None
        except Exception:
            H_a, mask_a = None, None
            meth = cv2.RANSAC
        cand_a = evaluate_model("affine", 6, 3, src, dst, H_a, mask_a, threshold, image_shape, spatial_grid)
        cand_a.estimator = "cv2.MAGSAC++" if (hasattr(cv2, 'USAC_MAGSAC') and meth == cv2.USAC_MAGSAC) else "cv2.RANSAC"
        candidates.append(cand_a)
        
    # 4. Homography (k=8)
    if "homography" in allowed_models and len(src) >= 4:
        try:
            meth = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
            H_h, mask_h = cv2.findHomography(src, dst, meth, threshold)
        except Exception:
            H_h, mask_h = None, None
            meth = cv2.RANSAC
        cand_h = evaluate_model("homography", 8, 4, src, dst, H_h, mask_h, threshold, image_shape, spatial_grid)
        cand_h.estimator = "cv2.MAGSAC++" if (hasattr(cv2, 'USAC_MAGSAC') and meth == cv2.USAC_MAGSAC) else "cv2.RANSAC"
        candidates.append(cand_h)
        
    # Selection
    valid_cands = [c for c in candidates if c.failure_reason is None and c.homography is not None]
    best_cand = None
    if valid_cands:
        # Sort by BIC score (lower is better)
        valid_cands.sort(key=lambda c: c.selection_score)
        
        # Additional heuristic: ensure sufficient spatial coverage improvement if choosing more complex models
        best_cand = valid_cands[0]
        # But we trust BIC for now, since it already penalizes complexity heavily via k * ln(N)
        best_cand.selected = True
        
    return candidates, best_cand
