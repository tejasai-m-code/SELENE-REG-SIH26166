from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import cv2

@dataclass
class SubpixelResult:
    method: str
    status: str
    dx: float
    dy: float
    residual: float
    iterations: int
    correlation: Optional[float]
    response: Optional[float]
    stability: str
    confidence: str
    failure_reason: Optional[str]

@dataclass
class ConsensusSubpixelResult:
    dx: float
    dy: float
    confidence: str
    active_methods: int
    diagnostics: List[SubpixelResult]
    failure_reason: Optional[str]

def _get_center_crop(img, crop_size=256):
    h, w = img.shape[:2]
    if h <= crop_size and w <= crop_size:
        return img.copy(), 0, 0
    ch, cw = min(h, crop_size), min(w, crop_size)
    y = (h - ch) // 2
    x = (w - cw) // 2
    return img[y:y+ch, x:x+cw].copy(), x, y

def taylor_refinement(src_aligned: np.ndarray, ref: np.ndarray, max_iters: int = 10, epsilon: float = 1e-4) -> SubpixelResult:
    """
    Taylor / Image-Gradient Refinement:
    I(x+dx, y+dy) ≈ I(x,y) + Ix*dx + Iy*dy
    => Ix*dx + Iy*dy = I_ref - I_src
    """
    try:
        # Use center crops to avoid edge artifacts from warping
        s_crop, _, _ = _get_center_crop(src_aligned)
        r_crop, _, _ = _get_center_crop(ref)
        
        s = cv2.GaussianBlur(s_crop, (5, 5), 1.0).astype(np.float32)
        r = cv2.GaussianBlur(r_crop, (5, 5), 1.0).astype(np.float32)
        
        dx, dy = 0.0, 0.0
        
        for i in range(max_iters):
            # Calculate gradients of reference image
            Ix = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            Iy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            
            # Warp source by current (dx, dy)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            s_warped = cv2.warpAffine(s, M, (s.shape[1], s.shape[0]), flags=cv2.INTER_LINEAR)
            
            diff = r - s_warped
            
            # Mask out black borders
            mask = (s_warped > 0) & (r > 0)
            if np.sum(mask) < 100:
                return SubpixelResult("taylor", "FAILED", 0.0, 0.0, 999.0, i, None, None, "UNSTABLE", "LOW", "Insufficient overlap")
                
            Ix_f = Ix[mask].flatten()
            Iy_f = Iy[mask].flatten()
            diff_f = diff[mask].flatten()
            
            # Solve [Ix Iy] * [ddx; ddy] = diff
            A = np.vstack([Ix_f, Iy_f]).T
            # Normal equations: A.T * A * delta = A.T * b
            ATA = A.T @ A
            ATb = A.T @ diff_f
            
            if np.linalg.det(ATA) < 1e-6:
                return SubpixelResult("taylor", "FAILED", float(dx), float(dy), 999.0, i, None, None, "ILL_CONDITIONED", "LOW", "Singular gradient matrix")
                
            delta = np.linalg.solve(ATA, ATb)
            dx += delta[0]
            dy += delta[1]
            
            if np.linalg.norm(delta) < epsilon:
                res = float(np.sqrt(np.mean(diff_f**2)))
                return SubpixelResult("taylor", "CONVERGED", float(dx), float(dy), res, i+1, None, None, "STABLE", "HIGH", None)
                
            if np.linalg.norm([dx, dy]) > 5.0:
                return SubpixelResult("taylor", "DIVERGED", float(dx), float(dy), 999.0, i+1, None, None, "UNSTABLE", "LOW", "Displacement too large")
                
        res = float(np.sqrt(np.mean(diff[mask]**2)))
        return SubpixelResult("taylor", "MAX_ITERS", float(dx), float(dy), res, max_iters, None, None, "MARGINAL", "MEDIUM", "Reached max iterations")
    except Exception as e:
        return SubpixelResult("taylor", "ERROR", 0.0, 0.0, 999.0, 0, None, None, "ERROR", "LOW", str(e))

def lk_refinement(src_aligned: np.ndarray, ref: np.ndarray) -> SubpixelResult:
    """
    Lucas-Kanade via cv2.calcOpticalFlowPyrLK on a grid of points.
    """
    try:
        s_crop, x0, y0 = _get_center_crop(src_aligned)
        r_crop, _, _ = _get_center_crop(ref)
        
        # Create grid of points
        h, w = s_crop.shape[:2]
        pts_s = []
        for y in range(h//4, 3*h//4, max(5, h//10)):
            for x in range(w//4, 3*w//4, max(5, w//10)):
                if s_crop[y, x] > 0 and r_crop[y, x] > 0:
                    pts_s.append([float(x), float(y)])
                    
        if len(pts_s) < 4:
            return SubpixelResult("lk", "FAILED", 0.0, 0.0, 999.0, 0, None, None, "UNSTABLE", "LOW", "Insufficient active points")
            
        pts_s = np.array(pts_s, dtype=np.float32).reshape(-1, 1, 2)
        
        pts_r, st, err = cv2.calcOpticalFlowPyrLK(
            s_crop, r_crop, pts_s, None, 
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        
        good_s = pts_s[st == 1]
        good_r = pts_r[st == 1]
        
        if len(good_s) < 4:
            return SubpixelResult("lk", "FAILED", 0.0, 0.0, 999.0, 0, None, None, "UNSTABLE", "LOW", "LK lost tracks")
            
        deltas = good_r - good_s
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        
        # Residual
        res = float(np.median(np.linalg.norm(deltas - np.array([dx, dy]), axis=1)))
        return SubpixelResult("lk", "CONVERGED", dx, dy, res, 20, None, None, "STABLE", "HIGH", None)
    except Exception as e:
        return SubpixelResult("lk", "ERROR", 0.0, 0.0, 999.0, 0, None, None, "ERROR", "LOW", str(e))

def ecc_refinement(src_aligned: np.ndarray, ref: np.ndarray) -> SubpixelResult:
    """
    Enhanced Correlation Coefficient subpixel shift.
    """
    try:
        s_crop, _, _ = _get_center_crop(src_aligned)
        r_crop, _, _ = _get_center_crop(ref)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        
        cc, warp_matrix = cv2.findTransformECC(
            s_crop, r_crop, warp_matrix, cv2.MOTION_TRANSLATION, criteria, None, 1
        )
        
        dx = float(warp_matrix[0, 2])
        dy = float(warp_matrix[1, 2])
        return SubpixelResult("ecc", "CONVERGED", dx, dy, 0.0, 50, float(cc), None, "STABLE", "HIGH", None)
    except Exception as e:
        return SubpixelResult("ecc", "ERROR", 0.0, 0.0, 999.0, 0, None, None, "ERROR", "LOW", str(e))

def phase_correlation_refinement(src_aligned: np.ndarray, ref: np.ndarray) -> SubpixelResult:
    """
    Phase correlation with Hanning window and subpixel peak fitting.
    """
    try:
        s_crop, _, _ = _get_center_crop(src_aligned)
        r_crop, _, _ = _get_center_crop(ref)
        
        s = np.float32(s_crop)
        r = np.float32(r_crop)
        
        shift, response = cv2.phaseCorrelate(s, r)
        dx, dy = float(shift[0]), float(shift[1])
        
        return SubpixelResult("phase_correlate", "CONVERGED", dx, dy, 0.0, 1, None, float(response), "STABLE", "HIGH", None)
    except Exception as e:
        return SubpixelResult("phase_correlate", "ERROR", 0.0, 0.0, 999.0, 0, None, None, "ERROR", "LOW", str(e))

def local_correlation_peak(src_aligned: np.ndarray, ref: np.ndarray) -> SubpixelResult:
    """
    Normalized cross correlation and subpixel parabolic peak fitting.
    """
    try:
        s_crop, _, _ = _get_center_crop(src_aligned, crop_size=128)
        r_crop, _, _ = _get_center_crop(ref, crop_size=160)
        
        if s_crop.size == 0 or r_crop.size == 0:
            raise ValueError("Crops too small")
            
        res = cv2.matchTemplate(r_crop, s_crop, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Max_loc is top-left of matched region. Center of s_crop is matched to max_loc + center
        # dx, dy in integer pixels
        cx_s = s_crop.shape[1] // 2
        cy_s = s_crop.shape[0] // 2
        
        cx_r = r_crop.shape[1] // 2
        cy_r = r_crop.shape[0] // 2
        
        # The center of the matched region in ref:
        matched_cx_r = max_loc[0] + cx_s
        matched_cy_r = max_loc[1] + cy_s
        
        idx = matched_cx_r - cx_r
        idy = matched_cy_r - cy_r
        
        # Subpixel parabolic fitting
        x, y = max_loc
        dx_sub, dy_sub = 0.0, 0.0
        if 0 < x < res.shape[1]-1 and 0 < y < res.shape[0]-1:
            # Fit parabola to 3 points in x
            val_m1_x = res[y, x-1]
            val_0_x  = res[y, x]
            val_p1_x = res[y, x+1]
            dx_sub = (val_m1_x - val_p1_x) / (2.0 * (val_m1_x - 2.0*val_0_x + val_p1_x) + 1e-6)
            
            # Fit parabola to 3 points in y
            val_m1_y = res[y-1, x]
            val_0_y  = res[y, x]
            val_p1_y = res[y+1, x]
            dy_sub = (val_m1_y - val_p1_y) / (2.0 * (val_m1_y - 2.0*val_0_y + val_p1_y) + 1e-6)
            
            # Cap subpixel shift at [-0.5, 0.5]
            dx_sub = np.clip(dx_sub, -0.5, 0.5)
            dy_sub = np.clip(dy_sub, -0.5, 0.5)
            
        dx = float(idx + dx_sub)
        dy = float(idy + dy_sub)
        
        return SubpixelResult("local_peak", "CONVERGED", dx, dy, 0.0, 1, float(max_val), None, "STABLE", "HIGH", None)
    except Exception as e:
        return SubpixelResult("local_peak", "ERROR", 0.0, 0.0, 999.0, 0, None, None, "ERROR", "LOW", str(e))

def compute_subpixel_consensus(results: List[SubpixelResult], max_variance: float = 2.0, src_aligned: np.ndarray = None, ref: np.ndarray = None) -> ConsensusSubpixelResult:
    valid_results = [r for r in results if r.status == "CONVERGED" and r.stability == "STABLE"]
    
    grad_energy = 999.0
    local_variance = 999.0
    overlap_pixels = 0
    ncc_quality = 1.0
    
    if src_aligned is not None and ref is not None:
        mask = (src_aligned > 0) & (ref > 0)
        overlap_pixels = int(np.sum(mask))
        if overlap_pixels > 100:
            gx = cv2.Sobel(src_aligned, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(src_aligned, cv2.CV_32F, 0, 1, ksize=3)
            grad_energy = float(np.mean(gx[mask]**2 + gy[mask]**2))
            local_variance = float(np.var(src_aligned[mask]))
            
            s_f32 = src_aligned[mask].astype(np.float32)
            r_f32 = ref[mask].astype(np.float32)
            if np.std(s_f32) > 1e-5 and np.std(r_f32) > 1e-5:
                ncc_quality = np.corrcoef(s_f32, r_f32)[0, 1]
            else:
                ncc_quality = 0.0
    
    if not valid_results:
        return ConsensusSubpixelResult(0.0, 0.0, "FAILED", 0, results, "No valid subpixel estimators converged")
        
    dxs = np.array([r.dx for r in valid_results])
    dys = np.array([r.dy for r in valid_results])
    
    med_dx = np.median(dxs)
    med_dy = np.median(dys)
    
    # Filter gross outliers from consensus
    dists = np.sqrt((dxs - med_dx)**2 + (dys - med_dy)**2)
    inliers = dists <= max_variance
    
    if np.sum(inliers) == 0:
        return ConsensusSubpixelResult(0.0, 0.0, "FAILED", 0, results, "No consensus reached")
        
    final_dx = float(np.mean(dxs[inliers]))
    final_dy = float(np.mean(dys[inliers]))
    
    num_successful = int(np.sum(inliers))
    dispersion = float(np.max(dists[inliers])) if num_successful > 1 else 0.0
    
    if overlap_pixels < 100 or local_variance < 5.0 or grad_energy < 10.0:
        conf = "INSUFFICIENT_TEXTURE"
    elif num_successful == 0:
        conf = "FAILED"
    elif num_successful < 2:
        conf = "LOW"
    elif dispersion > 0.5:
        conf = "UNSTABLE"
    elif num_successful >= 3 and dispersion < 0.2 and ncc_quality > 0.5:
        conf = "HIGH"
    elif num_successful >= 2:
        conf = "MEDIUM"
    else:
        conf = "LOW"
        
    return ConsensusSubpixelResult(final_dx, final_dy, conf, int(np.sum(inliers)), results, None)

def run_subpixel_refinement(src_aligned: np.ndarray, ref: np.ndarray) -> ConsensusSubpixelResult:
    # Texture check moved to consensus
    
    # Run the estimators
    results = [
        taylor_refinement(src_aligned, ref),
        lk_refinement(src_aligned, ref),
        ecc_refinement(src_aligned, ref),
        phase_correlation_refinement(src_aligned, ref),
        local_correlation_peak(src_aligned, ref)
    ]
    return compute_subpixel_consensus(results, src_aligned=src_aligned, ref=ref)
