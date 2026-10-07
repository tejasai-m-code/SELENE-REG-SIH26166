"""Measured local correspondence refinement methods.

These routines operate on the already verified correspondence set. They do not
turn a good-looking image into a claim of absolute sub-pixel accuracy; the
returned deltas are evidence that can be inspected and validated against known
synthetic transforms.

Phase correlation and ECC provide GLOBAL image-wide model refinement, while
Lucas-Kanade, Taylor expansion, Quadratic peak fitting, and CornerSubPix provide
LOCAL POINT-WISE correspondence refinement.
"""

from __future__ import annotations

from typing import Iterable
from time import perf_counter
import cv2
import numpy as np

from app.services.gpu_accelerator import is_cuda_available, phase_correlation_gpu


# Canonical mapping from UI aliases and alternative spellings to canonical backend identifiers
REFINEMENT_ALIASES: dict[str, str] = {
    # UI aliases -> backend canonical identifiers
    "taylor": "taylor_expansion",
    "phase": "phase_correlation",
    "quadratic": "quadratic_peak",
    "ecc": "ecc",
    "lucas_kanade": "lucas_kanade",
    "lk": "lucas_kanade",
    "corner_subpix": "corner_subpix",
    "corner": "corner_subpix",
    "auto": "auto",
    "ecc_alignment": "ecc",
    # Backend canonical names (pass-through)
    "taylor_expansion": "taylor_expansion",
    "phase_correlation": "phase_correlation",
    "quadratic_peak": "quadratic_peak",
}

CANONICAL_METHOD_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "auto": {
        "display_name": "Auto Selection (Lowest Residual)",
        "category": "hybrid",
        "description": "Evaluates candidate refinement methods and selects the method yielding lowest residual error.",
    },
    "taylor_expansion": {
        "display_name": "Taylor Expansion",
        "category": "point_wise",
        "description": "Local first-order brightness constancy linearization (inverse compositional LK).",
    },
    "phase_correlation": {
        "display_name": "Phase Correlation + Upsampling",
        "category": "global_translation",
        "description": "Fourier-domain cross-power spectrum with matrix-multiply DFT upsampling.",
    },
    "quadratic_peak": {
        "display_name": "Quadratic Peak Fitting",
        "category": "point_wise",
        "description": "2D quadric surface fit to local correlation response peak with Hessian validation.",
    },
    "ecc": {
        "display_name": "Enhanced Correlation Coefficient (ECC)",
        "category": "global_image",
        "description": "Iterative gradient-based image-wide homography optimization.",
    },
    "lucas_kanade": {
        "display_name": "Lucas–Kanade Optical Flow",
        "category": "point_wise",
        "description": "Iterative pyramidal LK optical flow on keypoint neighborhoods.",
    },
    "corner_subpix": {
        "display_name": "Corner SubPix",
        "category": "point_wise",
        "description": "Gradient-based orthogonal constraint corner refinement.",
    },
}

POINT_REFINEMENT_METHODS: set[str] = {
    "lucas_kanade",
    "taylor_expansion",
    "quadratic_peak",
    "corner_subpix",
}

GLOBAL_REFINEMENT_METHODS: set[str] = {
    "ecc",
    "phase_correlation",
}


def normalize_refinement_method(name: str) -> str | None:
    """Normalize a single method name or alias to its canonical identifier."""
    cleaned = str(name).strip().lower()
    return REFINEMENT_ALIASES.get(cleaned)


def normalize_refinement_methods(methods: Iterable[str] | str | None) -> list[str]:
    """Convert an arbitrary list or comma-separated string of method names/aliases
    into an ordered list of unique canonical method identifiers.
    Unknown tokens are omitted from canonical list.
    """
    if methods is None:
        return []
    if isinstance(methods, str):
        items = [item.strip() for item in methods.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in methods if str(item).strip()]

    canonical: list[str] = []
    for item in items:
        norm = normalize_refinement_method(item)
        if norm and norm not in canonical:
            canonical.append(norm)
    return canonical


def _points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


# =============================================================================
# 1. Matrix-Multiplication DFT Upsampling (Guizar-Sicairos, Thurman, & Fienup 2008)
# =============================================================================

def _upsampled_dft(
    data: np.ndarray,
    upsampled_region_size: tuple[int, int],
    upsample_factor: int = 1,
    axis_offsets: tuple[float, float] | None = None,
) -> np.ndarray:
    """Matrix-multiply 2D Discrete Fourier Transform upsampling.
    
    Guizar-Sicairos, Thurman, & Fienup (Optics Letters 33, 156-158, 2008).
    Computes 2D inverse Fourier transform at sub-pixel resolution via
    separable matrix multiplications without zero-padding the full array.
    """
    im2pi = 1j * 2.0 * np.pi
    offsets = axis_offsets if axis_offsets is not None else (0.0, 0.0)
    
    # Row transformation (axis 0)
    t_row = (np.arange(upsampled_region_size[0]) - offsets[0])[:, None] * np.fft.fftfreq(data.shape[0])[None, :]
    kern_row = np.exp(im2pi * t_row / upsample_factor)
    
    # Col transformation (axis 1)
    t_col = (np.arange(upsampled_region_size[1]) - offsets[1])[:, None] * np.fft.fftfreq(data.shape[1])[None, :]
    kern_col = np.exp(im2pi * t_col / upsample_factor)
    
    return kern_row @ data @ kern_col.T


def dft_upsample_registration(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    upsample_factor: int = 20,
) -> tuple[tuple[float, float], float]:
    """Efficient sub-pixel image registration by cross-power spectrum DFT upsampling.
    
    Guizar-Sicairos, Thurman, & Fienup (Optics Letters 33, 156-158, 2008).
    Returns ((dx, dy), peak_response) where (dx, dy) is the shift of src relative to ref.
    """
    ref_f = ref_img.astype(np.float64)
    src_f = src_img.astype(np.float64)
    h, w = ref_f.shape[:2]

    # Hanning window reduces boundary discontinuity artifacts
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)

    ref_w = (ref_f - np.mean(ref_f)) * window
    src_w = (src_f - np.mean(src_f)) * window

    F_ref = np.fft.fft2(ref_w)
    F_src = np.fft.fft2(src_w)

    image_product = F_src * np.conj(F_ref)
    norm = np.abs(image_product)
    image_product /= np.where(norm > 1e-12, norm, 1.0)

    cross_corr = np.fft.ifft2(image_product)
    maxima = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    midpoints = np.array([h / 2.0, w / 2.0])
    shifts = np.array(maxima, dtype=np.float64)
    for k in range(2):
        if shifts[k] > midpoints[k]:
            shifts[k] -= [h, w][k]

    if upsample_factor > 1:
        ups_size = int(np.ceil(upsample_factor * 1.5))
        dftshift = int(np.trunc(ups_size / 2.0))
        row_offset = dftshift - shifts[0] * upsample_factor
        col_offset = dftshift - shifts[1] * upsample_factor

        CC = _upsampled_dft(
            image_product,
            (ups_size, ups_size),
            upsample_factor,
            (row_offset, col_offset),
        )

        up_max = np.unravel_index(np.argmax(np.abs(CC)), CC.shape)
        up_shifts = (np.array(up_max, dtype=np.float64) - dftshift) / upsample_factor
        shifts = shifts + up_shifts

        peak_val = float(np.abs(CC[up_max]) / (h * w))
    else:
        peak_val = float(np.abs(cross_corr[maxima]) / (h * w))

    # Note: shifts is (dy, dx). Return (dx, dy).
    return (float(shifts[1]), float(shifts[0])), float(np.clip(peak_val, 0.0, 1.0))


# =============================================================================
# 2. Local Intensity Correlation 2D Quadratic Peak Fitting
# =============================================================================

def fit_2d_quadratic_peak(scores_grid: np.ndarray) -> tuple[float, float, bool, str]:
    """Fit a 2D quadric surface f(x, y) = c1*x^2 + c2*y^2 + c3*x*y + c4*x + c5*y + c6.
    
    Verifies that the Hessian matrix is strictly negative definite (local maximum).
    Returns (dx, dy, is_valid, status_message).
    """
    K = scores_grid.shape[0] // 2
    ys, xs = np.mgrid[-K:K+1, -K:K+1]
    xs = xs.flatten().astype(np.float64)
    ys = ys.flatten().astype(np.float64)
    zs = scores_grid.flatten().astype(np.float64)

    A = np.column_stack([xs**2, ys**2, xs * ys, xs, ys, np.ones_like(xs)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
        c1, c2, c3, c4, c5, c6 = coeffs

        # Hessian matrix H = [[2*c1, c3], [c3, 2*c2]]
        H = np.array([[2.0 * c1, c3], [c3, 2.0 * c2]], dtype=np.float64)
        det_H = 4.0 * c1 * c2 - c3**2

        # For local maximum (peak), H must be negative definite: c1 < 0, c2 < 0, det(H) > 0
        if c1 >= 0 or c2 >= 0 or det_H <= 1e-9:
            return 0.0, 0.0, False, "Hessian not negative definite (response surface has no local maximum)"

        grad = np.array([c4, c5], dtype=np.float64)
        delta = -np.linalg.solve(H, grad)
        dx, dy = float(delta[0]), float(delta[1])

        if abs(dx) > 1.0 or abs(dy) > 1.0:
            return 0.0, 0.0, False, "Extremum falls outside local correlation patch radius"

        return dx, dy, True, "Valid 2D quadratic peak"
    except (np.linalg.LinAlgError, ValueError) as err:
        return 0.0, 0.0, False, f"Quadratic fit failed: {err}"


def quadratic_peak(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit a 2D quadratic surface to local normalized cross-correlation response peak."""
    t0 = perf_counter()
    src_img = source_gray.astype(np.float32)
    ref_img = reference_gray.astype(np.float32)
    src = _points(source_points)
    ref = _points(reference_points)
    refined = ref.copy()
    valid = np.ones(len(ref), dtype=bool)
    point_details: list[dict] = []

    patch_size = 11
    radius = 2  # 5x5 correlation window

    for idx, (sx, sy) in enumerate(src):
        x, y = ref[idx]
        pt_t0 = perf_counter()
        source_patch = cv2.getRectSubPix(src_img, (patch_size, patch_size), (float(sx), float(sy)))
        if source_patch is None or float(source_patch.std()) < 1e-3:
            valid[idx] = False
            point_details.append({
                "index": idx + 1,
                "dx": 0.0,
                "dy": 0.0,
                "shift": 0.0,
                "iterations": 1,
                "converged": False,
                "status": "DIVERGED",
                "method": "quadratic_peak",
                "local_correlation": 0.0,
                "reason": "Low source patch variance",
                "time_ms": round((perf_counter() - pt_t0) * 1000, 3),
            })
            continue

        a = source_patch - float(source_patch.mean())
        norm_a = float(np.linalg.norm(a))
        if norm_a < 1e-6:
            valid[idx] = False
            point_details.append({
                "index": idx + 1,
                "dx": 0.0,
                "dy": 0.0,
                "shift": 0.0,
                "iterations": 1,
                "converged": False,
                "status": "DIVERGED",
                "method": "quadratic_peak",
                "local_correlation": 0.0,
                "reason": "Degenerate zero-variance patch",
                "time_ms": round((perf_counter() - pt_t0) * 1000, 3),
            })
            continue

        grid = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        rx_int, ry_int = int(round(x)), int(round(y))

        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                cand = cv2.getRectSubPix(ref_img, (patch_size, patch_size), (float(rx_int + ox), float(ry_int + oy)))
                if cand is not None and float(cand.std()) >= 1e-3:
                    b = cand - float(cand.mean())
                    norm_b = float(np.linalg.norm(b))
                    if norm_b > 1e-6:
                        grid[oy + radius, ox + radius] = float((a * b).sum() / (norm_a * norm_b))

        # Fit 2D quadric surface on the 5x5 grid
        p_dx, p_dy, p_ok, p_msg = fit_2d_quadratic_peak(grid)
        best_idx = np.unravel_index(np.argmax(grid), grid.shape)
        best_oy, best_ox = best_idx[0] - radius, best_idx[1] - radius
        best_corr = float(grid[best_idx])

        if p_ok:
            total_dx = best_ox + p_dx
            total_dy = best_oy + p_dy
            shift_mag = float(np.hypot(total_dx, total_dy))
            if shift_mag <= 1.5:
                refined[idx] = (round(x) + total_dx, round(y) + total_dy)
                valid[idx] = True
                status_str = "CONVERGED"
            else:
                valid[idx] = False
                status_str = "DIVERGED"
        else:
            # Fall back to integer correlation peak if quadratic fit was ambiguous
            if best_corr >= 0.3 and np.hypot(best_ox, best_oy) <= 1.0:
                refined[idx] = (round(x) + best_ox, round(y) + best_oy)
                valid[idx] = True
                status_str = "CONVERGED"
            else:
                valid[idx] = False
                status_str = "DIVERGED"

        dx_final = float(refined[idx, 0] - ref[idx, 0])
        dy_final = float(refined[idx, 1] - ref[idx, 1])
        point_details.append({
            "index": idx + 1,
            "dx": round(dx_final, 4),
            "dy": round(dy_final, 4),
            "shift": round(float(np.hypot(dx_final, dy_final)), 4),
            "iterations": 1,
            "converged": bool(valid[idx]),
            "status": status_str,
            "method": "quadratic_peak",
            "local_correlation": round(best_corr, 4),
            "reason": p_msg if p_ok else "Sub-pixel surface ambiguous; integer peak fallback applied",
            "time_ms": round((perf_counter() - pt_t0) * 1000, 3),
        })

    if return_details:
        return refined, valid, point_details
    return refined, valid


# =============================================================================
# 3. Lucas-Kanade Optical Flow (Point-wise iterative)
# =============================================================================

def lucas_kanade(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]:
    """Refine target positions with iterative LK optical flow using current match as flow."""
    src = _points(source_points).reshape(-1, 1, 2)
    initial_copy = _points(reference_points).copy()
    initial = initial_copy.reshape(-1, 1, 2).copy()
    n_pts = len(src)
    if n_pts == 0:
        empty_ref = initial_copy
        empty_val = np.zeros(0, dtype=bool)
        return (empty_ref, empty_val, []) if return_details else (empty_ref, empty_val)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        1e-4,
    )
    t0 = perf_counter()
    try:
        refined, status, err = cv2.calcOpticalFlowPyrLK(
            source_gray,
            reference_gray,
            src,
            initial,
            winSize=(15, 15),
            maxLevel=3,
            criteria=criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
    except cv2.error:
        refined = initial_copy
        status = np.zeros((n_pts, 1), dtype=np.uint8)
        err = np.zeros((n_pts, 1), dtype=np.float32)

    if refined is None or status is None:
        refined = initial_copy
        status = np.zeros((n_pts, 1), dtype=np.uint8)

    ref_pts = refined.reshape(-1, 2)
    val_pts = status.reshape(-1).astype(bool)
    init_pts = initial_copy.reshape(-1, 2)

    point_details: list[dict] = []
    if return_details:
        src_flat = src.reshape(-1, 2)
        for i in range(n_pts):
            dx = float(ref_pts[i, 0] - init_pts[i, 0])
            dy = float(ref_pts[i, 1] - init_pts[i, 1])
            is_conv = bool(val_pts[i])
            corr = float(1.0 / (1.0 + err[i, 0])) if (err is not None and i < len(err) and np.isfinite(err[i, 0])) else None
            point_details.append({
                "index": i + 1,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "shift": round(float(np.hypot(dx, dy)), 4),
                "iterations": 30 if is_conv else 0,
                "converged": is_conv,
                "status": "CONVERGED" if is_conv else "DIVERGED",
                "method": "lucas_kanade",
                "local_correlation": round(corr, 4) if corr is not None else None,
                "reason": "LK optical flow converged" if is_conv else "LK optical flow tracking failed",
                "time_ms": round((perf_counter() - t0) / max(1, n_pts) * 1000, 3),
            })
        return ref_pts, val_pts, point_details

    return ref_pts, val_pts


# =============================================================================
# 4. Taylor Expansion (Local Brightness Constancy Linearization)
# =============================================================================

def taylor_expansion(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    iterations: int = 4,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]:
    """Solve the local first-order brightness-constancy equation.

    For each correspondence, the reference-image gradient is the Jacobian and
    the source/reference patch difference is the residual. This is the local
    Taylor expansion used by inverse-compositional Lucas–Kanade methods.
    """
    src_img = source_gray.astype(np.float32)
    ref_img = reference_gray.astype(np.float32)
    gx = cv2.Sobel(ref_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_img, cv2.CV_32F, 0, 1, ksize=3)
    src = _points(source_points)
    initial = _points(reference_points)
    refined = initial.copy()
    valid = np.ones(len(refined), dtype=bool)
    radius = 3

    def sample(image: np.ndarray, x: float, y: float) -> float:
        height, width = image.shape[:2]
        if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
            return float("nan")
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        return float(
            image[y0, x0] * (1 - dx) * (1 - dy)
            + image[y0, x0 + 1] * dx * (1 - dy)
            + image[y0 + 1, x0] * (1 - dx) * dy
            + image[y0 + 1, x0 + 1] * dx * dy
        )

    point_details: list[dict] = []

    for idx, (sx, sy) in enumerate(src):
        pt_t0 = perf_counter()
        x, y = refined[idx]
        iters_run = 0
        converged = True
        reason = "Taylor expansion converged"

        for it in range(iterations):
            iters_run += 1
            rows = []
            errors = []
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    sv = sample(src_img, sx + ox, sy + oy)
                    rv = sample(ref_img, x + ox, y + oy)
                    dxv = sample(gx, x + ox, y + oy)
                    dyv = sample(gy, x + ox, y + oy)
                    if np.isfinite([sv, rv, dxv, dyv]).all():
                        rows.append((dxv, dyv))
                        errors.append(sv - rv)
            if len(rows) < 9:
                converged = False
                valid[idx] = False
                reason = "Insufficient valid patch samples within image bounds"
                break
            jacobian = np.asarray(rows, dtype=np.float32)
            residual = np.asarray(errors, dtype=np.float32)
            normal = jacobian.T @ jacobian
            if abs(float(np.linalg.det(normal))) < 1e-6:
                converged = False
                valid[idx] = False
                reason = "Normal matrix is numerically singular (degenerate patch gradient)"
                break
            delta = np.linalg.solve(normal, jacobian.T @ residual)
            delta = np.clip(delta, -1.5, 1.5)
            x += float(delta[0])
            y += float(delta[1])
            if float(np.linalg.norm(delta)) < 1e-3:
                reason = f"Taylor expansion converged in {iters_run} iterations (step < 1e-3 px)"
                break

        refined[idx] = (x, y)
        dx_final = float(x - initial[idx, 0])
        dy_final = float(y - initial[idx, 1])

        # Patch correlation measurement
        src_patch = cv2.getRectSubPix(src_img, (7, 7), (float(sx), float(sy)))
        ref_patch = cv2.getRectSubPix(ref_img, (7, 7), (float(x), float(y)))
        local_corr = None
        if src_patch is not None and ref_patch is not None and src_patch.std() > 1e-3 and ref_patch.std() > 1e-3:
            a = src_patch - float(src_patch.mean())
            b = ref_patch - float(ref_patch.mean())
            local_corr = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))

        point_details.append({
            "index": idx + 1,
            "dx": round(dx_final, 4),
            "dy": round(dy_final, 4),
            "shift": round(float(np.hypot(dx_final, dy_final)), 4),
            "iterations": iters_run,
            "converged": bool(valid[idx] and converged),
            "status": "CONVERGED" if (valid[idx] and converged) else "DIVERGED",
            "method": "taylor_expansion",
            "local_correlation": round(local_corr, 4) if local_corr is not None else None,
            "reason": reason,
            "time_ms": round((perf_counter() - pt_t0) * 1000, 3),
        })

    if return_details:
        return refined, valid, point_details
    return refined, valid


# =============================================================================
# 5. Corner SubPix Refinement
# =============================================================================

def corner_taylor(
    image: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV's gradient/Taylor corner refinement for corner-like features."""
    refined = _points(points).reshape(-1, 1, 2).copy()
    if len(refined) == 0:
        return refined.reshape(-1, 2), np.zeros(0, dtype=bool)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        1e-4,
    )
    try:
        result = cv2.cornerSubPix(
            image.astype(np.float32),
            refined,
            (5, 5),
            (-1, -1),
            criteria,
        )
    except cv2.error:
        return refined.reshape(-1, 2), np.zeros(len(refined), dtype=bool)
    if result is None:
        return refined.reshape(-1, 2), np.zeros(len(refined), dtype=bool)
    result = result.reshape(-1, 2)
    valid = np.isfinite(result).all(axis=1)
    return result, valid


# =============================================================================
# 6. ECC Alignment (Image-wide intensity optimization)
# =============================================================================

def ecc_alignment(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    initial_homography: np.ndarray,
    iterations: int = 120,
) -> tuple[np.ndarray, float | None, bool, str]:
    """Enhanced Correlation Coefficient (ECC) image-wide intensity-based transformation refinement.
    
    Georgios D. Evangelidis and Emmanouil Z. Psarakis (IEEE TPAMI 2008).
    """
    if initial_homography is None or not np.isfinite(initial_homography).all():
        return initial_homography, None, False, "Not applicable for this image pair: Initial geometric transformation is absent or degenerate."

    try:
        src = source_gray.astype(np.float32) / 255.0
        ref = reference_gray.astype(np.float32) / 255.0

        warp = initial_homography.astype(np.float32).copy()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            iterations,
            1e-6,
        )

        warp_inv = np.linalg.inv(warp).astype(np.float32)
        cc, refined_inv = cv2.findTransformECC(
            ref,
            src,
            warp_inv,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            5,
        )
        if cc is None or not np.isfinite(cc) or cc <= 0:
            return initial_homography, None, False, "ECC optimization did not yield positive correlation."

        refined = np.linalg.inv(refined_inv)
        if not np.isfinite(refined).all() or abs(refined[2, 2]) < 1e-12:
            return initial_homography, float(cc), False, "ECC refined matrix inversion was numerically singular."

        refined /= refined[2, 2]
        return refined, float(cc), True, f"ECC converged in {iterations} iterations with correlation coefficient {float(cc):.4f}."
    except (cv2.error, np.linalg.LinAlgError) as err:
        return initial_homography, None, False, f"ECC optimization failed: {err}"


# =============================================================================
# 7. Phase Correlation + Upsampling (Global translation)
# =============================================================================

def phase_correlation_subpixel(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    homography: np.ndarray | None = None,
    min_response: float = 0.03,
    max_shift: float = 5.0,
    upsample_factor: int = 20,
) -> tuple[np.ndarray | None, float | None, bool, tuple[float, float] | None, str, str]:
    """Fourier-domain phase correlation with matrix-multiplication DFT upsampling and telemetry."""
    hw_device = "GPU_CUDA" if is_cuda_available() else "CPU"
    try:
        height, width = reference_gray.shape[:2]
        if homography is not None:
            warped = cv2.warpPerspective(source_gray, homography, (width, height))
        else:
            warped = source_gray

        if is_cuda_available():
            dx, dy, resp = phase_correlation_gpu(warped, reference_gray)
            hw_device = "GPU_CUDA"
        else:
            # First obtain fast integer/coarse shift via cv2.phaseCorrelate
            shift, response = cv2.phaseCorrelate(
                warped.astype(np.float32),
                reference_gray.astype(np.float32),
            )
            if shift is None or not np.isfinite(shift).all():
                return homography, None, False, None, "Phase correlation produced non-finite shift.", hw_device
            resp = float(response) if response is not None else 0.0
            dx, dy = float(shift[0]), float(shift[1])
            hw_device = "CPU"

            # Apply matrix-multiply DFT upsampling around the peak for sub-pixel accuracy
            if upsample_factor > 1 and resp >= min_response:
                try:
                    (up_dx, up_dy), up_resp = dft_upsample_registration(
                        reference_gray, warped, upsample_factor=upsample_factor
                    )
                    # Blend or accept upsampled shift if consistent with coarse shift
                    if abs(up_dx - dx) < 1.0 and abs(up_dy - dy) < 1.0:
                        dx, dy = up_dx, up_dy
                        resp = max(resp, up_resp)
                except Exception:
                    pass  # Retain standard subpixel estimate from phaseCorrelate

        shift_mag = float(np.hypot(dx, dy))
        if resp < min_response:
            return homography, resp, False, (dx, dy), f"Phase response ({resp:.4f}) below threshold ({min_response:.2f}).", hw_device
        if shift_mag > max_shift:
            return homography, resp, False, (dx, dy), f"Phase shift magnitude ({shift_mag:.2f} px) exceeded subpixel limit ({max_shift:.1f} px).", hw_device

        translation = np.asarray(
            [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if homography is not None:
            refined = translation @ np.asarray(homography, dtype=np.float64)
            if not np.isfinite(refined).all() or abs(refined[2, 2]) < 1e-12:
                return homography, resp, False, (dx, dy), "Refined homography is numerically singular.", hw_device
            refined /= refined[2, 2]
        else:
            refined = translation

        return refined, resp, True, (dx, dy), f"Phase correlation + upsampling ({upsample_factor}x) shift ({dx:+.3f}, {dy:+.3f}) px, response {resp:.4f}.", hw_device
    except (cv2.error, ValueError, np.linalg.LinAlgError) as err:
        return homography, None, False, None, f"Phase correlation failed: {err}", hw_device


def phase_translation(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    homography: np.ndarray,
    min_response: float = 0.03,
    max_shift: float = 5.0,
) -> tuple[np.ndarray, float | None, bool, tuple[float, float] | None, str]:
    """Estimate a fractional global translation after the initial geometric warp."""
    ref_h, resp, used, shift, reason, _ = phase_correlation_subpixel(
        source_gray, reference_gray, homography, min_response=min_response, max_shift=max_shift
    )
    return ref_h if ref_h is not None else homography, resp, used, shift, reason


# =============================================================================
# 8. AUTO SELECT BEST REFINEMENT (Comparative Evaluation & Decision Card)
# =============================================================================

def compare_and_select_refinement(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    homography: np.ndarray,
    candidate_methods: list[str] | None = None,
) -> dict:
    """Evaluate candidate refinement methods against measurable residual quality and select the best.

    Selection Criteria:
    1. Method executes and converges without numerical degeneracy.
    2. Resulting transformation is physically plausible.
    3. Method achieves measurable residual RMSE reduction over raw correspondence baseline (>= 0.005 px).
    4. Lowest valid inlier residual RMSE wins with documented numerical justification.
    """
    from app.services.geometry import validate_transform_plausibility

    pts_s = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    pts_r = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    H_base = np.asarray(homography, dtype=np.float64).copy() if homography is not None else None

    # Compute raw baseline error
    raw_rmse = None
    if H_base is not None and len(pts_s) > 0:
        pred_raw = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), H_base).reshape(-1, 2)
        raw_res = np.linalg.norm(pred_raw - pts_r, axis=1)
        raw_rmse = float(np.sqrt(np.mean(raw_res ** 2)))

    if not candidate_methods or "auto" in candidate_methods:
        candidates = ["taylor_expansion", "lucas_kanade", "quadratic_peak", "phase_correlation", "ecc"]
    else:
        candidates = normalize_refinement_methods(candidate_methods)
        if "auto" in candidates:
            candidates.remove("auto")

    ledger: list[dict] = []
    method_point_details: dict[str, list[dict]] = {}
    best_method = "none"
    best_rmse = raw_rmse
    best_reduction = 0.0
    best_target = pts_r.copy()
    best_H = H_base.copy() if H_base is not None else None
    best_reason = "No candidate sub-pixel method achieved measurable RMSE reduction over raw correspondence baseline."

    for method in candidates:
        method_entry = {
            "method": method,
            "display_name": CANONICAL_METHOD_DESCRIPTIONS.get(method, {}).get("display_name", method),
            "category": CANONICAL_METHOD_DESCRIPTIONS.get(method, {}).get("category", "unknown"),
            "attempted": True,
            "succeeded": False,
            "raw_rmse_pixels": round(raw_rmse, 4) if raw_rmse is not None else None,
            "refined_rmse_pixels": None,
            "rmse_reduction_pixels": 0.0,
            "converged_points": 0,
            "convergence_rate": 0.0,
            "mean_shift_pixels": 0.0,
            "shift_stability": None,
            "local_correlation": None,
            "uncertainty_pixels": None,
            "hardware": "GPU_CUDA" if (method == "phase_correlation" and is_cuda_available()) else "CPU",
            "reason": "",
        }

        try:
            if method == "taylor_expansion":
                if len(pts_s) < 4:
                    method_entry["reason"] = f"Not applicable for this image pair: Need at least 4 correspondence points (got {len(pts_s)})."
                    ledger.append(method_entry)
                    continue

                new_target, valid, pt_details = taylor_expansion(source_gray, reference_gray, pts_s, pts_r, return_details=True)
                n_valid = int(np.sum(valid))
                method_entry["converged_points"] = n_valid
                method_entry["convergence_rate"] = round(float(n_valid / len(pts_s)), 3)
                method_point_details[method] = pt_details

                if n_valid >= 4 and H_base is not None:
                    shifts = np.linalg.norm(new_target[valid] - pts_r[valid], axis=1)
                    method_entry["mean_shift_pixels"] = round(float(np.mean(shifts)), 4)
                    method_entry["shift_stability"] = round(float(np.std(shifts)), 4)
                    corrs = [d["local_correlation"] for d in pt_details if d.get("local_correlation") is not None]
                    if corrs:
                        method_entry["local_correlation"] = round(float(np.mean(corrs)), 4)

                    from app.services.geometry import estimate_geometric_model
                    geom_eval = estimate_geometric_model(pts_s[valid], new_target[valid])
                    H_eval = geom_eval.homography if (geom_eval and geom_eval.is_valid) else H_base

                    pred = cv2.perspectiveTransform(pts_s[valid].reshape(-1, 1, 2), H_eval).reshape(-1, 2)
                    m_res = np.linalg.norm(pred - new_target[valid], axis=1)
                    m_rmse = float(np.sqrt(np.mean(m_res ** 2)))
                    reduction = float(raw_rmse - m_rmse) if raw_rmse is not None else 0.0
                    method_entry["succeeded"] = True
                    method_entry["refined_rmse_pixels"] = round(m_rmse, 4)
                    method_entry["rmse_reduction_pixels"] = round(reduction, 4)
                    method_entry["uncertainty_pixels"] = round(float(np.std(m_res) / np.sqrt(len(m_res))), 4) if len(m_res) > 1 else 0.0
                    method_entry["reason"] = f"Taylor linearization: {n_valid}/{len(pts_s)} points converged, shift={method_entry['mean_shift_pixels']:.3f} px."
                    if reduction > best_reduction and m_rmse < (best_rmse if best_rmse is not None else float("inf")):
                        best_method = method
                        best_rmse = m_rmse
                        best_reduction = reduction
                        best_target = new_target
                        best_H = H_eval
                        best_reason = f"Selected Taylor Expansion: residual reduced by {reduction:.3f} px ({raw_rmse:.3f} -> {m_rmse:.3f} px) across {n_valid} points."

            elif method == "lucas_kanade":
                if len(pts_s) < 4:
                    method_entry["reason"] = f"Not applicable for this image pair: Need at least 4 correspondence points (got {len(pts_s)})."
                    ledger.append(method_entry)
                    continue

                new_target, valid, pt_details = lucas_kanade(source_gray, reference_gray, pts_s, pts_r, return_details=True)
                n_valid = int(np.sum(valid))
                method_entry["converged_points"] = n_valid
                method_entry["convergence_rate"] = round(float(n_valid / len(pts_s)), 3)
                method_point_details[method] = pt_details

                if n_valid >= 4 and H_base is not None:
                    shifts = np.linalg.norm(new_target[valid] - pts_r[valid], axis=1)
                    method_entry["mean_shift_pixels"] = round(float(np.mean(shifts)), 4)
                    method_entry["shift_stability"] = round(float(np.std(shifts)), 4)
                    corrs = [d["local_correlation"] for d in pt_details if d.get("local_correlation") is not None]
                    if corrs:
                        method_entry["local_correlation"] = round(float(np.mean(corrs)), 4)

                    from app.services.geometry import estimate_geometric_model
                    geom_eval = estimate_geometric_model(pts_s[valid], new_target[valid])
                    H_eval = geom_eval.homography if (geom_eval and geom_eval.is_valid) else H_base

                    pred = cv2.perspectiveTransform(pts_s[valid].reshape(-1, 1, 2), H_eval).reshape(-1, 2)
                    m_res = np.linalg.norm(pred - new_target[valid], axis=1)
                    m_rmse = float(np.sqrt(np.mean(m_res ** 2)))
                    reduction = float(raw_rmse - m_rmse) if raw_rmse is not None else 0.0
                    method_entry["succeeded"] = True
                    method_entry["refined_rmse_pixels"] = round(m_rmse, 4)
                    method_entry["rmse_reduction_pixels"] = round(reduction, 4)
                    method_entry["uncertainty_pixels"] = round(float(np.std(m_res) / np.sqrt(len(m_res))), 4) if len(m_res) > 1 else 0.0
                    method_entry["reason"] = f"Lucas-Kanade optical flow: {n_valid}/{len(pts_s)} points converged, shift={method_entry['mean_shift_pixels']:.3f} px."
                    if reduction > best_reduction and m_rmse < (best_rmse if best_rmse is not None else float("inf")):
                        best_method = method
                        best_rmse = m_rmse
                        best_reduction = reduction
                        best_target = new_target
                        best_H = H_eval
                        best_reason = f"Selected Lucas-Kanade: residual reduced by {reduction:.3f} px ({raw_rmse:.3f} -> {m_rmse:.3f} px) across {n_valid} points."

            elif method == "quadratic_peak":
                if len(pts_s) < 4:
                    method_entry["reason"] = f"Not applicable for this image pair: Need at least 4 correspondence points (got {len(pts_s)})."
                    ledger.append(method_entry)
                    continue

                new_target, valid, pt_details = quadratic_peak(source_gray, reference_gray, pts_s, pts_r, return_details=True)
                n_valid = int(np.sum(valid))
                method_entry["converged_points"] = n_valid
                method_entry["convergence_rate"] = round(float(n_valid / len(pts_s)), 3)
                method_point_details[method] = pt_details

                if n_valid >= 4 and H_base is not None:
                    shifts = np.linalg.norm(new_target[valid] - pts_r[valid], axis=1)
                    method_entry["mean_shift_pixels"] = round(float(np.mean(shifts)), 4)
                    method_entry["shift_stability"] = round(float(np.std(shifts)), 4)
                    corrs = [d["local_correlation"] for d in pt_details if d.get("local_correlation") is not None]
                    if corrs:
                        method_entry["local_correlation"] = round(float(np.mean(corrs)), 4)

                    from app.services.geometry import estimate_geometric_model
                    geom_eval = estimate_geometric_model(pts_s[valid], new_target[valid])
                    H_eval = geom_eval.homography if (geom_eval and geom_eval.is_valid) else H_base

                    pred = cv2.perspectiveTransform(pts_s[valid].reshape(-1, 1, 2), H_eval).reshape(-1, 2)
                    m_res = np.linalg.norm(pred - new_target[valid], axis=1)
                    m_rmse = float(np.sqrt(np.mean(m_res ** 2)))
                    reduction = float(raw_rmse - m_rmse) if raw_rmse is not None else 0.0
                    method_entry["succeeded"] = True
                    method_entry["refined_rmse_pixels"] = round(m_rmse, 4)
                    method_entry["rmse_reduction_pixels"] = round(reduction, 4)
                    method_entry["uncertainty_pixels"] = round(float(np.std(m_res) / np.sqrt(len(m_res))), 4) if len(m_res) > 1 else 0.0
                    method_entry["reason"] = f"Quadratic peak: {n_valid}/{len(pts_s)} points converged, shift={method_entry['mean_shift_pixels']:.3f} px."
                    if reduction > best_reduction and m_rmse < (best_rmse if best_rmse is not None else float("inf")):
                        best_method = method
                        best_rmse = m_rmse
                        best_reduction = reduction
                        best_target = new_target
                        best_H = H_eval
                        best_reason = f"Selected Quadratic Peak: residual reduced by {reduction:.3f} px ({raw_rmse:.3f} -> {m_rmse:.3f} px) across {n_valid} points."

            elif method == "phase_correlation":
                ref_h, resp, used, shift, reason, hw = phase_correlation_subpixel(source_gray, reference_gray, H_base)
                method_entry["hardware"] = hw
                if used and ref_h is not None:
                    plaus, _, _ = validate_transform_plausibility(ref_h, source_gray.shape, reference_gray.shape)
                    if plaus:
                        pred = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), ref_h).reshape(-1, 2)
                        m_res = np.linalg.norm(pred - pts_r, axis=1)
                        m_rmse = float(np.sqrt(np.mean(m_res ** 2)))
                        reduction = float(raw_rmse - m_rmse) if raw_rmse is not None else 0.0
                        shift_mag = float(np.hypot(shift[0], shift[1])) if shift else 0.0
                        method_entry["succeeded"] = True
                        method_entry["refined_rmse_pixels"] = round(m_rmse, 4)
                        method_entry["rmse_reduction_pixels"] = round(reduction, 4)
                        method_entry["mean_shift_pixels"] = round(shift_mag, 4)
                        method_entry["local_correlation"] = round(resp, 4) if resp else None
                        method_entry["uncertainty_pixels"] = round(float(np.std(m_res) / np.sqrt(len(m_res))), 4) if len(m_res) > 1 else 0.0
                        method_entry["reason"] = reason
                        if reduction > best_reduction and m_rmse < (best_rmse if best_rmse is not None else float("inf")):
                            best_method = method
                            best_rmse = m_rmse
                            best_reduction = reduction
                            best_H = ref_h
                            best_reason = f"Selected Phase Correlation: residual reduced by {reduction:.3f} px ({raw_rmse:.3f} -> {m_rmse:.3f} px), response={resp:.4f}."
                    else:
                        method_entry["reason"] = "Refined transform failed physical plausibility check."
                else:
                    method_entry["reason"] = reason

            elif method == "ecc":
                ref_h, score, used, reason = ecc_alignment(source_gray, reference_gray, H_base)
                if used and ref_h is not None:
                    plaus, _, _ = validate_transform_plausibility(ref_h, source_gray.shape, reference_gray.shape)
                    if plaus:
                        pred = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), ref_h).reshape(-1, 2)
                        m_res = np.linalg.norm(pred - pts_r, axis=1)
                        m_rmse = float(np.sqrt(np.mean(m_res ** 2)))
                        reduction = float(raw_rmse - m_rmse) if raw_rmse is not None else 0.0
                        method_entry["succeeded"] = True
                        method_entry["refined_rmse_pixels"] = round(m_rmse, 4)
                        method_entry["rmse_reduction_pixels"] = round(reduction, 4)
                        method_entry["local_correlation"] = round(score, 4) if score else None
                        method_entry["uncertainty_pixels"] = round(float(np.std(m_res) / np.sqrt(len(m_res))), 4) if len(m_res) > 1 else 0.0
                        method_entry["reason"] = reason
                        if reduction > best_reduction and m_rmse < (best_rmse if best_rmse is not None else float("inf")):
                            best_method = method
                            best_rmse = m_rmse
                            best_reduction = reduction
                            best_H = ref_h
                            best_reason = f"Selected ECC Optimization: residual reduced by {reduction:.3f} px ({raw_rmse:.3f} -> {m_rmse:.3f} px, correlation={score:.4f})."
                    else:
                        method_entry["reason"] = "Refined ECC transform failed physical plausibility check."
                else:
                    method_entry["reason"] = reason

        except Exception as exc:
            method_entry["reason"] = f"Method execution error: {exc}"

        ledger.append(method_entry)

    status = "SUBPIXEL_VALIDATED" if best_reduction >= 0.005 else "SUBPIXEL_NOT_VALIDATED"
    if status == "SUBPIXEL_NOT_VALIDATED" and not best_reason.startswith("Selected"):
        best_reason = "No candidate sub-pixel method achieved measurable RMSE reduction over raw correspondence baseline."

    # Per-point telemetry for the selected method
    selected_point_details = method_point_details.get(best_method)
    if not selected_point_details and len(pts_s) > 0:
        # Create baseline point details from best_target / best_H
        selected_point_details = []
        for i in range(len(pts_s)):
            dx = float(best_target[i, 0] - pts_r[i, 0])
            dy = float(best_target[i, 1] - pts_r[i, 1])
            selected_point_details.append({
                "index": i + 1,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "shift": round(float(np.hypot(dx, dy)), 4),
                "iterations": 1,
                "converged": True,
                "status": "CONVERGED" if status == "SUBPIXEL_VALIDATED" else "UNREFINED",
                "method": best_method,
                "local_correlation": None,
                "reason": best_reason,
            })

    return {
        "selected_method": best_method,
        "candidate_methods": candidates,
        "raw_rmse_pixels": round(raw_rmse, 4) if raw_rmse is not None else None,
        "refined_rmse_pixels": round(best_rmse, 4) if best_rmse is not None else None,
        "rmse_improvement_pixels": round(best_reduction, 4),
        "status": status,
        "reason": best_reason,
        "refined_reference_points": best_target,
        "homography": best_H,
        "comparison_ledger": ledger,
        "point_details": selected_point_details or [],
    }