"""Robust geometric model selection, outlier rejection, and overlap validation.

Evaluates candidate transformations (Homography vs Affine) to prevent
projective over-fitting and degeneracy. Employs modern robust estimators
(USAC_MAGSAC with RANSAC fallback) and verifies valid physical image overlap.

Includes transformation plausibility validation: scale, rotation, shear,
translation magnitude, and determinant are independently checked against
physically reasonable bounds for planetary image registration.
"""

from dataclasses import dataclass, field
import math
import cv2
import numpy as np


@dataclass
class GeometricEstimationResult:
    homography: np.ndarray
    inlier_mask: np.ndarray
    model_type: str  # "homography" or "affine"
    estimator_used: str  # "USAC_MAGSAC" or "RANSAC"
    inlier_count: int
    inlier_ratio: float
    rmse_pixels: float | None
    is_valid: bool
    overlap_valid: bool
    overlap_ratio: float
    failure_reason: str | None = None
    transform_decomposition: dict | None = None
    diagnostics: dict = field(default_factory=dict)


def _check_quadrilateral_convexity(corners: np.ndarray) -> tuple[bool, float]:
    """Check that 4 2D points form a non-self-intersecting, convex quadrilateral with positive area."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4 or not np.isfinite(pts).all():
        return False, 0.0

    # Shoelace formula for signed area
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(area) < 1.0:
        return False, 0.0

    # Cross products of consecutive edges to verify strict convexity and consistent winding
    cross_signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        d1 = p2 - p1
        d2 = p3 - p2
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) > 1e-5:
            cross_signs.append(cross > 0)

    is_convex = len(cross_signs) == 4 and (all(cross_signs) or not any(cross_signs))
    return is_convex, abs(area)


def decompose_transform(H: np.ndarray) -> dict:
    """Decompose a 3×3 homography into interpretable geometric parameters.

    Extracts: scale_x, scale_y, rotation (degrees), shear, translation (px),
    determinant, and perspective distortion terms.
    """
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all():
        return {"valid": False, "reason": "Matrix is None, wrong shape, or contains non-finite values."}

    # Normalize so H[2,2] = 1
    if abs(H[2, 2]) < 1e-12:
        return {"valid": False, "reason": "H[2,2] is near zero — degenerate projective matrix."}
    Hn = H / H[2, 2]

    # Extract affine part (top-left 2×2)
    A = Hn[:2, :2].astype(np.float64)
    tx, ty = float(Hn[0, 2]), float(Hn[1, 2])

    # SVD decomposition of affine part: A = U S V^T
    # gives rotation and scale
    det = float(np.linalg.det(A))
    if abs(det) < 1e-12:
        return {"valid": False, "reason": "Affine part has zero determinant — singular transformation."}

    # Polar decomposition: A = R * S_stretch
    # where R is rotation and S_stretch is symmetric (scale + shear)
    U, s_vals, Vt = np.linalg.svd(A)
    scale_x = float(s_vals[0])
    scale_y = float(s_vals[1])

    # Rotation angle from polar decomposition
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # Flip to ensure proper rotation (not reflection)
        U[:, -1] *= -1
        R = U @ Vt
    rotation_rad = float(math.atan2(R[1, 0], R[0, 0]))
    rotation_deg = float(math.degrees(rotation_rad))

    # Shear: after removing rotation, the off-diagonal element
    try:
        R_inv = np.linalg.inv(R)
        S_stretch = R_inv @ A
        shear = float(S_stretch[0, 1]) if abs(S_stretch[0, 0]) > 1e-9 else 0.0
    except np.linalg.LinAlgError:
        shear = 0.0

    # Translation magnitude
    translation_magnitude = float(math.sqrt(tx * tx + ty * ty))

    # Perspective distortion
    perspective_x = float(abs(Hn[2, 0]))
    perspective_y = float(abs(Hn[2, 1]))

    return {
        "valid": True,
        "scale_x": round(scale_x, 6),
        "scale_y": round(scale_y, 6),
        "scale_ratio": round(scale_x / max(scale_y, 1e-9), 6),
        "rotation_degrees": round(rotation_deg, 4),
        "rotation_radians": round(rotation_rad, 6),
        "shear": round(shear, 6),
        "translation_x": round(tx, 3),
        "translation_y": round(ty, 3),
        "translation_magnitude_px": round(translation_magnitude, 3),
        "determinant": round(det, 6),
        "perspective_x": round(perspective_x, 8),
        "perspective_y": round(perspective_y, 8),
    }


def validate_transform_plausibility(
    H: np.ndarray,
    source_shape: tuple[int, int] | None = None,
    reference_shape: tuple[int, int] | None = None,
    max_scale: float = 5.0,
    min_scale: float = 0.15,
    max_rotation_deg: float = 45.0,
    max_shear: float = 0.5,
    max_translation_fraction: float = 3.0,
    max_perspective: float = 0.01,
) -> tuple[bool, list[str], dict]:
    """Validate whether a transformation is physically plausible for planetary image registration.

    Returns:
        (is_plausible, list_of_issues, decomposition_dict)

    Checks:
    - Scale within [min_scale, max_scale]
    - Scale anisotropy (scale_x / scale_y) within reasonable bounds
    - Rotation within ±max_rotation_deg
    - Shear below max_shear
    - Translation magnitude relative to image size
    - Determinant positive and bounded
    - Perspective distortion bounded
    """
    decomp = decompose_transform(H)
    issues: list[str] = []

    if not decomp.get("valid", False):
        return False, [decomp.get("reason", "Transform decomposition failed.")], decomp

    sx = decomp["scale_x"]
    sy = decomp["scale_y"]
    rot = abs(decomp["rotation_degrees"])
    shear_val = abs(decomp["shear"])
    det = decomp["determinant"]
    tx_mag = decomp["translation_magnitude_px"]
    px = decomp["perspective_x"]
    py = decomp["perspective_y"]

    # Scale checks
    if sx > max_scale or sy > max_scale:
        issues.append(f"Implausible magnification: scale=({sx:.2f}, {sy:.2f}), max allowed={max_scale:.1f}.")
    if sx < min_scale or sy < min_scale:
        issues.append(f"Implausible minification: scale=({sx:.2f}, {sy:.2f}), min allowed={min_scale:.2f}.")

    # Scale anisotropy (aspect ratio distortion)
    scale_ratio = decomp["scale_ratio"]
    if scale_ratio > 2.5 or scale_ratio < 0.4:
        issues.append(f"Extreme scale anisotropy: ratio={scale_ratio:.3f} (expected near 1.0).")

    # Rotation check
    if rot > max_rotation_deg:
        issues.append(f"Large rotation: {decomp['rotation_degrees']:.1f}°, max allowed=±{max_rotation_deg:.0f}°.")

    # Shear check
    if shear_val > max_shear:
        issues.append(f"Excessive shear: {shear_val:.4f}, max allowed={max_shear:.2f}.")

    # Translation magnitude relative to image size
    if source_shape is not None:
        img_diag = math.sqrt(source_shape[0] ** 2 + source_shape[1] ** 2)
        if tx_mag > max_translation_fraction * img_diag:
            issues.append(
                f"Translation ({tx_mag:.0f} px) exceeds {max_translation_fraction:.0f}× image diagonal ({img_diag:.0f} px)."
            )

    # Determinant: should be positive and not extreme
    if det <= 0:
        issues.append(f"Negative or zero determinant ({det:.4f}) — transformation includes reflection.")
    elif det > max_scale * max_scale * 2:
        issues.append(f"Determinant ({det:.4f}) implies extreme area expansion.")
    elif det < min_scale * min_scale * 0.5:
        issues.append(f"Determinant ({det:.6f}) implies extreme area contraction.")

    # Perspective distortion
    if px > max_perspective or py > max_perspective:
        issues.append(f"High perspective distortion: h31={px:.6f}, h32={py:.6f}, max={max_perspective:.4f}.")

    # Projected footprint and canvas expansion checks (when image shapes are available)
    if source_shape is not None:
        sh, sw = source_shape[:2]
        src_corners = np.array(
            [[0.0, 0.0], [float(sw), 0.0], [float(sw), float(sh)], [0.0, float(sh)]],
            dtype=np.float32,
        )
        try:
            proj_corners = cv2.perspectiveTransform(
                src_corners.reshape(-1, 1, 2), H.astype(np.float64)
            ).reshape(-1, 2)
        except cv2.error as err:
            issues.append(f"Perspective transformation of image corners failed: {err}")
            proj_corners = None

        if proj_corners is not None and np.isfinite(proj_corners).all():
            # Check projective denominator horizon (avoid division-by-near-zero singularities)
            test_pts = np.array([
                [0.0, 0.0], [float(sw), 0.0], [float(sw), float(sh)], [0.0, float(sh)],
                [float(sw) / 2.0, float(sh) / 2.0]
            ], dtype=np.float64)
            denoms = H[2, 0] * test_pts[:, 0] + H[2, 1] * test_pts[:, 1] + H[2, 2]
            if np.any(denoms <= 0.05) or np.any(denoms >= 6.0):
                issues.append(
                    f"Homography approaches projective horizon or has extreme denominator variation (min={np.min(denoms):.3f}, max={np.max(denoms):.3f})."
                )

            # Check quadrilateral convexity and area
            is_convex, proj_area = _check_quadrilateral_convexity(proj_corners)
            if not is_convex:
                issues.append("Projected source quadrilateral is degenerate or self-intersecting (folds over).")
            elif proj_area < 50.0:
                issues.append(f"Projected source area collapsed ({proj_area:.1f} px²).")

            min_x = float(np.min(proj_corners[:, 0]))
            min_y = float(np.min(proj_corners[:, 1]))
            max_x = float(np.max(proj_corners[:, 0]))
            max_y = float(np.max(proj_corners[:, 1]))
            proj_w = max_x - min_x
            proj_h = max_y - min_y

            src_area = float(sw * sh)
            area_ratio = proj_area / src_area if src_area > 0 else 1.0

            decomp["projected_width"] = round(proj_w, 2)
            decomp["projected_height"] = round(proj_h, 2)
            decomp["projected_area"] = round(proj_area, 2)
            decomp["projected_bbox"] = [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)]
            decomp["projected_corners"] = proj_corners.round(2).tolist()
            decomp["footprint_area_ratio"] = round(area_ratio, 4)

            if reference_shape is not None:
                rh, rw = reference_shape[:2]
                ref_area = float(rw * rh)
                decomp["reference_width"] = rw
                decomp["reference_height"] = rh
                expansion_factor = max(proj_w / max(rw, 1), proj_h / max(rh, 1))
                decomp["canvas_expansion_factor"] = round(expansion_factor, 3)
                decomp["ref_area_ratio"] = round(proj_area / ref_area if ref_area > 0 else 0.0, 4)

                if expansion_factor > 6.0:
                    issues.append(
                        f"Excessive canvas expansion: projected dimension is {expansion_factor:.1f}x reference image dimension."
                    )
                if min_x < -4.0 * rw or max_x > 5.0 * rw or min_y < -4.0 * rh or max_y > 5.0 * rh:
                    issues.append(
                        f"Projected coordinates are extreme (x in [{min_x:.0f}, {max_x:.0f}], y in [{min_y:.0f}, {max_y:.0f}])."
                    )

                # Compute IoU with reference box
                ref_corners = np.array(
                    [[0.0, 0.0], [float(rw), 0.0], [float(rw), float(rh)], [0.0, float(rh)]],
                    dtype=np.float32,
                )
                try:
                    intersect_area, _ = cv2.intersectConvexConvex(
                        proj_corners.astype(np.float32),
                        ref_corners.astype(np.float32),
                    )
                except cv2.error:
                    intersect_area = 0.0

                union_area = proj_area + ref_area - intersect_area
                iou = float(intersect_area / union_area) if union_area > 0 else 0.0
                decomp["intersection_area"] = round(intersect_area, 2)
                decomp["iou"] = round(iou, 4)

                if intersect_area <= 0.0:
                    issues.append("Projected source footprint has zero intersection with reference image.")
                else:
                    source_overlap_ratio = intersect_area / proj_area if proj_area > 0 else 0.0
                    fraction_of_ref = proj_area / ref_area if ref_area > 0 else 0.0
                    if source_overlap_ratio < 0.02:
                        issues.append(
                            f"Projected source overlap ({source_overlap_ratio * 100:.2f}%) is below operational threshold."
                        )
                    elif fraction_of_ref < 0.005 and source_overlap_ratio < 0.10:
                        issues.append(
                            f"Projected source footprint occupies only {fraction_of_ref * 100:.2f}% of reference image area."
                        )
        elif proj_corners is not None:
            issues.append("Projected source corners contain NaN or Infinite coordinates.")

    decomp["perspective_distortion"] = max(px, py)
    decomp["is_plausible"] = len(issues) == 0
    decomp["issues"] = issues

    is_plausible = len(issues) == 0
    return is_plausible, issues, decomp


def validate_overlap(
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    homography: np.ndarray,
    min_overlap_ratio: float = 0.02,
) -> tuple[bool, float, str]:
    """Verify that the projected source image overlaps the reference raster meaningfully.

    Checks:
    1. Projected corners are finite and form a convex quadrilateral (no self-intersecting bowtie).
    2. Area is positive and non-degenerate.
    3. Overlap area with reference bounding box exceeds the minimum threshold.
    """
    if homography is None or not np.isfinite(homography).all():
        return False, 0.0, "Homography contains NaN or non-finite coefficients."

    sh, sw = source_shape[:2]
    rh, rw = reference_shape[:2]

    src_corners = np.array(
        [[0.0, 0.0], [float(sw), 0.0], [float(sw), float(sh)], [0.0, float(sh)]],
        dtype=np.float32,
    )
    ref_corners = np.array(
        [[0.0, 0.0], [float(rw), 0.0], [float(rw), float(rh)], [0.0, float(rh)]],
        dtype=np.float32,
    )

    try:
        proj_corners = cv2.perspectiveTransform(
            src_corners.reshape(-1, 1, 2), homography.astype(np.float64)
        ).reshape(-1, 2)
    except cv2.error as err:
        return False, 0.0, f"Perspective transformation failed: {err}"

    if not np.isfinite(proj_corners).all():
        return False, 0.0, "Projected source corners contain NaN or Infinite coordinates."

    is_convex, proj_area = _check_quadrilateral_convexity(proj_corners)
    if not is_convex:
        return False, 0.0, "Projected source boundaries fold over or self-intersect (perspective degeneracy)."

    # Compute intersection of projected source and reference box
    try:
        # cv2.intersectConvexConvex expects float32 polygons
        intersect_area, _ = cv2.intersectConvexConvex(
            proj_corners.astype(np.float32),
            ref_corners.astype(np.float32),
        )
    except cv2.error:
        intersect_area = 0.0

    src_area = float(sw * sh)
    overlap_ratio = float(intersect_area / src_area) if src_area > 0 else 0.0

    if intersect_area <= 0.0:
        return False, 0.0, "Transformed source image lies entirely outside the reference image."
    if overlap_ratio < min_overlap_ratio:
        return (
            False,
            overlap_ratio,
            f"Overlap area ({overlap_ratio * 100:.2f}%) is below minimum operational threshold ({min_overlap_ratio * 100:.1f}%).",
        )

    return True, overlap_ratio, f"Valid image overlap: {overlap_ratio * 100:.2f}%."


def compute_residual_statistics(residuals: np.ndarray) -> dict:
    """Compute comprehensive statistical metrics over correspondence reprojection errors."""
    if len(residuals) == 0:
        return {
            "count": 0,
            "rmse": None,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
            "min": None,
            "std": None,
            "uncertainty": None,
        }
    res = np.asarray(residuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mean_val = float(np.mean(res))
    med_val = float(np.median(res))
    p90_val = float(np.percentile(res, 90))
    max_val = float(np.max(res))
    min_val = float(np.min(res))
    std_val = float(np.std(res))
    unc_val = float(std_val / math.sqrt(len(res))) if len(res) > 1 else 0.0
    return {
        "count": len(res),
        "rmse": round(rmse, 4),
        "mean": round(mean_val, 4),
        "median": round(med_val, 4),
        "p90": round(p90_val, 4),
        "max": round(max_val, 4),
        "min": round(min_val, 4),
        "std": round(std_val, 4),
        "uncertainty": round(unc_val, 4),
    }


def estimate_geometric_model(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    source_shape: tuple[int, int] | None = None,
    reference_shape: tuple[int, int] | None = None,
    ransac_threshold: float = 3.0,
    prefer_affine: bool = False,
    model_type: str = "auto",
) -> GeometricEstimationResult:
    """Robust geometric model estimation with model selection and sanity checks.

    Evaluates:
    - Homography (8-DOF) using USAC_MAGSAC with RANSAC fallback.
    - Affine (6-DOF) using robust RANSAC estimation.
    - Similarity (4-DOF, scale/rotation/translation) using RANSAC.
    - Translation (2-DOF, rigid shift) using robust median consensus.

    Supports explicit model request ("homography", "affine", "similarity", "translation")
    or "auto" hierarchical model selection.
    """
    pts_s = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    pts_r = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)

    n_pts = len(pts_s)
    if n_pts < 3:
        raise ValueError(f"At least 3 point correspondences are required (got {n_pts}).")

    diagnostics: dict = {
        "candidate_points": n_pts,
        "ransac_threshold": ransac_threshold,
        "requested_model_type": model_type,
        "models_evaluated": {},
    }

    # 1. Estimate Translation Model (2-DOF)
    H_trans = None
    mask_trans = None
    rmse_trans = None
    err_trans = None
    try:
        deltas = pts_r - pts_s
        dx_init = float(np.median(deltas[:, 0]))
        dy_init = float(np.median(deltas[:, 1]))
        init_err = np.linalg.norm(pts_s + np.array([dx_init, dy_init], dtype=np.float32) - pts_r, axis=1)
        raw_mask_trans = init_err <= ransac_threshold
        if np.sum(raw_mask_trans) >= 3:
            dx = float(np.mean(deltas[raw_mask_trans, 0]))
            dy = float(np.mean(deltas[raw_mask_trans, 1]))
        else:
            dx, dy = dx_init, dy_init
        H_trans = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        err_trans = np.linalg.norm(pts_s + np.array([dx, dy], dtype=np.float32) - pts_r, axis=1)
        mask_trans = err_trans <= ransac_threshold
        inliers_trans = int(np.sum(mask_trans))
        if inliers_trans > 0:
            rmse_trans = float(np.sqrt(np.mean(err_trans[mask_trans] ** 2)))
            diagnostics["models_evaluated"]["translation"] = {
                "inliers": inliers_trans,
                "inlier_ratio": float(inliers_trans / n_pts),
                "rmse_pixels": rmse_trans,
                "shift": [round(dx, 4), round(dy, 4)],
            }
    except Exception as err:
        diagnostics["models_evaluated"]["translation_error"] = str(err)

    # 2. Estimate Similarity Model (4-DOF: rotation, scale, translation)
    H_sim = None
    mask_sim = None
    rmse_sim = None
    err_sim = None
    if n_pts >= 3:
        try:
            M_sim, mask_sim_raw = cv2.estimateAffinePartial2D(
                pts_s,
                pts_r,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=5000,
                confidence=0.995,
            )
            if M_sim is not None and mask_sim_raw is not None:
                H_sim = np.eye(3, dtype=np.float64)
                H_sim[:2, :] = M_sim
                mask_sim = mask_sim_raw.ravel().astype(bool)
                inliers_sim = int(np.sum(mask_sim))
                if inliers_sim > 0:
                    pred_sim = cv2.transform(pts_s.reshape(-1, 1, 2), M_sim).reshape(-1, 2)
                    err_sim = np.linalg.norm(pred_sim - pts_r, axis=1)
                    rmse_sim = float(np.sqrt(np.mean(err_sim[mask_sim] ** 2)))
                    diagnostics["models_evaluated"]["similarity"] = {
                        "inliers": inliers_sim,
                        "inlier_ratio": float(inliers_sim / n_pts),
                        "rmse_pixels": rmse_sim,
                    }
        except cv2.error as err:
            diagnostics["models_evaluated"]["similarity_error"] = str(err)

    # 3. Estimate Affine Model (6-DOF)
    H_aff = None
    mask_aff = None
    rmse_aff = None
    err_aff = None
    if n_pts >= 4:
        try:
            M_aff, mask_aff_raw = cv2.estimateAffine2D(
                pts_s,
                pts_r,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=5000,
                confidence=0.995,
            )
            if M_aff is not None and mask_aff_raw is not None:
                H_aff = np.eye(3, dtype=np.float64)
                H_aff[:2, :] = M_aff
                mask_aff = mask_aff_raw.ravel().astype(bool)
                inliers_aff = int(np.sum(mask_aff))
                if inliers_aff > 0:
                    pred_aff = cv2.transform(pts_s.reshape(-1, 1, 2), M_aff).reshape(-1, 2)
                    err_aff = np.linalg.norm(pred_aff - pts_r, axis=1)
                    rmse_aff = float(np.sqrt(np.mean(err_aff[mask_aff] ** 2)))
                    diagnostics["models_evaluated"]["affine"] = {
                        "inliers": inliers_aff,
                        "inlier_ratio": float(inliers_aff / n_pts),
                        "rmse_pixels": rmse_aff,
                    }
        except cv2.error as err:
            diagnostics["models_evaluated"]["affine_error"] = str(err)

    # 4. Estimate Homography Model (8-DOF) using USAC_MAGSAC (fallback to RANSAC)
    H_hom = None
    mask_hom = None
    estimator_used = "USAC_MAGSAC"
    rmse_hom = None
    err_hom = None

    if n_pts >= 4:
        for est_code, est_name in (
            (getattr(cv2, "USAC_MAGSAC", cv2.RANSAC), "USAC_MAGSAC"),
            (cv2.RANSAC, "RANSAC"),
        ):
            try:
                candidate_H, candidate_mask = cv2.findHomography(
                    pts_s,
                    pts_r,
                    est_code,
                    ransac_threshold,
                    maxIters=10000,
                    confidence=0.995,
                )
                if candidate_H is not None and candidate_mask is not None and np.isfinite(candidate_H).all():
                    H_hom = candidate_H / candidate_H[2, 2]
                    mask_hom = candidate_mask.ravel().astype(bool)
                    estimator_used = est_name
                    inliers_hom = int(np.sum(mask_hom))
                    if inliers_hom > 0:
                        pred_hom = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), H_hom).reshape(-1, 2)
                        err_hom = np.linalg.norm(pred_hom - pts_r, axis=1)
                        rmse_hom = float(np.sqrt(np.mean(err_hom[mask_hom] ** 2)))
                        diagnostics["models_evaluated"]["homography"] = {
                            "estimator": est_name,
                            "inliers": inliers_hom,
                            "inlier_ratio": float(inliers_hom / n_pts),
                            "rmse_pixels": rmse_hom,
                        }
                    break
            except (cv2.error, ValueError):
                continue

    # 5. Assess Stability and Distortion of Homography
    homography_stable = False
    hom_plausibility_issues: list[str] = []
    hom_decomposition: dict | None = None
    if H_hom is not None:
        try:
            # Condition number check
            s = np.linalg.svd(H_hom)[1]
            cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
            diagnostics["homography_conditioning"] = cond

            # Projective distortion terms
            proj_x = abs(float(H_hom[2, 0]))
            proj_y = abs(float(H_hom[2, 1]))
            diagnostics["projective_distortion"] = {"h31": proj_x, "h32": proj_y}

            # Determinant of affine/linear part
            det = float(np.linalg.det(H_hom[:2, :2]))
            diagnostics["linear_determinant"] = det

            # Full plausibility validation (scale, rotation, shear, translation)
            plausible, plaus_issues, decomp = validate_transform_plausibility(
                H_hom,
                source_shape=source_shape,
                reference_shape=reference_shape,
            )
            hom_plausibility_issues = plaus_issues
            hom_decomposition = decomp
            diagnostics["homography_plausibility"] = {
                "plausible": plausible,
                "issues": plaus_issues,
                "decomposition": decomp,
            }

            # Homography is stable if condition number is not extreme,
            # projective distortion is bounded, determinant is positive,
            # AND transformation is physically plausible
            if cond < 1e5 and proj_x < 0.005 and proj_y < 0.005 and 0.05 < det < 20.0 and plausible:
                homography_stable = True
        except (ValueError, np.linalg.LinAlgError):
            homography_stable = False

    # 6. Model Selection Logic
    target_model = str(model_type or "auto").strip().lower()
    if prefer_affine and target_model == "auto":
        target_model = "affine"

    selected_model = "homography"
    selected_H = H_hom
    selected_mask = mask_hom
    selected_rmse = rmse_hom
    selected_errors = err_hom
    selected_decomposition = hom_decomposition

    if target_model == "translation" and H_trans is not None and mask_trans is not None and np.sum(mask_trans) >= 3:
        selected_model = "translation"
        selected_H = H_trans
        selected_mask = mask_trans
        selected_rmse = rmse_trans
        selected_errors = err_trans
        estimator_used = "MEDIAN_CONSENSUS"
        _, _, selected_decomposition = validate_transform_plausibility(H_trans, source_shape, reference_shape)
        diagnostics["selection_reason"] = "Explicitly selected Translation (2-DOF) model."
    elif target_model == "similarity" and H_sim is not None and mask_sim is not None and np.sum(mask_sim) >= 3:
        selected_model = "similarity"
        selected_H = H_sim
        selected_mask = mask_sim
        selected_rmse = rmse_sim
        selected_errors = err_sim
        estimator_used = "RANSAC"
        _, _, selected_decomposition = validate_transform_plausibility(H_sim, source_shape, reference_shape)
        diagnostics["selection_reason"] = "Explicitly selected Similarity (4-DOF) model."
    elif target_model == "affine" and H_aff is not None and mask_aff is not None and np.sum(mask_aff) >= 4:
        selected_model = "affine"
        selected_H = H_aff
        selected_mask = mask_aff
        selected_rmse = rmse_aff
        selected_errors = err_aff
        estimator_used = "RANSAC"
        _, _, selected_decomposition = validate_transform_plausibility(H_aff, source_shape, reference_shape)
        diagnostics["selection_reason"] = "Explicitly selected Affine (6-DOF) model."
    elif target_model == "homography" and H_hom is not None and mask_hom is not None and np.sum(mask_hom) >= 4:
        selected_model = "homography"
        selected_H = H_hom
        selected_mask = mask_hom
        selected_rmse = rmse_hom
        selected_errors = err_hom
        diagnostics["selection_reason"] = "Explicitly selected Homography (8-DOF) model."
    else:
        # Hierarchical auto-selection
        if homography_stable and H_hom is not None and mask_hom is not None and np.sum(mask_hom) >= 8:
            selected_model = "homography"
            selected_H = H_hom
            selected_mask = mask_hom
            selected_rmse = rmse_hom
            selected_errors = err_hom
            diagnostics["selection_reason"] = "Homography satisfies conditioning and projective constraints."
        elif H_aff is not None and mask_aff is not None and np.sum(mask_aff) >= 4:
            selected_model = "affine"
            selected_H = H_aff
            selected_mask = mask_aff
            selected_rmse = rmse_aff
            selected_errors = err_aff
            estimator_used = "RANSAC"
            _, _, selected_decomposition = validate_transform_plausibility(H_aff, source_shape, reference_shape)
            reason_parts = ["Homography exhibited projective instability or insufficient inliers (<8)"]
            if hom_plausibility_issues:
                reason_parts.append(f"issues: {'; '.join(hom_plausibility_issues)}")
            reason_parts.append("selected robust Affine model.")
            diagnostics["selection_reason"] = " — ".join(reason_parts)
        elif H_sim is not None and mask_sim is not None and np.sum(mask_sim) >= 3:
            selected_model = "similarity"
            selected_H = H_sim
            selected_mask = mask_sim
            selected_rmse = rmse_sim
            selected_errors = err_sim
            estimator_used = "RANSAC"
            _, _, selected_decomposition = validate_transform_plausibility(H_sim, source_shape, reference_shape)
            diagnostics["selection_reason"] = "Homography and Affine failed; selected robust Similarity model."
        elif H_trans is not None and mask_trans is not None and np.sum(mask_trans) >= 3:
            selected_model = "translation"
            selected_H = H_trans
            selected_mask = mask_trans
            selected_rmse = rmse_trans
            selected_errors = err_trans
            estimator_used = "MEDIAN_CONSENSUS"
            _, _, selected_decomposition = validate_transform_plausibility(H_trans, source_shape, reference_shape)
            diagnostics["selection_reason"] = "Geometric degrees of freedom collapsed; selected robust Translation model."
        elif H_hom is not None and mask_hom is not None:
            selected_model = "homography"
            selected_H = H_hom
            selected_mask = mask_hom
            selected_rmse = rmse_hom
            selected_errors = err_hom
            diagnostics["selection_reason"] = "Fallback to Homography."
        else:
            raise ValueError("All candidate geometric models (Homography, Affine, Similarity, Translation) failed.")

    inlier_residuals = selected_errors[selected_mask] if (selected_errors is not None and selected_mask is not None and np.any(selected_mask)) else np.array([])
    residual_stats = compute_residual_statistics(inlier_residuals)
    diagnostics["residual_analysis"] = residual_stats

    inlier_count = int(np.sum(selected_mask))
    inlier_ratio = float(inlier_count / n_pts) if n_pts > 0 else 0.0

    # 5. Overlap Validation
    overlap_valid = True
    overlap_ratio = 1.0
    failure_reason = None
    if source_shape is not None and reference_shape is not None:
        overlap_valid, overlap_ratio, reason = validate_overlap(
            source_shape, reference_shape, selected_H
        )
        diagnostics["overlap_check"] = {
            "valid": overlap_valid,
            "overlap_ratio": overlap_ratio,
            "note": reason,
        }
        if not overlap_valid:
            failure_reason = reason

    # Check for plausibility failure as an independent validity signal
    plausibility_valid = True
    if selected_decomposition and not selected_decomposition.get("valid", True):
        plausibility_valid = False
        if failure_reason is None:
            failure_reason = selected_decomposition.get("reason", "Transform decomposition failed.")

    return GeometricEstimationResult(
        homography=selected_H,
        inlier_mask=selected_mask,
        model_type=selected_model,
        estimator_used=estimator_used,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        rmse_pixels=selected_rmse,
        is_valid=inlier_count >= 4 and selected_H is not None and overlap_valid and plausibility_valid,
        overlap_valid=overlap_valid,
        overlap_ratio=overlap_ratio,
        failure_reason=failure_reason,
        transform_decomposition=selected_decomposition,
        diagnostics=diagnostics,
    )
