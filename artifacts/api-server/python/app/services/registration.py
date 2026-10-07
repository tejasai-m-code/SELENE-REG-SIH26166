import cv2
import numpy as np

from app.services.subpixel import (
    GLOBAL_REFINEMENT_METHODS,
    POINT_REFINEMENT_METHODS,
    compare_and_select_refinement,
    corner_taylor,
    lucas_kanade,
    normalize_refinement_method,
    normalize_refinement_methods,
    phase_translation,
    quadratic_peak,
    taylor_expansion,
)


def estimate_homography(source_points, reference_points, ransac_threshold=3.0):
    if len(source_points) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        ransac_threshold,
        maxIters=10000,
        confidence=0.995
    )

    if H is None or mask is None:
        raise ValueError("Homography estimation failed.")

    mask = mask.ravel().astype(bool)
    return H, mask


def refine_with_ecc(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    initial_homography: np.ndarray,
    iterations: int = 100,
) -> tuple[np.ndarray, float | None, bool, str]:
    """Optional intensity-based image-wide refinement using ECC.

    Convention verification:
    findTransformECC(templateImage, inputImage, warpMatrix, ...) finds warpMatrix
    such that warping inputImage to templateImage aligns with templateImage.
    Here:
      template = reference_gray
      input = source_gray
    The warpMatrix maps template/reference coords -> input/source coords (inverse warp).
    Starting from initial source->reference H, we invert it: warp_inv = inv(H).
    The refined source->reference H is inv(refined_inv).
    """
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
            5,  # gaussFiltSize=5 provides necessary smoothing for stable gradients
        )
        if cc is None or not np.isfinite(cc) or cc <= 0:
            return initial_homography, None, False, "ECC optimization did not yield positive correlation."

        refined = np.linalg.inv(refined_inv)
        if not np.isfinite(refined).all() or abs(refined[2, 2]) < 1e-12:
            return initial_homography, float(cc), False, "ECC refined matrix inversion was numerically singular."

        refined /= refined[2, 2]
        return refined, float(cc), True, f"ECC converged with correlation coefficient {float(cc):.4f}."
    except (cv2.error, np.linalg.LinAlgError) as err:
        return initial_homography, None, False, f"ECC optimization failed: {err}"


def refine_subpixel(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    homography: np.ndarray,
    methods: list[str] | str | None = None,
) -> dict:
    """Run only the explicitly requested refinement methods and report granular evidence.

    Accepts both UI aliases (taylor, phase, quadratic, ecc) and canonical identifiers
    (taylor_expansion, phase_correlation, quadratic_peak, ecc, lucas_kanade, corner_subpix).
    Distinguishes between global transformation refinement and local point-wise refinement.
    """
    raw_tokens: list[str] = []
    if isinstance(methods, str):
        raw_tokens = [tok.strip() for tok in methods.split(",") if tok.strip()]
    elif methods:
        raw_tokens = [str(tok).strip() for tok in methods if str(tok).strip()]

    canonical_selected = normalize_refinement_methods(raw_tokens)
    source = source_points.reshape(-1, 2).astype(np.float32)
    initial = reference_points.reshape(-1, 2).astype(np.float32)
    target = initial.copy()

    applied: list[str] = []
    method_status: list[dict] = []
    phase_response = None
    phase_used = False
    ecc_score = None
    ecc_used = False

    # Check for unrecognized tokens
    for raw in raw_tokens:
        norm = normalize_refinement_method(raw)
        if norm is None:
            method_status.append({
                "method": raw,
                "canonical_method": None,
                "requested": True,
                "attempted": False,
                "succeeded": False,
                "valid_points": 0,
                "category": "unknown",
                "reason": f"Unknown refinement method '{raw}'",
            })

    comparison_ledger: list[dict] = []
    point_details: list[dict] = []
    sel_method = None
    if "auto" in canonical_selected:
        auto_res = compare_and_select_refinement(source_gray, reference_gray, source, target, homography)
        comparison_ledger = auto_res.get("comparison_ledger", [])
        point_details = auto_res.get("point_details", [])
        sel_method = auto_res.get("selected_method")
        if sel_method and sel_method != "none":
            applied.append(f"auto({sel_method})")
            target = auto_res["refined_reference_points"]
            if auto_res.get("homography") is not None:
                homography = auto_res["homography"]
            subpixel_validation_status = auto_res["status"]
            subpixel_validation_reason = auto_res["reason"]
            method_status.append({
                "method": "auto",
                "canonical_method": "auto",
                "requested": True,
                "attempted": True,
                "succeeded": True,
                "selected_method": sel_method,
                "rmse_improvement_pixels": auto_res.get("rmse_improvement_pixels"),
                "category": "hybrid",
                "reason": auto_res.get("reason"),
            })
            if sel_method in POINT_REFINEMENT_METHODS:
                has_point_refinement = True
        else:
            method_status.append({
                "method": "auto",
                "canonical_method": "auto",
                "requested": True,
                "attempted": True,
                "succeeded": False,
                "selected_method": None,
                "category": "hybrid",
                "reason": auto_res.get("reason"),
            })

    # 1. Global image-wide methods (ECC and Phase Correlation)
    if "ecc" in canonical_selected and "auto" not in canonical_selected:
        refined_h, score, used, reason = refine_with_ecc(source_gray, reference_gray, homography, iterations=120)
        ecc_score = score
        ecc_used = used
        method_status.append({
            "method": "ecc",
            "canonical_method": "ecc",
            "requested": True,
            "attempted": True,
            "succeeded": bool(used),
            "correlation": score,
            "category": "global_image",
            "reason": reason if used else f"Not applicable for this image pair: {reason}",
        })
        if used:
            homography = refined_h
            applied.append("ecc")
            sel_method = "ecc"

    if "phase_correlation" in canonical_selected and "auto" not in canonical_selected:
        refined_h, response, used, shift, reason = phase_translation(source_gray, reference_gray, homography)
        phase_response, phase_used = response, used
        method_status.append({
            "method": "phase_correlation",
            "canonical_method": "phase_correlation",
            "requested": True,
            "attempted": True,
            "succeeded": bool(used),
            "response": response,
            "shift_xy": shift,
            "category": "global_translation",
            "reason": reason if used else f"Not applicable for this image pair: {reason}",
        })
        if used:
            homography = refined_h
            applied.append("phase_correlation")
            sel_method = "phase_correlation"

    # 2. Local point-wise correspondence refinement methods
    if "lucas_kanade" in canonical_selected and "auto" not in canonical_selected:
        if len(source) >= 4:
            new_target, valid, lk_details = lucas_kanade(source_gray, reference_gray, source, target, return_details=True)
            valid_count = int(np.sum(valid))
            if valid_count >= 4:
                mean_shift = float(np.mean(np.linalg.norm(new_target[valid] - target[valid], axis=1)))
                target = new_target
                applied.append("lucas_kanade")
                sel_method = "lucas_kanade"
                point_details = lk_details
                method_status.append({
                    "method": "lucas_kanade",
                    "canonical_method": "lucas_kanade",
                    "requested": True,
                    "attempted": True,
                    "succeeded": True,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Refined {valid_count}/{len(source)} points with mean shift {mean_shift:.3f} px.",
                })
            else:
                method_status.append({
                    "method": "lucas_kanade",
                    "canonical_method": "lucas_kanade",
                    "requested": True,
                    "attempted": True,
                    "succeeded": False,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Not applicable for this image pair: Insufficient points converged ({valid_count}/{len(source)} < 4).",
                })
        else:
            method_status.append({
                "method": "lucas_kanade",
                "canonical_method": "lucas_kanade",
                "requested": True,
                "attempted": False,
                "succeeded": False,
                "valid_points": len(source),
                "category": "point_wise",
                "reason": f"Not applicable for this image pair: Need at least 4 correspondence points to apply Lucas-Kanade (got {len(source)}).",
            })

    if "taylor_expansion" in canonical_selected and "auto" not in canonical_selected:
        if len(source) >= 4:
            new_target, valid, taylor_details = taylor_expansion(source_gray, reference_gray, source, target, return_details=True)
            valid_count = int(np.sum(valid))
            if valid_count >= 4:
                mean_shift = float(np.mean(np.linalg.norm(new_target[valid] - target[valid], axis=1)))
                target = new_target
                applied.append("taylor_expansion")
                sel_method = "taylor_expansion"
                point_details = taylor_details
                method_status.append({
                    "method": "taylor_expansion",
                    "canonical_method": "taylor_expansion",
                    "requested": True,
                    "attempted": True,
                    "succeeded": True,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Refined {valid_count}/{len(source)} points with mean shift {mean_shift:.3f} px.",
                })
            else:
                method_status.append({
                    "method": "taylor_expansion",
                    "canonical_method": "taylor_expansion",
                    "requested": True,
                    "attempted": True,
                    "succeeded": False,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Not applicable for this image pair: Insufficient points converged ({valid_count}/{len(source)} < 4).",
                })
        else:
            method_status.append({
                "method": "taylor_expansion",
                "canonical_method": "taylor_expansion",
                "requested": True,
                "attempted": False,
                "succeeded": False,
                "valid_points": len(source),
                "category": "point_wise",
                "reason": f"Not applicable for this image pair: Need at least 4 correspondence points to apply Taylor expansion (got {len(source)}).",
            })

    if "quadratic_peak" in canonical_selected and "auto" not in canonical_selected:
        if len(source) >= 4:
            new_target, valid, quad_details = quadratic_peak(source_gray, reference_gray, source, target, return_details=True)
            valid_count = int(np.sum(valid))
            if valid_count >= 4:
                mean_shift = float(np.mean(np.linalg.norm(new_target[valid] - target[valid], axis=1)))
                target = new_target
                applied.append("quadratic_peak")
                sel_method = "quadratic_peak"
                point_details = quad_details
                method_status.append({
                    "method": "quadratic_peak",
                    "canonical_method": "quadratic_peak",
                    "requested": True,
                    "attempted": True,
                    "succeeded": True,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Refined {valid_count}/{len(source)} points with mean shift {mean_shift:.3f} px.",
                })
            else:
                method_status.append({
                    "method": "quadratic_peak",
                    "canonical_method": "quadratic_peak",
                    "requested": True,
                    "attempted": True,
                    "succeeded": False,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Not applicable for this image pair: Insufficient points converged ({valid_count}/{len(source)} < 4).",
                })
        else:
            method_status.append({
                "method": "quadratic_peak",
                "canonical_method": "quadratic_peak",
                "requested": True,
                "attempted": False,
                "succeeded": False,
                "valid_points": len(source),
                "category": "point_wise",
                "reason": f"Not applicable for this image pair: Need at least 4 correspondence points to apply quadratic peak (got {len(source)}).",
            })

    if "corner_subpix" in canonical_selected and "auto" not in canonical_selected:
        if len(target) >= 4:
            new_target, valid = corner_taylor(reference_gray, target)
            valid_count = int(np.sum(valid))
            if valid_count >= 4:
                mean_shift = float(np.mean(np.linalg.norm(new_target[valid] - target[valid], axis=1)))
                target = new_target
                applied.append("corner_subpix")
                sel_method = "corner_subpix"
                method_status.append({
                    "method": "corner_subpix",
                    "canonical_method": "corner_subpix",
                    "requested": True,
                    "attempted": True,
                    "succeeded": True,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Refined {valid_count}/{len(source)} points with mean shift {mean_shift:.3f} px.",
                })
            else:
                method_status.append({
                    "method": "corner_subpix",
                    "canonical_method": "corner_subpix",
                    "requested": True,
                    "attempted": True,
                    "succeeded": False,
                    "valid_points": valid_count,
                    "category": "point_wise",
                    "reason": f"Not applicable for this image pair: Insufficient points converged ({valid_count}/{len(source)} < 4).",
                })
        else:
            method_status.append({
                "method": "corner_subpix",
                "canonical_method": "corner_subpix",
                "requested": True,
                "attempted": False,
                "succeeded": False,
                "valid_points": len(source),
                "category": "point_wise",
                "reason": f"Not applicable for this image pair: Need at least 4 correspondence points to apply corner_subpix (got {len(source)}).",
            })

    # A local point refinement is allowed to update the projective model only
    # when it produces a valid robust model. This keeps a noisy optional method
    # from silently replacing a stronger geometric estimate.
    has_point_refinement = any(item in applied for item in POINT_REFINEMENT_METHODS)
    if has_point_refinement and len(source) >= 4 and homography is not None:
        try:
            from app.services.geometry import estimate_geometric_model, validate_transform_plausibility
            is_affine_original = (
                abs(float(homography[2, 0])) < 1e-7 and
                abs(float(homography[2, 1])) < 1e-7 and
                abs(float(homography[2, 2]) - 1.0) < 1e-5
            )
            geom_sub = estimate_geometric_model(
                source,
                target,
                source_shape=source_gray.shape if source_gray is not None else None,
                reference_shape=reference_gray.shape if reference_gray is not None else None,
                ransac_threshold=3.0,
                prefer_affine=is_affine_original,
            )
            if geom_sub.is_valid and geom_sub.inlier_count >= 4:
                plaus, _, _ = validate_transform_plausibility(
                    geom_sub.homography,
                    source_shape=source_gray.shape if source_gray is not None else None,
                    reference_shape=reference_gray.shape if reference_gray is not None else None,
                )
                if plaus:
                    homography = geom_sub.homography
        except Exception:
            pass

    deltas = target - initial

    # Determine subpixel validation evidence state
    if not canonical_selected and not raw_tokens:
        subpixel_validation_status = "SUBPIXEL_NOT_VALIDATED"
        subpixel_validation_reason = "No sub-pixel refinement was requested."
    elif not applied:
        subpixel_validation_status = "SUBPIXEL_FAILED"
        subpixel_validation_reason = "Sub-pixel refinement methods were requested, but none succeeded."
    else:
        # Refinement succeeded, but pairwise real-world registration has no ground-truth reference
        subpixel_validation_status = "SUBPIXEL_NOT_VALIDATED"
        subpixel_validation_reason = (
            f"Applied {', '.join(applied)}; local/global refinement updated coordinates/model, "
            "but ungrounded pairwise registration lacks ground-truth fiducial reference for absolute validation."
        )

    # Ensure point_details fallback if none generated by point method
    if not point_details and len(source) > 0:
        point_details = [
            {
                "index": i + 1,
                "integer_coordinate": [int(round(float(initial[i, 0]))), int(round(float(initial[i, 1])))],
                "original_point": [round(float(initial[i, 0]), 4), round(float(initial[i, 1]), 4)],
                "refined_point": [round(float(target[i, 0]), 4), round(float(target[i, 1]), 4)],
                "dx": round(float(target[i, 0] - initial[i, 0]), 4),
                "dy": round(float(target[i, 1] - initial[i, 1]), 4),
                "shift": round(float(np.hypot(target[i, 0] - initial[i, 0], target[i, 1] - initial[i, 1])), 4),
                "iterations": 1,
                "converged": True if applied else False,
                "status": "CONVERGED" if applied else "UNREFINED",
                "method": sel_method or "none",
                "local_correlation": None,
                "reason": subpixel_validation_reason,
            }
            for i in range(len(source))
        ]

    return {
        "requested_methods": canonical_selected,
        "raw_requested_tokens": raw_tokens,
        "applied_methods": applied,
        "selected_method": sel_method,
        "method_status": method_status,
        "phase_response": phase_response,
        "phase_used": phase_used,
        "ecc_correlation": ecc_score,
        "ecc_used": ecc_used,
        "has_point_refinement": has_point_refinement,
        "subpixel_validation_status": subpixel_validation_status,
        "subpixel_validation_reason": subpixel_validation_reason,
        "initial_reference_points": initial,
        "refined_reference_points": target,
        "delta_xy": deltas,
        "homography": homography,
        "comparison_ledger": comparison_ledger,
        "point_details": point_details,
    }


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def compute_registered_composite(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    homography: np.ndarray,
    max_dimension: int = 4096,
) -> tuple[np.ndarray | None, dict]:
    """Compute a scientifically correct registered composite canvas.

    - Preserves reference-image scale (1:1 pixels).
    - Accurately places source image warped into reference coordinate space.
    - Handles negative transformed coordinates through rigid origin translation.
    - Avoids excessive empty margins with tight bounding box.
    - Returns composite canvas and bounding box metadata.
    """
    if homography is None or not np.isfinite(homography).all():
        return None, {"status": "FAILED", "reason": "Homography is invalid or non-finite."}

    src_h, src_w = source_image.shape[:2]
    ref_h, ref_w = reference_image.shape[:2]

    src_corners = np.array(
        [[0.0, 0.0], [float(src_w), 0.0], [float(src_w), float(src_h)], [0.0, float(src_h)]],
        dtype=np.float32,
    )
    ref_corners = np.array(
        [[0.0, 0.0], [float(ref_w), 0.0], [float(ref_w), float(ref_h)], [0.0, float(ref_h)]],
        dtype=np.float32,
    )

    try:
        proj_corners = cv2.perspectiveTransform(
            src_corners.reshape(-1, 1, 2), homography.astype(np.float64)
        ).reshape(-1, 2)
    except cv2.error as err:
        return None, {"status": "FAILED", "reason": f"Corner projection failed: {err}"}

    if not np.isfinite(proj_corners).all():
        return None, {"status": "FAILED", "reason": "Transformed corners contain NaN/inf."}

    # Combine reference corners (at origin) and projected source corners
    all_corners = np.vstack([ref_corners, proj_corners])
    min_x = float(np.min(all_corners[:, 0]))
    min_y = float(np.min(all_corners[:, 1]))
    max_x = float(np.max(all_corners[:, 0]))
    max_y = float(np.max(all_corners[:, 1]))

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    if canvas_w <= 0 or canvas_h <= 0:
        return None, {"status": "FAILED", "reason": "Non-positive canvas dimensions."}
    if canvas_w > max_dimension or canvas_h > max_dimension:
        return None, {"status": "FAILED", "reason": f"Canvas size ({canvas_w}x{canvas_h}) exceeds maximum limit {max_dimension}."}

    # Translation matrix to shift (min_x, min_y) to (0, 0)
    T = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_canvas = T @ homography.astype(np.float64)
    H_canvas /= H_canvas[2, 2]

    src_bgr = _to_bgr(source_image)
    ref_bgr = _to_bgr(reference_image)

    warped_src = cv2.warpPerspective(src_bgr, H_canvas, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    src_mask = cv2.warpPerspective(
        np.ones((src_h, src_w), dtype=np.uint8) * 255,
        H_canvas,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
    )

    ref_x = int(round(-min_x))
    ref_y = int(round(-min_y))

    composite = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    composite[ref_y:ref_y + ref_h, ref_x:ref_x + ref_w] = ref_bgr

    src_valid = src_mask > 0
    ref_valid = np.zeros((canvas_h, canvas_w), dtype=bool)
    ref_valid[ref_y:ref_y + ref_h, ref_x:ref_x + ref_w] = True

    both_valid = src_valid & ref_valid
    src_only = src_valid & ~ref_valid

    composite[src_only] = warped_src[src_only]
    composite[both_valid] = cv2.addWeighted(
        composite[both_valid], 0.5, warped_src[both_valid], 0.5, 0
    )

    meta = {
        "status": "SUCCESS",
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "offset_x": -min_x,
        "offset_y": -min_y,
        "reference_rect": [ref_x, ref_y, ref_w, ref_h],
        "projected_bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
    }
    return composite, meta


def warp_to_reference(source_image, reference_shape, homography):
    h, w = reference_shape[:2]
    return cv2.warpPerspective(source_image, homography, (w, h))


def draw_matches(
    source_gray,
    reference_gray,
    source_points,
    reference_points,
    inlier_mask=None,
    max_inlier_display=300,
    max_outlier_display=80,
    registration_status: str | None = None,
    homography: np.ndarray | None = None,
    rmse: float | None = None,
    iou: float | None = None,
    model_type: str | None = None,
    show_outliers: bool = False,
):
    """Render a canonical scientific correspondence visualization.

    - Side titles: SOURCE / MOVING IMAGE | REFERENCE / FIXED IMAGE
    - Inliers: prominent bright green lines (2-3px) with circular endpoint markers
    - Outliers: only drawn if show_outliers is explicitly enabled (default is False for clean scientific visualization)
    - Detected Overlap Region: projected source quadrilateral drawn with boundary and translucent fill
    - Common Feature: centroid crosshair on dominant correspondences
    - Text overlay badge: Inliers, Total, Ratio, Status, RMSE, IoU, Model
    - Legend: Verified inlier, Detected overlap (and Rejected outlier if show_outliers is True)
    """
    if source_gray.ndim == 2:
        src_vis = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)
    elif source_gray.shape[2] == 3:
        src_vis = source_gray.copy()
    else:
        src_vis = cv2.cvtColor(source_gray[:, :, 0], cv2.COLOR_GRAY2BGR)

    if reference_gray.ndim == 2:
        ref_vis = cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2BGR)
    elif reference_gray.shape[2] == 3:
        ref_vis = reference_gray.copy()
    else:
        ref_vis = cv2.cvtColor(reference_gray[:, :, 0], cv2.COLOR_GRAY2BGR)

    pts_s = source_points.reshape(-1, 2)
    pts_r = reference_points.reshape(-1, 2)
    n_total = len(pts_s)

    if inlier_mask is None:
        inlier_mask_arr = np.ones(n_total, dtype=bool)
    else:
        inlier_mask_arr = np.asarray(inlier_mask).ravel().astype(bool)
        if len(inlier_mask_arr) < n_total:
            pad = np.zeros(n_total - len(inlier_mask_arr), dtype=bool)
            inlier_mask_arr = np.concatenate([inlier_mask_arr, pad])

    inlier_indices = np.where(inlier_mask_arr[:n_total])[0]
    outlier_indices = np.where(~inlier_mask_arr[:n_total])[0]

    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    inlier_ratio = n_inliers / max(n_total, 1)

    sh, sw = src_vis.shape[:2]
    rh, rw = ref_vis.shape[:2]
    height = max(sh, rh)
    width = sw + rw

    # Build base canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:sh, :sw] = src_vis
    canvas[:rh, sw:sw + rw] = ref_vis

    # Adaptive line thickness based on canvas resolution (2-3 px)
    scale_factor = max(1.0, float(width) / 1000.0)
    inlier_thick = max(2, int(round(2.0 * min(scale_factor, 1.8))))
    outlier_thick = max(1, inlier_thick - 1)
    inlier_pt_rad = max(4, int(round(4.0 * min(scale_factor, 1.5))))
    outlier_pt_rad = max(2, inlier_pt_rad - 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw section headers
    cv2.rectangle(canvas, (0, 0), (sw, 24), (20, 20, 20), -1)
    cv2.putText(canvas, f"SOURCE / MOVING IMAGE ({sw}x{sh})", (10, 17), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (sw, 0), (width, 24), (25, 25, 25), -1)
    cv2.putText(canvas, f"REFERENCE / FIXED IMAGE ({rw}x{rh})", (sw + 10, 17), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Subtle separator line between images
    cv2.line(canvas, (sw, 0), (sw, height), (100, 100, 100), 2)

    # 1. Verified Correspondence Region (Section 5):
    # Enclosing convex hull derived strictly from geometrically verified inliers,
    # rendered on source image and projected onto reference image.
    has_correspondence_region = False
    if n_inliers >= 3:
        try:
            inlier_pts_s = pts_s[inlier_indices]
            inlier_pts_r = pts_r[inlier_indices]
            hull_s_idx = cv2.convexHull(inlier_pts_s.astype(np.float32), returnPoints=False)
            if hull_s_idx is not None and len(hull_s_idx) >= 3:
                src_hull_poly = inlier_pts_s[hull_s_idx.ravel()].astype(np.int32)
                # Translucent overlay on source side
                overlay_s = canvas.copy()
                cv2.fillPoly(overlay_s, [src_hull_poly], (30, 180, 80))  # Greenish translucent tint
                cv2.addWeighted(overlay_s, 0.20, canvas, 0.80, 0, canvas)
                cv2.polylines(canvas, [src_hull_poly], isClosed=True, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

                # Label on source side
                s_lbl = f"Verified correspondence region ({n_inliers} verified inliers)"
                (slw, slh), _ = cv2.getTextSize(s_lbl, font, 0.38, 1)
                sx = int(np.clip(src_hull_poly[0, 0] + 6, 8, max(8, sw - slw - 8)))
                sy = int(np.clip(src_hull_poly[0, 1] + 16, 38, max(38, sh - 8)))
                cv2.rectangle(canvas, (sx - 3, sy - slh - 3), (sx + slw + 3, sy + 3), (10, 25, 10), -1)
                cv2.putText(canvas, s_lbl, (sx, sy), font, 0.38, (100, 255, 160), 1, cv2.LINE_AA)

                # Project verified correspondence region to reference coordinate space
                ref_hull_poly = None
                if homography is not None and np.isfinite(homography).all():
                    proj_hull = cv2.perspectiveTransform(
                        src_hull_poly.astype(np.float32).reshape(-1, 1, 2),
                        homography.astype(np.float64),
                    ).reshape(-1, 2)
                    if np.isfinite(proj_hull).all() and (np.abs(proj_hull[:, 0]) < 2 * rw).all() and (np.abs(proj_hull[:, 1]) < 2 * rh).all():
                        ref_hull_poly = (proj_hull + np.array([sw, 0.0])).astype(np.int32)

                if ref_hull_poly is None:
                    # Fallback to direct convex hull of reference inliers shifted by sw
                    hull_r_idx = cv2.convexHull(inlier_pts_r.astype(np.float32), returnPoints=False)
                    if hull_r_idx is not None and len(hull_r_idx) >= 3:
                        ref_hull_poly = (inlier_pts_r[hull_r_idx.ravel()] + np.array([sw, 0.0])).astype(np.int32)

                if ref_hull_poly is not None:
                    has_correspondence_region = True
                    overlay_r = canvas.copy()
                    cv2.fillPoly(overlay_r, [ref_hull_poly], (30, 180, 80))
                    cv2.addWeighted(overlay_r, 0.20, canvas, 0.80, 0, canvas)
                    cv2.polylines(canvas, [ref_hull_poly], isClosed=True, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)
                    r_lbl = "Verified correspondence region (projected)"
                    (rlw, rlh), _ = cv2.getTextSize(r_lbl, font, 0.38, 1)
                    rx = int(np.clip(ref_hull_poly[0, 0] + 6, sw + 8, max(sw + 8, width - rlw - 8)))
                    ry = int(np.clip(ref_hull_poly[0, 1] + 16, 38, max(38, rh - 8)))
                    cv2.rectangle(canvas, (rx - 3, ry - rlh - 3), (rx + rlw + 3, ry + 3), (10, 25, 10), -1)
                    cv2.putText(canvas, r_lbl, (rx, ry), font, 0.38, (100, 255, 160), 1, cv2.LINE_AA)
        except Exception:
            pass

    # 2. Detected Overlap Region (Part 4): Project moving image footprint onto reference side
    has_valid_overlap_poly = False
    if homography is not None and np.isfinite(homography).all():
        try:
            src_corners = np.array(
                [[0.0, 0.0], [float(sw), 0.0], [float(sw), float(sh)], [0.0, float(sh)]],
                dtype=np.float32,
            )
            proj_corners = cv2.perspectiveTransform(
                src_corners.reshape(-1, 1, 2), homography.astype(np.float64)
            ).reshape(-1, 2)

            if np.isfinite(proj_corners).all():
                # Shift to reference side of canvas
                canvas_poly = (proj_corners + np.array([sw, 0.0])).astype(np.int32)
                # Verify bounds are within reasonable canvas area
                if (np.abs(canvas_poly[:, 0] - sw) < 3 * rw).all() and (np.abs(canvas_poly[:, 1]) < 3 * rh).all():
                    has_valid_overlap_poly = True
                    # Translucent overlay
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [canvas_poly], (30, 120, 240))  # Amber/orange glow
                    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
                    # Visible boundary
                    cv2.polylines(canvas, [canvas_poly], isClosed=True, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
                    # Label
                    lbl_x = int(np.clip(canvas_poly[0, 0] + 8, sw + 10, width - 180))
                    lbl_y = int(np.clip(canvas_poly[0, 1] + 18, 35, height - 10))
                    overlap_lbl = f"Detected overlap" if iou is None else f"Detected overlap (IoU: {iou:.3f})"
                    (otw, oth), _ = cv2.getTextSize(overlap_lbl, font, 0.42, 1)
                    cv2.rectangle(canvas, (lbl_x - 4, lbl_y - oth - 3), (lbl_x + otw + 4, lbl_y + 4), (0, 0, 0), -1)
                    cv2.putText(canvas, overlap_lbl, (lbl_x, lbl_y), font, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # Optional: Common feature crosshair if dominant inliers exist
    if n_inliers >= 4:
        ref_inliers = pts_r[inlier_indices]
        centroid = np.mean(ref_inliers, axis=0) + np.array([sw, 0.0])
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        if sw < cx < width and 0 < cy < height:
            cv2.circle(canvas, (cx, cy), 8, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, (cx, cy), (255, 200, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
            cv2.putText(canvas, "Common feature", (cx + 10, cy - 4), font, 0.38, (255, 220, 50), 1, cv2.LINE_AA)

    # 2. Draw outliers only if explicitly requested (hidden by default for clean scientific canonical view)
    if show_outliers:
        outlier_show = outlier_indices[:max_outlier_display]
        for idx in outlier_show:
            a = tuple(np.round(pts_s[idx]).astype(int))
            b = tuple(np.round(pts_r[idx]).astype(int) + np.array([sw, 0]))
            cv2.line(canvas, a, b, (40, 40, 220), outlier_thick, cv2.LINE_AA)
            cv2.circle(canvas, a, outlier_pt_rad, (60, 60, 240), -1)
            cv2.circle(canvas, b, outlier_pt_rad, (60, 60, 240), -1)

    # 3. Draw inliers prominently — bright green lines with distinct circular endpoints
    inlier_show = inlier_indices[:max_inlier_display]
    for idx in inlier_show:
        a = tuple(np.round(pts_s[idx]).astype(int))
        b = tuple(np.round(pts_r[idx]).astype(int) + np.array([sw, 0]))
        cv2.line(canvas, a, b, (0, 230, 80), inlier_thick, cv2.LINE_AA)
        # Endpoint with inner dot and outer ring for clarity
        cv2.circle(canvas, a, inlier_pt_rad, (0, 255, 100), -1)
        cv2.circle(canvas, a, inlier_pt_rad + 1, (0, 180, 50), 1, cv2.LINE_AA)
        cv2.circle(canvas, b, inlier_pt_rad, (0, 255, 100), -1)
        cv2.circle(canvas, b, inlier_pt_rad + 1, (0, 180, 50), 1, cv2.LINE_AA)

    # 4. Text overlay badge — comprehensive diagnostic metrics
    font_scale = 0.46
    thickness = 1
    text_color = (255, 255, 255)
    bg_color = (15, 15, 15)

    if registration_status:
        status_text = registration_status
    elif n_inliers >= 10 and inlier_ratio >= 0.10:
        status_text = "PASS"
    elif n_inliers >= 6:
        status_text = "REVIEW"
    else:
        status_text = "FAIL"

    status_colors = {
        "PASS": (0, 220, 0),
        "PASS_WITH_WARNING": (0, 210, 210),
        "REVIEW": (0, 165, 255),
        "FAIL": (40, 40, 240),
    }
    status_color = status_colors.get(status_text, (200, 200, 200))

    lines = [
        f"Status: {status_text}",
        f"Verified inliers: {n_inliers} / {n_total} ({inlier_ratio:.1%})",
    ]
    if rmse is not None and rmse > 0:
        lines.append(f"Residual RMSE: {rmse:.2f} px")
    if iou is not None and iou > 0:
        lines.append(f"Detected IoU: {iou:.3f}")
    if model_type:
        lines.append(f"Transform: {model_type.upper()}")

    y_start = height - 16 - (len(lines) - 1) * 22
    for i, line in enumerate(lines):
        y = y_start + i * 22
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(canvas, (4, y - th - 4), (14 + tw, y + 5), bg_color, -1)
        cv2.rectangle(canvas, (4, y - th - 4), (14 + tw, y + 5), (60, 60, 60), 1)
        color = status_color if "Status:" in line else text_color
        cv2.putText(canvas, line, (8, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # 5. Clean Legend in top-right corner
    legend_items = [
        (f"Verified inlier ({n_inliers})", (0, 230, 80)),
    ]
    if show_outliers:
        legend_items.append((f"Rejected outlier ({n_outliers})", (40, 40, 220)))
    if has_correspondence_region:
        legend_items.append(("Correspondence region", (0, 255, 120)))
    if has_valid_overlap_poly:
        legend_items.append(("Detected overlap", (0, 140, 255)))

    legend_w = 210
    legend_h = 10 + len(legend_items) * 22
    legend_x = width - legend_w - 8
    cv2.rectangle(canvas, (legend_x, 28), (width - 8, 28 + legend_h), (15, 15, 15), -1)
    cv2.rectangle(canvas, (legend_x, 28), (width - 8, 28 + legend_h), (60, 60, 60), 1)
    for i, (label, color) in enumerate(legend_items):
        y = 48 + i * 22
        if "overlap" in label.lower() or "region" in label.lower():
            cv2.rectangle(canvas, (legend_x + 6, y - 8), (legend_x + 18, y + 2), color, 2)
        else:
            cv2.circle(canvas, (legend_x + 12, y - 4), inlier_pt_rad, color, -1)
        cv2.putText(canvas, label, (legend_x + 24, y), font, 0.40, text_color, 1, cv2.LINE_AA)

    return canvas
