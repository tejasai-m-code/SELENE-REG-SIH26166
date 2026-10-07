"""Coarse-to-fine multi-scale registration strategy using image pyramids.

Preserves coordinate transformations across pyramid levels using exact
scale matrices: H_fine = inv(S_ref) @ H_coarse @ S_src.
"""

from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

from app.services.feature_matching import MatchResult, match_features
from app.services.geometry import estimate_geometric_model
from app.services.preprocessing import build_pyramid


@dataclass
class MultiScaleResult:
    matches: MatchResult
    homography: np.ndarray
    inlier_mask: np.ndarray
    levels_used: int
    coarse_homography: np.ndarray | None
    fine_prior_homography: np.ndarray | None
    scale_factor_source: float
    scale_factor_reference: float
    diagnostics: dict = field(default_factory=dict)


def _safe_notify(callback: Any, stage: str, stage_index: int, stage_count: int, progress: float, message: str) -> None:
    if callable(callback):
        try:
            callback(stage, stage_index, stage_count, progress, message)
        except Exception:
            pass


def match_multiscale(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    detector: str = "sift",
    ratio: float = 0.72,
    max_features: int = 8000,
    ransac_threshold: float = 3.0,
    levels: int = 2,
    progress_callback: Any = None,
) -> MultiScaleResult:
    """Execute coarse-to-fine multi-scale matching.

    1. Builds Gaussian pyramids for source and reference images.
    2. Matches features at the coarsest level to establish an initial geometric prior.
    3. Transforms coarse prior to full-resolution coordinates.
    4. Matches features at fine scale, filtering or augmenting using the coarse prior.
    """
    levels = max(1, min(int(levels), 3))
    sh, sw = source_gray.shape[:2]
    rh, rw = reference_gray.shape[:2]

    # If image dimensions are small, multi-scale downsampling creates sub-images
    # too small for reliable feature extraction; use single-scale.
    if min(sh, sw, rh, rw) < 200 or levels <= 1 or detector.lower() == "loftr":
        _safe_notify(progress_callback, "FEATURE_DETECTION", 4, 12, 0.33, f"Extracting {detector.upper()} keypoints and descriptors")
        _safe_notify(progress_callback, "FEATURE_MATCHING", 5, 12, 0.42, f"Matching candidate correspondences with ratio test ({ratio:.2f})")
        match_fine = match_features(source_gray, reference_gray, detector, ratio, max_features)
        geom = estimate_geometric_model(
            match_fine.source_points,
            match_fine.reference_points,
            source_shape=source_gray.shape,
            reference_shape=reference_gray.shape,
            ransac_threshold=ransac_threshold,
        )
        return MultiScaleResult(
            matches=match_fine,
            homography=geom.homography,
            inlier_mask=geom.inlier_mask,
            levels_used=1,
            coarse_homography=None,
            fine_prior_homography=None,
            scale_factor_source=1.0,
            scale_factor_reference=1.0,
            diagnostics={"strategy": "single_scale_direct", "levels": 1},
        )

    # 1. Build Pyramids
    _safe_notify(progress_callback, "FEATURE_DETECTION", 4, 12, 0.33, f"Building Gaussian pyramids and extracting {detector.upper()} features")
    pyr_src = build_pyramid(source_gray, levels=levels)
    pyr_ref = build_pyramid(reference_gray, levels=levels)

    coarse_src = pyr_src[-1]
    coarse_ref = pyr_ref[-1]

    c_sh, c_sw = coarse_src.shape[:2]
    c_rh, c_rw = coarse_ref.shape[:2]

    scale_s_x = float(c_sw) / float(sw)
    scale_s_y = float(c_sh) / float(sh)
    scale_r_x = float(c_rw) / float(rw)
    scale_r_y = float(c_rh) / float(rh)

    S_src = np.diag([scale_s_x, scale_s_y, 1.0])
    S_ref = np.diag([scale_r_x, scale_r_y, 1.0])
    S_ref_inv = np.linalg.inv(S_ref)

    # 2. Match at coarse scale
    H_coarse = None
    H_fine_prior = None
    coarse_matches_count = 0
    coarse_inliers_count = 0

    try:
        _safe_notify(progress_callback, "FEATURE_MATCHING", 5, 12, 0.42, f"Matching coarse pyramid correspondences (ratio={min(0.80, ratio + 0.05):.2f})")
        match_coarse = match_features(
            coarse_src,
            coarse_ref,
            detector,
            ratio=min(0.80, ratio + 0.05),  # slightly more permissive at coarse level
            max_features=max(500, max_features // 2),
        )
        coarse_matches_count = len(match_coarse.good_matches)
        geom_coarse = estimate_geometric_model(
            match_coarse.source_points,
            match_coarse.reference_points,
            source_shape=coarse_src.shape,
            reference_shape=coarse_ref.shape,
            ransac_threshold=ransac_threshold,
        )
        if geom_coarse.is_valid:
            H_coarse = geom_coarse.homography
            coarse_inliers_count = geom_coarse.inlier_count
            # Scale coarse homography to fine resolution
            H_fine_prior = S_ref_inv @ H_coarse @ S_src
            H_fine_prior /= H_fine_prior[2, 2]
    except (ValueError, np.linalg.LinAlgError):
        H_coarse = None
        H_fine_prior = None

    # 3. Match at fine scale
    _safe_notify(progress_callback, "FEATURE_MATCHING", 5, 12, 0.42, "Matching fine-resolution correspondences with prior filtering")
    match_fine = match_features(source_gray, reference_gray, detector, ratio, max_features)

    # If coarse prior is available, we can filter or augment fine correspondences
    fine_pts_s = match_fine.source_points.reshape(-1, 2)
    fine_pts_r = match_fine.reference_points.reshape(-1, 2)

    # Check consistency of fine matches with coarse prior
    if H_fine_prior is not None and len(fine_pts_s) >= 4:
        try:
            pred_from_prior = cv2.perspectiveTransform(
                fine_pts_s.reshape(-1, 1, 2).astype(np.float32), H_fine_prior
            ).reshape(-1, 2)
            prior_residuals = np.linalg.norm(pred_from_prior - fine_pts_r, axis=1)
            # If at least 4 fine points agree with coarse prior within 4x threshold,
            # use them to estimate final geometry with prior consistency
            agree_mask = prior_residuals <= (ransac_threshold * 4.0)
            if int(np.sum(agree_mask)) >= 4:
                geom_fine = estimate_geometric_model(
                    fine_pts_s[agree_mask],
                    fine_pts_r[agree_mask],
                    source_shape=source_gray.shape,
                    reference_shape=reference_gray.shape,
                    ransac_threshold=ransac_threshold,
                )
            else:
                geom_fine = estimate_geometric_model(
                    fine_pts_s,
                    fine_pts_r,
                    source_shape=source_gray.shape,
                    reference_shape=reference_gray.shape,
                    ransac_threshold=ransac_threshold,
                )
        except cv2.error:
            geom_fine = estimate_geometric_model(
                fine_pts_s,
                fine_pts_r,
                source_shape=source_gray.shape,
                reference_shape=reference_gray.shape,
                ransac_threshold=ransac_threshold,
            )
    else:
        geom_fine = estimate_geometric_model(
            fine_pts_s,
            fine_pts_r,
            source_shape=source_gray.shape,
            reference_shape=reference_gray.shape,
            ransac_threshold=ransac_threshold,
        )

    # Re-evaluate final inliers across all fine points with final homography
    final_pred = cv2.perspectiveTransform(
        fine_pts_s.reshape(-1, 1, 2).astype(np.float32), geom_fine.homography
    ).reshape(-1, 2)
    final_inlier_mask = np.linalg.norm(final_pred - fine_pts_r, axis=1) <= ransac_threshold

    diagnostics = {
        "strategy": "coarse_to_fine_pyramid",
        "levels": levels,
        "coarse_dimensions": (c_sw, c_sh),
        "fine_dimensions": (sw, sh),
        "coarse_matches": coarse_matches_count,
        "coarse_inliers": coarse_inliers_count,
        "fine_matches": len(match_fine.good_matches),
        "fine_inliers": int(np.sum(final_inlier_mask)),
        "coarse_prior_available": H_fine_prior is not None,
    }

    return MultiScaleResult(
        matches=match_fine,
        homography=geom_fine.homography,
        inlier_mask=final_inlier_mask,
        levels_used=levels,
        coarse_homography=H_coarse,
        fine_prior_homography=H_fine_prior,
        scale_factor_source=scale_s_x,
        scale_factor_reference=scale_r_x,
        diagnostics=diagnostics,
    )
