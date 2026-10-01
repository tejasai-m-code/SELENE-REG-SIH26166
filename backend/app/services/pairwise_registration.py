"""Reusable, scientifically conservative two-image registration workflow."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.services.feature_matching import FeatureResult, match_features, match_feature_artifacts, spatially_distribute_matches
from app.services.metrics import compute_metrics
from app.services.preprocessing import preprocess
from app.services.registration import draw_matches, estimate_homography, refine_with_ecc, warp_to_reference


@dataclass
class PairwiseRegistrationResult:
    homography: np.ndarray
    inlier_mask: np.ndarray
    metrics: dict
    registered: np.ndarray | None
    match_visualization: np.ndarray
    source_points: np.ndarray
    reference_points: np.ndarray
    raw_match_count: int
    post_distribution_match_count: int
    ecc_used: bool
    ecc_correlation: float | None
    detector: str
    subpixel_consensus: dict | None = None
    geometric_model: str = "homography"
    estimator: str = "cv2.RANSAC"
    geometric_candidates: list = None
    processing_time_seconds: float = 0.0
    raw_matches: list | None = None
    ratios: list | None = None
    source_keypoints: list | None = None
    reference_keypoints: list | None = None
    source_features: Any | None = None
    reference_features: Any | None = None
    feature_response_source: np.ndarray | None = None
    feature_response_reference: np.ndarray | None = None
    keypoints_source_map: np.ndarray | None = None
    keypoints_reference_map: np.ndarray | None = None


def register_pair(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    detector: str = "sift",
    ratio: float = 0.72,
    ransac_threshold: float = 3.0,
    illumination_normalization: bool = True,
    spatial_distribution: bool = True,
    ecc_refinement: bool = True,
    max_features: int = 8000,
    geometric_model: str = "auto",
    match_preview_max_side: int | None = None,
    include_registered: bool = True,
    source_gray: np.ndarray | None = None,
    reference_gray: np.ndarray | None = None,
    source_features: FeatureResult | None = None,
    reference_features: FeatureResult | None = None,
) -> PairwiseRegistrationResult:
    """Register ``source_image`` into the coordinate system of ``reference_image``.

    This is shared by the established pair endpoint and the multi-image builder so
    both modes use identical preprocessing, matching, and quality measurements.
    """
    ratio = max(0.55, min(float(ratio), 0.95))
    ransac_threshold = max(0.5, min(float(ransac_threshold), 20.0))
    max_features = max(500, min(int(max_features), 20000))

    from time import perf_counter
    started = perf_counter()
    src_gray = source_gray if source_gray is not None else preprocess(source_image, illumination_normalization)
    ref_gray = reference_gray if reference_gray is not None else preprocess(reference_image, illumination_normalization)

    from app.services.feature_matching import compute_features, _match_loftr
    if source_features is None and detector.lower() != "loftr":
        source_features = compute_features(src_gray, detector, max_features)
    if reference_features is None and detector.lower() != "loftr":
        reference_features = compute_features(ref_gray, detector, max_features)

    matches = match_feature_artifacts(source_features, reference_features, detector, ratio) if detector.lower() != "loftr" else _match_loftr(src_gray, ref_gray)
    raw_match_count = len(matches.good_matches)
    if spatial_distribution:
        matches = spatially_distribute_matches(matches, src_gray.shape)

    from app.services.geometry import estimate_and_select_model
    cands, best = estimate_and_select_model(matches.source_points, matches.reference_points, ransac_threshold, src_gray.shape, allowed_models=[geometric_model] if geometric_model != "auto" else None)
    if best is None:
        raise ValueError("Geometric model estimation failed or found insufficient inliers.")
    H, inlier_mask = best.homography, best.inlier_mask
    ecc_score = None
    ecc_used = False
    subpixel_data = None
    if ecc_refinement:
        # Instead of just ECC, run full Subpixel Consensus
        from app.services.subpixel import run_subpixel_refinement
        # First align source to reference using coarse H
        aligned_src = cv2.warpPerspective(src_gray, H, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR)
        sub_res = run_subpixel_refinement(aligned_src, ref_gray)
        
        subpixel_data = {
            "dx": sub_res.dx,
            "dy": sub_res.dy,
            "confidence": sub_res.confidence,
            "active_methods": sub_res.active_methods,
            "diagnostics": [
                {
                    "method": d.method, "status": d.status, "dx": d.dx, "dy": d.dy,
                    "residual": d.residual, "iterations": d.iterations, 
                    "correlation": d.correlation, "response": d.response,
                    "stability": d.stability, "confidence": d.confidence, "failure_reason": d.failure_reason
                } for d in sub_res.diagnostics
            ],
            "failure_reason": sub_res.failure_reason
        }
        
        accepted = False
        rejection_reason = None
        
        if sub_res.confidence in ["HIGH", "MODERATE"]:
            # Evaluate Phase 6 correlation
            mask_valid = (aligned_src > 0) & (ref_gray > 0)
            if np.sum(mask_valid) > 100:
                p6_res = float(np.corrcoef(ref_gray[mask_valid].flatten(), aligned_src[mask_valid].flatten())[0, 1])
                
                # Combine shifts
                H_shift = np.array([[1.0, 0.0, sub_res.dx], [0.0, 1.0, sub_res.dy], [0.0, 0.0, 1.0]], dtype=np.float64)
                H_cand = H_shift @ H
                
                # Evaluate Phase 7 correlation
                aligned_src_p7 = cv2.warpPerspective(src_gray, H_cand, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR)
                mask_valid_p7 = (aligned_src_p7 > 0) & (ref_gray > 0)
                p7_res = float(np.corrcoef(ref_gray[mask_valid_p7].flatten(), aligned_src_p7[mask_valid_p7].flatten())[0, 1])
                
                # Accept if correlation improves or stays practically identical (loss < 0.01)
                if p7_res >= p6_res - 0.01:
                    accepted = True
                    H = H_cand
                    ecc_used = True
                    
                    # Recalculate mask
                    predicted = cv2.perspectiveTransform(matches.source_points, H).reshape(-1, 2)
                    target = matches.reference_points.reshape(-1, 2)
                    inlier_mask = np.linalg.norm(predicted - target, axis=1) <= ransac_threshold
                else:
                    rejection_reason = f"Refinement worsened correlation: P6={p6_res:.4f}, P7={p7_res:.4f}"
            else:
                rejection_reason = "Insufficient overlap for acceptance test"
        else:
            rejection_reason = "Low consensus confidence"
            
        subpixel_data["accepted"] = accepted
        subpixel_data["rejection_reason"] = rejection_reason


    registered = warp_to_reference(source_image, reference_image.shape, H) if include_registered else None
    metrics = compute_metrics(
        matches.source_points, matches.reference_points, H, inlier_mask,
        src_gray.shape, ref_gray.shape,
        registered=registered, reference_image=reference_image,
    )
    from app.services.registration import draw_feature_response_map, draw_keypoint_map
    feature_response_source = None
    feature_response_reference = None
    keypoints_source_map = None
    keypoints_reference_map = None

    if source_features is not None:
        feature_response_source = draw_feature_response_map(src_gray.shape, source_features.points, source_features.responses, detector)
        keypoints_source_map = draw_keypoint_map(source_image, source_features.points)
    elif detector.lower() == "loftr":
        feature_response_source = draw_feature_response_map(src_gray.shape, matches.source_points, matches.ratios, detector)
        keypoints_source_map = draw_keypoint_map(source_image, matches.source_points)

    if reference_features is not None:
        feature_response_reference = draw_feature_response_map(ref_gray.shape, reference_features.points, reference_features.responses, detector)
        keypoints_reference_map = draw_keypoint_map(reference_image, reference_features.points)
    elif detector.lower() == "loftr":
        feature_response_reference = draw_feature_response_map(ref_gray.shape, matches.reference_points, matches.ratios, detector)
        keypoints_reference_map = draw_keypoint_map(reference_image, matches.reference_points)

    return PairwiseRegistrationResult(
        homography=H,
        inlier_mask=inlier_mask,
        metrics=metrics,
        registered=registered,
        match_visualization=_match_visualization(
            source_image, reference_image, matches.source_points, matches.reference_points,
            inlier_mask, match_preview_max_side,
        ),
        source_points=matches.source_points,
        reference_points=matches.reference_points,
        raw_match_count=raw_match_count,
        post_distribution_match_count=len(matches.good_matches),
        ecc_used=ecc_used,
        ecc_correlation=ecc_score,
        subpixel_consensus=subpixel_data,
        detector=detector.upper(),
        geometric_model=best.model_name,
        estimator=best.estimator,
        geometric_candidates=cands,
        processing_time_seconds=round(perf_counter() - started, 6),
        raw_matches=matches.good_matches if hasattr(matches, 'good_matches') else None,
        ratios=matches.ratios if hasattr(matches, 'ratios') else None,
        source_keypoints=matches.source_keypoints if hasattr(matches, 'source_keypoints') else None,
        reference_keypoints=matches.reference_keypoints if hasattr(matches, 'reference_keypoints') else None,
        source_features=source_features,
        reference_features=reference_features,
        feature_response_source=feature_response_source,
        feature_response_reference=feature_response_reference,
        keypoints_source_map=keypoints_source_map,
        keypoints_reference_map=keypoints_reference_map,
    )


def _match_visualization(source_gray, reference_gray, source_points, reference_points, inlier_mask, max_side):
    """Render a bounded preview for multi-image jobs without altering registration."""
    if not max_side or max(source_gray.shape + reference_gray.shape) <= max_side:
        return draw_matches(source_gray, reference_gray, source_points, reference_points, inlier_mask)

    def scale_image(image):
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale == 1.0:
            return image, scale
        return cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA), scale

    source_preview, source_scale = scale_image(source_gray)
    reference_preview, reference_scale = scale_image(reference_gray)
    return draw_matches(source_preview, reference_preview, source_points * source_scale, reference_points * reference_scale, inlier_mask)


def serialize_inlier_points(result: PairwiseRegistrationResult) -> list[dict]:
    source = result.source_points.reshape(-1, 2)
    reference = result.reference_points.reshape(-1, 2)
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2).astype(np.float32), result.homography).reshape(-1, 2)
    return [
        {
            "source": [round(float(source[i, 0]), 4), round(float(source[i, 1]), 4)],
            "reference": [round(float(reference[i, 0]), 4), round(float(reference[i, 1]), 4)],
            "dx": round(float(reference[i, 0] - projected[i, 0]), 4),
            "dy": round(float(reference[i, 1] - projected[i, 1]), 4),
            "error": round(float(np.linalg.norm(reference[i] - projected[i])), 4),
            "status": "inlier",
        }
        for i, accepted in enumerate(result.inlier_mask) if accepted
    ]


