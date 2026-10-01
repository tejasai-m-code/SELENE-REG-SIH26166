import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from app.services.pairwise_registration import PairwiseRegistrationResult, register_pair, _match_visualization
from app.services.registration import refine_with_ecc, warp_to_reference
from app.services.metrics import compute_metrics
from app.utils.image_utils import ScientificRaster

@dataclass
class ScaleDiagnostics:
    source_gsd: Optional[float]
    reference_gsd: Optional[float]
    gsd_source_evidence: str
    expected_scale_ratio: Optional[float]
    estimated_image_scale_ratio: Optional[float]
    scale_dispersion: Optional[float]
    supporting_matches: int
    scale_estimation_status: str
    fallback_status: str

@dataclass
class ScaleProvenance:
    gsd_source: str
    source_gsd_value: Optional[float]
    reference_gsd_value: Optional[float]
    gsd_units: str
    scale_evidence_type: str
    scale_ratio: Optional[float]
    scale_estimation_method: str
    scale_uncertainty: Optional[float]
    supporting_correspondence_count: int
    pyramid_levels: int
    pyramid_scale_factor: float
    coarse_to_fine_enabled: bool
    initialization_path: str
    fallback_path: str
    warnings: List[str] = field(default_factory=list)

def propagate_homography(H: np.ndarray, scale_ratio: float) -> np.ndarray:
    """
    Propagates a homography computed at scale S1 to scale S2.
    scale_ratio = S2 / S1 (e.g., 2.0 if moving to a 2x larger image)
    """
    if H is None or H.shape != (3, 3):
        return H
        
    S = np.array([
        [scale_ratio, 0, 0],
        [0, scale_ratio, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    S_inv = np.array([
        [1.0 / scale_ratio, 0, 0],
        [0, 1.0 / scale_ratio, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    H_fine = S @ H.astype(np.float64) @ S_inv
    H_fine /= H_fine[2, 2]
    return H_fine.astype(np.float32)

def build_image_pyramid(image: np.ndarray, levels: int, scale_factor: float = 0.5, min_size: int = 128) -> List[np.ndarray]:
    pyramid = [image]
    current = image
    for _ in range(1, levels):
        h, w = current.shape[:2]
        new_w, new_h = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
        if new_w < min_size or new_h < min_size:
            break
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    return pyramid

def estimate_relative_scale(pts_src: np.ndarray, pts_ref: np.ndarray) -> Tuple[Optional[float], int, Optional[float]]:
    """
    Robustly estimates relative scale (ref / src) using similarity transform.
    Returns (scale, inlier_count, scale_dispersion).
    Dispersion is calculated as the standard deviation of local scale ratios for inliers.
    """
    if pts_src is None or pts_ref is None or len(pts_src) < 4:
        return None, 0, None
        
    M, inliers = cv2.estimateAffinePartial2D(pts_src, pts_ref, method=cv2.RANSAC)
    if M is not None and inliers is not None:
        s = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        inlier_mask = inliers.ravel().astype(bool)
        count = int(np.sum(inlier_mask))
        
        if count >= 2:
            # calculate scale dispersion: median absolute deviation of pairwise distances among inliers
            in_src = pts_src[inlier_mask].reshape(-1, 2)
            in_ref = pts_ref[inlier_mask].reshape(-1, 2)
            
            # center
            c_src = np.mean(in_src, axis=0)
            c_ref = np.mean(in_ref, axis=0)
            d_src = np.linalg.norm(in_src - c_src, axis=1)
            d_ref = np.linalg.norm(in_ref - c_ref, axis=1)
            
            valid = d_src > 1e-3
            if np.sum(valid) > 0:
                local_scales = d_ref[valid] / d_src[valid]
                dispersion = float(np.std(local_scales))
            else:
                dispersion = None
        else:
            dispersion = None
            
        return float(s), count, dispersion
    return None, 0, None

def compute_expected_scale_ratio(gsd_src: Optional[float], gsd_ref: Optional[float]) -> Optional[float]:
    """
    Convention: scale_ratio = GSD_src / GSD_ref.
    If src GSD is 2.0 (coarse) and ref GSD is 1.0 (fine), 
    src pixels are 2x larger physical coverage, so src image needs 2x scaling to match ref visually.
    """
    if gsd_src is not None and gsd_ref is not None and gsd_ref > 0 and gsd_src > 0:
        if np.isfinite(gsd_src) and np.isfinite(gsd_ref):
            return float(gsd_src / gsd_ref)
    return None

def coarse_to_fine_register_pair(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
    gsd_unit: str = "UNKNOWN",
    levels: int = 3,
    scale_factor: float = 0.5,
    source_gray: Optional[np.ndarray] = None,
    reference_gray: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[PairwiseRegistrationResult, ScaleDiagnostics, ScaleProvenance]:
    warnings = []
    
    # 1. Expected ratio from metadata
    expected_ratio = compute_expected_scale_ratio(source_gsd, reference_gsd)
    
    # 2. Build Pyramids
    src_pyramid = build_image_pyramid(source_image, levels, scale_factor)
    ref_pyramid = build_image_pyramid(reference_image, levels, scale_factor)
    
    src_gray_base = source_gray if source_gray is not None else source_image
    ref_gray_base = reference_gray if reference_gray is not None else reference_image
    
    src_gray_pyramid = build_image_pyramid(src_gray_base, levels, scale_factor)
    ref_gray_pyramid = build_image_pyramid(ref_gray_base, levels, scale_factor)
    
    actual_levels = min(len(src_pyramid), len(ref_pyramid))
    L = actual_levels - 1
    
    # 3. Determine Initial Scale (S_init)
    S_init = 1.0
    init_path = "v1_fallback"
    evidence_type = "UNKNOWN"
    est_scale, support_count, est_dispersion = None, 0, None
    
    if expected_ratio is not None:
        S_init = expected_ratio
        init_path = "metadata_gsd"
        evidence_type = "KNOWN_METADATA"
    else:
        # Pre-flight feature matching to get image-derived scale
        try:
            preflight = register_pair(
                src_pyramid[L], ref_pyramid[L],
                source_gray=src_gray_pyramid[L],
                reference_gray=ref_gray_pyramid[L],
                include_registered=False, **kwargs
            )
            est_scale, support_count, est_dispersion = estimate_relative_scale(preflight.source_points, preflight.reference_points)
        except Exception as e:
            warnings.append(f"Image-derived scale estimation preflight failed: {e}")
            est_scale, support_count, est_dispersion = None, 0, None
        if est_scale is not None and support_count >= 10:
            S_init = est_scale
            init_path = "image_derived"
            evidence_type = "ESTIMATED_FROM_IMAGES"
    
    # 4. Apply Initial Scale to Coarsest Source
    # We explicitly warp the source image at the coarsest level by S_init to provide a physically scaled initialization
    if abs(S_init - 1.0) > 1e-3:
        h, w = src_pyramid[L].shape[:2]
        new_w, new_h = max(1, int(w * S_init)), max(1, int(h * S_init))
        src_coarse_scaled = cv2.resize(src_pyramid[L], (new_w, new_h), interpolation=cv2.INTER_AREA)
        src_gray_coarse_scaled = cv2.resize(src_gray_pyramid[L], (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        src_coarse_scaled = src_pyramid[L]
        src_gray_coarse_scaled = src_gray_pyramid[L]
        
    H_init_scale = np.array([
        [S_init, 0, 0],
        [0, S_init, 0],
        [0, 0, 1]
    ], dtype=np.float64)
        
    # 5. Base registration on scaled coarsest level
    coarse_kwargs = kwargs.copy()
    coarse_kwargs['include_registered'] = False
    try:
        coarsest_res = register_pair(
            src_coarse_scaled, ref_pyramid[L],
            source_gray=src_gray_coarse_scaled,
            reference_gray=ref_gray_pyramid[L],
            **coarse_kwargs
        )
        H_match = coarsest_res.homography
    except Exception as e:
        warnings.append(f"Coarse matching exception: {e}")
        H_match = None
    
    if H_match is None or H_match.shape != (3, 3):
        warnings.append("Coarse registration failed on scaled image. Falling back to unscaled V1.")
        # Total fallback
        fallback_res = register_pair(
            source_image, reference_image,
            source_gray=source_gray,
            reference_gray=reference_gray,
            **kwargs
        )
        
        diag = ScaleDiagnostics(
            source_gsd=source_gsd, reference_gsd=reference_gsd, gsd_source_evidence=evidence_type,
            expected_scale_ratio=expected_ratio, estimated_image_scale_ratio=est_scale,
            scale_dispersion=est_dispersion, supporting_matches=support_count, 
            scale_estimation_status="ESTIMATED_FROM_IMAGES" if support_count > 0 else "FAILED",
            fallback_status="V1_FALLBACK_EXECUTED"
        )
        prov = ScaleProvenance(
            gsd_source="ProductMetadata", source_gsd_value=source_gsd, reference_gsd_value=reference_gsd, gsd_units=gsd_unit,
            scale_evidence_type="UNKNOWN", scale_ratio=1.0, scale_estimation_method="NONE",
            scale_uncertainty=None, supporting_correspondence_count=0, pyramid_levels=actual_levels, pyramid_scale_factor=scale_factor,
            coarse_to_fine_enabled=False, initialization_path="none", fallback_path="v1_full_res", warnings=warnings
        )
        return fallback_res, diag, prov

    # Mathematical combination: H_coarse maps ORIGINAL coarse source to coarse ref.
    # p_ref = H_match * p_scaled = H_match * (H_init_scale * p_src) = (H_match * H_init_scale) * p_src
    H = (H_match.astype(np.float64) @ H_init_scale).astype(np.float32)
    H /= H[2, 2]

    # 6. Propagate and Refine
    up_scale = 1.0 / scale_factor
    ecc_used = coarsest_res.ecc_used
    ecc_score = coarsest_res.ecc_correlation
    
    for l in range(L - 1, -1, -1):
        H_propagated = propagate_homography(H, up_scale)
        try:
            H_refined, score, used = refine_with_ecc(src_gray_pyramid[l], ref_gray_pyramid[l], H_propagated, iterations=50)
            if used:
                H = H_refined
                ecc_score = score
                ecc_used = True
            else:
                H = H_propagated
        except Exception as e:
            warnings.append(f"ECC refinement failed at level {l}: {e}")
            H = H_propagated

    # 7. Map coarse points to full resolution
    total_scale = (1.0 / scale_factor) ** L
    
    # Remember, coarsest_res points are in the SCALED source coordinate space.
    # We must unscale them to the original coarse source space, then scale to full res.
    # Unscale: divide by S_init
    src_pts_orig_coarse = coarsest_res.source_points / S_init
    src_pts_full = src_pts_orig_coarse * total_scale
    ref_pts_full = coarsest_res.reference_points * total_scale
    
    predicted = cv2.perspectiveTransform(src_pts_full, H).reshape(-1, 2)
    target = ref_pts_full.reshape(-1, 2)
    ransac_thresh = kwargs.get('ransac_threshold', 3.0)
    inlier_mask = np.linalg.norm(predicted - target, axis=1) <= ransac_thresh
    
    # 8. Generate final outputs
    include_registered = kwargs.get('include_registered', True)
    registered = warp_to_reference(source_image, reference_image.shape, H) if include_registered else None
    
    metrics = compute_metrics(
        src_pts_full, ref_pts_full, H, inlier_mask,
        source_image.shape, reference_image.shape,
        registered=registered, reference_image=reference_image
    )
    
    match_vis = _match_visualization(
        source_image, reference_image, src_pts_full, ref_pts_full,
        inlier_mask, kwargs.get('match_preview_max_side', None)
    )
    
    # 7. Phase 7 Subpixel Refinement Consensus
    subpixel_consensus = None
    if kwargs.get('ecc_refinement', True):
        from app.services.subpixel import run_subpixel_refinement
        h, w = reference_image.shape[:2]
        src_aligned = cv2.warpPerspective(source_image, H, (w, h), flags=cv2.INTER_LINEAR)
        sub_res = run_subpixel_refinement(src_aligned, reference_image)
        
        dx, dy = sub_res.dx, sub_res.dy
        H_shift = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        H_candidate = H_shift @ H
        
        mask = (src_aligned > 0) & (reference_image > 0)
        rejection_reason = None
        accepted = False
        
        if sub_res.confidence in ["INSUFFICIENT_TEXTURE"]:
            rejection_reason = "INSUFFICIENT_TEXTURE"
        elif sub_res.confidence in ["FAILED"]:
            rejection_reason = "FAILED"
        elif sub_res.confidence in ["UNSTABLE"]:
            rejection_reason = "UNSTABLE"
        elif sub_res.confidence in ["LOW"]:
            rejection_reason = "LOW_CONFIDENCE_SUBPIXEL"
        elif np.sum(mask) <= 100:
            rejection_reason = "INSUFFICIENT_OVERLAP"
        else:
            corr_before = np.corrcoef(src_aligned[mask].astype(np.float32), reference_image[mask].astype(np.float32))[0, 1]
            src_sub = cv2.warpPerspective(source_image, H_candidate, (w, h), flags=cv2.INTER_LINEAR)
            mask2 = (src_sub > 0) & (reference_image > 0)
            if np.sum(mask2) > 100:
                corr_after = np.corrcoef(src_sub[mask2].astype(np.float32), reference_image[mask2].astype(np.float32))[0, 1]
                if corr_after >= corr_before:
                    accepted = True
                else:
                    rejection_reason = "Degraded NCC correlation"
            else:
                rejection_reason = "Insufficient overlap after refinement"
                
        if accepted:
            H = H_candidate
            
        subpixel_consensus = {
            "dx": float(sub_res.dx),
            "dy": float(sub_res.dy),
            "confidence": sub_res.confidence,
            "active_methods": sub_res.active_methods,
            "diagnostics": [
                {
                    "method": d.method,
                    "status": d.status,
                    "dx": float(d.dx),
                    "dy": float(d.dy),
                    "residual": float(d.residual) if d.residual is not None else None,
                    "iterations": d.iterations,
                    "correlation": float(d.correlation) if d.correlation is not None else None,
                    "response": float(d.response) if d.response is not None else None,
                    "stability": d.stability,
                    "confidence": d.confidence,
                    "failure_reason": d.failure_reason
                } for d in sub_res.diagnostics
            ],
            "failure_reason": sub_res.failure_reason,
            "accepted": accepted,
            "rejection_reason": None if accepted else rejection_reason,
            "acceptance_reason": "Scientific criterion met" if accepted else None
        }

    from app.services.registration import draw_feature_response_map, draw_keypoint_map
    feature_response_source = None
    feature_response_reference = None
    keypoints_source_map = None
    keypoints_reference_map = None

    try:
        if coarsest_res.source_features is not None:
            # Map coarse points back to original image coordinates
            # source_features.points are in src_coarse_scaled space.
            src_scale_ratio = total_scale / S_init
            src_points_full = coarsest_res.source_features.points * src_scale_ratio
            feature_response_source = draw_feature_response_map(source_image.shape, src_points_full, coarsest_res.source_features.responses, coarsest_res.detector)
            keypoints_source_map = draw_keypoint_map(source_image, src_points_full)
        elif coarsest_res.detector.lower() == "loftr":
            feature_response_source = draw_feature_response_map(source_image.shape, src_pts_full, coarsest_res.ratios, coarsest_res.detector)
            keypoints_source_map = draw_keypoint_map(source_image, src_pts_full)

        if coarsest_res.reference_features is not None:
            ref_scale_ratio = total_scale
            ref_points_full = coarsest_res.reference_features.points * ref_scale_ratio
            feature_response_reference = draw_feature_response_map(reference_image.shape, ref_points_full, coarsest_res.reference_features.responses, coarsest_res.detector)
            keypoints_reference_map = draw_keypoint_map(reference_image, ref_points_full)
        elif coarsest_res.detector.lower() == "loftr":
            feature_response_reference = draw_feature_response_map(reference_image.shape, ref_pts_full, coarsest_res.ratios, coarsest_res.detector)
            keypoints_reference_map = draw_keypoint_map(reference_image, ref_pts_full)
    except Exception as e:
        warnings.append(f"Failed to generate full-resolution feature maps: {e}")

    final_res = PairwiseRegistrationResult(
        homography=H,
        inlier_mask=inlier_mask,
        metrics=metrics,
        subpixel_consensus=subpixel_consensus,
        registered=registered,
        match_visualization=match_vis,
        source_points=src_pts_full,
        reference_points=ref_pts_full,
        raw_match_count=coarsest_res.raw_match_count,
        post_distribution_match_count=coarsest_res.post_distribution_match_count,
        ecc_used=ecc_used,
        ecc_correlation=ecc_score,
        detector=coarsest_res.detector,
        processing_time_seconds=0.0,
        feature_response_source=feature_response_source,
        feature_response_reference=feature_response_reference,
        keypoints_source_map=keypoints_source_map,
        keypoints_reference_map=keypoints_reference_map,
        source_features=coarsest_res.source_features,
        reference_features=coarsest_res.reference_features,
    )
    
    diag = ScaleDiagnostics(
        source_gsd=source_gsd, reference_gsd=reference_gsd, gsd_source_evidence=evidence_type,
        expected_scale_ratio=expected_ratio, estimated_image_scale_ratio=est_scale,
        scale_dispersion=est_dispersion, supporting_matches=support_count, 
        scale_estimation_status="ESTIMATED_FROM_IMAGES" if support_count > 0 else "UNKNOWN",
        fallback_status="NONE"
    )
    
    prov = ScaleProvenance(
        gsd_source="ProductMetadata" if expected_ratio else "UNKNOWN",
        source_gsd_value=source_gsd, reference_gsd_value=reference_gsd, gsd_units=gsd_unit,
        scale_evidence_type=evidence_type, scale_ratio=S_init,
        scale_estimation_method="RANSAC_SIMILARITY", scale_uncertainty=est_dispersion, 
        supporting_correspondence_count=support_count,
        pyramid_levels=actual_levels, pyramid_scale_factor=scale_factor,
        coarse_to_fine_enabled=True, initialization_path=init_path, fallback_path="none", warnings=warnings
    )
    
    return final_res, diag, prov


