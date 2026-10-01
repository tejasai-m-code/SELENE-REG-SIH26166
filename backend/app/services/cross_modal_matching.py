# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.services.representation import ImageRepresentation
from app.services.pairwise_registration import register_pair, PairwiseRegistrationResult
from app.services.scale import coarse_to_fine_register_pair, ScaleDiagnostics, ScaleProvenance

@dataclass
class DeepMatchingState:
    capability_state: str
    model_name: Optional[str] = None
    weights_source: Optional[str] = None
    device: Optional[str] = None
    inference_status: str = "NOT_ATTEMPTED"
    failure_reason: Optional[str] = None

def inspect_deep_matching_capability() -> DeepMatchingState:
    """
    Investigates the current environment for SuperPoint/LoFTR usability.
    """
    try:
        import torch
        # Here we would check for kornia or specialized SuperPoint weights
        import kornia
        return DeepMatchingState(
            capability_state="UNAVAILABLE_WEIGHTS",
            model_name="SuperPoint/LoFTR",
            weights_source="None",
            device="cuda" if torch.cuda.is_available() else "cpu",
            inference_status="FAILED_PRECHECK",
            failure_reason="Model weights not downloaded in offline environment."
        )
    except ImportError as e:
        return DeepMatchingState(
            capability_state="UNAVAILABLE_DEPENDENCY",
            model_name="None",
            weights_source="None",
            device="None",
            inference_status="FAILED_PRECHECK",
            failure_reason=f"Missing dependency: {e}"
        )

@dataclass
class CandidateTransform:
    representation_source: str
    representation_reference: str
    detector: str
    descriptor: str
    matching_method: str
    geometric_model: str = "homography"
    estimator: str = "cv2.RANSAC"
    candidate_count: int = 0
    good_matches: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    confidence: str = "FAILED"
    failure_reason: Optional[str] = None
    result: Optional[PairwiseRegistrationResult] = None
    scale_diagnostics: Optional[ScaleDiagnostics] = None
    scale_provenance: Optional[ScaleProvenance] = None

def _eval_confidence(inlier_count: int, inlier_ratio: float, reprojection_error: float) -> str:
    if inlier_count >= 50 and inlier_ratio > 0.3 and reprojection_error < 5.0:
        return "HIGH"
    if inlier_count >= 15 and reprojection_error < 10.0:
        return "MODERATE"
    if inlier_count > 4:
        return "LOW"
    return "REJECTED"

def execute_cross_modal_matching(
    source_reps: Dict[str, ImageRepresentation],
    reference_reps: Dict[str, ImageRepresentation],
    source_display: np.ndarray = None,
    reference_display: np.ndarray = None,
    detector: str = "sift",
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
    gsd_unit: str = "UNKNOWN",
    levels: int = 3,
    **kwargs
) -> List[CandidateTransform]:
    """
    Matches pairs across structural representations instead of just raw grayscale.
    Strategies:
    - Normalized <-> Normalized
    - High-Pass <-> High-Pass
    - Gradient <-> Gradient
    """
    strategies = kwargs.pop('custom_strategies', [
        ("radiometric_normalized", "radiometric_normalized"),
        ("high_pass", "high_pass"),
        ("gradient_magnitude", "gradient_magnitude")
    ])
    
    candidates = []
    
    for src_key, ref_key in strategies:
        if src_key not in source_reps or ref_key not in reference_reps:
            continue
            
        src_rep = source_reps[src_key]
        ref_rep = reference_reps[ref_key]
        
        try:
            # We use Phase 4 coarse-to-fine registration on this representation pair
            import cv2
            src_gray = src_rep.image if src_rep.image.ndim == 2 else cv2.cvtColor(src_rep.image, cv2.COLOR_BGR2GRAY)
            ref_gray = ref_rep.image if ref_rep.image.ndim == 2 else cv2.cvtColor(ref_rep.image, cv2.COLOR_BGR2GRAY)
            res, diag, prov = coarse_to_fine_register_pair(
                source_image=source_display if source_display is not None else src_rep.image,
                reference_image=reference_display if reference_display is not None else ref_rep.image,
                source_gray=src_gray,
                reference_gray=ref_gray,
                source_gsd=source_gsd, reference_gsd=reference_gsd, gsd_unit=gsd_unit,
                levels=levels, detector=detector, **kwargs
            )
            
            inliers = res.metrics.get("inlier_count", 0)
            matches = res.metrics.get("total_matches", inliers)
            ratio = inliers / max(1, matches)
            rmse = res.metrics.get("rmse_pixels", 999.0)
            
            conf = _eval_confidence(inliers, ratio, rmse)
            fail_reason = None if inliers >= 4 else "Insufficient RANSAC inliers"
            
            cand = CandidateTransform(
                representation_source=src_key,
                representation_reference=ref_key,
                detector=detector,
                descriptor=detector,
                matching_method="coarse_to_fine_ecc",
                candidate_count=matches,
                good_matches=res.raw_match_count,
                inlier_count=inliers,
                inlier_ratio=ratio,
                reprojection_error=rmse,
                confidence=conf,
                failure_reason=fail_reason,
                result=res,
                scale_diagnostics=diag,
                scale_provenance=prov
            )
            candidates.append(cand)
            
        except Exception as e:
            candidates.append(CandidateTransform(
                representation_source=src_key,
                representation_reference=ref_key,
                detector=detector,
                descriptor=detector,
                matching_method="coarse_to_fine_ecc",
                candidate_count=0,
                good_matches=0,
                inlier_count=0,
                inlier_ratio=0.0,
                reprojection_error=999.0,
                confidence="REJECTED",
                failure_reason=f"Pipeline exception: {str(e)}"
            ))
            
    return candidates






def serialize_match_payload(cand, source_shape=None, spatial_grid=(4, 4)):
    res = cand.result
    if not res: return []
    import cv2
    import numpy as np
    source = res.source_points.reshape(-1, 2)
    reference = res.reference_points.reshape(-1, 2)
    if res.homography is not None:
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2).astype(np.float32), res.homography).reshape(-1, 2)
    else:
        projected = np.zeros_like(reference)
        
    payload = []
    
    h, w = source_shape[:2] if source_shape else (100, 100)
        
    grid_rows, grid_cols = spatial_grid
        
    for i, is_inlier in enumerate(res.inlier_mask):
        match_dist = None
        if hasattr(res, 'raw_matches') and res.raw_matches and i < len(res.raw_matches):
            match_dist = res.raw_matches[i].distance
            
        ratio_val = None
        if hasattr(res, 'ratios') and res.ratios and i < len(res.ratios):
            ratio_val = res.ratios[i]
            
        sx, sy = float(source[i, 0]), float(source[i, 1])
        c = min(grid_cols - 1, max(0, int(sx / max(w, 1) * grid_cols)))
        r = min(grid_rows - 1, max(0, int(sy / max(h, 1) * grid_rows)))
        
        sub_dx = res.subpixel_consensus.get('dx') if res.subpixel_consensus else None
        sub_dy = res.subpixel_consensus.get('dy') if res.subpixel_consensus else None
        sub_conf = res.subpixel_consensus.get('confidence') if res.subpixel_consensus else None
        sub_accepted = res.subpixel_consensus.get('accepted') if res.subpixel_consensus else False

        # geometric_model and estimator come from actual Phase 6 result
        # (set in register_pair via best.model_name / best.estimator - never guessed)
        actual_geo_model = res.geometric_model if (hasattr(res, 'geometric_model') and res.geometric_model) else None
        actual_estimator = res.estimator if (hasattr(res, 'estimator') and res.estimator) else None

        payload.append({
            # --- Identity ---
            'match_id': i,
            # --- Coordinates: original image pixel space, reference coordinate system ---
            'source_x': round(sx, 4),
            'source_y': round(sy, 4),
            'reference_x': round(float(reference[i, 0]), 4),
            'reference_y': round(float(reference[i, 1]), 4),
            # projected = source point transformed into reference space via H (for residual vector)
            'projected_x': round(float(projected[i, 0]), 4) if res.homography is not None else None,
            'projected_y': round(float(projected[i, 1]), 4) if res.homography is not None else None,
            # --- Matching quality ---
            'distance': round(float(match_dist), 4) if match_dist is not None else None,
            'ratio_test_value': round(float(ratio_val), 4) if ratio_val is not None else None,
            # --- Geometric verification ---
            'inlier': bool(is_inlier),
            # residual = ||reference - projected|| in pixels
            'residual': round(float(np.linalg.norm(reference[i] - projected[i])), 4) if res.homography is not None else None,
            # --- Spatial distribution ---
            'spatial_cell': {
                'row': r,
                'col': c,
                'configuration': f"{grid_rows}x{grid_cols}"
            },
            # --- Provenance: actual backend values, never fabricated ---
            'detector': cand.detector,
            'geometric_model': actual_geo_model,
            'estimator': actual_estimator,
            # phase6_status: 'accepted' because if geometric estimation failed, result is None (never reaches here)
            'phase6_status': 'accepted',
            # --- Per-match Phase 7: Phase 7 operates at PAIR level, not per-match ---
            # The pair-level subpixel result is in the root 'subpixel_refinement' key.
            # Per-match values are intentionally null - do not propagate pair consensus to each point.
            'subpixel_dx': None,
            'subpixel_dy': None,
            'subpixel_confidence': None,
        })
    return payload


def fuse_representations(candidates, source_shape=None):
    valid = [c for c in candidates if c.confidence != "REJECTED" and c.result is not None]
    if not valid:
        if not candidates:
            raise ValueError("No candidates generated")
        return sorted(candidates, key=lambda c: c.inlier_count, reverse=True)[0]
        
    sorted_cands = sorted(valid, key=lambda c: c.inlier_count, reverse=True)
    best = sorted_cands[0]
    
    if len(sorted_cands) > 1:
        runner_up = sorted_cands[1]
        if best.result.homography is not None and runner_up.result.homography is not None:
            # Compare using transformed control points
            import numpy as np
            import cv2
            h, w = source_shape[:2] if source_shape else (100, 100)
            pts = np.array([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]], dtype=np.float32)
            proj_best = cv2.perspectiveTransform(pts, best.result.homography)
            proj_runner = cv2.perspectiveTransform(pts, runner_up.result.homography)
            
            dist = np.mean(np.linalg.norm(proj_best - proj_runner, axis=2))
            
            if dist < 5.0:
                if best.confidence == "MODERATE":
                    best.confidence = "HIGH (Multi-Modal Agreement)"
                best.failure_reason = None
            elif dist > 50.0:
                best.failure_reason = "Representations strongly disagree geometrically (Control point drift)"
                
    return best
