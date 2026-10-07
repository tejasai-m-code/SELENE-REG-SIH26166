from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.services.feature_matching import (
    FeatureResult,
    match_features,
    match_feature_artifacts,
    spatially_distribute_matches,
)
from app.services.multiscale import match_multiscale
from app.services.geometry import (
    estimate_geometric_model,
    validate_overlap,
    validate_transform_plausibility,
    decompose_transform,
)
from app.services.metrics import compute_metrics
from app.services.preprocessing import preprocess, preprocess_with_diagnostics, compare_illumination_pair
from app.services.registration import (
    compute_registered_composite,
    draw_matches,
    refine_subpixel,
    warp_to_reference,
)
from app.services.subpixel import normalize_refinement_methods
from app.services.gpu_accelerator import get_gpu_info, is_cuda_available


def _notify_progress(
    callback: Any,
    stage: str,
    stage_index: int,
    stage_count: int = 12,
    progress: float | None = None,
    message: str = "",
    status: str = "RUNNING",
    error: str | None = None,
) -> None:
    if callable(callback):
        try:
            if progress is None:
                progress = round(stage_index / stage_count, 2)
            callback(
                stage=stage,
                stage_index=stage_index,
                stage_count=stage_count,
                progress=progress,
                message=message,
                status=status,
                error=error,
            )
        except Exception:
            pass


@dataclass
class PairwiseRegistrationResult:
    homography: np.ndarray
    inlier_mask: np.ndarray
    metrics: dict
    registered: np.ndarray | None
    match_visualization: np.ndarray
    source_points: np.ndarray
    reference_points: np.ndarray
    final_reference_points: np.ndarray
    raw_match_count: int
    post_distribution_match_count: int
    ecc_used: bool
    ecc_correlation: float | None
    detector: str
    registration_status: str
    success: bool
    raw_rmse: float | None = None
    refined_rmse: float | None = None
    rmse_improvement: float | None = None
    processing_time_seconds: float = 0.0
    preprocessing: dict | None = None
    subpixel: dict | None = None
    inlier_investigation: dict | None = None
    geometric_model: str = "homography"
    estimator_used: str = "RANSAC"
    overlap_valid: bool = True
    overlap_ratio: float = 1.0
    gsd_handling: dict | None = None
    multiscale_diagnostics: dict | None = None
    stage_timings: dict | None = None
    footprint: dict | None = None
    composite: np.ndarray | None = None
    composite_metadata: dict | None = None
    match_confidences: list[float] | None = None

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
    match_preview_max_side: int | None = None,
    include_registered: bool = True,
    source_gray: np.ndarray | None = None,
    reference_gray: np.ndarray | None = None,
    source_features: FeatureResult | None = None,
    reference_features: FeatureResult | None = None,
    refinement_methods: list[str] | str | None = None,
    source_metadata: dict | None = None,
    reference_metadata: dict | None = None,
    radiometric_mode: str = "safe_normalization",
    representation: str = "structural",
    progress_callback: Any = None,
    prefer_affine: bool = False,
    geometric_model: str = "auto",
    execution_mode: str = "hybrid",
) -> PairwiseRegistrationResult:
    """Register ``source_image`` into the coordinate system of ``reference_image``.

    This is shared by the established pair endpoint and the multi-image builder so
    both modes use identical preprocessing, matching, and quality measurements.
    """
    ratio = max(0.55, min(float(ratio), 0.95))
    ransac_threshold = max(0.5, min(float(ransac_threshold), 20.0))
    max_features = max(500, min(int(max_features), 20000))

    started = perf_counter()
    src_preprocessed = None
    ref_preprocessed = None

    _notify_progress(
        progress_callback,
        stage="PREPROCESSING",
        stage_index=3,
        stage_count=12,
        progress=round(3 / 12, 2),
        message=f"Applying {representation} representation and illumination normalization",
    )

    # Phase 8: Candidate representations ledger and auto-selection evaluation
    auto_selection_data = None
    evaluated_candidates = ["raw", "percentile", "clahe", "gradient", "highpass", "retinex", "structural"]
    
    if str(representation).lower() == "auto":
        _notify_progress(
            progress_callback,
            stage="PREPROCESSING",
            stage_index=3,
            stage_count=12,
            progress=round(3 / 12, 2),
            message="Auto-evaluating 7 candidate radiometric representations with numerical validation",
        )
        
        candidate_scores = []
        representation_ledger = []
        best_candidate_rep = "structural"
        best_score = -1.0
        best_reason = "Default fallback"

        # Precompute candidate image pairs
        candidate_rasters = {}
        for cand in evaluated_candidates:
            s_cand = preprocess_with_diagnostics(source_image, illumination_normalization, cand, source_metadata, radiometric_mode)
            r_cand = preprocess_with_diagnostics(reference_image, illumination_normalization, cand, reference_metadata, radiometric_mode)
            candidate_rasters[cand] = (s_cand, r_cand)

        for cand in evaluated_candidates:
            t_cand_start = perf_counter()
            s_cand, r_cand = candidate_rasters[cand]
            s_img, r_img = s_cand.image, r_cand.image
            
            try:
                # Fast feature extraction & matching for representation evaluation
                ms_cand = match_multiscale(
                    s_img, r_img, detector, ratio, max_features=min(max_features, 3000), ransac_threshold=ransac_threshold, levels=2
                )
                cand_matches = ms_cand.matches
                raw_count = len(cand_matches.good_matches)
                if spatial_distribution:
                    cand_matches = spatially_distribute_matches(cand_matches, s_img.shape)
                
                geom_cand = estimate_geometric_model(
                    cand_matches.source_points,
                    cand_matches.reference_points,
                    source_shape=s_img.shape,
                    reference_shape=r_img.shape,
                    ransac_threshold=ransac_threshold,
                    prefer_affine=prefer_affine,
                    model_type=geometric_model,
                )
                
                inlier_cnt = int(np.sum(geom_cand.inlier_mask)) if geom_cand.inlier_mask is not None else 0
                tot_matches = len(cand_matches.source_points) if cand_matches.source_points is not None else 0
                inlier_rat = float(inlier_cnt / tot_matches) if tot_matches > 0 else 0.0
                cand_rmse = float(geom_cand.rmse_pixels) if geom_cand.rmse_pixels is not None else 999.0
                
                # Spatial coverage calculation
                if inlier_cnt >= 4 and geom_cand.inlier_mask is not None:
                    mask_flat = np.asarray(geom_cand.inlier_mask).ravel().astype(bool)
                    inlier_pts = np.asarray(cand_matches.source_points)[mask_flat].reshape(-1, 2)
                    if len(inlier_pts) >= 4:
                        x_span = float(np.ptp(inlier_pts[:, 0])) / max(1, s_img.shape[1])
                        y_span = float(np.ptp(inlier_pts[:, 1])) / max(1, s_img.shape[0])
                        cand_coverage = float(np.clip(x_span * y_span, 0.0, 1.0))
                    else:
                        cand_coverage = 0.0
                else:
                    cand_coverage = 0.0

                t_cand_eval = perf_counter() - t_cand_start

                # Status check based on engineering thresholds
                if inlier_cnt >= 12 and inlier_rat >= 0.20 and cand_rmse <= 3.5 and geom_cand.is_valid:
                    cand_status = "PASS"
                elif inlier_cnt >= 6 and inlier_rat >= 0.10 and cand_rmse <= 8.0 and geom_cand.is_valid:
                    cand_status = "WARN"
                else:
                    cand_status = "FAIL"

                # Numerical scoring: reward verified inliers, inlier ratio, and spatial coverage; penalize RMSE
                if geom_cand.is_valid and inlier_cnt >= 6:
                    score = float((inlier_cnt * (inlier_rat ** 0.5) * (0.2 + 0.8 * cand_coverage)) / (1.0 + 0.2 * cand_rmse))
                else:
                    score = 0.0

                ledger_entry = {
                    "representation": cand,
                    "matches": tot_matches,
                    "inliers": inlier_cnt,
                    "inlier_ratio": round(inlier_rat, 4),
                    "rmse": round(cand_rmse, 4) if cand_rmse < 900.0 else None,
                    "coverage": round(cand_coverage, 4),
                    "processing_time_s": round(t_cand_eval, 3),
                    "status": cand_status,
                    "score": round(score, 3),
                    "is_valid": bool(geom_cand.is_valid and inlier_cnt >= 6),
                }
                representation_ledger.append(ledger_entry)

                if score > best_score and geom_cand.is_valid and inlier_cnt >= 6:
                    best_score = score
                    best_candidate_rep = cand
                    best_reason = (
                        f"Ranked highest among 7 candidates with score {score:.2f}: "
                        f"{inlier_cnt} verified inliers ({inlier_rat:.1%}), "
                        f"residual RMSE {cand_rmse:.3f}px, coverage {cand_coverage:.1%}"
                    )
            except Exception as e:
                t_cand_eval = perf_counter() - t_cand_start
                representation_ledger.append({
                    "representation": cand,
                    "matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "rmse": None,
                    "coverage": 0.0,
                    "processing_time_s": round(t_cand_eval, 3),
                    "status": "FAIL",
                    "score": 0.0,
                    "is_valid": False,
                    "error": str(e),
                })

        # Check if all failed
        valid_candidates = [c for c in representation_ledger if c.get("is_valid")]
        if not valid_candidates:
            best_candidate_rep = "structural"
            best_reason = "No candidate representation satisfied minimal geometric validity; defaulted to structural baseline."

        # Assign selected representation for subsequent full pipeline execution
        effective_representation = best_candidate_rep
        src_preprocessed, ref_preprocessed = candidate_rasters[effective_representation]
        src_gray = src_preprocessed.image
        ref_gray = ref_preprocessed.image

        selected_ledger_entry = next((c for c in representation_ledger if c["representation"] == effective_representation), None)

        auto_selection_data = {
            "mode": "AUTO",
            "selected_representation": effective_representation,
            "selection_score": round(best_score, 3),
            "selection_reason": best_reason,
            "evaluated_candidates": evaluated_candidates,
            "ledger": representation_ledger,
            "metrics": {
                "inliers": selected_ledger_entry["inliers"] if selected_ledger_entry else 0,
                "inlier_ratio": selected_ledger_entry["inlier_ratio"] if selected_ledger_entry else 0.0,
                "rmse": selected_ledger_entry["rmse"] if selected_ledger_entry else None,
                "coverage": selected_ledger_entry["coverage"] if selected_ledger_entry else 0.0,
                "processing_time_s": selected_ledger_entry["processing_time_s"] if selected_ledger_entry else 0.0,
            }
        }
    else:
        effective_representation = str(representation).lower()
        if source_gray is None:
            src_preprocessed = preprocess_with_diagnostics(
                source_image,
                illumination_normalization,
                effective_representation,
                source_metadata,
                radiometric_mode,
            )
            src_gray = src_preprocessed.image
        else:
            src_gray = source_gray
        if reference_gray is None:
            ref_preprocessed = preprocess_with_diagnostics(
                reference_image,
                illumination_normalization,
                effective_representation,
                reference_metadata,
                radiometric_mode,
            )
            ref_gray = ref_preprocessed.image
        else:
            ref_gray = reference_gray

        auto_selection_data = {
            "mode": "MANUAL",
            "selected_representation": effective_representation,
            "selection_score": None,
            "selection_reason": f"Manual operator selection: {effective_representation}",
            "evaluated_candidates": [effective_representation],
            "ledger": [{
                "representation": effective_representation,
                "status": "PASS",
                "mode": "MANUAL",
            }],
        }

    stage_timings = {}
    stage_timings["preprocessing"] = perf_counter() - started

    # 1. Feature detection and correspondence matching
    t0 = perf_counter()
    multiscale_diagnostics = None
    try:
        if source_features is not None and reference_features is not None:
            _notify_progress(
                progress_callback,
                stage="FEATURE_DETECTION",
                stage_index=4,
                stage_count=12,
                progress=round(4 / 12, 2),
                message="Using precomputed keypoint descriptors",
            )
            _notify_progress(
                progress_callback,
                stage="FEATURE_MATCHING",
                stage_index=5,
                stage_count=12,
                progress=round(5 / 12, 2),
                message=f"Matching precomputed {detector.upper()} descriptors with ratio test",
            )
            matches = match_feature_artifacts(source_features, reference_features, detector, ratio)
            raw_match_count = len(matches.good_matches)
            if spatial_distribution:
                _notify_progress(
                    progress_callback,
                    stage="SPATIAL_FILTERING",
                    stage_index=6,
                    stage_count=12,
                    progress=round(6 / 12, 2),
                    message=f"Enforcing spatial distribution across {raw_match_count} matches",
                )
                matches = spatially_distribute_matches(matches, src_gray.shape)
            else:
                _notify_progress(
                    progress_callback,
                    stage="SPATIAL_FILTERING",
                    stage_index=6,
                    stage_count=12,
                    progress=round(6 / 12, 2),
                    message="Spatial distribution disabled; retaining raw correspondences",
                )

            _notify_progress(
                progress_callback,
                stage="GEOMETRIC_ESTIMATION",
                stage_index=7,
                stage_count=12,
                progress=round(7 / 12, 2),
                message=f"Estimating geometric model with RANSAC threshold {ransac_threshold:.1f}px",
            )
            geom = estimate_geometric_model(
                matches.source_points,
                matches.reference_points,
                source_shape=src_gray.shape,
                reference_shape=ref_gray.shape,
                ransac_threshold=ransac_threshold,
                prefer_affine=False
            )
            H_raw = geom.homography
            raw_inlier_mask = geom.inlier_mask
            model_type = geom.model_type
            estimator_used = geom.estimator_used
            overlap_valid = geom.overlap_valid
            overlap_ratio = geom.overlap_ratio
        else:
            # Check GSD scaling
            gsd_s = source_metadata.get("gsd_m") if source_metadata else None
            gsd_r = reference_metadata.get("gsd_m") if reference_metadata else None
            
            # Use MultiScale
            _notify_progress(
                progress_callback,
                stage="FEATURE_DETECTION",
                stage_index=4,
                stage_count=12,
                progress=round(4 / 12, 2),
                message=f"Extracting multi-scale {detector.upper()} keypoints and descriptors",
            )
            ms_result = match_multiscale(
                src_gray, ref_gray, detector, ratio, max_features, ransac_threshold, levels=2, progress_callback=progress_callback
            )
            matches = ms_result.matches
            multiscale_diagnostics = ms_result.diagnostics
            
            raw_match_count = len(matches.good_matches)
            if spatial_distribution:
                _notify_progress(
                    progress_callback,
                    stage="SPATIAL_FILTERING",
                    stage_index=6,
                    stage_count=12,
                    progress=round(6 / 12, 2),
                    message=f"Enforcing spatial distribution and adaptive grid filtering on {raw_match_count} matches",
                )
                matches = spatially_distribute_matches(matches, src_gray.shape)
            else:
                _notify_progress(
                    progress_callback,
                    stage="SPATIAL_FILTERING",
                    stage_index=6,
                    stage_count=12,
                    progress=round(6 / 12, 2),
                    message="Spatial distribution disabled; retaining raw correspondences",
                )
                
            # Re-evaluate robust model on final spatially distributed fine matches
            _notify_progress(
                progress_callback,
                stage="GEOMETRIC_ESTIMATION",
                stage_index=7,
                stage_count=12,
                progress=round(7 / 12, 2),
                message=f"Robust RANSAC geometric model estimation with threshold {ransac_threshold:.1f}px",
            )
            geom = estimate_geometric_model(
                matches.source_points,
                matches.reference_points,
                source_shape=src_gray.shape,
                reference_shape=ref_gray.shape,
                ransac_threshold=ransac_threshold,
                prefer_affine=prefer_affine,
                model_type=geometric_model,
            )
            if geom.is_valid:
                H_raw = geom.homography
                raw_inlier_mask = geom.inlier_mask
                model_type = geom.model_type
                estimator_used = geom.estimator_used
                overlap_valid = geom.overlap_valid
                overlap_ratio = geom.overlap_ratio
            else:
                H_raw = geom.homography
                raw_inlier_mask = geom.inlier_mask
                model_type = geom.model_type
                estimator_used = geom.estimator_used
                overlap_valid = False
                overlap_ratio = 0.0
    except ValueError as e:
        _notify_progress(
            progress_callback,
            stage="FAILED",
            stage_index=5,
            stage_count=12,
            progress=round(5 / 12, 2),
            message=f"Registration failed: {str(e)}",
            status="FAILED",
            error=str(e),
        )
        failed_investigation = {
            "raw_ratio_test_matches": 0,
            "spatially_selected_matches": 0,
            "geometrically_verified_matches": 0,
            "final_inliers": 0,
            "outliers_after_geometry": 0,
            "descriptor": detector.upper(),
            "ratio_test": float(ratio),
            "ransac_threshold_pixels": float(ransac_threshold),
            "quality_gate": {
                "status": "FAIL",
                "success": False,
                "case_classification": "CASE_C_POOR_CORRESPONDENCE",
                "primary_reason": str(e),
                "human_readable_summary": f"REJECTED (CASE_C_POOR_CORRESPONDENCE) — {str(e)}",
                "gate_checks": {
                    "minimum_inliers": {
                        "threshold": 6,
                        "actual": 0,
                        "passed": False,
                        "severity": "critical",
                        "description": "At least 6 geometrically verified inliers required for non-degenerate homography.",
                    }
                },
                "failed_checks": ["minimum_inliers"],
                "critical_failures": ["minimum_inliers"],
                "warnings": [],
                "reason": str(e),
            },
            "subpixel_validation": {
                "status": "SUBPIXEL_NOT_VALIDATED",
                "reason": "Registration aborted before sub-pixel refinement.",
            },
            "transform_decomposition": None,
            "transform_plausibility_issues": [str(e)],
        }
        return PairwiseRegistrationResult(
            homography=np.eye(3),
            inlier_mask=np.array([]),
            metrics={"failure_reason": str(e)},
            registered=None,
            match_visualization=np.zeros_like(src_gray),
            source_points=np.array([]),
            reference_points=np.array([]),
            final_reference_points=np.array([]),
            raw_match_count=0,
            post_distribution_match_count=0,
            ecc_used=False,
            ecc_correlation=None,
            detector=detector,
            registration_status="FAIL",
            success=False,
            processing_time_seconds=perf_counter() - started,
            preprocessing={
                "representation": src_preprocessed.method if src_preprocessed else representation,
                "source_method": src_preprocessed.method if src_preprocessed else "provided_preprocessed_image",
                "reference_method": ref_preprocessed.method if ref_preprocessed else "provided_preprocessed_image",
                "source_radiometric": src_preprocessed.radiometric if src_preprocessed else None,
                "reference_radiometric": ref_preprocessed.radiometric if ref_preprocessed else None,
                "available_representations": ["auto", "raw", "percentile", "clahe", "gradient", "highpass", "retinex", "structural"],
                "auto_selection": auto_selection_data,
            },
            subpixel=None,
            inlier_investigation=failed_investigation,
            geometric_model="none",
            estimator_used="none",
            overlap_valid=False,
            overlap_ratio=0.0,
            gsd_handling=None,
            multiscale_diagnostics=None,
            stage_timings=stage_timings
        )

    stage_timings["feature_detection_and_matching"] = perf_counter() - t0

    # Check validity
    is_valid = geom.is_valid and overlap_valid

    # Compute raw residuals before any sub-pixel refinement
    raw_src_pts = matches.source_points.reshape(-1, 2).astype(np.float32)
    raw_ref_pts = matches.reference_points.reshape(-1, 2).astype(np.float32)
    
    if H_raw is not None and np.isfinite(H_raw).all() and len(raw_src_pts) > 0:
        if model_type == "homography":
            raw_pred = cv2.perspectiveTransform(raw_src_pts.reshape(-1, 1, 2), H_raw).reshape(-1, 2)
        else:
            # H_raw for affine is 3x3 but we need 2x3 for cv2.transform
            raw_pred = cv2.transform(raw_src_pts.reshape(-1, 1, 2), H_raw[:2, :]).reshape(-1, 2)
            
        raw_residuals = np.linalg.norm(raw_pred - raw_ref_pts, axis=1)
        # Update raw_inlier_mask based on geometry result (which evaluates on the full set)
        raw_inlier_residuals = raw_residuals[raw_inlier_mask] if np.any(raw_inlier_mask) else np.array([])
        raw_rmse = float(np.sqrt(np.mean(raw_inlier_residuals ** 2))) if len(raw_inlier_residuals) else None
    else:
        raw_rmse = None


    # 3. Sub-pixel refinement (normalize UI aliases and execute selected methods)
    if refinement_methods is not None:
        canonical_methods = normalize_refinement_methods(refinement_methods)
    elif ecc_refinement:
        canonical_methods = ["ecc"]
    else:
        canonical_methods = []

    _notify_progress(
        progress_callback,
        stage="SUBPIXEL_REFINEMENT",
        stage_index=8,
        stage_count=12,
        progress=round(8 / 12, 2),
        message=f"Refining correspondences with sub-pixel methods ({', '.join(canonical_methods) if canonical_methods else 'none'})",
    )
    subpixel = refine_subpixel(
        src_gray,
        ref_gray,
        matches.source_points,
        matches.reference_points,
        H_raw,
        canonical_methods,
    )
    H_final = subpixel["homography"]
    ecc_row = next((item for item in subpixel["method_status"] if item.get("canonical_method") == "ecc" or item.get("method") == "ecc"), {})
    ecc_score = ecc_row.get("correlation")
    ecc_used = bool(ecc_row.get("succeeded", ecc_row.get("used", False)))

    # 4. Refined-correspondence data flow:
    # If point-wise subpixel refinement succeeded, final metrics and residuals MUST use
    # the refined reference points. If only global refinement or no refinement was applied,
    # the points are the original reference points.
    has_point_refinement = subpixel.get("has_point_refinement", False)
    if has_point_refinement:
        final_reference_points = subpixel["refined_reference_points"]
        correspondence_basis = "refined"
    else:
        final_reference_points = raw_ref_pts
        correspondence_basis = "raw"

    t1 = perf_counter()
    # 5. Plausibility check on H_final: if sub-pixel refinement corrupted geometry, revert to H_raw
    if H_final is not None and np.isfinite(H_final).all():
        is_h_final_plaus, plaus_issues, _ = validate_transform_plausibility(
            H_final, source_shape=src_gray.shape, reference_shape=ref_gray.shape
        )
        if not is_h_final_plaus and H_raw is not None:
            raw_plaus, _, _ = validate_transform_plausibility(
                H_raw, source_shape=src_gray.shape, reference_shape=ref_gray.shape
            )
            if raw_plaus:
                H_final = H_raw
                final_reference_points = raw_ref_pts
                correspondence_basis = "raw"

    # Final residuals and final inlier mask (consistent with final H and final points)
    if H_final is not None and np.isfinite(H_final).all() and len(raw_src_pts) > 0:
        if model_type == "homography":
            final_pred = cv2.perspectiveTransform(raw_src_pts.reshape(-1, 1, 2), H_final).reshape(-1, 2)
        else:
            final_pred = cv2.transform(raw_src_pts.reshape(-1, 1, 2), H_final[:2, :]).reshape(-1, 2)
        final_residuals = np.linalg.norm(final_pred - final_reference_points, axis=1)
        final_inlier_mask = final_residuals <= ransac_threshold
        final_inlier_residuals = final_residuals[final_inlier_mask] if np.any(final_inlier_mask) else np.array([])
        final_rmse = float(np.sqrt(np.mean(final_inlier_residuals ** 2))) if len(final_inlier_residuals) else None
    else:
        final_inlier_mask = np.zeros(len(raw_src_pts), dtype=bool)
        final_rmse = None
        is_valid = False

    # Scientific error guard (Section 2 & 9):
    # Refinement is only accepted if it does not degrade residual error.
    # If refinement increased RMSE (e.g. 0.28 px -> 1.21 px), revert to H_raw to maintain optimal registration!
    # Scientific error guard (Section 2 & 9):
    # Refinement is only validated if it improves residual error.
    # If a global refinement degraded RMSE (e.g. 0.28 px -> 1.21 px) or became implausible,
    # revert H_final to H_raw to maintain optimal registration and footprint overlap.
    if (
        subpixel.get("applied_methods")
        and raw_rmse is not None
        and final_rmse is not None
        and H_raw is not None
        and not has_point_refinement
        and final_rmse > (raw_rmse + 0.05)
    ):
        degraded_rmse = final_rmse
        H_final = H_raw
        final_rmse = raw_rmse
        subpixel["subpixel_validation_status"] = "SUBPIXEL_NOT_VALIDATED"
        subpixel["subpixel_validation_reason"] = (
            f"Global refinement increased residual error ({raw_rmse:.3f} px -> {degraded_rmse:.3f} px); "
            "reverted to raw robust geometric model to preserve registration accuracy."
        )
    elif subpixel.get("applied_methods") and raw_rmse is not None and final_rmse is not None:
        improvement = raw_rmse - final_rmse
        pct = (improvement / raw_rmse) * 100.0 if raw_rmse > 0 else 0.0
        if improvement >= 0.01:
            subpixel["subpixel_validation_status"] = "SUBPIXEL_VALIDATED"
            subpixel["subpixel_validation_reason"] = (
                f"Refinement confirmed: residual RMSE reduced by {improvement:.3f} px ({pct:.1f}% improvement) "
                f"across verified correspondences."
            )
        else:
            subpixel["subpixel_validation_status"] = "SUBPIXEL_NOT_VALIDATED"
            subpixel["subpixel_validation_reason"] = (
                f"Refinement did not improve residual RMSE ({raw_rmse:.3f} px vs {final_rmse:.3f} px); "
                "subpixel validation not confirmed."
            )

    stage_timings["subpixel_refinement"] = perf_counter() - t1

    # Calculate RMSE improvement if refinement was applied
    refined_rmse = final_rmse if (has_point_refinement or subpixel.get("applied_methods")) else None
    rmse_improvement = None
    rmse_improvement_percent = None
    if (has_point_refinement or subpixel.get("applied_methods")) and raw_rmse is not None and final_rmse is not None:
        rmse_improvement = float(raw_rmse - final_rmse)
        if raw_rmse > 0:
            rmse_improvement_percent = float(100.0 * (raw_rmse - final_rmse) / raw_rmse)

    # 6. Footprint computation
    _notify_progress(
        progress_callback,
        stage="FOOTPRINT_COMPUTATION",
        stage_index=9,
        stage_count=12,
        progress=round(9 / 12, 2),
        message="Computing lunar spatial footprints and polygon overlap coverage",
    )
    footprint_info = None
    if H_final is not None and np.isfinite(H_final).all():
        try:
            from app.services.footprint import create_image_footprint
            src_fp = create_image_footprint(src_gray.shape[:2], source_id="source")
            ref_fp = create_image_footprint(ref_gray.shape[:2], source_id="reference")
            warped_src_fp = src_fp.transform(H_final, target_frame="IMAGE_PIXEL")
            overlap_metrics = ref_fp.overlap_metrics(warped_src_fp)
            footprint_info = {
                "source_footprint": src_fp.to_dict(),
                "reference_footprint": ref_fp.to_dict(),
                "warped_source_footprint": warped_src_fp.to_dict(),
                "overlap": overlap_metrics,
                "status": "COMPUTED",
            }
        except Exception as e:
            footprint_info = {
                "status": "UNAVAILABLE",
                "reason": str(e),
            }
    else:
        footprint_info = {
            "status": "UNAVAILABLE",
            "reason": "Homography not available or invalid.",
        }

    # 7. Final warps, composite calculation, and full metrics calculation
    _notify_progress(
        progress_callback,
        stage="METRICS_EVALUATION",
        stage_index=10,
        stage_count=12,
        progress=round(10 / 12, 2),
        message="Evaluating residual RMSE, coverage, and 4-state scientific quality gate",
    )
    registered = warp_to_reference(source_image, reference_image.shape, H_final) if (include_registered and is_valid) else None
    composite_image, composite_meta = (
        compute_registered_composite(source_image, reference_image, H_final)
        if (include_registered and is_valid and H_final is not None)
        else (None, None)
    )
    metrics = compute_metrics(
        matches.source_points,
        final_reference_points,
        H_final,
        final_inlier_mask,
        src_gray.shape,
        ref_gray.shape,
        registered=registered,
        reference_image=reference_image,
    )
    # Augment metrics with explicit distinction between raw and refined
    metrics["raw_rmse_pixels"] = raw_rmse
    metrics["refined_rmse_pixels"] = refined_rmse
    metrics["rmse_improvement_pixels"] = rmse_improvement
    metrics["rmse_improvement_percent"] = rmse_improvement_percent
    metrics["correspondence_basis"] = correspondence_basis
    metrics["subpixel_validation_status"] = subpixel.get("subpixel_validation_status", "SUBPIXEL_NOT_VALIDATED")
    metrics["subpixel_validation_reason"] = subpixel.get("subpixel_validation_reason", "")
    if footprint_info and footprint_info.get("overlap"):
        metrics["footprint_iou"] = float(footprint_info["overlap"].get("iou", 0.0))
        metrics["overlap_ratio"] = float(footprint_info["overlap"].get("overlap_ratio_a", 0.0))

    # Canonical correspondence counts synchronized across all consumers
    inlier_count = int(np.sum(final_inlier_mask))
    total_match_count = int(len(final_inlier_mask))
    inlier_ratio = float(inlier_count / total_match_count) if total_match_count > 0 else 0.0
    outlier_count = int(total_match_count - inlier_count)

    metrics["inlier_count"] = inlier_count
    metrics["inliers"] = inlier_count
    metrics["n_inliers"] = inlier_count
    metrics["match_count"] = total_match_count
    metrics["matches"] = total_match_count
    metrics["n_matches"] = total_match_count
    metrics["outliers"] = outlier_count
    metrics["outlier_count"] = outlier_count
    metrics["inlier_ratio"] = inlier_ratio

    # 7. Quality Gate Logic: 4 Explicit States (PASS, PASS_WITH_WARNING, REVIEW, FAIL)
    coverage = float(metrics.get("source_spatial_coverage", 0.0))
    conditioning = metrics.get("transform_conditioning")

    gate_checks = {
        "minimum_inliers": {
            "threshold": 6,
            "actual": inlier_count,
            "passed": inlier_count >= 6,
            "severity": "critical",
            "description": "At least 6 geometrically verified inliers required for non-degenerate homography.",
        },
        "minimum_inlier_ratio": {
            "threshold": 0.10,
            "actual": round(inlier_ratio, 4),
            "passed": inlier_ratio >= 0.10,
            "severity": "critical",
            "description": "At least 10% of candidate matches must be geometrically verified inliers.",
        },
        "minimum_spatial_coverage": {
            "threshold": 0.125,
            "actual": round(coverage, 4),
            "passed": coverage >= 0.125,
            "severity": "critical",
            "description": "Inliers must span at least 12.5% of the image grid area to avoid clustered registration.",
        },
        "maximum_rmse": {
            "threshold": 8.0,
            "actual": round(final_rmse, 4) if final_rmse is not None else None,
            "passed": final_rmse is not None and final_rmse <= 8.0,
            "severity": "critical",
            "description": "Geometric residual RMSE must be <= 8.0 px (engineering quality threshold).",
        },
        "clustered_distribution": {
            "threshold": "GOOD_DISTRIBUTION or LIMITED_DISTRIBUTION",
            "actual": str(metrics.get("spatial_distribution_category", "LIMITED_DISTRIBUTION")),
            "passed": str(metrics.get("spatial_distribution_category", "LIMITED_DISTRIBUTION")) != "CLUSTERED_RISKY_DISTRIBUTION",
            "severity": "critical",
            "description": "Correspondences concentrated in a single small region cannot support reliable registration.",
        },
        "transform_conditioning": {
            "threshold": 1e6,
            "actual": round(conditioning, 2) if conditioning is not None and np.isfinite(conditioning) else None,
            "passed": conditioning is not None and np.isfinite(conditioning) and conditioning < 1e6,
            "severity": "warning",
            "description": "Homography condition number should indicate stable perspective projection.",
        },
        "inlier_abundance": {
            "threshold": 10,
            "actual": inlier_count,
            "passed": inlier_count >= 10,
            "severity": "warning",
            "description": "Strong evidence recommends at least 10 inliers.",
        },
        "spatial_uniformity": {
            "threshold": 0.20,
            "actual": round(float(metrics.get("spatial_uniformity", 0.0)), 4),
            "passed": float(metrics.get("spatial_uniformity", 0.0)) >= 0.20,
            "severity": "warning",
            "description": "Even spatial spread across the scene reduces local shear distortion.",
        },
    }

    # Add transform plausibility check if decomposition is available
    transform_decomp = None
    transform_plausibility_issues: list[str] = []
    if H_final is not None and np.isfinite(H_final).all():
        plausible, plaus_issues, transform_decomp = validate_transform_plausibility(
            H_final,
            source_shape=src_gray.shape,
            reference_shape=ref_gray.shape,
        )
        transform_plausibility_issues = plaus_issues
        gate_checks["transform_plausibility"] = {
            "threshold": "No implausible parameters",
            "actual": f"{len(plaus_issues)} issues" if plaus_issues else "Plausible",
            "passed": plausible,
            "severity": "critical",
            "description": "Transformation scale, rotation, shear must be physically plausible for image registration.",
            "issues": plaus_issues,
        }
        if not plausible:
            is_valid = False

    # Evaluate 4-state registration quality
    critical_failures = [
        name for name, check in gate_checks.items()
        if check["severity"] == "critical" and not check["passed"]
    ]
    warning_failures = [
        name for name, check in gate_checks.items()
        if check["severity"] == "warning" and not check["passed"]
    ]

    # Multi-factor scientific quality decision and explicit Case classification (Section 3 & 8)
    median_err = metrics.get("median_reprojection_error_pixels")
    p90_err = metrics.get("p90_reprojection_error_pixels")
    scale_val = transform_decomp.get("scale_x", 1.0) if transform_decomp else 1.0
    rot_val = transform_decomp.get("rotation_degrees", 0.0) if transform_decomp else 0.0
    trans_val = transform_decomp.get("translation_magnitude_px", 0.0) if transform_decomp else 0.0

    if inlier_count < 6 or inlier_ratio < 0.10:
        case_classification = "CASE_C_POOR_CORRESPONDENCE"
        primary_reason = (
            f"Insufficient verified correspondences (inliers={inlier_count}, ratio={inlier_ratio:.1%}; "
            "minimum required: 6 inliers and 10% inlier ratio)."
        )
        status = "FAIL"
    elif not plausible or (transform_plausibility_issues and len(transform_plausibility_issues) > 0):
        case_classification = "CASE_B_IMPLAUSIBLE_TRANSFORM"
        primary_reason = f"Global transform is inconsistent with verified correspondences: {'; '.join(transform_plausibility_issues)}."
        status = "FAIL" if (len(transform_plausibility_issues) > 1 or inlier_count < 6) else "REVIEW"
    elif critical_failures:
        case_classification = "CASE_B_CRITICAL_FAILURE"
        primary_reason = f"Failed critical stability criteria: {', '.join(critical_failures)}."
        status = "REVIEW"
    elif (scale_val > 1.8 or scale_val < 0.55) and overlap_ratio >= 0.08:
        case_classification = "CASE_D_ACCEPTED_SCALE_ADAPTED"
        primary_reason = (
            f"Scale-adapted registration accepted: optical scale difference is {scale_val:.2f}x "
            f"with {overlap_ratio:.1%} valid overlap in reference coordinate frame."
        )
        status = "PASS"
    elif warning_failures:
        case_classification = "CASE_A_ACCEPTED_WITH_WARNINGS"
        primary_reason = f"Passed core geometric criteria with operational warnings ({', '.join(warning_failures)})."
        status = "PASS_WITH_WARNING"
    else:
        case_classification = "CASE_A_ACCEPTED"
        primary_reason = "All configured geometric evidence and stability criteria verified."
        status = "PASS"

    if status in ("FAIL", "REVIEW"):
        reason = f"REJECTED ({case_classification}) — {primary_reason}"
    else:
        reason = f"ACCEPTED ({case_classification}) — {primary_reason}"

    rmse_str = f"{final_rmse:.2f} px" if final_rmse is not None else "N/A"
    med_str = f"{median_err:.2f} px" if median_err is not None else "N/A"
    p90_str = f"{p90_err:.2f} px" if p90_err is not None else "N/A"
    cov_str = f"{coverage:.1%}" if coverage is not None else "N/A"
    human_readable_summary = (
        f"{'ACCEPTED' if status in ('PASS', 'PASS_WITH_WARNING') else 'REJECTED'} ({case_classification}) — {primary_reason}\n"
        f"Inliers: {inlier_count} | Inlier ratio: {inlier_ratio:.1%} | Residual RMSE: {rmse_str} "
        f"(Median: {med_str}, 90th%: {p90_str}) | Coverage: {cov_str}\n"
        f"Transform: Scale {scale_val:.2f}x, Rotation {rot_val:.1f} deg, Translation {trans_val:.1f} px"
    )

    success = status in ("PASS", "PASS_WITH_WARNING")

    result_descriptor = detector.upper()
    inlier_investigation = {
        "raw_ratio_test_matches": raw_match_count,
        "spatially_selected_matches": len(matches.good_matches),
        "geometrically_verified_matches": inlier_count,
        "final_inliers": inlier_count,
        "outliers_after_geometry": int(len(final_inlier_mask) - inlier_count),
        "descriptor": result_descriptor,
        "ratio_test": float(ratio),
        "ransac_threshold_pixels": float(ransac_threshold),
        "spatial_distribution_enabled": bool(spatial_distribution),
        "spatial_grid": metrics.get("source_spatial_grid"),
        "quality_gate": {
            "status": status,
            "success": success,
            "case_classification": case_classification,
            "primary_reason": primary_reason,
            "human_readable_summary": human_readable_summary,
            "gate_checks": gate_checks,
            "failed_checks": critical_failures + warning_failures,
            "critical_failures": critical_failures,
            "warnings": warning_failures,
            "reason": reason,
        },
        "subpixel_validation": {
            "status": subpixel.get("subpixel_validation_status", "SUBPIXEL_NOT_VALIDATED"),
            "reason": subpixel.get("subpixel_validation_reason", ""),
            "selected_method": subpixel.get("selected_method"),
            "candidate_methods": subpixel.get("candidate_methods", ["taylor_expansion", "lucas_kanade", "quadratic_peak", "phase_correlation", "ecc"]),
            "raw_rmse_pixels": raw_rmse,
            "refined_rmse_pixels": refined_rmse,
            "rmse_improvement_pixels": rmse_improvement,
            "correspondence_basis": correspondence_basis,
            "comparison_ledger": subpixel.get("comparison_ledger", []),
        },
        "interpretation": (
            "Inliers are only correspondences that survived geometric residual verification; "
            "raw ratio-test matches are reported separately. Final metrics use refined correspondences "
            "when point-wise sub-pixel refinement is applied."
        ),
        "transform_decomposition": transform_decomp,
        "transform_plausibility_issues": transform_plausibility_issues,
    }

    preprocessing = {
        "representation": src_preprocessed.method if src_preprocessed else representation,
        "source_method": src_preprocessed.method if src_preprocessed else "provided_preprocessed_image",
        "reference_method": ref_preprocessed.method if ref_preprocessed else "provided_preprocessed_image",
        "source_radiometric": src_preprocessed.radiometric if src_preprocessed else None,
        "reference_radiometric": ref_preprocessed.radiometric if ref_preprocessed else None,
        "source_illumination": src_preprocessed.illumination_stats if src_preprocessed else None,
        "reference_illumination": ref_preprocessed.illumination_stats if ref_preprocessed else None,
        "illumination_comparison": compare_illumination_pair(src_gray, ref_gray) if src_gray is not None and ref_gray is not None else None,
        "hardware_acceleration": src_preprocessed.hardware_acceleration if src_preprocessed else None,
        "available_representations": ["auto", "raw", "percentile", "clahe", "gradient", "highpass", "retinex", "structural"],
        "auto_selection": auto_selection_data,
    }

    # 8. Match visualization: ensure drawn points and inlier mask correspond to the final reported correspondence set
    computed_iou = None
    if footprint_info and footprint_info.get("overlap"):
        computed_iou = footprint_info["overlap"].get("iou")
    elif overlap_ratio:
        computed_iou = overlap_ratio

    visualization = _match_visualization(
        src_gray,
        ref_gray,
        matches.source_points,
        final_reference_points,
        final_inlier_mask,
        match_preview_max_side,
        registration_status=status,
        homography=H_final,
        rmse=final_rmse,
        iou=computed_iou,
        model_type=model_type,
    )

    # Add phase 2 diagnostic metrics
    metrics["stage_timings"] = stage_timings

    # Hardware acceleration telemetry & hybrid scheduler statistics
    from app.services.hybrid_scheduler import get_global_scheduler
    scheduler = get_global_scheduler()
    scheduler_telemetry = scheduler.get_telemetry()
    gpu_telemetry = get_gpu_info()
    is_cuda = is_cuda_available()

    stages_hw = {
        "preprocessing": "GPU_CUDA" if (src_preprocessed and src_preprocessed.hardware_acceleration and "cuda" in str(src_preprocessed.hardware_acceleration.get("device_used", "")).lower()) else "CPU",
        "feature_detection": "CPU",  # Feature detection (SIFTEngine / ORB) runs on multi-core CPU
        "descriptor_matching": matches.diagnostics.get("acceleration", "GPU_CUDA" if is_cuda else "CPU_FALLBACK") if (hasattr(matches, "diagnostics") and isinstance(matches.diagnostics, dict)) else ("GPU_CUDA" if is_cuda else "CPU"),
        "spatial_distribution": "CPU",
        "ransac_geometry": "CPU",
        "subpixel_refinement": "GPU_CUDA" if (subpixel.get("applied_methods") and "phase_correlation" in subpixel.get("applied_methods", []) and is_cuda) else "CPU",
        "warping": "GPU_CUDA" if is_cuda else "CPU",
    }
    gpu_stages_count = sum(1 for v in stages_hw.values() if "GPU" in v)
    total_stages_count = len(stages_hw)
    gpu_pct = round(gpu_stages_count / total_stages_count * 100.0, 1)
    cpu_pct = round(100.0 - gpu_pct, 1)

    req_mode_clean = str(execution_mode).lower().strip()
    if req_mode_clean == "cpu":
        actual_mode = "CPU"
        fallback_reason = "Operator selected pure CPU execution"
    elif not is_cuda:
        actual_mode = "CPU FALLBACK"
        fallback_reason = "CUDA runtime not available on host system"
    elif req_mode_clean == "gpu":
        actual_mode = "GPU"
        fallback_reason = None
    else:
        actual_mode = "HYBRID (GPU + CPU)" if (gpu_stages_count > 0 and is_cuda) else "CPU FALLBACK"
        fallback_reason = None

    metrics["hardware_acceleration"] = {
        "mode": actual_mode,
        "execution_mode": actual_mode,
        "requested_mode": req_mode_clean.upper(),
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "cuda_active": is_cuda,
        "device_name": gpu_telemetry.get("device_name", "CPU"),
        "allocated_vram_mb": gpu_telemetry.get("allocated_memory_mb", 0.0),
        "reserved_vram_mb": gpu_telemetry.get("reserved_memory_mb", 0.0),
        "free_vram_mb": scheduler_telemetry.get("vram_free_mb", 0.0),
        "total_vram_mb": gpu_telemetry.get("total_vram_mb", 0.0),
        "gpu_execution_percentage": gpu_pct,
        "cpu_execution_percentage": cpu_pct,
        "stages": stages_hw,
        "scheduler_telemetry": scheduler_telemetry,
    }
    inlier_investigation["hardware_acceleration"] = metrics["hardware_acceleration"]

    gsd_handling = {}
    gsd_s = source_metadata.get("gsd_m") if source_metadata else None
    gsd_r = reference_metadata.get("gsd_m") if reference_metadata else None
    if gsd_s is None or gsd_r is None:
        gsd_handling["status"] = "UNAVAILABLE_NOT_PROVIDED"
    else:
        gsd_handling["status"] = "RESCALED"
        gsd_handling["ratio"] = gsd_s / gsd_r

    if not is_valid:
        status = "FAIL"
        success = False
        inlier_investigation["quality_gate"]["status"] = status
        inlier_investigation["quality_gate"]["success"] = success
        if not inlier_investigation["quality_gate"].get("reason"):
            inlier_investigation["quality_gate"]["reason"] = reason

    return PairwiseRegistrationResult(
        homography=H_final,
        inlier_mask=final_inlier_mask,
        metrics=metrics,
        registered=registered,
        match_visualization=visualization,
        source_points=matches.source_points,
        reference_points=matches.reference_points,
        final_reference_points=final_reference_points,
        raw_match_count=raw_match_count,
        post_distribution_match_count=len(matches.good_matches),
        ecc_used=ecc_used,
        ecc_correlation=ecc_score,
        detector=result_descriptor,
        registration_status=status,
        success=success,
        raw_rmse=raw_rmse,
        refined_rmse=refined_rmse,
        rmse_improvement=rmse_improvement,
        processing_time_seconds=round(perf_counter() - started, 6),
        preprocessing=preprocessing,
        subpixel=subpixel,
        inlier_investigation=inlier_investigation,
        geometric_model=model_type,
        estimator_used=estimator_used,
        overlap_valid=overlap_valid,
        overlap_ratio=overlap_ratio,
        gsd_handling=gsd_handling,
        multiscale_diagnostics=multiscale_diagnostics,
        stage_timings=stage_timings,
        footprint=footprint_info,
        composite=composite_image,
        composite_metadata=composite_meta,
        match_confidences=getattr(matches, "match_confidences", None),
    )


def _match_visualization(
    source_gray,
    reference_gray,
    source_points,
    reference_points,
    inlier_mask,
    max_side,
    registration_status=None,
    homography=None,
    rmse=None,
    iou=None,
    model_type=None,
    show_outliers=False,
):
    """Render a canonical correspondence visualization."""
    if not max_side or max(source_gray.shape + reference_gray.shape) <= max_side:
        return draw_matches(
            source_gray,
            reference_gray,
            source_points,
            reference_points,
            inlier_mask,
            registration_status=registration_status,
            homography=homography,
            rmse=rmse,
            iou=iou,
            model_type=model_type,
            show_outliers=show_outliers,
        )

    def scale_image(image):
        scale = min(1.0, max_side / max(image.shape[:2]))
        if scale == 1.0:
            return image, scale
        return (
            cv2.resize(
                image,
                (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            ),
            scale,
        )

    source_preview, source_scale = scale_image(source_gray)
    reference_preview, reference_scale = scale_image(reference_gray)

    H_scaled = None
    if homography is not None:
        try:
            S_ref = np.diag([reference_scale, reference_scale, 1.0])
            S_src_inv = np.diag([1.0 / source_scale, 1.0 / source_scale, 1.0])
            H_scaled = S_ref @ homography @ S_src_inv
            H_scaled /= H_scaled[2, 2]
        except Exception:
            H_scaled = None

    return draw_matches(
        source_preview,
        reference_preview,
        source_points * source_scale,
        reference_points * reference_scale,
        inlier_mask,
        registration_status=registration_status,
        homography=H_scaled,
        rmse=rmse,
        iou=iou,
        model_type=model_type,
        show_outliers=show_outliers,
    )


def serialize_inlier_points(result: PairwiseRegistrationResult) -> list[dict]:
    if result is None or result.homography is None or not np.any(result.inlier_mask):
        return []
    source = np.asarray(result.source_points).reshape(-1, 2)
    initial_ref = np.asarray(result.reference_points).reshape(-1, 2)
    final_ref = np.asarray(result.final_reference_points).reshape(-1, 2) if result.final_reference_points is not None and len(result.final_reference_points) else initial_ref
    if len(source) == 0 or len(initial_ref) == 0:
        return []
    try:
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2).astype(np.float32), result.homography.astype(np.float32))
        if projected is None:
            return []
        projected = projected.reshape(-1, 2)
    except (cv2.error, ValueError):
        return []
    subpixel = result.subpixel or {}
    methods = ",".join(subpixel.get("applied_methods", [])) or "none"

    pt_details_map = {
        d["index"]: d
        for d in (subpixel.get("point_details") or [])
        if isinstance(d, dict) and "index" in d
    }

    inlier_items = []
    for i, accepted in enumerate(result.inlier_mask):
        if not (accepted and i < len(source) and i < len(final_ref) and i < len(projected)):
            continue
        pt_detail = pt_details_map.get(i + 1, {})
        res = float(np.linalg.norm(final_ref[i] - projected[i]))
        raw_res = pt_detail.get("raw_residual")
        if raw_res is None:
            raw_res = float(np.linalg.norm(initial_ref[i] - projected[i]))
        improvement = float(raw_res - res) if (raw_res is not None and res is not None) else None
        shift_mag = float(np.hypot(final_ref[i, 0] - initial_ref[i, 0], final_ref[i, 1] - initial_ref[i, 1]))

        inlier_items.append({
            "source": [round(float(source[i, 0]), 4), round(float(source[i, 1]), 4)],
            "reference": [round(float(final_ref[i, 0]), 4), round(float(final_ref[i, 1]), 4)],
            "initial_reference": [round(float(initial_ref[i, 0]), 4), round(float(initial_ref[i, 1]), 4)],
            "refined_reference": [round(float(final_ref[i, 0]), 4), round(float(final_ref[i, 1]), 4)],
            "integer_coordinate": [int(round(float(initial_ref[i, 0]))), int(round(float(initial_ref[i, 1])))],
            "refinement_delta": [round(float(final_ref[i, 0] - initial_ref[i, 0]), 4), round(float(final_ref[i, 1] - initial_ref[i, 1]), 4)],
            "shift_magnitude": round(shift_mag, 4),
            "dx": round(float(final_ref[i, 0] - projected[i, 0]), 4),
            "dy": round(float(final_ref[i, 1] - projected[i, 1]), 4),
            "residual": round(res, 4),
            "raw_residual": round(raw_res, 4) if raw_res is not None else None,
            "refined_residual": round(res, 4),
            "residual_improvement": round(improvement, 4) if improvement is not None else None,
            "error": round(res, 4),
            "confidence": round(float(result.match_confidences[i]), 3) if getattr(result, "match_confidences", None) and i < len(result.match_confidences) else None,
            "refinement_method": methods,
            "iterations": pt_detail.get("iterations", 1),
            "converged": pt_detail.get("converged", True),
            "convergence_status": pt_detail.get("convergence_status", "CONVERGED"),
            "local_correlation": pt_detail.get("local_correlation"),
            "status": "inlier",
        })
    return inlier_items


def serialize_correspondences(result: PairwiseRegistrationResult) -> list[dict]:
    """Serialize all candidate correspondences with inlier/outlier status, coordinates, and residuals."""
    if result is None or len(result.source_points) == 0:
        return []
    source = np.asarray(result.source_points).reshape(-1, 2)
    initial_ref = np.asarray(result.reference_points).reshape(-1, 2)
    final_ref = (
        np.asarray(result.final_reference_points).reshape(-1, 2)
        if result.final_reference_points is not None and len(result.final_reference_points)
        else initial_ref
    )
    if len(source) == 0 or len(initial_ref) == 0:
        return []

    projected = None
    if result.homography is not None:
        try:
            p = cv2.perspectiveTransform(
                source.reshape(-1, 1, 2).astype(np.float32),
                result.homography.astype(np.float32),
            )
            if p is not None:
                projected = p.reshape(-1, 2)
        except (cv2.error, ValueError):
            projected = None

    inlier_mask = (
        np.asarray(result.inlier_mask, dtype=bool)
        if result.inlier_mask is not None
        else np.zeros(len(source), dtype=bool)
    )

    methods = ",".join(result.subpixel.get("applied_methods", [])) if (result and result.subpixel) else "none"
    pt_details_map = {
        d["index"]: d
        for d in (result.subpixel.get("point_details") or [])
        if isinstance(d, dict) and "index" in d
    } if (result and result.subpixel) else {}

    sw = max(1.0, float(np.max(source[:, 0])) * 1.05 if len(source) else 1024.0)
    sh = max(1.0, float(np.max(source[:, 1])) * 1.05 if len(source) else 1024.0)
    rw = max(1.0, float(np.max(final_ref[:, 0])) * 1.05 if len(final_ref) else 1024.0)
    rh = max(1.0, float(np.max(final_ref[:, 1])) * 1.05 if len(final_ref) else 1024.0)

    items = []
    n = min(len(source), len(final_ref))
    for i in range(n):
        is_inlier = bool(inlier_mask[i]) if i < len(inlier_mask) else False
        res = (
            float(np.linalg.norm(final_ref[i] - projected[i]))
            if (projected is not None and i < len(projected))
            else None
        )
        pt_detail = pt_details_map.get(i + 1, {})
        raw_res = pt_detail.get("raw_residual")
        if raw_res is None and projected is not None and i < len(projected):
            raw_res = float(np.linalg.norm(initial_ref[i] - projected[i]))

        improvement = float(raw_res - res) if (raw_res is not None and res is not None) else None
        shift_mag = float(np.hypot(final_ref[i, 0] - initial_ref[i, 0], final_ref[i, 1] - initial_ref[i, 1]))

        s_col = min(3, max(0, int(source[i, 0] / (sw / 4.0))))
        s_row = min(3, max(0, int(source[i, 1] / (sh / 4.0))))
        s_cell = s_row * 4 + s_col + 1

        r_col = min(3, max(0, int(final_ref[i, 0] / (rw / 4.0))))
        r_row = min(3, max(0, int(final_ref[i, 1] / (rh / 4.0))))
        r_cell = r_row * 4 + r_col + 1

        items.append({
            "index": i + 1,
            "source": [round(float(source[i, 0]), 4), round(float(source[i, 1]), 4)],
            "reference": [round(float(final_ref[i, 0]), 4), round(float(final_ref[i, 1]), 4)],
            "initial_reference": [round(float(initial_ref[i, 0]), 4), round(float(initial_ref[i, 1]), 4)],
            "refined_reference": [round(float(final_ref[i, 0]), 4), round(float(final_ref[i, 1]), 4)],
            "integer_coordinate": [int(round(float(initial_ref[i, 0]))), int(round(float(initial_ref[i, 1])))],
            "refinement_delta": [round(float(final_ref[i, 0] - initial_ref[i, 0]), 4), round(float(final_ref[i, 1] - initial_ref[i, 1]), 4)],
            "shift_magnitude": round(shift_mag, 4),
            "residual": round(res, 4) if res is not None else None,
            "raw_residual": round(raw_res, 4) if raw_res is not None else None,
            "refined_residual": round(res, 4) if res is not None else None,
            "residual_improvement": round(improvement, 4) if improvement is not None else None,
            "error": round(res, 4) if res is not None else None,
            "confidence": round(float(result.match_confidences[i]), 3) if getattr(result, "match_confidences", None) and i < len(result.match_confidences) else None,
            "refinement_method": methods,
            "iterations": pt_detail.get("iterations", 1),
            "converged": pt_detail.get("converged", is_inlier),
            "convergence_status": pt_detail.get("convergence_status", "CONVERGED" if is_inlier else "UNREFINED"),
            "local_correlation": pt_detail.get("local_correlation"),
            "status": "INLIER" if is_inlier else "OUTLIER",
            "is_inlier": is_inlier,
            "spatial_cell": s_cell,
            "spatial_coverage": f"Cell {s_cell} / 16 (Row {s_row + 1}/4, Col {s_col + 1}/4)",
            "source_cell": s_cell,
            "reference_cell": r_cell,
        })
    return items
