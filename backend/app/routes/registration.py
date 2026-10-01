from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.services.scale import coarse_to_fine_register_pair
from app.utils.image_utils import decode_upload, save_image, resize_for_preview
from app.services.radiometry import process_raster, RadiometricConfig
from app.services.representation import build_representations
from app.services.cross_modal_matching import execute_cross_modal_matching, fuse_representations, serialize_match_payload, inspect_deep_matching_capability

router = APIRouter(tags=["registration"])

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"


@router.post("/register")
async def register_images(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    detector: str = Form("sift"),
    ratio: float = Form(0.72),
    ransac_threshold: float = Form(3.0),
    illumination_normalization: bool = Form(True),
    spatial_distribution: bool = Form(True),
    ecc_refinement: bool = Form(True),
    max_features: int = Form(8000),
    geometric_model: str = Form("auto"),
):
    job_id = uuid4().hex[:12]
    try:
        source_bytes = await source.read()
        reference_bytes = await reference.read()

        src_obj = decode_upload(source_bytes, source.filename or "")
        ref_obj = decode_upload(reference_bytes, reference.filename or "")
        
        # Radiometry is processed inherently during representation generation
        config = RadiometricConfig(enable_illumination_normalization=illumination_normalization)
        src_reps = build_representations(src_obj, config)
        ref_reps = build_representations(ref_obj, config)

        candidates = execute_cross_modal_matching(
            src_reps, ref_reps,
            source_display=src_obj.display_image,
            reference_display=ref_obj.display_image,
            source_gsd=src_obj.metadata.spatial.gsd,
            reference_gsd=ref_obj.metadata.spatial.gsd,
            gsd_unit=src_obj.metadata.spatial.gsd_unit or "UNKNOWN",
            detector=detector, ratio=ratio, ransac_threshold=ransac_threshold,
            illumination_normalization=illumination_normalization,
            spatial_distribution=spatial_distribution,
            ecc_refinement=ecc_refinement,
            max_features=max_features,
            geometric_model=geometric_model,
        )

        best_candidate = fuse_representations(candidates, source_shape=src_reps[list(src_reps.keys())[0]].image.shape)
        if best_candidate.result is None:
            raise ValueError(f"Cross-modal registration failed. Best candidate reason: {best_candidate.failure_reason}")

        result = best_candidate.result
        geom_cands = result.geometric_candidates if result and hasattr(result, "geometric_candidates") else []
        geom_info = [
            {
                "model": c.model_name,
                "estimator": c.estimator,
                "parameter_count": c.parameter_count,
                "minimum_points": c.minimum_points,
                "candidate_count": c.candidate_count,
                "inlier_count": c.inlier_count,
                "inlier_ratio": c.inlier_ratio,
                "median_reprojection_error": c.median_reprojection_error,
                "mean_reprojection_error": c.mean_reprojection_error,
                "RMSE": c.rmse,
                "robust_residual": c.robust_residual,
                "spatial_coverage": c.spatial_coverage,
                "degeneracy_status": c.degeneracy_status,
                "conditioning": c.conditioning,
                "selection_score": c.selection_score,
                "selected": c.selected,
                "failure_reason": c.failure_reason,
            } for c in geom_cands if c is not None
        ] if geom_cands else []


        # Save representations visually for traceability.
        save_image(UPLOAD_DIR / f"{job_id}_source.png", resize_for_preview(src_obj.display_image))
        save_image(UPLOAD_DIR / f"{job_id}_reference.png", resize_for_preview(ref_obj.display_image))

        reg_path = OUTPUT_DIR / f"{job_id}_registered.png"
        match_path = OUTPUT_DIR / f"{job_id}_matches.png"
        save_image(reg_path, resize_for_preview(result.registered))
        save_image(match_path, resize_for_preview(result.match_visualization))
        
        feature_response_source_path = OUTPUT_DIR / f"{job_id}_feature_response_source.png"
        feature_response_reference_path = OUTPUT_DIR / f"{job_id}_feature_response_reference.png"
        keypoints_source_path = OUTPUT_DIR / f"{job_id}_keypoints_source.png"
        keypoints_reference_path = OUTPUT_DIR / f"{job_id}_keypoints_reference.png"
        
        save_image(feature_response_source_path, resize_for_preview(result.feature_response_source))
        save_image(feature_response_reference_path, resize_for_preview(result.feature_response_reference))
        save_image(keypoints_source_path, resize_for_preview(result.keypoints_source_map))
        save_image(keypoints_reference_path, resize_for_preview(result.keypoints_reference_map))

        deep_matching_state = inspect_deep_matching_capability()

        return {
            "success": True,
            "job_id": job_id,
            "problem_statement": "SIH26166",
            "pipeline": [
                "image ingestion",
                "scientific representation extraction",
                "multi-modal candidate matching",
                "representation fusion",
                f"{best_candidate.detector} feature detection",
                "ratio-test correspondence",
                "spatial distribution constraint" if spatial_distribution else "raw correspondence",
                "RANSAC geometric verification",
                "subpixel consensus refinement" if result.ecc_used else "geometric registration",
                "warp to reference",
                "RMSE / inlier / coverage evaluation",
            ],
            "source": {
                "filename": source.filename,
                "width": int(src_reps["radiometric_normalized"].image.shape[1]),
                "height": int(src_reps["radiometric_normalized"].image.shape[0]),
            },
            "reference": {
                "filename": reference.filename,
                "width": int(ref_reps["radiometric_normalized"].image.shape[1]),
                "height": int(ref_reps["radiometric_normalized"].image.shape[0]),
            },
            "deep_matching_state": deep_matching_state.__dict__,
            "cross_modal": {
                "selected_source_representation": best_candidate.representation_source,
                "selected_reference_representation": best_candidate.representation_reference,
                "fusion_confidence": best_candidate.confidence,
                "fusion_failure_reason": best_candidate.failure_reason,
                "selected_geometric_model": best_candidate.geometric_model,
                "geometric_estimator": best_candidate.estimator,
                "geometric_candidates": geom_info,
            },
            "settings": {
                "detector": best_candidate.detector,
                "ratio": float(ratio),
                "ransac_threshold": float(ransac_threshold),
                "illumination_normalization": illumination_normalization,
                "spatial_distribution": spatial_distribution,
                "ecc_refinement": ecc_refinement,
                "ecc_used": result.ecc_used,
                "ecc_correlation": result.ecc_correlation,
                "raw_match_count": result.raw_match_count,
                "post_distribution_match_count": result.post_distribution_match_count,
            },
            "subpixel_refinement": result.subpixel_consensus,
            "metrics": result.metrics,
            "homography": result.homography.tolist(),
            "inlier_points": serialize_match_payload(best_candidate, source_shape=src_reps[best_candidate.representation_source].image.shape),
            "outputs": {
                "registered_image": f"/outputs/{job_id}_registered.png",
                "match_visualization": f"/outputs/{job_id}_matches.png",
                "feature_response_source": f"/outputs/{job_id}_feature_response_source.png",
                "feature_response_reference": f"/outputs/{job_id}_feature_response_reference.png",
                "keypoints_source": f"/outputs/{job_id}_keypoints_source.png",
                "keypoints_reference": f"/outputs/{job_id}_keypoints_reference.png",
            },
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")


@router.get("/demo")
def demo_info():
    return {
        "message": "Upload two lunar images to POST /api/register.",
        "recommended_pairs": [
            "Chandrayaan-2 OHRC <-> LROC NAC",
            "Chandrayaan-2 TMC-2 <-> LROC WAC",
            "Chandrayaan-2 IIRS <-> LROC WAC",
        ],
        "note": "The prototype accepts standard raster images. Native PDS binary products should be converted/read with a mission-specific PDS/ISIS ingestion layer before upload.",
    }


