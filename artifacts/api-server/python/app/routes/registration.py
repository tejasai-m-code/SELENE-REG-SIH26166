from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.services.pairwise_registration import register_pair, serialize_inlier_points
from app.utils.image_utils import decode_upload, save_image, resize_for_preview

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
):
    job_id = uuid4().hex[:12]
    try:
        source_bytes = await source.read()
        reference_bytes = await reference.read()

        src_img = decode_upload(source_bytes, source.filename or "")
        ref_img = decode_upload(reference_bytes, reference.filename or "")

        result = register_pair(
            src_img, ref_img, detector, ratio, ransac_threshold,
            illumination_normalization, spatial_distribution, ecc_refinement, max_features,
        )

        # Keep original source/reference copies for traceability.
        save_image(UPLOAD_DIR / f"{job_id}_source.png", resize_for_preview(src_img))
        save_image(UPLOAD_DIR / f"{job_id}_reference.png", resize_for_preview(ref_img))

        reg_path = OUTPUT_DIR / f"{job_id}_registered.png"
        match_path = OUTPUT_DIR / f"{job_id}_matches.png"
        save_image(reg_path, resize_for_preview(result.registered))
        save_image(match_path, resize_for_preview(result.match_visualization))

        return {
            "success": True,
            "job_id": job_id,
            "problem_statement": "SIH26166",
            "pipeline": [
                "image ingestion",
                "grayscale + illumination normalization",
                f"{result.detector} feature detection",
                "ratio-test correspondence",
                "spatial distribution constraint" if spatial_distribution else "raw correspondence",
                "RANSAC geometric verification",
                "ECC intensity refinement" if result.ecc_used else "geometric registration",
                "warp to reference",
                "RMSE / inlier / coverage evaluation",
            ],
            "source": {
                "filename": source.filename,
                "width": int(src_img.shape[1]),
                "height": int(src_img.shape[0]),
            },
            "reference": {
                "filename": reference.filename,
                "width": int(ref_img.shape[1]),
                "height": int(ref_img.shape[0]),
            },
            "settings": {
                "detector": result.detector,
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
            "metrics": result.metrics,
            "homography": result.homography.tolist(),
            "inlier_points": serialize_inlier_points(result),
            "outputs": {
                "registered_image": f"/outputs/{job_id}_registered.png",
                "match_visualization": f"/outputs/{job_id}_matches.png",
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
            "Chandrayaan-2 OHRC ↔ LROC NAC",
            "Chandrayaan-2 TMC-2 ↔ LROC WAC",
            "Chandrayaan-2 IIRS ↔ LROC WAC",
        ],
        "note": "The prototype accepts standard raster images. Native PDS binary products should be converted/read with a mission-specific PDS/ISIS ingestion layer before upload.",
    }
