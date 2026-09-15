from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.services.preprocessing import preprocess
from app.services.feature_matching import match_features, spatially_distribute_matches
from app.services.registration import (
    estimate_homography,
    refine_with_ecc,
    warp_to_reference,
    draw_matches,
)
from app.services.metrics import compute_metrics
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

        src_gray = preprocess(src_img, illumination_normalization)
        ref_gray = preprocess(ref_img, illumination_normalization)

        matches = match_features(
            src_gray,
            ref_gray,
            detector_name=detector,
            ratio=max(0.55, min(float(ratio), 0.95)),
            max_features=max(500, min(int(max_features), 20000)),
        )

        raw_match_count = len(matches.good_matches)

        if spatial_distribution:
            matches = spatially_distribute_matches(matches, src_gray.shape)

        H, inlier_mask = estimate_homography(
            matches.source_points,
            matches.reference_points,
            ransac_threshold=max(0.5, min(float(ransac_threshold), 20.0)),
        )

        ecc_score = None
        ecc_used = False
        if ecc_refinement:
            H2, ecc_score, ecc_used = refine_with_ecc(
                src_gray, ref_gray, H, iterations=100
            )
            if ecc_used:
                H = H2
                # Recompute RANSAC inliers against refined H.
                pred = cv2.perspectiveTransform(
                    matches.source_points, H
                ).reshape(-1, 2)
                target = matches.reference_points.reshape(-1, 2)
                err = np.linalg.norm(pred - target, axis=1)
                inlier_mask = err <= max(0.5, float(ransac_threshold))

        registered = warp_to_reference(src_img, ref_img.shape, H)

        match_vis = draw_matches(
            src_gray,
            ref_gray,
            matches.source_points,
            matches.reference_points,
            inlier_mask,
        )

        metrics = compute_metrics(
            matches.source_points,
            matches.reference_points,
            H,
            inlier_mask,
            src_gray.shape,
            ref_gray.shape,
        )

        # Keep original source/reference copies for traceability.
        save_image(UPLOAD_DIR / f"{job_id}_source.png", resize_for_preview(src_img))
        save_image(UPLOAD_DIR / f"{job_id}_reference.png", resize_for_preview(ref_img))

        reg_path = OUTPUT_DIR / f"{job_id}_registered.png"
        match_path = OUTPUT_DIR / f"{job_id}_matches.png"
        save_image(reg_path, resize_for_preview(registered))
        save_image(match_path, resize_for_preview(match_vis))

        inlier_points = []
        src_pts = matches.source_points.reshape(-1, 2)
        ref_pts = matches.reference_points.reshape(-1, 2)
        for i, ok in enumerate(inlier_mask):
            if ok:
                inlier_points.append({
                    "source": [round(float(src_pts[i, 0]), 4), round(float(src_pts[i, 1]), 4)],
                    "reference": [round(float(ref_pts[i, 0]), 4), round(float(ref_pts[i, 1]), 4)],
                })

        return {
            "success": True,
            "job_id": job_id,
            "problem_statement": "SIH26166",
            "pipeline": [
                "image ingestion",
                "grayscale + illumination normalization",
                f"{detector.upper()} feature detection",
                "ratio-test correspondence",
                "spatial distribution constraint" if spatial_distribution else "raw correspondence",
                "RANSAC geometric verification",
                "ECC intensity refinement" if ecc_used else "geometric registration",
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
                "detector": detector.upper(),
                "ratio": float(ratio),
                "ransac_threshold": float(ransac_threshold),
                "illumination_normalization": illumination_normalization,
                "spatial_distribution": spatial_distribution,
                "ecc_refinement": ecc_refinement,
                "ecc_used": ecc_used,
                "ecc_correlation": ecc_score,
                "raw_match_count": raw_match_count,
                "post_distribution_match_count": len(matches.good_matches),
            },
            "metrics": metrics,
            "homography": H.tolist(),
            "inlier_points": inlier_points,
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
