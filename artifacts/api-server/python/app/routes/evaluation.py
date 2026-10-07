"""Synthetic robustness evaluation endpoint.

The endpoint is intentionally separate from the lunar-data registration path
so demo users cannot confuse controlled stress tests with mission validation.
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.evaluation import run_synthetic_robustness
from app.utils.image_utils import decode_upload

router = APIRouter(tags=["evaluation"])


@router.post("/evaluation/synthetic-robustness")
async def synthetic_robustness(
    image: UploadFile = File(...),
    detector: str = Form("sift"),
    ratio: float = Form(0.72),
    ransac_threshold: float = Form(3.0),
    illumination_normalization: bool = Form(True),
    spatial_distribution: bool = Form(True),
    ecc_refinement: bool = Form(True),
    max_features: int = Form(8000),
):
    try:
        data = await image.read()
        raster = decode_upload(data, image.filename or "")
        if raster.shape[0] * raster.shape[1] > 25_000_000:
            raise ValueError("Evaluation image exceeds the 25 MP local stress-test limit.")
        return run_synthetic_robustness(
            raster,
            {
                "detector": detector,
                "ratio": ratio,
                "ransac_threshold": ransac_threshold,
                "illumination_normalization": illumination_normalization,
                "spatial_distribution": spatial_distribution,
                "ecc_refinement": ecc_refinement,
                "max_features": max_features,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthetic evaluation failed: {exc}")
