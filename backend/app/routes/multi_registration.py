"""FastAPI routes for the synchronous local multi-image map-builder prototype."""

import csv
import json
from io import StringIO
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.services.multi_registration import run_multi_registration
from app.utils.image_utils import decode_upload, resize_for_preview, save_image
from app.services.radiometry import process_raster, RadiometricConfig

router = APIRouter(tags=["multi-registration"])
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MULTI_JOBS: dict[str, dict] = {}


def _metadata_for(files: list[UploadFile], metadata_json: str) -> list[dict]:
    try:
        supplied = json.loads(metadata_json or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("Image metadata must be valid JSON.") from exc
    if not isinstance(supplied, list):
        raise ValueError("Image metadata must be a JSON list when provided.")
    result = []
    for index, file in enumerate(files):
        item = supplied[index] if index < len(supplied) and isinstance(supplied[index], dict) else {}
        result.append({"filename": file.filename or f"Image {index + 1}", "label": str(item.get("label") or file.filename or f"Image {index + 1}"), "sensor": str(item.get("sensor") or "Unknown / Not provided"), "modality": str(item.get("modality") or "Unknown / Not provided"), "resolution": str(item.get("resolution") or "Unknown / Not provided"), "acquisition": str(item.get("acquisition") or "Unknown / Not provided"), "illumination": str(item.get("illumination") or "Unknown / Not provided")})
    return result


@router.post("/multi-registration/register")
async def register_image_collection(images: list[UploadFile] = File(...), metadata_json: str = Form("[]"), detector: str = Form("sift"), ratio: float = Form(0.72), ransac_threshold: float = Form(3.0), illumination_normalization: bool = Form(True), spatial_distribution: bool = Form(True), ecc_refinement: bool = Form(True), max_features: int = Form(8000)):
    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two images.")
    if len(images) > 12:
        raise HTTPException(status_code=400, detail="This local prototype accepts up to 12 images per map job.")
    job_id = uuid4().hex[:12]
    try:
        metadata, decoded = _metadata_for(images, metadata_json), []
        for index, file in enumerate(images):
            try:
                data_bytes = await file.read()
                if not data_bytes: raise ValueError("Empty file.")
                image_obj = decode_upload(data_bytes, file.filename or "")
                image_res = process_raster(image_obj, RadiometricConfig(enable_illumination_normalization=illumination_normalization))
                image = image_res.processing_image
                if image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError(f"{metadata[index]['filename']} is too large for this local prototype.")
                metadata[index].update({"index": index, "width": int(image.shape[1]), "height": int(image.shape[0]), "format": Path(file.filename or "").suffix.lower().lstrip(".") or "Unknown", "status": "ACCEPTED", "reason": ""})
                thumbnail_path = OUTPUT_DIR / f"{job_id}_thumbnail_{index}.png"
                save_image(thumbnail_path, resize_for_preview(image_res.display_image, 360))
                metadata[index]["thumbnail"] = f"/outputs/{thumbnail_path.name}"
                decoded.append(image)
            except Exception as exc:
                metadata[index].update({"index": index, "width": 0, "height": 0, "format": Path(file.filename or "").suffix.lower().lstrip(".") or "Unknown", "status": "REJECTED", "reason": str(exc), "thumbnail": None})
                decoded.append(None)
        result = run_multi_registration(decoded, {"detector": detector, "ratio": ratio, "ransac_threshold": ransac_threshold, "illumination_normalization": illumination_normalization, "spatial_distribution": spatial_distribution, "ecc_refinement": ecc_refinement, "max_features": max_features})
        pairs = {}
        for key, value in result.pair_outputs.items():
            match_path = OUTPUT_DIR / f"{job_id}_pair_{key}.png"
            save_image(match_path, resize_for_preview(value["match_visualization"]))
            pairs[key] = {"metrics": value["metrics"], "inlier_points": value["inlier_points"], "match_visualization": f"/outputs/{match_path.name}"}
        outputs = {}
        if result.mosaic is not None:
            mosaic_path = OUTPUT_DIR / f"{job_id}_relative_mosaic.png"
            save_image(mosaic_path, result.mosaic)
            outputs["mosaic"] = f"/outputs/{mosaic_path.name}"
        public = {"success": True, "job_id": job_id, "status": "COMPLETED" if result.mosaic is not None else "PARTIALLY COMPLETED", "map_type": "Relative Registered Lunar Mosaic", "images": metadata, "summary": result.summary, "graph": result.graph, "placement": result.placement, "mosaic": result.mosaic_info, "outputs": outputs, "pairs": pairs, "global_graph": result.global_graph, "p10_mosaic": result.p10_mosaic}
        MULTI_JOBS[job_id] = public
        return public
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="The multi-image registration job could not be completed.")


def _job_or_404(job_id: str) -> dict:
    if job_id not in MULTI_JOBS:
        raise HTTPException(status_code=404, detail="Map-builder job was not found in this local server session.")
    return MULTI_JOBS[job_id]


@router.get("/multi-registration/{job_id}/status")
def multi_status(job_id: str):
    job = _job_or_404(job_id)
    return {"job_id": job_id, "status": job["status"], "summary": job["summary"]}


@router.get("/multi-registration/{job_id}/graph")
def multi_graph(job_id: str):
    job = _job_or_404(job_id)
    return {"job_id": job_id, "graph": job["graph"], "placement": job["placement"]}


@router.get("/multi-registration/{job_id}/results")
def multi_results(job_id: str):
    return _job_or_404(job_id)


@router.get("/multi-registration/{job_id}/pair/{image_a}/{image_b}")
def multi_pair(job_id: str, image_a: int, image_b: int):
    pairs = _job_or_404(job_id)["pairs"]
    pair = pairs.get(f"{image_a}-{image_b}") or pairs.get(f"{image_b}-{image_a}")
    if pair is None:
        raise HTTPException(status_code=404, detail="No completed correspondence visualization exists for this pair.")
    return pair


@router.get("/multi-registration/{job_id}/export/graph.json")
def export_graph(job_id: str):
    job = _job_or_404(job_id)
    return {"job_id": job_id, "map_type": job["map_type"], "graph": job["graph"], "placement": job["placement"]}



@router.get("/multi-registration/{job_id}/export/report.json")
def export_report(job_id: str):
    job = _job_or_404(job_id)
    return {
        "product": "SELENE-REG",
        "problem_statement": "SIH26166",
        "job_id": job_id,
        "map_type": job["map_type"],
        "scientific_scope": job["summary"].get("scientific_note"),
        "images": job["images"],
        "summary": job["summary"],
        "graph": job["graph"],
        "placement": job["placement"],
        "mosaic": job["mosaic"],
        "pairs": {
            key: {
                "metrics": value["metrics"],
                "inlier_points": value["inlier_points"],
            }
            for key, value in job["pairs"].items()
        },
    }

@router.get("/multi-registration/{job_id}/export/metrics.csv", response_class=PlainTextResponse)
def export_metrics_csv(job_id: str):
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["image_a", "image_b", "status", "reason", "inlier_count", "inlier_ratio", "rmse_pixels", "source_spatial_coverage"])
    for edge in _job_or_404(job_id)["graph"]["edges"]:
        metrics = edge.get("metrics", {})
        writer.writerow([edge["image_a"], edge["image_b"], edge["status"], edge.get("reason") or "", metrics.get("inlier_count", ""), metrics.get("inlier_ratio", ""), metrics.get("rmse_pixels", ""), metrics.get("source_spatial_coverage", "")])
    return stream.getvalue()




@router.get('/dataset-inventory')
def dataset_inventory(root_path: str):
    import os
    if not os.path.exists(root_path) or not os.path.isdir(root_path):
        raise HTTPException(status_code=400, detail='Invalid root path provided.')
    
    from app.services.dataset_inventory import DatasetInventory
    inv = DatasetInventory(root_path)
    inv.scan()
    return {
        'summary': inv.get_summary(),
        'products': inv.products
    }

