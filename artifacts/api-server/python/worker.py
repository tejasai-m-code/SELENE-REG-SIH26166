"""CLI bridge used by the TypeScript API to run the preserved OpenCV engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.evaluation import run_synthetic_robustness
from app.services.multi_registration import run_multi_registration
from app.services.pairwise_registration import register_pair, serialize_correspondences, serialize_inlier_points
from app.services.subpixel import normalize_refinement_methods
from app.utils.image_utils import decode_upload, resize_for_preview, save_image


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _infer_sensor(filename: str, supplied: str | None) -> tuple[str, str]:
    explicit = str(supplied or "").strip()
    if explicit and explicit.upper() not in {"UNKNOWN", "UNKNOWN / NOT PROVIDED", "NOT PROVIDED"}:
        return explicit, "USER_SUPPLIED"
    name = filename.upper()
    for token, sensor in (
        ("OHRC", "OHRC"),
        ("TMC-2", "TMC-2"),
        ("TMC2", "TMC-2"),
        ("IIRS", "IIRS"),
        ("LROC-NAC", "LROC NAC"),
        ("LROC_NAC", "LROC NAC"),
        ("LROC-WAC", "LROC WAC"),
        ("SELENE", "SELENE / Kaguya"),
    ):
        if token in name:
            return sensor, "INFERRED_FROM_FILENAME"
    return "UNKNOWN / NOT PROVIDED", "UNKNOWN"


def _metadata(path: Path, image: np.ndarray, supplied: dict[str, Any] | None, sci_raster: Any = None) -> dict:
    supplied = supplied or {}
    sensor, sensor_source = _infer_sensor(path.name, supplied.get("sensor"))
    channels = int(image.shape[2]) if image.ndim == 3 else 1
    depth = int(image.dtype.itemsize * 8)
    from app.services.mission_metadata import parse_metadata_label
    mission_meta = parse_metadata_label(supplied)
    metadata_status = mission_meta.status.value
    provenance = (
        "MEASURED" if sensor_source == "USER_SUPPLIED" and metadata_status == "KNOWN"
        else "ESTIMATED" if sensor_source == "INFERRED_FROM_FILENAME"
        else "REFERENCE_PROFILE" if sensor in ("OHRC", "TMC-2", "IIRS")
        else "UNAVAILABLE"
    )

    base_meta = {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": channels,
        "bit_depth": depth,
        "sensor": sensor,
        "sensor_source": sensor_source,
        "gsd_m": _number(supplied.get("gsd_m")),
        "solar_incidence_deg": _number(supplied.get("solar_incidence_deg")),
        "solar_azimuth_deg": _number(supplied.get("solar_azimuth_deg")),
        "view_incidence_deg": _number(supplied.get("view_incidence_deg")),
        "coordinate_reference": supplied.get("coordinate_reference") or "NOT PROVIDED",
        "geotransform": supplied.get("geotransform") or "NOT PROVIDED",
        "availability": "LOCAL_UPLOAD",
        "radiance_scale": _number(supplied.get("radiance_scale")),
        "radiance_offset": _number(supplied.get("radiance_offset")),
        "metadata_status": metadata_status,
        "provenance": provenance,
    }

    if sci_raster is not None:
        if sensor == "UNKNOWN / NOT PROVIDED":
            base_meta["sensor"] = sci_raster.sensor_id.sensor
            base_meta["sensor_source"] = sci_raster.sensor_id.metadata_status
        else:
            base_meta["sensor"] = sensor
            base_meta["sensor_source"] = sensor_source
        base_meta["mission"] = sci_raster.sensor_id.mission
        base_meta["sensor_confidence_pct"] = sci_raster.sensor_id.confidence_pct
        base_meta["sensor_evidence"] = sci_raster.sensor_id.evidence
        if sci_raster.sensor_id.gsd_m is not None:
            base_meta["gsd_m"] = sci_raster.sensor_id.gsd_m
        base_meta["bit_depth"] = sci_raster.inspection.bit_depth
        base_meta["dtype"] = sci_raster.inspection.dtype
        base_meta["scientific_bands"] = sci_raster.inspection.scientific_bands
        base_meta["layer_classification"] = sci_raster.inspection.layer_classification
        base_meta["dynamic_range"] = [sci_raster.inspection.min_dn, sci_raster.inspection.max_dn]
        base_meta["valid_pixel_pct"] = sci_raster.inspection.valid_pixel_pct
        base_meta["saturation_pct"] = sci_raster.inspection.saturation_pct
        base_meta["scientific_inspection"] = sci_raster.inspection.to_dict()
        base_meta["sensor_identification"] = sci_raster.sensor_id.to_dict()

    return base_meta


def _write_preview(path: Path, image: np.ndarray) -> str:
    save_image(path, resize_for_preview(image))
    return path.name


def _emit_progress(
    job_id: str,
    stage: str,
    stage_index: int,
    stage_count: int,
    progress: float,
    message: str,
    status: str = "RUNNING",
    error: str | None = None,
) -> None:
    import sys
    payload = {
        "job_id": job_id or "",
        "stage": stage,
        "stage_index": stage_index,
        "stage_count": stage_count,
        "progress": round(float(progress), 2),
        "message": message,
        "status": status,
    }
    if error:
        payload["error"] = error
    sys.stderr.write(f"PROGRESS:{json.dumps(payload)}\n")
    sys.stderr.flush()


def register(source_path: Path, reference_path: Path, out_dir: Path, settings: dict, job_id: str | None = None) -> dict:
    job_id = job_id or settings.get("job_id") or ""

    _emit_progress(
        job_id,
        stage="INITIALIZING",
        stage_index=1,
        stage_count=12,
        progress=round(1 / 12, 2),
        message="Initializing registration pipeline and environment",
    )

    _emit_progress(
        job_id,
        stage="LOADING_INPUT",
        stage_index=2,
        stage_count=12,
        progress=round(2 / 12, 2),
        message="Decoding input imagery and extracting metadata",
    )

    from app.services.scientific_data import ingest_scientific_image
    source_bytes = source_path.read_bytes()
    reference_bytes = reference_path.read_bytes()
    source_sci = ingest_scientific_image(source_bytes, source_path.name, settings.get("source_metadata"))
    reference_sci = ingest_scientific_image(reference_bytes, reference_path.name, settings.get("reference_metadata"))
    source = source_sci.display_data
    reference = reference_sci.display_data
    source_metadata = _metadata(source_path, source, settings.get("source_metadata"), sci_raster=source_sci)
    reference_metadata = _metadata(reference_path, reference, settings.get("reference_metadata"), sci_raster=reference_sci)

    raw_methods = settings.get(
        "refinement_methods",
        "taylor,ecc,lucas_kanade,phase,quadratic",
    )
    methods = normalize_refinement_methods(raw_methods)

    def progress_callback(
        stage: str,
        stage_index: int,
        stage_count: int,
        progress: float,
        message: str,
        status: str = "RUNNING",
        error: str | None = None,
    ):
        _emit_progress(job_id, stage, stage_index, stage_count, progress, message, status=status, error=error)

    result = register_pair(
        source,
        reference,
        detector=str(settings.get("detector", "sift")),
        ratio=float(settings.get("ratio", 0.72)),
        ransac_threshold=float(settings.get("ransac_threshold", 3.0)),
        illumination_normalization=bool(settings.get("illumination_normalization", True)),
        spatial_distribution=bool(settings.get("spatial_distribution", True)),
        ecc_refinement="ecc" in {item.lower() for item in methods},
        max_features=int(settings.get("max_features", 8000)),
        refinement_methods=methods,
        source_metadata=source_metadata,
        reference_metadata=reference_metadata,
        radiometric_mode=str(settings.get("radiometric_mode", "safe_normalization")),
        representation=str(settings.get("representation", "structural")),
        progress_callback=progress_callback,
        prefer_affine=bool(settings.get("prefer_affine", False)),
        geometric_model=str(settings.get("geometric_model", "auto")),
        execution_mode=str(settings.get("execution_mode", "hybrid")),
    )

    _emit_progress(
        job_id,
        stage="VISUALIZATION",
        stage_index=11,
        stage_count=12,
        progress=round(11 / 12, 2),
        message="Generating registered imagery and correspondence visualizations",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    public_prefix = str(settings.get("public_prefix", "/api/outputs")).rstrip("/")
    registered = out_dir / "registered.png"
    composite_file = out_dir / "composite.png"
    matches = out_dir / "matches.png"
    if result.registered is not None:
        save_image(registered, resize_for_preview(result.registered))
    if getattr(result, "composite", None) is not None:
        save_image(composite_file, resize_for_preview(result.composite))
    if result.match_visualization is not None:
        save_image(matches, result.match_visualization)

    preprocessing_outputs: dict[str, dict[str, str]] = {}
    from app.services.preprocessing import preprocess_with_diagnostics

    selected_rep_for_previews = result.preprocessing.get("representation", "structural") if result.preprocessing else str(settings.get("representation", "structural"))
    if selected_rep_for_previews == "auto":
        selected_rep_for_previews = "structural"

    for label, image, metadata in (
        ("source", source, source_metadata),
        ("reference", reference, reference_metadata),
    ):
        prepared = preprocess_with_diagnostics(
            image,
            bool(settings.get("illumination_normalization", True)),
            selected_rep_for_previews,
            metadata,
            str(settings.get("radiometric_mode", "safe_normalization")),
        )
        preprocessing_outputs[label] = {
            "selected": _write_preview(out_dir / f"{label}_selected.png", prepared.stages["selected"]),
            "clahe": _write_preview(out_dir / f"{label}_clahe.png", prepared.stages["clahe"]),
            "gradient": _write_preview(out_dir / f"{label}_gradient.png", prepared.stages["gradient"]),
            "retinex": _write_preview(out_dir / f"{label}_retinex.png", prepared.stages["retinex"]),
            "shadow_mask": _write_preview(out_dir / f"{label}_shadow_mask.png", prepared.stages["shadow_mask"]),
        }

    payload = {
        "success": result.success,
        "registration_status": result.registration_status,
        "problem_statement": "SIH26166",
        "pipeline": [
            "image ingestion and metadata inspection",
            result.preprocessing.get("source_radiometric", {}).get("status", "safe normalization"),
            f"{result.preprocessing.get('representation', 'structural')} illumination representation",
            f"{result.detector} feature detection",
            "ratio-test correspondence",
            "spatial distribution constraint" if bool(settings.get("spatial_distribution", True)) else "raw correspondence",
            "RANSAC geometric verification",
            "sub-pixel refinement: " + ", ".join(result.subpixel.get("applied_methods", [])) if result.subpixel else "no sub-pixel method applied",
            "warp to reference",
            "inlier investigation and quality gate",
        ],
        "source": source_metadata,
        "reference": reference_metadata,
        "scientific_data": {
            "source": source_sci.to_summary_dict(),
            "reference": reference_sci.to_summary_dict(),
            "preservation_status": "RAW_DYNAMIC_RANGE_PRESERVED",
            "display_decoupled": True,
        },
        "settings": {
            **{key: value for key, value in settings.items() if key not in {"source_metadata", "reference_metadata"}},
            "refinement_methods": methods,
            "ecc_used": result.ecc_used,
            "ecc_correlation": result.ecc_correlation,
            "raw_match_count": result.raw_match_count,
            "post_distribution_match_count": result.post_distribution_match_count,
            "processing_time_seconds": result.processing_time_seconds,
        },
        "preprocessing": {
            **(result.preprocessing or {}),
            "outputs": preprocessing_outputs,
            "note": "Percentile/CLAHE/Retinex/structural preparation is intensity normalization, not calibration, unless valid radiance metadata is supplied.",
        },
        "metrics": result.metrics,
        "transform_decomposition": (
            (result.inlier_investigation.get("transform_decomposition") if result.inlier_investigation else None)
            or (result.metrics.get("transform_decomposition") if result.metrics else None)
        ),
        "geometric_model": result.geometric_model,
        "estimator_used": result.estimator_used,
        "overlap_validation": {
            "valid": result.overlap_valid,
            "ratio": result.overlap_ratio,
        },
        "gsd_handling": result.gsd_handling,
        "multiscale_diagnostics": result.multiscale_diagnostics,
        "stage_timings": result.stage_timings,
        "inlier_investigation": result.inlier_investigation,
        "subpixel": {
            "requested_methods": result.subpixel.get("requested_methods", []) if result.subpixel else [],
            "applied_methods": result.subpixel.get("applied_methods", []) if result.subpixel else [],
            "method_status": result.subpixel.get("method_status", []) if result.subpixel else [],
            "selected_method": result.subpixel.get("selected_method") if result.subpixel else None,
            "comparison_ledger": result.subpixel.get("comparison_ledger", []) if result.subpixel else [],
            "point_details": result.subpixel.get("point_details", []) if result.subpixel else [],
            "status": "APPLIED" if (result.subpixel and result.subpixel.get("applied_methods")) else "NOT_APPLIED",
            "refinement_applied": bool(result.subpixel and result.subpixel.get("applied_methods")),
            "phase_response": result.subpixel.get("phase_response") if result.subpixel else None,
            "subpixel_validation_status": result.subpixel.get("subpixel_validation_status", "SUBPIXEL_NOT_VALIDATED") if result.subpixel else "SUBPIXEL_NOT_VALIDATED",
            "subpixel_validation_reason": result.subpixel.get("subpixel_validation_reason", "") if result.subpixel else "",
            "raw_rmse_pixels": result.raw_rmse,
            "refined_rmse_pixels": result.refined_rmse,
            "rmse_improvement_pixels": result.rmse_improvement,
            "refined_point_count": int(len(result.subpixel.get("refined_reference_points", []))) if result.subpixel else 0,
            "mean_delta_pixels": (
                np.mean(np.linalg.norm(result.subpixel["delta_xy"], axis=1), dtype=float).item()
                if result.subpixel is not None and len(result.subpixel.get("delta_xy", []))
                else 0.0
            ),
        },
        "homography": result.homography.tolist(),
        "inlier_points": serialize_inlier_points(result),
        "correspondences": serialize_correspondences(result),
        "footprint": result.footprint,
        "outputs": {
            "registered_image": f"{public_prefix}/{registered.name}" if result.registered is not None else None,
            "composite_image": f"{public_prefix}/{composite_file.name}" if getattr(result, "composite", None) is not None else None,
            "match_visualization": f"{public_prefix}/{matches.name}" if result.match_visualization is not None else None,
            "composite_metadata": getattr(result, "composite_metadata", None),
            "preprocessing": {
                side: {name: f"{public_prefix}/{filename}" for name, filename in files.items()}
                for side, files in preprocessing_outputs.items()
            },
        },
    }

    _emit_progress(
        job_id,
        stage="COMPLETE",
        stage_index=12,
        stage_count=12,
        progress=1.0,
        message="Scientific registration pipeline complete",
        status="COMPLETE",
    )

    return payload


def register_multi(image_paths: list[Path], out_dir: Path, settings: dict, job_id: str | None = None) -> dict:
    job_id = job_id or settings.get("job_id") or ""

    def progress_callback(
        stage: str,
        stage_index: int,
        stage_count: int,
        progress: float,
        message: str,
        status: str = "RUNNING",
        error: str | None = None,
    ):
        _emit_progress(job_id, stage, stage_index, stage_count, progress, message, status=status, error=error)

    from app.services.scientific_data import ingest_scientific_image
    sci_rasters = [ingest_scientific_image(path.read_bytes(), path.name) for path in image_paths]
    images = [s.display_data for s in sci_rasters]
    metadata = []
    supplied = settings.get("multi_metadata") or []
    for index, (path, image, sci) in enumerate(zip(image_paths, images, sci_rasters)):
        item = supplied[index] if index < len(supplied) and isinstance(supplied[index], dict) else {}
        item = {**item, "sensor": item.get("sensor", settings.get("sensor", "UNKNOWN"))}
        record = _metadata(path, image, item, sci_raster=sci)
        record.update({
            "index": index,
            "label": str(item.get("label") or path.name),
            "modality": str(item.get("modality") or sci.sensor_id.mission),
            "thumbnail": None,
        })
        metadata.append(record)
    result = run_multi_registration(
        images,
        {
            "detector": str(settings.get("detector", "sift")),
            "ratio": float(settings.get("ratio", 0.72)),
            "ransac_threshold": float(settings.get("ransac_threshold", 3.0)),
            "illumination_normalization": bool(settings.get("illumination_normalization", True)),
            "spatial_distribution": bool(settings.get("spatial_distribution", True)),
            "ecc_refinement": bool(settings.get("ecc_refinement", True)),
            "max_features": int(settings.get("max_features", 8000)),
        },
        progress_callback=progress_callback,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    public_prefix = str(settings.get("public_prefix", "/api/outputs")).rstrip("/")
    outputs: dict[str, str] = {}
    if result.mosaic is not None:
        mosaic_path = out_dir / "relative_mosaic.png"
        save_image(mosaic_path, result.mosaic)
        outputs["mosaic"] = f"{public_prefix}/{mosaic_path.name}"
    pairs: dict[str, dict] = {}
    for key, pair in result.pair_outputs.items():
        match_path = out_dir / f"pair_{key}.png"
        save_image(match_path, resize_for_preview(pair["match_visualization"]))
        pairs[key] = {
            "source_index": pair.get("source_index"),
            "reference_index": pair.get("reference_index"),
            "metrics": pair["metrics"],
            "inlier_points": pair["inlier_points"],
            "correspondences": pair.get("correspondences", []),
            "homography": pair.get("homography"),
            "geometric_model": pair.get("geometric_model", "homography"),
            "estimator_used": pair.get("estimator_used", "USAC_MAGSAC"),
            "registration_status": pair.get("registration_status", "PASS"),
            "representation": pair.get("representation", "structural"),
            "cache_hit": pair["cache_hit"],
            "match_visualization": f"{public_prefix}/{match_path.name}",
        }
    return {
        "success": True,
        "job_id": job_id,
        "status": "COMPLETED" if result.mosaic is not None else "PARTIALLY_COMPLETED",
        "map_type": "Relative Registered Lunar Mosaic",
        "images": metadata,
        "summary": result.summary,
        "graph": result.graph,
        "placement": result.placement,
        "mosaic": result.mosaic_info,
        "outputs": outputs,
        "pairs": pairs,
        "scientific_note": "Placement is relative image-space registration. Mission control points and geographic validation are required for georeferencing.",
    }


def load_settings(settings_json: str | None = None, settings_file: str | Path | None = None) -> dict:
    """Load settings dictionary from either a JSON file or a JSON string.

    If both are supplied, settings_file takes precedence.
    """
    if settings_file:
        if settings_json:
            import sys
            print("Warning: Both --settings-file and --settings-json supplied; --settings-file takes precedence.", file=sys.stderr)
        settings_text = Path(settings_file).read_text(encoding="utf-8")
        return json.loads(settings_text)
    if settings_json:
        return json.loads(settings_json)
    raise ValueError("Either --settings-json or --settings-file must be provided.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pair", "multi", "robustness", "ingest-url", "validation-benchmark", "scan-dataset", "hardware-status", "gpu-self-test", "inspect-image", "list-detectors"), default="pair")
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument("--images", nargs="*")
    parser.add_argument("--url")
    parser.add_argument("--folder-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--public-prefix", default="/api/outputs")
    parser.add_argument("--settings-json", default=None)
    parser.add_argument("--settings-file", default=None)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    if args.mode == "hardware-status":
        from app.services.hardware_manager import get_hardware_status
        from app.services.hybrid_scheduler import get_global_scheduler
        status = get_hardware_status()
        status["hybrid_scheduler"] = get_global_scheduler().get_telemetry()
        print(json.dumps(status, default=str))
        return

    if args.mode == "gpu-self-test":
        from app.services.gpu_accelerator import run_gpu_self_test
        print(json.dumps(run_gpu_self_test(), default=str))
        return

    if args.mode == "inspect-image":
        from app.services.scientific_data import ingest_scientific_image
        if not args.source:
            raise ValueError("--source is required for inspect-image mode.")
        img_path = Path(args.source)
        sci_raster = ingest_scientific_image(img_path.read_bytes(), img_path.name)
        out = sci_raster.to_summary_dict()
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            preview_file = out_dir / f"preview_{img_path.stem}.jpg"
            save_image(preview_file, resize_for_preview(sci_raster.display_data))
            out["preview_url"] = f"{args.public_prefix}/{preview_file.name}"
        print(json.dumps(out, default=str))
        return

    if args.mode == "list-detectors":
        from app.services.feature_matching import get_detector_capabilities
        print(json.dumps(get_detector_capabilities(), default=str))
        return

    if args.mode == "scan-dataset":
        from app.services.dataset_scanner import DatasetScanner
        if not args.folder_path:
            raise ValueError("--folder-path is required for scan-dataset mode.")
        scanner = DatasetScanner(output_dir=Path(args.out_dir), public_prefix=args.public_prefix)
        manifest = scanner.scan_directory(args.folder_path, dataset_name=args.dataset_name)
        print(json.dumps(manifest, default=str))
        return

    if args.mode == "ingest-url":
        from app.services.url_ingestion import ingest_image_from_url
        if not args.url:
            raise ValueError("--url is required for ingest-url mode.")
        result = ingest_image_from_url(
            url_str=args.url,
            out_dir=Path(args.out_dir),
            public_prefix=args.public_prefix,
        )
        print(json.dumps(result.to_dict(), default=str))
        if not result.success:
            import sys
            sys.exit(1)
        return

    if args.mode == "validation-benchmark":
        from app.services.evaluation import run_validation_benchmark
        job_id = args.job_id or ""
        settings = {}
        if args.settings_json or args.settings_file:
            settings = load_settings(args.settings_json, args.settings_file)
        def progress_cb(stage, stage_index, stage_count, progress, message, status="RUNNING", error=None):
            _emit_progress(job_id, stage, stage_index, stage_count, progress, message, status=status, error=error)
        res = run_validation_benchmark(settings, progress_callback=progress_cb)
        print(json.dumps(res, default=str))
        return

    settings = load_settings(args.settings_json, args.settings_file)
    if args.mode == "robustness":
        image_path = Path(args.source)
        image = decode_upload(image_path.read_bytes(), image_path.name)
        print(json.dumps(run_synthetic_robustness(image, settings), default=str))
        return
    if args.mode == "multi":
        if not args.images or len(args.images) < 2:
            raise ValueError("At least two images are required for multi-image registration.")
        settings["public_prefix"] = args.public_prefix
        print(json.dumps(register_multi([Path(item) for item in args.images], Path(args.out_dir), settings, job_id=args.job_id), default=str))
        return
    if not args.source or not args.reference:
        raise ValueError("Both --source and --reference are required for pair mode.")
    settings["public_prefix"] = args.public_prefix
    payload = register(Path(args.source), Path(args.reference), Path(args.out_dir), settings, job_id=args.job_id)
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()