"""Cached, scientifically validated multi-image registration and mosaic orchestration."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Callable
import cv2
import numpy as np

from app.services.diagnostics import processing_device_diagnostics
from app.services.mosaic import build_relative_mosaic, MosaicCanvasTooLargeError
from app.services.pairwise_registration import PairwiseRegistrationResult, register_pair, serialize_inlier_points
from app.services.registration import draw_matches
from app.services.registration_cache import RegistrationCache
from app.services.registration_graph import (
    build_graph,
    confidence_for_metrics,
    place_largest_component,
    select_reference_node,
    evaluate_cycle_consistency,
)
from app.utils.image_utils import resize_for_preview, to_gray

DEFAULT_ACCEPTANCE = {
    "min_inliers": 6,
    "min_inlier_ratio": 0.10,
    "max_rmse_pixels": 8.0,
    "min_spatial_coverage": 0.125,
}

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


@dataclass
class MultiRegistrationResult:
    graph: dict
    placement: dict
    mosaic: np.ndarray | None
    mosaic_info: dict | None
    pair_outputs: dict
    summary: dict


def thumbnail_signature(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        cv2.resize(resize_for_preview(to_gray(image), 360), (64, 64), interpolation=cv2.INTER_AREA),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def select_candidate_pairs(
    images: list[np.ndarray],
    max_candidates: int = 66,
    max_neighbors_per_image: int = 6,
) -> list[dict]:
    n = len(images)
    sigs = [thumbnail_signature(x) for x in images]
    hists = [cv2.calcHist([s], [0], None, [32], [0, 256]) for s in sigs]

    if n <= 12:
        pairs = []
        for a, b in combinations(range(n), 2):
            score = float(cv2.compareHist(hists[a], hists[b], cv2.HISTCMP_CORREL))
            pairs.append({"image_a": a, "image_b": b, "screening_score": score})
        return pairs

    # For N > 12: scalable candidate discovery
    # Scale candidate budget so large batches (50, 100, 200) can form connected graphs
    candidate_budget = max(max_candidates, n * max_neighbors_per_image)
    selected_pairs: dict[tuple[int, int], float] = {}

    # 1. Temporal / spatial sequential adjacency (critical for orbital strip swaths)
    for i in range(n - 1):
        score_1 = float(cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_CORREL))
        selected_pairs[(i, i + 1)] = score_1
        if i + 2 < n:
            score_2 = float(cv2.compareHist(hists[i], hists[i + 2], cv2.HISTCMP_CORREL))
            selected_pairs[(i, i + 2)] = score_2

    # 2. Top-K nearest neighbors per image based on thumbnail correlation
    for a in range(n):
        scores_for_a = []
        for b in range(n):
            if a == b:
                continue
            pair_key = (min(a, b), max(a, b))
            if pair_key in selected_pairs:
                score = selected_pairs[pair_key]
            else:
                score = float(cv2.compareHist(hists[a], hists[b], cv2.HISTCMP_CORREL))
            scores_for_a.append((score, pair_key))

        scores_for_a.sort(key=lambda x: x[0], reverse=True)
        for score, pair_key in scores_for_a[:max_neighbors_per_image]:
            selected_pairs[pair_key] = score

    pair_list = [
        {"image_a": a, "image_b": b, "screening_score": score}
        for (a, b), score in selected_pairs.items()
    ]
    pair_list.sort(key=lambda x: x["screening_score"], reverse=True)
    return pair_list[:candidate_budget]



def _rejection_reason(m: dict, c: dict) -> str | None:
    if m.get("inlier_count", 0) < c.get("min_inliers", 6):
        return "insufficient reliable correspondences"
    if m.get("inlier_ratio", 0.0) < c.get("min_inlier_ratio", 0.10):
        return "inlier ratio below configured prototype threshold"
    if m.get("rmse_pixels") is None or m.get("rmse_pixels") > c.get("max_rmse_pixels", 8.0):
        return "reprojection error exceeds configured prototype threshold"
    if m.get("source_spatial_coverage", 0.0) < c.get("min_spatial_coverage", 0.125):
        return "insufficient spatial distribution of inlier matches"
    if m.get("spatial_distribution_category") == "CLUSTERED_RISKY_DISTRIBUTION":
        # Clustered matches in a small region cannot support reliable registration regardless of count
        return "correspondences severely clustered in small region"
    return None


def _payload(result: PairwiseRegistrationResult) -> dict:
    return {
        "homography": result.homography.tolist() if isinstance(result.homography, np.ndarray) else result.homography,
        "metrics": result.metrics,
        "source_points": result.source_points,
        "reference_points": result.reference_points,
        "final_reference_points": result.final_reference_points.tolist() if isinstance(result.final_reference_points, np.ndarray) else result.final_reference_points,
        "inlier_mask": result.inlier_mask,
        "raw_match_count": result.raw_match_count,
        "post_distribution_match_count": result.post_distribution_match_count,
        "ecc_used": result.ecc_used,
        "ecc_correlation": result.ecc_correlation,
        "detector": result.detector,
        "registration_status": result.registration_status,
        "geometric_model": result.geometric_model,
        "estimator_used": result.estimator_used,
        "overlap_valid": result.overlap_valid,
        "overlap_ratio": result.overlap_ratio,
        "processing_time_seconds": result.processing_time_seconds,
    }


def _cached(payload: dict, source_gray: np.ndarray, reference_gray: np.ndarray) -> PairwiseRegistrationResult:
    a = np.asarray(payload["source_points"], dtype=np.float32)
    b = np.asarray(payload["reference_points"], dtype=np.float32)
    mask = np.asarray(payload["inlier_mask"], dtype=bool)

    raw_final = payload.get("final_reference_points")
    if raw_final is not None and not isinstance(raw_final, str):
        try:
            final_ref = np.asarray(raw_final, dtype=np.float32)
        except (ValueError, TypeError):
            final_ref = b
    else:
        final_ref = b

    homo = np.asarray(payload.get("homography", np.eye(3)), dtype=float)

    try:
        vis = draw_matches(source_gray, reference_gray, a, final_ref, mask)
    except Exception:
        vis = np.zeros_like(source_gray)

    return PairwiseRegistrationResult(
        homography=homo,
        inlier_mask=mask,
        metrics=payload["metrics"],
        registered=None,
        match_visualization=vis,
        source_points=a,
        reference_points=b,
        final_reference_points=final_ref,
        raw_match_count=payload.get("raw_match_count", len(a)),
        post_distribution_match_count=payload.get("post_distribution_match_count", len(a)),
        ecc_used=payload.get("ecc_used", False),
        ecc_correlation=payload.get("ecc_correlation"),
        detector=payload.get("detector", "SIFT"),
        registration_status=payload.get("registration_status", "SUCCESS"),
        success=payload.get("registration_status") == "SUCCESS",
        geometric_model=payload.get("geometric_model", "homography"),
        estimator_used=payload.get("estimator_used", "USAC_MAGSAC"),
        overlap_valid=payload.get("overlap_valid", True),
        overlap_ratio=payload.get("overlap_ratio", 1.0),
        processing_time_seconds=payload.get("processing_time_seconds", 0.0),
    )


def _summary(edges: list[dict], images: int, graph: dict, placement: dict, mosaic_info: dict | None, cache, started: float, possible: int, candidates: int, stages: dict) -> dict:
    accepted = [e for e in edges if e.get("accepted", False)]
    ms = [e["metrics"] for e in accepted]
    rmses = [m["rmse_pixels"] for m in ms if m.get("rmse_pixels") is not None]
    covs = [m.get("source_spatial_coverage", 0.0) for m in ms]
    uniformities = [m.get("spatial_uniformity", 0.0) for m in ms]

    mean = lambda xs: float(np.mean(xs)) if xs else None
    placed_count = len(placement.get("transforms", {}))
    unplaced_count = images - placed_count

    cycle_diag = placement.get("cycle_diagnostics", {})

    return {
        "images_total": images,
        "images_loaded": images,
        "image_count": images,
        "multi_image_count": images,
        "images_registered": placed_count,
        "images_accepted": placed_count,
        "images_placed": placed_count,
        "images_unplaced": unplaced_count,
        "images_rejected": unplaced_count,
        "images_disconnected": len(graph.get("isolated_nodes", [])),
        "final_mosaic_contributors": placed_count,
        "placed_image_count": placed_count,
        "unregistered_images": placement.get("unplaced_nodes", []),
        "isolated_images": graph.get("isolated_nodes", []),
        "isolated_image_count": len(graph.get("isolated_nodes", [])),
        "total_possible_pairs": possible,
        "candidate_pairs": candidates,
        "candidate_pair_count": candidates,
        "rejected_by_coarse_screening": max(0, possible - candidates),
        "connected_ratio_display": f"{placed_count} / {images} images successfully connected",
        "rejection_details": placement.get("rejection_details", []),
        "global_optimization": placement.get("global_optimization", {}),
        "number_of_edges": len(edges),
        "successful_edges": len(accepted),
        "rejected_edges": len(edges) - len(accepted),
        "accepted_pairs": len(accepted),
        "rejected_pairs": len(edges) - len(accepted),
        "successful_pair_count": len(accepted),
        "failed_pair_count": len(edges) - len(accepted),
        "processed_pairs": sum(not e.get("cache_hit", False) for e in edges),
        "reused_pairs": sum(bool(e.get("cache_hit", False)) for e in edges),
        "graph_component_count": len(graph.get("components", [])),
        "connected_components": len(graph.get("components", [])),
        "largest_component_size": graph.get("largest_component_size", 0),
        "graph_connected": graph.get("connected", False),
        "is_fully_connected": graph.get("connected", False),
        "selected_reference": placement.get("root"),
        "selected_reference_node": placement.get("root"),
        "reference_selection_rationale": placement.get("reference_selection", {}).get("rationale"),
        "cycle_count": cycle_diag.get("cycle_count", 0),
        "cycle_consistency_error": cycle_diag.get("mean_cycle_error_pixels"),
        "cycle_consistency_error_max": cycle_diag.get("max_cycle_error_pixels"),
        "graph_cycle_consistent": cycle_diag.get("graph_cycle_consistent", True),
        "average_inliers": mean([m.get("inlier_count", 0) for m in ms]),
        "average_inliers_per_accepted_pair": mean([m.get("inlier_count", 0) for m in ms]),
        "average_inlier_ratio": mean([m.get("inlier_ratio", 0.0) for m in ms]),
        "spatial_coverage_mean": mean(covs),
        "average_source_spatial_coverage": mean(covs),
        "spatial_uniformity_mean": mean(uniformities),
        "rmse_mean": mean(rmses),
        "rmse_median": float(np.median(rmses)) if rmses else None,
        "rmse_min": float(np.min(rmses)) if rmses else None,
        "rmse_max": float(np.max(rmses)) if rmses else None,
        "rmse_statistics_pixels": {
            "mean": mean(rmses),
            "median": float(np.median(rmses)) if rmses else None,
            "minimum": float(np.min(rmses)) if rmses else None,
            "maximum": float(np.max(rmses)) if rmses else None,
        },
        "mosaic_coverage": mosaic_info.get("coverage_fraction") if mosaic_info else None,
        "mosaic_quality": mosaic_info.get("quality_label") if mosaic_info else None,
        "mosaic_overlap_fraction": mosaic_info.get("overlap_fraction") if mosaic_info else None,
        "mosaic_psnr_db": mosaic_info.get("overlap_psnr_db") if mosaic_info else None,
        "cache_hits": cache.stats["pair_hits"],
        "cache_misses": cache.stats["pair_misses"],
        "cache": cache.stats,
        "processing_time_seconds": round(perf_counter() - started, 4),
        "stage_timings_seconds": stages,
        "processing_diagnostics": processing_device_diagnostics(),
        "processing_device": processing_device_diagnostics()["device"],
        "scientific_note": "All placement is relative image-space registration. Target SIH dataset validation and geographic control are required for scientific georeferencing.",
    }


def run_multi_registration(
    images: list[np.ndarray],
    settings: dict | None = None,
    cache_dir: Path | None = None,
    metadata_list: list[dict] | None = None,
    progress_callback: Callable | None = None,
) -> MultiRegistrationResult:
    """Execute complete multi-image registration, graph solving, and mosaic pipeline."""
    from app.services.mission_metadata import parse_metadata_label
    from app.services.footprint import create_image_footprint

    def emit(stage: str, stage_idx: int, progress_val: float, msg: str, status: str = "RUNNING", err: str | None = None):
        if progress_callback:
            try:
                progress_callback(
                    stage=stage,
                    stage_index=stage_idx,
                    stage_count=15,
                    progress=progress_val,
                    message=msg,
                    status=status,
                    error=err,
                )
            except Exception:
                pass

    emit("INITIALIZING", 1, 0.05, "Initializing multi-frame registration pipeline")

    if not 2 <= len(images) <= 200:
        raise ValueError("Upload 2–200 valid images to build a registration map.")

    emit("VALIDATING_INPUTS", 2, 0.10, f"Validating {len(images)} input image rasters and channels")

    for i, img in enumerate(images):
        if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
            raise ValueError(f"Image {i} is invalid or has empty dimensions.")

    settings = settings or {}
    criteria = {**DEFAULT_ACCEPTANCE, **settings.get("acceptance", {})}
    blending_mode = settings.get("blending_mode", "distance_weighted")
    started = perf_counter()
    cache = RegistrationCache(cache_dir or CACHE_DIR)

    emit("LOADING_IMAGES", 3, 0.15, f"Extracting mission metadata for {len(images)} frames")

    raw_metas = metadata_list or settings.get("metadata") or []
    parsed_metadata = [
        parse_metadata_label(raw_metas[i]) if i < len(raw_metas) else parse_metadata_label({"image_index": i})
        for i in range(len(images))
    ]

    # 1. Feature extraction
    emit("BUILDING_REGISTRATION_GRAPH", 4, 0.20, f"Extracting features and forming pair candidate graph")
    artifacts = []
    for i, image in enumerate(images):
        artifacts.append(cache.get_features(image, settings))
        if len(images) > 12 and (i % 5 == 0 or i == len(images) - 1):
            emit(
                "BUILDING_REGISTRATION_GRAPH",
                4,
                round(0.15 + 0.05 * ((i + 1) / len(images)), 2),
                f"Processing image {i + 1} / {len(images)}",
            )
    candidates = select_candidate_pairs(images, int(settings.get("max_candidates", 66)))
    edges = []
    outputs = []
    pair_time = 0.0


    # 2. Pairwise registration
    for edge_id, candidate in enumerate(candidates):
        a, b = candidate["image_a"], candidate["image_b"]
        fa, ga, fea, _, _ = artifacts[a]
        fb, gb, feb, _, _ = artifacts[b]
        meta_a = parsed_metadata[a].to_dict()
        meta_b = parsed_metadata[b].to_dict()
        key = cache.pair_key(fa, fb, settings)
        edge = {
            **candidate,
            "edge_id": edge_id,
            "source_image": a,
            "target_image": b,
            "accepted": False,
            "cache_hit": False,
            "pair_id": f"{a}-{b}",
        }

        try:
            stored = cache.load_pair(key)
            if stored and stored.get("failure"):
                edge.update({
                    "status": "rejected",
                    "registration_status": "FAIL",
                    "success": False,
                    "reason": stored["failure"],
                    "confidence": 0.0,
                    "cache_hit": True,
                    "processing_time_seconds": 0.0,
                })
                edges.append(edge)
                continue

            if stored:
                result = _cached(stored, ga, gb)
                edge["cache_hit"] = True
            else:
                result = register_pair(
                    images[a],
                    images[b],
                    detector=settings.get("detector", "sift"),
                    ratio=float(settings.get("ratio", 0.72)),
                    ransac_threshold=float(settings.get("ransac_threshold", 3.0)),
                    illumination_normalization=bool(settings.get("illumination_normalization", True)),
                    spatial_distribution=bool(settings.get("spatial_distribution", True)),
                    ecc_refinement=bool(settings.get("ecc_refinement", True)),
                    max_features=int(settings.get("max_features", 8000)),
                    match_preview_max_side=1200,
                    include_registered=False,
                    source_gray=ga,
                    reference_gray=gb,
                    source_features=fea,
                    reference_features=feb,
                    source_metadata=meta_a,
                    reference_metadata=meta_b,
                )
                cache.save_pair(key, _payload(result))
                pair_time += result.processing_time_seconds

            reason = _rejection_reason(result.metrics, criteria)
            is_successful = result.success or (result.registration_status in {"PASS", "SUCCESS"})
            is_accepted = (reason is None) and is_successful

            edge.update({
                "status": "accepted" if is_accepted else "rejected",
                "accepted": is_accepted,
                "success": is_successful,
                "registration_status": result.registration_status,
                "reason": reason if reason is not None else ("success" if is_accepted else "registration failed"),
                "model_type": result.geometric_model,
                "estimator": result.estimator_used,
                "inlier_count": result.metrics.get("inlier_count", 0),
                "inlier_ratio": result.metrics.get("inlier_ratio", 0.0),
                "rmse_pixels": result.metrics.get("rmse_pixels"),
                "spatial_coverage": result.metrics.get("source_spatial_coverage", 0.0),
                "spatial_uniformity": result.metrics.get("spatial_uniformity", 0.0),
                "spatial_distribution_category": result.metrics.get("spatial_distribution_category", "LIMITED_DISTRIBUTION"),
                "overlap": result.overlap_ratio,
                "overlap_ratio": result.overlap_ratio,
                "conditioning": result.metrics.get("transform_conditioning"),
                "metrics": result.metrics,
                "homography": result.homography.tolist(),
                "confidence": confidence_for_metrics(result.metrics),
                "transformation_type": result.geometric_model,
                "ecc_used": result.ecc_used,
                "processing_time_seconds": result.processing_time_seconds,
            })
            outputs.append((a, b, result, edge["cache_hit"]))
        except ValueError as exc:
            cache.save_pair(
                key,
                {
                    "failure": str(exc),
                    "source_points": np.empty((0, 1, 2), dtype=np.float32),
                    "reference_points": np.empty((0, 1, 2), dtype=np.float32),
                    "inlier_mask": np.empty(0, dtype=bool),
                },
            )
            edge.update({
                "status": "rejected",
                "registration_status": "FAIL",
                "success": False,
                "reason": str(exc),
                "confidence": 0.0,
                "processing_time_seconds": 0.0,
            })
        pair_prog = 0.20 + ((edge_id + 1) / max(1, len(candidates))) * 0.40
        emit(
            "REGISTERING_IMAGE_PAIRS",
            6,
            round(pair_prog, 2),
            f"Registering frame {a+1}/{len(images)} against frame {b+1} (pair {edge_id+1}/{len(candidates)})",
        )
        edges.append(edge)

    # 3. Graph construction and transformation propagation
    graph_start = perf_counter()
    emit("ANALYZING_CONNECTIVITY", 7, 0.62, "Analyzing graph connectivity across registration edges")
    graph = build_graph(len(images), edges)
    emit("SELECTING_REFERENCE", 8, 0.66, "Selecting optimal reference coordinate node")
    emit("PROPAGATING_GLOBAL_TRANSFORMS", 9, 0.70, "Propagating global transforms via Maximum Spanning Tree")
    placement = place_largest_component(graph, [x.shape for x in images])
    placement_time = perf_counter() - graph_start

    # 4. Mosaic construction
    mosaic = None
    mosaic_info = None
    mosaic_time = 0.0

    if len(placement["transforms"]) >= 2:
        mosaic_start = perf_counter()
        emit("COMPUTING_GLOBAL_BOUNDS", 10, 0.75, f"Computing global bounds from {len(placement['transforms'])} placed frames")
        emit("PREPARING_MOSAIC_CANVAS", 11, 0.80, "Allocating global canvas coordinate frame with safety margin")
        emit("RENDERING_IMAGES", 12, 0.85, f"Warping {len(placement['transforms'])} placed images to global canvas")
        emit("BLENDING_OVERLAPS", 13, 0.90, f"Applying {blending_mode} blending across overlap seams")
        try:
            mosaic, mosaic_info = build_relative_mosaic(
                images,
                placement["transforms"],
                blending_mode=blending_mode,
            )
            emit("COMPUTING_MOSAIC_METRICS", 14, 0.95, "Evaluating mosaic quality, coverage, and overlap consistency")
        except MosaicCanvasTooLargeError as exc:
            mosaic_info = {
                "error": str(exc),
                "error_code": exc.code,
                "details": {
                    "code": exc.code,
                    "requested_width": exc.requested_width,
                    "requested_height": exc.requested_height,
                    "estimated_memory_mb": exc.estimated_memory_mb,
                    "images_count": exc.images_count,
                    "current_bounds": exc.current_bounds,
                    "configured_safety_limit": exc.configured_safety_limit,
                    "actionable_recommendation": exc.actionable_recommendation,
                },
                "label": "Relative Registered Lunar Mosaic",
            }
        except ValueError as exc:
            mosaic_info = {"error": str(exc), "label": "Relative Registered Lunar Mosaic"}
        mosaic_time = perf_counter() - mosaic_start

    from app.services.pairwise_registration import serialize_correspondences
    pair_outputs = {
        f"{a}-{b}": {
            "source_index": a,
            "reference_index": b,
            "metrics": r.metrics,
            "inlier_points": serialize_inlier_points(r),
            "correspondences": serialize_correspondences(r),
            "homography": r.homography.tolist() if isinstance(r.homography, np.ndarray) else r.homography,
            "geometric_model": r.geometric_model,
            "estimator_used": r.estimator_used,
            "registration_status": r.registration_status,
            "representation": r.preprocessing.get("representation", "structural") if r.preprocessing else "structural",
            "match_visualization": r.match_visualization,
            "cache_hit": hit,
        }
        for a, b, r, hit in outputs
    }

    if mosaic_info and "mosaic_transforms" in mosaic_info:
        mosaic_points = []
        for a, b, result, _ in outputs:
            if not result or not np.any(result.inlier_mask):
                continue
            mask = np.asarray(result.inlier_mask, dtype=bool)
            for image_index, points in ((a, result.source_points), (b, result.reference_points)):
                matrix = (
                    np.asarray(mosaic_info["mosaic_transforms"].get(str(image_index)), dtype=np.float32)
                    if str(image_index) in mosaic_info["mosaic_transforms"]
                    else None
                )
                if matrix is None or len(points) == 0:
                    continue
                pts_arr = np.asarray(points).reshape(-1, 2)
                if len(pts_arr) == len(mask):
                    selected = pts_arr[mask]
                else:
                    selected = np.empty((0, 2), dtype=np.float32)
                if len(selected):
                    try:
                        projected = cv2.perspectiveTransform(selected.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)
                        mosaic_points.extend(
                            {
                                "image_index": image_index,
                                "x": round(float(x), 3),
                                "y": round(float(y), 3),
                                "kind": "inlier",
                            }
                            for x, y in projected
                            if 0 <= x < mosaic_info["width"] and 0 <= y < mosaic_info["height"]
                        )
                    except (cv2.error, ValueError):
                        pass
        mosaic_info["match_points"] = mosaic_points[:1200]

    stages = {
        "feature_extraction": round(sum(x[4] for x in artifacts), 6),
        "pair_registration": round(pair_time, 6),
        "placement": round(placement_time, 6),
        "mosaic": round(mosaic_time, 6),
    }

    summary = _summary(
        edges,
        len(images),
        graph,
        placement,
        mosaic_info,
        cache,
        started,
        len(images) * (len(images) - 1) // 2,
        len(candidates),
        stages,
    )
    summary["acceptance_criteria"] = criteria

    emit("FINALIZING_OUTPUT", 14, 0.98, "Serializing inlier points and registration summary")
    emit("COMPLETE", 15, 1.0, "Multi-image registered mosaic complete", status="COMPLETE")

    return MultiRegistrationResult(graph, placement, mosaic, mosaic_info, pair_outputs, summary)
