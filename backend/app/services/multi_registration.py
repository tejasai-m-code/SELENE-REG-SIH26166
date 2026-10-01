# -*- coding: utf-8 -*-
"""Cached, incremental multi-image orchestration built on the pair engine.
Phase 9 extension: calls build_global_graph() after pairwise registration to
produce globally consistent poses with cycle-consistency checks and optimization.
"""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from app.services.diagnostics import processing_device_diagnostics
from app.services.mosaic import build_relative_mosaic
from app.services.pairwise_registration import PairwiseRegistrationResult, register_pair, serialize_inlier_points
from app.services.registration import draw_matches
from app.services.registration_cache import RegistrationCache
from app.services.registration_graph import build_graph, confidence_for_metrics, place_largest_component
from app.utils.image_utils import resize_for_preview, to_gray
# Phase 9: global graph optimization
from app.services.global_graph import build_global_graph, serialize_global_result

DEFAULT_ACCEPTANCE = {"min_inliers": 6, "min_inlier_ratio": .10, "max_rmse_pixels": 8., "min_spatial_coverage": .0625}
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
@dataclass
class MultiRegistrationResult:
    graph: dict
    placement: dict
    mosaic: np.ndarray | None
    mosaic_info: dict | None
    pair_outputs: dict
    summary: dict
    global_graph: dict | None = None   # Phase 9: serialized global graph result
    p10_mosaic: dict | None = None

def thumbnail_signature(image):
    return cv2.normalize(cv2.resize(resize_for_preview(to_gray(image), 360), (64, 64), interpolation=cv2.INTER_AREA), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
def select_candidate_pairs(images, max_candidates=28):
    sigs = []
    for x in images:
        try:
            sigs.append(thumbnail_signature(x) if x is not None else None)
        except Exception:
            sigs.append(None)
    pairs = []
    for a, b in combinations(range(len(images)), 2):
        if sigs[a] is None or sigs[b] is None: continue
        score = float(cv2.compareHist(cv2.calcHist([sigs[a]], [0], None, [32], [0, 256]), cv2.calcHist([sigs[b]], [0], None, [32], [0, 256]), cv2.HISTCMP_CORREL))
        pairs.append({"image_a": a, "image_b": b, "screening_score": score})
    return pairs if len(images) <= 8 else sorted(pairs, key=lambda x: x["screening_score"], reverse=True)[:max_candidates]
def _rejection_reason(m, c):
    if m["inlier_count"] < c["min_inliers"]: return "insufficient reliable correspondences"
    if m["inlier_ratio"] < c["min_inlier_ratio"]: return "inlier ratio below configured prototype threshold"
    if m["rmse_pixels"] is None or m["rmse_pixels"] > c["max_rmse_pixels"]: return "reprojection error exceeds configured prototype threshold"
    if m["source_spatial_coverage"] < c["min_spatial_coverage"]: return "insufficient spatial distribution of inlier matches"
def _payload(result):
    return {"homography": result.homography.tolist(), "metrics": result.metrics, "source_points": result.source_points, "reference_points": result.reference_points, "inlier_mask": result.inlier_mask, "raw_match_count": result.raw_match_count, "post_distribution_match_count": result.post_distribution_match_count, "ecc_used": result.ecc_used, "ecc_correlation": result.ecc_correlation, "detector": result.detector, "processing_time_seconds": result.processing_time_seconds}
def _cached(payload, source_gray, reference_gray):
    a,b,mask=payload["source_points"],payload["reference_points"],payload["inlier_mask"]
    return PairwiseRegistrationResult(np.asarray(payload["homography"],dtype=float),mask,payload["metrics"],None,draw_matches(source_gray,reference_gray,a,b,mask),a,b,payload["raw_match_count"],payload["post_distribution_match_count"],payload["ecc_used"],payload.get("ecc_correlation"),payload["detector"],payload.get("processing_time_seconds",0.0))
def _summary(edges, images, graph, placement, cache, started, possible, candidates, stages):
    accepted=[e for e in edges if e["accepted"]]; ms=[e["metrics"] for e in accepted]; rmses=[m["rmse_pixels"] for m in ms if m["rmse_pixels"] is not None]
    mean=lambda xs: float(np.mean(xs)) if xs else None
    return {"images_total":images,"images_registered":len(placement["transforms"]),"images_unplaced":images-len(placement["transforms"]),"image_count":images,"total_possible_pairs":possible,"candidate_pairs":candidates,"candidate_pair_count":candidates,"processed_pairs":sum(not e["cache_hit"] for e in edges),"reused_pairs":sum(e["cache_hit"] for e in edges),"accepted_pairs":len(accepted),"rejected_pairs":len(edges)-len(accepted),"successful_pair_count":len(accepted),"failed_pair_count":len(edges)-len(accepted),"placed_image_count":len(placement["transforms"]),"isolated_image_count":len(graph["isolated_nodes"]),"graph_component_count":len(graph["components"]),"connected_components":len(graph["components"]),"graph_connected":graph["connected"],"isolated_images":graph["isolated_nodes"],"average_inliers":mean([m["inlier_count"] for m in ms]),"average_inliers_per_accepted_pair":mean([m["inlier_count"] for m in ms]),"average_inlier_ratio":mean([m["inlier_ratio"] for m in ms]),"spatial_coverage_mean":mean([m["source_spatial_coverage"] for m in ms]),"average_source_spatial_coverage":mean([m["source_spatial_coverage"] for m in ms]),"rmse_mean":mean(rmses),"rmse_median":float(np.median(rmses)) if rmses else None,"rmse_min":float(np.min(rmses)) if rmses else None,"rmse_max":float(np.max(rmses)) if rmses else None,"rmse_statistics_pixels":{"mean":mean(rmses),"median":float(np.median(rmses)) if rmses else None,"minimum":float(np.min(rmses)) if rmses else None,"maximum":float(np.max(rmses)) if rmses else None},"cache_hits":cache.stats["pair_hits"],"cache_misses":cache.stats["pair_misses"],"cache":cache.stats,"processing_time_seconds":round(perf_counter()-started,3),"stage_timings_seconds":stages,"processing_diagnostics":processing_device_diagnostics(),"processing_device":processing_device_diagnostics()["device"],"scientific_note":"All placement is relative image-space registration. Target SIH dataset validation and geographic control are required for scientific georeferencing."}

def run_multi_registration(images, settings=None, cache_dir=None):
    if not 2 <= len(images) <= 12: raise ValueError("Upload 2–12 valid images to build a registration map.")
    settings=settings or {}; criteria={**DEFAULT_ACCEPTANCE,**settings.get("acceptance",{})}; started=perf_counter(); cache=RegistrationCache(cache_dir or CACHE_DIR)
    artifacts=[]
    for image in images:
        try:
            if image is None: raise ValueError()
            artifacts.append(cache.get_features(image,settings))
        except Exception:
            artifacts.append(None)
    candidates=select_candidate_pairs(images,int(settings.get("max_candidates",28))); edges=[]; outputs=[]; pair_time=0.
    for edge_id, candidate in enumerate(candidates):
        a,b=candidate["image_a"],candidate["image_b"]; fa,ga,fea,_,feature_time_a=artifacts[a]; fb,gb,feb,_,feature_time_b=artifacts[b]; key=cache.pair_key(fa,fb,settings); edge={**candidate,"edge_id":edge_id,"accepted":False,"cache_hit":False,"pair_id":f"{a}-{b}"}
        try:
            stored=cache.load_pair(key)
            if stored and stored.get("failure"):
                edge.update({"status":"rejected","reason":stored["failure"],"confidence":0.,"cache_hit":True,"processing_time_seconds":0.})
                edges.append(edge); continue
            if stored: result=_cached(stored,ga,gb); edge["cache_hit"]=True
            else:
                result=register_pair(images[a],images[b],detector=settings.get("detector","sift"),ratio=float(settings.get("ratio",.72)),ransac_threshold=float(settings.get("ransac_threshold",3)),illumination_normalization=bool(settings.get("illumination_normalization",True)),spatial_distribution=bool(settings.get("spatial_distribution",True)),ecc_refinement=bool(settings.get("ecc_refinement",True)),max_features=int(settings.get("max_features",8000)),match_preview_max_side=1200,include_registered=False,source_gray=ga,reference_gray=gb,source_features=fea,reference_features=feb); cache.save_pair(key,_payload(result)); pair_time+=result.processing_time_seconds
            reason=_rejection_reason(result.metrics,criteria); edge.update({"status":"accepted" if reason is None else "rejected","accepted":reason is None,"reason":reason,"metrics":result.metrics,"homography":result.homography.tolist(),"confidence":confidence_for_metrics(result.metrics),"transformation_type":"homography","ecc_used":result.ecc_used,"processing_time_seconds":result.processing_time_seconds,"spatial_distribution_status":"Spatial inlier distribution"}); outputs.append((a,b,result,edge["cache_hit"]))
        except ValueError as exc:
            cache.save_pair(key,{"failure":str(exc),"source_points":np.empty((0,1,2),dtype=np.float32),"reference_points":np.empty((0,1,2),dtype=np.float32),"inlier_mask":np.empty(0,dtype=bool)})
            edge.update({"status":"rejected","reason":str(exc),"confidence":0.,"processing_time_seconds":0.})
        edges.append(edge)
    graph_start=perf_counter(); graph=build_graph(len(images),edges); placement=place_largest_component(graph,[x.shape if x is not None else (0,0) for x in images]); placement_time=perf_counter()-graph_start; mosaic=mosaic_info=None; mosaic_time=0.
    if len(placement["transforms"])>=2:
        mosaic_start=perf_counter()
        try: mosaic,mosaic_info=build_relative_mosaic(images,placement["transforms"])
        except ValueError as exc: mosaic_info={"error":str(exc),"label":"Relative Registered Lunar Mosaic"}
        mosaic_time=perf_counter()-mosaic_start
    pair_outputs={f"{a}-{b}":{"metrics":r.metrics,"inlier_points":serialize_inlier_points(r),"match_visualization":r.match_visualization,"cache_hit":hit} for a,b,r,hit in outputs}
    if mosaic_info and "mosaic_transforms" in mosaic_info:
        mosaic_points=[]
        for a,b,result,_ in outputs:
            for image_index, points in ((a, result.source_points), (b, result.reference_points)):
                matrix=np.asarray(mosaic_info["mosaic_transforms"].get(str(image_index)), dtype=np.float32) if str(image_index) in mosaic_info["mosaic_transforms"] else None
                if matrix is None: continue
                selected=points.reshape(-1,2)[result.inlier_mask.ravel()]
                if len(selected):
                    projected=cv2.perspectiveTransform(selected.reshape(-1,1,2),matrix).reshape(-1,2)
                    mosaic_points.extend({"image_index":image_index,"x":round(float(x),3),"y":round(float(y),3),"kind":"inlier"} for x,y in projected if 0 <= x < mosaic_info["width"] and 0 <= y < mosaic_info["height"])
        mosaic_info["match_points"]=mosaic_points[:1200]
    stages={"feature_extraction":round(sum(x[4] for x in artifacts if x is not None),6),"pair_registration":round(pair_time,6),"placement":round(placement_time,6),"mosaic":round(mosaic_time,6)}; summary=_summary(edges,len(images),graph,placement,cache,started,len(images)*(len(images)-1)//2,len(candidates),stages); summary["acceptance_criteria"]=criteria

    # ---- Phase 9: Global Graph Optimization ----------------------------
    # Build rich node info from available metadata
    nodes_info = [
        {"image_id": i, "filename": f"image_{i}", "sensor": "Unknown",
         "width": int(images[i].shape[1]) if images[i] is not None else 0, "height": int(images[i].shape[0]) if images[i] is not None else 0}
        for i in range(len(images))
    ]
    # Pass subpixel_consensus from outputs back into edge data for Phase 9
    subpixel_by_pair = {}
    for a, b, reg_result, _ in outputs:
        sub = getattr(reg_result, "subpixel_consensus", None)
        subpixel_by_pair[f"{a}-{b}"] = sub

    # Enrich edges with subpixel data before passing to Phase 9
    enriched_edges = []
    for edge in edges:
        e = dict(edge)
        key = f"{edge['image_a']}-{edge['image_b']}"
        e["subpixel_consensus"] = subpixel_by_pair.get(key)
        enriched_edges.append(e)

    global_result = None
    try:
        global_graph_obj = build_global_graph(
            nodes_info=nodes_info,
            pairwise_edges=enriched_edges,
            image_shapes=[img.shape if img is not None else (0,0,0) for img in images],
        )
        global_result = serialize_global_result(global_graph_obj)
        # Expose key global metrics in summary
        summary["global_optimization"] = {
            "method": global_graph_obj.optimization.method,
            "status": global_graph_obj.optimization.status,
            "objective_before": global_graph_obj.optimization.objective_before,
            "objective_after": global_graph_obj.optimization.objective_after,
            "converged": global_graph_obj.optimization.converged,
        }
        summary["cycle_diagnostics_count"] = len(global_graph_obj.cycle_diagnostics)
        summary["inconsistent_cycles"] = sum(
            1 for c in global_graph_obj.cycle_diagnostics if not c.consistent
        )
    except Exception as exc:
        # Fallback: Phase 9 optimization unavailable; V1 placement remains valid
        global_result = {"error": str(exc), "fallback": "V1 tree placement used"}
        summary["global_optimization"] = {"status": "FALLBACK_V1", "reason": str(exc)}

    # Phase 10: Scientific mosaic using Phase 9 global poses
    p10_result = None
    try:
        from app.services.scientific_mosaic import build_scientific_mosaic, serialize_mosaic_result as ser_p10
        p10_obj = build_scientific_mosaic(
            images=images,
            global_graph_data=global_result,
            nodes_info=nodes_info,
            settings=settings,
        )
        p10_result = ser_p10(p10_obj)
        # Store display images to disk
        for comp in p10_obj.components:
            if comp.mosaic_display is not None:
                disp_path = CACHE_DIR.parent / 'outputs' / f'p10_mosaic_comp{comp.component_id}_{id(comp)}.png'
                try:
                    from app.utils.image_utils import save_image as _si
                    _si(disp_path, comp.mosaic_display)
                    # Add path to serialized result
                    for sd in p10_result.get('components', []):
                        if sd.get('component_id') == comp.component_id:
                            sd['mosaic_display_path'] = f'/outputs/{disp_path.name}'
                except Exception:
                    pass
    except Exception as exc:
        p10_result = {'error': str(exc), 'engine': 'phase10_scientific', 'fallback': 'V1_mosaic_used'}

    return MultiRegistrationResult(graph, placement, mosaic, mosaic_info, pair_outputs, summary,
                                   global_graph=global_result, p10_mosaic=p10_result)

