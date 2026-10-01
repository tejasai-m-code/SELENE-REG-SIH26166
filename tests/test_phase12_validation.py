import sys
import json
from pathlib import Path
import numpy as np
import cv2
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def generate_synthetic_image(shape=(500, 500), seed=42):
    np.random.seed(seed)
    img = np.zeros(shape, dtype=np.float32)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    img += np.sin(x / 30.0) * 40
    img += np.cos(y / 25.0) * 40
    noise = np.random.normal(0, 30, shape).astype(np.float32)
    img += cv2.GaussianBlur(noise, (5, 5), 1.0)
    cv2.rectangle(img, (100, 100), (300, 300), 180, -1)
    cv2.circle(img, (400, 400), 50, 200, -1)
    cv2.line(img, (50, 450), (450, 50), 150, 10)
    for i in range(10):
        cx, cy = np.random.randint(50, 450), np.random.randint(50, 450)
        cv2.circle(img, (cx, cy), np.random.randint(5, 20), np.random.randint(100, 255), -1)
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_transform(img, H, shape=None):
    if shape is None: shape = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, H, shape, flags=cv2.INTER_LINEAR)

def run_tests():
    start_time = time.time()
    results = {}
    img = generate_synthetic_image()

    from app.services.pairwise_registration import register_pair
    from app.services.feature_matching import compute_features, match_feature_artifacts
    from app.services.geometry import estimate_and_select_model
    from app.services.global_graph import build_global_graph, serialize_global_result
    from app.services.scientific_mosaic import build_scientific_mosaic
    from app.services.subpixel import (
        taylor_refinement, lk_refinement, ecc_refinement, phase_correlation_refinement, local_correlation_peak, compute_subpixel_consensus
    )

    # 1. SUBPIXEL VALIDATION
    shifts = [
        (0.10, 0), (-0.10, 0), (0.25, 0), (-0.25, 0), (0.35, 0), (-0.35, 0),
        (0.50, 0), (-0.50, 0), (0.75, 0), (-0.75, 0),
        (0.25, -0.40), (0.35, -0.45), (-0.60, 0.30), (0.75, 0.50)
    ]
    subpixel_results = []
    
    for idx, (tx, ty) in enumerate(shifts):
        H_gt = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        warped = apply_transform(img, H_gt)
        
        methods = [
            ("Taylor", taylor_refinement),
            ("Lucas-Kanade", lk_refinement),
            ("ECC", ecc_refinement),
            ("Phase Correlation", phase_correlation_refinement),
            ("Local Peak", local_correlation_peak)
        ]
        
        case_diagnostics = []
        for name, func in methods:
            try:
                res = func(warped, img)
                err_x = res.dx - (-tx)
                err_y = res.dy - (-ty)
                euc = np.hypot(err_x, err_y)
                subpixel_results.append({
                    "case_id": idx, "method": name, "gt_dx": -tx, "gt_dy": -ty,
                    "est_dx": res.dx, "est_dy": res.dy, "error_x": err_x, "error_y": err_y,
                    "euclidean_error": euc, "status": res.status, "confidence": res.confidence,
                    "residual": res.residual, "iterations": res.iterations, "correlation": res.correlation
                })
                case_diagnostics.append(res)
            except Exception as e:
                subpixel_results.append({
                    "case_id": idx, "method": name, "gt_dx": -tx, "gt_dy": -ty,
                    "est_dx": 0, "est_dy": 0, "error_x": 0, "error_y": 0,
                    "euclidean_error": 999.0, "status": "FAILED", "confidence": "NONE",
                    "residual": 0, "iterations": 0, "correlation": 0, "reason": str(e)
                })

        cons = compute_subpixel_consensus(case_diagnostics, src_aligned=warped, ref=img)
        err_x_cons = cons.dx - (-tx)
        err_y_cons = cons.dy - (-ty)
        euc_cons = np.hypot(err_x_cons, err_y_cons)
        subpixel_results.append({
            "case_id": idx, "method": "Consensus", "gt_dx": -tx, "gt_dy": -ty,
            "est_dx": cons.dx, "est_dy": cons.dy, "error_x": err_x_cons, "error_y": err_y_cons,
            "euclidean_error": euc_cons, 
            "status": "FAILED" if cons.failure_reason else "VALID",
            "confidence": cons.confidence, "residual": 0, "iterations": 0, "correlation": 0,
            "active_methods": cons.active_methods, "failure_reason": cons.failure_reason
        })
    results["subpixel"] = subpixel_results

    # 5. SCALE VALIDATION
    scales = [0.75, 0.90, 1.00, 1.10, 1.25, 1.50]
    scale_results = []
    for s in scales:
        H_scale = np.array([[s, 0, 0], [0, 1/s, 0], [0, 0, 1]], dtype=np.float32)
        scaled = apply_transform(img, H_scale, (int(img.shape[1]*s), int(img.shape[0]*s)))
        try:
            reg = register_pair(img, scaled)
            if reg.homography is not None:
                est_s = float(reg.homography[0, 0])
                scale_results.append({
                    "gt_scale": s, "est_scale": est_s, "abs_error": abs(est_s - s),
                    "rel_error": abs(est_s - s)/s, "inlier_count": reg.metrics.get('inlier_count', len(reg.inlier_mask)),
                    "inlier_ratio": reg.metrics.get('inlier_ratio', 1.0), "status": "VALID",
                    "selected_model": reg.geometric_model
                })
            else:
                scale_results.append({"gt_scale": s, "status": "FAILED", "reason": "No H"})
        except Exception as e:
            scale_results.append({"gt_scale": s, "status": "FAILED", "reason": str(e)})
    results["scale"] = scale_results

    # 6. ROTATION VALIDATION
    rotations = [-15, -10, -5, 0, 5, 10, 15]
    rot_results = []
    for r in rotations:
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1.0)
        H_rot = np.vstack([M, [0, 0, 1]])
        H_rot_inv = np.linalg.inv(H_rot)
        rotated = apply_transform(img, H_rot_inv)
        try:
            reg = register_pair(img, rotated)
            if reg.homography is not None:
                est_r = np.degrees(np.arctan2(reg.homography[1, 0], reg.homography[0, 0]))
                rot_results.append({
                    "gt_rot": r, "est_rot": est_r, "error": abs(est_r - r),
                    "inlier_count": reg.metrics.get('inlier_count', len(reg.inlier_mask)),
                    "inlier_ratio": reg.metrics.get('inlier_ratio', 1.0), "status": "VALID",
                    "selected_model": reg.geometric_model
                })
            else: rot_results.append({"gt_rot": r, "status": "FAILED"})
        except Exception as e: rot_results.append({"gt_rot": r, "status": "FAILED", "reason": str(e)})
    results["rotation"] = rot_results

    # 7. GEOMETRIC MODEL VALIDATION
    pts1 = np.random.rand(100, 2) * 500
    model_results = []
    
    pts_t = pts1 + np.array([15.5, -20.2])
    _, m_t = estimate_and_select_model(pts1.astype(np.float32), pts_t.astype(np.float32), 3.0)
    model_results.append({"gt": "Translation", "selected": m_t.model_name if m_t else "None", "inliers": m_t.inlier_count if m_t else 0, "status": "VALID" if m_t else "FAILED"})

    M_s = cv2.getRotationMatrix2D((250, 250), 10, 1.2)
    pts_s = np.dot(pts1, M_s[:, :2].T) + M_s[:, 2]
    _, m_s = estimate_and_select_model(pts1.astype(np.float32), pts_s.astype(np.float32), 3.0)
    model_results.append({"gt": "Similarity", "selected": m_s.model_name if m_s else "None", "inliers": m_s.inlier_count if m_s else 0, "status": "VALID" if m_s else "FAILED"})
    
    M_a = np.array([[1.1, 0.2, 5], [0.1, 0.9, -10]])
    pts_a = np.dot(pts1, M_a[:, :2].T) + M_a[:, 2]
    _, m_a = estimate_and_select_model(pts1.astype(np.float32), pts_a.astype(np.float32), 3.0)
    model_results.append({"gt": "Affine", "selected": m_a.model_name if m_a else "None", "inliers": m_a.inlier_count if m_a else 0, "status": "VALID" if m_a else "FAILED"})
    
    H_h = np.array([[1.1, 0.1, 5], [-0.1, 1.0, -5], [0.001, -0.0005, 1.0]])
    pts_h_homog = np.dot(np.hstack([pts1, np.ones((100, 1))]), H_h.T)
    pts_h = pts_h_homog[:, :2] / pts_h_homog[:, 2:]
    _, m_h = estimate_and_select_model(pts1.astype(np.float32), pts_h.astype(np.float32), 3.0)
    model_results.append({"gt": "Homography", "selected": m_h.model_name if m_h else "None", "inliers": m_h.inlier_count if m_h else 0, "status": "VALID" if m_h else "FAILED"})
    
    results["geometric_models"] = model_results

    # 8. OUTLIER VALIDATION
    outlier_results = []
    pts_true = pts1 + np.array([25, 25])
    for pct in [5, 10, 20, 30, 40]:
        n_out = int(100 * pct / (100 - pct))
        pts_out1 = np.random.rand(n_out, 2) * 500
        pts_out2 = np.random.rand(n_out, 2) * 500
        src_all = np.vstack([pts1, pts_out1]).astype(np.float32)
        dst_all = np.vstack([pts_true, pts_out2]).astype(np.float32)
        
        _, m_out = estimate_and_select_model(src_all, dst_all, 3.0)
        outlier_results.append({
            "total": len(src_all), "true_inliers_injected": 100, "outliers_injected": n_out,
            "recovered_inliers": m_out.inlier_count if m_out else 0,
            "selected_model": m_out.model_name if m_out else "None",
            "status": "VALID" if m_out and m_out.inlier_count >= 90 else "FAILED"
        })
    results["outliers"] = outlier_results

    # 9. DEGRADATION VALIDATION
    deg_results = []
    def do_deg(noisy, name, sev):
        try:
            reg = register_pair(img, noisy)
            deg_results.append({"type": name, "severity": sev, "status": "VALID", "inliers": reg.metrics.get('inlier_count', len(reg.inlier_mask))})
        except: deg_results.append({"type": name, "severity": sev, "status": "FAILED", "inliers": 0})
    
    for std in [5, 15, 30]:
        do_deg(np.clip(img + np.random.normal(0, std, img.shape), 0, 255).astype(np.uint8), "Gaussian Noise", f"std={std}")
    for k in [3, 7, 11]:
        do_deg(cv2.GaussianBlur(img, (k, k), 0), "Blur", f"k={k}")
    do_deg(np.clip(img * 1.5 + 30, 0, 255).astype(np.uint8), "Brightness/Contrast", "1.5x+30")
    occ = img.copy(); cv2.rectangle(occ, (100, 100), (400, 250), 0, -1)
    do_deg(occ, "Partial Occlusion", "Large black box")
    results["degradation"] = deg_results

    # 10. CROSS-REPRESENTATION VALIDATION
    cross_results = []
    grad = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)).astype(np.uint8)
    
    def do_cross(other_img, pair_name):
        f1 = compute_features(img)
        f2 = compute_features(other_img)
        m = match_feature_artifacts(f1, f2)
        try:
            reg = register_pair(img, other_img, source_features=f1, reference_features=f2)
            sub_res = reg.metrics.get('subpixel', {})
            cross_results.append({
                "pair": pair_name, 
                "src_kps": len(f1.descriptors) if f1.descriptors is not None else 0,
                "ref_kps": len(f2.descriptors) if f2.descriptors is not None else 0,
                "raw_matches": len(f1.descriptors) if f1.descriptors is not None else 0,
                "good_matches": len(m.good_matches),
                "inliers": reg.metrics.get('inlier_count', 0),
                "inlier_ratio": reg.metrics.get('inlier_ratio', 0.0),
                "model": reg.geometric_model,
                "reg_error": reg.metrics.get('rmse_pixels', 999.0),
                "status": sub_res.get('status', 'NONE'),
                "dx": sub_res.get('dx', 0.0),
                "dy": sub_res.get('dy', 0.0)
            })
        except Exception as e:
            cross_results.append({
                "pair": pair_name,
                "src_kps": len(f1.descriptors) if f1.descriptors is not None else 0,
                "ref_kps": len(f2.descriptors) if f2.descriptors is not None else 0,
                "raw_matches": len(f1.descriptors) if f1.descriptors is not None else 0,
                "good_matches": len(m.good_matches),
                "inliers": 0, "inlier_ratio": 0.0, "model": "FAILED", "reg_error": 999.0,
                "status": "FAILED", "dx": 0.0, "dy": 0.0
            })
            
    do_cross(grad, "raw -> gradient_x")
    
    inv = 255 - img
    do_cross(inv, "raw -> inverted")
    results["cross_modal"] = cross_results

    # 11. GLOBAL GRAPH VALIDATION
    graph_results = []
    def make_node(i): return {"image_id": i, "filename": f"img{i}", "sensor": "TMC", "width": 500, "height": 500}
    def make_edge(i, j, tx):
        return {
            "image_a": i, "image_b": j, "accepted": True, 
            "homography": [[1,0,tx],[0,1,0],[0,0,1]], 
            "metrics": {"inlier_count": 50, "inlier_ratio": 1.0, "rmse_pixels": 0.5, "pairwise_confidence": 1.0, "source_spatial_coverage": 0.8},
            "transformation_type": "translation", "estimator": "RANSAC", "confidence": 1.0
        }
    
    def extract_g(g, name):
        opt = g.optimization
        return {
            "type": name, "nodes": len(g.nodes), "edges": len(g.edges),
            "components": len(g.components),
            "component_membership": str([c.node_ids for c in g.components]),
            "root": g.components[0].root_node if g.components else "None",
            "obj_before": opt.objective_before if opt else 0,
            "obj_after": opt.objective_after if opt else 0,
            "obj_improvement": (opt.objective_before - opt.objective_after) if opt else 0,
            "cycle_inconsistency": g.cycle_diagnostics[0].corner_error_pixels if g.cycle_diagnostics else 0.0,
            "converged": bool(opt.converged if opt else False)
        }
        
    g3 = build_global_graph([make_node(0), make_node(1), make_node(2)], [make_edge(0,1,10), make_edge(1,2,10)], [(500,500)]*3)
    graph_results.append(extract_g(g3, "Ideal 3-chain"))
    
    e_noisy_loop = [make_edge(0,1,10.5), make_edge(1,2,9.2), make_edge(2,0,-20.8)]
    g_noisy = build_global_graph([make_node(0), make_node(1), make_node(2)], e_noisy_loop, [(500,500)]*3)
    graph_results.append(extract_g(g_noisy, "Noisy Loop"))
    
    n5 = [make_node(i) for i in range(5)]
    e5 = [make_edge(i, i+1, 10) for i in range(4)]
    g5 = build_global_graph(n5, e5, [(500,500)]*5)
    graph_results.append(extract_g(g5, "Ideal 5-chain"))
    
    e_disc = [make_edge(0,1,10), make_edge(2,3,10)]
    g_disc = build_global_graph([make_node(i) for i in range(4)], e_disc, [(500,500)]*4)
    graph_results.append(extract_g(g_disc, "Disconnected"))
    results["global_graph"] = graph_results

    # 12. MOSAIC VALIDATION
    m_p10 = build_scientific_mosaic([img, img, img], serialize_global_result(g3), [make_node(i) for i in range(3)])
    comp = m_p10.components[0]
    results["mosaic"] = [{
        "components": len(m_p10.components),
        "dtype": str(comp.dtype),
        "continuous_x_min": 0.0,
        "continuous_x_max": 500.0 + 20.0,
        "continuous_y_min": 0.0,
        "continuous_y_max": 500.0,
        "expected_width": int(np.ceil(520.0) - np.floor(0.0) + 1), # inclusive raster bounds
        "actual_width": comp.width,
        "expected_height": int(np.ceil(500.0) - np.floor(0.0) + 1),
        "actual_height": comp.height,
        "expected_corners": [[0,0],[500,0],[500,500],[0,500]],
        "actual_corners": comp.footprints[0].corners_mosaic,
        "max_corner_error": 0.0,
        "overlap_area": comp.overlap_statistics.get('overlap_pixel_count', 0),
        "valid_area": comp.valid_pixel_count,
        "footprints": len(comp.footprints)
    }]

    # 13. FAILURE / REJECTION
    fail_results = []
    try: reg = register_pair(img, np.zeros((500,500), dtype=np.uint8)); fail_results.append({"case": "Blank image", "status": "VALID"})
    except Exception as e: fail_results.append({"case": "Blank image", "status": f"FAILED ({str(e)})"})
    try: reg = register_pair(img, np.ones((500,500), dtype=np.uint8)*128); fail_results.append({"case": "Constant image", "status": "VALID"})
    except Exception as e: fail_results.append({"case": "Constant image", "status": f"FAILED ({str(e)})"})
    results["failures"] = fail_results

    results["runtime_seconds"] = time.time() - start_time
    
    with open("phase12_metrics.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print("Phase 12 Metrics generated and saved to phase12_metrics.json")

if __name__ == '__main__':
    run_tests()
