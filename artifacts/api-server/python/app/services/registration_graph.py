"""Graph construction, reference node selection, transformation propagation, and cycle consistency."""

from collections import defaultdict, deque
import cv2
import numpy as np


def confidence_for_metrics(metrics: dict) -> float:
    """Scientific multi-factor quality score for pairwise graph edges.

    Combines:
    - inlier count and inlier ratio
    - spatial coverage and spatial uniformity
    - reprojection error penalty
    - transform conditioning penalty
    """
    inliers = float(metrics.get("inlier_count", 0))
    ratio = float(metrics.get("inlier_ratio", 0.0))
    coverage = float(metrics.get("source_spatial_coverage", 0.0))
    uniformity = float(metrics.get("spatial_uniformity", 0.0))
    rmse = metrics.get("rmse_pixels")
    cond = metrics.get("transform_conditioning")

    if inliers < 4 or ratio <= 0.0:
        return 0.0

    # Base support score
    support = inliers * ratio

    # Spatial distribution multiplier (0.2 to 1.0)
    spatial_factor = (0.25 + 0.75 * max(0.0, min(1.0, coverage))) * (0.4 + 0.6 * max(0.0, min(1.0, uniformity)))

    # RMSE penalty (lower RMSE -> higher multiplier)
    rmse_factor = 1.0
    if rmse is not None and rmse > 0:
        rmse_factor = 1.0 / (1.0 + max(0.0, rmse / 2.0))

    # Conditioning penalty (ill-conditioned homography penalized)
    cond_factor = 1.0
    if cond is not None and cond > 1e4:
        cond_factor = max(0.1, 1e4 / cond)

    score = support * spatial_factor * rmse_factor * cond_factor
    return float(round(score, 4))


def build_graph(image_count: int, edges: list[dict]) -> dict:
    """Construct a multi-image registration graph from pairwise edges.

    IMAGE = NODE
    PAIRWISE REGISTRATION = EDGE

    Detects connected components, isolated nodes, and summarizes edge validity.
    """
    adjacency = defaultdict(list)
    accepted_edges = []
    rejected_edges = []

    for edge_index, edge in enumerate(edges):
        if edge.get("accepted", False):
            a, b = edge["image_a"], edge["image_b"]
            adjacency[a].append((b, edge_index))
            adjacency[b].append((a, edge_index))
            accepted_edges.append(edge)
        else:
            rejected_edges.append(edge)

    # Connected component discovery (BFS)
    components, unseen = [], set(range(image_count))
    while unseen:
        start = unseen.pop()
        component, queue = [], deque([start])
        while queue:
            node = queue.popleft()
            component.append(node)
            for other, _ in adjacency[node]:
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        components.append(sorted(component))

    # Sort components by descending size
    components.sort(key=len, reverse=True)
    isolated_nodes = sorted(node for node in range(image_count) if not adjacency[node])

    return {
        "nodes": list(range(image_count)),
        "edges": edges,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "components": components,
        "connected": len(components) == 1 and image_count > 0,
        "component_count": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "isolated_nodes": isolated_nodes,
        "isolated_count": len(isolated_nodes),
    }


def _maximum_spanning_tree(nodes: list[int], edges: list[dict]) -> list[dict]:
    """Find maximum-confidence spanning tree on a connected component using Kruskal's algorithm."""
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected = []
    # Sort edges descending by confidence score
    sorted_edges = sorted(edges, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    for edge in sorted_edges:
        a, b = find(edge["image_a"]), find(edge["image_b"])
        if a != b:
            parent[a] = b
            selected.append(edge)
    return selected


def select_reference_node(component: list[int], edges: list[dict], image_shapes: list[tuple] | None = None) -> tuple[int, dict]:
    """Select the optimal reference coordinate frame node for a connected component.

    Selection factors:
    1. Incident connection degree (number of reliable pairwise links).
    2. Sum of incident edge confidence scores.
    3. Closeness centrality (shortest path length to all other component nodes).
    4. Average reprojection RMSE of incident edges.
    5. Average spatial coverage of incident edges.

    Deterministic tie-breaker: lowest node index.
    """
    if not component:
        raise ValueError("Cannot select reference node from an empty component.")
    if len(component) == 1:
        return component[0], {"rationale": "Single isolated node in component", "scores": {component[0]: 1.0}}

    # Build incident edge map
    incident_edges = defaultdict(list)
    for edge in edges:
        if edge.get("accepted", False):
            a, b = edge["image_a"], edge["image_b"]
            if a in component and b in component:
                incident_edges[a].append(edge)
                incident_edges[b].append(edge)

    # Compute shortest path distances (BFS) for closeness centrality
    adj = defaultdict(list)
    for edge in edges:
        if edge.get("accepted", False):
            a, b = edge["image_a"], edge["image_b"]
            if a in component and b in component:
                adj[a].append(b)
                adj[b].append(a)

    scores = {}
    details = {}

    for node in component:
        inc = incident_edges[node]
        deg = len(inc)
        conf_sum = sum(float(e.get("confidence", 0.0)) for e in inc)

        rmses = [float(e["metrics"]["rmse_pixels"]) for e in inc if e.get("metrics", {}).get("rmse_pixels") is not None]
        avg_rmse = float(np.mean(rmses)) if rmses else 5.0

        coverages = [float(e.get("metrics", {}).get("source_spatial_coverage", 0.0)) for e in inc]
        avg_cov = float(np.mean(coverages)) if coverages else 0.0

        # Closeness centrality: sum of shortest path hop counts
        dist = {node: 0}
        q = deque([node])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    q.append(nbr)
        total_hops = sum(dist.get(target, len(component)) for target in component if target != node)
        closeness = float(len(component) - 1) / max(total_hops, 1)

        # Composite score
        # High degree, high confidence, high closeness centrality, high coverage, low RMSE
        score = (
            2.0 * deg
            + 1.5 * conf_sum
            + 4.0 * closeness
            + 2.0 * avg_cov
            - 0.5 * avg_rmse
        )
        scores[node] = float(round(score, 4))
        details[node] = {
            "degree": deg,
            "total_confidence": round(conf_sum, 3),
            "closeness_centrality": round(closeness, 3),
            "average_rmse": round(avg_rmse, 3),
            "average_coverage": round(avg_cov, 3),
            "composite_score": round(score, 4),
        }

    # Pick node with maximum score; tie-breaker: lowest index
    selected = max(component, key=lambda n: (scores[n], -n))

    rationale = (
        f"Image {selected} selected as reference node: degree={details[selected]['degree']}, "
        f"confidence_sum={details[selected]['total_confidence']}, "
        f"closeness={details[selected]['closeness_centrality']}, "
        f"avg_rmse={details[selected]['average_rmse']} px."
    )

    return selected, {
        "selected_reference_node": selected,
        "rationale": rationale,
        "node_evaluations": details,
    }


def evaluate_cycle_consistency(
    component: list[int],
    accepted_edges: list[dict],
    image_shapes: list[tuple],
    tolerance_pixels: float = 5.0,
) -> dict:
    """Evaluate transformation cycle consistency across all 3-cycles and 4-cycles in the graph.

    For each cycle (A -> B -> C -> A), composes transformations around the loop:
    H_cycle = H_{C->A} * H_{B->C} * H_{A->B}.
    Evaluates corner transfer error against the reference image coordinates:
    e_cycle = mean(||H_cycle * corners - corners||).
    """
    # Build edge lookup: (u, v) -> H_{u -> v}
    edge_map = {}
    edge_obj_map = {}
    adj = defaultdict(set)
    for edge in accepted_edges:
        a, b = edge["image_a"], edge["image_b"]
        if a in component and b in component:
            H = np.asarray(edge["homography"], dtype=np.float64)
            edge_map[(a, b)] = H
            edge_obj_map[(a, b)] = edge
            adj[a].add(b)
            try:
                H_inv = np.linalg.inv(H)
                edge_map[(b, a)] = H_inv
                edge_obj_map[(b, a)] = edge
                adj[b].add(a)
            except np.linalg.LinAlgError:
                pass

    # Find simple cycles of length 3 and 4
    discovered_cycles = []
    nodes = sorted(component)

    # 1. 3-cycles: a < b < c with all pairwise edges
    for i, a in enumerate(nodes):
        for b in sorted(adj[a]):
            if b <= a:
                continue
            for c in sorted(adj[b]):
                if c <= b:
                    continue
                if a in adj[c]:
                    discovered_cycles.append([a, b, c])

    # 2. 4-cycles (chords): a < b < c < d
    for i, a in enumerate(nodes):
        for b in sorted(adj[a]):
            if b <= a:
                continue
            for c in sorted(adj[b]):
                if c == a or c <= b:
                    continue
                for d in sorted(adj[c]):
                    if d in (a, b) or d <= c:
                        continue
                    if a in adj[d]:
                        discovered_cycles.append([a, b, c, d])

    evaluated_cycles = []
    cycle_errors = []
    suspect_edge_ids = set()

    for cycle_path in discovered_cycles:
        k = len(cycle_path)
        # Compose loop transformation: v0 -> v1 -> ... -> vk-1 -> v0
        H_loop = np.eye(3, dtype=np.float64)
        cycle_edges = []
        is_valid = True

        for idx in range(k):
            u = cycle_path[idx]
            v = cycle_path[(idx + 1) % k]
            if (u, v) not in edge_map:
                is_valid = False
                break
            H_loop = edge_map[(u, v)] @ H_loop
            cycle_edges.append(edge_obj_map[(u, v)]["edge_id"])

        if not is_valid or not np.isfinite(H_loop).all() or abs(H_loop[2, 2]) < 1e-12:
            continue

        H_loop = H_loop / H_loop[2, 2]

        # Evaluate corner transfer error on the root node of the cycle
        v0 = cycle_path[0]
        h, w = image_shapes[v0][:2]
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        try:
            warped_corners = cv2.perspectiveTransform(corners, H_loop.astype(np.float32))[0]
            orig_corners = corners[0]
            corner_error = float(np.mean(np.linalg.norm(warped_corners - orig_corners, axis=1)))
            matrix_deviation = float(np.linalg.norm(H_loop - np.eye(3), ord="fro"))
        except cv2.error:
            corner_error = 999.0
            matrix_deviation = 999.0

        is_consistent = corner_error <= tolerance_pixels
        cycle_errors.append(corner_error)

        if not is_consistent:
            for eid in cycle_edges:
                suspect_edge_ids.add(eid)

        evaluated_cycles.append({
            "path": cycle_path,
            "cycle_length": k,
            "edge_ids": cycle_edges,
            "corner_consistency_error_pixels": round(corner_error, 4),
            "matrix_deviation_frobenius": round(matrix_deviation, 4),
            "is_consistent": is_consistent,
            "status": "CONSISTENT" if is_consistent else "INCONSISTENT",
        })

    mean_err = float(np.mean(cycle_errors)) if cycle_errors else None
    max_err = float(np.max(cycle_errors)) if cycle_errors else None
    consistent_count = sum(c["is_consistent"] for c in evaluated_cycles)
    inconsistent_count = len(evaluated_cycles) - consistent_count

    return {
        "cycle_count": len(evaluated_cycles),
        "cycles": evaluated_cycles,
        "mean_cycle_error_pixels": round(mean_err, 4) if mean_err is not None else None,
        "max_cycle_error_pixels": round(max_err, 4) if max_err is not None else None,
        "consistent_cycles": consistent_count,
        "inconsistent_cycles": inconsistent_count,
        "suspect_edge_ids": sorted(suspect_edge_ids),
        "tolerance_pixels": tolerance_pixels,
        "graph_cycle_consistent": inconsistent_count == 0,
    }


def validate_propagated_transform(H: np.ndarray, image_shape: tuple[int, int]) -> tuple[bool, str]:
    """Validate that a propagated transformation is non-degenerate and physically valid."""
    if H is None or not np.isfinite(H).all():
        return False, "Transform contains NaN or infinite values."
    if abs(H[2, 2]) < 1e-12:
        return False, "Transform scale factor H[2, 2] is near zero."

    # Conditioning
    try:
        s = np.linalg.svd(H)[1]
        cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
        if cond > 1e7:
            return False, f"Transform is ill-conditioned (cond={cond:.1e})."
    except (ValueError, np.linalg.LinAlgError):
        return False, "SVD failed for transformation."

    # Orientation / determinant check of linear 2x2 part
    det = float(np.linalg.det(H[:2, :2]))
    if det <= 0.001 or det > 1000.0:
        return False, f"Transform linear determinant is degenerate or reversed (det={det:.4f})."

    # Corner bounding check
    h, w = image_shape[:2]
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    try:
        warped = cv2.perspectiveTransform(corners, H.astype(np.float32))[0]
        if not np.isfinite(warped).all():
            return False, "Transformed corners contain non-finite values."
        span_x = np.ptp(warped[:, 0])
        span_y = np.ptp(warped[:, 1])
        if span_x < 5.0 or span_y < 5.0 or span_x > 50000 or span_y > 50000:
            return False, f"Transformed bounds are unphysical (span=({span_x:.1f}, {span_y:.1f}))."
    except cv2.error as err:
        return False, f"Corner transform failed: {err}"

    return True, "Transform valid."


def place_largest_component(graph: dict, image_shapes: list[tuple]) -> dict:
    """Place the largest connected component in one relative image coordinate frame.

    Pipeline:
    1. Extract the largest connected component.
    2. Principled reference node selection (degree, centrality, quality).
    3. Maximum-confidence spanning tree traversal for mathematically verified transform propagation.
    4. Propagation validation (finite, invertible, conditioning, positive orientation).
    5. Non-tree edge consistency check and full cycle-consistency evaluation.
    """
    component = graph["components"][0] if graph["components"] else []
    accepted = [
        edge for edge in graph["edges"]
        if edge.get("accepted", False) and edge["image_a"] in component and edge["image_b"] in component
    ]

    if not component or len(accepted) == 0:
        return {
            "root": None,
            "selected_reference": None,
            "transforms": {},
            "tree_edge_ids": [],
            "consistency": [],
            "cycle_diagnostics": {"cycle_count": 0, "graph_cycle_consistent": True},
            "placed_nodes": [],
            "unplaced_nodes": graph["nodes"],
        }

    # 1. Select reference node
    root, ref_info = select_reference_node(component, accepted, image_shapes)

    # 2. Maximum Spanning Tree on edge confidence
    tree = _maximum_spanning_tree(component, accepted)
    tree_ids = {edge["edge_id"] for edge in tree}

    # Build bidirectional tree adjacency
    # edge homography H maps: image_a -> image_b (x_b ~ H * x_a)
    tree_adjacency = defaultdict(list)
    for edge in tree:
        H = np.asarray(edge["homography"], dtype=np.float64)
        tree_adjacency[edge["image_a"]].append((edge["image_b"], H))
        try:
            tree_adjacency[edge["image_b"]].append((edge["image_a"], np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            continue

    # 3. Propagate transformations from root
    # T_node maps: node -> root (x_root ~ T_node * x_node)
    transforms = {root: np.eye(3, dtype=np.float64)}
    queue = deque([root])
    failed_propagations = []

    while queue:
        known = queue.popleft()
        # For an adjacent node 'unknown', edge supplies H_known_to_unknown mapping known -> unknown.
        # Thus x_known ~ (H_known_to_unknown)^(-1) * x_unknown.
        # Since x_root ~ T_known * x_known, we have:
        # x_root ~ T_known * (H_known_to_unknown)^(-1) * x_unknown.
        # Therefore T_unknown = T_known * inv(H_known_to_unknown).
        for unknown, H_known_to_unknown in tree_adjacency[known]:
            if unknown not in transforms:
                try:
                    H_unknown_to_known = np.linalg.inv(H_known_to_unknown)
                    T_unknown = transforms[known] @ H_unknown_to_known
                    T_unknown /= T_unknown[2, 2]

                    # Validate propagated transformation
                    is_valid, reason = validate_propagated_transform(T_unknown, image_shapes[unknown])
                    if is_valid:
                        transforms[unknown] = T_unknown
                        queue.append(unknown)
                    else:
                        failed_propagations.append({"node": unknown, "reason": reason})
                except np.linalg.LinAlgError as err:
                    failed_propagations.append({"node": unknown, "reason": str(err)})

    # 4. Consistency of non-tree edges
    consistency = []
    for edge in accepted:
        if edge["edge_id"] in tree_ids or edge["image_a"] not in transforms or edge["image_b"] not in transforms:
            continue
        h, w = image_shapes[edge["image_a"]][:2]
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        # Direct propagation: A -> root via tree
        via_graph = cv2.perspectiveTransform(corners, transforms[edge["image_a"]].astype(np.float32))
        # Redundant path: A -> B via edge, then B -> root
        direct_to_root = transforms[edge["image_b"]] @ np.asarray(edge["homography"], dtype=np.float64)
        direct_to_root /= direct_to_root[2, 2]
        direct = cv2.perspectiveTransform(corners, direct_to_root.astype(np.float32))
        error = float(np.mean(np.linalg.norm(via_graph[0] - direct[0], axis=1)))
        consistency.append({
            "edge_id": edge["edge_id"],
            "image_a": edge["image_a"],
            "image_b": edge["image_b"],
            "corner_consistency_error_pixels": round(error, 4),
            "is_consistent": error <= 5.0,
        })

    # 5. Global Pose Graph Joint Optimization (drift reduction across redundant loops)
    optimized_transforms, opt_metrics = optimize_pose_graph(
        transforms, accepted, root, image_shapes, max_iters=6, relaxation_factor=0.25
    )
    transforms = optimized_transforms

    # 6. Full cycle consistency analysis
    cycle_diag = evaluate_cycle_consistency(component, accepted, image_shapes, tolerance_pixels=5.0)

    placed_nodes = sorted(transforms.keys())
    unplaced_nodes = sorted(set(graph["nodes"]) - set(placed_nodes))

    # 7. Explicit rejection reasons for unplaced images
    rejection_details = []
    for node in unplaced_nodes:
        if node in graph.get("isolated_nodes", []):
            rejection_details.append({
                "image_index": node,
                "status": "REJECTED",
                "reason": "Image isolated: no candidate edges satisfied geometric acceptance thresholds (minimum 6 inliers, 10% inlier ratio).",
            })
        else:
            fail_item = next((f for f in failed_propagations if f.get("node") == node), None)
            reason_str = fail_item["reason"] if fail_item else "Disconnected from primary connected component"
            rejection_details.append({
                "image_index": node,
                "status": "REJECTED",
                "reason": f"Geometric verification rejected: {reason_str}",
            })

    return {
        "root": root,
        "selected_reference": root,
        "reference_selection": ref_info,
        "transforms": {str(key): value.tolist() for key, value in transforms.items()},
        "tree_edge_ids": sorted(tree_ids),
        "consistency": consistency,
        "cycle_diagnostics": cycle_diag,
        "global_optimization": opt_metrics,
        "placed_nodes": placed_nodes,
        "unplaced_nodes": unplaced_nodes,
        "rejection_details": rejection_details,
        "failed_propagations": failed_propagations,
    }


def optimize_pose_graph(
    transforms: dict[int, np.ndarray],
    accepted_edges: list[dict],
    root: int,
    image_shapes: list[tuple],
    max_iters: int = 6,
    relaxation_factor: float = 0.25,
) -> tuple[dict[int, np.ndarray], dict]:
    """Iterative pose-graph relaxation to jointly optimize image poses and reduce accumulated chain drift."""
    if len(transforms) <= 2 or not accepted_edges:
        return transforms, {
            "initial_mean_residual_px": 0.0,
            "optimized_mean_residual_px": 0.0,
            "drift_reduction_px": 0.0,
            "iterations_performed": 0,
            "status": "SKIPPED_SMALL_GRAPH",
        }

    # Helper to compute edge corner transfer error
    def _compute_graph_residual(curr_t: dict[int, np.ndarray]) -> float:
        errors = []
        for e in accepted_edges:
            a, b = e["image_a"], e["image_b"]
            if a in curr_t and b in curr_t:
                h, w = image_shapes[a][:2]
                corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
                # Direct A -> root
                Ta = curr_t[a]
                pts_a = cv2.perspectiveTransform(corners, Ta.astype(np.float32))[0]
                # Via edge A -> B -> root
                H_ab = np.asarray(e["homography"], dtype=np.float64)
                Tb_Hab = curr_t[b] @ H_ab
                if abs(Tb_Hab[2, 2]) > 1e-12:
                    Tb_Hab /= Tb_Hab[2, 2]
                    pts_b = cv2.perspectiveTransform(corners, Tb_Hab.astype(np.float32))[0]
                    err = float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1)))
                    if np.isfinite(err):
                        errors.append(err)
        return float(np.mean(errors)) if errors else 0.0

    initial_residual = _compute_graph_residual(transforms)

    # Build incident edge observations for each node
    # Each observation is a predicted pose for node i from neighbor j
    # if (i, j) with H_ij: T_i ~ T_j * H_ij
    # if (j, i) with H_ji: T_i ~ T_j * inv(H_ji)
    incident_predictions: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for e in accepted_edges:
        a, b = e["image_a"], e["image_b"]
        conf = float(e.get("confidence", 1.0))
        H_ab = np.asarray(e["homography"], dtype=np.float64)
        incident_predictions[a].append((b, H_ab, conf))
        try:
            H_ba = np.linalg.inv(H_ab)
            incident_predictions[b].append((a, H_ba, conf))
        except np.linalg.LinAlgError:
            pass

    current_t = {k: np.copy(v) for k, v in transforms.items()}
    iters_done = 0

    for it in range(max_iters):
        updated_t = {k: np.copy(v) for k, v in current_t.items()}
        for node in current_t:
            if node == root:
                continue  # Root remains anchored at identity

            preds = incident_predictions.get(node, [])
            if not preds:
                continue

            # Weighted consensus of predicted poses
            sum_matrix = np.zeros((3, 3), dtype=np.float64)
            sum_weight = 0.0

            for nbr, H_node_to_nbr, conf in preds:
                if nbr in current_t:
                    T_pred = current_t[nbr] @ H_node_to_nbr
                    if abs(T_pred[2, 2]) > 1e-12 and np.isfinite(T_pred).all():
                        T_pred /= T_pred[2, 2]
                        sum_matrix += conf * T_pred
                        sum_weight += conf

            if sum_weight > 0:
                T_consensus = sum_matrix / sum_weight
                # Relaxation blend
                T_candidate = (1.0 - relaxation_factor) * current_t[node] + relaxation_factor * T_consensus
                if abs(T_candidate[2, 2]) > 1e-12:
                    T_candidate /= T_candidate[2, 2]
                    is_valid, _ = validate_propagated_transform(T_candidate, image_shapes[node])
                    if is_valid:
                        updated_t[node] = T_candidate

        # Test if update improved residual
        cand_residual = _compute_graph_residual(updated_t)
        if cand_residual <= initial_residual + 0.1:  # allow slight relaxation without blowup
            current_t = updated_t
            iters_done += 1
        else:
            break

    final_residual = _compute_graph_residual(current_t)
    drift_reduction = max(0.0, initial_residual - final_residual)

    return current_t, {
        "initial_mean_residual_px": round(initial_residual, 4),
        "optimized_mean_residual_px": round(final_residual, 4),
        "drift_reduction_px": round(drift_reduction, 4),
        "iterations_performed": iters_done,
        "status": "OPTIMIZED" if iters_done > 0 else "CONVERGED_AT_START",
    }
