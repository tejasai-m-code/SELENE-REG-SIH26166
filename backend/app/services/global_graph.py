# -*- coding: utf-8 -*-
"""
Phase 9 -- Global Registration Graph + Optimization Engine
SELENE-REG-X-FINAL-BUILD

Provides:
  - Rich graph node/edge representation with all pairwise evidence
  - Transparent edge validation (VALID / WEAK / REJECTED)
  - Transparent confidence/weight formula (documented below)
  - Connected-component analysis
  - Root selection by weighted graph degree
  - Initial global pose estimation by BFS tree traversal
  - Cycle-consistency checking (all triangles in the graph)
  - Weighted global pose optimization (scipy.optimize.minimize if available,
    otherwise weighted least-squares refinement using numpy -- documented)
  - Bad-edge robustness via iterative re-weighting
  - Full global metrics
  - Phase 7 subpixel acceptance integration
  - Per-component independent root selection (no forced cross-component bridges)

Transform convention (documented here and enforced throughout):
  T_i  : maps image-i pixels -> global/root coordinate frame
  T_ij : maps image-i pixels -> image-j pixels  (pairwise result)

  Composition rule:
    T_j  =  T_ij  @  T_i          (standard camera/warp chain)
    T_i  =  inv(T_ij)  @  T_j

  The root image r has T_r = I (identity).
  BFS propagates:  T_child = T_edge  @  T_parent
  where T_edge is oriented source->reference; orientation is corrected per edge direction.

Cycle-consistency check:
  For a closed triangle A->B->C->A:
    T_AB @ T_BC @ T_CA  should be ~identity
  Measure as:
    * corner_error: mean pixel deviation when composing the cycle around four image corners
    * matrix_frob:  ||cycle_product - I||_F

Confidence/weight formula (fully transparent):
  edge_weight = inlier_count * inlier_ratio * (0.25 + spatial_coverage)
  if Phase7 accepted:  edge_weight *= 1.25   (boost for subpixel acceptance)
  if model == 'translation': edge_weight *= 0.90  (slight penalty: fewer DOF verified)
  Clipped to [0, inf).  Higher = more trustworthy.

Global optimization:
  Given root pose T_r=I and initial BFS poses T_i, minimize:
    sum_over_valid_edges  w_ij * |T_j - T_ij @ T_i|^2   (Frobenius distance on 3x3 H)
  Using scipy.optimize.minimize (L-BFGS-B) if scipy is available.
  Otherwise uses one pass of weighted averaging (documented as fallback).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy import -- fallback documented if unavailable
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ===========================================================================
# DATA CLASSES
# ===========================================================================

@dataclass
class GraphNode:
    image_id: int
    filename: str
    sensor: str
    width: int
    height: int
    status: str = "placed"       # placed | isolated | disconnected


@dataclass
class GraphEdge:
    edge_id: int
    source_id: int
    reference_id: int
    # Transform: maps source pixels -> reference pixels
    transform: np.ndarray                        # 3x3 float64
    geometric_model: str
    estimator: str
    match_count: int
    inlier_count: int
    inlier_ratio: float
    rmse_pixels: Optional[float]
    median_residual: Optional[float]
    robust_residual: Optional[float]
    spatial_coverage: float
    pairwise_confidence: float                   # raw from confidence_for_metrics
    phase7_status: str                           # ACCEPTED | REJECTED | NOT_RUN
    phase7_accepted: bool
    subpixel_dx: Optional[float]
    subpixel_dy: Optional[float]
    # Validation result
    validation_status: str = "VALID"             # VALID | WEAK | REJECTED
    rejection_reason: Optional[str] = None
    # Optimization weight (derived from pairwise_confidence + Phase7 boost)
    edge_weight: float = 0.0
    # Provenance
    representation_source: Optional[str] = None
    representation_reference: Optional[str] = None


@dataclass
class ComponentInfo:
    component_id: int
    node_ids: list
    edge_ids: list
    root_node: int
    root_reason: str
    connected: bool


@dataclass
class CycleDiagnostic:
    nodes: list          # [a, b, c]
    cycle_length: int
    corner_error_pixels: float
    matrix_frob_error: float
    threshold_pixels: float
    consistent: bool
    weak_edge_id: Optional[int]


@dataclass
class OptimizationResult:
    method: str           # "scipy_lbfgsb" | "weighted_ls_fallback"
    status: str           # "converged" | "max_iter" | "fallback"
    iterations: int
    objective_before: float
    objective_after: float
    converged: bool
    message: str


@dataclass
class GlobalGraphResult:
    nodes: list                     # list[GraphNode]
    edges: list                     # list[GraphEdge]
    components: list                # list[ComponentInfo]
    cycle_diagnostics: list         # list[CycleDiagnostic]
    global_poses: dict              # {image_id: 3x3 np.ndarray}
    optimization: OptimizationResult
    global_metrics: dict


# ===========================================================================
# EDGE WEIGHT FORMULA
# ===========================================================================

def compute_edge_weight(
    inlier_count: int,
    inlier_ratio: float,
    spatial_coverage: float,
    phase7_accepted: bool,
    geometric_model: str,
) -> float:
    """
    Fully transparent edge weight formula.

    Base:  inlier_count * inlier_ratio * (0.25 + spatial_coverage)
    Phase7 boost: *1.25 if accepted
    Translation penalty: *0.90  (fewer DOF verified than homography)

    Returns float >= 0.
    """
    base = float(inlier_count) * float(inlier_ratio) * (0.25 + float(spatial_coverage))
    if phase7_accepted:
        base *= 1.25
    if geometric_model and geometric_model.lower() == "translation":
        base *= 0.90
    return max(0.0, base)


# ===========================================================================
# EDGE VALIDATION POLICY
# ===========================================================================

VALIDATION_POLICY = {
    "min_inliers_valid":        8,
    "min_inlier_ratio_valid":   0.12,
    "max_rmse_valid":           6.0,
    "min_spatial_coverage_valid": 0.10,
    "min_inliers_weak":         4,
    "min_inlier_ratio_weak":    0.06,
    "max_rmse_weak":            12.0,
    "min_spatial_coverage_weak": 0.04,
}


def validate_edge(
    inlier_count: int,
    inlier_ratio: float,
    rmse_pixels: Optional[float],
    spatial_coverage: float,
    geometric_model: str,
) -> tuple[str, Optional[str]]:
    """
    Returns (status, reason) where status in {VALID, WEAK, REJECTED}.
    """
    p = VALIDATION_POLICY

    # Hard rejection checks
    if inlier_count < p["min_inliers_weak"]:
        return "REJECTED", f"inlier_count={inlier_count} < min={p['min_inliers_weak']}"
    if inlier_ratio < p["min_inlier_ratio_weak"]:
        return "REJECTED", f"inlier_ratio={inlier_ratio:.3f} < min={p['min_inlier_ratio_weak']}"
    if rmse_pixels is not None and rmse_pixels > p["max_rmse_weak"]:
        return "REJECTED", f"rmse={rmse_pixels:.2f}px > max={p['max_rmse_weak']}px"
    if spatial_coverage < p["min_spatial_coverage_weak"]:
        return "REJECTED", f"spatial_coverage={spatial_coverage:.3f} < min={p['min_spatial_coverage_weak']}"
    if geometric_model and geometric_model.lower() == "degenerate":
        return "REJECTED", "geometric_model=DEGENERATE"

    # VALID checks
    rmse_ok = (rmse_pixels is None) or (rmse_pixels <= p["max_rmse_valid"])
    if (inlier_count >= p["min_inliers_valid"] and
            inlier_ratio >= p["min_inlier_ratio_valid"] and
            rmse_ok and
            spatial_coverage >= p["min_spatial_coverage_valid"]):
        return "VALID", None

    # Otherwise WEAK
    reasons = []
    if inlier_count < p["min_inliers_valid"]:
        reasons.append(f"low_inliers={inlier_count}")
    if inlier_ratio < p["min_inlier_ratio_valid"]:
        reasons.append(f"low_ratio={inlier_ratio:.3f}")
    if not rmse_ok:
        reasons.append(f"high_rmse={rmse_pixels:.2f}px")
    if spatial_coverage < p["min_spatial_coverage_valid"]:
        reasons.append(f"low_coverage={spatial_coverage:.3f}")
    return "WEAK", "; ".join(reasons)


# ===========================================================================
# BUILD RICH GRAPH FROM PAIRWISE RESULTS
# ===========================================================================

def build_global_graph(
    nodes_info: list[dict],
    pairwise_edges: list[dict],
    image_shapes: list[tuple],
    cycle_threshold_pixels: float = 4.0,
) -> GlobalGraphResult:
    """
    Build a full Phase 9 global registration graph from validated pairwise results.

    Parameters
    ----------
    nodes_info : list of dicts with keys: image_id, filename, sensor, width, height
    pairwise_edges : list of dicts -- the raw edges from multi_registration run,
                     each containing: image_a, image_b, accepted, metrics, homography,
                     confidence, status, reason, ecc_used, subpixel_consensus (optional),
                     geometric_model (optional), estimator (optional)
    image_shapes : list of (H, W[, C]) tuples matching node order
    cycle_threshold_pixels : corner error below which a cycle is CONSISTENT
    """
    n_images = len(nodes_info)

    # ---- Build node list -----------------------------------------------
    nodes = [
        GraphNode(
            image_id=nd["image_id"],
            filename=nd.get("filename", f"image_{nd['image_id']}"),
            sensor=nd.get("sensor", "Unknown"),
            width=nd.get("width", 0),
            height=nd.get("height", 0),
            status="placed",
        )
        for nd in nodes_info
    ]

    # ---- Build edge list with full diagnostics -------------------------
    edges: list[GraphEdge] = []
    for raw in pairwise_edges:
        if not raw.get("accepted", False):
            # Still include as rejected for transparency
            edge = GraphEdge(
                edge_id=raw.get("edge_id", len(edges)),
                source_id=raw["image_a"],
                reference_id=raw["image_b"],
                transform=np.eye(3, dtype=np.float64),
                geometric_model=raw.get("transformation_type", "unknown"),
                estimator=raw.get("estimator", "UNAVAILABLE"),
                match_count=raw.get("metrics", {}).get("match_count", 0) if raw.get("metrics") else 0,
                inlier_count=raw.get("metrics", {}).get("inlier_count", 0) if raw.get("metrics") else 0,
                inlier_ratio=raw.get("metrics", {}).get("inlier_ratio", 0.0) if raw.get("metrics") else 0.0,
                rmse_pixels=raw.get("metrics", {}).get("rmse_pixels") if raw.get("metrics") else None,
                median_residual=raw.get("metrics", {}).get("median_reprojection_error_pixels") if raw.get("metrics") else None,
                robust_residual=None,
                spatial_coverage=raw.get("metrics", {}).get("source_spatial_coverage", 0.0) if raw.get("metrics") else 0.0,
                pairwise_confidence=raw.get("confidence", 0.0),
                phase7_status="NOT_RUN",
                phase7_accepted=False,
                subpixel_dx=None,
                subpixel_dy=None,
                validation_status="REJECTED",
                rejection_reason=raw.get("reason", "pairwise registration rejected"),
                edge_weight=0.0,
            )
            edges.append(edge)
            continue

        m = raw.get("metrics", {}) or {}
        H_raw = raw.get("homography")
        if H_raw is None:
            continue
        H = np.asarray(H_raw, dtype=np.float64)

        inlier_count = int(m.get("inlier_count", 0))
        inlier_ratio = float(m.get("inlier_ratio", 0.0))
        rmse = m.get("rmse_pixels")
        if rmse is not None:
            rmse = float(rmse)
        spatial_cov = float(m.get("source_spatial_coverage", 0.0))
        geo_model = raw.get("transformation_type", "homography")

        # Phase 7 integration
        sub = raw.get("subpixel_consensus") or {}
        if sub and sub.get("accepted"):
            phase7_status = "ACCEPTED"
            phase7_accepted = True
            sub_dx = sub.get("dx")
            sub_dy = sub.get("dy")
            # Apply subpixel correction to H if accepted
            dx = float(sub_dx) if sub_dx is not None else 0.0
            dy = float(sub_dy) if sub_dy is not None else 0.0
            T_sub = np.eye(3, dtype=np.float64)
            T_sub[0, 2] = dx
            T_sub[1, 2] = dy
            H = T_sub @ H   # refine translation component
            H /= H[2, 2]
        elif sub:
            phase7_status = "REJECTED"
            phase7_accepted = False
            sub_dx = sub.get("dx")
            sub_dy = sub.get("dy")
        else:
            phase7_status = "NOT_RUN"
            phase7_accepted = False
            sub_dx = None
            sub_dy = None

        val_status, val_reason = validate_edge(
            inlier_count, inlier_ratio, rmse, spatial_cov, geo_model
        )
        weight = compute_edge_weight(
            inlier_count, inlier_ratio, spatial_cov, phase7_accepted, geo_model
        )
        if val_status == "REJECTED":
            weight = 0.0

        edge = GraphEdge(
            edge_id=raw.get("edge_id", len(edges)),
            source_id=raw["image_a"],
            reference_id=raw["image_b"],
            transform=H,
            geometric_model=geo_model,
            estimator=raw.get("estimator", "UNAVAILABLE"),
            match_count=int(m.get("match_count", inlier_count)),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_pixels=rmse,
            median_residual=m.get("median_reprojection_error_pixels"),
            robust_residual=None,
            spatial_coverage=spatial_cov,
            pairwise_confidence=float(raw.get("confidence", 0.0)),
            phase7_status=phase7_status,
            phase7_accepted=phase7_accepted,
            subpixel_dx=sub_dx,
            subpixel_dy=sub_dy,
            validation_status=val_status,
            rejection_reason=val_reason,
            edge_weight=weight,
        )
        edges.append(edge)

    # ---- Connected components (on VALID + WEAK edges) ------------------
    trusted_edges = [e for e in edges if e.validation_status in ("VALID", "WEAK")]
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)   # node -> [(neighbor, edge_idx)]
    for ei, edge in enumerate(edges):
        if edge.validation_status not in ("VALID", "WEAK"):
            continue
        adj[edge.source_id].append((edge.reference_id, ei))
        adj[edge.reference_id].append((edge.source_id, ei))

    visited = set()
    raw_components: list[list[int]] = []
    for node_id in range(n_images):
        if node_id in visited:
            continue
        comp: list[int] = []
        q: deque[int] = deque([node_id])
        visited.add(node_id)
        while q:
            n = q.popleft()
            comp.append(n)
            for neighbor, _ in adj[n]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        raw_components.append(sorted(comp))
    raw_components.sort(key=len, reverse=True)

    # Mark isolated nodes
    for node in nodes:
        if not adj[node.image_id]:
            node.status = "isolated"
        elif all(len(c) == 1 for c in raw_components if node.image_id in c):
            node.status = "isolated"

    # ---- Root selection per component ----------------------------------
    components: list[ComponentInfo] = []
    for cid, comp_nodes in enumerate(raw_components):
        comp_set = set(comp_nodes)
        comp_edge_ids = [
            ei for ei, edge in enumerate(edges)
            if edge.source_id in comp_set and edge.reference_id in comp_set
            and edge.validation_status in ("VALID", "WEAK")
        ]

        # Weighted degree per node within component
        w_degree: dict[int, float] = defaultdict(float)
        for ei in comp_edge_ids:
            e = edges[ei]
            w_degree[e.source_id] += e.edge_weight
            w_degree[e.reference_id] += e.edge_weight

        if comp_nodes:
            root = max(comp_nodes, key=lambda n: (w_degree.get(n, 0.0), -n))
            root_reason = (
                f"highest weighted degree={w_degree.get(root, 0.0):.3f} "
                f"among {len(comp_nodes)} nodes in component {cid}"
            )
        else:
            root = comp_nodes[0]
            root_reason = "single node"

        components.append(ComponentInfo(
            component_id=cid,
            node_ids=comp_nodes,
            edge_ids=comp_edge_ids,
            root_node=root,
            root_reason=root_reason,
            connected=len(comp_nodes) > 1,
        ))

    # ---- Initial global poses (BFS per component) ----------------------
    global_poses: dict[int, np.ndarray] = {}

    for comp in components:
        root = comp.root_node
        global_poses[root] = np.eye(3, dtype=np.float64)
        comp_set = set(comp.node_ids)

        # Build adjacency for BFS
        bfs_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ei in comp.edge_ids:
            e = edges[ei]
            bfs_adj[e.source_id].append((e.reference_id, ei))
            bfs_adj[e.reference_id].append((e.source_id, ei))

        # BFS: T_child = T_parent @ edge_transform  (oriented correctly)
        bfs_q: deque[int] = deque([root])
        bfs_visited = {root}
        while bfs_q:
            parent = bfs_q.popleft()
            T_parent = global_poses[parent]
            for neighbor, ei in bfs_adj[parent]:
                if neighbor in bfs_visited:
                    continue
                bfs_visited.add(neighbor)
                e = edges[ei]
                # Orient: if edge maps parent->neighbor, T_neighbor = T_edge @ T_parent
                #         if edge maps neighbor->parent, T_neighbor = inv(T_edge) @ T_parent
                if e.source_id == parent and e.reference_id == neighbor:
                    T_edge = e.transform                             # parent->neighbor
                    T_neighbor = T_edge @ T_parent
                else:
                    try:
                        T_edge_inv = np.linalg.inv(e.transform)     # neighbor->parent
                        T_neighbor = T_edge_inv @ T_parent
                    except np.linalg.LinAlgError:
                        T_neighbor = T_parent.copy()
                # Normalize projective scale
                if abs(T_neighbor[2, 2]) > 1e-12:
                    T_neighbor = T_neighbor / T_neighbor[2, 2]
                global_poses[neighbor] = T_neighbor
                bfs_q.append(neighbor)

    # ---- Cycle-consistency check ---------------------------------------
    cycle_diagnostics: list[CycleDiagnostic] = []

    # Find all triangles in trusted_edges graph
    edge_index_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ei, edge in enumerate(edges):
        if edge.validation_status not in ("VALID", "WEAK"):
            continue
        edge_index_map[(edge.source_id, edge.reference_id)].append(ei)
        edge_index_map[(edge.reference_id, edge.source_id)].append(ei)

    triangle_seen: set[frozenset] = set()
    for a_node in range(n_images):
        for b_node, ei_ab in adj[a_node]:
            for c_node, ei_bc in adj[b_node]:
                if c_node == a_node:
                    continue
                if (a_node, c_node) not in edge_index_map and (c_node, a_node) not in edge_index_map:
                    continue
                key = frozenset([a_node, b_node, c_node])
                if key in triangle_seen:
                    continue
                triangle_seen.add(key)

                # Get all three edge transforms
                ei_ca_list = edge_index_map.get((c_node, a_node), edge_index_map.get((a_node, c_node), []))
                if not ei_ca_list:
                    continue

                e_ab = edges[ei_ab]
                e_bc = edges[ei_bc]
                e_ca = edges[ei_ca_list[0]]

                # T_AB: a->b, T_BC: b->c, T_CA: c->a
                T_AB = _get_oriented(e_ab, a_node, b_node)
                T_BC = _get_oriented(e_bc, b_node, c_node)
                T_CA = _get_oriented(e_ca, c_node, a_node)

                if T_AB is None or T_BC is None or T_CA is None:
                    continue

                cycle = T_CA @ T_BC @ T_AB
                # Normalize
                if abs(cycle[2, 2]) > 1e-12:
                    cycle = cycle / cycle[2, 2]

                frob_err = float(np.linalg.norm(cycle - np.eye(3)))

                # Corner consistency test
                h_a, w_a = image_shapes[a_node][:2]
                corners = np.float32([[[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]])
                import cv2
                try:
                    corners_transformed = cv2.perspectiveTransform(corners, cycle)
                    corner_err = float(np.mean(np.linalg.norm(
                        corners[0] - corners_transformed[0], axis=1
                    )))
                except Exception:
                    corner_err = frob_err * 10.0

                consistent = corner_err < cycle_threshold_pixels

                # Identify weak edge if inconsistent
                weak_edge_id = None
                if not consistent:
                    # Weakest edge (lowest weight) is most likely culprit
                    candidate_edges = [e_ab, e_bc, e_ca]
                    weak_edge = min(candidate_edges, key=lambda e: e.edge_weight)
                    weak_edge_id = weak_edge.edge_id
                    # Down-weight the weak edge
                    weak_edge.edge_weight *= 0.5
                    if corner_err > 3 * cycle_threshold_pixels:
                        weak_edge.validation_status = "WEAK"
                        if weak_edge.rejection_reason:
                            weak_edge.rejection_reason += "; cycle_inconsistent"
                        else:
                            weak_edge.rejection_reason = "cycle_inconsistent"

                cycle_diagnostics.append(CycleDiagnostic(
                    nodes=[a_node, b_node, c_node],
                    cycle_length=3,
                    corner_error_pixels=corner_err,
                    matrix_frob_error=frob_err,
                    threshold_pixels=cycle_threshold_pixels,
                    consistent=consistent,
                    weak_edge_id=weak_edge_id,
                ))

    # ---- Global pose optimization -------------------------------------
    opt_result = _optimize_global_poses(
        nodes=list(range(n_images)),
        edges=edges,
        global_poses=global_poses,
        components=components,
    )

    # ---- Global metrics -----------------------------------------------
    valid_edges = [e for e in edges if e.validation_status == "VALID"]
    weak_edges  = [e for e in edges if e.validation_status == "WEAK"]
    rej_edges   = [e for e in edges if e.validation_status == "REJECTED"]
    all_rmse    = [e.rmse_pixels for e in valid_edges + weak_edges if e.rmse_pixels is not None]

    global_metrics = {
        "image_count": n_images,
        "edge_count": len(edges),
        "valid_edges": len(valid_edges),
        "weak_edges": len(weak_edges),
        "rejected_edges": len(rej_edges),
        "component_count": len(components),
        "cycle_count": len(cycle_diagnostics),
        "inconsistent_cycle_count": sum(1 for c in cycle_diagnostics if not c.consistent),
        "placed_image_count": len(global_poses),
        "isolated_image_count": n_images - len(global_poses),
        "graph_connected": len(components) == 1 and len(components[0].node_ids) == n_images,
        "mean_edge_rmse": float(np.mean(all_rmse)) if all_rmse else None,
        "median_edge_rmse": float(np.median(all_rmse)) if all_rmse else None,
        "max_edge_rmse": float(np.max(all_rmse)) if all_rmse else None,
        "optimization_method": opt_result.method,
        "optimization_converged": opt_result.converged,
        "validation_policy": VALIDATION_POLICY,
        "weight_formula": (
            "base = inlier_count * inlier_ratio * (0.25 + spatial_coverage); "
            "if phase7_accepted: base *= 1.25; "
            "if model=='translation': base *= 0.90"
        ),
        "scipy_available": SCIPY_AVAILABLE,
    }

    return GlobalGraphResult(
        nodes=nodes,
        edges=edges,
        components=components,
        cycle_diagnostics=cycle_diagnostics,
        global_poses=global_poses,
        optimization=opt_result,
        global_metrics=global_metrics,
    )


def _get_oriented(edge: GraphEdge, from_id: int, to_id: int) -> Optional[np.ndarray]:
    """Return edge transform oriented from_id -> to_id."""
    if edge.source_id == from_id and edge.reference_id == to_id:
        return edge.transform.copy()
    elif edge.source_id == to_id and edge.reference_id == from_id:
        try:
            inv = np.linalg.inv(edge.transform)
            return inv / inv[2, 2]
        except np.linalg.LinAlgError:
            return None
    return None


# ===========================================================================
# GLOBAL POSE OPTIMIZATION
# ===========================================================================

def _h_to_params(H: np.ndarray) -> np.ndarray:
    """Flatten H to 8 free parameters (H[2,2]=1 fixed)."""
    h = H.copy().ravel()
    return h[:8]  # drop H[2,2] which is fixed=1


def _params_to_h(params: np.ndarray) -> np.ndarray:
    H = np.zeros(9, dtype=np.float64)
    H[:8] = params
    H[8] = 1.0
    return H.reshape(3, 3)


def _optimization_objective(
    params_flat: np.ndarray,
    free_ids: list,
    root: int,
    root_pose: np.ndarray,
    edge_list: list,  # (i, j, T_ij, weight)
) -> float:
    """
    Minimize sum_edges  w * ||T_j - T_ij @ T_i||_F^2

    params_flat contains 8 * len(free_ids) concatenated homography params.
    """
    poses: dict[int, np.ndarray] = {root: root_pose}
    for k, img_id in enumerate(free_ids):
        H = np.zeros(9, dtype=np.float64)
        H[:8] = params_flat[k*8:(k+1)*8]
        H[8] = 1.0
        poses[img_id] = H.reshape(3, 3)

    total = 0.0
    for (i, j, T_ij, w) in edge_list:
        if i not in poses or j not in poses:
            continue
        T_i = poses[i]
        T_j = poses[j]
        T_ij_Ti = T_ij @ T_i
        s = T_ij_Ti[2, 2]
        if abs(s) > 1e-12:
            T_ij_Ti = T_ij_Ti / s
        diff = T_j - T_ij_Ti
        total += w * float(np.sum(diff**2))
    return total



def _optimize_global_poses(
    nodes: list[int],
    edges: list[GraphEdge],
    global_poses: dict[int, np.ndarray],
    components: list[ComponentInfo],
) -> OptimizationResult:
    """
    Run weighted global pose optimization.
    Uses scipy L-BFGS-B if available, else weighted-LS averaging (documented fallback).
    Optimizes each connected component independently.
    """
    if not global_poses:
        return OptimizationResult(
            method="none", status="no_poses", iterations=0,
            objective_before=0.0, objective_after=0.0,
            converged=True, message="No poses to optimize"
        )

    # Build edge list for trusted edges
    trusted_edges: list[tuple[int, int, np.ndarray, float]] = []
    for e in edges:
        if e.validation_status not in ("VALID", "WEAK") or e.edge_weight <= 0:
            continue
        if e.source_id not in global_poses or e.reference_id not in global_poses:
            continue
        trusted_edges.append((e.source_id, e.reference_id, e.transform, e.edge_weight))

    if not trusted_edges:
        return OptimizationResult(
            method="none", status="no_trusted_edges", iterations=0,
            objective_before=0.0, objective_after=0.0,
            converged=True, message="No trusted edges for optimization"
        )

    if not SCIPY_AVAILABLE:
        return _weighted_ls_fallback(nodes, edges, global_poses, trusted_edges)

    # ---- scipy L-BFGS-B per component --------------------------------
    total_iter = 0
    total_obj_before = 0.0
    total_obj_after  = 0.0
    all_converged = True
    messages = []

    for comp in components:
        if len(comp.node_ids) < 2:
            continue
        root = comp.root_node
        root_pose = global_poses.get(root, np.eye(3, dtype=np.float64))
        free_ids = [i for i in comp.node_ids if i != root]
        if not free_ids:
            continue

        comp_set = set(comp.node_ids)
        comp_edges = [(i, j, T, w) for (i, j, T, w) in trusted_edges
                      if i in comp_set and j in comp_set]
        if not comp_edges:
            continue

        x0 = np.concatenate([_h_to_params(global_poses.get(fid, np.eye(3))) for fid in free_ids])
        obj_before = _optimization_objective(x0, free_ids, root, root_pose, comp_edges)

        try:
            res = scipy_minimize(
                _optimization_objective,
                x0,
                args=(free_ids, root, root_pose, comp_edges),
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-9},
            )
            for k, img_id in enumerate(free_ids):
                H_opt = _params_to_h(res.x[k*8:(k+1)*8])
                if abs(H_opt[2, 2]) > 1e-12:
                    H_opt /= H_opt[2, 2]
                global_poses[img_id] = H_opt

            total_iter += res.nit
            total_obj_before += obj_before
            total_obj_after  += res.fun
            if not res.success:
                all_converged = False
            messages.append(res.message if hasattr(res.message, '__len__') else str(res.message))


        except Exception as exc:
            messages.append(f"comp_{comp.component_id}: {exc}")
            all_converged = False
            total_obj_before += obj_before
            total_obj_after  += obj_before

    return OptimizationResult(
        method="scipy_lbfgsb",
        status="converged" if all_converged else "partial",
        iterations=total_iter,
        objective_before=total_obj_before,
        objective_after=total_obj_after,
        converged=all_converged,
        message="; ".join(messages)[:500],
    )


def _weighted_ls_fallback(
    nodes: list[int],
    edges: list[GraphEdge],
    global_poses: dict[int, np.ndarray],
    trusted_edges: list[tuple[int, int, np.ndarray, float]],
) -> OptimizationResult:
    """
    Fallback when scipy unavailable:
    For each non-root image, compute weighted average of implied poses from each incident edge.
    This is NOT equivalent to scipy global optimization but is a principled improvement
    over pure BFS tree without using non-tree edges.

    Explicitly documented as: FEATURE-LEVEL BUNDLE ADJUSTMENT: NOT AVAILABLE
                              GLOBAL POSE OPTIMIZATION: WEIGHTED-LS FALLBACK
    """
    obj_before = 0.0
    for (i, j, T_ij, w) in trusted_edges:
        if i not in global_poses or j not in global_poses:
            continue
        T_pred = T_ij @ global_poses[i]
        s = T_pred[2, 2]
        if abs(s) > 1e-12:
            T_pred /= s
        diff = global_poses[j] - T_pred
        obj_before += w * float(np.sum(diff**2))

    # One pass: for each non-root, collect all implied T from incident edges, weight-average
    # Gather implied poses
    implied: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for (i, j, T_ij, w) in trusted_edges:
        if i in global_poses and j in global_poses:
            T_j_from_i = T_ij @ global_poses[i]
            s = T_j_from_i[2, 2]
            if abs(s) > 1e-12:
                T_j_from_i /= s
            implied[j].append((T_j_from_i, w))

    for img_id, preds in implied.items():
        if img_id not in global_poses:
            continue
        total_w = sum(w for _, w in preds)
        if total_w < 1e-12:
            continue
        avg_H = sum(H * w for H, w in preds) / total_w
        s = avg_H[2, 2]
        if abs(s) > 1e-12:
            avg_H /= s
        global_poses[img_id] = avg_H

    obj_after = 0.0
    for (i, j, T_ij, w) in trusted_edges:
        if i not in global_poses or j not in global_poses:
            continue
        T_pred = T_ij @ global_poses[i]
        s = T_pred[2, 2]
        if abs(s) > 1e-12:
            T_pred /= s
        diff = global_poses[j] - T_pred
        obj_after += w * float(np.sum(diff**2))

    return OptimizationResult(
        method="weighted_ls_fallback",
        status="fallback",
        iterations=1,
        objective_before=obj_before,
        objective_after=obj_after,
        converged=True,
        message=(
            "scipy not available. Used weighted-LS pose averaging. "
            "FEATURE-LEVEL BUNDLE ADJUSTMENT: NOT AVAILABLE. "
            "GLOBAL POSE OPTIMIZATION: WEIGHTED-LS FALLBACK IMPLEMENTED."
        ),
    )


# ===========================================================================
# SERIALIZATION HELPERS
# ===========================================================================

def serialize_global_result(result: GlobalGraphResult) -> dict:
    """Convert GlobalGraphResult to a JSON-serializable dict."""

    def _h(arr: np.ndarray) -> list:
        return arr.tolist()

    return {
        "nodes": [
            {
                "image_id": n.image_id,
                "filename": n.filename,
                "sensor": n.sensor,
                "width": n.width,
                "height": n.height,
                "status": n.status,
            }
            for n in result.nodes
        ],
        "edges": [
            {
                "edge_id": e.edge_id,
                "source_id": e.source_id,
                "reference_id": e.reference_id,
                "geometric_model": e.geometric_model,
                "estimator": e.estimator,
                "match_count": e.match_count,
                "inlier_count": e.inlier_count,
                "inlier_ratio": round(e.inlier_ratio, 4),
                "rmse_pixels": round(e.rmse_pixels, 4) if e.rmse_pixels is not None else None,
                "median_residual": round(e.median_residual, 4) if e.median_residual is not None else None,
                "spatial_coverage": round(e.spatial_coverage, 4),
                "pairwise_confidence": round(e.pairwise_confidence, 4),
                "phase7_status": e.phase7_status,
                "phase7_accepted": e.phase7_accepted,
                "subpixel_dx": round(e.subpixel_dx, 4) if e.subpixel_dx is not None else None,
                "subpixel_dy": round(e.subpixel_dy, 4) if e.subpixel_dy is not None else None,
                "validation_status": e.validation_status,
                "rejection_reason": e.rejection_reason,
                "edge_weight": round(e.edge_weight, 4),
                "transform": _h(e.transform),
            }
            for e in result.edges
        ],
        "connected_components": [
            {
                "component_id": c.component_id,
                "node_ids": c.node_ids,
                "edge_ids": c.edge_ids,
                "root_node": c.root_node,
                "root_reason": c.root_reason,
                "connected": c.connected,
                "node_count": len(c.node_ids),
                "edge_count": len(c.edge_ids),
            }
            for c in result.components
        ],
        "root_selection": {
            c.component_id: {
                "root": c.root_node,
                "reason": c.root_reason,
            }
            for c in result.components
        },
        "cycle_diagnostics": [
            {
                "nodes": cd.nodes,
                "cycle_length": cd.cycle_length,
                "corner_error_pixels": round(cd.corner_error_pixels, 4),
                "matrix_frob_error": round(cd.matrix_frob_error, 4),
                "threshold_pixels": cd.threshold_pixels,
                "consistent": cd.consistent,
                "weak_edge_id": cd.weak_edge_id,
            }
            for cd in result.cycle_diagnostics
        ],
        "optimization": {
            "method": result.optimization.method,
            "status": result.optimization.status,
            "iterations": result.optimization.iterations,
            "objective_before": round(result.optimization.objective_before, 6),
            "objective_after": round(result.optimization.objective_after, 6),
            "converged": result.optimization.converged,
            "message": result.optimization.message,
            "scipy_available": SCIPY_AVAILABLE,
            # optimization_type: exactly what is minimized and how.
            # "weighted_pose_graph": minimizes sum_edges w_ij * ||T_j - T_ij @ T_i||_F^2
            # over 3x3 homography matrices (H[2,2]=1 fixed) using scipy L-BFGS-B per component.
            # This is NOT feature-level reprojection-error bundle adjustment.
            "optimization_type": "weighted_pose_graph",
            "objective_description": (
                "Minimize sum_edges w_ij * ||T_j - T_ij @ T_i||_F^2  "
                "(Frobenius disagreement in matrix space, NOT feature reprojection error)"
            ),
            "bundle_adjustment_status": "NOT_IMPLEMENTED",
            "bundle_adjustment_note": (
                "Feature-level reprojection-error bundle adjustment is NOT implemented. "
                "The implemented optimizer is a weighted pose-graph optimizer that minimizes "
                "Frobenius matrix disagreement between globally estimated poses and pairwise "
                "edge constraints. This is a valid global consistency optimization but does "
                "not refine individual feature correspondences. "
                "Feature-level BA is a planned future enhancement."
            ),
        },

        "global_poses": {
            str(img_id): _h(pose)
            for img_id, pose in result.global_poses.items()
        },
        "global_metrics": result.global_metrics,
    }
