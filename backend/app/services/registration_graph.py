"""Graph construction and conservative image-space placement utilities."""

from collections import defaultdict, deque

import cv2
import numpy as np


def confidence_for_metrics(metrics: dict) -> float:
    """A transparent ranking score for graph edges, not a scientific benchmark."""
    return float(metrics["inlier_count"]) * float(metrics["inlier_ratio"]) * (
        0.25 + float(metrics["source_spatial_coverage"])
    )


def build_graph(image_count: int, edges: list[dict]) -> dict:
    adjacency = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        if not edge["accepted"]:
            continue
        a, b = edge["image_a"], edge["image_b"]
        adjacency[a].append((b, edge_index))
        adjacency[b].append((a, edge_index))

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

    components.sort(key=len, reverse=True)
    return {
        "nodes": list(range(image_count)),
        "edges": edges,
        "components": components,
        "connected": len(components) == 1,
        "isolated_nodes": sorted(node for node in range(image_count) if not adjacency[node]),
    }


def _maximum_spanning_tree(nodes: list[int], edges: list[dict]) -> list[dict]:
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected = []
    for edge in sorted(edges, key=lambda item: item["confidence"], reverse=True):
        a, b = find(edge["image_a"]), find(edge["image_b"])
        if a != b:
            parent[a] = b
            selected.append(edge)
    return selected


def place_largest_component(graph: dict, image_shapes: list[tuple]) -> dict:
    """Place the largest connected component in one relative image coordinate frame.

    A maximum-confidence spanning tree supplies stable initial transforms. Retained
    non-tree edges are evaluated as graph-consistency diagnostics; this prototype
    intentionally does not claim a full bundle-adjustment solution.
    """
    component = graph["components"][0] if graph["components"] else []
    accepted = [edge for edge in graph["edges"] if edge["accepted"] and edge["image_a"] in component and edge["image_b"] in component]
    if not component:
        return {"root": None, "transforms": {}, "tree_edge_ids": [], "consistency": []}

    degree = defaultdict(float)
    for edge in accepted:
        degree[edge["image_a"]] += edge["confidence"]
        degree[edge["image_b"]] += edge["confidence"]
    root = max(component, key=lambda node: (degree[node], -node))
    tree = _maximum_spanning_tree(component, accepted)
    tree_ids = {edge["edge_id"] for edge in tree}
    adjacency = defaultdict(list)
    for edge in tree:
        H = np.asarray(edge["homography"], dtype=np.float64)
        adjacency[edge["image_a"]].append((edge["image_b"], H))
        try:
            adjacency[edge["image_b"]].append((edge["image_a"], np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            continue

    transforms = {root: np.eye(3, dtype=np.float64)}
    queue = deque([root])
    while queue:
        known = queue.popleft()
        for unknown, H_known_to_unknown in adjacency[known]:
            if unknown not in transforms:
                # T_unknown maps unknown -> root; inverse edge maps unknown -> known.
                transforms[unknown] = transforms[known] @ np.linalg.inv(H_known_to_unknown)
                transforms[unknown] /= transforms[unknown][2, 2]
                queue.append(unknown)

    consistency = []
    for edge in accepted:
        if edge["edge_id"] in tree_ids or edge["image_a"] not in transforms or edge["image_b"] not in transforms:
            continue
        h, w = image_shapes[edge["image_a"]][:2]
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        via_graph = cv2.perspectiveTransform(corners, transforms[edge["image_a"]])
        direct_to_root = transforms[edge["image_b"]] @ np.asarray(edge["homography"], dtype=np.float64)
        direct = cv2.perspectiveTransform(corners, direct_to_root)
        error = float(np.mean(np.linalg.norm(via_graph[0] - direct[0], axis=1)))
        consistency.append({"edge_id": edge["edge_id"], "corner_consistency_error_pixels": error})

    return {
        "root": root,
        "transforms": {str(key): value.tolist() for key, value in transforms.items()},
        "tree_edge_ids": sorted(tree_ids),
        "consistency": consistency,
    }
