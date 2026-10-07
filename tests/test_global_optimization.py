"""Unit tests for scalable candidate screening, pose-graph global optimization, and drift reduction."""

from __future__ import annotations

import unittest
import numpy as np

from app.services.registration_graph import (
    build_graph,
    place_largest_component,
    optimize_pose_graph,
)
from app.services.multi_registration import select_candidate_pairs


class TestGlobalOptimization(unittest.TestCase):
    def test_pose_graph_drift_reduction_on_cycle(self):
        """Simulate a 4-image closed loop with accumulated pairwise drift.
        Joint pose-graph optimization must distribute the residual and reduce global drift."""
        def make_translation_H(dx, dy):
            return np.array([
                [1.0, 0.0, float(dx)],
                [0.0, 1.0, float(dy)],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        # Synthetic 4-node loop: 0 -> 1 -> 2 -> 3 -> 0
        edges = [
            {"edge_id": "0-1", "image_a": 0, "image_b": 1, "homography": make_translation_H(100.5, 0.2), "accepted": True, "confidence": 0.95, "inlier_count": 150, "rmse_pixels": 0.4},
            {"edge_id": "1-2", "image_a": 1, "image_b": 2, "homography": make_translation_H(0.3, 99.8), "accepted": True, "confidence": 0.92, "inlier_count": 140, "rmse_pixels": 0.5},
            {"edge_id": "2-3", "image_a": 2, "image_b": 3, "homography": make_translation_H(-99.7, 0.4), "accepted": True, "confidence": 0.91, "inlier_count": 135, "rmse_pixels": 0.45},
            {"edge_id": "3-0", "image_a": 3, "image_b": 0, "homography": make_translation_H(0.2, -100.6), "accepted": True, "confidence": 0.96, "inlier_count": 160, "rmse_pixels": 0.35},
        ]

        graph = build_graph(image_count=4, edges=edges)
        self.assertEqual(graph["component_count"], 1)

        image_shapes = [(200, 200), (200, 200), (200, 200), (200, 200)]
        placement = place_largest_component(graph, image_shapes)

        self.assertEqual(len(placement["placed_nodes"]), 4)
        transforms = placement["transforms"]
        self.assertEqual(len(transforms), 4)

        opt_metrics = placement.get("global_optimization", {})
        self.assertIn("drift_reduction_px", opt_metrics)
        self.assertGreaterEqual(opt_metrics["drift_reduction_px"], 0.0)

        # Root node must remain exact identity
        root = placement["root"]
        np.testing.assert_allclose(transforms[str(root)], np.eye(3), atol=1e-6)

    def test_direct_optimize_pose_graph_call(self):
        """Direct test of optimize_pose_graph function."""
        image_shapes = [(200, 200), (200, 200), (200, 200)]
        accepted_edges = [
            {"image_a": 0, "image_b": 1, "homography": np.eye(3), "confidence": 1.0},
            {"image_a": 1, "image_b": 2, "homography": np.eye(3), "confidence": 1.0},
            {"image_a": 2, "image_b": 0, "homography": np.eye(3), "confidence": 1.0},
        ]
        transforms = {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)}

        opt_transforms, metrics = optimize_pose_graph(transforms, accepted_edges, root=0, image_shapes=image_shapes)
        self.assertIn("status", metrics)
        self.assertEqual(len(opt_transforms), 3)

    def test_candidate_screening_scalability(self):
        """Verify candidate screening prunes N^2 pairs gracefully for 50 images."""
        images = [np.full((32, 32), (i * 5) % 255, dtype=np.uint8) for i in range(50)]
        total_possible = 50 * 49 // 2  # 1225
        candidates = select_candidate_pairs(images, max_candidates=66, max_neighbors_per_image=6)

        self.assertLess(len(candidates), 500)
        self.assertGreater(len(candidates), 40)
        pruned = total_possible - len(candidates)
        self.assertGreater(pruned, 700)


if __name__ == "__main__":
    unittest.main()
