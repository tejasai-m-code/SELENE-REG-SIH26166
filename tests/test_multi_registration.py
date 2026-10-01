import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, "backend")

from app.services.mosaic import build_relative_mosaic
from app.services.multi_registration import run_multi_registration, select_candidate_pairs
from app.services.registration_graph import build_graph, place_largest_component
from app.services.pairwise_registration import register_pair


class MultiRegistrationServiceTests(unittest.TestCase):
    def test_small_collections_screen_every_pair(self):
        images = [np.full((32, 32), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
        self.assertEqual(len(select_candidate_pairs(images)), 6)

    def test_graph_marks_isolated_images(self):
        edge = {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "confidence": 3.0, "homography": np.eye(3).tolist()}
        graph = build_graph(3, [edge])
        self.assertEqual(graph["isolated_nodes"], [2])
        self.assertEqual(graph["components"], [[0, 1], [2]])

    def test_relative_placement_uses_pair_transform(self):
        H = np.array([[1, 0, 12], [0, 1, 0], [0, 0, 1]], dtype=float).tolist()
        graph = build_graph(2, [{"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "confidence": 4.0, "homography": H}])
        placement = place_largest_component(graph, [(20, 20), (20, 20)])
        self.assertEqual(set(placement["transforms"]), {"0", "1"})

    def test_mosaic_fits_translated_images(self):
        image = np.full((10, 10), 255, dtype=np.uint8)
        mosaic, info = build_relative_mosaic([image, image], {"0": np.eye(3).tolist(), "1": [[1, 0, 8], [0, 1, 0], [0, 0, 1]]})
        self.assertGreaterEqual(info["width"], 18)
        self.assertEqual(info["placed_image_count"], 2)
        self.assertEqual(mosaic.ndim, 3)

    def test_existing_pair_engine_registers_a_translated_image(self):
        source = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(source, (80, 80), 20, 255, -1)
        cv2.rectangle(source, (140, 120), (200, 180), 180, -1)
        cv2.line(source, (30, 200), (200, 30), 220, 3)
        reference = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, 8]]), (240, 240))
        result = register_pair(source, reference, ecc_refinement=False, max_features=2000)
        self.assertGreaterEqual(result.metrics["inlier_count"], 4)
        self.assertLess(result.metrics["rmse_pixels"], 2.0)

    def test_multi_orchestrator_builds_a_relative_mosaic(self):
        source = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(source, (80, 80), 20, 255, -1)
        cv2.rectangle(source, (140, 120), (200, 180), 180, -1)
        cv2.line(source, (30, 200), (200, 30), 220, 3)
        reference = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, 8]]), (240, 240))
        result = run_multi_registration([source, reference], {"ecc_refinement": False, "max_features": 2000})
        self.assertEqual(result.summary["successful_pair_count"], 1)
        self.assertEqual(result.summary["placed_image_count"], 2)
        self.assertIsNotNone(result.mosaic)


if __name__ == "__main__":
    unittest.main()
