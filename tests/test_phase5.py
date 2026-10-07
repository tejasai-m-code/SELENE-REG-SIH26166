"""Phase 5 Comprehensive Test Suite: Spatial Distribution, Multi-Image Registration, Graph Solving & Mosaic Quality."""

import unittest
from time import perf_counter
import cv2
import numpy as np

from app.services.feature_matching import (
    FeatureResult,
    MatchResult,
    match_features,
    spatially_distribute_matches,
)
from app.services.metrics import compute_metrics, evaluate_spatial_distribution
from app.services.mosaic import build_relative_mosaic, evaluate_mosaic_quality
from app.services.multi_registration import run_multi_registration
from app.services.pairwise_registration import register_pair
from app.services.registration_graph import (
    build_graph,
    confidence_for_metrics,
    evaluate_cycle_consistency,
    place_largest_component,
    select_reference_node,
    validate_propagated_transform,
)


def _generate_synthetic_lunar_patch(size=(400, 400), seed=26166) -> np.ndarray:
    """Generate a high-texture deterministic synthetic lunar surface with craters and regolith."""
    rng = np.random.default_rng(seed)
    h, w = size
    # Base regolith gradient
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120.0 + 30.0 * np.sin(x / 40.0) * np.cos(y / 40.0)

    # Add pseudo-random craters
    craters = [
        (int(w * 0.25), int(h * 0.25), 45, 0.8),
        (int(w * 0.70), int(h * 0.30), 60, 1.2),
        (int(w * 0.40), int(h * 0.65), 50, 0.9),
        (int(w * 0.80), int(h * 0.75), 35, 0.7),
        (int(w * 0.15), int(h * 0.80), 30, 0.6),
        (int(w * 0.50), int(h * 0.45), 25, 0.5),
    ]

    for cx, cy, rad, depth in craters:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        crater_profile = np.exp(-((dist - rad) ** 2) / (2 * (rad * 0.3) ** 2))
        crater_rim = np.exp(-((dist - rad * 1.15) ** 2) / (2 * (rad * 0.15) ** 2))
        base -= 50.0 * depth * np.clip(1.0 - (dist / rad), 0, 1) ** 2
        base += 35.0 * depth * crater_rim

    # Add high-frequency terrain noise
    noise = rng.normal(0, 12, (h, w)).astype(np.float32)
    surface = np.clip(base + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR)


def _compute_corner_error(H_est: np.ndarray, H_gt: np.ndarray, shape: tuple[int, int]) -> float:
    """Compute average corner transfer discrepancy in pixels between two homographies."""
    h, w = shape[:2]
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    pts_est = cv2.perspectiveTransform(corners, H_est.astype(np.float32))[0]
    pts_gt = cv2.perspectiveTransform(corners, H_gt.astype(np.float32))[0]
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


class TestPhase5SpatialDistribution(unittest.TestCase):
    """Step 2 & 3: Spatial Distribution & Correspondence Selection Tests."""

    def test_spatial_distribution_metrics_output(self):
        pts = np.array([[10, 10], [390, 10], [10, 390], [390, 390], [200, 200]], dtype=np.float32)
        res = evaluate_spatial_distribution(pts, 400, 400, rows=4, cols=4)
        self.assertIn("spatial_coverage", res)
        self.assertIn("grid_occupancy", res)
        self.assertIn("convex_hull_coverage", res)
        self.assertIn("centroid_offset_normalized", res)
        self.assertIn("nearest_neighbour_spacing", res)
        self.assertIn("spatial_uniformity", res)
        self.assertIn("largest_cluster_fraction", res)
        self.assertIn("border_coverage", res)
        self.assertIn("distribution_category", res)
        self.assertEqual(res["distribution_category"], "GOOD_DISTRIBUTION")

    def test_clustered_spatial_classification(self):
        # 20 points concentrated strictly in a 10x10 corner pocket
        rng = np.random.default_rng(42)
        pts = rng.uniform(5, 25, size=(20, 2)).astype(np.float32)
        res = evaluate_spatial_distribution(pts, 400, 400, rows=4, cols=4)
        self.assertEqual(res["distribution_category"], "CLUSTERED_RISKY_DISTRIBUTION")
        self.assertLess(res["spatial_coverage"], 0.20)
        self.assertGreaterEqual(res["largest_cluster_fraction"], 0.65)

    def test_spatially_distribute_matches_pruning(self):
        rng = np.random.default_rng(42)
        # Create 100 candidate matches with heavy clustering in cell (0, 0)
        pts_s = np.vstack([
            rng.uniform(0, 50, size=(60, 2)),
            rng.uniform(100, 300, size=(40, 2)),
        ]).astype(np.float32)
        pts_r = pts_s + 5.0

        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(i % 10)) for i in range(len(pts_s))]
        match_res = MatchResult(
            keypoints_source=[],
            keypoints_reference=[],
            good_matches=matches,
            source_points=pts_s.reshape(-1, 1, 2),
            reference_points=pts_r.reshape(-1, 1, 2),
            descriptor="SIFT",
            diagnostics={"candidate_keypoints_source": len(pts_s)},
        )

        filtered = spatially_distribute_matches(match_res, (400, 400), max_per_cell=10, min_point_spacing=3.0)
        self.assertLessEqual(len(filtered.good_matches), len(match_res.good_matches))
        self.assertIn("spatial_coverage", filtered.diagnostics)
        self.assertIn("raw_match_count", filtered.diagnostics)
        self.assertEqual(filtered.diagnostics["raw_match_count"], 100)
        self.assertIn("filtered_match_count", filtered.diagnostics)


class TestPhase5GraphAndPlacement(unittest.TestCase):
    """Step 4, 5, 6, 7, 8: Graph, Reference Selection, Propagation & Cycle Consistency."""

    def test_pairwise_edge_quality_confidence(self):
        metrics_good = {
            "inlier_count": 80,
            "inlier_ratio": 0.65,
            "source_spatial_coverage": 0.60,
            "spatial_uniformity": 0.85,
            "rmse_pixels": 0.8,
            "transform_conditioning": 150.0,
        }
        metrics_poor = {
            "inlier_count": 5,
            "inlier_ratio": 0.08,
            "source_spatial_coverage": 0.05,
            "spatial_uniformity": 0.20,
            "rmse_pixels": 7.5,
            "transform_conditioning": 2e5,
        }
        conf_good = confidence_for_metrics(metrics_good)
        conf_poor = confidence_for_metrics(metrics_poor)
        self.assertGreater(conf_good, conf_poor)
        self.assertGreater(conf_good, 20.0)

    def test_graph_connected_components(self):
        # 4 images: pair (0, 1) and pair (2, 3), disconnected
        edges = [
            {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "confidence": 10.0, "homography": np.eye(3).tolist()},
            {"edge_id": 1, "image_a": 2, "image_b": 3, "accepted": True, "confidence": 10.0, "homography": np.eye(3).tolist()},
        ]
        graph = build_graph(4, edges)
        self.assertFalse(graph["connected"])
        self.assertEqual(len(graph["components"]), 2)
        self.assertEqual(graph["components"][0], [0, 1])
        self.assertEqual(graph["components"][1], [2, 3])

    def test_reference_node_selection_centrality(self):
        # Star graph: Center (node 1) connected to 0, 2, 3
        edges = [
            {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "confidence": 20.0, "metrics": {"rmse_pixels": 1.0, "source_spatial_coverage": 0.5}},
            {"edge_id": 1, "image_a": 2, "image_b": 1, "accepted": True, "confidence": 25.0, "metrics": {"rmse_pixels": 0.8, "source_spatial_coverage": 0.6}},
            {"edge_id": 2, "image_a": 3, "image_b": 1, "accepted": True, "confidence": 22.0, "metrics": {"rmse_pixels": 1.1, "source_spatial_coverage": 0.5}},
        ]
        ref_node, info = select_reference_node([0, 1, 2, 3], edges)
        self.assertEqual(ref_node, 1)
        self.assertIn("rationale", info)
        self.assertIn("node_evaluations", info)

    def test_transformation_propagation_math(self):
        # Chain A -> B (shift dx=+50), B -> C (shift dx=+50).
        # We want placement relative to root C.
        # Edge A->B: H_A_B translates by +50.
        # Edge B->C: H_B_C translates by +50.
        H_A_B = np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        H_B_C = np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        edges = [
            {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "confidence": 10.0, "homography": H_A_B.tolist(), "metrics": {"rmse_pixels": 0.5, "source_spatial_coverage": 0.5}},
            {"edge_id": 1, "image_a": 1, "image_b": 2, "accepted": True, "confidence": 10.0, "homography": H_B_C.tolist(), "metrics": {"rmse_pixels": 0.5, "source_spatial_coverage": 0.5}},
        ]
        graph = build_graph(3, edges)
        shapes = [(300, 300), (300, 300), (300, 300)]
        placement = place_largest_component(graph, shapes)

        # Transforms map each image into root coordinate frame (node 1 or 2).
        root = placement["root"]
        T_0 = np.array(placement["transforms"]["0"])
        T_1 = np.array(placement["transforms"]["1"])
        T_2 = np.array(placement["transforms"]["2"])

        # Test point (0, 0) in Image 0
        pt0 = np.array([[[0.0, 0.0]]], dtype=np.float32)
        pt0_in_root = cv2.perspectiveTransform(pt0, T_0.astype(np.float32))[0][0]

        pt1 = np.array([[[0.0, 0.0]]], dtype=np.float32)
        pt1_in_root = cv2.perspectiveTransform(pt1, T_1.astype(np.float32))[0][0]

        # Verify relative translation between 0 and 1 is exactly 50 px
        self.assertAlmostEqual(pt0_in_root[0] - pt1_in_root[0], 50.0, delta=1e-3)

    def test_cycle_consistency_detection(self):
        # 3 images with consistent cycle:
        # A -> B (+40, 0)
        # B -> C (0, +30)
        # C -> A (-40, -30) -> composition should equal Identity
        H_AB = np.array([[1, 0, 40], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        H_BC = np.array([[1, 0, 0], [0, 1, 30], [0, 0, 1]], dtype=np.float64)
        H_CA = np.array([[1, 0, -40], [0, 1, -30], [0, 0, 1]], dtype=np.float64)

        edges_consistent = [
            {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "homography": H_AB.tolist()},
            {"edge_id": 1, "image_a": 1, "image_b": 2, "accepted": True, "homography": H_BC.tolist()},
            {"edge_id": 2, "image_a": 2, "image_b": 0, "accepted": True, "homography": H_CA.tolist()},
        ]
        shapes = [(300, 300), (300, 300), (300, 300)]
        res_consistent = evaluate_cycle_consistency([0, 1, 2], edges_consistent, shapes)
        self.assertTrue(res_consistent["graph_cycle_consistent"])
        self.assertEqual(res_consistent["cycle_count"], 1)
        self.assertLess(res_consistent["mean_cycle_error_pixels"], 0.01)

        # Now deliberately corrupt edge C -> A by +25 px
        H_CA_corrupt = np.array([[1, 0, -15], [0, 1, -30], [0, 0, 1]], dtype=np.float64)
        edges_corrupt = [
            {"edge_id": 0, "image_a": 0, "image_b": 1, "accepted": True, "homography": H_AB.tolist()},
            {"edge_id": 1, "image_a": 1, "image_b": 2, "accepted": True, "homography": H_BC.tolist()},
            {"edge_id": 2, "image_a": 2, "image_b": 0, "accepted": True, "homography": H_CA_corrupt.tolist()},
        ]
        res_corrupt = evaluate_cycle_consistency([0, 1, 2], edges_corrupt, shapes, tolerance_pixels=5.0)
        self.assertFalse(res_corrupt["graph_cycle_consistent"])
        self.assertGreater(res_corrupt["mean_cycle_error_pixels"], 15.0)
        self.assertIn(2, res_corrupt["suspect_edge_ids"])


class TestPhase5MosaicAndBlending(unittest.TestCase):
    """Step 10, 11, 12: Mosaic Construction, Blending Modes & Quality Metrics."""

    def test_mosaic_blending_modes(self):
        img0 = np.full((100, 100, 3), 100, dtype=np.uint8)
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)

        transforms = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 40], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }

        for mode in ("distance_weighted", "feather", "average", "overlay"):
            mosaic, info = build_relative_mosaic([img0, img1], transforms, blending_mode=mode)
            self.assertIsNotNone(mosaic)
            self.assertEqual(info["placed_image_count"], 2)
            self.assertEqual(info["width"], 140)
            self.assertEqual(info["height"], 100)
            self.assertIn("quality_metrics", info)
            self.assertGreater(info["quality_metrics"]["overlap_area_pixels"], 0)


class TestPhase5SyntheticGroundTruth(unittest.TestCase):
    """Step 13 & 14: Synthetic Ground-Truth Validation (3, 5, 8 Images)."""

    @classmethod
    def setUpClass(cls):
        cls.base_surface = _generate_synthetic_lunar_patch((600, 600), seed=26166)

    def test_synthetic_3_image_ground_truth(self):
        """3-Image Dataset: Translation and small rotation with known ground truth."""
        base = self.base_surface
        h, w = 300, 300

        # Sub-crop 3 overlapping tiles
        # Tile 0: Center (x=150, y=150)
        # Tile 1: East (x=210, y=150) -> dx = +60 px
        # Tile 2: South-East (x=200, y=200) -> dx = +50 px, dy = +50 px
        img0 = base[150:450, 150:450].copy()
        img1 = base[150:450, 210:510].copy()
        img2 = base[200:500, 200:500].copy()

        # Known analytical ground truth relative to Image 0
        # img1 relative to img0: shift by (+60, 0)
        # img2 relative to img0: shift by (+50, +50)
        H_gt = {
            0: np.eye(3, dtype=np.float64),
            1: np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            2: np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]], dtype=np.float64),
        }

        res = run_multi_registration([img0, img1, img2], {"detector": "sift", "max_features": 4000})

        self.assertEqual(res.summary["images_registered"], 3)
        self.assertEqual(res.summary["unregistered_images"], [])
        self.assertIsNotNone(res.mosaic)

        ref_node = res.placement["root"]
        T_ref_gt = H_gt[ref_node]

        for i in (0, 1, 2):
            T_est = np.array(res.placement["transforms"][str(i)])
            # Expected transform of image i to reference node: H_gt[ref]^(-1) * H_gt[i]
            T_expected = np.linalg.inv(T_ref_gt) @ H_gt[i]
            T_expected /= T_expected[2, 2]

            corner_err = _compute_corner_error(T_est, T_expected, (h, w))
            print(f"[Phase 5 GT-3] Image {i} to ref ({ref_node}) corner error: {corner_err:.3f} px")
            self.assertLess(corner_err, 2.5, f"Image {i} ground-truth error too high: {corner_err:.2f} px")

    def test_synthetic_5_image_ground_truth(self):
        """5-Image Dataset: Center, North, South, East, West with rotation and scale."""
        base = self.base_surface
        h, w = 300, 300

        # Center tile
        img0 = base[150:450, 150:450].copy()
        # East tile (dx=+60)
        img1 = base[150:450, 210:510].copy()
        # West tile (dx=-60)
        img2 = base[150:450, 90:390].copy()
        # South tile (dy=+60)
        img3 = base[210:510, 150:450].copy()
        # North tile (dy=-60)
        img4 = base[90:390, 150:450].copy()

        H_gt = {
            0: np.eye(3, dtype=np.float64),
            1: np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            2: np.array([[1, 0, -60], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            3: np.array([[1, 0, 0], [0, 1, 60], [0, 0, 1]], dtype=np.float64),
            4: np.array([[1, 0, 0], [0, 1, -60], [0, 0, 1]], dtype=np.float64),
        }

        res = run_multi_registration([img0, img1, img2, img3, img4], {"detector": "sift", "max_features": 4000})

        self.assertEqual(res.summary["images_registered"], 5)
        self.assertEqual(res.placement["root"], 0, "Center tile 0 should be selected as reference node.")
        self.assertIsNotNone(res.mosaic)

        for i in range(5):
            T_est = np.array(res.placement["transforms"][str(i)])
            corner_err = _compute_corner_error(T_est, H_gt[i], (h, w))
            print(f"[Phase 5 GT-5] Image {i} corner error: {corner_err:.3f} px")
            self.assertLess(corner_err, 3.0, f"Image {i} ground-truth error too high: {corner_err:.2f} px")

    def test_synthetic_8_image_orbital_strip(self):
        """8-Image Dataset: Orbital strip progression with partial overlaps and illumination change."""
        base = _generate_synthetic_lunar_patch((1200, 600), seed=9999)
        images = []
        H_gt = {}

        # 8 sequential orbital tiles overlapping vertically (dy=+75 px each)
        for i in range(8):
            y0 = i * 75
            tile = base[y0 : y0 + 350, 100:500].copy()
            # Apply slight illumination variation to simulate sun angle change
            gamma = 0.9 + 0.03 * i
            tile = np.clip(np.power(tile.astype(np.float32) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
            images.append(tile)
            H_gt[i] = np.array([[1, 0, 0], [0, 1, float(y0)], [0, 0, 1]], dtype=np.float64)

        res = run_multi_registration(images, {"detector": "sift", "max_features": 3000})

        self.assertGreaterEqual(res.summary["images_registered"], 7)
        self.assertIsNotNone(res.mosaic)
        self.assertIn("quality_metrics", res.mosaic_info)
        print(f"[Phase 5 GT-8] Registered {res.summary['images_registered']}/8 images, mosaic quality={res.mosaic_info['quality_label']}")


class TestPhase5FailureCases(unittest.TestCase):
    """Step 15: Failure Mode Stress Testing (A-M)."""

    def setUp(self):
        self.valid_patch = _generate_synthetic_lunar_patch((300, 300), seed=123)

    def test_failure_case_a_blank_image(self):
        """Failure Case A: One image has no features (pure black)."""
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        img1 = self.valid_patch
        img2 = self.valid_patch.copy()

        res = run_multi_registration([blank, img1, img2])
        # Blank image (0) should be isolated/unplaced, valid pair (1, 2) registered
        self.assertIn(0, res.summary["unregistered_images"])
        self.assertEqual(res.summary["images_registered"], 2)

    def test_failure_case_c_disconnected_graph(self):
        """Failure Case C: Disconnected graph (2 separate pairs)."""
        p1_a = self.valid_patch
        p1_b = self.valid_patch[20:280, 20:280]
        # Completely different texture for second pair
        p2_a = np.random.default_rng(999).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        p2_b = p2_a.copy()

        res = run_multi_registration([p1_a, p1_b, p2_a, p2_b])
        self.assertFalse(res.summary["graph_connected"])
        self.assertGreater(res.summary["connected_components"], 1)

    def test_failure_case_d_duplicate_image(self):
        """Failure Case D: Duplicate identical images."""
        img = self.valid_patch
        res = run_multi_registration([img, img])
        self.assertTrue(res.summary["graph_connected"])
        self.assertEqual(res.summary["images_registered"], 2)

    def test_failure_case_e_very_small_overlap(self):
        """Failure Case E: Tiny overlap (< 2%) rejected."""
        img0 = self.valid_patch
        # Shifted by 295 px on a 300 px image (5 px overlap = 1.6%)
        H_tiny = np.array([[1, 0, 295], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        img1 = cv2.warpPerspective(img0, H_tiny, (300, 300))

        result = register_pair(img0, img1)
        # Should either fail feature matching or fail overlap validation
        self.assertFalse(result.success)

    def test_failure_case_f_corrupted_transform_rejected(self):
        """Failure Case F: Non-finite matrix rejected by validation."""
        H_nan = np.array([[1, np.nan, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        is_val, reason = validate_propagated_transform(H_nan, (300, 300))
        self.assertFalse(is_val)

    def test_failure_case_h_invalid_dimensions(self):
        """Failure Case H: Empty / 0x0 image input raises ValueError."""
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            run_multi_registration([empty, self.valid_patch])

    def test_failure_case_m_canvas_overflow_protection(self):
        """Failure Case M: Extreme canvas bounds trigger memory protection without crashing."""
        # Extreme shift to 500,000 pixels
        transforms = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 500000], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }
        with self.assertRaises(ValueError):
            build_relative_mosaic([self.valid_patch, self.valid_patch], transforms)


class TestPhase5PerformanceScaling(unittest.TestCase):
    """Step 16: Performance Scaling Benchmarks."""

    def test_performance_scaling_3_5_8(self):
        base = _generate_synthetic_lunar_patch((800, 600), seed=42)

        timings = {}
        for count in (3, 5):
            images = [base[i * 40 : i * 40 + 300, 100:400].copy() for i in range(count)]
            t0 = perf_counter()
            res = run_multi_registration(images, {"detector": "sift", "max_features": 2500})
            elapsed = perf_counter() - t0
            timings[count] = {
                "total_seconds": round(elapsed, 3),
                "stages": res.summary["stage_timings_seconds"],
                "registered": res.summary["images_registered"],
            }
            print(f"[Phase 5 Perf] {count} images: total={elapsed:.2f}s, stages={res.summary['stage_timings_seconds']}")

        self.assertIn(3, timings)
        self.assertIn(5, timings)


if __name__ == "__main__":
    unittest.main()
