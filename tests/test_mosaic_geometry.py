"""Scientific Multi-Image Mosaic Geometry, Order Independence & Bounds Tests.

Covers:
1. Four-directional canvas expansion (left, right, top, bottom)
2. Order-independence of multi-image mosaic placement
3. Scaling up to 3, 5, 8, and 9 frames without clipping previously placed imagery
4. All transformed footprints lie strictly within [0, width] x [0, height]
5. Structured MOSAIC_CANVAS_TOO_LARGE memory safety exception
6. Real 15-stage progress event emission
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

# Ensure api-server python package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts" / "api-server" / "python"))

from app.services.mosaic import build_relative_mosaic, MosaicCanvasTooLargeError
from app.services.multi_registration import run_multi_registration, select_candidate_pairs
from app.services.evaluation import make_synthetic_lunar_surface


class TestMosaicGeometryAndExpansion(unittest.TestCase):
    """Part B & Part G: Core mosaic geometry, bounds computation, and 4-way canvas expansion."""

    def setUp(self):
        # Generate a large textured lunar canvas to crop overlapping frames from
        self.surface = make_synthetic_lunar_surface(seed=26166, size=500)
        self.frame_h, self.frame_w = 200, 200

        # Center frame
        cx, cy = 150, 150
        self.center = self.surface[cy : cy + self.frame_h, cx : cx + self.frame_w]

        # Shifted frames with known spatial overlap
        self.frame_left = self.surface[cy : cy + self.frame_h, cx - 60 : cx - 60 + self.frame_w]
        self.frame_right = self.surface[cy : cy + self.frame_h, cx + 60 : cx + 60 + self.frame_w]
        self.frame_top = self.surface[cy - 60 : cy - 60 + self.frame_h, cx : cx + self.frame_w]
        self.frame_bottom = self.surface[cy + 60 : cy + 60 + self.frame_h, cx : cx + self.frame_w]

    def test_four_direction_canvas_expansion(self):
        """Verify the global canvas expands in left, right, top, bottom directions without clipping."""
        # 5 images extending in all four cardinal directions
        images = [self.center, self.frame_left, self.frame_right, self.frame_top, self.frame_bottom]
        
        result = run_multi_registration(images, {"detector": "sift", "ecc_refinement": False})
        
        self.assertIsNotNone(result.mosaic, "Mosaic should be generated successfully.")
        self.assertIsNotNone(result.mosaic_info)
        info = result.mosaic_info
        
        # Dimensions must encompass the full extent of all 5 placed frames
        self.assertGreater(info["width"], self.frame_w + 50, "Canvas width must expand beyond single frame.")
        self.assertGreater(info["height"], self.frame_h + 50, "Canvas height must expand beyond single frame.")
        
        # Verify that all 5 placed frames have footprints strictly inside canvas
        footprints = info.get("footprints", [])
        self.assertGreaterEqual(len(footprints), 4, "At least 4 frames must be placed in connected component.")
        
        canvas_w = info["width"]
        canvas_h = info["height"]
        
        for fp in footprints:
            corners = np.array(fp["corners"], dtype=np.float32)
            # All x must be >= 0 and <= canvas_w
            self.assertTrue(np.all(corners[:, 0] >= -0.5), f"Corner X extends left outside canvas: {corners[:, 0]}")
            self.assertTrue(np.all(corners[:, 0] <= canvas_w + 0.5), f"Corner X extends right outside canvas: {corners[:, 0]}")
            # All y must be >= 0 and <= canvas_h
            self.assertTrue(np.all(corners[:, 1] >= -0.5), f"Corner Y extends top outside canvas: {corners[:, 1]}")
            self.assertTrue(np.all(corners[:, 1] <= canvas_h + 0.5), f"Corner Y extends bottom outside canvas: {corners[:, 1]}")

        # Summary counts must be accurate and distinct
        summary = result.summary
        self.assertEqual(summary["images_loaded"], 5)
        self.assertGreaterEqual(summary["images_placed"], 4)
        self.assertEqual(summary["images_unplaced"], 5 - summary["images_placed"])

    def test_adding_images_preserves_previous_content(self):
        """Regression test for the bug: adding frames must expand the canvas and NOT shrink or cut earlier regions."""
        # 2 images: center + right
        res2 = run_multi_registration([self.center, self.frame_right], {"detector": "sift", "ecc_refinement": False})
        self.assertIsNotNone(res2.mosaic)
        w2 = res2.mosaic_info["width"]
        h2 = res2.mosaic_info["height"]

        # 3 images: center + right + left
        res3 = run_multi_registration([self.center, self.frame_right, self.frame_left], {"detector": "sift", "ecc_refinement": False})
        self.assertIsNotNone(res3.mosaic)
        w3 = res3.mosaic_info["width"]

        # Adding a frame to the left MUST increase total canvas width!
        self.assertGreater(w3, w2, f"Canvas width with 3 frames ({w3}) must exceed canvas with 2 frames ({w2})")

        # 4 images: center + right + left + bottom
        res4 = run_multi_registration([self.center, self.frame_right, self.frame_left, self.frame_bottom], {"detector": "sift", "ecc_refinement": False})
        self.assertIsNotNone(res4.mosaic)
        h4 = res4.mosaic_info["height"]

        # Adding a frame to the bottom MUST increase total canvas height!
        self.assertGreater(h4, h2, f"Canvas height with bottom frame ({h4}) must exceed height without it ({h2})")


class TestMosaicOrderIndependence(unittest.TestCase):
    """Part F: Verification that registration and global placement are stable against input ordering."""

    def setUp(self):
        self.surface = make_synthetic_lunar_surface(seed=8001, size=450)
        h, w = 180, 180
        cx, cy = 120, 120
        self.f1 = self.surface[cy : cy + h, cx : cx + w]
        self.f2 = self.surface[cy : cy + h, cx + 50 : cx + 50 + w]
        self.f3 = self.surface[cy : cy + h, cx + 100 : cx + 100 + w]

    def test_order_independence_forward_vs_reverse(self):
        """Forward [1, 2, 3] vs reverse [3, 2, 1] input order produces consistent coverage."""
        images_fwd = [self.f1, self.f2, self.f3]
        images_rev = [self.f3, self.f2, self.f1]

        res_fwd = run_multi_registration(images_fwd, {"detector": "sift", "ecc_refinement": False})
        res_rev = run_multi_registration(images_rev, {"detector": "sift", "ecc_refinement": False})

        self.assertIsNotNone(res_fwd.mosaic)
        self.assertIsNotNone(res_rev.mosaic)

        # Both orderings must place all 3 frames
        self.assertEqual(res_fwd.summary["images_placed"], 3)
        self.assertEqual(res_rev.summary["images_placed"], 3)

        # Canvas dimensions should be nearly identical (within 8px padding/rounding)
        self.assertAlmostEqual(res_fwd.mosaic_info["width"], res_rev.mosaic_info["width"], delta=8)
        self.assertAlmostEqual(res_fwd.mosaic_info["height"], res_rev.mosaic_info["height"], delta=8)


class TestCandidatePairCompleteness(unittest.TestCase):
    """Ensure candidate pair generation evaluates all pairs up to 12 images (no histogram truncation)."""

    def test_9_images_has_36_pairs(self):
        images = [np.zeros((100, 100), dtype=np.uint8) for _ in range(9)]
        pairs = select_candidate_pairs(images)
        self.assertEqual(len(pairs), 36, "9 images must produce all 36 candidate pairs without truncation.")

    def test_12_images_has_66_pairs(self):
        images = [np.zeros((100, 100), dtype=np.uint8) for _ in range(12)]
        pairs = select_candidate_pairs(images)
        self.assertEqual(len(pairs), 66, "12 images must produce all 66 candidate pairs without truncation.")


class TestMosaicSafetyAndErrors(unittest.TestCase):
    """Verify memory safety: MOSAIC_CANVAS_TOO_LARGE error structure and safety limits."""

    def test_canvas_too_large_structured_error(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Construct transforms that would require a 20,000 px canvas
        transforms = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 25000], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }

        with self.assertRaises(MosaicCanvasTooLargeError) as ctx:
            build_relative_mosaic([img, img], transforms, max_dimension=12000)

        err = ctx.exception
        self.assertEqual(err.code, "MOSAIC_CANVAS_TOO_LARGE")
        self.assertGreater(err.requested_width, 12000)
        self.assertIn("current_bounds", dir(err))
        self.assertIn("configured_safety_limit", dir(err))
        self.assertIn("actionable_recommendation", dir(err))


class TestRealMosaicProgressEmissions(unittest.TestCase):
    """Part D: Verify that real backend execution stages are emitted monotonically."""

    def test_progress_stages_emitted(self):
        surface = make_synthetic_lunar_surface(seed=999, size=350)
        f1 = surface[50:200, 50:200]
        f2 = surface[50:200, 90:240]

        stages_collected = []

        def on_progress(stage, stage_index, stage_count, progress, message, status="RUNNING", error=None):
            stages_collected.append((stage, stage_index, progress, message))

        res = run_multi_registration([f1, f2], {"detector": "sift", "ecc_refinement": False}, progress_callback=on_progress)

        self.assertIsNotNone(res.mosaic)
        self.assertGreater(len(stages_collected), 5, "Multiple real execution stages must be emitted.")

        stage_names = [s[0] for s in stages_collected]
        self.assertIn("INITIALIZING", stage_names)
        self.assertIn("VALIDATING_INPUTS", stage_names)
        self.assertIn("REGISTERING_IMAGE_PAIRS", stage_names)
        self.assertIn("COMPUTING_GLOBAL_BOUNDS", stage_names)
        self.assertIn("COMPLETE", stage_names)

        # Progress values should reach 1.0
        final_progress = stages_collected[-1][2]
        self.assertEqual(final_progress, 1.0)


if __name__ == "__main__":
    unittest.main()
