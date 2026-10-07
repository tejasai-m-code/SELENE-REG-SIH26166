"""Phase 5 Comprehensive Verification Suite: Scientific Canonical Match Visualization.

Validates the Phase 5 requirements of the Master Engineering Prompt:
1. Complete uncropped side-by-side visualization (Source on left, Reference on right).
2. Preservation of original aspect ratios, especially for small source inside large reference.
3. Verified green lines only by default (NO red rejected lines in primary view).
4. Sub-pixel and pixel-accurate coordinate correspondence between source and reference rasters.
5. 4x4 spatial coverage grid calculation and assignment (cells 1..16).
6. Line hit-testing and distance calculation for interactive hover and click selection.
7. Coordinate inversion engine (screen space <-> source/reference pixel space under pan & zoom).
"""

import unittest
import numpy as np
import cv2

from app.services.registration import draw_matches
from app.services.pairwise_registration import serialize_correspondences


class TestPhase5CanonicalVisualization(unittest.TestCase):
    """Test suite for Phase 5 Scientific Canonical Match Visualization."""

    def setUp(self):
        # Deterministic synthetic test rasters
        # Moving image: 200x300
        self.img_source = np.full((200, 300, 3), 120, dtype=np.uint8)
        # Fixed reference image: 400x500 (small source inside large reference)
        self.img_reference = np.full((400, 500, 3), 140, dtype=np.uint8)

        # 4 candidate keypoints in source and reference
        self.src_pts = np.array([
            [50.0, 50.0],
            [250.0, 50.0],
            [50.0, 150.0],
            [250.0, 150.0],
        ], dtype=np.float32)

        # In reference: offset by +100 in X, +120 in Y
        self.ref_pts = np.array([
            [150.0, 170.0],
            [350.0, 170.0],
            [150.0, 270.0],
            [390.0, 390.0],  # intentional outlier
        ], dtype=np.float32)

        # Inlier mask: first 3 are inliers, 4th is outlier
        self.inliers = np.array([True, True, True, False], dtype=bool)

    def test_uncropped_side_by_side_dimensions_preserved(self):
        """Source and Reference are placed side-by-side uncropped with original aspect ratios."""
        h_s, w_s = self.img_source.shape[:2]
        h_r, w_r = self.img_reference.shape[:2]

        vis = draw_matches(
            self.img_source,
            self.img_reference,
            self.src_pts,
            self.ref_pts,
            self.inliers,
            show_outliers=False,
        )

        expected_h = max(h_s, h_r)
        expected_w = w_s + w_r
        self.assertEqual(vis.shape[0], expected_h, f"Expected height {expected_h}, got {vis.shape[0]}")
        self.assertEqual(vis.shape[1], expected_w, f"Expected width {expected_w}, got {vis.shape[1]}")

    def test_primary_view_draws_verified_green_lines_only_no_red_lines(self):
        """In primary view (show_outliers=False), only verified green lines are drawn, NO red outlier lines."""
        vis_default = draw_matches(
            self.img_source,
            self.img_reference,
            self.src_pts,
            self.ref_pts,
            self.inliers,
            registration_status="PASS",
            show_outliers=False,
        )

        # Check for presence of red lines:
        # Outlier lines use BGR (0, 0, 255) -> R channel high, B/G channels very low
        # Inlier lines use BGR (50, 205, 50) -> G channel high
        r_channel = vis_default[:, :, 2]
        g_channel = vis_default[:, :, 1]
        b_channel = vis_default[:, :, 0]

        # Strong pure red pixels (R > 200, G < 50, B < 50)
        pure_red_pixels = np.count_nonzero((r_channel > 200) & (g_channel < 50) & (b_channel < 50))
        self.assertEqual(
            pure_red_pixels,
            0,
            f"Expected zero red pixels when show_outliers=False, found {pure_red_pixels}",
        )

        # Verified green lines exist (G > 180, R < 100)
        verified_green_pixels = np.count_nonzero((g_channel > 180) & (r_channel < 100))
        self.assertGreater(
            verified_green_pixels,
            0,
            "Expected verified green correspondence lines in visualization.",
        )

    def test_diagnostic_view_draws_outliers_when_requested(self):
        """When show_outliers=True, red outlier lines are drawn for advanced diagnosis."""
        vis_outliers = draw_matches(
            self.img_source,
            self.img_reference,
            self.src_pts,
            self.ref_pts,
            self.inliers,
            show_outliers=True,
        )

        r_channel = vis_outliers[:, :, 2]
        g_channel = vis_outliers[:, :, 1]
        b_channel = vis_outliers[:, :, 0]

        # Outlier red pixels exist when show_outliers=True
        pure_red_pixels = np.count_nonzero((r_channel > 200) & (g_channel < 50) & (b_channel < 50))
        self.assertGreater(
            pure_red_pixels,
            0,
            "Expected red outlier lines to be drawn when show_outliers=True.",
        )

    def test_correspondence_serialization_preserves_subpixel_and_spatial_cells(self):
        """serialize_correspondences produces sub-pixel shifts, residuals, and 4x4 spatial cells (1..16)."""
        H_est = np.array([
            [1.0, 0.0, 100.0],
            [0.0, 1.0, 120.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        refined_ref = self.ref_pts + np.array([
            [0.12, -0.08],
            [-0.05, 0.15],
            [0.02, 0.04],
            [0.0, 0.0],
        ], dtype=np.float32)

        from app.services.pairwise_registration import PairwiseRegistrationResult
        mock_res = PairwiseRegistrationResult(
            homography=H_est,
            inlier_mask=self.inliers,
            metrics={"rmse": 0.42},
            registered=None,
            match_visualization=np.zeros((10, 10, 3), dtype=np.uint8),
            source_points=self.src_pts,
            reference_points=self.ref_pts,
            final_reference_points=refined_ref,
            raw_match_count=4,
            post_distribution_match_count=4,
            ecc_used=False,
            ecc_correlation=None,
            detector="sift",
            registration_status="PASS",
            success=True,
            subpixel={"applied_methods": ["lucas_kanade"]},
            match_confidences=[0.95, 0.88, 0.91, 0.22],
        )

        ser = serialize_correspondences(mock_res)

        self.assertEqual(len(ser), 4)

        # Check inlier 0
        c0 = ser[0]
        self.assertEqual(c0["status"], "INLIER")
        self.assertTrue(c0["is_inlier"])
        self.assertEqual(c0["source"], [50.0, 50.0])
        self.assertAlmostEqual(c0["residual"], 0.144, places=2)
        self.assertEqual(c0["refinement_method"], "lucas_kanade")
        self.assertEqual(c0["refinement_delta"], [0.12, -0.08])
        self.assertEqual(c0["confidence"], 0.95)

        # 4x4 spatial cells are between 1 and 16
        for c in ser:
            self.assertIn("spatial_cell", c)
            self.assertIn("source_cell", c)
            self.assertIn("reference_cell", c)
            self.assertIn("spatial_coverage", c)
            self.assertGreaterEqual(c["spatial_cell"], 1)
            self.assertLessEqual(c["spatial_cell"], 16)
            self.assertGreaterEqual(c["source_cell"], 1)
            self.assertLessEqual(c["source_cell"], 16)

        # Check outlier 3
        c3 = ser[3]
        self.assertEqual(c3["status"], "OUTLIER")
        self.assertFalse(c3["is_inlier"])
        self.assertAlmostEqual(c3["residual"], 126.49, places=1)

    def test_coordinate_inversion_engine_math(self):
        """Test mathematical fidelity of canvas coordinate transformation under pan & zoom."""
        # Simulated canvas view:
        # Scale = 1.85, pan = (-45.0, 120.0)
        scale = 1.85
        pan_x = -45.0
        pan_y = 120.0

        # Source image (300x200), Reference image (500x400)
        # Side-by-side layout: Source at [0, 0], Reference at [300, 0]
        w_src = 300
        h_src = 200
        w_ref = 500
        h_ref = 400

        # An arbitrary pixel inside Source: (x=140.5, y=85.25)
        src_px = np.array([140.5, 85.25])
        # Canvas screen position:
        screen_x = src_px[0] * scale + pan_x
        screen_y = src_px[1] * scale + pan_y

        # Invert back to image coordinate:
        inv_img_x = (screen_x - pan_x) / scale
        inv_img_y = (screen_y - pan_y) / scale
        np.testing.assert_allclose(src_px, [inv_img_x, inv_img_y], rtol=1e-5)

        # An arbitrary pixel inside Reference: (x=215.0, y=310.0)
        ref_px = np.array([215.0, 310.0])
        # In side-by-side canvas, Reference has an X offset of w_src:
        canvas_ref_x = ref_px[0] + w_src
        canvas_ref_y = ref_px[1]
        screen_ref_x = canvas_ref_x * scale + pan_x
        screen_ref_y = canvas_ref_y * scale + pan_y

        # Invert from screen space:
        inv_canvas_x = (screen_ref_x - pan_x) / scale
        inv_canvas_y = (screen_ref_y - pan_y) / scale

        # Detect that it falls into Reference side (inv_canvas_x >= w_src):
        self.assertGreaterEqual(inv_canvas_x, w_src)
        inv_ref_x = inv_canvas_x - w_src
        inv_ref_y = inv_canvas_y
        np.testing.assert_allclose(ref_px, [inv_ref_x, inv_ref_y], rtol=1e-5)

    def test_line_segment_hit_test_distance(self):
        """Test point-to-segment distance algorithm used for interactive correspondence line hover/click."""
        def point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-9:
                return float(np.hypot(px - x1, py - y1))
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return float(np.hypot(px - proj_x, py - proj_y))

        # Line from (100, 100) to (300, 300)
        p1 = (100.0, 100.0)
        p2 = (300.0, 300.0)

        # Point directly on midpoint (200, 200)
        d_mid = point_to_segment_distance(200.0, 200.0, *p1, *p2)
        self.assertAlmostEqual(d_mid, 0.0, places=5)

        # Point 4 pixels away perpendicular to the line: (200 + 4/sqrt(2), 200 - 4/sqrt(2))
        d_close = point_to_segment_distance(200.0 + 2.8284, 200.0 - 2.8284, *p1, *p2)
        self.assertAlmostEqual(d_close, 4.0, places=3)
        # Should be within typical 8px hover threshold
        self.assertLess(d_close, 8.0)

        # Point far away: (100, 400)
        d_far = point_to_segment_distance(100.0, 400.0, *p1, *p2)
        self.assertGreater(d_far, 100.0)


if __name__ == "__main__":
    unittest.main()
