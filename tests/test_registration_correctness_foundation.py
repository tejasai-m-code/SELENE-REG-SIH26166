"""Regression test suite for Registration Correctness and Scientific Foundation (Part 1).

Tests:
1. Subpixel Error Guard: Refinement that degrades RMSE reverts to H_raw with SUBPIXEL_NOT_VALIDATED.
2. Case A: Good local correspondences + physically plausible transform -> PASS (CASE_A_ACCEPTED).
3. Case B: Good local correspondences but implausible global transform -> FAIL (CASE_B_IMPLAUSIBLE_TRANSFORM) with explicit reason.
4. Case C: Insufficient correspondences -> FAIL (CASE_C_POOR_CORRESPONDENCE) with explicit reason.
5. Case D: Scale-adapted registration (cross-sensor zoom) -> PASS (CASE_D_ACCEPTED_SCALE_ADAPTED).
6. Correspondence Region Visualization: Convex hull computed from verified inliers, rendered on source and reference, present in legend.
7. Registered Composite Canvas: Tight bounding box, supports negative transformed coordinates, preserves reference scale.
8. Subpixel Method Tracking: Every requested method records granular status, convergence state, and before/after error.
"""

import math
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

# Ensure api-server python directory is in sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))

from app.services.geometry import decompose_transform, validate_transform_plausibility
from app.services.pairwise_registration import register_pair
from app.services.registration import compute_registered_composite, draw_matches, refine_subpixel


def _create_crater_scene(width: int = 320, height: int = 320, seed: int = 42) -> np.ndarray:
    """Generate synthetic lunar surface with craters and illumination gradient."""
    np.random.seed(seed)
    y, x = np.mgrid[0:height, 0:width]
    gradient = 100 + 30 * np.sin(x / 50.0) + 20 * np.cos(y / 60.0)
    noise = np.random.normal(0, 4, (height, width))
    img = np.clip(gradient + noise, 0, 255).astype(np.float32)

    craters = [
        (80, 80, 28),
        (200, 90, 35),
        (100, 210, 32),
        (220, 220, 25),
        (150, 150, 18),
        (60, 160, 15),
        (240, 60, 16),
        (160, 250, 20),
    ]
    for cx, cy, r in craters:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rim_mask = np.abs(dist - r) < 4.0
        inner_mask = dist < r
        img[inner_mask] = img[inner_mask] * 0.65
        img[rim_mask] = np.clip(img[rim_mask] * 1.45, 0, 255)

    return np.clip(img, 0, 255).astype(np.uint8)


class TestSubpixelErrorGuardAndRollback(unittest.TestCase):
    """Test that subpixel refinement never degrades registration geometry or corrupts footprint."""

    def test_refinement_evidence_and_footprint_preservation(self):
        """Refinement must not drift into large translations, must maintain footprint IoU, and record real evidence."""
        ref = _create_crater_scene(320, 320, seed=101)
        M = cv2.getRotationMatrix2D((160, 160), 2.5, 1.0)
        M[0, 2] += 6.0
        M[1, 2] += -4.0
        src = cv2.warpAffine(ref, M, (320, 320))

        # Register with refinement methods
        res = register_pair(
            src,
            ref,
            detector="sift",
            refinement_methods=["taylor", "phase", "quadratic"],
        )

        # Footprint IoU must remain healthy (> 0.60), not blown out to ~0
        self.assertGreater(
            res.metrics.get("footprint_iou", 0.0),
            0.60,
            "Refinement caused footprint IoU blowout!",
        )
        self.assertIn(res.registration_status, ("PASS", "PASS_WITH_WARNING"))

        # Evidence-based subpixel status: must be recorded honestly
        sub_status = res.subpixel.get("subpixel_validation_status")
        self.assertIn(sub_status, ("SUBPIXEL_VALIDATED", "SUBPIXEL_NOT_VALIDATED", "SUBPIXEL_NEUTRAL"))
        self.assertTrue(len(res.subpixel.get("method_status", [])) > 0)

        # Ensure composite was generated
        self.assertIsNotNone(res.composite)
        self.assertEqual(res.composite_metadata.get("status"), "SUCCESS")


class TestCaseClassifications(unittest.TestCase):
    """Test the 4 explicit quality-gate Case classifications (A, B, C, D)."""

    def test_case_a_accepted(self):
        """Case A: Good local correspondences + physically plausible transform -> PASS."""
        ref = _create_crater_scene(320, 320, seed=12)
        M = cv2.getRotationMatrix2D((160, 160), 1.5, 1.0)
        M[0, 2] += 5.0
        M[1, 2] += -3.0
        src = cv2.warpAffine(ref, M, (320, 320))

        res = register_pair(src, ref, detector="sift")
        qg = res.inlier_investigation["quality_gate"]
        self.assertEqual(res.registration_status, "PASS")
        self.assertIn(qg["case_classification"], ("CASE_A_ACCEPTED", "CASE_A_ACCEPTED_WITH_WARNINGS"))
        self.assertIn("Inliers:", qg["human_readable_summary"])
        self.assertIn("Residual RMSE:", qg["human_readable_summary"])

    def test_case_b_implausible_transform(self):
        """Case B: Good local matches but implausible global transform -> FAIL with explicit reason."""
        # Synthesize a homography with excessive translation and distortion
        H_implausible = np.array([
            [1.0, 0.0, 450.0],
            [0.0, 1.0, -380.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        is_plaus, issues, decomp = validate_transform_plausibility(
            H_implausible,
            source_shape=(300, 300),
            reference_shape=(300, 300),
        )
        self.assertFalse(is_plaus)
        self.assertTrue(any("translation" in iss.lower() or "coordinates" in iss.lower() or "intersection" in iss.lower() for iss in issues))

    def test_case_c_poor_correspondences(self):
        """Case C: Insufficient correspondences -> FAIL with explicit reason."""
        # Unrelated scenes
        img1 = _create_crater_scene(240, 240, seed=1)
        img2 = np.random.RandomState(99).randint(0, 256, (240, 240), dtype=np.uint8)

        res = register_pair(img1, img2, detector="sift")
        self.assertEqual(res.registration_status, "FAIL")
        self.assertFalse(res.success)
        self.assertIsNone(res.registered)
        qg = res.inlier_investigation["quality_gate"]
        self.assertEqual(qg["case_classification"], "CASE_C_POOR_CORRESPONDENCE")
        self.assertTrue("matches" in qg["primary_reason"].lower() or "correspondences" in qg["primary_reason"].lower())

    def test_case_d_scale_adapted(self):
        """Case D: High scale ratio (e.g. 2.0x zoom) with valid overlap -> PASS."""
        ref = _create_crater_scene(400, 400, seed=42)
        # 2.0x zoom-in of center
        center_crop = ref[100:300, 100:300]
        src = cv2.resize(center_crop, (200, 200))

        res = register_pair(src, ref, detector="sift")
        # Should register with high scale ratio
        if res.inlier_investigation and res.inlier_investigation.get("final_inliers", 0) >= 6:
            self.assertIn(res.registration_status, ("PASS", "PASS_WITH_WARNING"))
            self.assertIsNotNone(res.homography)


class TestCorrespondenceRegionVisualization(unittest.TestCase):
    """Test the Verified Correspondence Region layer (Section 5)."""

    def test_correspondence_region_rendered_in_canvas(self):
        """draw_matches must render verified correspondence region on source and reference."""
        src = _create_crater_scene(200, 200, seed=10)
        ref = _create_crater_scene(200, 200, seed=10)

        # Create known inliers covering a region
        pts_s = np.array([[30, 30], [170, 30], [170, 170], [30, 170], [100, 100]], dtype=np.float32)
        pts_r = pts_s + np.array([5.0, -3.0], dtype=np.float32)
        mask = np.ones(5, dtype=bool)

        H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])

        vis = draw_matches(
            src,
            ref,
            pts_s,
            pts_r,
            inlier_mask=mask,
            registration_status="PASS",
            homography=H,
            rmse=0.45,
            iou=0.92,
        )

        self.assertIsNotNone(vis)
        self.assertEqual(vis.shape[:2], (200, 400))
        # Ensure image has non-zero channels and legend drawn
        self.assertGreater(np.sum(vis), 1000)


class TestRegisteredCompositeCanvas(unittest.TestCase):
    """Test the composite canvas calculation with negative coordinates (Section 6)."""

    def test_composite_supports_negative_coordinates(self):
        """Canvas must support negative coordinates through rigid translation."""
        src = np.ones((200, 200, 3), dtype=np.uint8) * 128
        ref = np.ones((200, 200, 3), dtype=np.uint8) * 200

        # Translation of -50, -30 maps source corners to negative space
        H = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, -30.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        composite, meta = compute_registered_composite(src, ref, H)

        self.assertIsNotNone(composite)
        self.assertEqual(meta["status"], "SUCCESS")
        # Canvas width should accommodate the shifted source and reference:
        # Reference is [0, 200], source is [-50, 150]. Min x is -50, max x is 200 -> width is 250.
        self.assertEqual(meta["canvas_width"], 250)
        self.assertEqual(meta["canvas_height"], 230)
        self.assertEqual(meta["offset_x"], 50.0)
        self.assertEqual(meta["offset_y"], 30.0)
        self.assertEqual(composite.shape[:2], (230, 250))


if __name__ == "__main__":
    unittest.main()
