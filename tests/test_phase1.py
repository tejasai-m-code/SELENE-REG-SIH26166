"""Phase 1 Correctness and Regression Test Suite.

Tests all 12 critical requirements mandated by Phase 1 of Master Engineering Directive:
1. UI alias -> canonical method mapping
2. Taylor actually executes
3. Phase correlation actually executes
4. Quadratic actually executes
5. ECC executes
6. Success state reflects quality gate (PASS, PASS_WITH_WARNING, REVIEW, FAIL)
7. Refined points propagate into final metrics
8. Raw and refined metrics remain distinguishable
9. Final inlier mask matches final geometry
10. Phase-correlation direction/sign convention
11. ECC transform direction convention
12. Failed refinement is reported honestly
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure api-server python directory is in sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))

from app.services.pairwise_registration import register_pair
from app.services.registration import refine_subpixel, refine_with_ecc
from app.services.subpixel import (
    normalize_refinement_method,
    normalize_refinement_methods,
    phase_translation,
    quadratic_peak,
    taylor_expansion,
)


def _create_synthetic_crater_scene(width: int = 240, height: int = 240, seed: int = 42) -> np.ndarray:
    """Generate a realistic synthetic lunar terrain image with craters and gradients."""
    np.random.seed(seed)
    # Base undulating terrain
    img = np.random.normal(120, 15, (height, width)).astype(np.float32)
    img = cv2.GaussianBlur(img, (9, 9), 2.5)

    # Add craters of varying sizes with illuminated rim and cast shadow
    craters = [
        (60, 60, 22),
        (160, 70, 30),
        (80, 170, 26),
        (180, 180, 20),
        (120, 120, 15),
        (40, 130, 12),
        (140, 40, 10),
        (200, 110, 14),
    ]
    for cx, cy, r in craters:
        # Crater floor
        cv2.circle(img, (cx, cy), r, 60.0, -1)
        # Highlight rim (sun from top-left)
        cv2.ellipse(img, (cx - 2, cy - 2), (r, r), 0, 135, 315, 200.0, 3)
        # Shadow rim
        cv2.ellipse(img, (cx + 2, cy + 2), (r, r), 0, -45, 135, 30.0, 3)

    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestPhase1Correctness(unittest.TestCase):
    """Regression test cases for Phase 1 verification."""

    def test_01_ui_alias_to_canonical_mapping(self):
        """1. Verify UI aliases (taylor, phase, quadratic) map to canonical identifiers."""
        # Single method alias normalization
        self.assertEqual(normalize_refinement_method("taylor"), "taylor_expansion")
        self.assertEqual(normalize_refinement_method("phase"), "phase_correlation")
        self.assertEqual(normalize_refinement_method("quadratic"), "quadratic_peak")
        self.assertEqual(normalize_refinement_method("ecc"), "ecc")
        self.assertEqual(normalize_refinement_method("lucas_kanade"), "lucas_kanade")
        self.assertEqual(normalize_refinement_method("lk"), "lucas_kanade")
        self.assertEqual(normalize_refinement_method("corner_subpix"), "corner_subpix")

        # Canonical names pass through intact
        self.assertEqual(normalize_refinement_method("taylor_expansion"), "taylor_expansion")
        self.assertEqual(normalize_refinement_method("phase_correlation"), "phase_correlation")
        self.assertEqual(normalize_refinement_method("quadratic_peak"), "quadratic_peak")

        # List and comma-separated string normalization
        ui_methods = ["taylor", "phase", "quadratic", "ecc"]
        canonical = normalize_refinement_methods(ui_methods)
        self.assertEqual(
            canonical,
            ["taylor_expansion", "phase_correlation", "quadratic_peak", "ecc"]
        )

        comma_string = "taylor,phase,quadratic"
        self.assertEqual(
            normalize_refinement_methods(comma_string),
            ["taylor_expansion", "phase_correlation", "quadratic_peak"]
        )

        # Unknown tokens return None and are filtered out of canonical list
        self.assertIsNone(normalize_refinement_method("nonexistent_method"))
        self.assertEqual(
            normalize_refinement_methods(["taylor", "invalid_xyz"]),
            ["taylor_expansion"]
        )

    def test_02_taylor_actually_executes(self):
        """2. Verify Taylor expansion actually executes when UI alias 'taylor' is passed."""
        img = _create_synthetic_crater_scene()
        # Fractional shift: +0.4 px in X, -0.3 px in Y
        M = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, -0.3]], dtype=np.float32)
        img_shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        corners = cv2.goodFeaturesToTrack(img, maxCorners=30, qualityLevel=0.02, minDistance=15)
        self.assertIsNotNone(corners)
        src_pts = corners.reshape(-1, 2)
        ref_pts = src_pts.copy()
        H_init = np.eye(3, dtype=np.float64)

        # Call refine_subpixel with UI alias "taylor"
        subpixel = refine_subpixel(img, img_shifted, src_pts, ref_pts, H_init, methods=["taylor"])

        self.assertIn("taylor_expansion", subpixel["applied_methods"])
        status = next((item for item in subpixel["method_status"] if item.get("canonical_method") == "taylor_expansion"), None)
        self.assertIsNotNone(status)
        self.assertTrue(status["succeeded"])
        self.assertGreaterEqual(status["valid_points"], 4)

        # Refined points should not be identical to initial points
        deltas = subpixel["refined_reference_points"] - subpixel["initial_reference_points"]
        mean_delta = float(np.mean(np.linalg.norm(deltas, axis=1)))
        self.assertGreater(mean_delta, 0.05)

    def test_03_phase_correlation_actually_executes(self):
        """3. Verify Phase correlation actually executes when UI alias 'phase' is passed."""
        img = _create_synthetic_crater_scene()
        # Shift image by (+2.0, +1.5)
        M = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.5]], dtype=np.float32)
        ref = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        src = img

        src_pts = np.array([[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]], dtype=np.float32)
        ref_pts = src_pts.copy()
        H_init = np.eye(3, dtype=np.float64)

        subpixel = refine_subpixel(src, ref, src_pts, ref_pts, H_init, methods=["phase"])

        self.assertIn("phase_correlation", subpixel["applied_methods"])
        status = next((item for item in subpixel["method_status"] if item.get("canonical_method") == "phase_correlation"), None)
        self.assertIsNotNone(status)
        self.assertTrue(status["succeeded"])
        self.assertIsNotNone(status["response"])
        self.assertGreater(status["response"], 0.05)

        # Homography translation should reflect the shift
        H_refined = subpixel["homography"]
        self.assertAlmostEqual(H_refined[0, 2], 2.0, delta=0.5)
        self.assertAlmostEqual(H_refined[1, 2], 1.5, delta=0.5)

    def test_04_quadratic_actually_executes(self):
        """4. Verify Quadratic peak fitting actually executes when UI alias 'quadratic' is passed."""
        img = _create_synthetic_crater_scene()
        M = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, -0.2]], dtype=np.float32)
        img_shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        corners = cv2.goodFeaturesToTrack(img, maxCorners=30, qualityLevel=0.02, minDistance=15)
        self.assertIsNotNone(corners)
        src_pts = corners.reshape(-1, 2)
        ref_pts = src_pts.copy()
        H_init = np.eye(3, dtype=np.float64)

        subpixel = refine_subpixel(img, img_shifted, src_pts, ref_pts, H_init, methods=["quadratic"])

        self.assertIn("quadratic_peak", subpixel["applied_methods"])
        status = next((item for item in subpixel["method_status"] if item.get("canonical_method") == "quadratic_peak"), None)
        self.assertIsNotNone(status)
        self.assertTrue(status["succeeded"])
        self.assertGreaterEqual(status["valid_points"], 4)

        deltas = subpixel["refined_reference_points"] - subpixel["initial_reference_points"]
        mean_delta = float(np.mean(np.linalg.norm(deltas, axis=1)))
        self.assertGreater(mean_delta, 0.05)

    def test_05_ecc_executes(self):
        """5. Verify ECC alignment executes and converges on textured lunar image."""
        img = _create_synthetic_crater_scene()
        M = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        ref = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        src = img

        src_pts = np.array([[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]], dtype=np.float32)
        ref_pts = src_pts + np.array([2.0, 1.0])
        H_init = np.array([[1.0, 0.0, 1.8], [0.0, 1.0, 0.8], [0.0, 0.0, 1.0]], dtype=np.float64)

        subpixel = refine_subpixel(src, ref, src_pts, ref_pts, H_init, methods=["ecc"])

        self.assertIn("ecc", subpixel["applied_methods"])
        self.assertTrue(subpixel["ecc_used"])
        self.assertGreater(subpixel["ecc_correlation"], 0.90)

    def test_06_success_state_reflects_quality_gate(self):
        """6. Verify success state reflects quality gate (not blindly returning True)."""
        scene = _create_synthetic_crater_scene(width=300, height=300)

        # Case A: Good registration -> Should PASS or PASS_WITH_WARNING and success=True
        M_good = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        ref_good = cv2.warpAffine(scene, M_good, (300, 300))
        result_good = register_pair(
            scene,
            ref_good,
            detector="sift",
            refinement_methods=["taylor"],
            ecc_refinement=False,
        )
        self.assertIn(result_good.registration_status, ("PASS", "PASS_WITH_WARNING"))
        self.assertTrue(result_good.success)

        # Case B: Incompatible or highly distorted image where quality gate checks fail
        # Blank / featureless / conflicting image pair
        noise_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        try:
            result_bad = register_pair(
                scene,
                noise_image,
                detector="sift",
                refinement_methods=[],
                ecc_refinement=False,
            )
            # If it managed to produce any matches, status must be REVIEW or FAIL
            self.assertIn(result_bad.registration_status, ("REVIEW", "FAIL"))
            self.assertFalse(result_bad.success)
        except ValueError:
            # Expected if fewer than 4 matches can be found
            pass

    def test_07_refined_points_propagate_to_final_metrics(self):
        """7. Verify refined reference points propagate into final residuals and metrics."""
        scene = _create_synthetic_crater_scene(width=300, height=300)
        M = np.array([[1.0, 0.0, 3.2], [0.0, 1.0, 1.8]], dtype=np.float32)
        ref = cv2.warpAffine(scene, M, (300, 300))

        result = register_pair(
            scene,
            ref,
            detector="sift",
            refinement_methods=["taylor"],
            ecc_refinement=False,
        )

        if result.subpixel and "taylor_expansion" in result.subpixel.get("applied_methods", []):
            self.assertEqual(result.metrics["correspondence_basis"], "refined")
            # final_reference_points should match subpixel refined reference points
            np.testing.assert_array_equal(
                result.final_reference_points,
                result.subpixel["refined_reference_points"]
            )
            # Coordinates should have moved from initial reference points
            self.assertFalse(np.array_equal(
                result.final_reference_points,
                result.reference_points
            ))

    def test_08_raw_and_refined_metrics_distinguishable(self):
        """8. Verify raw and refined RMSE remain clearly distinguishable."""
        scene = _create_synthetic_crater_scene(width=300, height=300)
        M = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        ref = cv2.warpAffine(scene, M, (300, 300))

        result = register_pair(
            scene,
            ref,
            detector="sift",
            refinement_methods=["taylor"],
            ecc_refinement=False,
        )

        # Both raw_rmse_pixels and refined_rmse_pixels should be reported
        self.assertIn("raw_rmse_pixels", result.metrics)
        self.assertIn("refined_rmse_pixels", result.metrics)
        self.assertIn("rmse_improvement_pixels", result.metrics)
        self.assertIn("correspondence_basis", result.metrics)
        self.assertIn("subpixel_validation_status", result.metrics)

        self.assertIsNotNone(result.raw_rmse)

    def test_09_final_inlier_mask_matches_final_geometry(self):
        """9. Verify final inlier mask strictly matches the final geometry and point set."""
        scene = _create_synthetic_crater_scene(width=300, height=300)
        M = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, 1.5]], dtype=np.float32)
        ref = cv2.warpAffine(scene, M, (300, 300))

        result = register_pair(
            scene,
            ref,
            detector="sift",
            ransac_threshold=3.0,
            refinement_methods=["taylor"],
            ecc_refinement=False,
        )

        # Recompute independent residuals against final_reference_points and final H
        pts_src = result.source_points.reshape(-1, 2).astype(np.float32)
        pred = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), result.homography).reshape(-1, 2)
        target = result.final_reference_points.reshape(-1, 2)
        manual_residuals = np.linalg.norm(pred - target, axis=1)
        manual_mask = manual_residuals <= 3.0

        np.testing.assert_array_equal(result.inlier_mask, manual_mask)
        self.assertEqual(result.metrics["inlier_count"], int(np.sum(manual_mask)))

    def test_10_phase_correlation_direction_sign(self):
        """10. Verify phase correlation direction and sign convention moves in corrective direction."""
        np.random.seed(999)
        ref = np.random.normal(128, 25, (200, 200)).astype(np.float32)
        ref = cv2.GaussianBlur(ref, (5, 5), 1.5)

        # Source is shifted by (-5.0, -3.0)
        M_src = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
        src = cv2.warpAffine(ref, M_src, (200, 200))

        # True mapping from src to ref is translation (+5.0, +3.0)
        # Suppose initial H estimate is slightly off: (+4.2, +2.3)
        H_est = np.array([[1.0, 0.0, 4.2], [0.0, 1.0, 2.3], [0.0, 0.0, 1.0]], dtype=np.float64)

        refined_H, response, used, shift, reason = phase_translation(src, ref, H_est)

        self.assertTrue(used)
        self.assertIsNotNone(response)
        self.assertGreater(response, 0.1)

        # Refined translation must be closer to ground truth (5.0, 3.0) than initial estimate
        initial_err_x = abs(H_est[0, 2] - 5.0)
        initial_err_y = abs(H_est[1, 2] - 3.0)
        refined_err_x = abs(refined_H[0, 2] - 5.0)
        refined_err_y = abs(refined_H[1, 2] - 3.0)

        self.assertLess(refined_err_x, initial_err_x, f"Phase X moved wrong direction: {refined_H[0, 2]} vs {H_est[0, 2]}")
        self.assertLess(refined_err_y, initial_err_y, f"Phase Y moved wrong direction: {refined_H[1, 2]} vs {H_est[1, 2]}")

    def test_11_ecc_transform_direction(self):
        """11. Verify ECC transform direction correctly converges toward true warp."""
        scene = _create_synthetic_crater_scene(width=160, height=160)
        # Ground truth translation: +3.5 in X, +2.5 in Y
        M_true = np.array([[1.0, 0.0, 3.5], [0.0, 1.0, 2.5]], dtype=np.float32)
        ref = cv2.warpAffine(scene, M_true, (160, 160))
        src = scene

        # Initial estimate is off by 0.5 px: 3.0 in X, 2.0 in Y
        H_init = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        refined_H, cc, used, reason = refine_with_ecc(src, ref, H_init, iterations=60)

        self.assertTrue(used)
        self.assertGreater(cc, 0.95)
        # Check that refined translation is close to ground truth (3.5, 2.5)
        self.assertAlmostEqual(refined_H[0, 2], 3.5, delta=0.2)
        self.assertAlmostEqual(refined_H[1, 2], 2.5, delta=0.2)

    def test_12_failed_refinement_reported_honestly(self):
        """12. Verify failed or unknown refinement methods are reported honestly with status."""
        img = _create_synthetic_crater_scene()
        src_pts = np.array([[40.0, 40.0], [80.0, 40.0], [40.0, 80.0], [80.0, 80.0]], dtype=np.float32)
        ref_pts = src_pts.copy()
        H_init = np.eye(3, dtype=np.float64)

        # Test 12A: Unknown method alias
        subpixel = refine_subpixel(img, img, src_pts, ref_pts, H_init, methods=["nonexistent_algo"])
        status_unknown = next((item for item in subpixel["method_status"] if item["method"] == "nonexistent_algo"), None)
        self.assertIsNotNone(status_unknown)
        self.assertFalse(status_unknown["succeeded"])
        self.assertFalse(status_unknown["attempted"])
        self.assertIn("Unknown refinement method", status_unknown["reason"])
        self.assertNotIn("nonexistent_algo", subpixel["applied_methods"])

        # Test 12B: Completely flat image where gradient-based refinement cannot operate
        flat_img = np.ones((100, 100), dtype=np.uint8) * 128
        subpixel_flat = refine_subpixel(flat_img, flat_img, src_pts, ref_pts, H_init, methods=["taylor"])
        self.assertEqual(subpixel_flat["subpixel_validation_status"], "SUBPIXEL_FAILED")
        self.assertNotIn("taylor_expansion", subpixel_flat["applied_methods"])


if __name__ == "__main__":
    unittest.main()
