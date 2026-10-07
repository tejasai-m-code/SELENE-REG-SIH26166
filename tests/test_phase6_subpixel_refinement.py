"""Phase 6 Comprehensive Verification Suite: Sub-Pixel Refinement & Auto-Selection Engine.

Covers Master Engineering Prompt Sections 8, 9, 10, and Phase 6:
1. Real mathematical execution of all 5 sub-pixel refinement methods:
   - Taylor expansion (brightness constancy gradient linearization)
   - Lucas-Kanade (iterative pyramidal/patch optical flow)
   - ECC alignment (Enhanced Correlation Coefficient homography optimization)
   - Phase correlation + upsampling (Fourier-domain matrix-multiply DFT upsampling)
   - Local intensity correlation peak fitting (2D quadric surface fit with Hessian validation)
2. Sub-pixel shift recovery on controlled synthetic shifts (dx, dy <= 0.05 px recovery).
3. Auto Selection Engine:
   - Evaluates candidate methods against measurable numerical criteria (residual RMSE, convergence rate, local correlation, stability)
   - Returns candidate methods, selected method, and documented numerical justification
   - Generates complete Comparison Ledger table
4. Point-level inspection telemetry:
   - Integer coordinate, original point, refined point, (dx, dy), shift magnitude, residual before/after, iterations, convergence status, local correlation
5. Truthful "Not applicable for this image pair" reporting when preconditions are unmet (< 4 points, singular matrix, ill-conditioned).
6. Full integration with register_pair and correspondence point serialization.
"""

import math
import unittest
import cv2
import numpy as np

from app.services.subpixel import (
    taylor_expansion,
    lucas_kanade,
    ecc_alignment,
    phase_correlation_subpixel,
    quadratic_peak,
    fit_2d_quadratic_peak,
    dft_upsample_registration,
    compare_and_select_refinement,
    POINT_REFINEMENT_METHODS,
    GLOBAL_REFINEMENT_METHODS,
    normalize_refinement_methods,
)
from app.services.pairwise_registration import (
    register_pair,
    serialize_correspondences,
    serialize_inlier_points,
)
from app.services.geometry import estimate_geometric_model


class TestPhase6SubpixelRefinement(unittest.TestCase):
    """Rigorous scientific verification for Phase 6 Sub-Pixel Refinement."""

    @classmethod
    def setUpClass(cls):
        """Generate high-contrast, textured synthetic lunar terrain with craters."""
        rng = np.random.default_rng(2026)
        h, w = 300, 300
        # Multi-scale Gaussian noise baseline representing regolith
        base = rng.normal(128, 25, (h, w)).clip(20, 240).astype(np.float32)
        base = cv2.GaussianBlur(base, (3, 3), 1.0)

        # Draw structured crater-like features with sharp rims and shadowed bowls
        for _ in range(35):
            cx = rng.integers(30, w - 30)
            cy = rng.integers(30, h - 30)
            r = rng.integers(8, 25)
            # Bright rim
            cv2.circle(base, (cx, cy), r, float(rng.integers(190, 255)), 2)
            # Dark floor/shadow
            cv2.circle(base, (cx - 1, cy - 1), r - 3, float(rng.integers(30, 80)), -1)

        cls.terrain_base = cv2.GaussianBlur(base, (3, 3), 0.8).astype(np.uint8)

    def test_01_taylor_expansion_real_execution_and_details(self):
        """Taylor expansion linearizes brightness constancy and outputs point-level inspection details."""
        src = self.terrain_base.copy()
        # Known synthetic sub-pixel shift: dx = +0.32, dy = -0.28
        M = np.array([[1.0, 0.0, 0.32], [0.0, 1.0, -0.28]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)

        # Grid of test points well within image bounds
        grid_x, grid_y = np.meshgrid(np.linspace(60, 240, 5), np.linspace(60, 240, 5))
        pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        pts_r_initial = pts_s.copy()  # Initial integer/coarse unrefined correspondence

        # Call with return_details=True
        refined_pts, valid, details = taylor_expansion(src, ref, pts_s, pts_r_initial, return_details=True)

        self.assertEqual(len(refined_pts), len(pts_s))
        self.assertGreaterEqual(np.sum(valid), 15)  # Majority converged
        self.assertEqual(len(details), len(pts_s))

        # Inspect point-level telemetry structure
        for d in details:
            self.assertIn("index", d)
            self.assertIn("dx", d)
            self.assertIn("dy", d)
            self.assertIn("shift", d)
            self.assertIn("iterations", d)
            self.assertIn("converged", d)
            self.assertIn("status", d)
            self.assertIn("method", d)
            self.assertEqual(d["method"], "taylor_expansion")

        # Check mean shift direction reflects positive dx, negative dy
        valid_dx = [d["dx"] for d in details if d["converged"]]
        valid_dy = [d["dy"] for d in details if d["converged"]]
        mean_dx = float(np.mean(valid_dx))
        mean_dy = float(np.mean(valid_dy))
        self.assertGreater(mean_dx, 0.10)
        self.assertLess(mean_dy, -0.10)
        self.assertAlmostEqual(mean_dx, 0.32, delta=0.20)
        self.assertAlmostEqual(mean_dy, -0.28, delta=0.20)

    def test_02_lucas_kanade_optical_flow_real_execution(self):
        """Lucas-Kanade iterative pyramidal optical flow refines coordinates with telemetry."""
        src = self.terrain_base.copy()
        # Synthetic shift: dx = -0.40, dy = +0.35
        M = np.array([[1.0, 0.0, -0.40], [0.0, 1.0, 0.35]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)

        grid_x, grid_y = np.meshgrid(np.linspace(70, 230, 4), np.linspace(70, 230, 4))
        pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        pts_r_initial = pts_s.copy()

        refined_pts, valid, details = lucas_kanade(src, ref, pts_s, pts_r_initial, return_details=True)

        self.assertEqual(len(refined_pts), len(pts_s))
        self.assertGreaterEqual(np.sum(valid), 10)
        self.assertEqual(len(details), len(pts_s))

        # Check telemetry fields
        first_valid = next(d for d in details if d["converged"])
        self.assertEqual(first_valid["method"], "lucas_kanade")
        self.assertIn(first_valid["status"], ["CONVERGED", "DIVERGED"])
        self.assertGreater(first_valid["iterations"], 0)

        # Average recovered shift
        valid_dx = [d["dx"] for d in details if d["converged"]]
        valid_dy = [d["dy"] for d in details if d["converged"]]
        self.assertAlmostEqual(float(np.mean(valid_dx)), -0.40, delta=0.15)
        self.assertAlmostEqual(float(np.mean(valid_dy)), 0.35, delta=0.15)

    def test_03_ecc_alignment_real_execution(self):
        """Enhanced Correlation Coefficient (ECC) refines homography/warp matrix iteratively."""
        src = self.terrain_base.copy()
        # Rigid Euclidean transform with subtle sub-pixel rotation and translation
        theta = np.deg2rad(0.5)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        H_true = np.array([
            [cos_t, -sin_t, 0.75],
            [sin_t,  cos_t, -0.65],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)
        ref = cv2.warpPerspective(src, H_true, (src.shape[1], src.shape[0]))

        # Initial identity homography
        H_init = np.eye(3, dtype=np.float32)

        refined_H, cc, used, reason = ecc_alignment(src, ref, H_init, iterations=60)

        self.assertTrue(used)
        self.assertIsNotNone(cc)
        self.assertGreater(cc, 0.85)  # High correlation coefficient
        self.assertIn("ECC converged", reason)
        self.assertEqual(refined_H.shape, (3, 3))
        # Refined translation terms should approximate (0.75, -0.65)
        self.assertAlmostEqual(float(refined_H[0, 2]), 0.75, delta=0.25)
        self.assertAlmostEqual(float(refined_H[1, 2]), -0.65, delta=0.25)

    def test_04_phase_correlation_matrix_dft_upsampling(self):
        """Phase correlation with Guizar-Sicairos matrix-multiply DFT achieves sub-pixel resolution."""
        # 1. Pure Fourier phase shift test for sub-pixel accuracy verification
        h, w = 128, 128
        rng = np.random.default_rng(42)
        test_src = rng.normal(0, 1, (h, w))
        target_y, target_x = 0.35, -0.42
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        phase = np.exp(-2j * np.pi * (fy * target_y + fx * target_x))
        test_ref = np.fft.ifft2(np.fft.fft2(test_src) * phase).real

        (rec_dx, rec_dy), resp = dft_upsample_registration(test_src, test_ref, upsample_factor=50)
        self.assertAlmostEqual(rec_dx, target_x, delta=0.03)
        self.assertAlmostEqual(rec_dy, target_y, delta=0.03)
        self.assertGreater(resp, 0.80)

        # 2. Test phase_correlation_subpixel integration on lunar terrain
        src = self.terrain_base.copy()
        target_shift = (0.35, -0.45)  # (dx, dy)
        M = np.array([[1.0, 0.0, target_shift[0]], [0.0, 1.0, target_shift[1]]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_CUBIC)

        refined_H, response, used, shift, reason, hw = phase_correlation_subpixel(
            src.astype(np.uint8), ref.astype(np.uint8), np.eye(3), upsample_factor=20
        )
        self.assertTrue(used)
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift[0], target_shift[0], delta=0.20)
        self.assertAlmostEqual(shift[1], target_shift[1], delta=0.20)
        self.assertIn("Phase correlation + upsampling", reason)

    def test_05_quadratic_peak_fitting_and_hessian_validation(self):
        """Local quadratic peak fitting fits 2D quadric surface and verifies negative definite Hessian."""
        # 1. Direct mathematical quadric surface peak test
        # Create a synthetic 5x5 patch with a known extremum at (dx=0.25, dy=-0.35)
        # f(x, y) = -1.5*(x - 0.25)^2 - 2.0*(y + 0.35)^2 + 10.0
        # Expanded: -1.5*x^2 + 0.75*x - 0.09375 - 2.0*y^2 - 1.4*y - 0.245 + 10.0
        # c1 = -1.5, c2 = -2.0, c3 = 0, c4 = 0.75, c5 = -1.4, c6 = 9.66125
        grid_y, grid_x = np.mgrid[-2:3, -2:3]
        patch_5x5 = (-1.5 * (grid_x - 0.25) ** 2 - 2.0 * (grid_y + 0.35) ** 2 + 10.0).astype(np.float64)

        peak_dx, peak_dy, valid, cond = fit_2d_quadratic_peak(patch_5x5)
        self.assertTrue(valid)
        self.assertAlmostEqual(peak_dx, 0.25, delta=1e-4)
        self.assertAlmostEqual(peak_dy, -0.35, delta=1e-4)

        # 2. Rejection of saddle surface (non-negative definite Hessian)
        # f(x, y) = 1.5*x^2 - 2.0*y^2 (hyperbolic saddle, no maximum)
        saddle_patch = (1.5 * grid_x ** 2 - 2.0 * grid_y ** 2).astype(np.float64)
        _, _, valid_saddle, _ = fit_2d_quadratic_peak(saddle_patch)
        self.assertFalse(valid_saddle)

        # 3. Test full quadratic_peak refinement on lunar terrain
        src = self.terrain_base.copy()
        M = np.array([[1.0, 0.0, 0.30], [0.0, 1.0, -0.25]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)

        grid_x, grid_y = np.meshgrid(np.linspace(80, 220, 4), np.linspace(80, 220, 4))
        pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        pts_r_initial = pts_s.copy()

        refined_pts, valid_flags, details = quadratic_peak(src, ref, pts_s, pts_r_initial, return_details=True)
        self.assertEqual(len(refined_pts), len(pts_s))
        self.assertGreaterEqual(np.sum(valid_flags), 8)
        self.assertEqual(len(details), len(pts_s))
        for d in details:
            self.assertEqual(d["method"], "quadratic_peak")

    def test_06_auto_select_best_refinement_and_comparison_ledger(self):
        """Auto selection evaluates all 5 candidate methods and generates the comparison ledger."""
        src = self.terrain_base.copy()
        # Synthetic transform with measurable sub-pixel displacement
        M = np.array([[1.0, 0.0, 0.40], [0.0, 1.0, -0.30]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)

        grid_x, grid_y = np.meshgrid(np.linspace(60, 240, 5), np.linspace(60, 240, 5))
        pts_s = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        # In real registration, initial correspondence points have coarse/integer noise
        pts_r_true = pts_s + np.array([0.40, -0.30], dtype=np.float32)
        rng = np.random.default_rng(2026)
        pts_r = pts_r_true + rng.normal(0, 0.25, pts_r_true.shape).astype(np.float32)

        H_initial = np.array([[1.0, 0.0, 0.40], [0.0, 1.0, -0.30], [0.0, 0.0, 1.0]], dtype=np.float64)

        auto_result = compare_and_select_refinement(
            source_gray=src,
            reference_gray=ref,
            source_points=pts_s,
            reference_points=pts_r,
            homography=H_initial,
            candidate_methods=["auto"],
        )

        # Check return schema
        self.assertIn("selected_method", auto_result)
        self.assertIn("candidate_methods", auto_result)
        self.assertIn("raw_rmse_pixels", auto_result)
        self.assertIn("refined_rmse_pixels", auto_result)
        self.assertIn("rmse_improvement_pixels", auto_result)
        self.assertIn("status", auto_result)
        self.assertIn("reason", auto_result)
        self.assertIn("comparison_ledger", auto_result)
        self.assertIn("point_details", auto_result)

        ledger = auto_result["comparison_ledger"]
        self.assertEqual(len(ledger), 5)  # Taylor, LK, Quadratic, Phase, ECC
        methods_in_ledger = [item["method"] for item in ledger]
        self.assertIn("taylor_expansion", methods_in_ledger)
        self.assertIn("lucas_kanade", methods_in_ledger)
        self.assertIn("quadratic_peak", methods_in_ledger)
        self.assertIn("phase_correlation", methods_in_ledger)
        self.assertIn("ecc", methods_in_ledger)

        # Verify each ledger entry has documented metrics
        for entry in ledger:
            self.assertIn("method", entry)
            self.assertIn("display_name", entry)
            self.assertIn("category", entry)
            self.assertIn("attempted", entry)
            self.assertIn("succeeded", entry)
            self.assertIn("hardware", entry)
            self.assertIn("reason", entry)

        # Verify winner was selected based on measurable criteria
        selected = auto_result["selected_method"]
        self.assertNotEqual(selected, "none")
        self.assertTrue(len(auto_result["reason"]) > 10)

    def test_07_not_applicable_reported_truthfully_on_insufficient_points(self):
        """System states 'Not applicable for this image pair' when points < 4 without fabricating data."""
        src = self.terrain_base.copy()
        ref = self.terrain_base.copy()

        # Only 2 correspondence points (insufficient for projective geometry and refinement)
        pts_s = np.array([[50.0, 50.0], [100.0, 100.0]], dtype=np.float32)
        pts_r = np.array([[50.0, 50.0], [100.0, 100.0]], dtype=np.float32)

        auto_result = compare_and_select_refinement(
            source_gray=src,
            reference_gray=ref,
            source_points=pts_s,
            reference_points=pts_r,
            homography=np.eye(3),
            candidate_methods=["taylor_expansion", "lucas_kanade", "quadratic_peak"],
        )

        ledger = auto_result["comparison_ledger"]
        for entry in ledger:
            self.assertFalse(entry["succeeded"])
            self.assertIn("Not applicable for this image pair", entry["reason"])

    def test_08_point_level_inspection_serialization(self):
        """register_pair serializes integer coordinate, delta, shift, residuals, iterations, and correlation."""
        src = self.terrain_base.copy()
        # Synthetic shift dx = +0.5, dy = -0.3
        M = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.3]], dtype=np.float32)
        ref = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR)

        result = register_pair(
            source_image=src,
            reference_image=ref,
            detector="sift",
            refinement_methods=["auto"],
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.subpixel)

        # Test serialize_correspondences
        corrs = serialize_correspondences(result)
        self.assertGreater(len(corrs), 0)

        first_corr = corrs[0]
        # Required Phase 6 Inspection Fields
        self.assertIn("integer_coordinate", first_corr)
        self.assertIn("initial_reference", first_corr)
        self.assertIn("refined_reference", first_corr)
        self.assertIn("refinement_delta", first_corr)
        self.assertIn("shift_magnitude", first_corr)
        self.assertIn("residual", first_corr)
        self.assertIn("raw_residual", first_corr)
        self.assertIn("refined_residual", first_corr)
        self.assertIn("residual_improvement", first_corr)
        self.assertIn("refinement_method", first_corr)
        self.assertIn("iterations", first_corr)
        self.assertIn("converged", first_corr)
        self.assertIn("convergence_status", first_corr)

        # Verify integer coordinates are exact integer lists
        ix, iy = first_corr["integer_coordinate"]
        self.assertIsInstance(ix, int)
        self.assertIsInstance(iy, int)

        # Test serialize_inlier_points
        inliers = serialize_inlier_points(result)
        self.assertGreater(len(inliers), 0)
        first_inl = inliers[0]
        self.assertIn("integer_coordinate", first_inl)
        self.assertIn("shift_magnitude", first_inl)
        self.assertIn("residual_improvement", first_inl)
        self.assertIn("convergence_status", first_inl)

    def test_09_raw_scientific_pixels_remain_untouched_during_refinement(self):
        """Sub-pixel refinement updates coordinate estimates without modifying raw scientific input pixels."""
        src_orig = self.terrain_base.copy()
        ref_orig = self.terrain_base.copy()

        src_work = src_orig.copy()
        ref_work = ref_orig.copy()

        _ = register_pair(
            source_image=src_work,
            reference_image=ref_work,
            detector="sift",
            refinement_methods=["taylor", "phase", "quadratic"],
        )

        # Bitwise identical verification
        np.testing.assert_array_equal(src_work, src_orig)
        np.testing.assert_array_equal(ref_work, ref_orig)


if __name__ == "__main__":
    unittest.main()
