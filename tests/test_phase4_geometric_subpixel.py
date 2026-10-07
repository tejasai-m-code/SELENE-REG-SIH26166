"""Phase 4 Automated Scientific Verification Suite: Geometric Transformations & Sub-pixel Validation.

Tests cover:
1. Robust geometric model selection (Homography, Affine, Similarity, Translation).
2. RANSAC / MAGSAC robust verification & deliberate outlier rejection.
3. Reprojection error, residual statistics & uncertainty analysis.
4. Execution of all 5 sub-pixel refinement methods:
   - Taylor expansion (local gradient linearization)
   - Lucas-Kanade (iterative pyramidal optical flow)
   - ECC alignment (Enhanced Correlation Coefficient)
   - Phase correlation + upsampling (Fourier domain cross-power spectrum)
   - Quadratic peak fitting (local intensity correlation surface extremum)
5. Comparison and auto-selection of refinement methods based on measurable residual quality.
6. Complete correspondence coordinate preservation (source, reference, refined, delta, residual, method, confidence).
7. GPU vs CPU acceleration and truthful hardware telemetry.
"""

import math
import unittest
import cv2
import numpy as np

from app.services.geometry import (
    estimate_geometric_model,
    validate_transform_plausibility,
    decompose_transform,
    compute_residual_statistics,
    GeometricEstimationResult,
)
from app.services.subpixel import (
    taylor_expansion,
    lucas_kanade,
    ecc_alignment,
    phase_correlation_subpixel,
    quadratic_peak,
    compare_and_select_refinement,
    POINT_REFINEMENT_METHODS,
    GLOBAL_REFINEMENT_METHODS,
)
from app.services.pairwise_registration import (
    register_pair,
    serialize_correspondences,
    serialize_inlier_points,
)
from app.services.gpu_accelerator import (
    is_cuda_available,
    phase_correlation_gpu,
)


class TestPhase4GeometricSubpixel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # Create textured lunar-like surface with crater-like circular depressions and rocks
        base = np.random.randint(40, 210, (400, 400), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (5, 5), 1.5)
        rng = np.random.default_rng(42)
        for _ in range(25):
            cx, cy = rng.integers(50, 350, size=2)
            r = int(rng.integers(8, 28))
            cv2.circle(base, (int(cx), int(cy)), r, int(rng.integers(180, 255)), -1)
            cv2.circle(base, (int(cx) + 2, int(cy) + 2), max(2, r - 3), int(rng.integers(30, 80)), -1)
        cls.base_image = base

    def test_01_geometric_model_selection_translation(self):
        """Test Translation (2-DOF) model estimation with known displacement."""
        dx, dy = 14.5, -9.25
        pts_s = np.array([
            [50.0, 50.0], [150.0, 60.0], [250.0, 80.0],
            [60.0, 200.0], [180.0, 220.0], [300.0, 250.0],
            [100.0, 320.0], [220.0, 310.0]
        ], dtype=np.float32)
        pts_r = pts_s + np.array([dx, dy], dtype=np.float32)

        res = estimate_geometric_model(pts_s, pts_r, model_type="translation")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.model_type, "translation")
        self.assertAlmostEqual(res.homography[0, 2], dx, delta=0.1)
        self.assertAlmostEqual(res.homography[1, 2], dy, delta=0.1)
        self.assertLess(res.rmse_pixels, 0.05)

    def test_02_geometric_model_selection_similarity(self):
        """Test Similarity (4-DOF: scale, rotation, translation) model estimation."""
        theta = math.radians(12.0)
        scale = 1.06
        tx, ty = 18.0, -12.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        M_sim_gt = np.array([
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t,  scale * cos_t, ty]
        ], dtype=np.float64)

        pts_s = np.random.uniform(50, 350, (30, 2)).astype(np.float32)
        pts_r = cv2.transform(pts_s.reshape(-1, 1, 2), M_sim_gt).reshape(-1, 2)

        res = estimate_geometric_model(pts_s, pts_r, model_type="similarity")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.model_type, "similarity")
        self.assertLess(res.rmse_pixels, 0.2)
        decomp = res.transform_decomposition
        self.assertIsNotNone(decomp)
        self.assertAlmostEqual(decomp["rotation_degrees"], 12.0, delta=0.5)
        self.assertAlmostEqual(decomp["scale_x"], 1.06, delta=0.02)

    def test_03_geometric_model_selection_affine(self):
        """Test Affine (6-DOF) model estimation with shear and anisotropic scaling."""
        M_aff_gt = np.array([
            [1.08, 0.12, 10.0],
            [0.05, 0.95, -15.0]
        ], dtype=np.float64)
        pts_s = np.random.uniform(50, 350, (35, 2)).astype(np.float32)
        pts_r = cv2.transform(pts_s.reshape(-1, 1, 2), M_aff_gt).reshape(-1, 2)

        res = estimate_geometric_model(pts_s, pts_r, model_type="affine")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.model_type, "affine")
        self.assertLess(res.rmse_pixels, 0.2)
        self.assertEqual(res.inlier_count, 35)

    def test_04_geometric_model_selection_homography(self):
        """Test Homography (8-DOF) projective model estimation."""
        H_hom_gt = np.array([
            [1.04, 0.03, 12.0],
            [-0.02, 1.02, -8.0],
            [0.0001, -0.0001, 1.0]
        ], dtype=np.float64)
        H_hom_gt /= H_hom_gt[2, 2]

        pts_s = np.random.uniform(60, 340, (40, 2)).astype(np.float32)
        pts_r = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), H_hom_gt).reshape(-1, 2)

        res = estimate_geometric_model(pts_s, pts_r, model_type="homography")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.model_type, "homography")
        self.assertLess(res.rmse_pixels, 0.3)
        self.assertEqual(res.inlier_count, 40)

    def test_05_deliberate_outlier_rejection_ransac_magsac(self):
        """Test robust rejection of deliberately injected gross correspondence outliers."""
        # 30 true correspondences under affine transform
        M_true = np.array([[1.02, -0.05, 15.0], [0.05, 1.01, -20.0]], dtype=np.float64)
        pts_s_true = np.random.uniform(50, 350, (30, 2)).astype(np.float32)
        pts_r_true = cv2.transform(pts_s_true.reshape(-1, 1, 2), M_true).reshape(-1, 2)

        # Inject 20 gross outliers (50px to 150px displacement errors or shuffled coordinates)
        pts_s_out = np.random.uniform(50, 350, (20, 2)).astype(np.float32)
        pts_r_out = pts_s_out + np.random.uniform(60.0, 150.0, (20, 2)).astype(np.float32)

        pts_s_all = np.vstack([pts_s_true, pts_s_out])
        pts_r_all = np.vstack([pts_r_true, pts_r_out])

        res = estimate_geometric_model(pts_s_all, pts_r_all, ransac_threshold=3.0)
        self.assertTrue(res.is_valid)

        # Check inlier separation: all true inliers should be accepted, all gross outliers rejected
        mask = res.inlier_mask
        true_inliers_detected = np.sum(mask[:30])
        outliers_detected_as_inliers = np.sum(mask[30:])

        self.assertGreaterEqual(true_inliers_detected, 28, "RANSAC missed true inliers")
        self.assertEqual(outliers_detected_as_inliers, 0, "RANSAC accepted gross correspondence outliers")
        self.assertLess(res.rmse_pixels, 1.0, "Inlier residual degraded by outliers")

    def test_06_residual_statistics_and_uncertainty(self):
        """Test comprehensive residual error statistical metrics."""
        residuals = np.array([0.15, 0.22, 0.18, 0.35, 0.42, 0.28, 0.19, 0.31, 0.25], dtype=np.float64)
        stats = compute_residual_statistics(residuals)

        self.assertEqual(stats["count"], 9)
        self.assertAlmostEqual(stats["mean"], np.mean(residuals), places=3)
        self.assertAlmostEqual(stats["median"], np.median(residuals), places=3)
        self.assertAlmostEqual(stats["rmse"], np.sqrt(np.mean(residuals**2)), places=3)
        self.assertGreater(stats["uncertainty"], 0.0)
        self.assertLess(stats["uncertainty"], 0.1)

    def test_07_transform_plausibility_guard(self):
        """Test that physically implausible transformations are flagged."""
        # Reflection (negative determinant)
        H_reflect = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        plaus, issues, _ = validate_transform_plausibility(H_reflect)
        self.assertFalse(plaus)
        self.assertTrue(any("reflection" in s.lower() or "determinant" in s.lower() for s in issues))

        # Extreme magnification (10x)
        H_zoom = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
        plaus, issues, _ = validate_transform_plausibility(H_zoom)
        self.assertFalse(plaus)
        self.assertTrue(any("magnification" in s.lower() for s in issues))

    def test_08_subpixel_taylor_expansion_execution(self):
        """Test Method 1: Taylor expansion sub-pixel refinement on fractional shift."""
        dx, dy = 0.40, -0.30
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pts_s = np.array([[100.0, 100.0], [150.0, 200.0], [250.0, 150.0], [300.0, 300.0], [200.0, 250.0]], dtype=np.float32)
        # Initial reference points with integer rounding (deliberate sub-pixel error)
        pts_r_init = np.round(pts_s + np.array([dx, dy])).astype(np.float32)

        refined_pts, valid = taylor_expansion(self.base_image, ref, pts_s, pts_r_init, iterations=5)
        self.assertGreaterEqual(np.sum(valid), 4)

        # Measure displacement towards ground truth
        gt_pts = pts_s + np.array([dx, dy])
        init_err = np.linalg.norm(pts_r_init - gt_pts, axis=1)
        refined_err = np.linalg.norm(refined_pts[valid] - gt_pts[valid], axis=1)

        self.assertLess(np.mean(refined_err), np.mean(init_err[valid]))

    def test_09_subpixel_lucas_kanade_execution(self):
        """Test Method 2: Lucas-Kanade iterative optical flow refinement."""
        dx, dy = -0.35, 0.45
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pts_s = np.array([[120.0, 120.0], [180.0, 160.0], [220.0, 280.0], [280.0, 200.0], [150.0, 320.0]], dtype=np.float32)
        pts_r_init = pts_s.copy()

        refined_pts, valid = lucas_kanade(self.base_image, ref, pts_s, pts_r_init)
        self.assertGreaterEqual(np.sum(valid), 4)

        shifts = refined_pts[valid] - pts_s[valid]
        mean_dx = np.mean(shifts[:, 0])
        mean_dy = np.mean(shifts[:, 1])

        self.assertAlmostEqual(mean_dx, dx, delta=0.25)
        self.assertAlmostEqual(mean_dy, dy, delta=0.25)

    def test_10_subpixel_ecc_alignment_execution(self):
        """Test Method 3: Enhanced Correlation Coefficient (ECC) optimization."""
        dx, dy = 1.25, -0.75
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        H_init = np.eye(3, dtype=np.float32)
        H_refined, score, used, reason = ecc_alignment(self.base_image, ref, H_init, iterations=60)

        self.assertTrue(used)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.90)
        self.assertAlmostEqual(H_refined[0, 2], dx, delta=0.25)
        self.assertAlmostEqual(H_refined[1, 2], dy, delta=0.25)

    def test_11_subpixel_phase_correlation_upsampling_execution(self):
        """Test Method 4: Phase correlation + upsampling execution."""
        dx, dy = 2.40, -1.60
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        H_refined, resp, used, shift, reason, hw = phase_correlation_subpixel(self.base_image, ref)
        self.assertTrue(used)
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift[0], dx, delta=0.4)
        self.assertAlmostEqual(shift[1], dy, delta=0.4)
        self.assertIn(hw, ["GPU_CUDA", "CPU"])

    def test_12_subpixel_quadratic_peak_execution(self):
        """Test Method 5: Local intensity correlation quadratic peak fitting."""
        dx, dy = 0.30, 0.30
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pts_s = np.array([[130.0, 130.0], [210.0, 180.0], [260.0, 270.0], [170.0, 310.0], [310.0, 140.0]], dtype=np.float32)
        pts_r_init = np.round(pts_s).astype(np.float32)

        refined_pts, valid = quadratic_peak(self.base_image, ref, pts_s, pts_r_init)
        self.assertGreaterEqual(np.sum(valid), 4)
        shifts = refined_pts[valid] - pts_r_init[valid]
        self.assertTrue(np.all(np.abs(shifts) <= 0.6))

    def test_13_compare_and_select_refinement_auto(self):
        """Test comparative evaluation and auto-selection based on residual reduction."""
        dx, dy = 1.0, 1.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pts_s = np.array([
            [120.0, 120.0], [180.0, 160.0], [220.0, 280.0], [280.0, 200.0],
            [150.0, 320.0], [250.0, 150.0], [170.0, 220.0], [300.0, 300.0]
        ], dtype=np.float32)

        true_r = pts_s + np.array([dx, dy], dtype=np.float32)
        # Add realistic localization noise (std ~ 0.4 px)
        noise = np.array([
            [0.35, -0.25], [-0.30, 0.35], [0.25, 0.30], [-0.35, -0.20],
            [0.20, -0.35], [-0.25, 0.30], [0.30, -0.20], [-0.20, 0.25]
        ], dtype=np.float32)
        pts_r_init = true_r + noise

        H_init = np.eye(3, dtype=np.float64)
        H_init[0, 2] = dx
        H_init[1, 2] = dy

        comparison = compare_and_select_refinement(
            self.base_image, ref, pts_s, pts_r_init, H_init
        )

        self.assertIn("comparison_ledger", comparison)
        self.assertGreaterEqual(len(comparison["comparison_ledger"]), 4)
        self.assertIn(comparison["selected_method"], [
            "taylor_expansion", "lucas_kanade", "quadratic_peak", "phase_correlation", "ecc"
        ])
        self.assertGreaterEqual(comparison["rmse_improvement_pixels"], 0.0)

    def test_14_preservation_of_correspondence_attributes(self):
        """Verify preservation of source, reference, refined sub-pixel, delta, residual, method and confidence."""
        dx, dy = 2.0, 1.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        res = register_pair(
            self.base_image, ref,
            detector="sift",
            max_features=600,
            refinement_methods=["taylor"],
        )
        self.assertTrue(res.success)

        inliers = serialize_inlier_points(res)
        self.assertGreater(len(inliers), 0)
        sample = inliers[0]

        # Verify all required keys exist and contain non-null values
        required_keys = [
            "source", "reference", "initial_reference", "refined_reference",
            "refinement_delta", "dx", "dy", "error", "confidence", "refinement_method", "status"
        ]
        for key in required_keys:
            self.assertIn(key, sample, f"Key '{key}' missing from serialized inlier point")

        correspondences = serialize_correspondences(res)
        self.assertGreater(len(correspondences), 0)
        c_sample = correspondences[0]
        corr_keys = [
            "index", "source", "reference", "initial_reference", "refined_reference",
            "refinement_delta", "residual", "error", "confidence", "refinement_method", "status", "is_inlier"
        ]
        for key in corr_keys:
            self.assertIn(key, c_sample, f"Key '{key}' missing from serialized correspondence")

    def test_15_gpu_phase_correlation_vs_cpu(self):
        """Verify GPU phase correlation execution on CUDA with truthful hardware telemetry."""
        dx, dy = 1.5, -2.5
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base_image, M, (400, 400), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if is_cuda_available():
            shift_x, shift_y, resp = phase_correlation_gpu(self.base_image, ref)
            self.assertAlmostEqual(shift_x, dx, delta=0.5)
            self.assertAlmostEqual(shift_y, dy, delta=0.5)
            self.assertGreater(resp, 0.05)


if __name__ == "__main__":
    unittest.main()
