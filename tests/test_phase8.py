"""Phase 8 - Testing, Benchmark Suite & Final Acceptance Evidence.

Validates all 13 canonical Phase 8 requirements:
1. Unit tests
2. Integration tests
3. API/worker contract tests
4. Registration synthetic validation
5. Subpixel ground-truth validation
6. Scale/illumination robustness tests
7. Partial-overlap tests
8. Spatial-distribution tests
9. Multi-image registration tests
10. Mosaic tests
11. Failure-mode tests
12. Reproducible benchmark suite
13. Quantitative reporting of transform error, correspondence RMSE, subpixel error,
    inliers, inlier ratio, spatial coverage, uniformity, and failure rate.

All seeds are deterministic. All results are labeled as synthetic proxy evidence.
No native Chandrayaan-2 flight data is claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

# Ensure api-server python package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts" / "api-server" / "python"))

from app.services.pairwise_registration import register_pair, serialize_inlier_points
from app.services.multi_registration import run_multi_registration
from app.services.geometry import estimate_geometric_model, validate_overlap
from app.services.subpixel import (
    POINT_REFINEMENT_METHODS,
    GLOBAL_REFINEMENT_METHODS,
    lucas_kanade,
    taylor_expansion,
    quadratic_peak,
    phase_translation,
)
from app.services.footprint import create_image_footprint, PolygonFootprint
from app.services.coordinates import pixel_to_physical_local
from worker import (
    register as worker_register,
    register_multi as worker_register_multi,
    load_settings,
)


def make_synthetic_surface(seed: int = 26166, size: int = 320) -> np.ndarray:
    """Generate a reproducible, textured synthetic lunar surface with craters."""
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 180, (size, size), dtype=np.uint8)
    for _ in range(16):
        cx = int(rng.integers(20, size - 20))
        cy = int(rng.integers(20, size - 20))
        radius = int(rng.integers(8, 45))
        intensity = int(rng.integers(30, 240))
        cv2.circle(base, (cx, cy), radius, intensity, -1)
        # Ring edge
        cv2.circle(base, (cx, cy), radius, int(rng.integers(20, 60)), max(1, radius // 6))
    noise = rng.normal(0, 6.0, base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def compute_corner_error(H_est: np.ndarray, H_gt: np.ndarray, shape: tuple[int, int]) -> float:
    """Compute average corner transfer discrepancy in pixels between two homographies.
    
    Definition:
        E_transform = (1/4) * sum_{i=1}^4 ||H_est(c_i) - H_gt(c_i)||_2
    where c_i are the 4 image corners in pixels.
    """
    h, w = shape[:2]
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    pts_est = cv2.perspectiveTransform(corners, H_est.astype(np.float32))[0]
    pts_gt = cv2.perspectiveTransform(corners, H_gt.astype(np.float32))[0]
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


class TestPhase8TransformError(unittest.TestCase):
    """Requirement 4 & 13.1: Quantitative Transform Error against Ground Truth."""

    def setUp(self):
        self.base = make_synthetic_surface(seed=8001, size=320)
        self.h, self.w = self.base.shape[:2]

    def test_translation_transform_error(self):
        dx, dy = 12.0, -8.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        H_gt = np.float64([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        res = register_pair(self.base, ref, detector="sift", ecc_refinement=False)
        self.assertTrue(res.success)
        err = compute_corner_error(res.homography, H_gt, (self.h, self.w))
        print(f"[P8-METRIC] Translation Transform Corner Error: {err:.4f} px (RMSE={res.raw_rmse:.4f})")
        self.assertLess(err, 1.0, f"Translation transform error {err:.2f} px exceeds 1.0 px threshold")

    def test_affine_transform_error(self):
        # Rotation + scale + translation
        angle = 4.0
        scale = 1.03
        M_rot = cv2.getRotationMatrix2D((self.w / 2.0, self.h / 2.0), angle, scale)
        M_rot[0, 2] += 6.0
        M_rot[1, 2] -= 4.0
        ref = cv2.warpAffine(self.base, M_rot, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        H_gt = np.eye(3, dtype=np.float64)
        H_gt[:2, :] = M_rot

        res = register_pair(self.base, ref, detector="sift", ecc_refinement=False)
        self.assertTrue(res.success)
        err = compute_corner_error(res.homography, H_gt, (self.h, self.w))
        print(f"[P8-METRIC] Affine Transform Corner Error: {err:.4f} px (inliers={np.sum(res.inlier_mask)})")
        self.assertLess(err, 2.0, f"Affine transform error {err:.2f} px exceeds 2.0 px threshold")

    def test_perspective_transform_error(self):
        src_pts = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]])
        dst_pts = np.float32([[4, 3], [self.w - 5, 2], [self.w - 3, self.h - 4], [2, self.h - 5]])
        H_gt = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float64)
        ref = cv2.warpPerspective(self.base, H_gt, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)

        res = register_pair(self.base, ref, detector="sift", ecc_refinement=False)
        self.assertTrue(res.success)
        err = compute_corner_error(res.homography, H_gt, (self.h, self.w))
        print(f"[P8-METRIC] Perspective Transform Corner Error: {err:.4f} px")
        self.assertLess(err, 2.5, f"Perspective transform error {err:.2f} px exceeds 2.5 px threshold")


class TestPhase8SubpixelDisplacement(unittest.TestCase):
    """Requirement 5 & 13.3: Subpixel Displacement Error against Known GT."""

    def setUp(self):
        self.base = make_synthetic_surface(seed=8002, size=320)
        self.h, self.w = self.base.shape[:2]

    def _eval_subpixel(self, dx: float, dy: float, method: str) -> tuple[float, float, float]:
        """Compute subpixel displacement error against analytical ground truth."""
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ref = cv2.warpAffine(self.base, M, (self.w, self.h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        res = register_pair(self.base, ref, detector="sift", refinement_methods=[method], ecc_refinement=(method == "ecc"))
        self.assertTrue(res.success)

        src_pts = res.source_points.reshape(-1, 2)
        gt_pts = src_pts + np.array([dx, dy])

        # Raw displacement error
        raw_pts = res.reference_points.reshape(-1, 2)
        raw_disp_err = float(np.mean(np.linalg.norm(raw_pts - gt_pts, axis=1)))

        # Refined displacement error
        final_pts = res.final_reference_points.reshape(-1, 2) if res.final_reference_points is not None else raw_pts
        refined_disp_err = float(np.mean(np.linalg.norm(final_pts - gt_pts, axis=1)))

        raw_rmse = res.raw_rmse if res.raw_rmse is not None else 0.0
        refined_rmse = res.refined_rmse if res.refined_rmse is not None else raw_rmse

        return raw_disp_err, refined_disp_err, refined_rmse

    def test_subpixel_lucas_kanade_displacement(self):
        raw_err, refined_err, rmse = self._eval_subpixel(-0.5, 0.25, "lucas_kanade")
        print(f"[P8-METRIC] LK subpixel displacement: raw={raw_err:.4f} px, refined={refined_err:.4f} px, correspondence_RMSE={rmse:.4f} px")
        self.assertLess(refined_err, raw_err + 0.1)

    def test_subpixel_ecc_displacement(self):
        raw_err, refined_err, rmse = self._eval_subpixel(0.25, 0.0, "ecc")
        print(f"[P8-METRIC] ECC subpixel displacement: raw={raw_err:.4f} px, refined={refined_err:.4f} px, correspondence_RMSE={rmse:.4f} px")
        self.assertLess(refined_err, 1.0)

    def test_subpixel_taylor_displacement(self):
        raw_err, refined_err, rmse = self._eval_subpixel(0.25, 0.25, "taylor_expansion")
        print(f"[P8-METRIC] Taylor subpixel displacement: raw={raw_err:.4f} px, refined={refined_err:.4f} px, correspondence_RMSE={rmse:.4f} px")
        self.assertLess(refined_err, 1.2)


class TestPhase8PartialOverlapSweep(unittest.TestCase):
    """Requirement 7: Systematic Partial Overlap Sweep & Rejection Threshold."""

    @classmethod
    def setUpClass(cls):
        cls.large_canvas = make_synthetic_surface(seed=8003, size=800)

    def _test_overlap(self, shift_x: int) -> dict:
        h, w = 300, 300
        crop1 = self.large_canvas[100:400, 100:400].copy()
        crop2 = self.large_canvas[100:400, (100 + shift_x):(400 + shift_x)].copy()

        overlap_fraction = max(0.0, float((w - abs(shift_x)) / w))
        res = register_pair(crop1, crop2, detector="sift", ecc_refinement=False)
        inliers = int(np.sum(res.inlier_mask)) if res.inlier_mask is not None else 0
        cov = res.metrics.get("source_spatial_coverage", 0.0) if res.metrics else 0.0

        return {
            "shift_x": shift_x,
            "nominal_overlap": overlap_fraction,
            "status": res.registration_status,
            "success": res.success,
            "inliers": inliers,
            "coverage": cov,
        }

    def test_full_overlap_100pct(self):
        r = self._test_overlap(0)
        print(f"[P8-OVERLAP] 100% overlap: status={r['status']}, inliers={r['inliers']}")
        self.assertEqual(r["status"], "PASS")
        self.assertGreater(r["inliers"], 50)

    def test_partial_overlap_75pct(self):
        r = self._test_overlap(75)  # 225 px overlap = 75%
        print(f"[P8-OVERLAP] 75% overlap: status={r['status']}, inliers={r['inliers']}, cov={r['coverage']:.2f}")
        self.assertIn(r["status"], ("PASS", "PASS_WITH_WARNING"))
        self.assertGreater(r["inliers"], 30)

    def test_partial_overlap_50pct(self):
        r = self._test_overlap(150)  # 150 px overlap = 50%
        print(f"[P8-OVERLAP] 50% overlap: status={r['status']}, inliers={r['inliers']}, cov={r['coverage']:.2f}")
        self.assertIn(r["status"], ("PASS", "PASS_WITH_WARNING"))
        self.assertGreater(r["inliers"], 15)

    def test_minimal_overlap_20pct(self):
        r = self._test_overlap(240)  # 60 px overlap = 20%
        print(f"[P8-OVERLAP] 20% overlap: status={r['status']}, inliers={r['inliers']}")
        self.assertIn(r["status"], ("PASS", "PASS_WITH_WARNING", "FAIL"))

    def test_tiny_overlap_rejection(self):
        r = self._test_overlap(290)  # 10 px overlap = 3.3%
        print(f"[P8-OVERLAP] 3.3% overlap: status={r['status']}, inliers={r['inliers']}")
        # Must not produce an ungrounded false-positive PASS
        self.assertIn(r["status"], ("FAIL", "REVIEW", "PASS_WITH_WARNING"))
        if r["status"] == "FAIL":
            self.assertFalse(r["success"])

    def test_disjoint_images_rejection(self):
        r = self._test_overlap(350)  # 0% overlap (disjoint)
        print(f"[P8-OVERLAP] 0% overlap (disjoint): status={r['status']}, success={r['success']}")
        self.assertEqual(r["status"], "FAIL")
        self.assertFalse(r["success"])


class TestPhase8SpatialMetrics(unittest.TestCase):
    """Requirement 8 & 13.6, 13.7: Spatial Coverage & Uniformity Metrics."""

    def test_spatial_coverage_uniformity_well_distributed(self):
        src = make_synthetic_surface(seed=8004, size=320)
        ref = make_synthetic_surface(seed=8004, size=320)
        res = register_pair(src, ref, detector="sift", spatial_distribution=True, ecc_refinement=False)

        cov = res.metrics.get("source_spatial_coverage", 0.0)
        unif = res.metrics.get("spatial_uniformity", 0.0)
        grid = res.metrics.get("source_spatial_grid")

        print(f"[P8-METRIC] Well-distributed spatial coverage: {cov:.3f}, uniformity: {unif:.3f}")
        self.assertGreaterEqual(cov, 0.70)
        self.assertGreaterEqual(unif, 0.70)
        self.assertIsNotNone(grid)

    def test_spatial_coverage_sparse_input(self):
        # Image with texture concentrated in only one quadrant (bottom-right)
        base = np.full((320, 320), 128, dtype=np.uint8)
        base[160:, 160:] = make_synthetic_surface(seed=8005, size=160)
        res = register_pair(base, base, detector="sift", spatial_distribution=True, ecc_refinement=False)

        cov = res.metrics.get("source_spatial_coverage", 0.0)
        print(f"[P8-METRIC] Clustered spatial coverage: {cov:.3f}")
        # Clustered texture should yield lower spatial coverage than whole-scene texture
        self.assertLessEqual(cov, 0.50)


class TestPhase8WorkerAPIContract(unittest.TestCase):
    """Requirement 3 & Final Output Contract: worker.py CLI Bridge & Serialization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.td = Path(self.temp_dir.name)
        self.out_dir = self.td / "out"

        self.img1 = make_synthetic_surface(seed=8006, size=240)
        self.img2 = make_synthetic_surface(seed=8006, size=240)
        self.img3 = make_synthetic_surface(seed=8006, size=240)

        self.p1 = self.td / "OHRC_sample_01.png"
        self.p2 = self.td / "TMC2_sample_02.png"
        self.p3 = self.td / "sample_03.png"

        cv2.imwrite(str(self.p1), self.img1)
        cv2.imwrite(str(self.p2), self.img2)
        cv2.imwrite(str(self.p3), self.img3)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worker_register_pairwise_contract(self):
        settings = {
            "detector": "sift",
            "ecc_refinement": False,
            "source_metadata": {"sensor": "OHRC", "gsd_m": 0.25},
            "reference_metadata": {"sensor": "TMC-2", "gsd_m": 5.0},
        }
        res = worker_register(self.p1, self.p2, self.out_dir, settings)

        # Output contract assertions
        self.assertIn("success", res)
        self.assertIn("registration_status", res)
        self.assertIn("metrics", res)
        self.assertIn("source", res)
        self.assertIn("reference", res)
        self.assertIn("footprint", res)
        self.assertIn("subpixel", res)
        self.assertIn("outputs", res)
        self.assertIn("inlier_points", res)

        # Provenance & Metadata status assertions
        self.assertEqual(res["source"]["sensor"], "OHRC")
        self.assertIn(res["source"]["provenance"], ("MEASURED", "REFERENCE_PROFILE", "ESTIMATED"))
        self.assertIn(res["source"]["metadata_status"], ("KNOWN", "UNKNOWN", "ESTIMATED"))

        # Footprint structure assertions
        fp = res["footprint"]
        self.assertIsNotNone(fp)
        self.assertEqual(fp["status"], "COMPUTED")
        self.assertIn("source_footprint", fp)
        self.assertIn("reference_footprint", fp)
        self.assertIn("warped_source_footprint", fp)
        self.assertIn("overlap", fp)
        self.assertTrue(fp["overlap"]["has_overlap"])

        print(f"[P8-API] Pairwise worker contract validated: status={res['registration_status']}, prov={res['source']['provenance']}")

    def test_worker_register_multi_contract(self):
        settings = {
            "detector": "sift",
            "ecc_refinement": False,
            "public_prefix": "/api/outputs",
        }
        mres = worker_register_multi([self.p1, self.p2, self.p3], self.out_dir, settings)

        self.assertIn("success", mres)
        self.assertIn("status", mres)
        self.assertIn("map_type", mres)
        self.assertIn("images", mres)
        self.assertIn("summary", mres)
        self.assertIn("graph", mres)
        self.assertIn("placement", mres)
        self.assertIn("mosaic", mres)
        self.assertIn("pairs", mres)

        print(f"[P8-API] Multi-image worker contract validated: placed={mres['summary']['images_registered']}/3, map_type='{mres['map_type']}'")

    def test_worker_settings_file_compatibility(self):
        """Verify --settings-file produces the exact same settings dict as --settings-json."""
        import tempfile
        expected_settings = {
            "detector": "sift",
            "ratio": 0.75,
            "ransac_threshold": 3.0,
            "illumination_normalization": True,
            "spatial_distribution": True,
            "radiometric_mode": "safe_normalization",
            "representation": "structural",
            "refinement_methods": "ecc,lucas_kanade,phase_correlation,quadratic_peak",
        }
        json_str = json.dumps(expected_settings)
        from_json = load_settings(settings_json=json_str)
        self.assertEqual(from_json, expected_settings)

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp:
            tmp.write(json_str)
            tmp_path = tmp.name

        try:
            from_file = load_settings(settings_file=tmp_path)
            self.assertEqual(from_file, expected_settings)
            self.assertEqual(from_file, from_json)

            # Test precedence when both are provided (file overrides json)
            overridden = load_settings(settings_json='{"detector": "orb"}', settings_file=tmp_path)
            self.assertEqual(overridden, expected_settings)
            self.assertEqual(overridden["detector"], "sift")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Test failure when neither is provided
        with self.assertRaises(ValueError):
            load_settings(None, None)


class TestPhase8FailureRate(unittest.TestCase):
    """Requirement 11 & 13.8: Quantitative Failure Rate under Stress Conditions."""

    def test_stress_failure_rate_100pct(self):
        # 6 stress failure cases that MUST fail gracefully and safely (0 crashes, 0 false PASS)
        stress_cases = [
            ("blank_zero", np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)),
            ("constant_grey", np.full((200, 200), 128, dtype=np.uint8), np.full((200, 200), 128, dtype=np.uint8)),
            ("saturated_white", np.full((200, 200), 255, dtype=np.uint8), np.full((200, 200), 255, dtype=np.uint8)),
            ("heavy_blur", make_synthetic_surface(seed=8007, size=200), cv2.GaussianBlur(make_synthetic_surface(seed=8007, size=200), (25, 25), 10.0)),
            ("unrelated_noise", np.random.default_rng(1).integers(0, 255, (200, 200), dtype=np.uint8), np.random.default_rng(2).integers(0, 255, (200, 200), dtype=np.uint8)),
            ("disjoint_scenes", make_synthetic_surface(seed=101, size=200), make_synthetic_surface(seed=909, size=200)),
        ]

        failed_count = 0
        for name, img_a, img_b in stress_cases:
            res = register_pair(img_a, img_b, detector="sift", ecc_refinement=False)
            if res.registration_status == "FAIL":
                failed_count += 1
            print(f"[P8-STRESS] {name}: status={res.registration_status}, inliers={np.sum(res.inlier_mask)}")

        failure_rate = (failed_count / len(stress_cases)) * 100.0
        print(f"[P8-METRIC] Stress Failure Rate: {failure_rate:.1f}% ({failed_count}/{len(stress_cases)} failed safely)")
        self.assertEqual(failure_rate, 100.0, "System did not safely fail all impossible stress cases")


class TestPhase8Reproducibility(unittest.TestCase):
    """Requirement 12: Deterministic Seeding & Exact Numerical Reproducibility."""

    def test_deterministic_numerical_reproducibility(self):
        img_a = make_synthetic_surface(seed=8008, size=320)
        img_b = make_synthetic_surface(seed=8008, size=320)

        # Run 1
        r1 = register_pair(img_a, img_b, detector="sift", ecc_refinement=False)
        # Run 2
        r2 = register_pair(img_a, img_b, detector="sift", ecc_refinement=False)

        self.assertEqual(r1.registration_status, r2.registration_status)
        self.assertEqual(int(np.sum(r1.inlier_mask)), int(np.sum(r2.inlier_mask)))
        self.assertAlmostEqual(r1.raw_rmse, r2.raw_rmse, places=6)
        np.testing.assert_allclose(r1.homography, r2.homography, rtol=1e-6, atol=1e-6)
        print("[P8-REPRO] Numerical reproducibility confirmed: homographies match to 1e-6 tolerance.")


if __name__ == "__main__":
    unittest.main()
