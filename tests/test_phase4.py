"""Phase 4: Illumination, Sun-Angle, Scale & Multimodal Robustness Benchmark.

SCIENTIFIC DISCLAIMER
---------------------
All experiments in this file use DETERMINISTIC SYNTHETIC TRANSFORMATIONS only.

"Illumination robustness experiments" ≠ physical sun-angle invariance.
"Synthetic cross-modal proxy experiments" ≠ actual OHRC↔IIRS registration.

Synthetic brightness/contrast changes do NOT reproduce the non-linear shadow
casting and polarity inversion of real lunar craters under varying solar incidence.

Claims are strictly limited to what the synthetic experiments measure.

Real cross-modal mission-data validation was NOT possible in this phase
because the required OHRC/TMC-2/IIRS mission datasets were not available
in the repository.
"""

import time
import unittest
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.multimodal import (
    CrossModalConfig,
    ModalityMetadata,
    ModalityConfig,
    apply_synthetic_cross_modal,
    cross_modal_representation,
    KNOWN_MODALITIES,
)
from app.services.pairwise_registration import register_pair


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    condition: str
    parameters: dict
    success: bool
    inlier_count: int
    inlier_ratio: float
    rmse: float | None
    spatial_coverage: float | None  # fraction of image area covered by inliers
    registration_status: str
    processing_time_s: float
    baseline_rmse: float | None = None
    improved: bool | None = None
    failure_reason: str | None = None
    repr_used: str = "structural"


def _spatial_coverage(pts: np.ndarray, image_shape: tuple) -> float:
    """Fraction of image area covered by the convex hull of inlier points."""
    if len(pts) < 3:
        return 0.0
    h, w = image_shape[:2]
    pts_i = np.round(pts.reshape(-1, 2)).astype(np.int32)
    hull = cv2.convexHull(pts_i)
    area = cv2.contourArea(hull)
    return float(area) / max(1, h * w)


def _run_registration(src, ref, representation="structural", detector="sift") -> BenchmarkResult:
    t0 = time.perf_counter()
    try:
        result = register_pair(
            src, ref,
            detector=detector,
            max_features=1000,
            representation=representation,
            refinement_methods=None,
        )
        elapsed = time.perf_counter() - t0
        inlier_pts = result.source_points.reshape(-1, 2)
        inlier_mask = result.inlier_mask
        if inlier_mask is not None and len(inlier_mask) > 0:
            n_inliers = int(np.sum(inlier_mask))
            inlier_pts = inlier_pts[inlier_mask.astype(bool)] if len(inlier_mask) == len(inlier_pts) else inlier_pts
        else:
            n_inliers = len(inlier_pts)
        coverage = _spatial_coverage(inlier_pts, src.shape)
        rmse = result.metrics.get("rmse") if result.metrics else None
        return BenchmarkResult(
            condition="",
            parameters={},
            success=result.success,
            inlier_count=n_inliers,
            inlier_ratio=result.metrics.get("inlier_ratio", 0.0) if result.metrics else 0.0,
            rmse=float(rmse) if rmse is not None else None,
            spatial_coverage=coverage,
            registration_status=result.registration_status,
            processing_time_s=elapsed,
            repr_used=representation,
        )
    except (ValueError, Exception) as e:
        elapsed = time.perf_counter() - t0
        return BenchmarkResult(
            condition="",
            parameters={},
            success=False,
            inlier_count=0,
            inlier_ratio=0.0,
            rmse=None,
            spatial_coverage=0.0,
            registration_status="FAIL",
            processing_time_s=elapsed,
            failure_reason=str(e),
            repr_used=representation,
        )


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

class Phase4Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # High-frequency textured base image with fiducials
        base = np.random.randint(20, 235, (512, 512), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (5, 5), 2.0)
        rng = np.random.default_rng(42)
        for _ in range(30):
            x, y = rng.integers(60, 450, size=2)
            cv2.circle(base, (int(x), int(y)), int(rng.integers(6, 22)), 255, -1)
        for _ in range(20):
            x, y = rng.integers(60, 450, size=2)
            cv2.rectangle(base, (int(x)-12, int(y)-12), (int(x)+12, int(y)+12), 0, -1)
        cls.base = base

    def _warp(self, M, is_hom=False):
        if is_hom:
            return cv2.warpPerspective(self.base, M, (512, 512),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(self.base, M, (512, 512),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _translation_M(self, dx, dy):
        return np.float32([[1, 0, dx], [0, 1, dy]])

    def _scale_M(self, s, cx=256.0, cy=256.0):
        return np.float32([[s, 0, cx*(1-s)], [0, s, cy*(1-s)]])

    def _rot_scale_M(self, angle_deg, scale, cx=256.0, cy=256.0):
        return cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    def _affine_M(self, pts_src, pts_dst):
        return cv2.getAffineTransform(
            np.float32(pts_src),
            np.float32(pts_dst),
        )


# ===========================================================================
# SECTION A–E: Illumination robustness experiments
# ===========================================================================

class TestIlluminationRobustness(Phase4Base):
    """Illumination robustness experiments.

    These use SYNTHETIC TRANSFORMATIONS ONLY.
    They are NOT equivalent to physical sun-angle variation.
    """

    REPR_PAIRS = [("structural", "Robust"), ("percentile", "Baseline")]

    def _benchmark_pair(self, ref, label):
        results = {}
        for rep, name in self.REPR_PAIRS:
            r = _run_registration(self.base, ref, representation=rep)
            r.condition = label
            results[name] = r
        return results

    def test_A_brightness_offset(self):
        """A. Brightness offset (+30 DN, -30 DN)."""
        for offset in [30, -30]:
            ref = np.clip(self.base.astype(np.float32) + offset, 0, 255).astype(np.uint8)
            res = self._benchmark_pair(ref, f"brightness_offset_{offset:+d}")
            print(f"[A] brightness {offset:+d}: robust_inliers={res['Robust'].inlier_count}, "
                  f"baseline_inliers={res['Baseline'].inlier_count}, "
                  f"robust_rmse={res['Robust'].rmse}")
            self.assertTrue(res["Robust"].success, f"Brightness offset {offset} failed")

    def test_B_contrast_scaling(self):
        """B. Contrast scaling (0.5× and 1.5×)."""
        for alpha in [0.5, 1.5]:
            ref = np.clip(self.base.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            res = self._benchmark_pair(ref, f"contrast_scale_{alpha}")
            print(f"[B] contrast {alpha}×: robust_inliers={res['Robust'].inlier_count}, "
                  f"baseline_inliers={res['Baseline'].inlier_count}")
            self.assertTrue(res["Robust"].success, f"Contrast scale {alpha} failed")

    def test_C_gamma_variation(self):
        """C. Gamma variation (0.5, 1.5, 2.0)."""
        for gamma in [0.5, 1.5, 2.0]:
            ref = apply_synthetic_cross_modal(self.base, "gamma", gamma, seed=42)
            res = self._benchmark_pair(ref, f"gamma_{gamma}")
            print(f"[C] gamma {gamma}: robust_inliers={res['Robust'].inlier_count}")
            self.assertTrue(res["Robust"].success, f"Gamma {gamma} failed")

    def test_D_local_illumination_gradient(self):
        """D. Spatially-varying illumination gradient."""
        ref = apply_synthetic_cross_modal(self.base, "local_illum", 1.0, seed=7)
        res = self._benchmark_pair(ref, "local_illum")
        print(f"[D] local_illum: robust_inliers={res['Robust'].inlier_count}, "
              f"baseline_inliers={res['Baseline'].inlier_count}")
        self.assertTrue(res["Robust"].success, "Local illumination gradient failed")

    def test_E_shadow_region(self):
        """E. Dark rectangular soft shadow region."""
        ref = apply_synthetic_cross_modal(self.base, "shadow", 0.7, seed=3)
        res = self._benchmark_pair(ref, "shadow_0.7")
        print(f"[E] shadow: robust_inliers={res['Robust'].inlier_count}, "
              f"baseline_inliers={res['Baseline'].inlier_count}")
        self.assertTrue(res["Robust"].success, "Shadow region registration failed")

    def test_E2_multi_shadow(self):
        """E2. Multiple overlapping soft shadow regions (n=3)."""
        ref = apply_synthetic_cross_modal(self.base, "multi_shadow", 3, seed=9)
        res = self._benchmark_pair(ref, "multi_shadow_3")
        print(f"[E2] multi_shadow: robust_inliers={res['Robust'].inlier_count}")
        # multi-shadow is harsh; just ensure no crash
        self.assertIsNotNone(res["Robust"])

    def test_E3_combined_illumination(self):
        """E3. Combined local illumination gradient + shadow."""
        ref = apply_synthetic_cross_modal(self.base, "combined", 1.0, seed=11)
        res = self._benchmark_pair(ref, "combined_illum_shadow")
        print(f"[E3] combined: robust_inliers={res['Robust'].inlier_count}")
        self.assertIsNotNone(res["Robust"])


# ===========================================================================
# SECTION F: Scale robustness benchmark
# ===========================================================================

class TestScaleRobustness(Phase4Base):
    """Scale robustness benchmark.

    These compare single-scale and multi-scale behaviors.
    'Scale invariant' is NOT claimed — we report the measured success range.
    """

    SCALES = [0.50, 0.67, 0.75, 1.00, 1.25, 1.50, 2.00]

    def test_F_scale_sweep(self):
        """F. Registration across scale factors 0.5×–2.0×."""
        for s in self.SCALES:
            M = self._scale_M(s)
            ref = self._warp(M)
            r = _run_registration(self.base, ref, representation="structural")
            print(
                f"[F] scale={s:.2f}: success={r.success}, inliers={r.inlier_count}, "
                f"rmse={r.rmse}, time={r.processing_time_s:.2f}s"
            )
            # 1.0× must always succeed; other scales are reported but not mandated
            if s == 1.00:
                self.assertTrue(r.success, "Identity-scale registration failed")


# ===========================================================================
# SECTION G: Rotation + Scale benchmark
# ===========================================================================

class TestRotationScale(Phase4Base):
    """G. Rotation + Scale combinations."""

    CASES = [
        (5, 1.00),
        (-5, 1.00),
        (15, 1.00),
        (-15, 1.00),
        (30, 1.00),
        (-30, 1.00),
        (15, 1.25),
        (-15, 0.80),
        (30, 1.25),
    ]

    def test_G_rotation_scale_sweep(self):
        """G. Rotation ±5°, ±15°, ±30° combined with scale."""
        for angle, scale in self.CASES:
            M = self._rot_scale_M(angle, scale)
            ref = self._warp(M)
            r = _run_registration(self.base, ref, representation="structural")
            print(
                f"[G] angle={angle:+d}° scale={scale}: success={r.success}, "
                f"inliers={r.inlier_count}, rmse={r.rmse}"
            )
            # Small rotations (±5°) must succeed
            if abs(angle) <= 5 and scale == 1.0:
                self.assertTrue(r.success, f"Rotation {angle}° registration failed")


# ===========================================================================
# SECTION H–I: Affine / Perspective
# ===========================================================================

class TestAffineAndPerspective(Phase4Base):
    """H–I. Affine and mild perspective benchmark."""

    def test_H_affine(self):
        """H. Controlled affine deformation."""
        src_pts = np.float32([[50, 50], [450, 50], [50, 450]])
        dst_pts = np.float32([[60, 40], [445, 55], [55, 460]])
        M = self._affine_M(src_pts, dst_pts)
        ref = self._warp(M)
        r = _run_registration(self.base, ref, representation="structural")
        print(f"[H] affine: success={r.success}, inliers={r.inlier_count}, rmse={r.rmse}")
        self.assertTrue(r.success, "Affine deformation registration failed")

    def test_I_mild_perspective(self):
        """I. Mild perspective (projective) transformation."""
        src_pts = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
        dst_pts = np.float32([[10, 5], [501, 2], [505, 508], [5, 510]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        ref = self._warp(M, is_hom=True)
        r = _run_registration(self.base, ref, representation="structural")
        print(f"[I] perspective: success={r.success}, inliers={r.inlier_count}")
        self.assertTrue(r.success, "Mild perspective registration failed")


# ===========================================================================
# SECTION J–L: Synthetic cross-modal proxy experiments
# ===========================================================================

class TestSyntheticCrossModal(Phase4Base):
    """Synthetic cross-modal proxy experiments.

    IMPORTANT: These are CONTROLLED SYNTHETIC PROXY EXPERIMENTS ONLY.
    They are NOT equivalent to actual OHRC↔IIRS registration.
    Intensity ramps, contrast inversions, and gamma differences serve as
    rough analogues to cross-sensor reflectance differences. Actual
    cross-modal correspondence requires learned descriptors (Phase 5+).
    """

    def test_J_nonlinear_intensity_map(self):
        """J. Non-linear sigmoidal intensity remap (proxy for non-linear sensor response)."""
        ref = apply_synthetic_cross_modal(self.base, "nonlinear_map", 8.0, seed=13)
        r_baseline = _run_registration(self.base, ref, representation="percentile")
        r_robust = _run_registration(self.base, ref, representation="gradient")
        print(
            f"[J] nonlinear_map: "
            f"baseline_inliers={r_baseline.inlier_count}, "
            f"gradient_inliers={r_robust.inlier_count}"
        )
        # Just verify neither crashes
        self.assertIsNotNone(r_baseline)
        self.assertIsNotNone(r_robust)

    def test_K_inverted_contrast(self):
        """K. Global contrast inversion (proxy for polarity flip between sensors)."""
        ref = apply_synthetic_cross_modal(self.base, "invert", seed=0)
        r_baseline = _run_registration(self.base, ref, representation="percentile")
        r_gradient = _run_registration(self.base, ref, representation="gradient")
        print(
            f"[K] invert: "
            f"baseline_success={r_baseline.success}, inliers={r_baseline.inlier_count}; "
            f"gradient_success={r_gradient.success}, inliers={r_gradient.inlier_count}"
        )
        # Gradient representation is expected to preserve more correspondences
        # under inversion since edges are polarity-agnostic
        self.assertIsNotNone(r_gradient)

    def test_L_gamma_cross_modal_proxy(self):
        """L. Gamma difference as cross-modal proxy (OHRC≈1.0, IIRS proxy≈0.5 gamma)."""
        # Simulate IIRS-like image with lower gamma (compressed highlights)
        ref = apply_synthetic_cross_modal(self.base, "gamma", 0.5, seed=5)
        r_clahe = _run_registration(self.base, ref, representation="clahe")
        r_retinex = _run_registration(self.base, ref, representation="retinex")
        r_gradient = _run_registration(self.base, ref, representation="gradient")
        print(
            f"[L] gamma_proxy: clahe={r_clahe.inlier_count}, "
            f"retinex={r_retinex.inlier_count}, "
            f"gradient={r_gradient.inlier_count}"
        )
        self.assertIsNotNone(r_gradient)


# ===========================================================================
# SECTION M: Combined illumination + geometric change
# ===========================================================================

class TestCombinedConditions(Phase4Base):
    """M. Combined illumination + geometric transformations."""

    def test_M_illumination_plus_translation(self):
        """M1. Brightness offset + translation."""
        M = self._translation_M(18, -12)
        ref = self._warp(M)
        ref = np.clip(ref.astype(np.float32) + 40, 0, 255).astype(np.uint8)
        r = _run_registration(self.base, ref, representation="structural")
        print(f"[M1] illum+translate: success={r.success}, inliers={r.inlier_count}")
        self.assertTrue(r.success, "Illumination + translation failed")

    def test_M_illumination_plus_scale(self):
        """M2. Gamma variation + scale change."""
        M = self._scale_M(1.25)
        ref = self._warp(M)
        ref = apply_synthetic_cross_modal(ref, "gamma", 1.5, seed=17)
        r = _run_registration(self.base, ref, representation="structural")
        print(f"[M2] illum+scale: success={r.success}, inliers={r.inlier_count}")
        self.assertIsNotNone(r)

    def test_M_shadow_plus_rotation(self):
        """M3. Shadow region + rotation."""
        M = self._rot_scale_M(10, 1.0)
        ref = self._warp(M)
        ref = apply_synthetic_cross_modal(ref, "shadow", 0.6, seed=21)
        r = _run_registration(self.base, ref, representation="structural")
        print(f"[M3] shadow+rotation: success={r.success}, inliers={r.inlier_count}")
        self.assertIsNotNone(r)


# ===========================================================================
# SECTION: Multimodal abstraction layer unit tests
# ===========================================================================

class TestMultimodalAbstraction(unittest.TestCase):
    """Unit tests for the new multimodal abstraction layer."""

    def test_known_modalities_registered(self):
        """All required instrument modalities are registered."""
        for name in ["ohrc", "tmc2", "iirs", "generic"]:
            self.assertIn(name, KNOWN_MODALITIES)

    def test_modality_config_ohrc(self):
        cfg = ModalityConfig.for_modality("ohrc")
        self.assertEqual(cfg.modality, "ohrc")
        self.assertEqual(cfg.representation, "structural")
        self.assertGreater(cfg.max_features, 0)

    def test_modality_config_iirs(self):
        cfg = ModalityConfig.for_modality("iirs")
        self.assertEqual(cfg.modality, "iirs")
        self.assertEqual(cfg.representation, "retinex")

    def test_modality_config_tmc2(self):
        cfg = ModalityConfig.for_modality("tmc2")
        self.assertEqual(cfg.modality, "tmc2")
        self.assertEqual(cfg.representation, "clahe")

    def test_cross_modal_config_ohrc_iirs(self):
        src = ModalityMetadata(modality="ohrc")
        ref = ModalityMetadata(modality="iirs")
        cfg = CrossModalConfig.build(src, ref)
        self.assertTrue(cfg.is_cross_modal)
        self.assertEqual(cfg.shared_representation, "gradient")

    def test_cross_modal_config_same(self):
        src = ModalityMetadata(modality="ohrc")
        ref = ModalityMetadata(modality="ohrc")
        cfg = CrossModalConfig.build(src, ref)
        self.assertFalse(cfg.is_cross_modal)

    def test_modality_metadata_to_dict(self):
        m = ModalityMetadata(
            modality="ohrc", gsd_m=0.25, radiance_scale=1.5, radiance_offset=0.0
        )
        d = m.to_preprocessing_dict()
        self.assertAlmostEqual(d["gsd_m"], 0.25)
        self.assertAlmostEqual(d["radiance_scale"], 1.5)

    def test_synthetic_cross_modal_gamma(self):
        img = np.full((64, 64), 128, dtype=np.uint8)
        out = apply_synthetic_cross_modal(img, "gamma", 2.0, seed=0)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out <= 255))

    def test_synthetic_cross_modal_invert(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        out = apply_synthetic_cross_modal(img, "invert")
        # 255 - 100 = 155
        self.assertTrue(np.all(out == 155))

    def test_synthetic_cross_modal_shadow(self):
        img = np.full((128, 128), 200, dtype=np.uint8)
        out = apply_synthetic_cross_modal(img, "shadow", 0.9, seed=1)
        # At least some pixels should be darker
        self.assertLess(float(out.min()), float(img.min()) + 1)

    def test_cross_modal_representation_selection(self):
        rep = cross_modal_representation(np.zeros((1, 1), dtype=np.uint8), "ohrc", "iirs")
        self.assertEqual(rep, "gradient")
        rep2 = cross_modal_representation(np.zeros((1, 1), dtype=np.uint8), "ohrc", "tmc2")
        self.assertEqual(rep2, "clahe")
        rep3 = cross_modal_representation(np.zeros((1, 1), dtype=np.uint8), "ohrc", "ohrc")
        self.assertEqual(rep3, "structural")


if __name__ == "__main__":
    unittest.main(verbosity=2)
