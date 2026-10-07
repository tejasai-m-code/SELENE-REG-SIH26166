import unittest
import numpy as np
import cv2
import os

from app.services.preprocessing import (
    preprocess_with_diagnostics,
    preprocess,
    compute_illumination_statistics,
    compare_illumination_pair,
    validate_illumination_robustness,
    percentile_normalize,
)
from app.services.gpu_accelerator import (
    gpu_gaussian_blur,
    gpu_sobel_gradient,
    gpu_multiscale_retinex,
    gpu_highpass_filter,
    is_cuda_available,
)
from app.services.pairwise_registration import register_pair


class TestPhase2RadiometricIlluminationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        h, w = 450, 450
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        cls.terrain = cv2.GaussianBlur(img, (9, 9), 2.0)
        # Add distinct crater-like craters and features spread across quadrants
        for i in range(16):
            cx = np.random.randint(40, w - 40)
            cy = np.random.randint(40, h - 40)
            r = np.random.randint(10, 28)
            cv2.circle(cls.terrain, (cx, cy), r, (255, 255, 255), -1)
            cv2.circle(cls.terrain, (cx, cy), int(r * 0.7), (20, 20, 20), -1)
            cv2.rectangle(cls.terrain, (cx - 5, cy - 5), (cx + 5, cy + 5), (200, 200, 200), -1)

    def test_raw_scientific_preservation_in_preprocessing(self):
        """Verify raw uint16 dynamic range is preserved un-clipped in stages."""
        # Create synthetic 16-bit raster with lunar DN values up to 4095
        raw_16bit = (self.terrain.astype(np.float32) * 16.0).astype(np.uint16)
        self.assertGreater(raw_16bit.max(), 255)
        self.assertEqual(raw_16bit.dtype, np.uint16)

        result = preprocess_with_diagnostics(raw_16bit, representation="structural")
        self.assertEqual(result.image.dtype, np.uint8)
        # Raw scientific stage preserves exact 16-bit matrix
        self.assertIn("raw_scientific", result.stages)
        self.assertEqual(result.stages["raw_scientific"].dtype, np.uint16)
        self.assertEqual(int(result.stages["raw_scientific"].max()), int(raw_16bit.max()))

    def test_radiometric_calibration_metadata_handling(self):
        """Verify dark current subtraction and gain/scale calibration."""
        # 1. With calibration metadata
        meta = {
            "dark_current": 10.0,
            "gain": 1.5,
            "radiance_scale": 0.05,
            "radiance_offset": 0.01,
        }
        res_cal = preprocess_with_diagnostics(self.terrain, metadata=meta, radiometric_mode="metadata_calibration")
        self.assertEqual(res_cal.radiometric["status"], "CALIBRATED_FROM_SUPPLIED_METADATA")
        self.assertEqual(res_cal.radiometric["mode"], "metadata_calibration")
        self.assertAlmostEqual(res_cal.radiometric["scale"], 0.05)

        # 2. Without calibration metadata (must truthfully state unavailable)
        res_uncal = preprocess_with_diagnostics(self.terrain, metadata={}, radiometric_mode="safe_normalization")
        self.assertEqual(res_uncal.radiometric["status"], "SAFE_NORMALIZATION_ONLY")
        self.assertIn("Radiometric calibration metadata unavailable", res_uncal.radiometric["note"])

    def test_illumination_statistics_computation(self):
        """Verify computation of physical illumination metrics and Shannon entropy."""
        stats = compute_illumination_statistics(self.terrain)
        self.assertIn("mean_brightness", stats)
        self.assertIn("dynamic_contrast_ratio", stats)
        self.assertIn("shadow_fraction_pct", stats)
        self.assertIn("illumination_gradient_magnitude", stats)
        self.assertIn("entropy", stats)

        self.assertGreater(stats["mean_brightness"], 0.0)
        self.assertGreater(stats["dynamic_contrast_ratio"], 0.0)
        self.assertGreaterEqual(stats["shadow_fraction_pct"], 0.0)
        self.assertGreater(stats["entropy"], 1.0)

    def test_gpu_acceleration_and_truthful_telemetry(self):
        """Verify GPU spatial filtering with truthful device telemetry and CPU fallback."""
        img = self.terrain.copy()
        
        # 1. Gaussian Blur
        blur, dev_blur = gpu_gaussian_blur(img, sigma=15.0)
        self.assertEqual(blur.shape, img.shape)
        self.assertIn(dev_blur["device_used"], ["cuda:0", "cpu"])
        self.assertGreater(dev_blur["timing_ms"], 0.0)

        # 2. Sobel Gradient
        grad, dev_grad = gpu_sobel_gradient(img)
        self.assertEqual(grad.shape, img.shape)
        self.assertIn(dev_grad["device_used"], ["cuda:0", "cpu"])

        # 3. Retinex
        ret, dev_ret = gpu_multiscale_retinex(img, scales=(15.0, 45.0, 120.0))
        self.assertEqual(ret.shape, img.shape)
        self.assertIn(dev_ret["device_used"], ["cuda:0", "cpu"])

        # 4. High-pass filter
        hp, dev_hp = gpu_highpass_filter(img, sigma=9.0)
        self.assertEqual(hp.shape, img.shape)
        self.assertIn(dev_hp["device_used"], ["cuda:0", "cpu"])

    def test_illumination_divergence_between_different_sun_angles(self):
        """Verify comparison metrics between same terrain under two differing sun illumination conditions."""
        h, w = self.terrain.shape
        
        # Illumination Condition A: Low solar elevation from West (gradient left-to-right)
        grad_a = np.tile(np.linspace(1.4, 0.6, w, dtype=np.float32), (h, 1))
        cond_a = np.clip(self.terrain.astype(np.float32) * grad_a, 0, 255).astype(np.uint8)

        # Illumination Condition B: Low solar elevation from East (gradient right-to-left) + shadow shift
        grad_b = np.tile(np.linspace(0.6, 1.4, w, dtype=np.float32), (h, 1))
        cond_b = np.clip(self.terrain.astype(np.float32) * grad_b, 0, 255).astype(np.uint8)

        comparison = compare_illumination_pair(cond_a, cond_b)
        self.assertIn("mean_absolute_difference", comparison)
        self.assertIn("normalized_cross_correlation", comparison)
        self.assertIn("mutual_information", comparison)
        self.assertGreater(comparison["mean_absolute_difference"], 10.0)

    def test_matching_robustness_under_different_illumination(self):
        """Scientifically validate matching improvement under differing illumination."""
        h, w = self.terrain.shape
        # Create transformed reference image (translation + slight rotation)
        M_geom = cv2.getRotationMatrix2D((w // 2, h // 2), 4.0, 1.0)
        M_geom[0, 2] += 20.0
        M_geom[1, 2] -= 15.0
        ref_geom = cv2.warpAffine(self.terrain, M_geom, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Condition 1: Source has strong westward illumination gradient
        grad_src = np.tile(np.linspace(1.5, 0.5, w, dtype=np.float32), (h, 1))
        src_illuminated = np.clip(self.terrain.astype(np.float32) * grad_src, 0, 255).astype(np.uint8)

        # Condition 2: Reference has opposite eastward illumination gradient
        grad_ref = np.tile(np.linspace(0.5, 1.5, w, dtype=np.float32), (h, 1))
        ref_illuminated = np.clip(ref_geom.astype(np.float32) * grad_ref, 0, 255).astype(np.uint8)

        # Validate robustness using structural representation
        val = validate_illumination_robustness(
            src_illuminated,
            ref_illuminated,
            detector="sift",
            representation="structural",
        )

        self.assertIn("baseline_raw", val)
        self.assertIn("illumination_normalized", val)
        self.assertIn("measurable_improvements", val)

        # Structural representation must successfully register the pair
        self.assertTrue(val["illumination_normalized"]["success"])
        self.assertGreater(val["illumination_normalized"]["inliers"], 10)
        # Robustness gain must be verified
        self.assertTrue(val["measurable_improvements"]["robustness_gain"])

    def test_pairwise_registration_output_illumination_diagnostics(self):
        """Verify that pairwise registration returns complete illumination diagnostics in preprocessing."""
        res = register_pair(
            self.terrain,
            self.terrain,
            detector="sift",
            representation="structural",
            illumination_normalization=True,
        )
        self.assertTrue(res.success)
        prep = res.preprocessing
        self.assertIsNotNone(prep)
        self.assertIn("source_illumination", prep)
        self.assertIn("reference_illumination", prep)
        self.assertIn("illumination_comparison", prep)
        self.assertIn("hardware_acceleration", prep)
        self.assertIn("source_radiometric", prep)


if __name__ == "__main__":
    unittest.main()
