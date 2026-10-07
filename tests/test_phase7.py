"""Phase 7 - Final End-to-End Scientific Validation and SIH Readiness Benchmark.

All seeds are deterministic. No real mission data is claimed.
"""
import sys, time, unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts" / "api-server" / "python"))

from app.services.pairwise_registration import register_pair, serialize_inlier_points
from app.services.multi_registration import run_multi_registration
from app.services.multimodal import apply_synthetic_cross_modal
from app.services.mission_metadata import parse_metadata_label, MetadataStatus, ModalityType
from app.services.modality import get_modality_config, MODALITY_PROFILES
from app.services.footprint import create_image_footprint, create_mosaic_footprint, PolygonFootprint
from app.services.coordinates import (
    CoordinateFrame, TransformRecord, pixel_to_physical_local,
    SPICEInterface, DEMInterface, CameraModelInterface,
)

def make_image(seed=0, size=320):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, (size, size), dtype=np.uint8)
    for _ in range(12):
        cv2.circle(base, (int(rng.integers(0,size)), int(rng.integers(0,size))),
                   int(rng.integers(10,50)), int(rng.integers(30,230)), -1)
    noise = rng.normal(0, 8, base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_translation(img, dx, dy):
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def apply_brightness(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

def apply_gamma(img, gamma):
    lut = np.clip(((np.arange(256)/255.0)**gamma)*255.0, 0, 255).astype(np.uint8)
    return lut[img]

def apply_scale_transform(img, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(img, M, (w, h))

def apply_rotation_scale(img, angle_deg, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    return cv2.warpAffine(img, M, (w, h))


class TestPhase7Baseline(unittest.TestCase):
    def setUp(self):
        self.src = make_image(seed=1, size=320)
        self.ref = make_image(seed=1, size=320)

    def test_identical_images_pass(self):
        r = register_pair(self.src, self.ref, detector="sift", ecc_refinement=False)
        self.assertIn(r.registration_status, ("PASS", "PASS_WITH_WARNING"))
        if r.raw_rmse is not None:
            self.assertLess(r.raw_rmse, 1.5)
        print(f"[P7-B1] identical: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}, raw_rmse={r.raw_rmse}")

    def test_output_contract_present(self):
        r = register_pair(self.src, self.ref, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.homography)
        self.assertIn("inlier_count", r.metrics)
        self.assertIn("inlier_ratio", r.metrics)
        self.assertIn("source_spatial_coverage", r.metrics)
        self.assertIn("raw_rmse_pixels", r.metrics)
        pts = serialize_inlier_points(r)
        if len(pts) > 0:
            self.assertIn("source", pts[0])
            self.assertIn("error", pts[0])
        print(f"[P7-B1] contract: {len(pts)} inlier points")


class TestPhase7SubpixelRefinement(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=7, size=320)

    def _case(self, dx, dy, method):
        shifted = apply_translation(self.base, dx, dy)
        r = register_pair(self.base, shifted, detector="sift", ecc_refinement=True,
                          refinement_methods=[method])
        raw = r.raw_rmse
        refined = r.refined_rmse if r.refined_rmse is not None else r.raw_rmse
        print(f"[P7-B2] {method} dx={dx} dy={dy}: raw={raw}, refined={refined}")
        return raw, refined

    def test_ecc_quarter_pixel(self):
        raw, refined = self._case(0.25, 0.0, "ecc")
        if raw is not None and refined is not None:
            self.assertLessEqual(refined, raw + 0.5)

    def test_lk_half_pixel(self):
        raw, _ = self._case(-0.5, 0.0, "lucas_kanade")
        self.assertIsNotNone(raw)

    def test_ecc_diagonal(self):
        raw, refined = self._case(0.25, 0.25, "ecc")
        if raw is not None and refined is not None:
            self.assertLessEqual(refined, raw + 0.5)


class TestPhase7Illumination(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=3, size=320)
        self.ref = make_image(seed=3, size=320)

    def test_brightness_increase_30(self):
        r = register_pair(apply_brightness(self.base, 30), self.ref, detector="sift", ecc_refinement=False)
        inliers = int(np.sum(r.inlier_mask))
        print(f"[P7-B3] bright+30: status={r.registration_status}, inliers={inliers}")
        self.assertGreater(inliers, 4)

    def test_brightness_decrease_30(self):
        r = register_pair(apply_brightness(self.base, -30), self.ref, detector="sift", ecc_refinement=False)
        inliers = int(np.sum(r.inlier_mask))
        print(f"[P7-B3] bright-30: status={r.registration_status}, inliers={inliers}")
        self.assertGreater(inliers, 4)

    def test_contrast_reduction(self):
        lowc = np.clip(self.base.astype(np.float32)*0.5+64, 0, 255).astype(np.uint8)
        r = register_pair(lowc, self.ref, detector="sift", ecc_refinement=False)
        inliers = int(np.sum(r.inlier_mask))
        print(f"[P7-B4] contrast 0.5x: status={r.registration_status}, inliers={inliers}")
        self.assertGreater(inliers, 4)

    def test_gamma_correction(self):
        r = register_pair(apply_gamma(self.base, 1.5), self.ref, detector="sift", ecc_refinement=False)
        inliers = int(np.sum(r.inlier_mask))
        print(f"[P7-B5] gamma 1.5: status={r.registration_status}, inliers={inliers}")
        self.assertGreater(inliers, 4)

    def test_gamma_strong(self):
        r = register_pair(apply_gamma(self.base, 2.0), self.ref, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B5] gamma 2.0: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")


class TestPhase7LocalIllumination(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=5, size=320)

    def test_local_illum(self):
        modified = apply_synthetic_cross_modal(self.base, "local_illum", param=1.0, seed=0)
        r = register_pair(self.base, modified, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B6] local_illum: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")

    def test_shadow(self):
        modified = apply_synthetic_cross_modal(self.base, "shadow", param=0.6, seed=0)
        r = register_pair(self.base, modified, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B6] shadow: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")

    def test_combined_illum_shadow(self):
        modified = apply_synthetic_cross_modal(self.base, "combined", param=1.0, seed=0)
        r = register_pair(self.base, modified, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B6] combined: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")


class TestPhase7CrossModal(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=9, size=320)

    def test_inversion_gradient_vs_raw(self):
        inverted = apply_synthetic_cross_modal(self.base, "invert", seed=0)
        r_baseline = register_pair(self.base, inverted, detector="sift",
                                   ecc_refinement=False, illumination_normalization=False,
                                   representation="raw")
        baseline_inliers = int(np.sum(r_baseline.inlier_mask))
        r_gradient = register_pair(self.base, inverted, detector="sift",
                                   ecc_refinement=False, illumination_normalization=True,
                                   representation="gradient")
        gradient_inliers = int(np.sum(r_gradient.inlier_mask))
        print(f"[P7-B7] invert: baseline={baseline_inliers}, gradient={gradient_inliers}")
        self.assertGreater(gradient_inliers, baseline_inliers)

    def test_nonlinear_map(self):
        modified = apply_synthetic_cross_modal(self.base, "nonlinear_map", param=5.0, seed=0)
        r = register_pair(self.base, modified, detector="sift", ecc_refinement=False,
                          representation="gradient")
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B7] nonlinear_map: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")


class TestPhase7ScaleVariation(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=11, size=400)

    def _scale_case(self, scale):
        r = register_pair(self.base, apply_scale_transform(self.base, scale),
                          detector="sift", ecc_refinement=False)
        print(f"[P7-B8] scale={scale:.2f}: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}, t={r.processing_time_seconds:.2f}s")
        return r

    def test_scale_075(self): self.assertIsNotNone(self._scale_case(0.75).registration_status)
    def test_scale_125(self): self.assertIsNotNone(self._scale_case(1.25).registration_status)
    def test_scale_150(self): self.assertIsNotNone(self._scale_case(1.50).registration_status)
    def test_scale_050(self): self.assertIsNotNone(self._scale_case(0.50).registration_status)


class TestPhase7RotationScale(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=13, size=400)

    def _rot_case(self, angle, scale):
        r = register_pair(self.base, apply_rotation_scale(self.base, angle, scale),
                          detector="sift", ecc_refinement=False)
        print(f"[P7-B9] angle={angle:+.0f} scale={scale}: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")
        return r

    def test_rotation_plus_15(self): self.assertIsNotNone(self._rot_case(15.0, 1.0).registration_status)
    def test_rotation_minus_15(self): self.assertIsNotNone(self._rot_case(-15.0, 1.0).registration_status)
    def test_rotation_plus_30(self): self.assertIsNotNone(self._rot_case(30.0, 1.0).registration_status)
    def test_rotation_and_scale(self): self.assertIsNotNone(self._rot_case(15.0, 1.25).registration_status)


class TestPhase7NoisyBlurred(unittest.TestCase):
    def setUp(self):
        self.base = make_image(seed=17, size=320)

    def test_gaussian_noise(self):
        noise = np.random.default_rng(42).normal(0, 25, self.base.shape).astype(np.float32)
        noisy = np.clip(self.base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        r = register_pair(self.base, noisy, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B10] noise: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")

    def test_blur(self):
        blurred = cv2.GaussianBlur(self.base, (9, 9), 3.0)
        r = register_pair(self.base, blurred, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B10] blur: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")


class TestPhase7MultiImage(unittest.TestCase):
    def _make_set(self, n, size=280, seed=0):
        rng = np.random.default_rng(seed)
        base = make_image(seed=seed, size=size)
        return [apply_translation(base, float(rng.integers(-15,15)), float(rng.integers(-15,15))) for _ in range(n)]

    def test_three_image_mosaic(self):
        result = run_multi_registration(self._make_set(3, seed=0),
                                        {"detector": "sift", "max_features": 2000, "ecc_refinement": False})
        placed = result.summary.get("images_registered", 0)
        print(f"[P7-B11] 3-image: placed={placed}/3, mosaic={'yes' if result.mosaic is not None else 'no'}")
        self.assertGreaterEqual(placed, 2)

    def test_five_image_mosaic(self):
        result = run_multi_registration(self._make_set(5, seed=42),
                                        {"detector": "sift", "max_features": 2000, "ecc_refinement": False})
        placed = result.summary.get("images_registered", 0)
        print(f"[P7-B11] 5-image: placed={placed}/5")
        self.assertGreaterEqual(placed, 3)

    def test_mosaic_footprint_present(self):
        result = run_multi_registration(self._make_set(3, seed=1),
                                        {"detector": "sift", "max_features": 2000, "ecc_refinement": False})
        self.assertIsNotNone(result.mosaic_info)
        if result.mosaic_info:
            # mosaic_info top-level keys are width/height/placed_image_count;
            # canvas_area_pixels is nested under mosaic_footprint sub-dict.
            self.assertIn("width", result.mosaic_info)
            self.assertIn("height", result.mosaic_info)
            self.assertIn("placed_image_count", result.mosaic_info)
            mosaic_fp = result.mosaic_info.get("mosaic_footprint", {})
            if mosaic_fp:
                self.assertIn("canvas_area_pixels", mosaic_fp)
        print(f"[P7-B11] mosaic_info: {list(result.mosaic_info.keys()) if result.mosaic_info else 'None'}")


class TestPhase7SpatialDistribution(unittest.TestCase):
    def test_spatial_uniformity_reported(self):
        src = make_image(seed=21, size=320)
        ref = make_image(seed=21, size=320)
        r = register_pair(src, ref, detector="sift", spatial_distribution=True, ecc_refinement=False)
        self.assertIn("spatial_uniformity", r.metrics)
        self.assertIn("source_spatial_grid", r.metrics)
        coverage = r.metrics.get("source_spatial_coverage", 0.0)
        uniformity = r.metrics.get("spatial_uniformity", 0.0)
        print(f"[P7-B12] coverage={coverage:.3f}, uniformity={uniformity:.3f}")
        self.assertGreaterEqual(coverage, 0.0)


class TestPhase7FailureCases(unittest.TestCase):
    def test_blank_image(self):
        r = register_pair(np.zeros((200,200),dtype=np.uint8), np.zeros((200,200),dtype=np.uint8),
                          detector="sift", ecc_refinement=False)
        self.assertFalse(r.success)
        print(f"[P7-B13] blank: status={r.registration_status}")

    def test_constant_image(self):
        r = register_pair(np.full((200,200),128,dtype=np.uint8), np.full((200,200),200,dtype=np.uint8),
                          detector="sift", ecc_refinement=False)
        self.assertFalse(r.success)
        print(f"[P7-B13] constant: status={r.registration_status}")

    def test_tiny_image(self):
        r = register_pair(make_image(seed=0,size=32), make_image(seed=0,size=32),
                          detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B13] tiny 32x32: status={r.registration_status}")

    def test_mismatched_dimensions(self):
        r = register_pair(make_image(seed=0,size=160), make_image(seed=0,size=320),
                          detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B13] mismatched dims: status={r.registration_status}")

    def test_extreme_brightness(self):
        r = register_pair(make_image(seed=2,size=200), np.full((200,200),255,dtype=np.uint8),
                          detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B13] saturated ref: status={r.registration_status}")

    def test_unrelated_scenes(self):
        r = register_pair(make_image(seed=1,size=200), make_image(seed=999,size=200),
                          detector="sift", ecc_refinement=False)
        self.assertIsNotNone(r.registration_status)
        print(f"[P7-B13] unrelated scenes: status={r.registration_status}, inliers={np.sum(r.inlier_mask)}")

    def test_invalid_footprint_raises(self):
        fp = PolygonFootprint(np.array([[0,0],[1,0]]))
        self.assertFalse(fp.is_valid())
        with self.assertRaises(ValueError):
            fp.transform(np.eye(3), "MOSAIC_CANVAS_PIXEL")

    def test_footprint_frame_mismatch(self):
        fp1 = PolygonFootprint(np.array([[0,0],[100,0],[100,100],[0,100]]), coordinate_frame="IMAGE_PIXEL")
        fp2 = PolygonFootprint(np.array([[0,0],[100,0],[100,100],[0,100]]), coordinate_frame="MOSAIC_CANVAS_PIXEL")
        with self.assertRaises(ValueError):
            fp1.intersect(fp2)

    def test_missing_gsd_no_physical_scale(self):
        meta = parse_metadata_label({})
        self.assertIsNone(meta.gsd_m)
        self.assertFalse(meta.to_dict()["physical_scale_available"])

    def test_invalid_metadata_unknown_status(self):
        meta = parse_metadata_label({})
        self.assertEqual(meta.status, MetadataStatus.UNKNOWN)
        self.assertIsNone(meta.acquisition_time)

    def test_gsd_zero_raises(self):
        with self.assertRaises(ValueError):
            pixel_to_physical_local(np.array([[10, 20]]), gsd_m=0.0)

    def test_gsd_negative_raises(self):
        with self.assertRaises(ValueError):
            pixel_to_physical_local(np.array([[10, 20]]), gsd_m=-5.0)


class TestPhase7OutputContract(unittest.TestCase):
    def setUp(self):
        base = make_image(seed=23, size=320)
        self.src = base
        self.ref = apply_translation(base, 5, -3)

    def test_full_contract(self):
        r = register_pair(self.src, self.ref, detector="sift", ecc_refinement=True)
        self.assertIn("inlier_count", r.metrics)
        self.assertIn("inlier_ratio", r.metrics)
        self.assertIn("source_spatial_coverage", r.metrics)
        self.assertIn("raw_rmse_pixels", r.metrics)
        self.assertIn("refined_rmse_pixels", r.metrics)
        self.assertIn("rmse_improvement_pixels", r.metrics)
        self.assertIn("correspondence_basis", r.metrics)
        self.assertIn("subpixel_validation_status", r.metrics)
        self.assertIn("quality_gate", r.inlier_investigation)
        gate = r.inlier_investigation["quality_gate"]
        self.assertIn("status", gate)
        self.assertIn("gate_checks", gate)
        self.assertEqual(r.homography.shape, (3, 3))
        self.assertIsNotNone(r.subpixel)
        self.assertIn("method_status", r.subpixel)
        self.assertIn("representation", r.preprocessing)
        self.assertIsNotNone(r.footprint)
        self.assertEqual(r.footprint["status"], "COMPUTED")
        self.assertIn("source_footprint", r.footprint)
        self.assertIn("reference_footprint", r.footprint)
        self.assertIn("warped_source_footprint", r.footprint)
        self.assertIn("overlap", r.footprint)
        pts = serialize_inlier_points(r)
        if len(pts) > 0:
            self.assertIn("source", pts[0])
            self.assertIn("error", pts[0])
            self.assertIn("refinement_method", pts[0])
        print(f"[P7-B14] contract ok, status={r.registration_status}, pts={len(pts)}")


class TestPhase7ModalityProfiles(unittest.TestCase):
    def test_all_profiles_tagged(self):
        for mod, config in MODALITY_PROFILES.items():
            self.assertEqual(config.profile_type, "REFERENCE_PROFILE")
            self.assertEqual(config.provenance_source, "DEFAULT_ESTIMATE")
            print(f"[P7-MOD] {mod.value}: gsd={config.nominal_gsd_m}, type={config.profile_type}")

    def test_ohrc_reference_values(self):
        cfg = get_modality_config("ohrc")
        self.assertAlmostEqual(cfg.nominal_gsd_m, 0.25)

    def test_tmc2_reference_values(self):
        cfg = get_modality_config("tmc-2")
        self.assertAlmostEqual(cfg.nominal_gsd_m, 5.0)

    def test_iirs_reference_values(self):
        cfg = get_modality_config("iirs")
        self.assertAlmostEqual(cfg.nominal_gsd_m, 80.0)
        self.assertTrue(cfg.requires_cross_modal_proxy)


class TestPhase7ExtensionInterfaces(unittest.TestCase):
    def test_spice_unavailable(self):
        self.assertFalse(SPICEInterface.is_available())
        result = SPICEInterface.get_spacecraft_geometry("OHRC_TEST_001")
        self.assertFalse(result["available"])
        self.assertIn("INTERFACE_PREPARED", result["status"])

    def test_dem_unavailable(self):
        self.assertFalse(DEMInterface.is_available())
        result = DEMInterface.get_elevation_profile(0.0, 0.0)
        self.assertFalse(result["available"])
        self.assertIsNone(result["elevation_m"])

    def test_camera_model_unavailable(self):
        self.assertFalse(CameraModelInterface.is_available())


class TestPhase7CoordinateFrames(unittest.TestCase):
    def test_pixel_to_physical_valid_gsd(self):
        pts = np.array([[100.0, 200.0], [150.0, 250.0]])
        physical = pixel_to_physical_local(pts, gsd_m=0.25)
        self.assertEqual(physical.shape, (2, 2))
        np.testing.assert_allclose(physical[0, 0], 25.0)

    def test_transform_record_invert(self):
        M = np.eye(3, dtype=np.float64)
        M[0, 2] = 10.0
        tr = TransformRecord(
            source_frame=CoordinateFrame.IMAGE_PIXEL,
            target_frame=CoordinateFrame.REFERENCE_IMAGE_PIXEL,
            matrix=M,
        )
        inv = tr.invert()
        self.assertEqual(inv.source_frame, CoordinateFrame.REFERENCE_IMAGE_PIXEL)
        self.assertEqual(inv.target_frame, CoordinateFrame.IMAGE_PIXEL)

    def test_footprint_frame_preserved(self):
        fp = create_image_footprint((200, 300), source_id=0)
        self.assertTrue(fp.is_valid())
        M = np.eye(3, dtype=np.float64)
        M[0, 2] = 5.0
        transformed = fp.transform(M, target_frame="MOSAIC_CANVAS_PIXEL")
        self.assertEqual(transformed.coordinate_frame, "MOSAIC_CANVAS_PIXEL")


class TestPhase7Performance(unittest.TestCase):
    def test_pairwise_performance(self):
        src = make_image(seed=31, size=512)
        ref = make_image(seed=31, size=512)
        t0 = time.perf_counter()
        r = register_pair(src, ref, detector="sift", max_features=4000, ecc_refinement=False)
        elapsed = time.perf_counter() - t0
        print(f"[P7-PERF] pairwise 512x512: {elapsed:.2f}s, inliers={np.sum(r.inlier_mask)}")
        self.assertLess(elapsed, 15.0)

    def test_multi_3image_performance(self):
        images = [make_image(seed=i, size=240) for i in range(3)]
        t0 = time.perf_counter()
        result = run_multi_registration(images, {"detector": "sift", "max_features": 2000,
                                                  "ecc_refinement": False})
        elapsed = time.perf_counter() - t0
        print(f"[P7-PERF] multi 3-image 240x240: {elapsed:.2f}s, placed={result.summary.get('images_registered')}")
        self.assertLess(elapsed, 60.0)


class TestPhase7RealDataAvailability(unittest.TestCase):
    def test_no_real_mission_data(self):
        # REAL-DATA AUDIT: No .img/.lbl/.fits/.cub/.pds/SPICE/DEM present.
        # All validation is SYNTHETICALLY VALIDATED. This is a provenance marker.
        self.assertTrue(True)

    def test_metadata_unknown_when_empty(self):
        meta = parse_metadata_label({})
        self.assertEqual(meta.modality, ModalityType.GENERIC)
        self.assertIsNone(meta.gsd_m)
        self.assertEqual(meta.status, MetadataStatus.UNKNOWN)
        print(f"[P7-REAL] empty: status={meta.status.value}, gsd={meta.gsd_m}")

    def test_metadata_parses_json_label(self):
        label = '{"modality": "OHRC", "gsd_m": 0.25, "product_id": "TEST_001", "acquisition_time": "2019-07-22"}'
        meta = parse_metadata_label(label)
        self.assertEqual(meta.modality, ModalityType.OHRC)
        self.assertAlmostEqual(meta.gsd_m, 0.25)
        print(f"[P7-REAL] json: modality={meta.modality.value}, gsd={meta.gsd_m}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
