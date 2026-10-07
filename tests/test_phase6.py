"""Phase 6 Test Suite: Mission-Data Readiness, Geospatial Layer & Footprint Engine."""

import unittest
import numpy as np
import cv2

from app.services.mission_metadata import (
    MetadataStatus,
    ModalityType,
    MissionImageMetadata,
    SolarGeometry,
    SensorGeometry,
    SpectralMetadata,
    parse_metadata_label,
    parse_pvl_text,
)
from app.services.modality import (
    MODALITY_PROFILES,
    get_modality_config,
    compute_gsd_scale_factor,
)
from app.services.footprint import (
    PolygonFootprint,
    create_image_footprint,
    create_mosaic_footprint,
)
from app.services.coordinates import (
    CoordinateFrame,
    TransformRecord,
    pixel_to_physical_local,
    SPICEInterface,
    DEMInterface,
    CameraModelInterface,
)
from app.services.multi_registration import run_multi_registration


class TestPhase6MissionMetadata(unittest.TestCase):
    """Step 2, 3, 4: Mission Metadata Model, Modality Configuration & Label Parsing."""

    def test_parse_json_metadata_label(self):
        json_str = """
        {
            "instrument": "OHRC",
            "mission": "Chandrayaan-2",
            "product_id": "ch2_ohr_ncp_20201110T053421123_d_img_d18",
            "width": 12000,
            "height": 4000,
            "gsd_m": 0.25,
            "pixel_size_um": 7.0,
            "incidence_angle_deg": 48.5,
            "emission_angle_deg": 3.2,
            "phase_angle_deg": 50.1,
            "spacecraft_altitude_km": 100.2
        }
        """
        meta = parse_metadata_label(json_str)
        self.assertEqual(meta.modality, ModalityType.OHRC)
        self.assertEqual(meta.product_id, "ch2_ohr_ncp_20201110T053421123_d_img_d18")
        self.assertEqual(meta.image_width, 12000)
        self.assertEqual(meta.image_height, 4000)
        self.assertEqual(meta.gsd_m, 0.25)
        self.assertEqual(meta.solar_geometry.incidence_angle_deg, 48.5)
        self.assertEqual(meta.solar_geometry.status, MetadataStatus.KNOWN)
        self.assertEqual(meta.sensor_geometry.spacecraft_altitude_km, 100.2)
        self.assertEqual(meta.status, MetadataStatus.KNOWN)

    def test_parse_pvl_text_label(self):
        pvl_str = """
        /* Chandrayaan-2 TMC-2 Synthetic PDS Header */
        INSTRUMENT_NAME = "TMC-2";
        MISSION_NAME = "Chandrayaan-2";
        PRODUCT_ID = "CH2_TMC_NC_20191012";
        IMAGE_WIDTH = 4000;
        IMAGE_HEIGHT = 8000;
        SPATIAL_RESOLUTION = 5.0;
        INCIDENCE_ANGLE = 35.2;
        """
        meta = parse_metadata_label(pvl_str)
        self.assertEqual(meta.modality, ModalityType.TMC2)
        self.assertEqual(meta.gsd_m, 5.0)
        self.assertEqual(meta.image_width, 4000)
        self.assertEqual(meta.solar_geometry.incidence_angle_deg, 35.2)

    def test_parse_xml_label(self):
        xml_str = """
        <Product_Observational>
            <instrument>IIRS</instrument>
            <product_id>CH2_IIR_HYP_20210214</product_id>
            <bands>256</bands>
            <gsd_m>80.0</gsd_m>
        </Product_Observational>
        """
        meta = parse_metadata_label(xml_str)
        self.assertEqual(meta.modality, ModalityType.IIRS)
        self.assertEqual(meta.spectral.channel_count, 256)
        self.assertEqual(meta.gsd_m, 80.0)

    def test_missing_metadata_defaults_to_unknown(self):
        meta = parse_metadata_label({})
        self.assertEqual(meta.modality, ModalityType.GENERIC)
        self.assertIsNone(meta.gsd_m)
        self.assertEqual(meta.solar_geometry.status, MetadataStatus.UNKNOWN)
        self.assertEqual(meta.sensor_geometry.status, MetadataStatus.UNKNOWN)
        self.assertEqual(meta.status, MetadataStatus.UNKNOWN)

    def test_modality_configuration_profiles(self):
        ohrc_cfg = get_modality_config("OHRC")
        self.assertEqual(ohrc_cfg.modality, ModalityType.OHRC)
        self.assertEqual(ohrc_cfg.nominal_gsd_m, 0.25)
        self.assertEqual(ohrc_cfg.spectral_type, "panchromatic")

        tmc_cfg = get_modality_config("TMC-2")
        self.assertEqual(tmc_cfg.modality, ModalityType.TMC2)
        self.assertEqual(tmc_cfg.nominal_gsd_m, 5.0)

        iirs_cfg = get_modality_config("IIRS")
        self.assertEqual(iirs_cfg.modality, ModalityType.IIRS)
        self.assertEqual(iirs_cfg.nominal_gsd_m, 80.0)
        self.assertTrue(iirs_cfg.requires_cross_modal_proxy)


class TestPhase6GSDAndScale(unittest.TestCase):
    """Step 5 & 15: GSD Resolution Scaling and Physical Distance Conversion."""

    def test_gsd_scale_factor_calculation(self):
        meta_ohrc = MissionImageMetadata(gsd_m=0.25)
        meta_tmc = MissionImageMetadata(gsd_m=5.0)
        ratio, msg = compute_gsd_scale_factor(meta_ohrc, meta_tmc)
        self.assertAlmostEqual(ratio, 0.05, delta=1e-5)
        self.assertIn("Valid GSD ratio", msg)

    def test_gsd_unavailable_handling(self):
        meta_generic = MissionImageMetadata(gsd_m=None)
        meta_ohrc = MissionImageMetadata(gsd_m=0.25)
        ratio, msg = compute_gsd_scale_factor(meta_generic, meta_ohrc)
        self.assertIsNone(ratio)
        self.assertIn("unavailable", msg.lower())

    def test_pixel_to_physical_local_meters(self):
        pts_pixel = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 200.0]], dtype=np.float64)
        gsd = 0.25  # meters/pixel
        pts_phys = pixel_to_physical_local(pts_pixel, gsd)
        self.assertEqual(pts_phys[0, 0], 0.0)
        self.assertEqual(pts_phys[1, 0], 25.0)   # 100 px * 0.25 m/px = 25.0 m
        self.assertEqual(pts_phys[2, 1], 50.0)   # 200 px * 0.25 m/px = 50.0 m

    def test_pixel_to_physical_with_origin_offset(self):
        pts_pixel = np.array([[150.0, 150.0]], dtype=np.float64)
        gsd = 5.0  # meters/pixel
        pts_phys = pixel_to_physical_local(pts_pixel, gsd, origin_pixels=(100.0, 100.0))
        self.assertEqual(pts_phys[0, 0], 250.0)  # (150 - 100) * 5 = 250 m
        self.assertEqual(pts_phys[0, 1], 250.0)


class TestPhase6Footprints(unittest.TestCase):
    """Step 7, 8, 9, 10, 16: Footprint Creation, Transformation, Intersection, and Overlap."""

    def test_image_footprint_creation(self):
        fp = create_image_footprint((300, 400), source_id="img0")
        self.assertTrue(fp.is_valid())
        self.assertEqual(fp.area, 120000.0)
        self.assertEqual(fp.bounding_box, (0.0, 0.0, 400.0, 300.0))
        self.assertEqual(fp.coordinate_frame, "IMAGE_PIXEL")

    def test_footprint_transformation_affine(self):
        fp = create_image_footprint((100, 100), source_id="img0")
        # Translate by (+50, +50)
        T = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        warped_fp = fp.transform(T, target_frame="REFERENCE_IMAGE_PIXEL")
        self.assertTrue(warped_fp.is_valid())
        self.assertEqual(warped_fp.area, 10000.0)
        self.assertEqual(warped_fp.bounding_box, (50.0, 50.0, 150.0, 150.0))
        self.assertEqual(warped_fp.coordinate_frame, "REFERENCE_IMAGE_PIXEL")

    def test_footprint_overlap_metrics_partial(self):
        # Footprint A: (0..100, 0..100) -> Area 10,000
        # Footprint B: (50..150, 0..100) -> Area 10,000
        # Overlap: (50..100, 0..100) -> Area 5,000
        fp_a = create_image_footprint((100, 100), source_id="a")
        fp_a.coordinate_frame = "COMMON_FRAME"

        fp_b = PolygonFootprint(
            vertices=np.array([[50, 0], [150, 0], [150, 100], [50, 100]], dtype=np.float64),
            coordinate_frame="COMMON_FRAME",
            source_id="b",
        )

        metrics = fp_a.overlap_metrics(fp_b)
        self.assertEqual(metrics["intersection_area"], 5000.0)
        self.assertEqual(metrics["union_area"], 15000.0)
        self.assertAlmostEqual(metrics["iou"], 0.3333, delta=1e-3)
        self.assertEqual(metrics["overlap_ratio_a"], 0.5)
        self.assertEqual(metrics["overlap_ratio_b"], 0.5)
        self.assertTrue(metrics["has_overlap"])

    def test_footprint_overlap_disjoint(self):
        fp_a = PolygonFootprint(
            vertices=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64),
            coordinate_frame="COMMON_FRAME",
            source_id="a",
        )
        fp_b = PolygonFootprint(
            vertices=np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float64),
            coordinate_frame="COMMON_FRAME",
            source_id="b",
        )
        metrics = fp_a.overlap_metrics(fp_b)
        self.assertEqual(metrics["intersection_area"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertFalse(metrics["has_overlap"])

    def test_coordinate_frame_mismatch_raises_error(self):
        fp_a = create_image_footprint((100, 100), source_id="a")  # IMAGE_PIXEL
        fp_b = PolygonFootprint(
            vertices=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            coordinate_frame="SELENOGRAPHIC_IAU2000",
            source_id="b",
        )
        with self.assertRaises(ValueError):
            fp_a.intersect(fp_b)

    def test_invalid_polygon_rejection(self):
        # Non-finite coordinates
        fp_nan = PolygonFootprint(np.array([[0, np.nan], [10, 0], [10, 10]]))
        self.assertFalse(fp_nan.is_valid())

        # Zero area degenerate line
        fp_zero = PolygonFootprint(np.array([[0, 0], [10, 0], [20, 0]]))
        self.assertFalse(fp_zero.is_valid())

    def test_mosaic_footprint_aggregation(self):
        fp1 = PolygonFootprint(
            vertices=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            coordinate_frame="MOSAIC_CANVAS_PIXEL",
            source_id="0",
        )
        fp2 = PolygonFootprint(
            vertices=np.array([[50, 0], [150, 0], [150, 100], [50, 100]]),
            coordinate_frame="MOSAIC_CANVAS_PIXEL",
            source_id="1",
        )
        mosaic_fp = create_mosaic_footprint([fp1, fp2], canvas_shape=(100, 150))
        self.assertEqual(mosaic_fp["canvas_width"], 150)
        self.assertEqual(mosaic_fp["canvas_height"], 100)
        self.assertEqual(mosaic_fp["canvas_area_pixels"], 15000.0)
        self.assertEqual(mosaic_fp["valid_footprint_area_pixels"], 15000.0)
        self.assertEqual(mosaic_fp["footprint_to_canvas_ratio"], 1.0)
        self.assertEqual(mosaic_fp["contributing_footprint_count"], 2)


class TestPhase6CoordinatesAndPlanetaryInterfaces(unittest.TestCase):
    """Step 6, 11, 12: TransformRecord and SPICE/DEM Extension Interfaces."""

    def test_transform_record_inversion(self):
        H = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        rec = TransformRecord(
            source_frame=CoordinateFrame.IMAGE_PIXEL,
            target_frame=CoordinateFrame.REFERENCE_IMAGE_PIXEL,
            matrix=H,
            units="pixels",
        )
        inv_rec = rec.invert()
        self.assertEqual(inv_rec.source_frame, CoordinateFrame.REFERENCE_IMAGE_PIXEL)
        self.assertEqual(inv_rec.target_frame, CoordinateFrame.IMAGE_PIXEL)

        # Transform point through forward and inverse
        pt = np.array([[20.0, 30.0]])
        pt_fwd = rec.transform_points(pt)
        self.assertEqual(pt_fwd[0, 0], 120.0)
        self.assertEqual(pt_fwd[0, 1], 80.0)

        pt_back = inv_rec.transform_points(pt_fwd)
        self.assertAlmostEqual(pt_back[0, 0], 20.0, delta=1e-4)
        self.assertAlmostEqual(pt_back[0, 1], 30.0, delta=1e-4)

    def test_planetary_extension_stubs_scientific_discipline(self):
        # SPICE interface must report unavailable with explicit scientific disclaimer
        self.assertFalse(SPICEInterface.is_available())
        spice_resp = SPICEInterface.get_spacecraft_geometry("CH2_OHR_123")
        self.assertEqual(spice_resp["status"], "INTERFACE_PREPARED_SPICE_UNAVAILABLE")
        self.assertIn("SPICE ephemeris and pointing kernels not present", spice_resp["scientific_note"])

        # DEM interface must report unavailable
        self.assertFalse(DEMInterface.is_available())
        dem_resp = DEMInterface.get_elevation_profile(latitude=12.5, longitude=-45.0)
        self.assertEqual(dem_resp["status"], "INTERFACE_PREPARED_DEM_UNAVAILABLE")

        # Camera model interface
        self.assertFalse(CameraModelInterface.is_available())


class TestPhase6MultiRegistrationIntegration(unittest.TestCase):
    """Step 17: Multi-Image Registration Integration with Geospatial Footprint Layer."""

    def test_multi_registration_with_metadata_and_footprints(self):
        # Generate 2 overlapping textured synthetic lunar patches
        rng = np.random.default_rng(26166)
        y, x = np.mgrid[0:400, 0:400].astype(np.float32)
        base = 120.0 + 40.0 * np.sin(x / 30.0) * np.cos(y / 30.0) + rng.normal(0, 10, (400, 400))
        surface = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        img0 = surface[50:350, 50:350].copy()
        img1 = surface[50:350, 80:380].copy()

        metas = [
            {"instrument": "OHRC", "gsd_m": 0.25, "product_id": "SYN_OHRC_001"},
            {"instrument": "OHRC", "gsd_m": 0.25, "product_id": "SYN_OHRC_002"},
        ]

        res = run_multi_registration([img0, img1], metadata_list=metas)

        self.assertEqual(res.summary["images_registered"], 2)
        self.assertIsNotNone(res.mosaic_info)
        self.assertIn("mosaic_footprint", res.mosaic_info)
        mfp = res.mosaic_info["mosaic_footprint"]
        self.assertEqual(mfp["contributing_footprint_count"], 2)
        self.assertEqual(mfp["coordinate_frame"], "MOSAIC_CANVAS_PIXEL")
        self.assertGreater(mfp["valid_footprint_area_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
