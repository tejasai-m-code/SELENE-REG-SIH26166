"""Comprehensive Phase 1 Ingestion, Scientific Data Inspection, and Sensor Identification Tests.

Mandated by SIH 26166 Phase 1:
1. Native directory / folder scanning and recursive file discovery
2. Multiple file ingestion and manifest generation
3. PDS3, PDS4, GeoTIFF, XML, and JSON metadata inspection
4. Evidence-based sensor auto-identification (OHRC, TMC-2, IIRS, LROC NAC/WAC, Kaguya, Unknown)
5. Layer & band count clarification (1-channel panchromatic vs 3-channel RGB vs multiband)
6. Bit-depth inspection (8-bit, 16-bit uint16, float32)
7. Raw scientific data preservation and display preview decoupling
8. Radiometric statistics: min, max, mean, median, std, dynamic range span, valid %, saturation %
9. 32-bin radiometric histogram generation
10. Format matrix: PNG, JPEG, TIFF (8-bit and 16-bit), and uncompressed rasters
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

# Ensure api-server python directory is in sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))

from app.services.scientific_data import (
    ScientificRaster,
    compute_scientific_inspection,
    create_display_preview,
    identify_sensor_evidence,
    ingest_scientific_image,
)
from app.services.dataset_scanner import scan_local_dataset
from worker import _metadata, register


class TestPhase1ScientificIngestion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="selene_phase1_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_raw_16bit_preservation_and_preview_decoupling(self):
        """Verify 16-bit TIFF dynamic range is preserved and display preview is decoupled."""
        h, w = 150, 150
        raw_16u = np.linspace(1000, 52000, h * w, dtype=np.uint16).reshape((h, w))
        tif_path = self.temp_dir / "sample_16bit.tif"
        cv2.imwrite(str(tif_path), raw_16u)

        sci_raster = ingest_scientific_image(tif_path)

        # 1. Scientific data must preserve uint16 and full dynamic range
        self.assertEqual(sci_raster.raw_data.dtype, np.uint16)
        self.assertEqual(sci_raster.raw_data.shape, (h, w))
        self.assertEqual(int(sci_raster.raw_data.min()), 1000)
        self.assertEqual(int(sci_raster.raw_data.max()), 52000)

        # 2. Inspection must record 16-bit and accurate DN metrics
        insp = sci_raster.inspection
        self.assertEqual(insp.bit_depth, 16)
        self.assertEqual(insp.dtype, "uint16")
        self.assertAlmostEqual(insp.min_dn, 1000.0, places=1)
        self.assertAlmostEqual(insp.max_dn, 52000.0, places=1)
        self.assertAlmostEqual(insp.dynamic_range_span, 51000.0, places=1)
        self.assertEqual(insp.valid_pixel_pct, 100.0)

        # 3. Display preview must be normalized 8-bit copy without mutating raw data
        self.assertEqual(sci_raster.display_data.dtype, np.uint8)
        self.assertEqual(sci_raster.display_data.shape, (h, w))
        self.assertEqual(sci_raster.raw_data.dtype, np.uint16)  # Raw remains intact!

    def test_02_evidence_based_sensor_identification_matrix(self):
        """Verify sensor auto-identification uses real evidence and never fabricates labels."""
        # A. Chandrayaan-2 OHRC filename token
        ohrc_res = identify_sensor_evidence("ch2_ohr_ncp_20200115_d18.png")
        self.assertIn("OHRC", ohrc_res.sensor)
        self.assertEqual(ohrc_res.mission, "Chandrayaan-2")
        self.assertEqual(ohrc_res.gsd_m, 0.25)
        self.assertEqual(ohrc_res.metadata_status, "INFERRED_FROM_FILENAME")
        self.assertGreater(len(ohrc_res.evidence), 0)

        # B. Chandrayaan-2 TMC-2 filename token
        tmc_res = identify_sensor_evidence("ch2_tmc_ncp_strip1.tif")
        self.assertIn("TMC-2", tmc_res.sensor)
        self.assertEqual(tmc_res.mission, "Chandrayaan-2")
        self.assertEqual(tmc_res.gsd_m, 5.0)

        # C. Chandrayaan-2 IIRS filename token
        iirs_res = identify_sensor_evidence("ch2_iir_band12.tif")
        self.assertIn("IIRS", iirs_res.sensor)
        self.assertEqual(iirs_res.mission, "Chandrayaan-2")
        self.assertEqual(iirs_res.gsd_m, 80.0)

        # D. LROC NAC filename pattern
        nac_res = identify_sensor_evidence("M1105199859RE_thumb.png")
        self.assertIn("LRO NAC", nac_res.sensor)
        self.assertEqual(nac_res.mission, "Lunar Reconnaissance Orbiter")
        self.assertEqual(nac_res.gsd_m, 0.5)

        # E. LROC WAC filename pattern
        wac_res = identify_sensor_evidence("lroc_wac_polar_mosaic.tif")
        self.assertIn("LRO WAC", wac_res.sensor)
        self.assertEqual(wac_res.gsd_m, 100.0)

        # F. SELENE / Kaguya filename pattern
        kaguya_res = identify_sensor_evidence("TC_SP_01_N80E000.png")
        self.assertIn("SELENE", kaguya_res.sensor)
        self.assertEqual(kaguya_res.mission, "SELENE (Kaguya)")

        # G. Unknown image without metadata -> MUST NOT FABRICATE SENSOR!
        unknown_res = identify_sensor_evidence("custom_moon_crater_patch.png")
        self.assertEqual(unknown_res.sensor, "UNKNOWN / NEEDS METADATA")
        self.assertEqual(unknown_res.confidence_pct, 0.0)
        self.assertEqual(unknown_res.metadata_status, "UNKNOWN")
        self.assertIsNone(unknown_res.gsd_m)

        # H. Metadata override verification
        meta_supplied = {"sensor": "CHANDRAYAAN-2 OHRC", "gsd_m": 0.25}
        verified_res = identify_sensor_evidence("arbitrary_filename.png", metadata=meta_supplied)
        self.assertIn("OHRC", verified_res.sensor)
        self.assertEqual(verified_res.metadata_status, "VERIFIED")
        self.assertEqual(verified_res.confidence_pct, 95.0)

    def test_03_layer_and_channel_classification(self):
        """Verify layer count investigation clarifies channels vs scientific layers."""
        # 1-channel grayscale
        gray = np.full((64, 64), 128, dtype=np.uint8)
        insp_gray = compute_scientific_inspection(gray)
        self.assertEqual(insp_gray.channels, 1)
        self.assertEqual(insp_gray.scientific_bands, 1)
        self.assertIn("Panchromatic Grayscale", insp_gray.layer_classification)

        # 3-channel RGB
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        insp_rgb = compute_scientific_inspection(rgb)
        self.assertEqual(insp_rgb.channels, 3)
        self.assertEqual(insp_rgb.scientific_bands, 3)
        self.assertIn("RGB", insp_rgb.layer_classification)

        # 4-channel RGBA
        rgba = np.full((64, 64, 4), 255, dtype=np.uint8)
        insp_rgba = compute_scientific_inspection(rgba)
        self.assertEqual(insp_rgba.channels, 4)
        self.assertTrue(insp_rgba.has_alpha)
        self.assertIn("Alpha", insp_rgba.layer_classification)

    def test_04_radiometric_histogram_and_saturation(self):
        """Verify 32-bin histogram edges and saturation percentage computation."""
        # Saturated image: 20% pixels at 255
        arr = np.full((100, 100), 50, dtype=np.uint8)
        arr[:20, :] = 255  # 2000 pixels = 20%
        insp = compute_scientific_inspection(arr)

        self.assertEqual(len(insp.histogram_bins), 33)
        self.assertEqual(len(insp.histogram_counts), 32)
        self.assertEqual(sum(insp.histogram_counts), 10000)
        self.assertAlmostEqual(insp.saturation_pct, 20.0, places=1)
        self.assertEqual(insp.min_dn, 50.0)
        self.assertEqual(insp.max_dn, 255.0)

    def test_05_format_matrix_ingestion(self):
        """Test ingestion of standard formats (PNG, JPG, TIFF) without errors."""
        base_arr = np.random.randint(20, 220, (120, 120), dtype=np.uint8)

        # PNG
        png_path = self.temp_dir / "test.png"
        cv2.imwrite(str(png_path), base_arr)
        sci_png = ingest_scientific_image(png_path)
        self.assertEqual(sci_png.inspection.dimensions[:2], [120, 120])

        # JPEG
        jpg_path = self.temp_dir / "test.jpg"
        cv2.imwrite(str(jpg_path), base_arr)
        sci_jpg = ingest_scientific_image(jpg_path)
        self.assertEqual(sci_jpg.inspection.dimensions[:2], [120, 120])

        # 8-bit TIFF
        tif_path = self.temp_dir / "test.tif"
        cv2.imwrite(str(tif_path), base_arr)
        sci_tif = ingest_scientific_image(tif_path)
        self.assertEqual(sci_tif.inspection.bit_depth, 8)

    def test_06_folder_scanning_with_manifest_enrichment(self):
        """Verify recursive folder scanner populates scientific inspection and sensor evidence."""
        sub = self.temp_dir / "orbit_105"
        sub.mkdir(parents=True, exist_ok=True)

        img1 = np.full((100, 100), 100, dtype=np.uint8)
        cv2.imwrite(str(sub / "ch2_ohr_sample1.png"), img1)

        img2 = np.full((120, 120), 4000, dtype=np.uint16)
        cv2.imwrite(str(sub / "ch2_tmc_sample2.tif"), img2)

        out_dir = self.temp_dir / "manifest_out"
        manifest = scan_local_dataset(self.temp_dir, dataset_name="Test Strip", output_dir=out_dir)

        self.assertEqual(manifest["images_count"], 2)
        files = manifest["files"]
        ohr_file = next(f for f in files if "ohr" in f["filename"].lower())
        tmc_file = next(f for f in files if "tmc" in f["filename"].lower())

        self.assertIn("OHRC", ohr_file["sensor"])
        self.assertEqual(ohr_file["bit_depth"], 8)
        self.assertEqual(ohr_file["status"], "READY")

        self.assertIn("TMC", tmc_file["sensor"])
        self.assertEqual(tmc_file["bit_depth"], 16)
        self.assertEqual(tmc_file["dtype"], "uint16")
        self.assertGreater(tmc_file["min_dn"], 3990)

    def test_07_worker_metadata_integration(self):
        """Verify worker _metadata and register attach scientific_inspection and sensor evidence."""
        img16 = np.full((80, 80), 5000, dtype=np.uint16)
        path = self.temp_dir / "ch2_ohr_test_worker.png"
        cv2.imwrite(str(path), img16)

        sci = ingest_scientific_image(path)
        meta = _metadata(path, sci.display_data, {}, sci_raster=sci)

        self.assertIn("scientific_inspection", meta)
        self.assertIn("sensor_identification", meta)
        self.assertEqual(meta["bit_depth"], 16)
        self.assertIn("OHRC", meta["sensor"])
        self.assertEqual(meta["sensor_source"], "INFERRED_FROM_FILENAME")


if __name__ == "__main__":
    unittest.main()
