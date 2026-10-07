"""Unit tests for dataset scanning, manifest generation, and scientific format handling."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from app.services.dataset_scanner import (
    scan_local_dataset,
    read_pds_label,
    SUPPORTED_RASTER_EXTENSIONS,
    METADATA_EXTENSIONS,
)


class TestDatasetScanner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="selene_test_dataset_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scan_demo_data_directory(self):
        """Verify scanner on real workspace demo_data directory."""
        demo_dir = Path("demo_data").resolve()
        if not demo_dir.exists():
            self.skipTest("demo_data directory not found")

        out_dir = Path(self.temp_dir) / "manifests"
        manifest = scan_local_dataset(
            folder_path=str(demo_dir),
            dataset_name="Demo Chandrayaan-2 Strip",
            output_dir=str(out_dir),
            public_prefix="/api/outputs/manifests",
        )

        self.assertGreater(manifest["files_discovered"], 0)
        self.assertGreater(manifest["images_count"], 0)
        self.assertIn("manifest_id", manifest)
        self.assertEqual(manifest["dataset_name"], "Demo Chandrayaan-2 Strip")

        # Check file items
        files = manifest["files"]
        ready_images = [f for f in files if f["status"] == "READY"]
        self.assertGreater(len(ready_images), 0)

        for img in ready_images:
            fmt = img["format"].lower()
            self.assertTrue(fmt in SUPPORTED_RASTER_EXTENSIONS or f".{fmt}" in SUPPORTED_RASTER_EXTENSIONS)
            self.assertIsNotNone(img["dimensions"])
            self.assertGreater(img["size_bytes"], 0)

    def test_unsupported_file_reporting_not_silent(self):
        """Unsupported files must report explicit diagnostic reason, not fail silently."""
        unsupported_file = Path(self.temp_dir) / "document.docx"
        unsupported_file.write_text("Binary mock docx content")

        txt_file = Path(self.temp_dir) / "notes.txt"
        txt_file.write_text("Scientific observer field notes")

        manifest = scan_local_dataset(
            folder_path=self.temp_dir,
            dataset_name="Test Unsupported",
            output_dir=self.temp_dir,
        )

        self.assertEqual(manifest["files_discovered"], 2)
        self.assertEqual(manifest["images_count"], 0)
        self.assertGreater(manifest["unsupported_count"], 0)

        unsupported_items = [f for f in manifest["files"] if f["status"] == "UNSUPPORTED"]
        self.assertTrue(len(unsupported_items) >= 1)
        for item in unsupported_items:
            self.assertIsNotNone(item["reason"])
            self.assertTrue("Non-raster" in item["reason"] or "Unsupported" in item["reason"])

    def test_recursive_nested_structures(self):
        """Verify recursive folder discovery across nested calibrated subdirectories."""
        sub1 = Path(self.temp_dir) / "calibrated" / "orbit_102"
        sub1.mkdir(parents=True, exist_ok=True)

        sub2 = Path(self.temp_dir) / "calibrated" / "orbit_103" / "browse"
        sub2.mkdir(parents=True, exist_ok=True)

        img1 = np.full((100, 100), 128, dtype=np.uint8)
        cv2.imwrite(str(sub1 / "frame_a.png"), img1)

        img2 = np.full((120, 120), 200, dtype=np.uint8)
        cv2.imwrite(str(sub2 / "frame_b.tif"), img2)

        manifest = scan_local_dataset(
            folder_path=self.temp_dir,
            dataset_name="Nested Orbits",
            output_dir=self.temp_dir,
        )

        self.assertEqual(manifest["images_count"], 2)
        filenames = {f["filename"] for f in manifest["files"]}
        self.assertIn("frame_a.png", filenames)
        self.assertIn("frame_b.tif", filenames)

    def test_pds_label_extraction_without_fabrication(self):
        """Verify PDS3/4 label parser extracts true metadata and does not fabricate values."""
        lbl_content = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 2048
INSTRUMENT_NAME = "OHRC"
TARGET_NAME = "MOON"
SPACECRAFT_NAME = "CHANDRAYAAN-2"
IMAGE_LINES = 1024
LINE_SAMPLES = 1024
PIXEL_RESOLUTION = 0.25 <METERS/PIXEL>
END
"""
        lbl_file = Path(self.temp_dir) / "ch2_ohrc_001.lbl"
        lbl_file.write_text(lbl_content)

        meta = read_pds_label(str(lbl_file))
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("INSTRUMENT_NAME"), "OHRC")
        self.assertEqual(meta.get("TARGET_NAME"), "MOON")
        self.assertEqual(int(meta.get("IMAGE_LINES")), 1024)
        self.assertEqual(meta.get("dimensions"), [1024, 1024])
        self.assertEqual(meta.get("gsd_m"), 0.25)
        # Should not fabricate non-existent keys
        self.assertNotIn("SUN_AZIMUTH", meta)


if __name__ == "__main__":
    unittest.main()
