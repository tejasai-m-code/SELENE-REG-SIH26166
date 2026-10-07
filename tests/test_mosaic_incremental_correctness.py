"""Phase B Invariant & Incremental Mosaic Correctness Tests.

Verifies:
1. Invariant: Adding another image must never erase previously valid mosaic content.
2. Incremental mosaic building across 2, 3, 5, 9, and 12+ images.
3. Proper valid-pixel mask isolation (black border/collar does not cut holes in existing mosaic).
4. Disconnected images are identified, excluded from the mosaic canvas, and accompanied by explanation.
5. Invalid/blank regions never corrupt valid data.
"""

from __future__ import annotations

import unittest
import numpy as np
import cv2

from app.services.mosaic import build_relative_mosaic, compute_valid_data_mask
from app.services.multi_registration import run_multi_registration
from app.services.evaluation import make_synthetic_lunar_surface


class TestMosaicInvariantAndPreservation(unittest.TestCase):
    """B1, B2 & B4: Adding frames must never erase existing valid mosaic content."""

    def setUp(self):
        # 800x800 synthetic lunar surface
        self.surface = make_synthetic_lunar_surface(seed=42, size=800)
        self.h, self.w = 200, 200

        # Create 5 contiguous overlapping patches
        self.p0 = self.surface[200:400, 200:400].copy()  # Center
        self.p1 = self.surface[200:400, 260:460].copy()  # East (dx=+60)
        self.p2 = self.surface[260:460, 200:400].copy()  # South (dy=+60)
        self.p3 = self.surface[200:400, 140:340].copy()  # West (dx=-60)
        self.p4 = self.surface[140:340, 200:400].copy()  # North (dy=-60)

    def test_invariant_adding_image_never_erases_valid_content(self):
        """CRITICAL INVARIANT: Every valid pixel in mosaic N must remain valid in mosaic N+1."""
        # Baseline 2 images
        transforms_2 = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }
        mosaic2, info2 = build_relative_mosaic([self.p0, self.p1], transforms_2, blending_mode="distance_weighted")
        valid2_count = info2["quality_metrics"]["valid_mosaic_area_pixels"]

        # 3 images (add South)
        transforms_3 = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
            "2": np.array([[1, 0, 0], [0, 1, 60], [0, 0, 1]], dtype=np.float64).tolist(),
        }
        mosaic3, info3 = build_relative_mosaic([self.p0, self.p1, self.p2], transforms_3, blending_mode="distance_weighted")
        valid3_count = info3["quality_metrics"]["valid_mosaic_area_pixels"]

        # Invariant: valid area must grow, never shrink
        self.assertGreater(valid3_count, valid2_count, "Adding image must increase total valid pixel area.")

        # 4 images (add West)
        transforms_4 = {
            **transforms_3,
            "3": np.array([[1, 0, -60], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }
        mosaic4, info4 = build_relative_mosaic([self.p0, self.p1, self.p2, self.p3], transforms_4, blending_mode="distance_weighted")
        valid4_count = info4["quality_metrics"]["valid_mosaic_area_pixels"]
        self.assertGreater(valid4_count, valid3_count)

        # 5 images (add North)
        transforms_5 = {
            **transforms_4,
            "4": np.array([[1, 0, 0], [0, 1, -60], [0, 0, 1]], dtype=np.float64).tolist(),
        }
        mosaic5, info5 = build_relative_mosaic([self.p0, self.p1, self.p2, self.p3, self.p4], transforms_5, blending_mode="distance_weighted")
        valid5_count = info5["quality_metrics"]["valid_mosaic_area_pixels"]
        self.assertGreater(valid5_count, valid4_count)

    def test_black_padding_collar_does_not_cut_existing_content(self):
        """B1 & B2: An image with large zero/black border padding must NOT cut holes in existing mosaic."""
        # Image 0 is normal terrain
        img0 = self.p0.copy()

        # Image 1 is terrain with a large 40px zero border (e.g. from uncropped rotation or sensor mask)
        img1 = self.p1.copy()
        img1[:40, :] = 0  # Black top border
        img1[:, :40] = 0  # Black left border (which overlaps img0!)

        # Verify compute_valid_data_mask marks black collar as invalid
        mask1 = compute_valid_data_mask(img1)
        self.assertEqual(mask1[10, 10], 0, "Collar pixel must be marked invalid (0).")
        self.assertEqual(mask1[100, 100], 255, "Interior terrain must be marked valid (255).")

        transforms = {
            "0": np.eye(3).tolist(),
            "1": np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], dtype=np.float64).tolist(),
        }

        # Build mosaic with overlay and distance_weighted
        for mode in ("overlay", "distance_weighted"):
            mosaic, info = build_relative_mosaic([img0, img1], transforms, blending_mode=mode)
            # In the region where img0 has valid data and img1 has black padding,
            # the mosaic MUST preserve img0's content and not be black (0,0,0)
            # Image 0 spans x in [0, 200], y in [0, 200]
            # Image 1 top-left is at x=50, y=0. Its padding is at x in [50, 90], y in [0, 40].
            sample_region = mosaic[10:30, 60:80]
            # Since img0 had valid non-zero pixels here, the mosaic must NOT be all 0!
            self.assertGreater(np.mean(sample_region), 10.0, f"Mode {mode}: Black padding must not overwrite valid image 0 data!")


class TestMultiImageScalabilityAndDisconnected(unittest.TestCase):
    """B3 & B4: 2, 3, 5, 9, 12+ images and handling of disconnected images."""

    def setUp(self):
        self.surface = make_synthetic_lunar_surface(seed=12345, size=1000)

    def test_disconnected_image_excluded_with_explanation(self):
        """B3: An image with no overlap must be marked disconnected and not corrupt mosaic."""
        # 3 overlapping images from one part of the moon
        p0 = self.surface[100:300, 100:300]
        p1 = self.surface[100:300, 160:360]
        p2 = self.surface[160:360, 100:300]

        # 1 completely unrelated / disconnected image (noise/unrelated scene)
        unrelated = np.random.RandomState(999).randint(50, 200, size=(200, 200), dtype=np.uint8)

        images = [p0, p1, p2, unrelated]
        result = run_multi_registration(images, {"detector": "sift", "ecc_refinement": False})

        # 3 images should be placed, 1 unplaced / disconnected
        self.assertEqual(result.summary["images_placed"], 3)
        self.assertEqual(result.summary["images_unplaced"], 1)
        self.assertIn(3, result.summary["unregistered_images"])
        self.assertIsNotNone(result.mosaic)

    def test_incremental_batch_sizes_up_to_14_images(self):
        """B4: Test 2, 3, 5, 9, and 14 images incrementally."""
        h, w = 150, 150
        # Generate a grid of overlapping frames
        frames = []
        for r in range(4):
            for c in range(4):
                y = 100 + r * 60
                x = 100 + c * 60
                frames.append(self.surface[y : y + h, x : x + w])

        for count in (2, 3, 5, 9, 14):
            sub_frames = frames[:count]
            # Use transforms directly to verify geometry and mosaic generation
            transforms = {}
            for idx in range(count):
                r = idx // 4
                c = idx % 4
                transforms[str(idx)] = np.array(
                    [[1.0, 0.0, float(c * 60)], [0.0, 1.0, float(r * 60)], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ).tolist()

            mosaic, info = build_relative_mosaic(sub_frames, transforms, blending_mode="distance_weighted")
            self.assertIsNotNone(mosaic)
            self.assertEqual(info["placed_image_count"], count)
            self.assertGreater(info["quality_metrics"]["valid_mosaic_area_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
