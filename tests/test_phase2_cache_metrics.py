import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "backend")

from app.services.metrics import spatial_grid
from app.services.multi_registration import run_multi_registration
from app.services.registration_cache import RegistrationCache, image_fingerprint


def sample_images(count=2):
    source = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(source, (80, 80), 20, 255, -1)
    cv2.rectangle(source, (140, 120), (200, 180), 180, -1)
    cv2.line(source, (30, 200), (200, 30), 220, 3)
    return [cv2.warpAffine(source, np.float32([[1, 0, index * 8], [0, 1, index * 5]]), (240, 240)) for index in range(count)]


class PhaseTwoCacheAndMetricsTests(unittest.TestCase):
    def test_fingerprints_change_only_with_image_content(self):
        image = sample_images()[0]
        self.assertEqual(image_fingerprint(image), image_fingerprint(image.copy()))
        image[0, 0] = 1
        self.assertNotEqual(image_fingerprint(sample_images()[0]), image_fingerprint(image))

    def test_configuration_changes_invalidate_feature_and_pair_keys(self):
        cache = RegistrationCache(Path(tempfile.mkdtemp()))
        image = sample_images()[0]
        _, _, _, first_hit, _ = cache.get_features(image, {"detector": "sift", "max_features": 1000})
        _, _, _, second_hit, _ = cache.get_features(image, {"detector": "sift", "max_features": 1000})
        _, _, _, changed_hit, _ = cache.get_features(image, {"detector": "orb", "max_features": 1000})
        self.assertFalse(first_hit); self.assertTrue(second_hit); self.assertFalse(changed_hit)
        fp = image_fingerprint(image)
        self.assertNotEqual(cache.pair_key(fp, fp, {"ratio": .72}), cache.pair_key(fp, fp, {"ratio": .8}))

    def test_repeat_and_incremental_runs_reuse_old_pair_evidence(self):
        cache_dir, images = Path(tempfile.mkdtemp()), sample_images(3)
        first = run_multi_registration(images, {"ecc_refinement": False, "max_features": 2000}, cache_dir)
        repeat = run_multi_registration(images, {"ecc_refinement": False, "max_features": 2000}, cache_dir)
        added = run_multi_registration(sample_images(4), {"ecc_refinement": False, "max_features": 2000}, cache_dir)
        self.assertEqual(first.summary["processed_pairs"], 3)
        self.assertEqual(repeat.summary["processed_pairs"], 0)
        self.assertEqual(repeat.summary["reused_pairs"], 3)
        self.assertEqual(added.summary["processed_pairs"], 3)
        self.assertEqual(added.summary["reused_pairs"], 3)

    def test_spatial_grid_counts_and_coverage_are_real(self):
        grid = spatial_grid(np.array([[1, 1], [2, 2], [99, 99], [50, 50]], dtype=np.float32), 100, 100)
        self.assertEqual(grid["total_inliers"], 4)
        self.assertEqual(grid["populated_cells"], 3)
        self.assertEqual(grid["coverage_percent"], 18.75)

