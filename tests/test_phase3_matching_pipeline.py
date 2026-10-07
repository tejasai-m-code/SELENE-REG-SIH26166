import unittest
import numpy as np
import cv2

from app.services.feature_matching import (
    create_detector,
    compute_features,
    match_feature_artifacts,
    match_features,
    spatially_distribute_matches,
    get_detector_capabilities,
)
from app.services.gpu_accelerator import (
    match_descriptors_gpu,
    is_cuda_available,
)
from app.services.pairwise_registration import (
    register_pair,
    serialize_correspondences,
    serialize_inlier_points,
)


class TestPhase3MatchingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        h, w = 450, 450
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        cls.image = cv2.GaussianBlur(img, (7, 7), 1.5)
        # Add craters and distinct features across quadrants
        for i in range(16):
            cx = np.random.randint(50, w - 50)
            cy = np.random.randint(50, h - 50)
            r = np.random.randint(10, 25)
            cv2.circle(cls.image, (cx, cy), r, (255, 255, 255), -1)
            cv2.circle(cls.image, (cx, cy), int(r * 0.6), (30, 30, 30), -1)
            cv2.rectangle(cls.image, (cx - 6, cy - 6), (cx + 6, cy + 6), (180, 180, 180), -1)

    def test_detector_capabilities_audit(self):
        """Audit detector registry: classical available, uninstalled learned marked accurately."""
        caps = get_detector_capabilities()
        self.assertIn("sift", caps)
        self.assertIn("orb", caps)
        self.assertIn("akaze", caps)
        self.assertIn("loftr", caps)
        self.assertIn("superpoint", caps)

        self.assertEqual(caps["sift"]["status"], "AVAILABLE")
        self.assertEqual(caps["orb"]["status"], "AVAILABLE")
        self.assertIn(caps["loftr"]["status"], ["NOT AVAILABLE", "AVAILABLE"])
        self.assertEqual(caps["superpoint"]["status"], "NOT AVAILABLE")

        # Requesting unavailable learned matcher must raise explicit ValueError
        with self.assertRaises(ValueError) as ctx:
            create_detector("superpoint")
        self.assertIn("NOT AVAILABLE", str(ctx.exception))

    def test_classical_detectors_extraction(self):
        """Verify SIFT, ORB, and AKAZE extract valid sub-pixel keypoints and descriptors."""
        for d_name in ["sift", "orb", "akaze"]:
            feat = compute_features(self.image, detector_name=d_name, max_features=1000)
            self.assertGreater(len(feat.points), 50, f"{d_name} found too few keypoints")
            self.assertIsNotNone(feat.descriptors)
            self.assertEqual(len(feat.points), len(feat.descriptors))
            # Verify coordinates are floating-point sub-pixel within bounds
            self.assertTrue(np.issubdtype(feat.points.dtype, np.floating))
            self.assertTrue((feat.points[:, 0] >= 0).all() and (feat.points[:, 0] < self.image.shape[1]).all())
            self.assertTrue((feat.points[:, 1] >= 0).all() and (feat.points[:, 1] < self.image.shape[0]).all())

    def test_lowe_ratio_and_mutual_cross_check_filtering(self):
        """Verify ratio test and mutual consistency filter out ambiguous matches."""
        dx, dy = 15.0, -10.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]), borderMode=cv2.BORDER_REFLECT)

        feat_s = compute_features(self.image, "sift", 1000)
        feat_r = compute_features(shifted, "sift", 1000)

        # Standard ratio test + mutual cross-check
        match_res = match_feature_artifacts(
            feat_s, feat_r, detector_name="sift", ratio=0.75, mutual_check=True
        )

        self.assertGreater(len(match_res.good_matches), 30)
        self.assertIn("ratio_passed", match_res.diagnostics)
        self.assertIn("mutual_consistent", match_res.diagnostics)
        self.assertIn("unique_matches", match_res.diagnostics)

        # Verify no duplicate reference indices in matches
        ref_indices = [m.trainIdx for m in match_res.good_matches]
        self.assertEqual(len(ref_indices), len(set(ref_indices)), "Found duplicate reference matches")

    def test_deterministic_match_confidence(self):
        """Verify real deterministic match confidence values in [0.05, 0.99]."""
        dx, dy = 12.0, 8.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]), borderMode=cv2.BORDER_REFLECT)

        feat_s = compute_features(self.image, "sift", 500)
        feat_r = compute_features(shifted, "sift", 500)
        match_res = match_feature_artifacts(feat_s, feat_r, detector_name="sift", ratio=0.75)

        self.assertEqual(len(match_res.good_matches), len(match_res.match_confidences))
        for conf in match_res.match_confidences:
            self.assertGreaterEqual(conf, 0.05)
            self.assertLessEqual(conf, 0.99)

    def test_spatial_distribution_filtering(self):
        """Verify bucketed spatial grid distribution prevents localized clustering."""
        feat_s = compute_features(self.image, "sift", 1500)
        feat_r = compute_features(self.image, "sift", 1500)
        match_res = match_feature_artifacts(feat_s, feat_r, detector_name="sift", ratio=0.8)

        distributed = spatially_distribute_matches(
            match_res,
            self.image.shape,
            grid_rows=4,
            grid_cols=4,
            max_per_cell=15,
        )

        self.assertLessEqual(len(distributed.good_matches), len(match_res.good_matches))
        self.assertGreater(len(distributed.good_matches), 10)
        self.assertEqual(len(distributed.source_points), len(distributed.good_matches))
        self.assertIn("spatial_coverage", distributed.diagnostics)
        self.assertGreater(distributed.diagnostics["spatial_coverage"], 0.25)

    def test_coordinate_correspondence_integrity_and_serialization(self):
        """Verify source/reference coordinates match ground-truth displacement and serialize accurately."""
        dx, dy = 20.0, -15.0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]), borderMode=cv2.BORDER_REFLECT)

        res = register_pair(self.image, shifted, detector="sift")
        self.assertTrue(res.success)
        self.assertGreater(int(np.sum(res.inlier_mask)), 15)

        corrs = serialize_correspondences(res)
        self.assertGreater(len(corrs), 0)
        first_inlier = next((c for c in corrs if c["is_inlier"]), None)
        self.assertIsNotNone(first_inlier)
        self.assertIn("source", first_inlier)
        self.assertIn("reference", first_inlier)
        self.assertIn("confidence", first_inlier)

        # Coordinate correspondence displacement check on inliers
        src_pt = np.array(first_inlier["source"])
        ref_pt = np.array(first_inlier["reference"])
        expected_ref = src_pt + np.array([dx, dy])
        displacement_err = np.linalg.norm(ref_pt - expected_ref)
        self.assertLess(displacement_err, 2.0, "Matched points deviate from geometric ground truth")

    def test_gpu_accelerated_descriptor_matching(self):
        """Verify PyTorch CUDA descriptor matching executes accurately with CPU fallback."""
        rng = np.random.default_rng(42)
        n = 200
        dim = 128
        des_s = rng.standard_normal((n, dim)).astype(np.float32)
        # Reference has 50 identical descriptors + shift
        des_r = rng.standard_normal((n, dim)).astype(np.float32)
        des_r[:50] = des_s[:50]

        matches, diag = match_descriptors_gpu(des_s, des_r, ratio=0.75, mutual_check=True)
        self.assertGreater(len(matches), 30)
        self.assertIn(diag["acceleration"], ["GPU_CUDA", "CPU_FALLBACK"])
        self.assertGreater(diag["execution_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
