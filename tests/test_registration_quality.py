"""Regression tests for pair registration quality gate.

Tests the specific failure mode identified in the SELENE-REG-X bug report:

1. A deliberately wrong image pair must produce FAIL or REVIEW, not PASS.
2. Matches concentrated in one tiny region must not pass the quality gate.
3. Transformation plausibility validation rejects extreme scale/rotation/shear.
4. A genuinely corresponding pair with distributed matches must PASS.
5. Spatial distribution category correctly flags clustered distributions.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure api-server python directory is in sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))

from app.services.pairwise_registration import register_pair
from app.services.geometry import (
    decompose_transform,
    validate_transform_plausibility,
    estimate_geometric_model,
)
from app.services.metrics import evaluate_spatial_distribution
from app.services.registration import draw_matches


def _create_crater_scene(width=300, height=300, seed=42):
    """Generate a realistic synthetic lunar terrain image with craters."""
    rng = np.random.default_rng(seed)
    img = rng.normal(120, 15, (height, width)).astype(np.float32)
    img = cv2.GaussianBlur(img, (9, 9), 2.5)
    craters = [
        (60, 60, 22), (160, 70, 30), (80, 170, 26), (180, 180, 20),
        (120, 120, 15), (40, 130, 12), (140, 40, 10), (200, 110, 14),
    ]
    for cx, cy, r in craters:
        cv2.circle(img, (cx, cy), r, float(rng.integers(70, 95)), -1)
        cv2.circle(img, (cx, cy), r, float(rng.integers(150, 190)), 2)
        cv2.circle(img, (cx - r // 4, cy - r // 4), max(2, r // 3), float(rng.integers(50, 70)), -1)
    for _ in range(15):
        x, y, sz = int(rng.integers(10, width - 10)), int(rng.integers(10, height - 10)), int(rng.integers(2, 6))
        cv2.circle(img, (x, y), sz, float(rng.integers(80, 200)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _create_different_scene(width=300, height=300, seed=99):
    """Generate a completely different image — NOT the same terrain."""
    rng = np.random.default_rng(seed)
    img = rng.normal(160, 20, (height, width)).astype(np.float32)
    # Add random circles in different positions than _create_crater_scene
    for _ in range(10):
        x = int(rng.integers(20, width - 20))
        y = int(rng.integers(20, height - 20))
        r = int(rng.integers(5, 15))
        cv2.circle(img, (x, y), r, float(rng.integers(40, 120)), -1)
    # Add grid pattern to make it visually completely different
    for i in range(0, width, 30):
        cv2.line(img, (i, 0), (i, height), float(rng.integers(100, 200)), 1)
    for j in range(0, height, 30):
        cv2.line(img, (0, j), (width, j), float(rng.integers(100, 200)), 1)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestWrongPairRejection(unittest.TestCase):
    """A deliberately wrong image pair must NOT receive PASS."""

    def test_wrong_pair_must_fail_or_review(self):
        """Two completely different scenes should produce FAIL or REVIEW, never PASS."""
        scene_a = cv2.cvtColor(_create_crater_scene(seed=42), cv2.COLOR_GRAY2BGR)
        scene_b = cv2.cvtColor(_create_different_scene(seed=99), cv2.COLOR_GRAY2BGR)

        result = register_pair(
            scene_a, scene_b,
            detector="sift",
            ratio=0.72,
            ransac_threshold=3.0,
            include_registered=False,
        )

        self.assertIn(result.registration_status, ("FAIL", "REVIEW"),
                       f"Wrong pair should FAIL or REVIEW, got {result.registration_status}")
        self.assertFalse(result.success,
                          "Wrong pair should NOT be marked as successful")

    def test_wrong_pair_not_misleading_pass(self):
        """Even if many tentative matches exist, a wrong pair must not PASS."""
        # Use larger images to potentially generate more spurious matches
        scene_a = cv2.cvtColor(_create_crater_scene(400, 400, seed=1), cv2.COLOR_GRAY2BGR)
        scene_b = cv2.cvtColor(_create_different_scene(400, 400, seed=2), cv2.COLOR_GRAY2BGR)

        result = register_pair(
            scene_a, scene_b,
            detector="sift",
            ratio=0.80,  # Intentionally permissive ratio
            ransac_threshold=5.0,  # Intentionally permissive threshold
            max_features=10000,
            include_registered=False,
        )

        self.assertNotEqual(result.registration_status, "PASS",
                             f"Wrong pair should never PASS even with permissive settings")


class TestCorrectPairPasses(unittest.TestCase):
    """A genuinely corresponding pair with known transformation must PASS."""

    def test_same_scene_with_small_translation(self):
        """Same scene with small known translation must PASS."""
        base = cv2.cvtColor(_create_crater_scene(300, 300, seed=42), cv2.COLOR_GRAY2BGR)

        # Apply known small translation
        M = np.float32([[1, 0, 12], [0, 1, -8]])
        shifted = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT101)

        result = register_pair(
            base, shifted,
            detector="sift",
            ratio=0.72,
            ransac_threshold=3.0,
            include_registered=False,
        )

        self.assertIn(result.registration_status, ("PASS", "PASS_WITH_WARNING"),
                       f"Correct pair should PASS, got {result.registration_status}")
        self.assertTrue(result.success)

    def test_same_scene_with_rotation(self):
        """Same scene rotated by 5 degrees must PASS."""
        base = cv2.cvtColor(_create_crater_scene(300, 300, seed=42), cv2.COLOR_GRAY2BGR)

        # Apply known small rotation
        center = (150, 150)
        M = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        rotated = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT101)

        result = register_pair(
            base, rotated,
            detector="sift",
            ratio=0.72,
            ransac_threshold=3.0,
            include_registered=False,
        )

        self.assertIn(result.registration_status, ("PASS", "PASS_WITH_WARNING"),
                       f"Correct pair with small rotation should PASS, got {result.registration_status}")


class TestClusteredMatchRejection(unittest.TestCase):
    """Matches concentrated in a tiny region must be flagged as risky."""

    def test_clustered_distribution_detected(self):
        """Points concentrated in one corner should be CLUSTERED_RISKY_DISTRIBUTION."""
        # All points in a small region (top-left 20% of image)
        rng = np.random.default_rng(42)
        width, height = 400, 400
        n_points = 30
        pts = rng.uniform([0, 0], [width * 0.15, height * 0.15], size=(n_points, 2)).astype(np.float32)

        result = evaluate_spatial_distribution(pts, width, height, rows=4, cols=4)
        self.assertEqual(result["distribution_category"], "CLUSTERED_RISKY_DISTRIBUTION",
                          f"Concentrated matches should be CLUSTERED, got {result['distribution_category']}")

    def test_well_distributed_points_good(self):
        """Points spread across the image should be GOOD_DISTRIBUTION."""
        rng = np.random.default_rng(42)
        width, height = 400, 400
        n_points = 40
        pts = rng.uniform([10, 10], [width - 10, height - 10], size=(n_points, 2)).astype(np.float32)

        result = evaluate_spatial_distribution(pts, width, height, rows=4, cols=4)
        self.assertIn(result["distribution_category"],
                       ("GOOD_DISTRIBUTION", "LIMITED_DISTRIBUTION"),
                       f"Well-distributed points should not be CLUSTERED")


class TestTransformPlausibility(unittest.TestCase):
    """Transformation plausibility validation."""

    def test_identity_is_plausible(self):
        """Identity matrix should be fully plausible."""
        H = np.eye(3, dtype=np.float64)
        plausible, issues, decomp = validate_transform_plausibility(H)
        self.assertTrue(plausible, f"Identity should be plausible: {issues}")
        self.assertEqual(len(issues), 0)

    def test_small_translation_is_plausible(self):
        """Small translation should be plausible."""
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 15.0  # 15px translation
        H[1, 2] = -10.0
        plausible, issues, decomp = validate_transform_plausibility(H, source_shape=(300, 300))
        self.assertTrue(plausible, f"Small translation should be plausible: {issues}")

    def test_extreme_scale_is_implausible(self):
        """10x magnification should be implausible."""
        H = np.eye(3, dtype=np.float64)
        H[0, 0] = 10.0
        H[1, 1] = 10.0
        plausible, issues, decomp = validate_transform_plausibility(H)
        self.assertFalse(plausible, "10x scale should be implausible")
        self.assertTrue(any("magnification" in iss.lower() or "scale" in iss.lower() for iss in issues))

    def test_large_rotation_is_flagged(self):
        """60 degree rotation should be flagged."""
        angle = np.radians(60)
        c, s = np.cos(angle), np.sin(angle)
        H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        plausible, issues, decomp = validate_transform_plausibility(H, max_rotation_deg=45.0)
        self.assertFalse(plausible, "60° rotation should be implausible with 45° limit")

    def test_extreme_shear_is_implausible(self):
        """Extreme shear should be implausible."""
        H = np.array([[1, 2.0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        plausible, issues, decomp = validate_transform_plausibility(H)
        self.assertFalse(plausible, "Extreme shear should be implausible")

    def test_negative_determinant_is_implausible(self):
        """Reflection (negative determinant) should be implausible."""
        H = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        plausible, issues, decomp = validate_transform_plausibility(H)
        self.assertFalse(plausible, "Reflection should be implausible")

    def test_decompose_transform_extracts_rotation(self):
        """Verify rotation extraction from a known rotation matrix."""
        angle = np.radians(15)
        c, s = np.cos(angle), np.sin(angle)
        H = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        decomp = decompose_transform(H)
        self.assertTrue(decomp["valid"])
        self.assertAlmostEqual(decomp["rotation_degrees"], 15.0, places=1)
        self.assertAlmostEqual(decomp["scale_x"], 1.0, places=2)
        self.assertAlmostEqual(decomp["scale_y"], 1.0, places=2)


class TestVisualizationHasLegendAndStats(unittest.TestCase):
    """Match visualization must include inlier/outlier distinction and stats overlay."""

    def test_visualization_is_color_image(self):
        """draw_matches must return a 3-channel color image."""
        src = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        ref = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        pts_s = np.array([[20, 20], [50, 50], [80, 80], [30, 70], [70, 30]], dtype=np.float32)
        pts_r = np.array([[25, 22], [52, 48], [78, 82], [35, 68], [73, 33]], dtype=np.float32)
        mask = np.array([True, True, True, False, False])

        vis = draw_matches(src, ref, pts_s, pts_r, mask, registration_status="PASS")
        self.assertEqual(vis.ndim, 3)
        self.assertEqual(vis.shape[2], 3)
        self.assertEqual(vis.shape[1], 200)  # Two 100px images side by side

    def test_visualization_wider_than_single_image(self):
        """Result should be wider than either input (side-by-side layout)."""
        src = np.random.randint(50, 200, (80, 120), dtype=np.uint8)
        ref = np.random.randint(50, 200, (80, 100), dtype=np.uint8)
        pts_s = np.array([[20, 20]], dtype=np.float32)
        pts_r = np.array([[25, 22]], dtype=np.float32)
        mask = np.array([True])

        vis = draw_matches(src, ref, pts_s, pts_r, mask)
        self.assertEqual(vis.shape[1], 220)  # 120 + 100


class TestQualityGateStates(unittest.TestCase):
    """Quality gate must produce correct 4-state classification."""

    def test_quality_gate_present_in_output(self):
        """Registration result must contain quality gate with all required fields."""
        base = cv2.cvtColor(_create_crater_scene(240, 240, seed=42), cv2.COLOR_GRAY2BGR)
        M = np.float32([[1, 0, 8], [0, 1, -5]])
        shifted = cv2.warpAffine(base, M, (240, 240), borderMode=cv2.BORDER_REFLECT101)

        result = register_pair(base, shifted, include_registered=False)
        inv = result.inlier_investigation
        self.assertIn("quality_gate", inv)

        gate = inv["quality_gate"]
        self.assertIn("status", gate)
        self.assertIn("gate_checks", gate)
        self.assertIn(gate["status"], ("PASS", "PASS_WITH_WARNING", "REVIEW", "FAIL"))

        # The gate_checks should include the new checks
        checks = gate["gate_checks"]
        self.assertIn("minimum_spatial_coverage", checks)
        self.assertIn("clustered_distribution", checks)

        # Spatial coverage threshold should be 0.125 (not the old 0.0625)
        self.assertEqual(checks["minimum_spatial_coverage"]["threshold"], 0.125)


if __name__ == "__main__":
    unittest.main()
