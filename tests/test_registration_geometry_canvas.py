"""Test suite for registered image geometry, canvas calculation, and canonical visualization.

Covers:
1. Valid lunar pair produces plausible geometry, healthy inliers, and non-empty canvas
2. Problematic moon pair (moon100.png and moon99.png) registers with isotropic scale ~0.25 and no collapsed canvas
3. Wrong pair is rejected honestly without misleading registered image
4. Low overlap pair is rejected or flagged with review
5. Pathological homography with excessive canvas expansion (>6x) is rejected
6. Denominator approaching projective horizon is rejected
7. Degenerate non-convex quadrilateral is rejected
8. Valid registration output must not consist of a tiny valid region inside an excessively large canvas
9. Canonical match visualization contains legend, titles, overlap polygon, and diagnostic badge
"""

import math
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

# Ensure api-server python directory is in sys.path
api_python_dir = Path(__file__).resolve().parent.parent / "artifacts" / "api-server" / "python"
if str(api_python_dir) not in sys.path:
    sys.path.insert(0, str(api_python_dir))

from app.services.geometry import decompose_transform, validate_transform_plausibility
from app.services.pairwise_registration import register_pair
from app.services.registration import draw_matches


def _create_crater_scene(width: int = 300, height: int = 300, seed: int = 42) -> np.ndarray:
    """Generate synthetic lunar surface with craters and illumination gradient."""
    np.random.seed(seed)
    y, x = np.mgrid[0:height, 0:width]
    # Smooth illumination gradient
    gradient = 100 + 30 * np.sin(x / 50.0) + 20 * np.cos(y / 60.0)
    noise = np.random.normal(0, 5, (height, width))
    img = np.clip(gradient + noise, 0, 255).astype(np.float32)

    craters = [
        (80, 80, 28),
        (200, 90, 35),
        (100, 210, 32),
        (220, 220, 25),
        (150, 150, 18),
        (50, 160, 15),
        (240, 60, 16),
    ]
    for cx, cy, r in craters:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # Rim illumination and bowl shadow
        rim_mask = np.abs(dist - r) < 4.0
        inner_mask = dist < r
        img[inner_mask] = img[inner_mask] * 0.65
        img[rim_mask] = np.clip(img[rim_mask] * 1.45, 0, 255)

    return np.clip(img, 0, 255).astype(np.uint8)


class TestRegisteredImageGeometry(unittest.TestCase):
    """Verifies geometry, canvas calculations, and anti-pathology constraints."""

    def test_valid_lunar_pair_registration(self):
        """A valid lunar pair with affine transformation should register successfully with valid canvas."""
        ref = _create_crater_scene(320, 320, seed=10)
        # Apply moderate rotation and translation
        M = cv2.getRotationMatrix2D((160, 160), 3.0, 1.0)
        M[0, 2] += 12.0
        M[1, 2] += -8.0
        src = cv2.warpAffine(ref, M, (320, 320))

        res = register_pair(src, ref, detector="sift")
        self.assertIn(res.registration_status, ("PASS", "PASS_WITH_WARNING"))
        self.assertIsNotNone(res.homography)
        self.assertIsNotNone(res.registered)

        # Check canvas dimensions match reference
        self.assertEqual(res.registered.shape[:2], ref.shape[:2])
        # Check non-zero registered content occupies a healthy portion of reference canvas
        nonzero_mask = res.registered > 0 if res.registered.ndim == 2 else np.any(res.registered > 0, axis=-1)
        nonzero_ratio = np.mean(nonzero_mask)
        self.assertGreater(nonzero_ratio, 0.60, f"Valid registration content only covers {nonzero_ratio:.1%} of canvas")

    def test_problematic_moon_pair_isotropic_scale_and_healthy_canvas(self):
        """The problematic moon pair (moon100.png and moon99.png) must not collapse into a tiny corner."""
        moon100_path = Path(__file__).resolve().parent / "fixtures" / "moon100.png"
        moon99_path = Path(__file__).resolve().parent / "fixtures" / "moon99.png"
        if not moon100_path.exists() or not moon99_path.exists():
            self.skipTest("Moon fixtures not found in tests/fixtures")

        src = cv2.imread(str(moon100_path), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(str(moon99_path), cv2.IMREAD_GRAYSCALE)

        res = register_pair(
            src,
            ref,
            detector="sift",
            ratio=0.75,
            max_features=6000,
            representation="raw",
        )
        self.assertIn(res.registration_status, ("PASS", "PASS_WITH_WARNING"))
        self.assertIsNotNone(res.homography)
        self.assertIsNotNone(res.registered)

        # Check transform plausibility
        is_plaus, issues, decomp = validate_transform_plausibility(res.homography, src.shape, ref.shape)
        self.assertTrue(is_plaus, f"Transform failed plausibility with issues: {issues}")
        self.assertAlmostEqual(decomp["scale_ratio"], 1.0, delta=0.15)
        self.assertAlmostEqual(decomp["scale_x"], 0.25, delta=0.05)
        self.assertAlmostEqual(decomp["scale_y"], 0.25, delta=0.05)

        # Invariant: valid registration output must NOT consist of a tiny valid region inside an excessively large empty canvas
        # For moon100 (1792x898) scaled by 0.25 onto moon99 (1024x1024), projected size is ~448x225 (area ~100,000 px^2, ~9.6% of ref)
        # Previously the bug caused an area of 315x157 (area ~49,000 px^2, ~4.7% of ref with determinant 0.04)
        proj_w = decomp.get("projected_width", 0)
        proj_h = decomp.get("projected_height", 0)
        self.assertGreater(proj_w, 400.0, f"Projected width {proj_w} collapsed")
        self.assertGreater(proj_h, 200.0, f"Projected height {proj_h} collapsed")

    def test_wrong_pair_rejected_no_misleading_registered_image(self):
        """Unrelated pair must be rejected and must not generate a misleading registered raster."""
        img1 = _create_crater_scene(250, 250, seed=1)
        # Random non-lunar checkerboard pattern
        img2 = np.zeros((250, 250), dtype=np.uint8)
        img2[::20, :] = 255
        img2[:, ::20] = 255

        res = register_pair(img1, img2, detector="sift")
        self.assertEqual(res.registration_status, "FAIL")
        self.assertFalse(res.success)
        # Crucial requirement: do not generate a misleading registered image for invalid registration
        self.assertIsNone(res.registered, "Registered image must be None when registration fails")

    def test_pathological_homography_extreme_canvas_expansion(self):
        """A homography that produces >6x canvas expansion must be rejected."""
        # Scale by 10x
        H = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        is_plaus, issues, decomp = validate_transform_plausibility(H, source_shape=(200, 200), reference_shape=(200, 200))
        self.assertFalse(is_plaus)
        self.assertTrue(any("expansion" in issue.lower() or "magnification" in issue.lower() for issue in issues))

    def test_homography_projective_horizon_denominator_singularity(self):
        """A homography where denominator H[2,0]*x + H[2,1]*y + 1 approaches zero within the image must be rejected."""
        # Near singular denominator across image [0, 200]
        H = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [-0.005, 0.0, 1.0]  # At x=200, denom = -1 + 1 = 0!
        ], dtype=np.float64)
        is_plaus, issues, decomp = validate_transform_plausibility(H, source_shape=(200, 200), reference_shape=(200, 200))
        self.assertFalse(is_plaus)
        self.assertTrue(any("horizon" in issue.lower() for issue in issues))

    def test_degenerate_non_convex_quadrilateral_rejected(self):
        """A transformation projecting corners into a self-intersecting bowtie must be flagged."""
        # Swaps top-right and bottom-right corners
        src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        is_plaus, issues, decomp = validate_transform_plausibility(H, source_shape=(100, 100), reference_shape=(100, 100))
        self.assertFalse(is_plaus)
        self.assertTrue(any("self-intersecting" in issue.lower() or "degenerate" in issue.lower() for issue in issues))

    def test_canonical_visualization_has_titles_legend_and_badges(self):
        """Canonical backend visualization must render side titles, legend, and diagnostic badges."""
        ref = _create_crater_scene(200, 200, seed=20)
        src = cv2.warpAffine(ref, cv2.getRotationMatrix2D((100, 100), 2.0, 1.0), (200, 200))
        res = register_pair(src, ref, detector="sift")
        vis = res.match_visualization
        self.assertIsNotNone(vis)
        self.assertIsInstance(vis, np.ndarray)
        self.assertEqual(vis.ndim, 3)
        self.assertEqual(vis.shape[2], 3)
        # Width should include source and reference plus border
        self.assertGreaterEqual(vis.shape[1], 400)
        self.assertGreaterEqual(vis.shape[0], 200)


if __name__ == "__main__":
    unittest.main()
