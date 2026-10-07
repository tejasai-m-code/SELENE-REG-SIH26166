import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

from app.services.geometry import estimate_geometric_model, validate_overlap
from app.services.multiscale import match_multiscale
from app.services.pairwise_registration import register_pair

class TestPhase2Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic highly textured source image
        np.random.seed(42)
        # Random noise blurred creates some texture
        img = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
        cls.src_image = cv2.GaussianBlur(img, (9, 9), 2.0)
        # Add some distinct features
        for i in range(10):
            cv2.circle(cls.src_image, (np.random.randint(50, 450), np.random.randint(50, 450)), np.random.randint(5, 20), (255, 255, 255), -1)
            cv2.rectangle(cls.src_image, (np.random.randint(50, 450), np.random.randint(50, 450)), (np.random.randint(50, 450), np.random.randint(50, 450)), (0, 0, 0), -1)
            
    def _apply_transform_and_test(self, M, is_homography=False, noise=False, blur=False, brightness=0, partial=False):
        if is_homography:
            ref_image = cv2.warpPerspective(self.src_image, M, (500, 500), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            ref_image = cv2.warpAffine(self.src_image, M, (500, 500), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
        if blur:
            ref_image = cv2.GaussianBlur(ref_image, (5, 5), 1.0)
        if noise:
            noise_img = np.random.normal(0, 10, ref_image.shape).astype(np.float32)
            ref_image = np.clip(ref_image.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)
        if brightness != 0:
            ref_image = np.clip(ref_image.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
            
        if partial:
            # Crop a region
            ref_image[0:200, :] = 0
            
        result = register_pair(
            self.src_image, ref_image, detector="sift", max_features=1000
        )
        return result

    def test_A_translation(self):
        M = np.float32([[1, 0, 50], [0, 1, -30]])
        res = self._apply_transform_and_test(M)
        self.assertTrue(res.success)
        
    def test_B_rotation(self):
        M = cv2.getRotationMatrix2D((250, 250), 15, 1.0)
        res = self._apply_transform_and_test(M)
        self.assertTrue(res.success)
        
    def test_C_scale(self):
        M = cv2.getRotationMatrix2D((250, 250), 0, 0.8)
        res = self._apply_transform_and_test(M)
        self.assertTrue(res.success)
        
    def test_D_rotation_and_scale(self):
        M = cv2.getRotationMatrix2D((250, 250), -20, 1.2)
        res = self._apply_transform_and_test(M)
        self.assertTrue(res.success)
        
    def test_E_affine_distortion(self):
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[40,60],[220,40],[60,210]])
        M = cv2.getAffineTransform(pts1, pts2)
        res = self._apply_transform_and_test(M)
        self.assertTrue(res.success)
        
    def test_F_perspective_distortion(self):
        pts1 = np.float32([[50,50],[450,50],[50,450],[450,450]])
        pts2 = np.float32([[60,60],[430,40],[40,430],[450,450]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        res = self._apply_transform_and_test(M, is_homography=True)
        self.assertTrue(res.success)
        
    def test_G_noise(self):
        M = np.float32([[1, 0, 20], [0, 1, 20]])
        res = self._apply_transform_and_test(M, noise=True)
        self.assertTrue(res.success)
        
    def test_H_blur(self):
        M = np.float32([[1, 0, -20], [0, 1, -20]])
        res = self._apply_transform_and_test(M, blur=True)
        self.assertTrue(res.success)
        
    def test_I_brightness(self):
        M = np.float32([[1, 0, 10], [0, 1, 10]])
        res = self._apply_transform_and_test(M, brightness=50)
        self.assertTrue(res.success)
        
    def test_J_partial_overlap(self):
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        res = self._apply_transform_and_test(M, partial=True)
        self.assertTrue(res.success)
        
    def test_blank_image(self):
        blank = np.zeros_like(self.src_image)
        res = register_pair(self.src_image, blank)
        self.assertFalse(res.success)
        self.assertEqual(res.registration_status, "FAIL")
        
    def test_low_texture(self):
        low = np.full_like(self.src_image, 128)
        cv2.circle(low, (250, 250), 2, (255, 255, 255), -1) # One feature
        res = register_pair(self.src_image, low)
        self.assertFalse(res.success)
        
    def test_gsd_handling(self):
        # We simulate GSD scaling.
        # This will be handled if we pass source_metadata and reference_metadata
        res = register_pair(
            self.src_image, self.src_image,
            source_metadata={"gsd_m": 0.5},
            reference_metadata={"gsd_m": 1.0}
        )
        self.assertEqual(res.gsd_handling["status"], "RESCALED")
        self.assertEqual(res.gsd_handling["ratio"], 0.5)

    def test_magsac_model_selection(self):
        pts_s = np.array([[0,0],[100,0],[100,100],[0,100], [50,50]], dtype=float)
        pts_r = pts_s + np.array([10, 10])
        
        geom = estimate_geometric_model(pts_s, pts_r, (500, 500), (500, 500))
        self.assertTrue(geom.is_valid)
        
    def test_overlap_validation(self):
        # Good overlap
        M = np.eye(3)
        valid, ratio, reason = validate_overlap((100, 100), (100, 100), M)
        self.assertTrue(valid)
        self.assertEqual(ratio, 1.0)
        
        # Zero overlap
        M[0, 2] = 200 # Translate completely out
        valid, ratio, reason = validate_overlap((100, 100), (100, 100), M)
        self.assertFalse(valid)
        self.assertEqual(ratio, 0.0)

if __name__ == '__main__':
    unittest.main()
