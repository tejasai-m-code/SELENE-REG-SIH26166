import asyncio
import json
from app.services.cross_modal_matching import execute_cross_modal_matching
from app.services.radiometry import RadiometricConfig
from app.utils.image_utils import ScientificRaster
from app.services.metadata import ProductMetadata
from app.routes.registration import register_images
import numpy as np
import cv2
from fastapi import UploadFile
import io

class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
    async def read(self):
        return self.content

async def run_test():
    print("=== PHASE 8 API & SERIALIZATION TESTS ===")
    np.random.seed(42)
    src_img = (np.random.rand(400, 400) * 255).astype(np.uint8)
    src_img = cv2.GaussianBlur(src_img, (5, 5), 2.0)
    
    # Add some strong features
    for i in range(20):
        x, y = np.random.randint(20, 380, 2)
        cv2.circle(src_img, (x, y), 5, 255, -1)
    
    success, encoded = cv2.imencode('.png', src_img)
    src_bytes = encoded.tobytes()
    
    ref_img = src_img.copy()
    M = np.float32([[1, 0, 5.0], [0, 1, -2.5]])
    ref_img = cv2.warpAffine(ref_img, M, (400, 400))
    success, encoded = cv2.imencode('.png', ref_img)
    ref_bytes = encoded.tobytes()

    src_file = MockUploadFile("src.png", src_bytes)
    ref_file = MockUploadFile("ref.png", ref_bytes)

    res = await register_images(
        source=src_file, reference=ref_file, detector="sift", ratio=0.9,
        ransac_threshold=3.0, illumination_normalization=False,
        spatial_distribution=False, ecc_refinement=True, max_features=1000,
        geometric_model="translation"
    )
    
    pts = res.get('inlier_points', [])
    print(pts[0])

if __name__ == "__main__":
    asyncio.run(run_test())
