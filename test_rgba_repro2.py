import cv2
import numpy as np

# Create an RGBA image
img = np.zeros((100, 100, 4), dtype=np.uint8)
img[:, :, :3] = 255
img[:, :, 3] = 128
cv2.imwrite("test_rgba.png", img)

# Test decoding it using the utils directly
from backend.app.utils.image_utils import decode_upload
from backend.app.services.radiometry import process_raster, RadiometricConfig

with open("test_rgba.png", "rb") as f:
    data = f.read()

try:
    raster = decode_upload(data, "test_rgba.png")
    print(f"Decoded raster shape: {raster.data.shape}")
    raster_res = process_raster(raster, RadiometricConfig())
    print(f"Processed raster shape: {raster_res.processing_image.shape}")
except Exception as e:
    print(f"Error: {e}")
