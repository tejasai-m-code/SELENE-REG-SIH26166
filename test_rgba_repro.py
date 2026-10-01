import cv2
import numpy as np
import httpx
import os

# Create an RGBA image
img = np.zeros((100, 100, 4), dtype=np.uint8)
img[:, :, :3] = 255
img[:, :, 3] = 128
cv2.imwrite("test_rgba.png", img)

# Test decoding it using the utils directly
from backend.app.utils.image_utils import decode_upload
with open("test_rgba.png", "rb") as f:
    data = f.read()

try:
    raster = decode_upload(data, "test_rgba.png")
    print(f"Decoded raster shape: {raster.data.shape}")
except Exception as e:
    print(f"decode_upload error: {e}")

# What if it's not a PNG but something else? Or what if decode_upload fails elsewhere?
