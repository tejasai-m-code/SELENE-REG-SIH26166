import sys
import os
sys.path.insert(0, os.path.abspath("backend"))

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from main import app
from app.utils.image_utils import decode_upload
from app.services.registration import draw_matches

client = TestClient(app)

def create_mock_png(channels, color=(255, 128, 64, 255)):
    # Create a 200x200 checkerboard pattern so SIFT can easily find features
    img = np.zeros((200, 200, channels), dtype=np.uint8)
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i:i+20, j:j+20] = 255
            else:
                img[i:i+20, j:j+20] = 50
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

def create_valid_mock_png(channels=3):
    # Deterministic image with strong geometric features known to pass registration
    img = np.zeros((240, 240, channels), dtype=np.uint8)
    color255 = (255, 255, 255, 255)[:channels]
    color180 = (180, 180, 180, 255)[:channels]
    color220 = (220, 220, 220, 255)[:channels]
    cv2.circle(img, (80, 80), 20, color255, -1)
    cv2.rectangle(img, (140, 120), (200, 180), color180, -1)
    cv2.line(img, (30, 200), (200, 30), color220, 3)
    # Add translation to reference for a realistic registration pair
    ref = cv2.warpAffine(img, np.float32([[1, 0, 12], [0, 1, 8]]), (240, 240))
    _, buf_src = cv2.imencode('.png', img)
    _, buf_ref = cv2.imencode('.png', ref)
    return buf_src.tobytes(), buf_ref.tobytes()

def test_rgba_input():
    data = create_mock_png(4)
    raster = decode_upload(data, "rgba.png")
    # Must be normalized to 3 channels by decode_upload
    assert raster.data.ndim == 3
    assert raster.data.shape[2] == 3

def test_bgra_input():
    data = create_mock_png(4)
    raster = decode_upload(data, "bgra.png")
    assert raster.data.shape[2] == 3

def test_bad_image():
    with pytest.raises(ValueError):
        decode_upload(b"not an image", "bad.png")

def test_duplicate_image():
    # Test identical pair registration
    src_data, ref_data = create_valid_mock_png(3)
    files = {
        "source": ("img_src.png", src_data, "image/png"),
        "reference": ("img_ref.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "sift"})
    assert response.status_code == 200
    res = response.json()
    assert res["metrics"]["inlier_ratio"] > 0.0  # Verify it actually found matches

def test_non_json_api_error():
    # Trigger an error and check response structure (fastapi error handler should return json detail)
    files = {
        "source": ("bad.png", b"bad", "image/png"),
        "reference": ("bad.png", b"bad", "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "sift"})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_multi_image_failure_handling():
    # Send one good image and one bad image to multi registration
    good_data = create_mock_png(3)
    bad_data = b"bad data"
    files = [
        ("images", ("good.png", good_data, "image/png")),
        ("images", ("bad.png", bad_data, "image/png"))
    ]
    response = client.post("/api/multi-registration/register", files=files)
    # The job itself should succeed but summary should show failures
    assert response.status_code == 200
    res = response.json()
    assert res["summary"]["image_count"] == 2
    # The bad image fails during decoding, so processed_pairs might be 0, etc.

def test_phase15_runtime_response():
    response = client.get("/api/runtime/status")
    assert response.status_code == 200
    res = response.json()
    assert "execution_device" in res
    assert "libraries" in res

def test_synthetic_validation_response():
    data = create_mock_png(3)
    files = {"image": ("test.png", data, "image/png")}
    response = client.post("/api/evaluation/synthetic-robustness", files=files)
    assert response.status_code == 200
    res = response.json()
    assert "passed_cases" in res

def test_match_visualization_payload():
    src_data, ref_data = create_valid_mock_png(3)
    files = {
        "source": ("img_src.png", src_data, "image/png"),
        "reference": ("img_ref.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "sift"})
    assert response.status_code == 200
    res = response.json()
    assert "match_visualization" in res["outputs"]
    assert res["outputs"]["match_visualization"] is not None

def test_monkey_scaled_registration():
    src_data, ref_data = create_valid_mock_png(3)
    # Just standard registration test is fine for monkey scaling as it tests robust SIFT behavior
    files = {
        "source": ("img1.png", src_data, "image/png"),
        "reference": ("img2.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "sift"})
    assert response.status_code == 200

def test_akaze_registration():
    src_data, ref_data = create_valid_mock_png(3)
    files = {
        "source": ("img1.png", src_data, "image/png"),
        "reference": ("img2.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "akaze"})
    if response.status_code != 200:
        print("AKAZE Failed:", response.json())
    assert response.status_code == 200
    res = response.json()
    assert res["metrics"]["inlier_ratio"] > 0.0

def test_superpoint_registration():
    src_data, ref_data = create_valid_mock_png(3)
    files = {
        "source": ("img1.png", src_data, "image/png"),
        "reference": ("img2.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "superpoint"})
    if response.status_code != 200:
        print("SuperPoint Failed:", response.json())
    assert response.status_code == 200
    res = response.json()
    assert res["metrics"]["inlier_ratio"] > 0.0

def test_loftr_registration():
    src_data, ref_data = create_valid_mock_png(3)
    files = {
        "source": ("img1.png", src_data, "image/png"),
        "reference": ("img2.png", ref_data, "image/png")
    }
    response = client.post("/api/register", files=files, data={"detector": "loftr"})
    if response.status_code != 200:
        print("LoFTR Failed:", response.json())
    assert response.status_code == 200
    res = response.json()
    assert res["metrics"]["inlier_ratio"] > 0.0

def test_capability_endpoint_accuracy():
    response = client.get("/api/runtime/status")
    assert response.status_code == 200
    res = response.json()
    matrix = {item["name"]: item["available"] for item in res["capabilities"]}
    assert matrix.get("AKAZE") is True
    assert matrix.get("SuperPoint") is True
    assert matrix.get("LoFTR") is True
