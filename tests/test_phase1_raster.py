import cv2
import numpy as np
import tempfile
from pathlib import Path
from app.utils.image_utils import decode_upload

def test_phase1_raster():
    print("Testing Phase 1 Scientific Raster Preservation...")
    
    # 1. uint8 input preserves uint8
    img_uint8 = np.zeros((100, 100), dtype=np.uint8)
    img_uint8[50:60, 50:60] = 255
    ret, buf = cv2.imencode('.png', img_uint8)
    raster = decode_upload(buf.tobytes(), "test_uint8.png")
    assert raster.data.dtype == np.uint8, f"Expected uint8, got {raster.data.dtype}"
    print("uint8 preserved.")
    
    # 2. uint16 input preserves uint16
    img_uint16 = np.zeros((100, 100), dtype=np.uint16)
    img_uint16[50:60, 50:60] = 65000
    ret, buf = cv2.imencode('.png', img_uint16)
    raster16 = decode_upload(buf.tobytes(), "test_uint16.png")
    assert raster16.data.dtype == np.uint16, f"Expected uint16, got {raster16.data.dtype}"
    assert raster16.display_image.dtype == np.uint8, "Display image must be uint8"
    assert raster16.processing_image.dtype == np.uint8, "Processing image must be uint8"
    assert raster16.metadata.raster.dtype == "uint16", "Metadata must record uint16"
    print("uint16 preserved.")
    
    # 3. float32 input preserves float32
    # cv2.imencode doesn't support float32 to png, so we can save to tiff and read
    img_float32 = np.zeros((100, 100), dtype=np.float32)
    img_float32[50:60, 50:60] = 1.0
    
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tf:
        tf_path = tf.name
    cv2.imwrite(tf_path, img_float32)
    
    with open(tf_path, "rb") as f:
        data32 = f.read()
    
    raster32 = decode_upload(data32, "test_float32.tiff")
    Path(tf_path).unlink()
    
    assert raster32.data.dtype == np.float32, f"Expected float32, got {raster32.data.dtype}"
    assert raster32.metadata.raster.dtype == "float32", "Metadata must record float32"
    print("float32 preserved.")
    
    print("All Phase 1 tests passed.")

if __name__ == "__main__":
    test_phase1_raster()
