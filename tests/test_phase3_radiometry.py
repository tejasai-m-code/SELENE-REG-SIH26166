import cv2
import numpy as np
from app.utils.image_utils import decode_upload
from app.services.radiometry import (
    process_raster, 
    RadiometricConfig, 
    compute_diagnostics,
    compare_illumination
)

def test_phase3():
    print("Testing Phase 3 Radiometric Processing...")
    
    # Create simple float array
    raw = np.array([[0, 1], [10, 100]], dtype=np.float32)
    _, buf = cv2.imencode('.tiff', raw)
    raster = decode_upload(buf.tobytes(), "test.tiff")
    
    # A. Known scale/offset (Synthetic test metadata)
    raster.metadata.radiometric.scale_factor = 2.0
    raster.metadata.radiometric.offset = 10.0
    
    res1 = process_raster(raster, RadiometricConfig(enable_calibration=True, enable_illumination_normalization=False))
    
    assert res1.provenance.calibration_applied == True
    expected = np.array([[10, 12], [30, 210]], dtype=np.float32)
    assert np.allclose(res1.scientific_data, expected), f"Calibration failed. Got {res1.scientific_data}"
    
    # B. Missing scale/offset
    raster.metadata.radiometric.scale_factor = None
    raster.metadata.radiometric.offset = None
    res2 = process_raster(raster, RadiometricConfig(enable_calibration=True))
    assert res2.provenance.calibration_applied == False
    assert np.allclose(res2.scientific_data, raw), "Raw data mutated when missing calibration"
    
    # C/D. Integer preservation
    raw_int = np.array([[0, 255]], dtype=np.uint8)
    _, buf_int = cv2.imencode('.png', raw_int)
    raster_int = decode_upload(buf_int.tobytes(), "test.png")
    res3 = process_raster(raster_int, RadiometricConfig(enable_calibration=False))
    assert res3.scientific_data.dtype == np.uint8
    assert res3.scientific_data.shape == raw_int.shape
    
    # F. Constant image illumination
    raster_const = decode_upload(cv2.imencode('.png', np.ones((10,10), dtype=np.uint8))[1].tobytes(), "test.png")
    res4 = process_raster(raster_const, RadiometricConfig(enable_calibration=False, enable_illumination_normalization=True))
    assert np.isfinite(res4.scientific_data).all(), "NaNs in constant image normalization"
    
    # H/I. Diagnostics
    d1 = compute_diagnostics(raw)
    d2 = compute_diagnostics(raw * 2)
    diff = compare_illumination(d1, d2)
    assert diff.mean_diff != 0
    
    print("All Phase 3 tests passed.")

if __name__ == "__main__":
    test_phase3()
