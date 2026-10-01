import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from app.utils.image_utils import ScientificRaster, normalize_to_uint8

@dataclass
class IlluminationDiagnostics:
    mean_intensity: float
    median_intensity: float
    p01: float
    p99: float
    robust_contrast: float
    gradient_mean: float
    shadow_fraction: float

@dataclass
class PairwiseIlluminationDiagnostics:
    mean_diff: float
    contrast_diff: float
    gradient_mean_diff: float

@dataclass
class RadiometricConfig:
    enable_calibration: bool = True
    enable_illumination_normalization: bool = False
    illumination_method: str = "local_contrast"  # or "high_pass"
    blur_kernel_size: int = 51

@dataclass
class RadiometricProvenance:
    input_dtype: str
    calibration_applied: bool
    calibration_source: str
    scale_factor: Optional[float]
    offset: Optional[float]
    input_units: str
    output_units: str
    illumination_normalization_applied: bool
    illumination_method: str
    warnings: List[str] = field(default_factory=list)

@dataclass
class RadiometricResult:
    scientific_data: np.ndarray
    processing_image: np.ndarray
    display_image: np.ndarray
    provenance: RadiometricProvenance
    diagnostics: IlluminationDiagnostics


def compute_diagnostics(image: np.ndarray) -> IlluminationDiagnostics:
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
        
    gray_f = gray.astype(np.float32)
    valid = gray_f[np.isfinite(gray_f)]
    if valid.size == 0:
        return IlluminationDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    mean_val = float(np.mean(valid))
    median_val = float(np.median(valid))
    p01, p99 = np.percentile(valid, [1, 99])
    contrast = float(p99 - p01)
    
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    valid_grad = grad_mag[np.isfinite(grad_mag)]
    grad_mean = float(np.mean(valid_grad)) if valid_grad.size > 0 else 0.0
    
    shadow_thresh = median_val * 0.25
    shadow_frac = float(np.sum(valid < shadow_thresh) / valid.size)
    
    return IlluminationDiagnostics(
        mean_intensity=mean_val,
        median_intensity=median_val,
        p01=float(p01),
        p99=float(p99),
        robust_contrast=contrast,
        gradient_mean=grad_mean,
        shadow_fraction=shadow_frac
    )

def compare_illumination(d1: IlluminationDiagnostics, d2: IlluminationDiagnostics) -> PairwiseIlluminationDiagnostics:
    return PairwiseIlluminationDiagnostics(
        mean_diff=d1.mean_intensity - d2.mean_intensity,
        contrast_diff=d1.robust_contrast - d2.robust_contrast,
        gradient_mean_diff=d1.gradient_mean - d2.gradient_mean
    )

def apply_radiometric_calibration(raster: ScientificRaster) -> Tuple[np.ndarray, bool, Optional[float], Optional[float], str, List[str]]:
    warnings = []
    scale = raster.metadata.radiometric.scale_factor
    offset = raster.metadata.radiometric.offset
    
    if scale is not None and offset is not None:
        data_f = raster.data.astype(np.float64)  # Use float64 to prevent precision loss during scaling
        calibrated = (data_f * scale) + offset
        return calibrated.astype(np.float32), True, scale, offset, "metadata", warnings
    else:
        warnings.append("Radiometric scale/offset unavailable. Calibration NOT APPLIED.")
        return raster.data.copy(), False, None, None, "none", warnings

def apply_illumination_normalization(image: np.ndarray, method: str, ksize: int) -> Tuple[np.ndarray, List[str]]:
    warnings = []
    img_f = image.astype(np.float32)
    
    if not np.isfinite(img_f).any():
        warnings.append("Image contains no finite values. Normalization skipped.")
        return img_f, warnings
        
    k = ksize if ksize % 2 == 1 else ksize + 1
    
    if method == "local_contrast":
        blur = cv2.GaussianBlur(img_f, (k, k), 0)
        norm = img_f / (blur + 1e-5)
        return norm, warnings
    elif method == "high_pass":
        blur = cv2.GaussianBlur(img_f, (k, k), 0)
        norm = img_f - blur
        return norm, warnings
    else:
        warnings.append(f"Unknown illumination method: {method}. Skipped.")
        return img_f, warnings

def process_raster(raster: ScientificRaster, config: RadiometricConfig) -> RadiometricResult:
    # 1. Calibration
    if config.enable_calibration:
        calibrated_data, calib_applied, scale, offset, calib_src, calib_warns = apply_radiometric_calibration(raster)
    else:
        calibrated_data = raster.data.copy()
        calib_applied, scale, offset, calib_src, calib_warns = False, None, None, "disabled", []
        
    # 2. Illumination Normalization
    illum_applied = False
    illum_data = calibrated_data
    illum_warns = []
    if config.enable_illumination_normalization:
        illum_data, illum_warns = apply_illumination_normalization(calibrated_data, config.illumination_method, config.blur_kernel_size)
        illum_applied = True
        
    # 3. Processing and Display Representations
    proc_img = normalize_to_uint8(illum_data)
    disp_img = normalize_to_uint8(illum_data)
    
    # 4. Diagnostics on the purely calibrated (or raw) scientific data
    diagnostics = compute_diagnostics(calibrated_data)
    
    # 5. Provenance
    input_units = raster.metadata.radiometric.units or "UNKNOWN"
    output_units = input_units if not calib_applied else (raster.metadata.radiometric.units or "UNKNOWN")
    
    prov = RadiometricProvenance(
        input_dtype=str(raster.data.dtype),
        calibration_applied=calib_applied,
        calibration_source=calib_src,
        scale_factor=scale,
        offset=offset,
        input_units=input_units,
        output_units=output_units,
        illumination_normalization_applied=illum_applied,
        illumination_method=config.illumination_method if illum_applied else "none",
        warnings=calib_warns + illum_warns
    )
    
    return RadiometricResult(
        scientific_data=calibrated_data,
        processing_image=proc_img,
        display_image=disp_img,
        provenance=prov,
        diagnostics=diagnostics
    )
