import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.utils.image_utils import ScientificRaster
from app.services.radiometry import process_raster, RadiometricConfig

@dataclass
class RepresentationProvenance:
    source_name: str
    preprocessing_method: str
    warnings: List[str] = field(default_factory=list)

@dataclass
class ImageRepresentation:
    name: str
    dtype: str
    shape: tuple
    source: str
    preprocessing_method: str
    validity_status: str
    warnings: List[str]
    provenance: RepresentationProvenance
    image: np.ndarray

def generate_high_pass(image_uint8: np.ndarray, ksize: int = 15) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_uint8, (ksize, ksize), 0)
    # Using cv2.subtract with a shift or just direct subtraction via int16
    high_pass_int = cv2.addWeighted(image_uint8, 1.5, blurred, -0.5, 0)
    return high_pass_int

def generate_gradient_magnitude(image_uint8: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(image_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_uint8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    # Normalize to 0-255
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def build_representations(raster: ScientificRaster, config: RadiometricConfig) -> Dict[str, ImageRepresentation]:
    reps = {}
    
    # 1. Scientific Data (Do not destroy)
    sci_data = raster.data
    reps["raw_scientific"] = ImageRepresentation(
        name="raw_scientific",
        dtype=str(sci_data.dtype),
        shape=sci_data.shape,
        source="raster.data",
        preprocessing_method="None",
        validity_status="VALID",
        warnings=[],
        provenance=RepresentationProvenance("ScientificRaster", "None"),
        image=sci_data
    )
    
    # 2. Radiometric/Illumination Normalized
    res = process_raster(raster, config)
    proc_img = res.processing_image
    
    reps["radiometric_normalized"] = ImageRepresentation(
        name="radiometric_normalized",
        dtype=str(proc_img.dtype),
        shape=proc_img.shape,
        source="process_raster",
        preprocessing_method="radiometric_calibration + illumination_normalization" if config.enable_illumination_normalization else "radiometric_calibration",
        validity_status="VALID",
        warnings=[],
        provenance=RepresentationProvenance("process_raster", "Phase 3 Pipeline"),
        image=proc_img
    )
    
    # 3. High Pass / Structural
    hp_img = generate_high_pass(proc_img)
    reps["high_pass"] = ImageRepresentation(
        name="high_pass",
        dtype=str(hp_img.dtype),
        shape=hp_img.shape,
        source="radiometric_normalized",
        preprocessing_method="gaussian_blur_subtraction",
        validity_status="VALID",
        warnings=[],
        provenance=RepresentationProvenance("high_pass_filter", "OpenCV cv2.addWeighted"),
        image=hp_img
    )
    
    # 4. Gradient Magnitude
    grad_img = generate_gradient_magnitude(proc_img)
    reps["gradient_magnitude"] = ImageRepresentation(
        name="gradient_magnitude",
        dtype=str(grad_img.dtype),
        shape=grad_img.shape,
        source="radiometric_normalized",
        preprocessing_method="sobel_magnitude_normalized",
        validity_status="VALID",
        warnings=[],
        provenance=RepresentationProvenance("gradient_filter", "OpenCV Sobel"),
        image=grad_img
    )
    
    return reps
