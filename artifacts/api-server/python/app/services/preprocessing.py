import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any

from app.utils.image_utils import to_gray
from app.services.gpu_accelerator import (
    gpu_gaussian_blur,
    gpu_sobel_gradient,
    gpu_multiscale_retinex,
    gpu_highpass_filter,
    is_cuda_available,
)


@dataclass
class PreprocessResult:
    image: np.ndarray
    method: str
    radiometric: dict
    stages: dict
    illumination_stats: dict | None = None
    hardware_acceleration: dict | None = None


def percentile_normalize(gray: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Robust percentile stretching into 8-bit dynamic range [0, 255]."""
    gray = gray.astype(np.float32)
    valid = np.isfinite(gray)
    if not valid.any():
        return np.zeros_like(gray, dtype=np.uint8)

    values = gray[valid]
    lo, hi = float(np.percentile(values, p_low)), float(np.percentile(values, p_high))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)

    out = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _clahe(gray: np.ndarray, clip_limit: float = 2.5, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)


def _highpass(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """High-pass filter suppressing broad illumination field with GPU acceleration."""
    hp_img, dev_info = gpu_highpass_filter(gray, sigma=9.0)
    return np.clip(hp_img, 0, 255).astype(np.uint8), dev_info


def _gradient(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Gradient magnitude representation invariant to monotonic illumination bias."""
    mag, dev_info = gpu_sobel_gradient(gray)
    return percentile_normalize(mag), dev_info


def _retinex(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-Scale Retinex (MSR) estimating reflectance under variable solar incidence."""
    msr, dev_info = gpu_multiscale_retinex(gray, scales=(15.0, 45.0, 120.0))
    return percentile_normalize(msr), dev_info


def _shadow_mask(gray: np.ndarray) -> np.ndarray:
    """Extract deep shadow regions using adaptive thresholding and morphological opening."""
    threshold = float(np.percentile(gray, 18))
    mask = (gray <= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _background_illumination_field(gray: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    """Estimate low-frequency macro-topography solar illumination field."""
    bg, _ = gpu_gaussian_blur(gray, sigma=sigma)
    return bg


def compute_illumination_statistics(gray: np.ndarray) -> dict[str, Any]:
    """Compute measurable radiometric and illumination metrics across lunar surface."""
    f = gray.astype(np.float32)
    valid = np.isfinite(f)
    if not valid.any():
        return {
            "mean_brightness": 0.0,
            "median_brightness": 0.0,
            "dynamic_contrast_ratio": 0.0,
            "shadow_fraction_pct": 0.0,
            "illumination_gradient_magnitude": 0.0,
            "entropy": 0.0,
            "estimated_sun_direction_deg": None,
        }

    valid_vals = f[valid]
    mean_val = float(np.mean(valid_vals))
    median_val = float(np.median(valid_vals))
    p5 = float(np.percentile(valid_vals, 5))
    p95 = float(np.percentile(valid_vals, 95))
    contrast_ratio = float((p95 - p5) / (p95 + p5 + 1e-5))

    # Shadow fraction (deep shadowed pixels below 15th percentile or near detector noise floor)
    shadow_thresh = min(p5 + 5.0, 30.0)
    shadow_pixels = np.sum(valid_vals <= shadow_thresh)
    shadow_fraction_pct = float(round((shadow_pixels / len(valid_vals)) * 100.0, 2))

    # Low-frequency illumination background gradient
    bg = _background_illumination_field(gray, sigma=35.0)
    gx = cv2.Sobel(bg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bg, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mean_gradient = float(round(float(np.mean(mag)), 4))

    # Shannon entropy
    counts, _ = np.histogram(valid_vals, bins=32)
    probs = counts / (np.sum(counts) + 1e-8)
    probs = probs[probs > 0]
    entropy = float(round(-np.sum(probs * np.log2(probs)), 3))

    # Sun direction proxy from shadow gradient orientation
    angles = np.arctan2(gy, gx) * 180.0 / np.pi
    weights = mag.flatten()
    if np.sum(weights) > 1e-4:
        # Circular mean weighted by gradient strength
        sin_mean = np.sum(weights * np.sin(angles.flatten() * np.pi / 180.0))
        cos_mean = np.sum(weights * np.cos(angles.flatten() * np.pi / 180.0))
        sun_azimuth = float(round((np.arctan2(sin_mean, cos_mean) * 180.0 / np.pi) % 360.0, 1))
    else:
        sun_azimuth = None

    return {
        "mean_brightness": round(mean_val, 2),
        "median_brightness": round(median_val, 2),
        "dynamic_contrast_ratio": round(contrast_ratio, 4),
        "shadow_fraction_pct": shadow_fraction_pct,
        "illumination_gradient_magnitude": mean_gradient,
        "entropy": entropy,
        "estimated_sun_direction_deg": sun_azimuth,
    }


def compare_illumination_pair(source_gray: np.ndarray, reference_gray: np.ndarray) -> dict[str, Any]:
    """Measure radiometric and illumination divergence between two terrain views."""
    s_stats = compute_illumination_statistics(source_gray)
    r_stats = compute_illumination_statistics(reference_gray)

    # Resize to common dimension for spatial difference metrics
    target_shape = (256, 256)
    s_res = cv2.resize(source_gray.astype(np.float32), target_shape, interpolation=cv2.INTER_AREA)
    r_res = cv2.resize(reference_gray.astype(np.float32), target_shape, interpolation=cv2.INTER_AREA)

    # Mean Absolute Difference (normalized)
    mad = float(np.mean(np.abs(s_res - r_res)))

    # Normalized Cross Correlation (NCC)
    s_zero = s_res - np.mean(s_res)
    r_zero = r_res - np.mean(r_res)
    denom = np.sqrt(np.sum(s_zero ** 2) * np.sum(r_zero ** 2))
    ncc = float(np.sum(s_zero * r_zero) / (denom + 1e-7))

    # Mutual Information proxy
    hist_2d, _, _ = np.histogram2d(s_res.ravel(), r_res.ravel(), bins=20)
    p_xy = hist_2d / float(np.sum(hist_2d))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = p_x[:, None] * p_y[None, :]
    nz = p_xy > 0
    mi = float(np.sum(p_xy[nz] * np.log2(p_xy[nz] / (p_x_p_y[nz] + 1e-12))))

    return {
        "source": s_stats,
        "reference": r_stats,
        "mean_absolute_difference": round(mad, 2),
        "normalized_cross_correlation": round(ncc, 4),
        "mutual_information": round(mi, 4),
        "illumination_gradient_delta": round(abs(s_stats["illumination_gradient_magnitude"] - r_stats["illumination_gradient_magnitude"]), 4),
        "shadow_fraction_delta": round(abs(s_stats["shadow_fraction_pct"] - r_stats["shadow_fraction_pct"]), 2),
    }


def _radiometric_gray(image: np.ndarray, metadata: dict | None, mode: str) -> tuple[np.ndarray, dict]:
    """Execute truthful radiometric preprocessing or document uncalibrated status."""
    metadata = metadata or {}
    raw = to_gray(image).astype(np.float32)

    # 1. Dark current / bias and gain calibration if metadata supplied
    dark_current = metadata.get("dark_current", metadata.get("bias"))
    gain = metadata.get("gain")
    scale = metadata.get("radiance_scale")
    offset = metadata.get("radiance_offset", 0.0)

    calibration_available = (
        mode == "metadata_calibration"
        and (
            (isinstance(scale, (int, float)) and np.isfinite(scale))
            or (isinstance(gain, (int, float)) and np.isfinite(gain))
        )
    )

    if calibration_available:
        calibrated = raw.copy()
        if dark_current is not None and np.isfinite(dark_current):
            calibrated = np.maximum(0.0, calibrated - float(dark_current))
        if gain is not None and np.isfinite(gain):
            calibrated = calibrated * float(gain)
        if scale is not None and np.isfinite(scale):
            calibrated = calibrated * float(scale) + float(offset)

        return calibrated, {
            "mode": "metadata_calibration",
            "status": "CALIBRATED_FROM_SUPPLIED_METADATA",
            "scale": float(scale) if scale is not None else None,
            "offset": float(offset) if offset is not None else 0.0,
            "dark_current": float(dark_current) if dark_current is not None else None,
            "gain": float(gain) if gain is not None else None,
        }

    # Truthful fallback: Calibration metadata absent
    return raw, {
        "mode": "safe_normalization",
        "status": "SAFE_NORMALIZATION_ONLY",
        "note": "Radiometric calibration metadata unavailable — geometric/radiometric normalization applied using image-derived statistics.",
    }


def preprocess_with_diagnostics(
    image: np.ndarray,
    illumination_normalization: bool = True,
    representation: str = "structural",
    metadata: dict | None = None,
    radiometric_mode: str = "safe_normalization",
) -> PreprocessResult:
    """Execute complete radiometric and illumination preprocessing pipeline."""
    raw_gray, radiometric = _radiometric_gray(image, metadata, radiometric_mode)
    normalized = percentile_normalize(raw_gray)
    clahe = _clahe(normalized)

    # GPU-accelerated spatial filters with CPU fallback
    gradient, grad_dev = _gradient(normalized)
    highpass, hp_dev = _highpass(clahe)
    retinex, ret_dev = _retinex(normalized)
    shadow = _shadow_mask(normalized)
    bg_field = _background_illumination_field(normalized)

    representations = {
        # Preserve full span for raw representation
        "raw": percentile_normalize(raw_gray, 0.0, 100.0) if raw_gray.max() > 255 else np.clip(raw_gray, 0, 255).astype(np.uint8),
        "percentile": normalized,
        "clahe": clahe,
        "gradient": gradient,
        "highpass": highpass,
        "retinex": retinex,
        # Structural representation combines local CLAHE contrast with high-pass edge/crater contours
        "structural": cv2.addWeighted(clahe, 0.70, highpass, 0.30, 0),
    }

    rep_key = str(representation).lower()
    if rep_key == "auto":
        # Default fallback representation if auto called directly on single image
        selected = representations["structural"]
        method = "structural"
    elif not illumination_normalization:
        selected = representations["percentile"]
        method = "percentile_without_illumination_normalization"
    else:
        selected = representations.get(rep_key, representations["structural"])
        method = rep_key if rep_key in representations else "structural"

    illum_stats = compute_illumination_statistics(selected)

    hw_info = {
        "cuda_available": is_cuda_available(),
        "device_used": ret_dev.get("device_used", "cpu"),
        "retinex_ms": ret_dev.get("timing_ms", 0.0),
        "gradient_ms": grad_dev.get("timing_ms", 0.0),
        "highpass_ms": hp_dev.get("timing_ms", 0.0),
    }

    return PreprocessResult(
        image=np.asarray(selected, dtype=np.uint8),
        method=method,
        radiometric=radiometric,
        stages={
            "raw_scientific": image,
            "raw_grayscale": raw_gray,
            "percentile": normalized,
            "clahe": clahe,
            "gradient": gradient,
            "highpass": highpass,
            "retinex": retinex,
            "shadow_mask": shadow,
            "background_field": bg_field,
            "selected": selected,
        },
        illumination_stats=illum_stats,
        hardware_acceleration=hw_info,
    )


def preprocess(
    image: np.ndarray,
    illumination_normalization: bool = True,
    representation: str = "structural",
    metadata: dict | None = None,
    radiometric_mode: str = "safe_normalization",
) -> np.ndarray:
    """Preprocess image array and return uint8 normalized feature raster."""
    return preprocess_with_diagnostics(
        image,
        illumination_normalization,
        representation,
        metadata,
        radiometric_mode,
    ).image


def validate_illumination_robustness(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    detector: str = "sift",
    representation: str = "structural",
) -> dict[str, Any]:
    """Scientifically validate matching improvement under differing illumination.

    Executes a before/after benchmark comparing baseline raw matching vs.
    illumination-normalized representation matching.
    """
    from app.services.pairwise_registration import register_pair

    # 1. Baseline: Raw representation without illumination normalization
    res_baseline = register_pair(
        source_image,
        reference_image,
        detector=detector,
        illumination_normalization=False,
        representation="raw",
    )

    # 2. Normalized: Selected representation with active illumination normalization
    res_normalized = register_pair(
        source_image,
        reference_image,
        detector=detector,
        illumination_normalization=True,
        representation=representation,
    )

    baseline_matches = res_baseline.raw_match_count or 0
    normalized_matches = res_normalized.raw_match_count or 0
    baseline_inliers = int(np.sum(res_baseline.inlier_mask)) if res_baseline.inlier_mask is not None else 0
    normalized_inliers = int(np.sum(res_normalized.inlier_mask)) if res_normalized.inlier_mask is not None else 0

    inlier_improvement_pct = 0.0
    if baseline_inliers > 0:
        inlier_improvement_pct = round(((normalized_inliers - baseline_inliers) / baseline_inliers) * 100.0, 2)
    elif normalized_inliers > 0:
        inlier_improvement_pct = 100.0

    rmse_baseline = res_baseline.refined_rmse
    rmse_normalized = res_normalized.refined_rmse
    rmse_delta = None
    if rmse_baseline is not None and rmse_normalized is not None:
        rmse_delta = round(rmse_baseline - rmse_normalized, 4)

    return {
        "baseline_raw": {
            "success": res_baseline.success,
            "matches": baseline_matches,
            "inliers": baseline_inliers,
            "rmse": rmse_baseline,
            "status": res_baseline.registration_status,
        },
        "illumination_normalized": {
            "success": res_normalized.success,
            "matches": normalized_matches,
            "inliers": normalized_inliers,
            "rmse": rmse_normalized,
            "status": res_normalized.registration_status,
        },
        "measurable_improvements": {
            "additional_inliers": normalized_inliers - baseline_inliers,
            "inlier_improvement_pct": inlier_improvement_pct,
            "rmse_improvement_pixels": rmse_delta,
            "robustness_gain": normalized_inliers >= baseline_inliers,
        },
        "detector_used": detector,
        "representation_tested": representation,
    }


def build_pyramid(gray: np.ndarray, levels: int = 1):
    levels = max(1, min(int(levels), 4))
    pyramid = [gray]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid
