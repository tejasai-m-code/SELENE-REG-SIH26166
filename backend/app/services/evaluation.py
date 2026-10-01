"""Controlled synthetic robustness evaluation for SELENE-REG.

This module deliberately labels its results as synthetic. It measures how the
current correspondence pipeline behaves when a raster is transformed by known
rotation, scale, illumination and noise changes. It is not a substitute for
validation on Chandrayaan-2/LROC/SELENE mission products.
"""
from __future__ import annotations

from time import perf_counter
import cv2
import numpy as np

from app.services.pairwise_registration import register_pair


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _transform(image: np.ndarray, angle: float, scale: float, gamma: float,
               gain: float, blur: float, noise_sigma: float) -> np.ndarray:
    base = _to_bgr(image)
    h, w = base.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    transformed = cv2.warpAffine(
        base, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    x = transformed.astype(np.float32) / 255.0
    x = np.clip(gain * np.power(np.clip(x, 0, 1), gamma), 0, 1)
    transformed = (x * 255).astype(np.uint8)
    if blur > 0:
        transformed = cv2.GaussianBlur(transformed, (0, 0), blur)
    if noise_sigma > 0:
        rng = np.random.default_rng(26166)
        noise = rng.normal(0, noise_sigma, transformed.shape).astype(np.float32)
        transformed = np.clip(transformed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return transformed


SCENARIOS = [
    ("baseline", 0.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    ("rotation_8deg", 8.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    ("scale_0.82x", 0.0, 0.82, 1.0, 1.0, 0.0, 0.0),
    ("scale_1.18x", 0.0, 1.18, 1.0, 1.0, 0.0, 0.0),
    ("illumination_dark", 0.0, 1.0, 1.18, 0.62, 0.0, 0.0),
    ("illumination_bright", 0.0, 1.0, 0.86, 1.28, 0.0, 0.0),
    ("blur_noise", 0.0, 1.0, 1.0, 1.0, 0.8, 5.0),
]


def run_synthetic_robustness(image: np.ndarray, settings: dict | None = None) -> dict:
    settings = settings or {}
    started = perf_counter()
    results = []
    base = _to_bgr(image)

    for name, angle, scale, gamma, gain, blur, noise in SCENARIOS:
        variant = _transform(base, angle, scale, gamma, gain, blur, noise)
        case_start = perf_counter()
        try:
            result = register_pair(
                base, variant,
                detector=settings.get("detector", "sift"),
                ratio=float(settings.get("ratio", 0.72)),
                ransac_threshold=float(settings.get("ransac_threshold", 3.0)),
                illumination_normalization=bool(settings.get("illumination_normalization", True)),
                spatial_distribution=bool(settings.get("spatial_distribution", True)),
                ecc_refinement=bool(settings.get("ecc_refinement", True)),
                max_features=int(settings.get("max_features", 8000)),
                match_preview_max_side=900,
                include_registered=False,
            )
            m = result.metrics
            accepted = (
                m["inlier_count"] >= 6
                and m["inlier_ratio"] >= 0.10
                and m["rmse_pixels"] is not None
                and m["rmse_pixels"] <= 8.0
                and m["source_spatial_coverage"] >= 0.0625
            )
            results.append({
                "scenario": name,
                "status": "PASS" if accepted else "REVIEW",
                "inlier_count": m["inlier_count"],
                "inlier_ratio": m["inlier_ratio"],
                "rmse_pixels": m["rmse_pixels"],
                "spatial_coverage": m["source_spatial_coverage"],
                "processing_time_seconds": round(perf_counter() - case_start, 4),
            })
        except Exception as exc:
            results.append({
                "scenario": name,
                "status": "ERROR",
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "rmse_pixels": None,
                "spatial_coverage": 0.0,
                "processing_time_seconds": round(perf_counter() - case_start, 4),
                "error": str(exc),
            })

    passed = sum(r["status"] == "PASS" for r in results)
    return {
        "evaluation_type": "synthetic_robustness_stress_test",
        "scientific_note": (
            "Controlled raster perturbations only. These results demonstrate "
            "software robustness trends and must not be presented as evidence "
            "of cross-mission or real lunar-data performance."
        ),
        "thresholds": {
            "min_inliers": 6,
            "min_inlier_ratio": 0.10,
            "max_rmse_pixels": 8.0,
            "min_spatial_coverage": 0.0625,
        },
        "results": results,
        "passed_cases": passed,
        "total_cases": len(results),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
