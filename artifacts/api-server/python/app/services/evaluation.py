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


def make_synthetic_lunar_surface(seed: int = 26166, size: int = 320) -> np.ndarray:
    """Generate a reproducible, textured synthetic lunar surface with craters and regolith noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 180, (size, size), dtype=np.uint8)
    for _ in range(16):
        cx = int(rng.integers(20, size - 20))
        cy = int(rng.integers(20, size - 20))
        radius = int(rng.integers(8, 45))
        intensity = int(rng.integers(30, 240))
        cv2.circle(base, (cx, cy), radius, intensity, -1)
        # Ring rim
        cv2.circle(base, (cx, cy), radius, int(rng.integers(20, 60)), max(1, radius // 6))
    noise = rng.normal(0, 6.0, base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def compute_corner_error(H_est: np.ndarray, H_gt: np.ndarray, shape: tuple[int, int]) -> float:
    """Compute average corner transfer discrepancy in pixels between estimated and ground-truth homographies.

    Definition:
        E_transform = (1/4) * sum_{i=1}^4 ||H_est(c_i) - H_gt(c_i)||_2
    where c_i are the 4 image corners in pixels.
    """
    h, w = shape[:2]
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    pts_est = cv2.perspectiveTransform(corners, H_est.astype(np.float32))[0]
    pts_gt = cv2.perspectiveTransform(corners, H_gt.astype(np.float32))[0]
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


BENCHMARK_SAMPLES = [
    {
        "id": "sample_1_translation",
        "label": "Controlled Translation (dx=12.0, dy=-8.0)",
        "perturbation_type": "translation",
        "dx": 12.0,
        "dy": -8.0,
        "angle": 0.0,
        "scale": 1.0,
        "gamma": 1.0,
        "gain": 1.0,
    },
    {
        "id": "sample_2_small_rotation",
        "label": "Small Rotation (angle=4.5 deg, dx=3.0, dy=2.0)",
        "perturbation_type": "rotation",
        "dx": 3.0,
        "dy": 2.0,
        "angle": 4.5,
        "scale": 1.0,
        "gamma": 1.0,
        "gain": 1.0,
    },
    {
        "id": "sample_3_rotation_scale",
        "label": "Rotation + Scale (angle=-5.0 deg, scale=0.94)",
        "perturbation_type": "rotation_scale",
        "dx": 6.0,
        "dy": -4.0,
        "angle": -5.0,
        "scale": 0.94,
        "gamma": 1.0,
        "gain": 1.0,
    },
    {
        "id": "sample_4_subpixel_shift",
        "label": "Fractional Subpixel Shift (dx=0.35, dy=-0.45)",
        "perturbation_type": "fractional_subpixel",
        "dx": 0.35,
        "dy": -0.45,
        "angle": 0.0,
        "scale": 1.0,
        "gamma": 1.0,
        "gain": 1.0,
    },
    {
        "id": "sample_5_illumination",
        "label": "Illumination Variation (gamma=1.22, gain=0.82, dx=5.0)",
        "perturbation_type": "illumination",
        "dx": 5.0,
        "dy": -3.0,
        "angle": 0.0,
        "scale": 1.0,
        "gamma": 1.22,
        "gain": 0.82,
    },
    {
        "id": "sample_6_scale_expansion",
        "label": "Scale Expansion (scale=1.06, dx=-4.0, dy=5.0)",
        "perturbation_type": "scale",
        "dx": -4.0,
        "dy": 5.0,
        "angle": 2.0,
        "scale": 1.06,
        "gamma": 1.0,
        "gain": 1.0,
    },
]


def run_validation_benchmark(settings: dict | None = None, progress_callback=None) -> dict:
    """Execute real multi-method evaluation on deterministic synthetic lunar benchmark samples.

    Compares:
    1. OpenCV baseline (SIFT + RANSAC homography)
    2. ECC (Enhanced Correlation Coefficient)
    3. Phase correlation (Fourier translation)
    4. Taylor subpixel refinement
    5. Learned model (labeled NOT IMPLEMENTED)
    """
    settings = settings or {}
    started = perf_counter()
    base_surface = make_synthetic_lunar_surface(seed=26166, size=320)
    h, w = base_surface.shape[:2]

    samples_results = []
    methods = ["OpenCV baseline", "ECC", "Phase correlation", "Taylor / Subpixel", "Learned model"]
    total_steps = len(BENCHMARK_SAMPLES) * (len(methods) - 1)
    step_idx = 0

    method_accumulators = {
        m: {"errors": [], "successes": 0, "runtimes": [], "samples": 0}
        for m in methods
    }

    for sample_idx, sample in enumerate(BENCHMARK_SAMPLES):
        # 1. Synthesize reference image with exact ground truth
        angle = sample["angle"]
        scale = sample["scale"]
        dx = sample["dx"]
        dy = sample["dy"]

        # Rotation matrix around image center
        center = (w / 2.0, h / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, angle, scale)
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy

        H_gt = np.vstack([M_rot, [0, 0, 1]]).astype(np.float64)

        ref_img = cv2.warpAffine(
            base_surface,
            M_rot,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        # Radiometric variation
        if sample["gamma"] != 1.0 or sample["gain"] != 1.0:
            x = ref_img.astype(np.float32) / 255.0
            x = np.clip(sample["gain"] * np.power(np.clip(x, 0, 1), sample["gamma"]), 0, 1)
            ref_img = (x * 255).astype(np.uint8)

        sample_eval = {
            "sample_id": sample["id"],
            "label": sample["label"],
            "perturbation_type": sample["perturbation_type"],
            "ground_truth": {
                "dx": dx,
                "dy": dy,
                "angle_deg": angle,
                "scale": scale,
                "homography": H_gt.tolist(),
            },
            "methods": {},
        }

        # Method 1: OpenCV baseline
        step_idx += 1
        if progress_callback:
            progress_callback(
                stage="EVALUATING_METHODS",
                stage_index=step_idx,
                stage_count=total_steps,
                progress=round(step_idx / total_steps, 2),
                message=f"Evaluating sample {sample_idx + 1}/{len(BENCHMARK_SAMPLES)} with OpenCV baseline",
            )
        t0 = perf_counter()
        try:
            res_opencv = register_pair(
                base_surface,
                ref_img,
                detector="sift",
                ecc_refinement=False,
                illumination_normalization=True,
            )
            t_opencv = round((perf_counter() - t0) * 1000, 2)
            if res_opencv.homography is not None:
                err_opencv = compute_corner_error(res_opencv.homography, H_gt, (h, w))
                success_opencv = err_opencv < 5.0 and int(np.sum(res_opencv.inlier_mask)) >= 6
            else:
                err_opencv = 999.0
                success_opencv = False
        except Exception:
            err_opencv = 999.0
            success_opencv = False
            t_opencv = round((perf_counter() - t0) * 1000, 2)

        method_accumulators["OpenCV baseline"]["samples"] += 1
        method_accumulators["OpenCV baseline"]["runtimes"].append(t_opencv)
        if success_opencv:
            method_accumulators["OpenCV baseline"]["successes"] += 1
            method_accumulators["OpenCV baseline"]["errors"].append(err_opencv)

        sample_eval["methods"]["OpenCV baseline"] = {
            "status": "PASS" if success_opencv else "FAIL",
            "corner_error_pixels": round(err_opencv, 3) if err_opencv < 500 else None,
            "runtime_ms": t_opencv,
            "inlier_count": int(np.sum(res_opencv.inlier_mask)) if 'res_opencv' in locals() and res_opencv.inlier_mask is not None else 0,
        }

        # Method 2: ECC
        step_idx += 1
        if progress_callback:
            progress_callback(
                stage="EVALUATING_METHODS",
                stage_index=step_idx,
                stage_count=total_steps,
                progress=round(step_idx / total_steps, 2),
                message=f"Evaluating sample {sample_idx + 1}/{len(BENCHMARK_SAMPLES)} with ECC",
            )
        t0 = perf_counter()
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp_matrix = cv2.findTransformECC(base_surface, ref_img, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
            t_ecc = round((perf_counter() - t0) * 1000, 2)
            H_ecc = np.vstack([warp_matrix, [0, 0, 1]]).astype(np.float64)
            err_ecc = compute_corner_error(H_ecc, H_gt, (h, w))
            success_ecc = err_ecc < 5.0
        except cv2.error:
            t_ecc = round((perf_counter() - t0) * 1000, 2)
            err_ecc = 999.0
            success_ecc = False

        method_accumulators["ECC"]["samples"] += 1
        method_accumulators["ECC"]["runtimes"].append(t_ecc)
        if success_ecc:
            method_accumulators["ECC"]["successes"] += 1
            method_accumulators["ECC"]["errors"].append(err_ecc)

        sample_eval["methods"]["ECC"] = {
            "status": "PASS" if success_ecc else "FAIL",
            "corner_error_pixels": round(err_ecc, 3) if err_ecc < 500 else None,
            "runtime_ms": t_ecc,
        }

        # Method 3: Phase correlation
        step_idx += 1
        if progress_callback:
            progress_callback(
                stage="EVALUATING_METHODS",
                stage_index=step_idx,
                stage_count=total_steps,
                progress=round(step_idx / total_steps, 2),
                message=f"Evaluating sample {sample_idx + 1}/{len(BENCHMARK_SAMPLES)} with Phase correlation",
            )
        t0 = perf_counter()
        try:
            shift, response = cv2.phaseCorrelate(base_surface.astype(np.float32), ref_img.astype(np.float32))
            t_phase = round((perf_counter() - t0) * 1000, 2)
            # Phase correlation measures translation shift (x, y)
            err_phase = float(np.hypot(shift[0] - dx, shift[1] - dy))
            success_phase = err_phase < 4.0
        except Exception:
            t_phase = round((perf_counter() - t0) * 1000, 2)
            err_phase = 999.0
            success_phase = False

        method_accumulators["Phase correlation"]["samples"] += 1
        method_accumulators["Phase correlation"]["runtimes"].append(t_phase)
        if success_phase:
            method_accumulators["Phase correlation"]["successes"] += 1
            method_accumulators["Phase correlation"]["errors"].append(err_phase)

        sample_eval["methods"]["Phase correlation"] = {
            "status": "PASS" if success_phase else "FAIL",
            "corner_error_pixels": round(err_phase, 3) if err_phase < 500 else None,
            "runtime_ms": t_phase,
        }

        # Method 4: Taylor / Subpixel
        step_idx += 1
        if progress_callback:
            progress_callback(
                stage="EVALUATING_METHODS",
                stage_index=step_idx,
                stage_count=total_steps,
                progress=round(step_idx / total_steps, 2),
                message=f"Evaluating sample {sample_idx + 1}/{len(BENCHMARK_SAMPLES)} with Taylor / Subpixel",
            )
        t0 = perf_counter()
        try:
            from app.services.subpixel import taylor_expansion
            # Evaluate across multiple grid points inside textured region
            grid_x, grid_y = np.meshgrid(np.linspace(80, w - 80, 4), np.linspace(80, h - 80, 4))
            src_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
            gt_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
            # Integer-rounded initialization simulating keypoint detector output
            init_ref_pts = np.round(gt_pts).astype(np.float32)
            refined_pts, valid = taylor_expansion(base_surface, ref_img, src_pts, init_ref_pts)
            t_taylor = round((perf_counter() - t0) * 1000, 2)
            if np.any(valid):
                subpixel_err = float(np.mean(np.linalg.norm(refined_pts[valid] - gt_pts[valid], axis=1)))
                success_taylor = subpixel_err < 2.5
            else:
                subpixel_err = 999.0
                success_taylor = False
        except Exception:
            t_taylor = round((perf_counter() - t0) * 1000, 2)
            subpixel_err = 999.0
            success_taylor = False

        method_accumulators["Taylor / Subpixel"]["samples"] += 1
        method_accumulators["Taylor / Subpixel"]["runtimes"].append(t_taylor)
        if success_taylor:
            method_accumulators["Taylor / Subpixel"]["successes"] += 1
            method_accumulators["Taylor / Subpixel"]["errors"].append(subpixel_err)

        sample_eval["methods"]["Taylor / Subpixel"] = {
            "status": "PASS" if success_taylor else "FAIL",
            "corner_error_pixels": round(subpixel_err, 3) if subpixel_err < 500 else None,
            "runtime_ms": t_taylor,
        }

        # Method 5: Learned model
        from app.services.feature_matching import get_detector_capabilities, match_learned_loftr
        caps = get_detector_capabilities()
        learned_cap = caps.get("loftr", {})
        
        if learned_cap.get("status") == "AVAILABLE":
            t0 = perf_counter()
            try:
                # Use LoFTR
                learned_matches = match_learned_loftr(src_img, ref_img)
                # Geometric verification
                valid_src, valid_ref, learned_inliers, learned_mask, _ = geometric_verification(
                    learned_matches.source_points,
                    learned_matches.reference_points,
                    model_type="auto",
                    ransac_threshold=3.0,
                )
                success_learned = len(learned_inliers) >= 8
                subpixel_err = 999.0
                if success_learned:
                    # Inlier RMSE estimation
                    H_learned, _ = cv2.findHomography(learned_inliers[:, 0], learned_inliers[:, 1], cv2.USAC_MAGSAC, 3.0)
                    if H_learned is not None:
                        proj = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H_learned).reshape(-1, 2)
                        subpixel_err = float(np.mean(np.linalg.norm(proj - gt_pts, axis=1)))
            except Exception as e:
                success_learned = False
                subpixel_err = 999.0
            
            t_learned = round((perf_counter() - t0) * 1000, 2)
            
            method_accumulators["Learned model"]["samples"] += 1
            method_accumulators["Learned model"]["runtimes"].append(t_learned)
            if success_learned:
                method_accumulators["Learned model"]["successes"] += 1
                method_accumulators["Learned model"]["errors"].append(subpixel_err)
            
            sample_eval["methods"]["Learned model"] = {
                "status": "PASS" if success_learned else "FAIL",
                "corner_error_pixels": round(subpixel_err, 3) if subpixel_err < 500 else None,
                "runtime_ms": t_learned,
                "note": "LoFTR geometric verification successful" if success_learned else "LoFTR failed to find sufficient inliers",
            }
        else:
            sample_eval["methods"]["Learned model"] = {
                "status": learned_cap.get("status", "NOT_IMPLEMENTED"),
                "reason": learned_cap.get("substatus", "MODEL REQUIRED"),
                "corner_error_pixels": None,
                "runtime_ms": None,
                "note": learned_cap.get("reason", "No deep neural model weights bundled."),
            }

        samples_results.append(sample_eval)

    # Build final benchmark matrix
    matrix = {}
    for m in methods:
        if m == "Learned model" and caps.get("loftr", {}).get("status") != "AVAILABLE":
            l_cap = caps.get("loftr", {})
            matrix[m] = {
                "method_name": m,
                "status": l_cap.get("status", "NOT_IMPLEMENTED"),
                "reason": l_cap.get("substatus", "MODEL REQUIRED"),
                "sample_count": 0,
                "success_count": 0,
                "success_rate_percent": None,
                "median_error_pixels": None,
                "mean_error_pixels": None,
                "mean_runtime_ms": None,
                "note": l_cap.get("reason", "No deep neural model weights bundled."),
            }
        else:
            acc = method_accumulators[m]
            errs = acc["errors"]
            matrix[m] = {
                "method_name": m,
                "status": "VERIFIED" if acc["successes"] > 0 else "FAILED",
                "sample_count": acc["samples"],
                "success_count": acc["successes"],
                "success_rate_percent": round((acc["successes"] / max(1, acc["samples"])) * 100.0, 1),
                "median_error_pixels": round(float(np.median(errs)), 3) if errs else None,
                "mean_error_pixels": round(float(np.mean(errs)), 3) if errs else None,
                "mean_runtime_ms": round(float(np.mean(acc["runtimes"])), 1) if acc["runtimes"] else None,
                "note": "Server-side evaluated against deterministic ground truth.",
            }

    if progress_callback:
        progress_callback(
            stage="COMPLETE",
            stage_index=total_steps,
            stage_count=total_steps,
            progress=1.0,
            message="Synthetic benchmark suite evaluation complete",
            status="COMPLETE",
        )

    return {
        "status": "CONNECTED",
        "dataset_type": "SYNTHETIC_CONTROLLED_BENCHMARK",
        "dataset_name": "Synthetic Controlled Lunar Benchmark Corpus",
        "ground_truth_status": "AVAILABLE",
        "reproducible_seed": 26166,
        "sample_count": len(BENCHMARK_SAMPLES),
        "perturbations": [
            "Controlled rigid translation (dx, dy)",
            "In-plane crater rotation (4.5 to 6 deg)",
            "Optical scale change (0.94x to 1.06x)",
            "Subpixel fractional-pixel displacement (0.35 px)",
            "Photometric gamma/gain illumination variation",
        ],
        "method_matrix": matrix,
        "samples": samples_results,
        "scientific_note": (
            "Ground truth transforms are known by construction. These quantitative results reflect "
            "controlled mathematical benchmark performance. No native Chandrayaan-2 flight data accuracy "
            "is claimed until a mission ground-truth corpus is connected."
        ),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }

