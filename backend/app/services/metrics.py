import cv2
import numpy as np


def project_points(points, H):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _conditioning(H):
    try:
        matrix = np.asarray(H, dtype=np.float64)
        if matrix.shape != (3, 3):
            return None
        singular = np.linalg.svd(matrix)[1]
        if singular[-1] <= 1e-12:
            return float("inf")
        return float(singular[0] / singular[-1])
    except (ValueError, np.linalg.LinAlgError):
        return None


def _uniformity(points, width, height, rows=8, cols=8):
    if len(points) == 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "largest_cluster_fraction": 0.0}
    grid = spatial_grid(points, width, height, rows, cols)
    counts = np.asarray(grid["counts"], dtype=np.float64).ravel()
    total = counts.sum()
    if total <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "largest_cluster_fraction": 0.0}
    p = counts[counts > 0] / total
    entropy = float(-np.sum(p * np.log2(p)))
    max_entropy = float(np.log2(rows * cols))
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
        "largest_cluster_fraction": float(np.max(counts) / total),
    }


def _image_metrics(registered, reference):
    """Downsampled image metrics; only used when image arrays are explicitly supplied."""
    if registered is None or reference is None:
        return {}
    try:
        a = registered if registered.ndim == 2 else cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
        b = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        h = min(a.shape[0], b.shape[0], 512)
        w = min(a.shape[1], b.shape[1], 512)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = a - b
        mse = float(np.mean(diff * diff))
        psnr = 999.9 if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
        # Global normalized mutual information, histogram-based and intentionally lightweight.
        ai = np.clip(a.astype(np.int32), 0, 255)
        bi = np.clip(b.astype(np.int32), 0, 255)
        joint, _, _ = np.histogram2d(ai.ravel(), bi.ravel(), bins=32, range=((0, 256), (0, 256)))
        joint = joint / max(joint.sum(), 1.0)
        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)
        nz = joint > 0
        h_ab = float(-np.sum(joint[nz] * np.log2(joint[nz])))
        h_a = float(-np.sum(pa[pa > 0] * np.log2(pa[pa > 0])))
        h_b = float(-np.sum(pb[pb > 0] * np.log2(pb[pb > 0])))
        nmi = float((h_a + h_b) / max(h_ab, 1e-9))
        # A global SSIM-style statistic without pulling in another dependency.
        mu_a, mu_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        cov = float(np.mean((a - mu_a) * (b - mu_b)))
        c1, c2 = 6.5025, 58.5225
        ssim = float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                     max((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2), 1e-9))
        return {"psnr_db": psnr, "ssim": max(-1.0, min(1.0, ssim)), "nmi": nmi}
    except (ValueError, cv2.error, FloatingPointError):
        return {}


def compute_metrics(source_points, reference_points, H, inlier_mask, source_shape, reference_shape,
                    registered=None, reference_image=None):
    src = np.asarray(source_points).reshape(-1, 2)
    ref = np.asarray(reference_points).reshape(-1, 2)
    pred = project_points(src, H)

    errors = np.linalg.norm(pred - ref, axis=1)
    inlier_errors = errors[inlier_mask] if np.any(inlier_mask) else np.array([])

    src_h, src_w = source_shape[:2]
    ref_h, ref_w = reference_shape[:2]

    inlier_src = src[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2))
    inlier_ref = ref[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2))

    source_grid = spatial_grid(inlier_src, src_w, src_h)
    reference_grid = spatial_grid(inlier_ref, ref_w, ref_h)
    uniformity = _uniformity(inlier_src, src_w, src_h)

    rmse = float(np.sqrt(np.mean(inlier_errors ** 2))) if len(inlier_errors) else None
    mean_error = float(np.mean(inlier_errors)) if len(inlier_errors) else None
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else None
    p90_error = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) else None
    max_error = float(np.max(inlier_errors)) if len(inlier_errors) else None
    residual_std = float(np.std(inlier_errors)) if len(inlier_errors) else None

    metrics = {
        "match_count": int(len(src)),
        "inlier_count": int(np.sum(inlier_mask)),
        "inlier_ratio": float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0,
        "rmse_pixels": rmse,
        "mean_reprojection_error_pixels": mean_error,
        "median_reprojection_error_pixels": median_error,
        "p90_reprojection_error_pixels": p90_error,
        "max_reprojection_error_pixels": max_error,
        "residual_std_pixels": residual_std,
        "source_spatial_coverage": source_grid["coverage_percent"] / 100.0,
        "reference_spatial_coverage": reference_grid["coverage_percent"] / 100.0,
        "spatial_uniformity": uniformity["normalized_entropy"],
        "largest_spatial_cluster_fraction": uniformity["largest_cluster_fraction"],
        "source_spatial_grid": source_grid,
        "reference_spatial_grid": reference_grid,
        "transform_conditioning": _conditioning(H),
    }
    metrics.update(_image_metrics(registered, reference_image))
    # Conservative evidence score: descriptive, not a claim of scientific accuracy.
    components = []
    if metrics["inlier_ratio"] is not None:
        components.append(min(1.0, max(0.0, metrics["inlier_ratio"])))
    components.append(min(1.0, max(0.0, metrics["source_spatial_coverage"])))
    components.append(min(1.0, max(0.0, metrics["spatial_uniformity"])))
    if rmse is not None:
        components.append(1.0 / (1.0 + max(0.0, rmse)))
    metrics["evidence_score"] = float(np.mean(components)) if components else 0.0
    return metrics


def spatial_grid(points, width, height, rows=4, cols=4):
    counts = [[0 for _ in range(cols)] for _ in range(rows)]
    for x, y in np.asarray(points).reshape(-1, 2):
        col = min(cols - 1, max(0, int(x / max(width, 1) * cols)))
        row = min(rows - 1, max(0, int(y / max(height, 1) * rows)))
        counts[row][col] += 1
    populated = sum(value > 0 for row in counts for value in row)
    return {"rows": rows, "cols": cols, "counts": counts, "populated_cells": populated, "total_cells": rows * cols, "total_inliers": int(sum(map(sum, counts))), "coverage_percent": round(populated / float(rows * cols) * 100, 3)}
