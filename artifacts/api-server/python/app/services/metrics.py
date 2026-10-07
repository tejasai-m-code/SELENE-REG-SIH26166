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
        psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
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


def evaluate_spatial_distribution(points, width: int, height: int, rows: int = 4, cols: int = 4) -> dict:
    """Comprehensive spatial correspondence-quality evaluation.

    Evaluates:
    - spatial_coverage: fraction of grid cells occupied
    - grid_occupancy: occupied cells / total cells
    - occupied_cell_ratio: same as grid_occupancy
    - convex_hull_coverage: area of 2D convex hull / image area
    - centroid_distribution: offset from image center normalized by half diagonal
    - nearest_neighbour_spacing: distance statistics between closest point pairs
    - spatial_uniformity: normalized entropy across grid cells
    - largest_cluster_fraction: maximum cell count / total count
    - border_coverage: count of the 4 borders (top, bottom, left, right) with points in outer 15% margin
    - distribution_category: 'GOOD_DISTRIBUTION', 'LIMITED_DISTRIBUTION', or 'CLUSTERED_RISKY_DISTRIBUTION'
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n_pts = len(pts)

    if n_pts == 0 or width <= 0 or height <= 0:
        return {
            "spatial_coverage": 0.0,
            "grid_occupancy": 0.0,
            "occupied_cell_ratio": 0.0,
            "convex_hull_coverage": 0.0,
            "centroid_offset_normalized": 1.0,
            "nearest_neighbour_spacing": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "spatial_uniformity": 0.0,
            "largest_cluster_fraction": 0.0,
            "border_coverage": {"top": False, "bottom": False, "left": False, "right": False, "count": 0},
            "distribution_category": "LIMITED_DISTRIBUTION",
        }

    # 1. Grid occupancy & Uniformity
    grid = spatial_grid(pts, width, height, rows, cols)
    occupancy = float(grid["populated_cells"] / max(grid["total_cells"], 1))
    uniformity = _uniformity(pts, width, height, rows=rows, cols=cols)

    # 2. Convex Hull Coverage
    hull_coverage = 0.0
    if n_pts >= 3:
        try:
            hull = cv2.convexHull(pts)
            hull_area = float(cv2.contourArea(hull))
            hull_coverage = float(hull_area / max(width * height, 1))
        except (cv2.error, ValueError):
            hull_coverage = 0.0

    # 3. Centroid Distribution
    centroid = np.mean(pts, axis=0)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    max_dist = float(np.linalg.norm(center))
    dist_to_center = float(np.linalg.norm(centroid - center))
    norm_centroid_offset = float(dist_to_center / max_dist) if max_dist > 0 else 0.0

    # 4. Nearest Neighbor Spacing
    if n_pts >= 2:
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        nn_dist = np.sqrt(np.min(dist_sq, axis=1))
        nn_stats = {
            "mean": float(np.mean(nn_dist)),
            "median": float(np.median(nn_dist)),
            "min": float(np.min(nn_dist)),
            "max": float(np.max(nn_dist)),
        }
    else:
        nn_stats = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    # 5. Border Coverage (outer 15% margins)
    margin_x = 0.15 * width
    margin_y = 0.15 * height
    b_left = bool(np.any(pts[:, 0] <= margin_x))
    b_right = bool(np.any(pts[:, 0] >= width - margin_x))
    b_top = bool(np.any(pts[:, 1] <= margin_y))
    b_bottom = bool(np.any(pts[:, 1] >= height - margin_y))
    border_count = int(b_left + b_right + b_top + b_bottom)
    border_info = {
        "top": b_top,
        "bottom": b_bottom,
        "left": b_left,
        "right": b_right,
        "count": border_count,
    }

    # 6. Distribution Category Classification
    # Prevent a registration passing merely because points are concentrated in one small textured region.
    # Distinguish: GOOD DISTRIBUTION, LIMITED DISTRIBUTION, CLUSTERED / RISKY DISTRIBUTION
    largest_cluster = uniformity["largest_cluster_fraction"]
    if n_pts >= 8 and (largest_cluster >= 0.65 or (hull_coverage < 0.02 and occupancy <= 0.15)):
        category = "CLUSTERED_RISKY_DISTRIBUTION"
    elif occupancy >= 0.30 and uniformity["normalized_entropy"] >= 0.40 and (hull_coverage >= 0.05 or n_pts >= 12):
        category = "GOOD_DISTRIBUTION"
    else:
        category = "LIMITED_DISTRIBUTION"

    return {
        "spatial_coverage": occupancy,
        "grid_occupancy": occupancy,
        "occupied_cell_ratio": occupancy,
        "convex_hull_coverage": hull_coverage,
        "centroid_offset_normalized": norm_centroid_offset,
        "nearest_neighbour_spacing": nn_stats,
        "spatial_uniformity": uniformity["normalized_entropy"],
        "largest_cluster_fraction": largest_cluster,
        "border_coverage": border_info,
        "distribution_category": category,
    }


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

    source_spatial = evaluate_spatial_distribution(inlier_src, src_w, src_h)
    reference_spatial = evaluate_spatial_distribution(inlier_ref, ref_w, ref_h)

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
        "source_spatial_coverage": source_spatial["spatial_coverage"],
        "reference_spatial_coverage": reference_spatial["spatial_coverage"],
        "spatial_uniformity": source_spatial["spatial_uniformity"],
        "largest_spatial_cluster_fraction": source_spatial["largest_cluster_fraction"],
        "spatial_distribution_category": source_spatial["distribution_category"],
        "convex_hull_coverage": source_spatial["convex_hull_coverage"],
        "nearest_neighbour_spacing": source_spatial["nearest_neighbour_spacing"],
        "source_spatial_distribution": source_spatial,
        "reference_spatial_distribution": reference_spatial,
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

