import cv2
import numpy as np


def project_points(points, H):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def compute_metrics(source_points, reference_points, H, inlier_mask, source_shape, reference_shape):
    src = np.asarray(source_points).reshape(-1, 2)
    ref = np.asarray(reference_points).reshape(-1, 2)
    pred = project_points(src, H)

    errors = np.linalg.norm(pred - ref, axis=1)
    inlier_errors = errors[inlier_mask] if np.any(inlier_mask) else np.array([])

    src_h, src_w = source_shape[:2]
    ref_h, ref_w = reference_shape[:2]

    inlier_src = src[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2))
    inlier_ref = ref[inlier_mask] if np.any(inlier_mask) else np.empty((0, 2))

    def coverage(points, w, h, rows=4, cols=4):
        if len(points) == 0:
            return 0.0
        occupied = set()
        for x, y in points:
            c = min(cols - 1, max(0, int(x / max(w, 1) * cols)))
            r = min(rows - 1, max(0, int(y / max(h, 1) * rows)))
            occupied.add((r, c))
        return len(occupied) / float(rows * cols)

    rmse = float(np.sqrt(np.mean(inlier_errors ** 2))) if len(inlier_errors) else None
    mean_error = float(np.mean(inlier_errors)) if len(inlier_errors) else None
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else None

    return {
        "match_count": int(len(src)),
        "inlier_count": int(np.sum(inlier_mask)),
        "inlier_ratio": float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0,
        "rmse_pixels": rmse,
        "mean_reprojection_error_pixels": mean_error,
        "median_reprojection_error_pixels": median_error,
        "source_spatial_coverage": coverage(inlier_src, src_w, src_h),
        "reference_spatial_coverage": coverage(inlier_ref, ref_w, ref_h),
    }
