import cv2
import numpy as np


def estimate_homography(source_points, reference_points, ransac_threshold=3.0):
    if len(source_points) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        ransac_threshold,
        maxIters=10000,
        confidence=0.995
    )

    if H is None or mask is None:
        raise ValueError("Homography estimation failed.")

    mask = mask.ravel().astype(bool)
    return H, mask


def refine_with_ecc(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    initial_homography: np.ndarray,
    iterations: int = 100,
):
    """Optional intensity-based refinement after feature/RANSAC initialization."""
    try:
        src = source_gray.astype(np.float32) / 255.0
        ref = reference_gray.astype(np.float32) / 255.0

        warp = initial_homography.astype(np.float32).copy()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            iterations,
            1e-6,
        )

        # ECC expects a warp mapping template/reference to input when using
        # WARP_INVERSE_MAP. Starting from our source->reference H, invert it.
        warp_inv = np.linalg.inv(warp).astype(np.float32)
        cc, refined_inv = cv2.findTransformECC(
            ref,
            src,
            warp_inv,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            1
        )
        refined = np.linalg.inv(refined_inv)
        refined /= refined[2, 2]
        return refined, float(cc), True
    except cv2.error:
        return initial_homography, None, False
    except np.linalg.LinAlgError:
        return initial_homography, None, False


def warp_to_reference(source_image, reference_shape, homography):
    h, w = reference_shape[:2]
    return cv2.warpPerspective(source_image, homography, (w, h))


def draw_matches(
    source_gray,
    reference_gray,
    source_points,
    reference_points,
    inlier_mask=None,
    max_points=180,
):
    src_vis = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR) if source_gray.ndim == 2 else source_gray.copy()
    ref_vis = cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2BGR) if reference_gray.ndim == 2 else reference_gray.copy()

    pts_s = source_points.reshape(-1, 2)
    pts_r = reference_points.reshape(-1, 2)

    if inlier_mask is None:
        selected = np.arange(len(pts_s))
    else:
        selected = np.where(inlier_mask)[0]

    selected = selected[:max_points]

    sh, sw = src_vis.shape[:2]
    rh, rw = ref_vis.shape[:2]
    height = max(sh, rh)
    width = sw + rw

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:sh, :sw] = src_vis
    canvas[:rh, sw:sw+rw] = ref_vis

    for idx in selected:
        a = tuple(np.round(pts_s[idx]).astype(int))
        b = tuple(np.round(pts_r[idx]).astype(int) + np.array([sw, 0]))
        cv2.circle(canvas, a, 3, (0, 255, 0), -1)
        cv2.circle(canvas, b, 3, (0, 255, 0), -1)
        cv2.line(canvas, a, b, (255, 190, 0), 1, cv2.LINE_AA)

    return canvas

def draw_feature_response_map(image_shape, points, responses, descriptor="SIFT"):
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    if len(points) > 0 and responses is not None and len(responses) == len(points):
        responses = np.array(responses, dtype=np.float32)
        r_max = np.max(responses) if np.max(responses) > 0 else 1.0
        norm_resp = responses / r_max
        for pt, r in zip(points.reshape(-1, 2), norm_resp):
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(heatmap, (x, y), max(2, int(15 * r)), float(r), -1)
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_color
    else:
        return np.zeros((h, w, 3), dtype=np.uint8)

def draw_keypoint_map(image, points):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for pt in points.reshape(-1, 2):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 5, 1)
    return vis
