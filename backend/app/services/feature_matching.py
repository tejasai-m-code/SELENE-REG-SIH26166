from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class MatchResult:
    keypoints_source: list
    keypoints_reference: list
    good_matches: list
    source_points: np.ndarray
    reference_points: np.ndarray
    descriptor: str


def create_detector(name: str = "sift", max_features: int = 8000):
    name = name.lower()
    if name == "orb":
        return cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=10
        )
    if name == "akaze":
        return cv2.AKAZE_create()
    return cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=4,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6
    )


def match_features(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    detector_name: str = "sift",
    ratio: float = 0.72,
    max_features: int = 8000,
) -> MatchResult:
    detector = create_detector(detector_name, max_features=max_features)

    kp_s, des_s = detector.detectAndCompute(source_gray, None)
    kp_r, des_r = detector.detectAndCompute(reference_gray, None)

    if des_s is None or des_r is None:
        raise ValueError("Feature detector found insufficient usable descriptors.")

    if detector_name.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_s, des_r, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        raise ValueError(
            f"Only {len(good)} reliable matches found. "
            "Try a higher ratio, a different detector, or images with more overlap."
        )

    src_pts = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return MatchResult(
        keypoints_source=kp_s,
        keypoints_reference=kp_r,
        good_matches=good,
        source_points=src_pts,
        reference_points=ref_pts,
        descriptor=detector_name.upper(),
    )


def spatially_distribute_matches(
    match_result: MatchResult,
    image_shape,
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_per_cell: int = 30,
) -> MatchResult:
    h, w = image_shape[:2]
    src = match_result.source_points.reshape(-1, 2)

    cells = {}
    for idx, (x, y) in enumerate(src):
        c = min(grid_cols - 1, max(0, int(x / max(w, 1) * grid_cols)))
        r = min(grid_rows - 1, max(0, int(y / max(h, 1) * grid_rows)))
        cells.setdefault((r, c), []).append(idx)

    keep = []
    for _, indices in cells.items():
        indices = sorted(
            indices,
            key=lambda i: match_result.good_matches[i].distance
        )
        keep.extend(indices[:max_per_cell])

    keep = sorted(set(keep))
    if len(keep) < 4:
        return match_result

    return MatchResult(
        keypoints_source=match_result.keypoints_source,
        keypoints_reference=match_result.keypoints_reference,
        good_matches=[match_result.good_matches[i] for i in keep],
        source_points=match_result.source_points[keep],
        reference_points=match_result.reference_points[keep],
        descriptor=match_result.descriptor,
    )
