from dataclasses import dataclass
import cv2
import numpy as np
import logging

_superpoint_model = None
_loftr_model = None

def _get_superpoint():
    global _superpoint_model
    if _superpoint_model is None:
        try:
            import torch
            try:
                from kornia.feature import SuperPoint
                _superpoint_model = SuperPoint(True).eval()
            except ImportError:
                from lightglue import SuperPoint
                _superpoint_model = SuperPoint(max_num_keypoints=2048).eval()
        except ImportError as e:
            raise RuntimeError(f"Dependencies missing for SuperPoint: {e}")
    return _superpoint_model

def _get_loftr():
    global _loftr_model
    if _loftr_model is None:
        try:
            import torch
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            from kornia.feature import LoFTR
            _loftr_model = LoFTR(pretrained='outdoor').eval()
        except ImportError as e:
            raise RuntimeError(f"Dependencies missing for LoFTR: {e}")
    return _loftr_model

def _compute_superpoint(gray: np.ndarray, max_features: int) -> FeatureResult:
    import torch
    if gray.dtype != np.float32:
        gray_t = gray.astype(np.float32) / 255.0
    else:
        gray_t = gray
    tensor = torch.from_numpy(gray_t).unsqueeze(0).unsqueeze(0)
    model = _get_superpoint()
    with torch.no_grad():
        if hasattr(model, 'extract'):
            out = model({'image': tensor})
            keypoints = out['keypoints'][0].cpu().numpy()
            descriptors = out['descriptors'][0].cpu().numpy()
            responses = out['keypoint_scores'][0].cpu().numpy()
        else:
            out = model(tensor)
            keypoints = out['keypoints'][0].cpu().numpy()
            descriptors = out['descriptors'][0].cpu().numpy().T
            responses = out.get('scores', [torch.ones(len(keypoints))])[0].cpu().numpy() if 'scores' in out else out.get('keypoint_scores', [torch.ones(len(keypoints))])[0].cpu().numpy()
    if len(keypoints) == 0:
        raise ValueError("SuperPoint found no features.")
    
    # We can cap the max_features simply by slicing since SuperPoint outputs them by score
    if len(keypoints) > max_features:
        keypoints = keypoints[:max_features]
        descriptors = descriptors[:max_features]
        responses = responses[:max_features]
        
    return FeatureResult(keypoints, descriptors, "SUPERPOINT", responses)

def _match_loftr(source_gray: np.ndarray, reference_gray: np.ndarray) -> MatchResult:
    import torch
    if source_gray.dtype != np.float32:
        src_t = source_gray.astype(np.float32) / 255.0
    else:
        src_t = source_gray
    if reference_gray.dtype != np.float32:
        ref_t = reference_gray.astype(np.float32) / 255.0
    else:
        ref_t = reference_gray
        
    tensor_src = torch.from_numpy(src_t).unsqueeze(0).unsqueeze(0)
    tensor_ref = torch.from_numpy(ref_t).unsqueeze(0).unsqueeze(0)
    model = _get_loftr()
    with torch.no_grad():
        out = model({"image0": tensor_src, "image1": tensor_ref})
        
    kp_src = out['keypoints0'].cpu().numpy()
    kp_ref = out['keypoints1'].cpu().numpy()
    
    if len(kp_src) < 4:
        raise ValueError(f"LoFTR found insufficient matches ({len(kp_src)}).")
        
    good_matches = []
    ratios = []
    for i in range(len(kp_src)):
        m = cv2.DMatch(i, i, 0)
        good_matches.append(m)
        ratios.append(0.0)
        
    return MatchResult(
        keypoints_source=[],
        keypoints_reference=[],
        good_matches=good_matches,
        source_points=kp_src.reshape(-1, 1, 2),
        reference_points=kp_ref.reshape(-1, 1, 2),
        ratios=ratios,
        descriptor="LOFTR"
    )

@dataclass
class MatchResult:
    keypoints_source: list
    keypoints_reference: list
    good_matches: list
    ratios: list
    source_points: np.ndarray
    reference_points: np.ndarray
    descriptor: str


@dataclass
class FeatureResult:
    points: np.ndarray
    descriptors: np.ndarray
    descriptor: str
    responses: np.ndarray = None


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
        if hasattr(cv2, 'AKAZE_create'):
            return cv2.AKAZE_create()
        else:
            return cv2.xfeatures2d.AKAZE_create()
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
    if detector_name.lower() == "loftr":
        return _match_loftr(source_gray, reference_gray)
        
    return match_feature_artifacts(
        compute_features(source_gray, detector_name, max_features),
        compute_features(reference_gray, detector_name, max_features), detector_name, ratio,
    )


def compute_features(gray: np.ndarray, detector_name: str = "sift", max_features: int = 8000) -> FeatureResult:
    if detector_name.lower() == "superpoint":
        return _compute_superpoint(gray, max_features)
        
    if gray.dtype != np.uint8:
        import cv2
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    keypoints, descriptors = create_detector(detector_name, max_features=max_features).detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        raise ValueError("Feature detector found insufficient usable descriptors.")
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    return FeatureResult(np.float32([item.pt for item in keypoints]), descriptors, detector_name.upper(), responses)


def match_feature_artifacts(source_features: FeatureResult, reference_features: FeatureResult, detector_name: str = "sift", ratio: float = 0.72) -> MatchResult:
    des_s, des_r = source_features.descriptors, reference_features.descriptors

    if detector_name.lower() == "orb":
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_s, des_r, k=2)

    good = []
    ratios_list = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
            ratios_list.append(m.distance / max(n.distance, 1e-6))

    if len(good) < 4:
        raise ValueError(
            f"Only {len(good)} reliable matches found. "
            "Try a higher ratio, a different detector, or images with more overlap."
        )

    src_pts = source_features.points[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    ref_pts = reference_features.points[[m.trainIdx for m in good]].reshape(-1, 1, 2)

    return MatchResult(
        keypoints_source=[],
        keypoints_reference=[],
        good_matches=good,
        source_points=src_pts,
        reference_points=ref_pts,
        ratios=ratios_list,
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
        ratios=[match_result.ratios[i] for i in keep],
        source_points=match_result.source_points[keep],
        reference_points=match_result.reference_points[keep],
        descriptor=match_result.descriptor,
    )

