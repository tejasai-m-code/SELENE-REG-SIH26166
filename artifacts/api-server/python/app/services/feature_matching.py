from dataclasses import dataclass, field
import cv2
import numpy as np

from app.services.gpu_accelerator import is_cuda_available, match_descriptors_gpu, get_gpu_info

LEARNED_AVAILABLE = False
LEARNED_REASON = "Model weights or dependencies unavailable."
try:
    import torch
    import kornia
    from kornia.feature import LoFTR
    # Check if loftr checkpoint exists to avoid silent downloads
    import os
    
    # 1. Project models directory
    # Go up from artifacts/api-server/python/app/services to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    local_models_dir = os.path.join(project_root, "models")
    
    # 2. Configured MODEL_PATH
    env_models_dir = os.environ.get("MODEL_PATH")
    
    # 3. Torch hub cache
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    
    # Search for LoFTR checkpoints
    checkpoint_found = False
    found_location = ""
    
    for search_dir in [local_models_dir, env_models_dir, cache_dir]:
        if search_dir and os.path.exists(search_dir):
            if any("loftr" in f.lower() and (f.endswith(".ckpt") or f.endswith(".pt")) for f in os.listdir(search_dir)):
                checkpoint_found = True
                found_location = search_dir
                break

    if checkpoint_found:
        LEARNED_AVAILABLE = True
        LEARNED_REASON = f"Kornia LoFTR checkpoint found at {found_location}."
    else:
        LEARNED_REASON = "Kornia installed but LoFTR checkpoint not found in models/ or torch cache."
except ImportError:
    LEARNED_REASON = "Python packages 'torch' and 'kornia' not installed."



@dataclass
class MatchResult:
    keypoints_source: list
    keypoints_reference: list
    good_matches: list
    source_points: np.ndarray
    reference_points: np.ndarray
    descriptor: str
    detector_requested: str = "SIFT"
    detector_used: str = "SIFT"
    fallback_triggered: bool = False
    match_confidences: list[float] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class FeatureResult:
    points: np.ndarray
    descriptors: np.ndarray | None
    descriptor: str
    keypoints: list = field(default_factory=list)


def get_detector_capabilities() -> dict[str, dict[str, Any]]:
    """Return runtime capability matrix for classical and learned feature matchers."""
    return {
        "sift": {
            "name": "SIFT",
            "category": "CLASSICAL",
            "status": "AVAILABLE",
            "reason": "OpenCV SIFT compiled and operational",
            "description": "Scale-Invariant Feature Transform (DoG extrema + 128D gradient descriptors)",
            "gpu_accelerated_matching": is_cuda_available(),
        },
        "orb": {
            "name": "ORB",
            "category": "CLASSICAL",
            "status": "AVAILABLE",
            "reason": "OpenCV ORB compiled and operational",
            "description": "Oriented FAST and Rotated BRIEF (256-bit binary descriptors)",
            "gpu_accelerated_matching": is_cuda_available(),
        },
        "akaze": {
            "name": "AKAZE",
            "category": "CLASSICAL",
            "status": "AVAILABLE" if hasattr(cv2, "AKAZE_create") else "NOT AVAILABLE",
            "reason": "OpenCV AKAZE operational" if hasattr(cv2, "AKAZE_create") else "AKAZE bindings not compiled in OpenCV",
            "description": "Accelerated-KAZE (non-linear diffusion scale space)",
            "gpu_accelerated_matching": is_cuda_available(),
        },
        "superpoint": {
            "name": "SuperPoint",
            "category": "LEARNED",
            "status": "NOT AVAILABLE",
            "substatus": "MODEL REQUIRED",
            "reason": "Required model checkpoint unavailable. External .pt weights (superpoint_v1.pth) required.",
            "description": "Self-Supervised Interest Point Detection and Description",
            "gpu_accelerated_matching": False,
        },
        "loftr": {
            "name": "LoFTR",
            "category": "LEARNED",
            "status": "AVAILABLE" if LEARNED_AVAILABLE else "NOT AVAILABLE",
            "substatus": "AVAILABLE" if LEARNED_AVAILABLE else "MODEL REQUIRED",
            "reason": LEARNED_REASON,
            "description": "Detector-Free Local Feature Matching with Transformers",
            "gpu_accelerated_matching": is_cuda_available() if LEARNED_AVAILABLE else False,
        },
        "lightglue": {
            "name": "LightGlue",
            "category": "LEARNED",
            "status": "NOT AVAILABLE",
            "substatus": "DEPENDENCY MISSING",
            "reason": "LightGlue weights not bundled.",
            "description": "Local Feature Matching at Light Speed with Transformers",
            "gpu_accelerated_matching": False,
        },
        "superglue": {
            "name": "SuperGlue",
            "category": "LEARNED",
            "status": "NOT AVAILABLE",
            "substatus": "MODEL REQUIRED",
            "reason": "Required model checkpoint unavailable. External graph neural network weights (.pth) required.",
            "description": "Graph Neural Network Correspondence Matching",
            "gpu_accelerated_matching": False,
        },
    }


def create_detector(name: str = "sift", max_features: int = 8000):
    name = name.lower()
    if name in ("loftr", "superpoint", "superglue", "lightglue"):
        raise ValueError(
            f"Learned matcher '{name.upper()}' status is NOT AVAILABLE (MODEL REQUIRED / DEPENDENCY MISSING). "
            f"Required model checkpoint or dependency is unavailable in current runtime. "
            "Available operational detectors are: SIFT, ORB, AKAZE."
        )
    if name == "orb":
        return cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=10,
        )
    if name == "akaze":
        if hasattr(cv2, "AKAZE_create"):
            return cv2.AKAZE_create()
        # Graceful fallback if AKAZE is not compiled into OpenCV build
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=4,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6,
    )

def match_learned_loftr(gray_source: np.ndarray, gray_reference: np.ndarray) -> MatchResult:
    if not LEARNED_AVAILABLE:
        raise ValueError(f"Learned matcher LoFTR is NOT AVAILABLE: {LEARNED_REASON}")
    import torch
    from kornia.feature import LoFTR
    import kornia
    device = torch.device('cuda' if is_cuda_available() else 'cpu')
    matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    # Convert numpy arrays to torch tensors (B, C, H, W) in [0, 1]
    ts1 = kornia.utils.image_to_tensor(gray_source, keepdim=False).float().to(device) / 255.0
    ts2 = kornia.utils.image_to_tensor(gray_reference, keepdim=False).float().to(device) / 255.0
    
    with torch.inference_mode():
        input_dict = {"image0": ts1, "image1": ts2}
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidences = correspondences['confidence'].cpu().numpy()
    
    # Create fake DMatch objects to be compatible with classical result format
    good_matches = []
    kp_source = []
    kp_reference = []
    for i, (p0, p1, conf) in enumerate(zip(mkpts0, mkpts1, confidences)):
        kp_source.append(cv2.KeyPoint(float(p0[0]), float(p0[1]), 1.0))
        kp_reference.append(cv2.KeyPoint(float(p1[0]), float(p1[1]), 1.0))
        # use (1 - conf) as distance so higher confidence is lower distance
        good_matches.append(cv2.DMatch(i, i, float(1.0 - conf)))
        
    return MatchResult(
        keypoints_source=kp_source,
        keypoints_reference=kp_reference,
        good_matches=good_matches,
        source_points=mkpts0,
        reference_points=mkpts1,
        descriptor="LoFTR",
        detector_requested="LoFTR",
        detector_used="LoFTR",
        fallback_triggered=False,
        match_confidences=confidences.tolist(),
        diagnostics={
            "acceleration": "GPU_CUDA" if is_cuda_available() else "CPU",
            "acceleration_device": "NVIDIA GPU" if is_cuda_available() else "CPU",
            "raw_knn_matches": len(good_matches),
            "ratio_passed": len(good_matches),
            "mutual_consistent": len(good_matches),
            "unique_matches": len(good_matches)
        }
    )


def compute_features(gray: np.ndarray, detector_name: str = "sift", max_features: int = 8000) -> FeatureResult:
    detector = create_detector(detector_name, max_features=max_features)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return FeatureResult(np.empty((0, 2), dtype=np.float32), None, detector_name.upper(), [])
    pts = np.float32([item.pt for item in keypoints])
    return FeatureResult(pts, descriptors, detector_name.upper(), list(keypoints))


def match_feature_artifacts(
    source_features: FeatureResult,
    reference_features: FeatureResult,
    detector_name: str = "sift",
    ratio: float = 0.72,
    mutual_check: bool = True,
    image_shape_source: tuple[int, int] | None = None,
    image_shape_reference: tuple[int, int] | None = None,
) -> MatchResult:
    des_s = source_features.descriptors
    des_r = reference_features.descriptors

    diagnostics = {
        "candidate_keypoints_source": len(source_features.points),
        "candidate_keypoints_reference": len(reference_features.points),
        "raw_knn_matches": 0,
        "ratio_passed": 0,
        "mutual_consistent": 0,
        "unique_matches": 0,
        "valid_coordinates": 0,
    }

    if des_s is None or des_r is None or len(des_s) < 2 or len(des_r) < 2:
        raise ValueError(
            f"Insufficient descriptors found: source={len(source_features.points)}, "
            f"reference={len(reference_features.points)}. Need at least 2 features per image."
        )

    # GPU-accelerated matching with seamless CPU fallback
    good = None
    if is_cuda_available():
        try:
            metric = "hamming" if "ORB" in detector_name.upper() else "euclidean"
            gpu_matches, gpu_diag = match_descriptors_gpu(
                des_s, des_r, ratio=ratio, mutual_check=mutual_check, metric=metric
            )
            diagnostics["acceleration"] = gpu_diag.get("acceleration", "GPU_CUDA")
            diagnostics["acceleration_device"] = gpu_diag.get("device", "NVIDIA GPU")
            diagnostics["matching_time_ms"] = gpu_diag.get("execution_time_ms")
            diagnostics["raw_knn_matches"] = gpu_diag.get("raw_candidates_src", len(des_s))
            diagnostics["ratio_passed"] = len(gpu_matches)
            diagnostics["mutual_consistent"] = len(gpu_matches)
            diagnostics["unique_matches"] = len(gpu_matches)
            good = [cv2.DMatch(int(s), int(r), float(d)) for s, r, d in gpu_matches]
        except Exception:
            good = None

    if good is None:
        norm = cv2.NORM_HAMMING if "ORB" in detector_name.upper() else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)

        # 1. Forward matching (source -> reference) with ratio test
        knn_forward = matcher.knnMatch(des_s, des_r, k=2)
        diagnostics["raw_knn_matches"] = len(knn_forward)

        forward_good = {}
        for pair in knn_forward:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                forward_good[m.queryIdx] = m

        diagnostics["ratio_passed"] = len(forward_good)

        # 2. Mutual consistency check (cross-check)
        if mutual_check and len(des_s) >= 2 and len(des_r) >= 2:
            knn_backward = matcher.knnMatch(des_r, des_s, k=2)
            backward_good = {}
            for pair in knn_backward:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio * n.distance:
                    backward_good[m.queryIdx] = m

            mutual_matches = []
            for q_idx, m_fwd in forward_good.items():
                t_idx = m_fwd.trainIdx
                if t_idx in backward_good and backward_good[t_idx].trainIdx == q_idx:
                    mutual_matches.append(m_fwd)

            diagnostics["mutual_consistent"] = len(mutual_matches)

            # If mutual consistency leaves at least 6 matches, use it;
            # otherwise gracefully fall back to unidirectional ratio test
            if len(mutual_matches) >= 6:
                candidates = mutual_matches
            else:
                candidates = list(forward_good.values())
        else:
            candidates = list(forward_good.values())

        # 3. Duplicate reference match suppression (keep match with smallest descriptor distance)
        unique_by_target = {}
        for m in candidates:
            if m.trainIdx not in unique_by_target or m.distance < unique_by_target[m.trainIdx].distance:
                unique_by_target[m.trainIdx] = m

        good = list(unique_by_target.values())
        diagnostics["unique_matches"] = len(good)
        diagnostics["acceleration"] = "CPU_FALLBACK"
        diagnostics["acceleration_device"] = "CPU"

    # 4. Coordinate validation (reject NaN, Inf, and out-of-bounds points)
    valid_good = []
    src_pts_list = []
    ref_pts_list = []
    confidences = []
    is_binary = "ORB" in detector_name.upper()
    for m in good:
        sp = source_features.points[m.queryIdx]
        rp = reference_features.points[m.trainIdx]
        if not np.isfinite(sp).all() or not np.isfinite(rp).all():
            continue
        if image_shape_source is not None:
            sh, sw = image_shape_source[:2]
            if sp[0] < 0 or sp[0] >= sw or sp[1] < 0 or sp[1] >= sh:
                continue
        if image_shape_reference is not None:
            rh, rw = image_shape_reference[:2]
            if rp[0] < 0 or rp[0] >= rw or rp[1] < 0 or rp[1] >= rh:
                continue
        valid_good.append(m)
        src_pts_list.append(sp)
        ref_pts_list.append(rp)
        # Real deterministic match confidence from descriptor distance
        if is_binary:
            conf = float(np.clip(1.0 - (m.distance / 128.0), 0.05, 0.99))
        else:
            conf = float(np.clip(1.0 - (m.distance / 350.0), 0.05, 0.99))
        confidences.append(round(conf, 4))

    diagnostics["valid_coordinates"] = len(valid_good)

    if len(valid_good) < 4:
        raise ValueError(
            f"Only {len(valid_good)} reliable matches found after filtering "
            f"(ratio={ratio}, candidates_src={diagnostics['candidate_keypoints_source']}, "
            f"candidates_ref={diagnostics['candidate_keypoints_reference']}). "
            "Minimum 4 matches required for geometric model estimation."
        )

    src_pts = np.asarray(src_pts_list, dtype=np.float32).reshape(-1, 1, 2)
    ref_pts = np.asarray(ref_pts_list, dtype=np.float32).reshape(-1, 1, 2)

    return MatchResult(
        keypoints_source=[],
        keypoints_reference=[],
        good_matches=valid_good,
        source_points=src_pts,
        reference_points=ref_pts,
        descriptor=detector_name.upper(),
        detector_requested=detector_name.upper(),
        detector_used=detector_name.upper(),
        fallback_triggered=False,
        match_confidences=confidences,
        diagnostics=diagnostics,
    )


def match_features(
    source_gray: np.ndarray,
    reference_gray: np.ndarray,
    detector_name: str = "sift",
    ratio: float = 0.72,
    max_features: int = 8000,
    enable_fallback: bool = True,
) -> MatchResult:
    """Robust feature matching with detector fallback strategy.

    If the requested detector yields insufficient matches (< 8), automatically
    attempts secondary detectors (SIFT -> ORB or ORB -> SIFT) and records
    which detector actually succeeded.
    """
    if detector_name.lower() == "loftr":
        try:
            match_res = match_learned_loftr(source_gray, reference_gray)
            match_res.diagnostics["attempts"] = [{"detector": "loftr", "matches": len(match_res.good_matches), "status": "success"}]
            return match_res
        except Exception as e:
            if not enable_fallback:
                raise ValueError(f"LoFTR matching failed: {e}")
            # If fallback is enabled, fall through to classical detectors
            detectors_to_try = ["sift", "orb"]
    else:
        detectors_to_try = [detector_name.lower()]
        if enable_fallback:
            fallback_order = ["sift", "orb"]
            for d in fallback_order:
                if d != detector_name.lower() and d not in detectors_to_try:
                    detectors_to_try.append(d)

    last_error = None
    attempts = []

    for idx, d_name in enumerate(detectors_to_try):
        try:
            feat_s = compute_features(source_gray, d_name, max_features)
            feat_r = compute_features(reference_gray, d_name, max_features)
            match_res = match_feature_artifacts(
                feat_s,
                feat_r,
                detector_name=d_name,
                ratio=ratio,
                mutual_check=True,
                image_shape_source=source_gray.shape,
                image_shape_reference=reference_gray.shape,
            )
            match_res.detector_requested = detector_name.upper()
            match_res.detector_used = d_name.upper()
            match_res.fallback_triggered = (idx > 0)
            attempts.append({"detector": d_name, "matches": len(match_res.good_matches), "status": "success"})
            match_res.diagnostics["attempts"] = attempts

            # If primary detector found plenty of matches, return immediately
            if len(match_res.good_matches) >= 8 or idx == len(detectors_to_try) - 1:
                return match_res

            # If matches were marginal (< 8) and there is a fallback detector, save as candidate
            candidate_match = match_res
        except ValueError as err:
            last_error = err
            attempts.append({"detector": d_name, "matches": 0, "status": str(err)})
            continue

    if "candidate_match" in locals():
        candidate_match.diagnostics["attempts"] = attempts
        return candidate_match

    raise ValueError(
        f"Feature matching failed for all attempted detectors ({', '.join(detectors_to_try)}). "
        f"Last error: {last_error}"
    )


def spatially_distribute_matches(
    match_result: MatchResult,
    image_shape,
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_per_cell: int = 30,
    min_point_spacing: float = 3.0,
) -> MatchResult:
    """Select a spatially well-distributed, non-clustered subset of correspondences.

    Filters duplicate or overly clustered matches within local neighborhoods while
    preserving strong matches and broad spatial coverage. Attaches comprehensive
    spatial diagnostics.
    """
    from app.services.metrics import evaluate_spatial_distribution

    h, w = image_shape[:2]
    src = match_result.source_points.reshape(-1, 2)
    raw_count = len(src)

    if raw_count < 4:
        match_result.diagnostics.update({
            "raw_match_count": raw_count,
            "filtered_match_count": raw_count,
            "final_match_count": raw_count,
            "spatial_coverage": 0.0,
            "spatial_uniformity": 0.0,
            "grid_occupancy": 0.0,
            "distribution_category": "LIMITED_DISTRIBUTION",
        })
        return match_result

    # If match count is already small (<= 12), preserve all valid points without pruning
    if raw_count <= 12:
        spatial = evaluate_spatial_distribution(src, w, h, rows=grid_rows, cols=grid_cols)
        match_result.diagnostics.update({
            "raw_match_count": raw_count,
            "filtered_match_count": raw_count,
            "final_match_count": raw_count,
            "spatial_coverage": spatial["spatial_coverage"],
            "spatial_uniformity": spatial["spatial_uniformity"],
            "grid_occupancy": spatial["grid_occupancy"],
            "distribution_category": spatial["distribution_category"],
            "spatial_distribution": spatial,
        })
        return match_result

    # 1. Bucket by grid cell
    cells: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(src):
        c = min(grid_cols - 1, max(0, int(x / max(w, 1) * grid_cols)))
        r = min(grid_rows - 1, max(0, int(y / max(h, 1) * grid_rows)))
        cells.setdefault((r, c), []).append(idx)

    # 2. Select top matches per cell sorted deterministically by descriptor match distance
    keep: list[int] = []
    for _, indices in cells.items():
        sorted_indices = sorted(
            indices,
            key=lambda i: (match_result.good_matches[i].distance, i)
        )
        keep.extend(sorted_indices[:max_per_cell])

    keep = sorted(set(keep))

    # Guard: if spatial filtering was too aggressive, fall back to best matches
    if len(keep) < 4:
        all_sorted = sorted(range(raw_count), key=lambda i: (match_result.good_matches[i].distance, i))
        keep = sorted(all_sorted[:min(raw_count, max_per_cell * 4)])

    filtered_count = len(keep)
    final_src = match_result.source_points[keep]
    final_ref = match_result.reference_points[keep]
    final_matches = [match_result.good_matches[i] for i in keep]
    final_confidences = (
        [match_result.match_confidences[i] for i in keep]
        if match_result.match_confidences and len(match_result.match_confidences) >= raw_count
        else []
    )

    spatial = evaluate_spatial_distribution(final_src.reshape(-1, 2), w, h, rows=grid_rows, cols=grid_cols)

    diagnostics = dict(match_result.diagnostics)
    diagnostics.update({
        "raw_match_count": raw_count,
        "filtered_match_count": filtered_count,
        "final_match_count": filtered_count,
        "spatial_coverage": spatial["spatial_coverage"],
        "spatial_uniformity": spatial["spatial_uniformity"],
        "grid_occupancy": spatial["grid_occupancy"],
        "distribution_category": spatial["distribution_category"],
        "spatial_distribution": spatial,
    })

    return MatchResult(
        keypoints_source=match_result.keypoints_source,
        keypoints_reference=match_result.keypoints_reference,
        good_matches=final_matches,
        source_points=final_src,
        reference_points=final_ref,
        descriptor=match_result.descriptor,
        detector_requested=match_result.detector_requested,
        detector_used=match_result.detector_used,
        fallback_triggered=match_result.fallback_triggered,
        match_confidences=final_confidences,
        diagnostics=diagnostics,
    )

