"""Small disk-backed cache for reproducible multi-image registration artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.services.feature_matching import FeatureResult, compute_features
from app.services.preprocessing import preprocess

CACHE_VERSION = "phase2-v1"

def canonical_key(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()

def image_fingerprint(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = sha256(); digest.update(str(contiguous.shape).encode()); digest.update(str(contiguous.dtype).encode()); digest.update(contiguous.tobytes())
    return digest.hexdigest()

class RegistrationCache:
    def __init__(self, root: Path):
        self.root = root; self.features = root / "features"; self.pairs = root / "pairs"
        self.features.mkdir(parents=True, exist_ok=True); self.pairs.mkdir(parents=True, exist_ok=True)
        self.stats = {"image_hits": 0, "image_misses": 0, "feature_hits": 0, "feature_misses": 0, "pair_hits": 0, "pair_misses": 0}

    @staticmethod
    def feature_config(settings: dict) -> dict:
        return {"cache_version": CACHE_VERSION, "preprocessing_version": "clahe-percentile-detail-v1", "illumination_normalization": bool(settings.get("illumination_normalization", True)), "detector": str(settings.get("detector", "sift")).lower(), "max_features": int(settings.get("max_features", 8000)), "opencv": cv2.__version__}

    @staticmethod
    def pair_config(settings: dict, feature_config: dict) -> dict:
        return {"cache_version": CACHE_VERSION, "features": feature_config, "ratio": float(settings.get("ratio", 0.72)), "ransac_threshold": float(settings.get("ransac_threshold", 3.0)), "ecc_refinement": bool(settings.get("ecc_refinement", True)), "spatial_distribution": bool(settings.get("spatial_distribution", True))}

    def get_features(self, image: np.ndarray, settings: dict):
        fingerprint = image_fingerprint(image); config = self.feature_config(settings); key = canonical_key({"image": fingerprint, "config": config}); path = self.features / f"{key}.npz"
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    gray = data["gray"]; feature = FeatureResult(data["keypoints"].astype(np.float32), data["descriptors"], str(data["descriptor"].item()))
                self.stats["image_hits"] += 1; self.stats["feature_hits"] += 1
                return fingerprint, gray, feature, True, 0.0
            except (OSError, ValueError, KeyError): path.unlink(missing_ok=True)
        started = perf_counter(); gray = preprocess(image, config["illumination_normalization"]); feature = compute_features(gray, config["detector"], config["max_features"])
        np.savez_compressed(path, gray=gray, keypoints=feature.points, descriptors=feature.descriptors, descriptor=np.array(feature.descriptor))
        self.stats["image_misses"] += 1; self.stats["feature_misses"] += 1
        return fingerprint, gray, feature, False, perf_counter() - started

    def pair_key(self, source_fingerprint: str, reference_fingerprint: str, settings: dict) -> str:
        return canonical_key({"source": source_fingerprint, "reference": reference_fingerprint, "config": self.pair_config(settings, self.feature_config(settings))})

    def load_pair(self, key: str):
        path = self.pairs / f"{key}.npz"
        if not path.exists(): self.stats["pair_misses"] += 1; return None
        try:
            with np.load(path, allow_pickle=False) as data:
                result = json.loads(str(data["metadata"].item())); result.update(source_points=data["source_points"].astype(np.float32), reference_points=data["reference_points"].astype(np.float32), inlier_mask=data["inlier_mask"].astype(bool))
            self.stats["pair_hits"] += 1; return result
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            path.unlink(missing_ok=True); self.stats["pair_misses"] += 1; return None

    def save_pair(self, key: str, payload: dict) -> None:
        metadata = {k: v for k, v in payload.items() if k not in {"source_points", "reference_points", "inlier_mask"}}
        np.savez_compressed(self.pairs / f"{key}.npz", metadata=np.array(json.dumps(metadata, default=str)), source_points=payload["source_points"], reference_points=payload["reference_points"], inlier_mask=payload["inlier_mask"].astype(np.uint8))
