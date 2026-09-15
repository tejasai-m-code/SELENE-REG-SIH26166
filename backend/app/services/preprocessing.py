import cv2
import numpy as np

from app.utils.image_utils import to_gray


def percentile_normalize(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    valid = np.isfinite(gray)
    if not valid.any():
        return np.zeros_like(gray, dtype=np.uint8)

    values = gray[valid]
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)

    out = np.clip((gray - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def preprocess(image: np.ndarray, illumination_normalization: bool = True) -> np.ndarray:
    gray = to_gray(image)

    if illumination_normalization:
        # CLAHE improves local contrast under different illumination.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(percentile_normalize(gray))
        # Mild high-pass style normalization without destroying crater edges.
        background = cv2.GaussianBlur(gray, (0, 0), 9)
        detail = cv2.addWeighted(gray, 1.5, background, -0.5, 0)
        gray = np.clip(detail, 0, 255).astype(np.uint8)
    else:
        gray = percentile_normalize(gray)

    return gray


def build_pyramid(gray: np.ndarray, levels: int = 1):
    levels = max(1, min(int(levels), 4))
    pyramid = [gray]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid
