from pathlib import Path
import cv2
import numpy as np


ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


def decode_upload(data: bytes, filename: str = "") -> np.ndarray:
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            "For ISRO/LROC scientific products, export/convert the image to GeoTIFF/TIFF/PNG/JPEG "
            "or add a mission-specific PDS reader before ingestion."
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    # Convert common 16-bit/float data to displayable uint8 while preserving relative contrast.
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)

    return image


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[2] > 1:
        channels = []
        for c in range(x.shape[2]):
            channels.append(normalize_to_uint8(x[..., c]))
        return np.stack(channels, axis=2)

    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)

    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)

    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not save output image: {path}")


def resize_for_preview(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
