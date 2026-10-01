"""Relative image-space mosaic construction with feathered weighted blending."""

import cv2
import numpy as np


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def build_relative_mosaic(images: list[np.ndarray], transforms: dict[str, list]) -> tuple[np.ndarray, dict]:
    """Warp placed images to a fitting canvas and blend with distance weights."""
    placed = sorted(int(key) for key in transforms)
    if not placed:
        raise ValueError("No connected images are available for mosaic generation.")

    matrices = {index: np.asarray(transforms[str(index)], dtype=np.float64) for index in placed}
    all_corners = []
    footprints = []
    for index in placed:
        h, w = images[index].shape[:2]
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        warped = cv2.perspectiveTransform(corners, matrices[index].astype(np.float32))[0]
        all_corners.append(warped)
        footprints.append({"image_index": index, "corners": warped.tolist()})
    all_corners = np.vstack(all_corners)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = int(max_x - min_x + 1), int(max_y - min_y + 1)
    if width < 1 or height < 1 or width > 12000 or height > 12000 or width * height > 50_000_000:
        raise ValueError("The relative mosaic canvas is too large for this local prototype.")

    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    translated_footprints = []
    for index in placed:
        image = _to_bgr(images[index])
        H = translation @ matrices[index]
        warped = cv2.warpPerspective(image, H, (width, height))
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(source_mask, H, (width, height))
        # A blurred binary footprint produces a lightweight feather at overlaps.
        feather = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 8)
        feather *= (mask > 0)
        accumulator += warped.astype(np.float32) * feather[..., None]
        weights += feather
        h, w = image.shape[:2]
        corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H.astype(np.float32))[0]
        translated_footprints.append({"image_index": index, "corners": corners.tolist()})

    mosaic = (accumulator / np.maximum(weights[..., None], 1e-6)).clip(0, 255).astype(np.uint8)
    return mosaic, {
        "width": width,
        "height": height,
        "placed_image_count": len(placed),
        "footprints": translated_footprints,
        "mosaic_transforms": {str(index): (translation @ matrices[index]).tolist() for index in placed},
        "coordinate_system": "relative_image_space",
        "label": "Relative Registered Lunar Mosaic",
    }
