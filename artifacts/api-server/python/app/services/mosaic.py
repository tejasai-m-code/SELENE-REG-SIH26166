"""Relative image-space mosaic construction with multiple blending modes and quality metrics."""

import cv2
import numpy as np

from app.services.gpu_accelerator import is_cuda_available, warp_perspective_gpu


class MosaicCanvasTooLargeError(ValueError):
    """Structured exception raised when the computed global mosaic canvas exceeds safety limits."""

    def __init__(
        self,
        width: int,
        height: int,
        images_count: int,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        max_dimension: int,
        max_total_pixels: int,
    ):
        est_mem_mb = round((width * height * 3 * 4 * 2) / (1024 * 1024), 2)
        self.code = "MOSAIC_CANVAS_TOO_LARGE"
        self.requested_width = width
        self.requested_height = height
        self.estimated_memory_mb = est_mem_mb
        self.images_count = images_count
        self.current_bounds = {"min_x": int(min_x), "min_y": int(min_y), "max_x": int(max_x), "max_y": int(max_y)}
        self.configured_safety_limit = {"max_dimension": int(max_dimension), "max_total_pixels": int(max_total_pixels)}
        self.actionable_recommendation = (
            f"The combined spatial extent of the {images_count} placed frames requires a canvas of "
            f"{width}x{height} (~{est_mem_mb} MB memory), exceeding the configured limit of {max_dimension}px. "
            "Consider subsetting the input images to a tighter geographic strip or increasing max_dimension in settings."
        )
        msg = f"[MOSAIC_CANVAS_TOO_LARGE] Canvas dimensions ({width}x{height}) exceed safety limits. {self.actionable_recommendation}"
        super().__init__(msg)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Cannot convert None image.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _compute_distance_weight(mask: np.ndarray) -> np.ndarray:
    """Compute exact Euclidean distance transform from boundary for distance-weighted blending."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d > 0:
        dist /= max_d
    return dist.astype(np.float32)


def compute_valid_data_mask(image: np.ndarray) -> np.ndarray:
    """Compute binary mask of genuine valid data pixels (255=valid, 0=invalid/padding).

    Distinguishes genuine image data from zero-padding, nodata collars, and transparent alpha.
    Preserves genuine interior dark features (e.g. crater shadows).
    """
    if image is None or image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    h, w = image.shape[:2]

    # 1. 4-channel image with Alpha channel
    if image.ndim == 3 and image.shape[2] == 4:
        return (image[:, :, 3] > 0).astype(np.uint8) * 255

    # 2. Check for outer nodata collar / zero padding
    if image.ndim == 3:
        is_zero = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
    else:
        is_zero = (image == 0)

    # If the image is completely zero, return all zeros (no valid pixels)
    if is_zero.all():
        return np.zeros((h, w), dtype=np.uint8)

    # Check if any border pixel is zero
    has_border_zeros = bool(
        is_zero[0, :].any() or is_zero[-1, :].any() or
        is_zero[:, 0].any() or is_zero[:, -1].any()
    )

    if not has_border_zeros:
        return np.full((h, w), 255, dtype=np.uint8)

    # Connected components of zero pixels to identify outer boundary padding
    num_labels, labels = cv2.connectedComponents(is_zero.astype(np.uint8), connectivity=4)

    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)  # 0 represents non-zero foreground

    if border_labels:
        is_padding = np.isin(labels, list(border_labels))
        return (~is_padding).astype(np.uint8) * 255

    return np.full((h, w), 255, dtype=np.uint8)


def _compute_feather_weight(mask: np.ndarray, blur_sigma: float = 8.0) -> np.ndarray:
    """Compute Gaussian-blurred feathering weight inside the valid mask region."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)
    feather = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), blur_sigma)
    feather *= (mask > 0)
    return feather


def evaluate_mosaic_quality(
    mosaic: np.ndarray,
    occupancy_map: np.ndarray,
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    width: int,
    height: int,
    placed_count: int,
) -> dict:
    """Measure objective scientific quality metrics for the generated mosaic.

    Metrics include:
    - valid mosaic area (pixels and fraction)
    - overlap area (pixels and fraction)
    - overlap consistency (PSNR / mean pixel difference in overlap zones)
    - seam intensity discontinuity
    - overall quality assessment
    """
    total_canvas_pixels = width * height
    valid_mask = occupancy_map > 0
    valid_area_pixels = int(np.sum(valid_mask))
    coverage_fraction = float(valid_area_pixels / max(total_canvas_pixels, 1))

    overlap_mask = occupancy_map >= 2
    overlap_area_pixels = int(np.sum(overlap_mask))
    overlap_fraction = float(overlap_area_pixels / max(valid_area_pixels, 1))

    # Overlap seam and consistency evaluation
    overlap_psnr = None
    mean_overlap_diff = None
    seam_discontinuity = None

    if overlap_area_pixels > 20 and len(warped_images) >= 2:
        diffs = []
        for i in range(len(warped_images)):
            for j in range(i + 1, len(warped_images)):
                mutual = (warped_masks[i] > 0) & (warped_masks[j] > 0)
                if np.sum(mutual) > 20:
                    img_i = warped_images[i].astype(np.float32)
                    img_j = warped_images[j].astype(np.float32)
                    diff = np.abs(img_i[mutual] - img_j[mutual])
                    diffs.append(np.mean(diff))

        if diffs:
            mean_overlap_diff = float(np.mean(diffs))
            mse = float(np.mean([d ** 2 for d in diffs]))
            overlap_psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-6)))

    # Edge gradient discontinuity on mosaic
    if valid_area_pixels > 100:
        gray_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY) if mosaic.ndim == 3 else mosaic
        sobel_x = cv2.Sobel(gray_mosaic, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_mosaic, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        if np.any(overlap_mask):
            seam_discontinuity = float(np.mean(grad_mag[overlap_mask]))

    # Quality category rating
    if placed_count >= 2 and overlap_fraction > 0.05:
        if overlap_psnr is not None and overlap_psnr > 25.0:
            quality_label = "EXCELLENT"
        elif overlap_psnr is not None and overlap_psnr > 18.0:
            quality_label = "GOOD"
        else:
            quality_label = "ACCEPTABLE"
    elif placed_count >= 1:
        quality_label = "ACCEPTABLE"
    else:
        quality_label = "DEGRADED"

    return {
        "width": width,
        "height": height,
        "total_canvas_pixels": total_canvas_pixels,
        "valid_mosaic_area_pixels": valid_area_pixels,
        "coverage_fraction": round(coverage_fraction, 4),
        "overlap_area_pixels": overlap_area_pixels,
        "overlap_fraction": round(overlap_fraction, 4),
        "contributing_images": placed_count,
        "mean_overlap_pixel_difference": round(mean_overlap_diff, 3) if mean_overlap_diff is not None else None,
        "overlap_consistency_psnr_db": round(overlap_psnr, 2) if overlap_psnr is not None else None,
        "seam_discontinuity_gradient": round(seam_discontinuity, 3) if seam_discontinuity is not None else None,
        "quality_label": quality_label,
    }


def build_relative_mosaic(
    images: list[np.ndarray],
    transforms: dict[str, list],
    blending_mode: str = "distance_weighted",
    max_dimension: int = 12000,
    max_total_pixels: int = 50_000_000,
    pad: int = 0,
) -> tuple[np.ndarray, dict]:
    """Warp placed images into a common reference canvas with seamless multi-band blending.

    Supported blending modes:
    - 'distance_weighted': Optimal linear distance-transform feathering
    - 'feather': Gaussian-blurred boundary feathering
    - 'average': Simple arithmetic average of overlapping rasters
    - 'overlay': Deterministic top-down layering
    """
    placed = sorted(int(key) for key in transforms)
    if not placed:
        raise ValueError("No connected images are available for mosaic generation.")

    for idx in placed:
        if idx < 0 or idx >= len(images) or images[idx] is None or images[idx].size == 0:
            raise ValueError(f"Image {idx} is invalid or missing.")

    matrices = {}
    for index in placed:
        M = np.asarray(transforms[str(index)], dtype=np.float64)
        if not np.isfinite(M).all() or abs(M[2, 2]) < 1e-12:
            raise ValueError(f"Transformation matrix for image {index} contains NaN or invalid scale.")
        matrices[index] = M

    # 1. Compute bounding box across all projected corners
    all_corners = []
    footprints = []
    for index in placed:
        h, w = images[index].shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Image {index} has invalid dimensions: {w}x{h}.")
        corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        try:
            warped = cv2.perspectiveTransform(corners, matrices[index].astype(np.float32))[0]
            if not np.isfinite(warped).all():
                raise ValueError(f"Transformed corners for image {index} contain non-finite coordinates.")
            all_corners.append(warped)
            footprints.append({"image_index": index, "corners": warped.tolist()})
        except cv2.error as err:
            raise ValueError(f"Failed to project corners for image {index}: {err}")

    all_corners_arr = np.vstack(all_corners)
    min_x, min_y = np.floor(all_corners_arr.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners_arr.max(axis=0)).astype(int)

    pad = max(0, int(pad))
    width = max(1, int(max_x - min_x) + 2 * pad)
    height = max(1, int(max_y - min_y) + 2 * pad)

    if width < 1 or height < 1:
        raise ValueError("Computed mosaic canvas has non-positive dimensions.")
    if width > max_dimension or height > max_dimension or (width * height) > max_total_pixels:
        raise MosaicCanvasTooLargeError(
            width=width,
            height=height,
            images_count=len(placed),
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            max_dimension=max_dimension,
            max_total_pixels=max_total_pixels,
        )

    # 2. Shift translation to map (min_x, min_y) to (pad, pad)
    translation = np.array(
        [[1.0, 0.0, float(-min_x + pad)], [0.0, 1.0, float(-min_y + pad)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    occupancy_map = np.zeros((height, width), dtype=np.uint8)
    overlay_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    warped_images = []
    warped_masks = []
    translated_footprints = []
    mosaic_transforms = {}

    for index in placed:
        raw_image = images[index]
        image = _to_bgr(raw_image)
        H_canvas = translation @ matrices[index]
        H_canvas /= H_canvas[2, 2]
        mosaic_transforms[str(index)] = H_canvas.tolist()

        # Compute independent valid data mask (excluding padding/collar)
        source_mask = compute_valid_data_mask(raw_image)
        if not np.any(source_mask):
            continue

        if is_cuda_available() and (width * height) <= 25_000_000:
            warped = warp_perspective_gpu(image, H_canvas, (width, height))
        else:
            warped = cv2.warpPerspective(image, H_canvas, (width, height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(source_mask, H_canvas, (width, height), flags=cv2.INTER_NEAREST)


        # Boundary anti-seam check: cv2.warpPerspective with INTER_LINEAR interpolates with canvas zeros at border.
        # Mask out boundary interpolation pixels so black canvas padding never bleeds dark seams into mosaic.
        mask_linear = cv2.warpPerspective((source_mask > 0).astype(np.float32), H_canvas, (width, height), flags=cv2.INTER_LINEAR)
        binary_mask = ((mask > 0) & (mask_linear >= 0.4)).astype(np.uint8)

        if not np.any(binary_mask):
            continue

        occupancy_map += binary_mask
        warped_images.append(warped)
        warped_masks.append(binary_mask)

        # 3. Blending computation
        valid_bool = binary_mask > 0
        if blending_mode == "overlay":
            # True painter's algorithm: write where valid; existing pixels outside valid_bool are untouched
            overlay_canvas[valid_bool] = warped[valid_bool]
            weights[valid_bool] = 1.0
        elif blending_mode == "distance_weighted":
            w_img = _compute_distance_weight(binary_mask)
            accumulator += warped.astype(np.float32) * w_img[..., None]
            weights += w_img
        elif blending_mode == "feather":
            w_img = _compute_feather_weight(binary_mask, blur_sigma=8.0)
            accumulator += warped.astype(np.float32) * w_img[..., None]
            weights += w_img
        elif blending_mode == "average":
            w_img = binary_mask.astype(np.float32)
            accumulator += warped.astype(np.float32) * w_img[..., None]
            weights += w_img
        else:
            w_img = _compute_distance_weight(binary_mask)
            accumulator += warped.astype(np.float32) * w_img[..., None]
            weights += w_img

        h, w = image.shape[:2]
        corners_c = cv2.perspectiveTransform(
            np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]),
            H_canvas.astype(np.float32)
        )[0]
        translated_footprints.append({
            "image_index": index,
            "corners": corners_c.tolist(),
            "vertices": corners_c.tolist(),
        })

    # 4. Normalize canvas by weights
    if blending_mode == "overlay":
        mosaic = overlay_canvas
    else:
        mosaic = (accumulator / np.maximum(weights[..., None], 1e-6)).clip(0, 255).astype(np.uint8)


    # 5. Measure quality metrics
    quality = evaluate_mosaic_quality(
        mosaic,
        occupancy_map,
        warped_images,
        warped_masks,
        width,
        height,
        len(placed),
    )

    # 6. Structured Footprints
    from app.services.footprint import PolygonFootprint, create_mosaic_footprint
    structured_fps = []
    for tf in translated_footprints:
        structured_fps.append(
            PolygonFootprint(
                vertices=np.array(tf["corners"], dtype=np.float64),
                coordinate_frame="MOSAIC_CANVAS_PIXEL",
                source_id=str(tf["image_index"]),
            )
        )
    mosaic_fp = create_mosaic_footprint(structured_fps, (height, width))

    info = {
        "width": width,
        "height": height,
        "placed_image_count": len(placed),
        "blending_mode": blending_mode,
        "footprints": translated_footprints,
        "global_bounds": {
            "min_x": int(min_x),
            "min_y": int(min_y),
            "max_x": int(max_x),
            "max_y": int(max_y),
            "pad": int(pad),
        },
        "mosaic_transforms": mosaic_transforms,
        "mosaic_footprint": mosaic_fp,
        "quality_metrics": quality,
        "coverage_fraction": quality["coverage_fraction"],
        "overlap_fraction": quality["overlap_fraction"],
        "overlap_psnr_db": quality["overlap_consistency_psnr_db"],
        "quality_label": quality["quality_label"],
        "coordinate_system": "relative_image_space",
        "label": "Relative Registered Lunar Mosaic",
    }

    return mosaic, info
