"""Planetary image footprint geometry, transformation, and overlap analysis."""

from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class PolygonFootprint:
    vertices: np.ndarray  # Shape: (N, 2)
    coordinate_frame: str = "IMAGE_PIXEL"
    source_id: str = "0"
    properties: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 2)

    @property
    def area(self) -> float:
        """Compute polygon area via the Shoelace formula."""
        if len(self.vertices) < 3 or not np.isfinite(self.vertices).all():
            return 0.0
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y)."""
        if len(self.vertices) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 1])),
        )

    def is_valid(self) -> bool:
        """Validate polygon vertices against non-finite values, degeneracy, and minimum area."""
        if len(self.vertices) < 3:
            return False
        if not np.isfinite(self.vertices).all():
            return False
        if self.area < 1e-4:
            return False
        return True

    def transform(self, matrix: np.ndarray, target_frame: str, densify_samples: int = 8) -> "PolygonFootprint":
        """Transform footprint polygon into a new coordinate frame.

        Edge-densification samples intermediate points along polygon edges before
        transformation to guarantee accurate boundary projection under non-linear
        or projective homographies.
        """
        if not self.is_valid():
            raise ValueError("Cannot transform an invalid or degenerate footprint.")

        M = np.asarray(matrix, dtype=np.float64)
        if M.shape not in ((3, 3), (2, 3)) or not np.isfinite(M).all():
            raise ValueError("Transformation matrix must be a finite 3x3 or 2x3 array.")

        # Densify polygon edges
        densified_pts = []
        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            for t in np.linspace(0.0, 1.0, densify_samples, endpoint=False):
                densified_pts.append((1.0 - t) * p1 + t * p2)

        pts_arr = np.array(densified_pts, dtype=np.float32).reshape(-1, 1, 2)

        if M.shape == (3, 3):
            warped = cv2.perspectiveTransform(pts_arr, M).reshape(-1, 2)
        else:
            warped = cv2.transform(pts_arr, M).reshape(-1, 2)

        return PolygonFootprint(
            vertices=warped,
            coordinate_frame=target_frame,
            source_id=self.source_id,
            properties=dict(self.properties),
        )

    def intersect(self, other: "PolygonFootprint") -> "PolygonFootprint":
        """Compute spatial intersection with another footprint in the same coordinate frame."""
        if self.coordinate_frame != other.coordinate_frame:
            raise ValueError(
                f"Coordinate frame mismatch: '{self.coordinate_frame}' vs '{other.coordinate_frame}'. "
                "Footprints must be in the same coordinate frame for polygon operations."
            )
        if not self.is_valid() or not other.is_valid():
            return PolygonFootprint(np.empty((0, 2)), coordinate_frame=self.coordinate_frame, source_id="intersection")

        # Use OpenCV convex intersection
        p1 = self.vertices.astype(np.float32)
        p2 = other.vertices.astype(np.float32)

        # Simplify to convex hull if needed
        h1 = cv2.convexHull(p1)
        h2 = cv2.convexHull(p2)

        try:
            area, intersect_pts = cv2.intersectConvexConvex(h1, h2)
            if area > 0 and intersect_pts is not None and len(intersect_pts) >= 3:
                return PolygonFootprint(
                    vertices=intersect_pts.reshape(-1, 2),
                    coordinate_frame=self.coordinate_frame,
                    source_id=f"intersect_{self.source_id}_{other.source_id}",
                    properties={"intersection_area": float(area)},
                )
        except cv2.error:
            pass

        return PolygonFootprint(np.empty((0, 2)), coordinate_frame=self.coordinate_frame, source_id="intersection")

    def overlap_metrics(self, other: "PolygonFootprint") -> dict:
        """Compute intersection area, union area, and IoU overlap ratios."""
        intersection = self.intersect(other)
        int_area = intersection.area
        area_a = self.area
        area_b = other.area
        union_area = area_a + area_b - int_area

        iou = float(int_area / union_area) if union_area > 0 else 0.0
        cov_a = float(int_area / area_a) if area_a > 0 else 0.0
        cov_b = float(int_area / area_b) if area_b > 0 else 0.0

        return {
            "intersection_area": round(int_area, 4),
            "union_area": round(union_area, 4),
            "area_source_a": round(area_a, 4),
            "area_source_b": round(area_b, 4),
            "iou": round(iou, 4),
            "overlap_ratio_a": round(cov_a, 4),
            "overlap_ratio_b": round(cov_b, 4),
            "coordinate_frame": self.coordinate_frame,
            "has_overlap": int_area > 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "source_id": str(self.source_id),
            "coordinate_frame": self.coordinate_frame,
            "vertices": [[round(float(x), 3), round(float(y), 3)] for x, y in self.vertices],
            "vertex_count": len(self.vertices),
            "area": round(self.area, 3),
            "bounding_box": [round(v, 3) for v in self.bounding_box],
            "is_valid": self.is_valid(),
        }


def create_image_footprint(image_shape: tuple[int, int], source_id: str | int = 0) -> PolygonFootprint:
    """Create a rectangular image-space footprint from raster dimensions."""
    h, w = image_shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}.")
    corners = np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=np.float64)
    return PolygonFootprint(
        vertices=corners,
        coordinate_frame="IMAGE_PIXEL",
        source_id=str(source_id),
        properties={"width": w, "height": h},
    )


def create_mosaic_footprint(
    contributing_footprints: list[PolygonFootprint],
    canvas_shape: tuple[int, int],
) -> dict:
    """Compute the multi-image mosaic footprint, distinguishing valid data area from bounding canvas."""
    h, w = canvas_shape[:2]
    canvas_area = float(w * h)

    valid_fps = [fp for fp in contributing_footprints if fp.is_valid()]
    if not valid_fps:
        return {
            "canvas_width": w,
            "canvas_height": h,
            "canvas_area_pixels": canvas_area,
            "valid_footprint_area_pixels": 0.0,
            "footprint_to_canvas_ratio": 0.0,
            "contributing_footprint_count": 0,
            "footprints": [],
            "coordinate_frame": "MOSAIC_CANVAS_PIXEL",
        }

    # Combined convex hull of all transformed footprint vertices
    all_vertices = np.vstack([fp.vertices for fp in valid_fps])
    try:
        hull = cv2.convexHull(all_vertices.astype(np.float32)).reshape(-1, 2)
        valid_area = PolygonFootprint(hull, coordinate_frame="MOSAIC_CANVAS_PIXEL").area
    except cv2.error:
        valid_area = sum(fp.area for fp in valid_fps)

    ratio = float(valid_area / canvas_area) if canvas_area > 0 else 0.0

    return {
        "canvas_width": w,
        "canvas_height": h,
        "canvas_area_pixels": round(canvas_area, 2),
        "valid_footprint_area_pixels": round(valid_area, 2),
        "footprint_to_canvas_ratio": round(min(1.0, ratio), 4),
        "contributing_footprint_count": len(valid_fps),
        "footprints": [fp.to_dict() for fp in valid_fps],
        "hull_vertices": [[round(float(x), 3), round(float(y), 3)] for x, y in hull] if 'hull' in locals() else [],
        "coordinate_frame": "MOSAIC_CANVAS_PIXEL",
    }
