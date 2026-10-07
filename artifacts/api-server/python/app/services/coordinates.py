"""Coordinate systems, frame-tagged transformations, and planetary extension points."""

from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np


class CoordinateFrame(str, Enum):
    IMAGE_PIXEL = "IMAGE_PIXEL"
    REFERENCE_IMAGE_PIXEL = "REFERENCE_IMAGE_PIXEL"
    MOSAIC_CANVAS_PIXEL = "MOSAIC_CANVAS_PIXEL"
    PHYSICAL_LOCAL_METERS = "PHYSICAL_LOCAL_METERS"
    SELENOGRAPHIC_IAU2000 = "SELENOGRAPHIC_IAU2000"


@dataclass
class TransformRecord:
    source_frame: CoordinateFrame | str
    target_frame: CoordinateFrame | str
    matrix: np.ndarray
    units: str = "pixels"
    valid: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply the transformation to 2D coordinates."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        if self.matrix.shape == (3, 3):
            warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.matrix.astype(np.float32)).reshape(-1, 2)
        elif self.matrix.shape == (2, 3):
            warped = cv2.transform(pts.reshape(-1, 1, 2), self.matrix.astype(np.float32)).reshape(-1, 2)
        else:
            raise ValueError(f"Unsupported transform matrix shape: {self.matrix.shape}")

        return warped

    def invert(self) -> "TransformRecord":
        """Invert transformation and reverse source/target coordinate frames."""
        if self.matrix.shape == (3, 3):
            inv_mat = np.linalg.inv(self.matrix)
            inv_mat /= inv_mat[2, 2]
        elif self.matrix.shape == (2, 3):
            inv_mat = cv2.invertAffineTransform(self.matrix)
        else:
            raise ValueError("Matrix cannot be inverted.")

        return TransformRecord(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            matrix=inv_mat,
            units=self.units,
            valid=self.valid,
            metadata={**self.metadata, "inverted": True},
        )

    def to_dict(self) -> dict:
        return {
            "source_frame": str(self.source_frame),
            "target_frame": str(self.target_frame),
            "matrix": self.matrix.tolist(),
            "units": self.units,
            "valid": self.valid,
            "metadata": self.metadata,
        }


def pixel_to_physical_local(
    points: np.ndarray,
    gsd_m: float,
    origin_pixels: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Convert pixel coordinates to local physical distances in meters using known GSD."""
    if gsd_m <= 0:
        raise ValueError("GSD must be positive to compute physical coordinates.")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ox, oy = origin_pixels
    physical_m = np.empty_like(pts)
    physical_m[:, 0] = (pts[:, 0] - ox) * gsd_m
    physical_m[:, 1] = (pts[:, 1] - oy) * gsd_m
    return physical_m


# -------------------------------------------------------------------------
# Planetary Extension Interfaces (SPICE / DEM / Camera Model)
# -------------------------------------------------------------------------

class SPICEInterface:
    """Extension interface for NAIF SPICE kernel geometry."""

    STATUS = "INTERFACE_PREPARED_SPICE_UNAVAILABLE"
    SCIENTIFIC_NOTE = "SPICE ephemeris and pointing kernels not present in repository. Native SPICE calculations disabled."

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def get_spacecraft_geometry(cls, product_id: str, timestamp: str | None = None) -> dict:
        return {
            "status": cls.STATUS,
            "available": False,
            "scientific_note": cls.SCIENTIFIC_NOTE,
            "product_id": product_id,
        }


class DEMInterface:
    """Extension interface for Digital Elevation Models and terrain relief correction."""

    STATUS = "INTERFACE_PREPARED_DEM_UNAVAILABLE"
    SCIENTIFIC_NOTE = "Planetary DEM raster tiles not present in repository. Elevation-based relief displacement corrections disabled."

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def get_elevation_profile(cls, latitude: float, longitude: float) -> dict:
        return {
            "status": cls.STATUS,
            "available": False,
            "scientific_note": cls.SCIENTIFIC_NOTE,
            "elevation_m": None,
        }


class CameraModelInterface:
    """Extension interface for rigorous pushbroom / framing sensor camera models."""

    STATUS = "INTERFACE_PREPARED_CAMERA_MODEL_UNAVAILABLE"

    @classmethod
    def is_available(cls) -> bool:
        return False
