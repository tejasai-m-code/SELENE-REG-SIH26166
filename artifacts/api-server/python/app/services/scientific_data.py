"""Scientific Image Ingestion, Data Inspection, and Sensor Auto-Identification.

Complies with SIH 26166 Phase 1 requirements:
- Decouples raw scientific data (preserves original dtype e.g. uint16, int16, float32,
  original channels, dynamic range, and metadata) from display preview (8-bit normalized).
- Rigorous statistical inspection: dimensions, channels, bands, bit-depth, dtype,
  min/max/mean/median/std, valid-pixel %, saturation %, dynamic range, and histogram.
- Layer count investigation: prevents confusion between image channels, scientific bands,
  and UI layers.
- Evidence-based sensor auto-identification: OHRC, TMC-2, IIRS, LROC NAC, LROC WAC,
  SELENE/Kaguya, with transparent evidence logging and explicit UNKNOWN/NEEDS METADATA fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


@dataclass
class SensorIdentification:
    sensor: str
    mission: str
    gsd_m: float | None
    confidence_pct: float
    evidence: list[str]
    metadata_status: str  # "VERIFIED" | "INFERRED_FROM_FILENAME" | "USER_SUPPLIED" | "UNKNOWN"
    band_info: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "mission": self.mission,
            "gsd_m": self.gsd_m,
            "confidence_pct": round(self.confidence_pct, 1),
            "evidence": self.evidence,
            "metadata_status": self.metadata_status,
            "band_info": self.band_info,
        }


@dataclass
class ScientificInspection:
    dimensions: list[int]  # [width, height, channels]
    width: int
    height: int
    ndim: int
    channels: int
    scientific_bands: int
    layer_classification: str
    bit_depth: int
    dtype: str
    min_dn: float
    max_dn: float
    mean_dn: float
    median_dn: float
    std_dn: float
    dynamic_range_span: float
    valid_pixel_pct: float
    saturation_pct: float
    histogram_bins: list[float]
    histogram_counts: list[int]
    has_alpha: bool
    compression: str
    display_normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "width": self.width,
            "height": self.height,
            "ndim": self.ndim,
            "channels": self.channels,
            "scientific_bands": self.scientific_bands,
            "layer_classification": self.layer_classification,
            "bit_depth": self.bit_depth,
            "dtype": self.dtype,
            "min_dn": round(self.min_dn, 3),
            "max_dn": round(self.max_dn, 3),
            "mean_dn": round(self.mean_dn, 3),
            "median_dn": round(self.median_dn, 3),
            "std_dn": round(self.std_dn, 3),
            "dynamic_range_span": round(self.dynamic_range_span, 3),
            "valid_pixel_pct": round(self.valid_pixel_pct, 2),
            "saturation_pct": round(self.saturation_pct, 2),
            "histogram_bins": [round(b, 2) for b in self.histogram_bins],
            "histogram_counts": self.histogram_counts,
            "has_alpha": self.has_alpha,
            "compression": self.compression,
            "display_normalized": self.display_normalized,
        }


@dataclass
class ScientificRaster:
    raw_data: np.ndarray
    display_data: np.ndarray
    matching_data: np.ndarray
    inspection: ScientificInspection
    sensor_id: SensorIdentification
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sensor": self.sensor_id.to_dict(),
            "inspection": self.inspection.to_dict(),
            "metadata": self.metadata,
        }


def identify_sensor_evidence(
    filename: str = "",
    metadata: dict[str, Any] | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    channels: int = 1,
) -> SensorIdentification:
    """Identify lunar sensor and mission using genuine evidence without fabrication."""
    evidence: list[str] = []
    meta = metadata or {}
    fn_upper = (filename or "").upper()

    # 1. First check explicit user-supplied or embedded label metadata
    supp_sensor = meta.get("sensor") or meta.get("instrument") or meta.get("instrument_name")
    if supp_sensor and str(supp_sensor).upper() not in ("UNKNOWN", "UNKNOWN / NOT PROVIDED", ""):
        sens_str = str(supp_sensor).strip()
        sens_upper = sens_str.upper()
        if "OHRC" in sens_upper:
            evidence.append(f"Metadata tag indicates OHRC: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 0.25)
            return SensorIdentification(
                sensor="Chandrayaan-2 OHRC (0.25m/px)",
                mission="Chandrayaan-2",
                gsd_m=gsd,
                confidence_pct=95.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Panchromatic High-Resolution Optical (0.25m)",
            )
        if "TMC" in sens_upper:
            evidence.append(f"Metadata tag indicates TMC-2: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 5.0)
            return SensorIdentification(
                sensor="Chandrayaan-2 TMC-2 (5.0m/px)",
                mission="Chandrayaan-2",
                gsd_m=gsd,
                confidence_pct=95.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Terrain Mapping Stereo Camera (5.0m)",
            )
        if "IIRS" in sens_upper:
            evidence.append(f"Metadata tag indicates IIRS: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 80.0)
            return SensorIdentification(
                sensor="Chandrayaan-2 IIRS (80m/px)",
                mission="Chandrayaan-2",
                gsd_m=gsd,
                confidence_pct=95.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Imaging Infrared Spectrometer (256 bands, 80m)",
            )
        if "NAC" in sens_upper or "LROC_NAC" in sens_upper or "LRO NAC" in sens_upper:
            evidence.append(f"Metadata tag indicates LROC NAC: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 0.5)
            return SensorIdentification(
                sensor="LRO NAC (0.5m/px)",
                mission="Lunar Reconnaissance Orbiter",
                gsd_m=gsd,
                confidence_pct=95.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Narrow Angle Camera Optical (0.5m)",
            )
        if "WAC" in sens_upper or "LROC_WAC" in sens_upper or "LRO WAC" in sens_upper:
            evidence.append(f"Metadata tag indicates LROC WAC: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 100.0)
            return SensorIdentification(
                sensor="LRO WAC (100m/px)",
                mission="Lunar Reconnaissance Orbiter",
                gsd_m=gsd,
                confidence_pct=95.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Wide Angle Camera Multispectral (100m)",
            )
        if "KAGUYA" in sens_upper or "SELENE" in sens_upper:
            evidence.append(f"Metadata tag indicates SELENE/Kaguya: '{supp_sensor}'")
            gsd = float(meta.get("gsd_m") or 10.0)
            return SensorIdentification(
                sensor="SELENE / Kaguya (10m/px)",
                mission="SELENE (Kaguya)",
                gsd_m=gsd,
                confidence_pct=90.0,
                evidence=evidence,
                metadata_status="VERIFIED",
                band_info="Terrain Camera Optical (10m)",
            )

    # 2. Check filename token patterns
    if re.search(r"(?:CH2[_-]?OHR|OHRC)", fn_upper):
        evidence.append(f"Filename contains Chandrayaan-2 OHRC signature: '{filename}'")
        return SensorIdentification(
            sensor="Chandrayaan-2 OHRC (0.25m/px)",
            mission="Chandrayaan-2",
            gsd_m=0.25,
            confidence_pct=85.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Panchromatic High-Resolution Optical (0.25m GSD)",
        )

    if re.search(r"(?:CH2[_-]?TMC|TMC2|TMC[_-]2)", fn_upper):
        evidence.append(f"Filename contains Chandrayaan-2 TMC-2 signature: '{filename}'")
        return SensorIdentification(
            sensor="Chandrayaan-2 TMC-2 (5.0m/px)",
            mission="Chandrayaan-2",
            gsd_m=5.0,
            confidence_pct=85.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Terrain Mapping Stereo Camera (5.0m GSD)",
        )

    if re.search(r"(?:CH2[_-]?IIR|IIRS)", fn_upper):
        evidence.append(f"Filename contains Chandrayaan-2 IIRS signature: '{filename}'")
        return SensorIdentification(
            sensor="Chandrayaan-2 IIRS (80m/px)",
            mission="Chandrayaan-2",
            gsd_m=80.0,
            confidence_pct=85.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Imaging Infrared Spectrometer (80m GSD)",
        )

    if re.search(r"(?:M[0-9]{7,10}[RLE]|LROC[_-]?NAC|\bNAC\b|_NAC|NAC_)", fn_upper):
        evidence.append(f"Filename matches LROC NAC product pattern: '{filename}'")
        return SensorIdentification(
            sensor="LRO NAC (0.5m/px)",
            mission="Lunar Reconnaissance Orbiter",
            gsd_m=0.5,
            confidence_pct=80.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Narrow Angle Camera Optical (0.5m GSD)",
        )

    if re.search(r"(?:LROC[_-]?WAC|WAC)", fn_upper):
        evidence.append(f"Filename matches LROC WAC signature: '{filename}'")
        return SensorIdentification(
            sensor="LRO WAC (100m/px)",
            mission="Lunar Reconnaissance Orbiter",
            gsd_m=100.0,
            confidence_pct=80.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Wide Angle Camera (100m GSD)",
        )

    if re.search(r"(?:TC[_-]|KAGUYA|SELENE)", fn_upper):
        evidence.append(f"Filename matches SELENE/Kaguya pattern: '{filename}'")
        return SensorIdentification(
            sensor="SELENE / Kaguya (10m/px)",
            mission="SELENE (Kaguya)",
            gsd_m=10.0,
            confidence_pct=75.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_FILENAME",
            band_info="Terrain Camera Optical (10m GSD)",
        )

    # 3. Secondary check: raster characteristics
    if channels >= 100:
        evidence.append(f"High spectral band count ({channels} bands) indicates imaging spectrometer")
        return SensorIdentification(
            sensor="Chandrayaan-2 IIRS (80m/px)",
            mission="Chandrayaan-2",
            gsd_m=80.0,
            confidence_pct=60.0,
            evidence=evidence,
            metadata_status="INFERRED_FROM_DIMENSIONS",
            band_info=f"Imaging Infrared Spectrometer ({channels} spectral channels)",
        )

    # 4. If no reliable evidence, do NOT fabricate!
    evidence.append("No embedded PDS/GeoTIFF sensor tags or recognized mission tokens found in filename.")
    return SensorIdentification(
        sensor="UNKNOWN / NEEDS METADATA",
        mission="Unknown Lunar Platform",
        gsd_m=None,
        confidence_pct=0.0,
        evidence=evidence,
        metadata_status="UNKNOWN",
        band_info="Panchromatic / Optical Raster (Standard)",
    )


def compute_scientific_inspection(
    raw_array: np.ndarray,
    compression: str = "Uncompressed / Lossless",
) -> ScientificInspection:
    """Calculate comprehensive radiometric and quality statistics without mutating raw data."""
    arr = raw_array
    ndim = arr.ndim
    h, w = arr.shape[:2]
    channels = int(arr.shape[2]) if ndim == 3 else 1
    has_alpha = channels == 4

    bit_depth = int(arr.dtype.itemsize * 8)
    dtype_name = str(arr.dtype)

    # Layer count clarification
    if channels == 1:
        layer_class = "Panchromatic Grayscale (1 Scientific Layer)"
        bands = 1
    elif channels == 3:
        layer_class = "RGB Color Composite (3 Image Channels / 1 Tri-band Layer)"
        bands = 3
    elif channels == 4:
        layer_class = "RGBA (3 Color Channels + 1 Alpha Transparency Mask)"
        bands = 3
    else:
        layer_class = f"Multispectral / Hyperspectral ({channels} Scientific Bands)"
        bands = channels

    # Focus on first channel or intensity plane for single scalar statistics
    sample_plane = arr[..., 0] if (ndim == 3 and channels > 1) else arr
    flat_sample = sample_plane.ravel().astype(np.float64)

    # Valid pixel calculation
    if np.issubdtype(arr.dtype, np.floating):
        valid_mask = np.isfinite(flat_sample)
    else:
        # Non-zero or standard range
        valid_mask = flat_sample >= 0

    valid_count = int(np.sum(valid_mask))
    total_count = max(1, flat_sample.size)
    valid_pct = float(round((valid_count / total_count) * 100.0, 2))

    if valid_count > 0:
        valid_vals = flat_sample[valid_mask]
        min_dn = float(np.min(valid_vals))
        max_dn = float(np.max(valid_vals))
        mean_dn = float(np.mean(valid_vals))
        median_dn = float(np.median(valid_vals))
        std_dn = float(np.std(valid_vals))
        range_span = max_dn - min_dn

        # Saturation check: max possible value for bit depth
        if np.issubdtype(arr.dtype, np.integer):
            max_possible = float(np.iinfo(arr.dtype).max)
            sat_count = int(np.sum(valid_vals >= (max_possible - 1)))
        else:
            sat_count = int(np.sum(valid_vals >= 1.0)) if max_dn <= 1.0 else int(np.sum(valid_vals >= (max_dn - 1e-4)))

        saturation_pct = float(round((sat_count / valid_count) * 100.0, 2))

        # 32-bin histogram over the actual dynamic range
        hist_counts, bin_edges = np.histogram(valid_vals, bins=32)
        bins_list = [float(e) for e in bin_edges]
        counts_list = [int(c) for c in hist_counts]
    else:
        min_dn = max_dn = mean_dn = median_dn = std_dn = range_span = saturation_pct = 0.0
        bins_list = [0.0] * 33
        counts_list = [0] * 32

    return ScientificInspection(
        dimensions=[w, h, channels],
        width=w,
        height=h,
        ndim=ndim,
        channels=channels,
        scientific_bands=bands,
        layer_classification=layer_class,
        bit_depth=bit_depth,
        dtype=dtype_name,
        min_dn=min_dn,
        max_dn=max_dn,
        mean_dn=mean_dn,
        median_dn=median_dn,
        std_dn=std_dn,
        dynamic_range_span=range_span,
        valid_pixel_pct=valid_pct,
        saturation_pct=saturation_pct,
        histogram_bins=bins_list,
        histogram_counts=counts_list,
        has_alpha=has_alpha,
        compression=compression,
        display_normalized=True,
    )


def create_display_preview(raw_array: np.ndarray) -> np.ndarray:
    """Create a separate 8-bit normalized preview for UI rendering without corrupting scientific radiometry.

    Preserves 1%-99% relative contrast with robust clipping.
    """
    arr = raw_array
    if arr.dtype == np.uint8:
        return arr.copy()

    # Convert to float for normalization
    x = arr.astype(np.float32)
    if x.ndim == 3 and x.shape[2] > 1:
        channels_8u = []
        for c in range(min(3, x.shape[2])):
            ch = x[..., c]
            finite = np.isfinite(ch)
            if not finite.any():
                channels_8u.append(np.zeros(ch.shape, dtype=np.uint8))
                continue
            vals = ch[finite]
            lo, hi = np.percentile(vals, [1, 99])
            if hi <= lo:
                lo, hi = float(vals.min()), float(vals.max())
            if hi <= lo:
                channels_8u.append(np.zeros(ch.shape, dtype=np.uint8))
                continue
            norm = np.clip((ch - lo) / (hi - lo), 0, 1)
            channels_8u.append((norm * 255).astype(np.uint8))
        return np.stack(channels_8u, axis=2)

    # 2D grayscale or single channel
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)

    vals = x[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)

    norm = np.clip((x - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def ingest_scientific_image(
    data_or_path: bytes | str | Path,
    filename: str = "",
    supplied_metadata: dict[str, Any] | None = None,
) -> ScientificRaster:
    """Authoritative scientific ingestion entrypoint.

    Decouples raw scientific data from display representation.
    """
    fn = filename
    meta = dict(supplied_metadata or {})
    raw_img: np.ndarray | None = None

    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        fn = fn or p.name
        # Read unchanged to preserve 16-bit / multi-channel data
        raw_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            # Try fromfile for binary or special raster
            try:
                raw_bytes = p.read_bytes()
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                raw_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            except Exception:
                pass
    elif isinstance(data_or_path, bytes):
        arr = np.frombuffer(data_or_path, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if raw_img is None or raw_img.size == 0:
        raise ValueError(f"Could not decode scientific raster from input ({fn or 'bytes'}).")

    # 1. Scientific Inspection (preserves original bit-depth and DN range)
    inspection = compute_scientific_inspection(raw_img)

    # 2. Sensor Auto-Identification with real evidence
    sensor_id = identify_sensor_evidence(
        filename=fn,
        metadata=meta,
        shape=raw_img.shape,
        dtype=raw_img.dtype,
        channels=inspection.channels,
    )

    # 3. Create decoupled 8-bit display preview (strictly for canvas rendering)
    display_preview = create_display_preview(raw_img)

    # 4. Create matching representation
    if display_preview.ndim == 3 and display_preview.shape[2] == 4:
        matching_img = cv2.cvtColor(display_preview, cv2.COLOR_BGRA2GRAY)
    elif display_preview.ndim == 3 and display_preview.shape[2] == 3:
        matching_img = cv2.cvtColor(display_preview, cv2.COLOR_BGR2GRAY)
    else:
        matching_img = display_preview

    return ScientificRaster(
        raw_data=raw_img,
        display_data=display_preview,
        matching_data=matching_img,
        inspection=inspection,
        sensor_id=sensor_id,
        metadata=meta,
        filename=fn,
    )
