"""Structured planetary mission-image metadata model and generic label parsing."""

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any


class MetadataStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    ESTIMATED = "ESTIMATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ModalityType(str, Enum):
    OHRC = "OHRC"
    TMC2 = "TMC-2"
    IIRS = "IIRS"
    GENERIC = "GENERIC"


@dataclass
class SolarGeometry:
    incidence_angle_deg: float | None = None
    emission_angle_deg: float | None = None
    phase_angle_deg: float | None = None
    sub_solar_azimuth_deg: float | None = None
    solar_distance_au: float | None = None
    status: MetadataStatus = MetadataStatus.UNKNOWN


@dataclass
class SensorGeometry:
    spacecraft_altitude_km: float | None = None
    slant_range_km: float | None = None
    spacecraft_roll_deg: float | None = None
    spacecraft_pitch_deg: float | None = None
    spacecraft_yaw_deg: float | None = None
    sensor_azimuth_deg: float | None = None
    status: MetadataStatus = MetadataStatus.UNKNOWN


@dataclass
class SpectralMetadata:
    channel_count: int = 1
    band_names: list[str] = field(default_factory=lambda: ["PAN"])
    center_wavelengths_nm: list[float] = field(default_factory=list)
    bandwidths_nm: list[float] = field(default_factory=list)
    calibration_units: str = "DN"
    status: MetadataStatus = MetadataStatus.UNKNOWN


@dataclass
class MissionImageMetadata:
    modality: ModalityType = ModalityType.GENERIC
    instrument_name: str = "Generic Optical Sensor"
    mission_name: str = "Generic Lunar Dataset"
    product_id: str = "UNKNOWN_PRODUCT"
    acquisition_time: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    gsd_m: float | None = None
    pixel_size_um: float | None = None
    units: str = "DN"
    bit_depth: int = 8
    image_orientation_deg: float = 0.0
    spectral: SpectralMetadata = field(default_factory=SpectralMetadata)
    solar_geometry: SolarGeometry = field(default_factory=SolarGeometry)
    sensor_geometry: SensorGeometry = field(default_factory=SensorGeometry)
    coordinate_reference_system: str | None = None
    label_format: str = "DICTIONARY"
    status: MetadataStatus = MetadataStatus.UNKNOWN
    raw_properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "instrument_name": self.instrument_name,
            "mission_name": self.mission_name,
            "product_id": self.product_id,
            "acquisition_time": self.acquisition_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "gsd_m": self.gsd_m,
            "pixel_size_um": self.pixel_size_um,
            "units": self.units,
            "bit_depth": self.bit_depth,
            "image_orientation_deg": self.image_orientation_deg,
            "spectral": {
                "channel_count": self.spectral.channel_count,
                "band_names": self.spectral.band_names,
                "center_wavelengths_nm": self.spectral.center_wavelengths_nm,
                "bandwidths_nm": self.spectral.bandwidths_nm,
                "calibration_units": self.spectral.calibration_units,
                "status": self.spectral.status.value,
            },
            "solar_geometry": {
                "incidence_angle_deg": self.solar_geometry.incidence_angle_deg,
                "emission_angle_deg": self.solar_geometry.emission_angle_deg,
                "phase_angle_deg": self.solar_geometry.phase_angle_deg,
                "status": self.solar_geometry.status.value,
            },
            "sensor_geometry": {
                "spacecraft_altitude_km": self.sensor_geometry.spacecraft_altitude_km,
                "slant_range_km": self.sensor_geometry.slant_range_km,
                "status": self.sensor_geometry.status.value,
            },
            "coordinate_reference_system": self.coordinate_reference_system,
            "label_format": self.label_format,
            "status": self.status.value,
            "physical_scale_available": self.gsd_m is not None and self.gsd_m > 0,
        }


def _normalize_modality(name: str | None) -> ModalityType:
    if not name:
        return ModalityType.GENERIC
    n = name.strip().upper().replace("-", "").replace("_", "")
    if "OHRC" in n:
        return ModalityType.OHRC
    if "TMC" in n or "TMC2" in n:
        return ModalityType.TMC2
    if "IIRS" in n:
        return ModalityType.IIRS
    return ModalityType.GENERIC


def parse_pvl_text(text: str) -> dict[str, str]:
    """Parse generic PVL / key-value parameter format found in planetary labels."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("/*") or line.startswith("#"):
            continue
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip().strip('";').strip()
            result[key] = val
    return result


def parse_metadata_label(content_or_dict: str | bytes | dict, format_hint: str | None = None) -> MissionImageMetadata:
    """Robust, generic metadata reader supporting JSON, XML, text/PVL labels, and dictionaries.

    If fields cannot be determined with certainty, they are marked UNKNOWN rather than guessing.
    """
    raw_dict: dict[str, Any] = {}
    label_format = format_hint or "DICTIONARY"

    if isinstance(content_or_dict, dict):
        raw_dict = content_or_dict
        label_format = "DICTIONARY"
    elif isinstance(content_or_dict, (str, bytes)):
        text = content_or_dict.decode("utf-8", errors="replace") if isinstance(content_or_dict, bytes) else content_or_dict
        text_clean = text.strip()
        if text_clean.startswith("{") and text_clean.endswith("}"):
            try:
                raw_dict = json.loads(text_clean)
                label_format = "JSON"
            except json.JSONDecodeError:
                raw_dict = parse_pvl_text(text_clean)
                label_format = "PVL_TEXT"
        elif text_clean.startswith("<") and text_clean.endswith(">"):
            # Lightweight XML key-value tag extractor
            label_format = "XML"
            matches = re.findall(r"<([a-zA-Z0-9_]+)>([^<]+)</\1>", text_clean)
            raw_dict = {k.upper(): v.strip() for k, v in matches}
        else:
            raw_dict = parse_pvl_text(text_clean)
            label_format = "PVL_TEXT"

    # Normalize keys to lower for lookup
    lookup = {k.lower(): v for k, v in raw_dict.items()}

    # 1. Modality & Instrument
    modality_str = lookup.get("modality") or lookup.get("instrument") or lookup.get("instrument_name") or lookup.get("sensor")
    modality = _normalize_modality(str(modality_str) if modality_str else None)

    inst_name = str(lookup.get("instrument_name") or lookup.get("instrument") or modality.value)
    mission_name = str(lookup.get("mission_name") or lookup.get("mission") or "Chandrayaan-2" if modality != ModalityType.GENERIC else "Generic Lunar Dataset")
    product_id = str(lookup.get("product_id") or lookup.get("product_id_name") or lookup.get("productid") or "UNKNOWN_PRODUCT")
    acq_time = lookup.get("acquisition_time") or lookup.get("start_time") or lookup.get("stop_time")

    # 2. Dimensions & GSD
    def _parse_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _parse_int(val):
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    width = _parse_int(lookup.get("image_width") or lookup.get("width") or lookup.get("samples") or lookup.get("line_samples"))
    height = _parse_int(lookup.get("image_height") or lookup.get("height") or lookup.get("lines"))
    gsd_m = _parse_float(lookup.get("gsd_m") or lookup.get("gsd") or lookup.get("spatial_resolution") or lookup.get("resolution"))
    pixel_size = _parse_float(lookup.get("pixel_size_um") or lookup.get("pixel_size"))
    bit_depth = _parse_int(lookup.get("bit_depth") or lookup.get("bits_per_pixel")) or 8
    units = str(lookup.get("units") or lookup.get("radiance_units") or "DN")

    # 3. Solar geometry
    inc = _parse_float(lookup.get("incidence_angle_deg") or lookup.get("incidence_angle"))
    emi = _parse_float(lookup.get("emission_angle_deg") or lookup.get("emission_angle"))
    phase = _parse_float(lookup.get("phase_angle_deg") or lookup.get("phase_angle"))
    solar_status = MetadataStatus.KNOWN if (inc is not None or emi is not None) else MetadataStatus.UNKNOWN
    solar = SolarGeometry(
        incidence_angle_deg=inc,
        emission_angle_deg=emi,
        phase_angle_deg=phase,
        status=solar_status,
    )

    # 4. Sensor geometry
    alt = _parse_float(lookup.get("spacecraft_altitude_km") or lookup.get("altitude"))
    slant = _parse_float(lookup.get("slant_range_km") or lookup.get("slant_range"))
    sensor_status = MetadataStatus.KNOWN if (alt is not None or slant is not None) else MetadataStatus.UNKNOWN
    sensor = SensorGeometry(
        spacecraft_altitude_km=alt,
        slant_range_km=slant,
        status=sensor_status,
    )

    # 5. Spectral information
    channels = _parse_int(lookup.get("channel_count") or lookup.get("bands")) or 1
    band_names = lookup.get("band_names")
    if isinstance(band_names, str):
        band_names = [b.strip() for b in band_names.split(",")]
    elif not isinstance(band_names, list):
        band_names = ["PAN"] if channels == 1 else [f"Band_{i+1}" for i in range(channels)]

    spectral = SpectralMetadata(
        channel_count=channels,
        band_names=band_names,
        calibration_units=units,
        status=MetadataStatus.KNOWN if (channels > 1 or modality == ModalityType.IIRS) else MetadataStatus.UNKNOWN,
    )

    overall_status = MetadataStatus.KNOWN if (gsd_m is not None or modality != ModalityType.GENERIC or product_id != "UNKNOWN_PRODUCT") else MetadataStatus.UNKNOWN

    return MissionImageMetadata(
        modality=modality,
        instrument_name=inst_name,
        mission_name=mission_name,
        product_id=product_id,
        acquisition_time=str(acq_time) if acq_time else None,
        image_width=width,
        image_height=height,
        gsd_m=gsd_m,
        pixel_size_um=pixel_size,
        units=units,
        bit_depth=bit_depth,
        image_orientation_deg=_parse_float(lookup.get("image_orientation_deg")) or 0.0,
        spectral=spectral,
        solar_geometry=solar,
        sensor_geometry=sensor,
        coordinate_reference_system=str(lookup.get("coordinate_reference_system") or lookup.get("projection") or "IMAGE_PIXEL_RELATIVE"),
        label_format=label_format,
        status=overall_status,
        raw_properties=raw_dict,
    )
