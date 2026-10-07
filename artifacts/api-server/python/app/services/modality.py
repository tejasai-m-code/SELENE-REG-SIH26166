"""Modality-aware abstraction and processing profiles for Chandrayaan-2 instruments (OHRC, TMC-2, IIRS)."""

from dataclasses import dataclass, field
import numpy as np

from app.services.mission_metadata import MissionImageMetadata, ModalityType


@dataclass
class ModalityConfig:
    modality: ModalityType
    name: str
    nominal_gsd_m: float | None
    spectral_type: str  # "panchromatic", "stereo_panchromatic", "hyperspectral", "generic_optical"
    recommended_detector: str
    recommended_representation: str
    default_preprocessing_mode: str
    requires_cross_modal_proxy: bool = False
    description: str = ""
    spectral_range_nm: tuple[float, float] | None = None
    profile_type: str = "REFERENCE_PROFILE"
    provenance_source: str = "DEFAULT_ESTIMATE"


MODALITY_PROFILES: dict[ModalityType, ModalityConfig] = {
    ModalityType.OHRC: ModalityConfig(
        modality=ModalityType.OHRC,
        name="Orbiter High Resolution Camera (OHRC)",
        nominal_gsd_m=0.25,
        spectral_type="panchromatic",
        recommended_detector="sift",
        recommended_representation="structural",
        default_preprocessing_mode="clahe",
        requires_cross_modal_proxy=False,
        description="Ultra-high resolution optical panchromatic framing camera (~0.25–0.32 m/pixel nominal reference).",
        spectral_range_nm=(450.0, 900.0),
        profile_type="REFERENCE_PROFILE",
        provenance_source="DEFAULT_ESTIMATE",
    ),
    ModalityType.TMC2: ModalityConfig(
        modality=ModalityType.TMC2,
        name="Terrain Mapping Camera-2 (TMC-2)",
        nominal_gsd_m=5.0,
        spectral_type="stereo_panchromatic",
        recommended_detector="sift",
        recommended_representation="structural",
        default_preprocessing_mode="safe_normalization",
        requires_cross_modal_proxy=False,
        description="Stereo triplet pushbroom imaging camera (Fore, Nadir, Aft) at ~5 m/pixel nominal reference GSD.",
        spectral_range_nm=(500.0, 850.0),
        profile_type="REFERENCE_PROFILE",
        provenance_source="DEFAULT_ESTIMATE",
    ),
    ModalityType.IIRS: ModalityConfig(
        modality=ModalityType.IIRS,
        name="Imaging Infra-Red Spectrometer (IIRS)",
        nominal_gsd_m=80.0,
        spectral_type="hyperspectral",
        recommended_detector="sift",
        recommended_representation="gradient",
        default_preprocessing_mode="percentile",
        requires_cross_modal_proxy=True,
        description="Hyperspectral imaging spectrometer (256 bands, ~0.8–5.0 um nominal reference) at ~80 m/pixel GSD.",
        spectral_range_nm=(800.0, 5000.0),
        profile_type="REFERENCE_PROFILE",
        provenance_source="DEFAULT_ESTIMATE",
    ),
    ModalityType.GENERIC: ModalityConfig(
        modality=ModalityType.GENERIC,
        name="Generic Lunar Sensor",
        nominal_gsd_m=None,
        spectral_type="generic_optical",
        recommended_detector="sift",
        recommended_representation="structural",
        default_preprocessing_mode="clahe",
        requires_cross_modal_proxy=False,
        description="Standard generic optical raster without mission-specific geometric constraints.",
        spectral_range_nm=None,
        profile_type="REFERENCE_PROFILE",
        provenance_source="DEFAULT_ESTIMATE",
    ),
}


def get_modality_config(modality: ModalityType | str) -> ModalityConfig:
    if isinstance(modality, str):
        mod_upper = modality.upper().strip().replace("-", "").replace("_", "")
        if "OHRC" in mod_upper:
            m_type = ModalityType.OHRC
        elif "TMC" in mod_upper or "TMC2" in mod_upper:
            m_type = ModalityType.TMC2
        elif "IIRS" in mod_upper:
            m_type = ModalityType.IIRS
        else:
            m_type = ModalityType.GENERIC
    else:
        m_type = modality

    return MODALITY_PROFILES.get(m_type, MODALITY_PROFILES[ModalityType.GENERIC])


def compute_gsd_scale_factor(
    source_meta: MissionImageMetadata | dict | None,
    target_meta: MissionImageMetadata | dict | None,
) -> tuple[float | None, str]:
    """Compute physical resolution scaling ratio between source and target rasters.

    Returns:
    (gsd_ratio, status_message)
    where gsd_ratio = source_gsd / target_gsd.
    """
    if source_meta is None or target_meta is None:
        return None, "GSD metadata unavailable for one or both images."

    gsd_s = source_meta.gsd_m if isinstance(source_meta, MissionImageMetadata) else source_meta.get("gsd_m") or source_meta.get("gsd")
    gsd_t = target_meta.gsd_m if isinstance(target_meta, MissionImageMetadata) else target_meta.get("gsd_m") or target_meta.get("gsd")

    if gsd_s is None or gsd_t is None or gsd_s <= 0 or gsd_t <= 0:
        return None, "GSD values not provided or non-positive; physical scale unavailable."

    ratio = float(gsd_s / gsd_t)
    return ratio, f"Valid GSD ratio: source={gsd_s:.2f} m/px, target={gsd_t:.2f} m/px, ratio={ratio:.4f}."
