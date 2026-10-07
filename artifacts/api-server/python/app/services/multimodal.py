"""Multimodal abstraction layer for SELENE-REG-X.

Defines a clean interface so that future modality-specific preprocessing strategies
can plug into the same registration engine without modifying core matching logic.

Currently supported modalities (classical, no learned models):
  - OHRC  : High-resolution panchromatic (0.25 m/px typical)
  - TMC2  : Terrain Mapping Camera 2 panchromatic (5 m/px typical)
  - IIRS  : Imaging IR Spectrometer, mapped to a single-band luminance proxy
  - GENERIC: Any unknown grayscale or RGB source

Workflow:
    MODALITY
        ↓
    MODALITY-SPECIFIC PREPROCESSING  (via ModalityConfig)
        ↓
    COMMON STRUCTURAL REPRESENTATION (via preprocessing.py)
        ↓
    FEATURE / CORRESPONDENCE ENGINE  (via feature_matching.py / multiscale.py)
        ↓
    GEOMETRIC REGISTRATION           (via geometry.py)
        ↓
    SUBPIXEL REFINEMENT              (via registration.py / subpixel.py)

IMPORTANT: This module does NOT import real PDS metadata or SPICE kernels.
If real calibration data is available it must be supplied by the caller as a
plain Python dict conforming to the `ModalityMetadata` schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------

KNOWN_MODALITIES = {
    "ohrc": {
        "display_name": "OHRC (Orbiter High Resolution Camera)",
        "instrument": "OHRC",
        "mission": "Chandrayaan-2",
        "spectral_domain": "panchromatic",
        "typical_gsd_m": 0.25,
        "bit_depth": 8,
        "recommended_representation": "structural",
        "notes": "High spatial detail. Shadow edges are sharp.",
    },
    "tmc2": {
        "display_name": "TMC-2 (Terrain Mapping Camera 2)",
        "instrument": "TMC-2",
        "mission": "Chandrayaan-2",
        "spectral_domain": "panchromatic",
        "typical_gsd_m": 5.0,
        "bit_depth": 8,
        "recommended_representation": "clahe",
        "notes": "Wide-area context. Coarser texture than OHRC.",
    },
    "iirs": {
        "display_name": "IIRS (Imaging Infrared Spectrometer)",
        "instrument": "IIRS",
        "mission": "Chandrayaan-2",
        "spectral_domain": "near_infrared",
        "typical_gsd_m": 29.0,
        "bit_depth": 12,
        "recommended_representation": "retinex",
        "notes": (
            "Thermal/spectral domain. Intensity relationships differ from optical. "
            "Gradient and retinex representations are preferred for cross-modal matching. "
            "Cross-OHRC↔IIRS is a hard open problem; classical features are unreliable."
        ),
    },
    "generic": {
        "display_name": "Generic Grayscale / RGB",
        "instrument": "unknown",
        "mission": "unknown",
        "spectral_domain": "unknown",
        "typical_gsd_m": None,
        "bit_depth": 8,
        "recommended_representation": "structural",
        "notes": "No instrument-specific preprocessing applied.",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModalityMetadata:
    """Caller-supplied metadata for one image in a pair.

    All fields are optional; defaults are used when information is absent.
    Do NOT invent values that are not actually known for the dataset.
    """
    modality: str = "generic"
    gsd_m: float | None = None
    wavelength_nm: float | None = None
    channel_name: str | None = None
    radiance_scale: float | None = None
    radiance_offset: float | None = None
    solar_incidence_deg: float | None = None
    extra: dict = field(default_factory=dict)

    def to_preprocessing_dict(self) -> dict:
        """Return a dict compatible with `preprocess_with_diagnostics(metadata=…)`."""
        d: dict[str, Any] = {}
        if self.gsd_m is not None:
            d["gsd_m"] = self.gsd_m
        if self.radiance_scale is not None:
            d["radiance_scale"] = self.radiance_scale
        if self.radiance_offset is not None:
            d["radiance_offset"] = self.radiance_offset
        d.update(self.extra)
        return d


@dataclass
class ModalityConfig:
    """Per-modality preprocessing recipe."""
    modality: str
    representation: str
    radiometric_mode: str
    illumination_normalization: bool
    detector: str
    max_features: int
    ratio: float
    notes: str = ""

    @classmethod
    def for_modality(cls, modality: str) -> "ModalityConfig":
        """Return a sensible default config for the named modality."""
        m = modality.lower().strip()
        known = KNOWN_MODALITIES.get(m, KNOWN_MODALITIES["generic"])
        rep = known["recommended_representation"]

        if m == "iirs":
            return cls(
                modality=m,
                representation="retinex",
                radiometric_mode="safe_normalization",
                illumination_normalization=True,
                detector="sift",
                max_features=4000,
                ratio=0.75,
                notes=(
                    "IIRS uses retinex to partially suppress broad illumination variation. "
                    "Classical SIFT features are used with slightly relaxed ratio threshold."
                ),
            )
        if m == "ohrc":
            return cls(
                modality=m,
                representation="structural",
                radiometric_mode="safe_normalization",
                illumination_normalization=True,
                detector="sift",
                max_features=8000,
                ratio=0.72,
                notes="OHRC: high-resolution structural representation.",
            )
        if m == "tmc2":
            return cls(
                modality=m,
                representation="clahe",
                radiometric_mode="safe_normalization",
                illumination_normalization=True,
                detector="sift",
                max_features=6000,
                ratio=0.72,
                notes="TMC-2: CLAHE to compensate for moderate dynamic-range variation.",
            )
        # generic
        return cls(
            modality="generic",
            representation="structural",
            radiometric_mode="safe_normalization",
            illumination_normalization=True,
            detector="sift",
            max_features=8000,
            ratio=0.72,
            notes="Generic: default structural representation.",
        )


# ---------------------------------------------------------------------------
# Cross-modal representation helpers
# ---------------------------------------------------------------------------

def cross_modal_representation(
    image: np.ndarray,
    source_modality: str = "generic",
    reference_modality: str = "generic",
) -> str:
    """Return the recommended shared representation for a cross-modal pair.

    This is a heuristic based on the known spectral domain pairing.
    It does NOT guarantee that the chosen representation will succeed;
    actual performance must be measured experimentally.

    NOTE: ohrc↔iirs is the hardest pairing. 'gradient' is recommended but
    classical features may still fail due to fundamentally different
    reflectance relationships. Learned multimodal models (Phase 5+) are
    the intended long-term solution for this pairing.
    """
    s = source_modality.lower()
    r = reference_modality.lower()

    # Same modality – use the standard recommended representation
    if s == r:
        cfg = ModalityConfig.for_modality(s)
        return cfg.representation

    # Cross-modal pairings
    pairing = frozenset([s, r])
    if pairing == frozenset(["ohrc", "iirs"]):
        return "gradient"  # structural gradients most modality-agnostic
    if pairing == frozenset(["ohrc", "tmc2"]):
        return "clahe"     # both panchromatic, CLAHE handles scale-induced contrast diff
    if pairing == frozenset(["tmc2", "iirs"]):
        return "retinex"   # retinex suppresses broad illumination gradients

    return "structural"  # safe default


@dataclass
class CrossModalConfig:
    """Configuration object for a registered image pair with different modalities."""
    source_modality: ModalityMetadata
    reference_modality: ModalityMetadata
    shared_representation: str
    source_config: ModalityConfig
    reference_config: ModalityConfig
    is_cross_modal: bool
    pairing_notes: str

    @classmethod
    def build(
        cls,
        source_meta: ModalityMetadata,
        reference_meta: ModalityMetadata,
    ) -> "CrossModalConfig":
        src_cfg = ModalityConfig.for_modality(source_meta.modality)
        ref_cfg = ModalityConfig.for_modality(reference_meta.modality)
        is_cross = source_meta.modality.lower() != reference_meta.modality.lower()
        shared_rep = cross_modal_representation(
            np.zeros((1, 1), dtype=np.uint8),
            source_meta.modality,
            reference_meta.modality,
        )
        notes = (
            f"Source: {source_meta.modality.upper()}, "
            f"Reference: {reference_meta.modality.upper()}. "
            f"Shared representation: '{shared_rep}'. "
        )
        if is_cross:
            notes += (
                "Cross-modal pair: intensity relationships may differ significantly. "
                "Gradient/structural representations provide more modality-agnostic features. "
                "Results should be treated as synthetic proxy experiments unless "
                "validated on actual instrument data."
            )
        return cls(
            source_modality=source_meta,
            reference_modality=reference_meta,
            shared_representation=shared_rep,
            source_config=src_cfg,
            reference_config=ref_cfg,
            is_cross_modal=is_cross,
            pairing_notes=notes,
        )


# ---------------------------------------------------------------------------
# Synthetic cross-modal transformations (for experimental use only)
# ---------------------------------------------------------------------------

def apply_synthetic_cross_modal(
    image: np.ndarray,
    mode: str = "gamma",
    param: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Apply a deterministic synthetic transformation that mimics how different
    imaging modalities might produce different intensity profiles from the same scene.

    IMPORTANT: These are NOT equivalent to actual OHRC↔IIRS cross-modal pairs.
    They are controlled synthetic proxy experiments only.

    Supported modes:
        gamma            – power-law intensity remap (models log-linear sensor difference)
        invert           – global contrast inversion (models reflectance polarity flip)
        nonlinear_map    – sigmoidal intensity remap (models non-linear response curve)
        local_illum      – spatially varying gain gradient (models solar incidence variation)
        shadow           – dark rectangular/soft blended shadow region
        multi_shadow     – multiple overlapping shadow regions
        combined         – local_illum + shadow combined
    """
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)

    if mode == "gamma":
        gamma = max(0.1, float(param))
        out = (img / 255.0) ** gamma
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    if mode == "invert":
        return np.clip(255.0 - img, 0, 255).astype(np.uint8)

    if mode == "nonlinear_map":
        k = float(param)
        x = img / 255.0
        out = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        # rescale to fill [0, 255]
        out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    if mode == "local_illum":
        h, w = img.shape[:2]
        angle = rng.uniform(0, np.pi)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        gradient = (x_coords * np.cos(angle) + y_coords * np.sin(angle))
        gradient = (gradient - gradient.min()) / max(gradient.max() - gradient.min(), 1e-6)
        strength = float(param)
        gain = 1.0 - strength * 0.4 * gradient
        out = img * gain
        return np.clip(out, 0, 255).astype(np.uint8)

    if mode == "shadow":
        out = img.copy()
        h, w = img.shape[:2]
        sw = max(10, int(w * 0.25))
        sh = max(10, int(h * 0.30))
        x0 = rng.integers(0, max(1, w - sw))
        y0 = rng.integers(0, max(1, h - sh))
        # Build soft shadow with Gaussian blur
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0:y0+sh, x0:x0+sw] = 1.0
        mask = cv2.GaussianBlur(mask, (31, 31), 10.0)
        shadow_strength = float(param)
        gain = 1.0 - shadow_strength * mask
        out_f = img * gain
        return np.clip(out_f, 0, 255).astype(np.uint8)

    if mode == "multi_shadow":
        out_f = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        n_shadows = int(param)
        for _ in range(n_shadows):
            sw = max(10, int(rng.uniform(0.1, 0.3) * w))
            sh = max(10, int(rng.uniform(0.1, 0.3) * h))
            x0 = rng.integers(0, max(1, w - sw))
            y0 = rng.integers(0, max(1, h - sh))
            mask = np.zeros((h, w), dtype=np.float32)
            mask[y0:y0+sh, x0:x0+sw] = 1.0
            mask = cv2.GaussianBlur(mask, (21, 21), 7.0)
            strength = rng.uniform(0.3, 0.7)
            out_f *= (1.0 - strength * mask)
        return np.clip(out_f, 0, 255).astype(np.uint8)

    if mode == "combined":
        # local illumination + shadow
        step1 = apply_synthetic_cross_modal(image, "local_illum", param, seed)
        step2 = apply_synthetic_cross_modal(step1, "shadow", 0.6, seed + 1)
        return step2

    # Fallback: return unchanged
    return np.clip(img, 0, 255).astype(np.uint8)
