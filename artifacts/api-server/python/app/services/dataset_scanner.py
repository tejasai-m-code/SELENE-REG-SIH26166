"""Recursive local dataset scanner, scientific raster parser, and manifest generator.

Supports:
- Recursive discovery of directory structures (e.g. dataset/zip files/..., dataset/calibrated/20260130/...)
- Safe extraction and inspection of ZIP archives
- Standard raster formats (PNG, JPG/JPEG, TIFF, GeoTIFF, BMP, WEBP)
- Scientific raster formats (PDS3/PDS4 .IMG, VICAR, FITS, ENVI .bsq/.bil/.raw)
- Associated metadata extraction (.lbl, .xml, .json, .txt)
- Safe handling of unsupported formats without silent failure
- Dataset Manifest generation with file properties and preview generation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import zipfile

import cv2
import numpy as np

from app.services.mission_metadata import parse_metadata_label
from app.utils.image_utils import resize_for_preview, save_image

# File extensions categorized
STANDARD_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SCIENTIFIC_RASTER_EXTS = {".img", ".cub", ".jp2", ".fits", ".fit", ".bil", ".bsq", ".raw", ".dat"}
METADATA_EXTS = {".lbl", ".xml", ".json", ".txt", ".pvl"}
ARCHIVE_EXTS = {".zip"}

SUPPORTED_RASTER_EXTENSIONS = STANDARD_IMAGE_EXTS | SCIENTIFIC_RASTER_EXTS
METADATA_EXTENSIONS = METADATA_EXTS


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        # Extract first numeric substring (handles PDS3 unit brackets like "0.25 <METERS/PIXEL>")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
        if m:
            try:
                f = float(m.group(0))
                return f if np.isfinite(f) else None
            except Exception:
                pass
        return None



def _safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_pds3_label_text(text: str) -> dict[str, Any]:
    """Extract key-value pairs from standard PDS3 / PVL label text."""
    data = {}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("/*") or line == "END":
            continue
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip().strip('"').strip("'")
            # Remove trailing comments
            if "/*" in val:
                val = val.split("/*")[0].strip()
            data[key] = val
    return data


def extract_metadata_from_file(file_path: Path) -> dict[str, Any]:
    """Safely extract metadata from XML, LBL, JSON, or TXT sidecar without inventing values."""
    ext = file_path.suffix.lower()
    meta: dict[str, Any] = {}
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")[:100000]
        if ext == ".json":
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        if ext in {".lbl", ".pvl", ".txt"} or ("PDS_VERSION_ID" in content or "RECORD_TYPE" in content):
            pds_dict = parse_pds3_label_text(content)
            if pds_dict:
                sensor = (
                    pds_dict.get("INSTRUMENT_ID")
                    or pds_dict.get("INSTRUMENT_NAME")
                    or pds_dict.get("INSTRUMENT_HOST_NAME")
                )
                gsd = _safe_float(
                    pds_dict.get("PIXEL_RESOLUTION")
                    or pds_dict.get("GROUND_RESOLUTION")
                    or pds_dict.get("MAP_RESOLUTION")
                )
                sol_inc = _safe_float(pds_dict.get("SOLAR_INCIDENCE_ANGLE") or pds_dict.get("INCIDENCE_ANGLE"))
                sol_az = _safe_float(pds_dict.get("SOLAR_AZIMUTH_ANGLE") or pds_dict.get("SUB_SOLAR_AZIMUTH"))
                view_inc = _safe_float(pds_dict.get("EMISSION_ANGLE") or pds_dict.get("VIEW_INCIDENCE_ANGLE"))
                target = pds_dict.get("TARGET_NAME") or "MOON"
                lines = _safe_int(pds_dict.get("LINES") or pds_dict.get("IMAGE_LINES"))
                samples = _safe_int(pds_dict.get("LINE_SAMPLES") or pds_dict.get("IMAGE_LINE_SAMPLES"))
                sample_bits = _safe_int(pds_dict.get("SAMPLE_BITS") or 8)

                if sensor:
                    meta["sensor"] = sensor
                if gsd is not None:
                    meta["gsd_m"] = gsd
                if sol_inc is not None:
                    meta["solar_incidence_deg"] = sol_inc
                if sol_az is not None:
                    meta["solar_azimuth_deg"] = sol_az
                if view_inc is not None:
                    meta["view_incidence_deg"] = view_inc
                if target:
                    meta["target"] = target
                if lines and samples:
                    meta["dimensions"] = [int(lines), int(samples)]
                if sample_bits:
                    meta["sample_bits"] = sample_bits
                meta["pds_label_keys_found"] = len(pds_dict)
                return meta

        if ext == ".xml":
            # Simple regex parser for common PDS4/ISRO XML tags without requiring heavy lxml
            sensor_match = re.search(r"<instrument_name>([^<]+)</instrument_name>", content, re.IGNORECASE) or \
                           re.search(r"<instrument_id>([^<]+)</instrument_id>", content, re.IGNORECASE)
            if sensor_match:
                meta["sensor"] = sensor_match.group(1).strip()

            gsd_match = re.search(r"<pixel_resolution[^>]*>([^<]+)</pixel_resolution>", content, re.IGNORECASE) or \
                        re.search(r"<ground_resolution[^>]*>([^<]+)</ground_resolution>", content, re.IGNORECASE)
            if gsd_match:
                meta["gsd_m"] = _safe_float(gsd_match.group(1).strip())

            inc_match = re.search(r"<solar_incidence_angle[^>]*>([^<]+)</solar_incidence_angle>", content, re.IGNORECASE)
            if inc_match:
                meta["solar_incidence_deg"] = _safe_float(inc_match.group(1).strip())

            az_match = re.search(r"<solar_azimuth_angle[^>]*>([^<]+)</solar_azimuth_angle>", content, re.IGNORECASE)
            if az_match:
                meta["solar_azimuth_deg"] = _safe_float(az_match.group(1).strip())

            target_match = re.search(r"<target_name>([^<]+)</target_name>", content, re.IGNORECASE)
            if target_match:
                meta["target"] = target_match.group(1).strip()

    except Exception:
        pass
    return meta


def read_pds_label(file_path: str | Path) -> dict[str, Any]:
    """Parse and return dictionary of PDS label key-values and parsed fields."""
    p = Path(file_path)
    try:
        content = p.read_text(encoding="utf-8", errors="replace")[:100000]
        raw = parse_pds3_label_text(content)
        parsed = extract_metadata_from_file(p)
        return {**raw, **parsed}
    except Exception:
        return {}



def read_scientific_raster(
    file_path: Path,
    sidecar_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """Attempt to decode a scientific raster.

    Returns:
        (image_array, error_message, extracted_metadata)
    """
    ext = file_path.suffix.lower()
    meta = dict(sidecar_meta or {})

    # 1. Standard read through OpenCV first (handles GeoTIFF, JP2, TIFF, PNG, BMP)
    if ext in STANDARD_IMAGE_EXTS or ext in {".jp2", ".tif", ".tiff"}:
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                channels = int(img.shape[2]) if img.ndim == 3 else 1
                meta["dimensions"] = [h, w, channels]
                return img, None, meta
        except Exception:
            pass

    # 2. Handle PDS .IMG format
    if ext == ".img":
        # Check if OpenCV can directly decode it (some IMG are raw TIFFs or standard rasters)
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                meta["dimensions"] = [h, w, 1 if img.ndim == 2 else img.shape[2]]
                return img, None, meta
        except Exception:
            pass

        # Check for attached or detached PDS label
        label_text = ""
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(8192)
                text_peek = header_bytes.decode("ascii", errors="ignore")
                if "PDS_VERSION_ID" in text_peek or "RECORD_TYPE" in text_peek:
                    label_text = text_peek
        except Exception:
            pass

        # If detached label exists (e.g. .lbl with same name in same directory)
        detached_lbl = file_path.with_suffix(".lbl")
        if not label_text and detached_lbl.exists():
            try:
                label_text = detached_lbl.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        if label_text:
            pds_info = parse_pds3_label_text(label_text)
            lines = _safe_int(pds_info.get("LINES") or pds_info.get("IMAGE_LINES"))
            samples = _safe_int(pds_info.get("LINE_SAMPLES") or pds_info.get("IMAGE_LINE_SAMPLES"))
            sample_bits = _safe_int(pds_info.get("SAMPLE_BITS") or 8)
            rec_bytes = _safe_int(pds_info.get("RECORD_BYTES") or 0)
            ptr_image = _safe_int(pds_info.get("^IMAGE") or 1)

            if lines and samples:
                dtype = np.uint8 if sample_bits == 8 else np.uint16 if sample_bits == 16 else np.float32
                expected_bytes = lines * samples * (sample_bits // 8)
                file_size = file_path.stat().st_size
                header_offset = (ptr_image - 1) * rec_bytes if rec_bytes and ptr_image > 1 else (file_size - expected_bytes if file_size >= expected_bytes else 0)
                if header_offset >= 0 and file_size >= header_offset + expected_bytes:
                    try:
                        raw = np.fromfile(str(file_path), dtype=dtype, count=lines * samples, offset=header_offset)
                        raw = raw.reshape((lines, samples))
                        # Normalize to uint8 for visual registration processing
                        if raw.dtype != np.uint8:
                            norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        else:
                            norm = raw
                        meta["dimensions"] = [lines, samples, 1]
                        meta["sample_bits"] = sample_bits
                        return norm, None, meta
                    except Exception as err:
                        return None, f"Failed reading raw PDS raster bytes: {err}", meta

        return None, "Unsupported scientific raster format: .IMG requires detached PDS label with valid image pointers.", meta

    # 3. For any other raster that failed to decode
    return None, f"Unsupported scientific raster format: {ext.upper()} parser not available in current environment.", meta


class DatasetScanner:
    """Recursively scans folders, handles zip archives, parses rasters and sidecars,

    and produces an authoritative Dataset Manifest.
    """

    def __init__(self, output_dir: str | Path | None = None, public_prefix: str = "/api/outputs/manifests"):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "selene_manifests"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.public_prefix = public_prefix.rstrip("/")

    def scan_directory(self, folder_path: str | Path, dataset_name: str | None = None) -> dict[str, Any]:
        root = Path(folder_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        name = dataset_name or root.name or "Lunar Dataset"
        manifest_id = f"manifest_{int(cv2.getTickCount())}"
        manifest_dir = self.output_dir / manifest_id
        previews_dir = manifest_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        # 1. Unpack any nested zip files to temporary directory for scanning
        extracted_dirs: list[Path] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if Path(fname).suffix.lower() in ARCHIVE_EXTS:
                    zip_path = Path(dirpath) / fname
                    target_extract = manifest_dir / "extracted" / zip_path.stem
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(target_extract)
                            extracted_dirs.append(target_extract)
                    except Exception:
                        pass

        # 2. Gather all files recursively
        all_scan_paths: list[Path] = []
        for p in root.rglob("*"):
            if p.is_file():
                all_scan_paths.append(p)
        for ext_dir in extracted_dirs:
            for p in ext_dir.rglob("*"):
                if p.is_file():
                    all_scan_paths.append(p)

        # 3. Separate rasters and metadata files
        metadata_map: dict[str, Path] = {}
        for p in all_scan_paths:
            ext = p.suffix.lower()
            if ext in METADATA_EXTS:
                metadata_map[p.stem.lower()] = p

        files_manifest: list[dict[str, Any]] = []
        images_count = 0
        metadata_count = len(metadata_map)
        unsupported_count = 0
        total_size_bytes = 0

        for p in all_scan_paths:
            total_size_bytes += p.stat().st_size
            ext = p.suffix.lower()
            if ext in METADATA_EXTS or ext in ARCHIVE_EXTS:
                continue

            rel_path = str(p.relative_to(root)) if root in p.parents else p.name
            size_bytes = p.stat().st_size
            stem_key = p.stem.lower()

            # Check if associated metadata exists
            sidecar_meta: dict[str, Any] = {}
            has_metadata = False
            if stem_key in metadata_map:
                sidecar_meta = extract_metadata_from_file(metadata_map[stem_key])
                has_metadata = bool(sidecar_meta)

            # Check format type
            is_std = ext in STANDARD_IMAGE_EXTS
            is_sci = ext in SCIENTIFIC_RASTER_EXTS

            if not is_std and not is_sci:
                unsupported_count += 1
                files_manifest.append({
                    "filename": p.name,
                    "relative_path": rel_path,
                    "absolute_path": str(p),
                    "format": ext.upper().lstrip(".") or "UNKNOWN",
                    "size_bytes": size_bytes,
                    "dimensions": None,
                    "metadata_status": "UNAVAILABLE",
                    "metadata": {},
                    "status": "UNSUPPORTED",
                    "reason": f"Non-raster file format ({ext})",
                    "preview_url": None,
                })
                continue

            # Attempt to decode raster
            img, err, extracted_meta = read_scientific_raster(p, sidecar_meta)
            combined_meta = {**sidecar_meta, **extracted_meta}

            if img is not None:
                images_count += 1
                h, w = img.shape[:2]
                channels = int(img.shape[2]) if img.ndim == 3 else 1
                preview_filename = f"thumb_{images_count}_{p.stem}.jpg"
                preview_path = previews_dir / preview_filename
                
                # Decouple display preview from scientific raster
                from app.services.scientific_data import (
                    compute_scientific_inspection,
                    identify_sensor_evidence,
                    create_display_preview,
                )
                disp_preview = create_display_preview(img)
                save_image(preview_path, resize_for_preview(disp_preview, max_side=320))

                sci_inspect = compute_scientific_inspection(img)
                sensor_id = identify_sensor_evidence(
                    filename=p.name,
                    metadata=combined_meta,
                    shape=img.shape,
                    dtype=img.dtype,
                    channels=channels,
                )

                sensor_name = sensor_id.sensor
                gsd_val = sensor_id.gsd_m or combined_meta.get("gsd_m")
                meta_summary = f"{sensor_name} ({gsd_val}m/px)" if gsd_val else sensor_name

                files_manifest.append({
                    "filename": p.name,
                    "relative_path": rel_path,
                    "absolute_path": str(p),
                    "format": ext.upper().lstrip("."),
                    "size_bytes": size_bytes,
                    "dimensions": [w, h, channels],
                    "width": w,
                    "height": h,
                    "channels": channels,
                    "bit_depth": sci_inspect.bit_depth,
                    "dtype": sci_inspect.dtype,
                    "scientific_bands": sci_inspect.scientific_bands,
                    "layer_classification": sci_inspect.layer_classification,
                    "min_dn": sci_inspect.min_dn,
                    "max_dn": sci_inspect.max_dn,
                    "mean_dn": sci_inspect.mean_dn,
                    "median_dn": sci_inspect.median_dn,
                    "std_dn": sci_inspect.std_dn,
                    "dynamic_range_span": sci_inspect.dynamic_range_span,
                    "valid_pixel_pct": sci_inspect.valid_pixel_pct,
                    "saturation_pct": sci_inspect.saturation_pct,
                    "histogram": {
                        "bins": sci_inspect.histogram_bins,
                        "counts": sci_inspect.histogram_counts,
                    },
                    "sensor": sensor_id.sensor,
                    "mission": sensor_id.mission,
                    "gsd_m": sensor_id.gsd_m,
                    "sensor_confidence_pct": sensor_id.confidence_pct,
                    "sensor_evidence": sensor_id.evidence,
                    "metadata_status": sensor_id.metadata_status if sensor_id.metadata_status != "UNKNOWN" else ("AVAILABLE" if combined_meta else "UNAVAILABLE"),
                    "metadata_summary": meta_summary,
                    "metadata": combined_meta,
                    "status": "READY",
                    "reason": None,
                    "preview_url": f"{self.public_prefix}/{manifest_id}/previews/{preview_filename}",
                })
            else:
                unsupported_count += 1
                files_manifest.append({
                    "filename": p.name,
                    "relative_path": rel_path,
                    "absolute_path": str(p),
                    "format": ext.upper().lstrip("."),
                    "size_bytes": size_bytes,
                    "dimensions": None,
                    "metadata_status": "AVAILABLE" if combined_meta else "UNAVAILABLE",
                    "metadata_summary": "Metadata extracted, raster decoding unsupported",
                    "metadata": combined_meta,
                    "status": "UNSUPPORTED",
                    "reason": err or "Unsupported scientific raster format",
                    "preview_url": None,
                })

        manifest = {
            "dataset_name": name,
            "manifest_id": manifest_id,
            "root_path": str(root),
            "files_discovered": len(all_scan_paths),
            "images_count": images_count,
            "metadata_count": metadata_count,
            "unsupported_count": unsupported_count,
            "total_size_bytes": total_size_bytes,
            "files": files_manifest,
        }

        # Save manifest JSON to disk
        manifest_file = manifest_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest


LocalDatasetScanner = DatasetScanner


def scan_local_dataset(
    folder_path: str | Path,
    dataset_name: str | None = None,
    output_dir: str | Path | None = None,
    public_prefix: str = "/api/outputs/manifests",
) -> dict[str, Any]:
    """Convenience helper to scan a local folder and generate a dataset manifest."""
    scanner = DatasetScanner(output_dir=output_dir, public_prefix=public_prefix)
    return scanner.scan_directory(folder_path, dataset_name=dataset_name)

