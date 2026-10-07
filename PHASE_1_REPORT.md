# PHASE 1 REPORT — Scientific Image Ingestion & Radiometric Data Inspection

**Project:** SELENE-REG-X — SIH Problem Statement 26166  
**Phase:** Phase 1 — Complete Image Ingestion & Scientific Data Inspection  
**Status:** Completed & Verified  
**Date:** 2026-10-02  

---

## 1. Executive Summary

Phase 1 establishes a scientific ingestion foundation for lunar optical and multispectral remote sensing data. Prior to this phase, ingestion across the frontend and worker relied on manual text-path inputs or implicit assumptions that clipped 16-bit sensor data down to 8 bits upon reading, fabricated sensor assignments without evidence verification, and lacked radiometric telemetry.

All Phase 1 requirements have been implemented in-place without rebuilding or discarding working registration pipelines:
1. **Native Folder Browsing & Multi-File Selection**: The application now supports native OS directory selection using browser `<input type="file" webkitdirectory directory multiple />` alongside recursive server-side dataset scanning and batch archive decompression.
2. **Evidence-Based Sensor Auto-Identification**: Replaced arbitrary guesswork with a rule-based deterministic classifier that inspects filename conventions, PDS3/PDS4 labels, GeoTIFF GDAL metadata tags, aspect ratios, and spatial resolutions. Unknown rasters remain explicitly labeled `UNKNOWN / NEEDS METADATA` with zero fabricated metrics or confidence scores.
3. **Layer Count Investigation & Clarification**: Clarified the ambiguity between raster memory channels, spectral bands, and UI rendering layers, displaying explicit diagnostic notes.
4. **Radiometric & Bit-Depth Inspection**: Extracted native data types (`uint8`, `uint16`, `int16`, `float32`), calculated dynamic range spans, valid vs. saturated pixel percentages, mean, median, standard deviation, and computed a 32-bin radiometric histogram.
5. **Preservation of Raw Scientific Values**: Raw DN values are strictly preserved in their native bit-depth (`raw_data`), while an independent 8-bit contrast-normalized raster (`display_data`) is generated solely for visualization and OpenCV feature detectors.

All 267 automated tests pass (266 passed, 1 skipped).

---

## 2. Ingestion Architecture Upgrades

### 2.1 Native Folder & Multi-File Selection
- **Frontend Directory Upload (`DatasetManifestModal.tsx`)**:
  - Added native folder picker (`data-testid="button-browse-folder"`) utilizing `<input type="file" webkitdirectory directory multiple />`.
  - Added multi-file picker (`data-testid="button-browse-files"`) supporting simultaneous batch selection of GeoTIFF, TIFF, PNG, and JPEG files.
  - Automatically transfers folder structures via `POST /api/dataset/upload-folder`, reconstructing directory trees and triggering metadata scanning.
- **DropZone & Workstation Inspection (`App.tsx` & `DropZone`)**:
  - Added direct **[Inspect Data]** button (`data-testid="button-inspect-source"` and `data-testid="button-inspect-reference"`) on moving and fixed raster cards.
  - Enables instant bit-depth and radiometric inspection of any loaded raster prior to launching registration.

### 2.2 Scientific Data Service (`scientific_data.py`)
Implemented `app.services.scientific_data` with:
- `ScientificRaster`: Immutable dataclass encapsulating `raw_data` (native dtype), `display_data` (uint8 preview), bit depth, channels, shape, and sensor identification.
- `ingest_scientific_image()`: Safely loads TIFF/GeoTIFF, PNG, and JPEG files using OpenCV (`IMREAD_UNCHANGED`), Pillow, and TIFF tag parsers. Decouples raw DN matrix from UI display normalization.
- `compute_scientific_inspection()`: Calculates exact radiometric statistics across all channels.
- `identify_sensor_evidence()`: Runs multi-factor evidence classification.

---

## 3. Evidence-Based Sensor Auto-Identification

The identification engine evaluates file signatures, dimensions, and metadata without fabricating mission origins:

| Target Sensor | Mission / Instrument | Identification Evidence Rules | Nominal GSD | Supported Bit-Depths |
|---|---|---|---|---|
| **OHRC** | Chandrayaan-2 Optical High Resolution Camera | Pattern `ch2_ohr_ncp_...`, `*OHRC*`, or aspect ratio > 4.0 with GSD ~ 0.25 m | 0.25 m/px | 10-bit / 12-bit in `uint16`, `uint8` |
| **TMC-2** | Chandrayaan-2 Terrain Mapping Camera-2 | Pattern `ch2_tmc_...`, `*TMC2*`, `*TMC-2*`, or 3-view stereo triplets (Fore/Nadir/Aft) | 5.0 m/px | 10-bit / 12-bit in `uint16`, `uint8` |
| **TMC** | Chandrayaan-1 Terrain Mapping Camera | Pattern `ch1_tmc_...` or `*TMC*` without `ch2` prefix | 5.0 m/px | 10-bit / 12-bit in `uint16`, `uint8` |
| **IIRS** | Chandrayaan-2 Imaging Infrared Spectrometer | Pattern `ch2_iir_...`, `*IIRS*`, or spectral bands count $\ge 200$ | 80.0 m/px | 12-bit / 14-bit in `uint16`, `float32` |
| **LROC NAC** | Lunar Reconnaissance Orbiter Narrow Angle Camera | Pattern `M[0-9]{7,10}[RLE]`, `*NAC*`, `*LROC_NAC*`, or long pushbroom strip (> 5000 px) | 0.50 m/px | 12-bit packed in `uint16`, `uint8` |
| **LROC WAC** | Lunar Reconnaissance Orbiter Wide Angle Camera | Pattern `*WAC*`, `*LROC_WAC*`, or 7-band multispectral push-frame | 100.0 m/px | 12-bit in `uint16`, `uint8` |
| **KAGUYA TC** | SELENE Kaguya Terrain Camera | Pattern `TC_...` or `*KAGUYA*` / `*SELENE_TC*` | 10.0 m/px | 10-bit / 12-bit in `uint16`, `uint8` |
| **UNKNOWN** | Metadata Missing / Unrecognized | No matching tags, regexes, or dimensions | User-defined | Preserved as loaded |

> **Verification Rule**: If no concrete metadata, regex, or aspect-ratio evidence matches, the system returns `UNKNOWN / NEEDS METADATA` with `confidence_pct = 0.0`. It does NOT default to OHRC or TMC-2 arbitrarily.

---

## 4. Layer Count Investigation & Clarification

### 4.1 Root Cause of Ambiguity
In GIS and remote sensing workflows, the term *"layers"* is frequently conflated across three distinct concepts:
1. **Memory Array Channels ($C$)**: The 3rd dimension of the loaded NumPy tensor (e.g., shape `(H, W, 1)` for panchromatic, `(H, W, 3)` for RGB/Bayer, or `(H, W, B)` for hyperspectral cubes).
2. **Spectral Bands**: Discrete physical wavelength intervals acquired by a sensor (e.g., 256 bands between 800 nm and 5000 nm for Chandrayaan-2 IIRS, 7 UV-VIS bands for LROC WAC, or 1 broad panchromatic band for OHRC).
3. **UI / Rendering Overlays**: Composition layers displayed on the frontend canvas (e.g., base reference raster, warped moving raster, checkerboard overlay, difference map, and correspondence vector arrows).

### 4.2 System Clarification & UI Diagnostics
`ScientificDataInspectorModal.tsx` and `scientific_data.py` now explicitly categorize the raster structure:
- **Panchromatic Mode (`channels == 1`)**: Single scientific band representing surface reflectance/radiance.
- **Multichannel Color/Bayer Mode (`channels == 3` or `4`)**: 3-channel optical or synthesized raster.
- **Hyperspectral Mode (`channels > 4`)**: Multi-band spectral cube with band index telemetry.
- **Diagnostic Note**: The inspector modal renders an informative callout disambiguating native file channels from UI composite layers.

---

## 5. Radiometric & Bit-Depth Inspection

For every ingested raster, the pipeline computes rigorous radiometric statistics across valid surface pixels:
- **Native Data Type**: Reported directly from NumPy array (`uint8`, `uint16`, `int16`, `float32`).
- **Bit-Depth Span**: Inferred based on dynamic range ceiling (e.g., 8-bit: $2^8-1=255$, 10-bit: 1023, 12-bit: 4095, 14-bit: 16383, 16-bit: 65535).
- **Statistical Moments**:
  - Minimum DN ($DN_{\min}$)
  - Maximum DN ($DN_{\max}$)
  - Dynamic Range Span ($\Delta DN = DN_{\max} - DN_{\min}$)
  - Mean ($\mu$)
  - Median
  - Standard Deviation ($\sigma$)
- **Saturation & Data Integrity**:
  - `saturated_pct`: Percentage of pixels at or above maximum saturation threshold ($255$ for 8-bit, $4095$ for 12-bit, $65535$ for 16-bit).
  - `valid_pct`: Percentage of pixels excluding non-data / border zero padding.
- **32-Bin Radiometric Histogram**: Computed uniformly across the raster's valid dynamic range and visualized as an interactive SVG histogram in the inspector modal.

---

## 6. End-to-End Test Suite & Verification Results

A dedicated test suite `tests/test_phase1_ingestion_inspection.py` was implemented to validate all Phase 1 components:

1. `test_16bit_raw_preservation_and_decoupled_preview`: Confirms 16-bit GeoTIFF retaining uint16 DN values up to 4095 without clamping or dynamic range compression, while providing a decoupled 8-bit display preview.
2. `test_sensor_auto_identification_evidence_matrix`: Tests classification across OHRC, TMC-2, TMC-1, IIRS, LROC NAC, LROC WAC, Kaguya, and unlabelled images (ensuring unlabelled data yields `UNKNOWN` with 0% confidence).
3. `test_layer_count_clarification`: Validates channel-to-spectral band categorization across 1-channel, 3-channel, and 32-channel hyperspectral cubes.
4. `test_radiometric_histogram_and_saturation`: Tests min, max, mean, std, 32-bin histogram sum, and saturation flag on synthetic over-saturated data.
5. `test_ingest_scientific_image_formats`: Verifies lossless reading and inspection across PNG, JPEG, TIFF (8-bit), and TIFF (16-bit).
6. `test_dataset_scanner_folder_ingestion_and_inspection`: Confirms recursive folder scanning generates complete manifest entries with attached `scientific_inspection` and `sensor_identification` payloads.
7. `test_worker_cli_inspect_image_mode`: Validates worker CLI `--mode inspect-image` execution, JSON response format, and schema conformance.

### Test Execution Output
```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Branches\Projects\SIH26166\SELENE-REG-X-SIH26166
configfile: pyproject.toml
plugins: anyio-4.14.2
collected 267 items

tests\test_dataset_scanner.py ....                                       [  1%]
tests\test_frontend_hooks_regression.py .                                [  1%]
tests\test_global_optimization.py ...                                    [  2%]
tests\test_hardware_manager.py ....                                      [  4%]
tests\test_mosaic_geometry.py .......                                    [  7%]
tests\test_mosaic_incremental_correctness.py ....                        [  8%]
tests\test_phase1.py ............                                        [ 13%]
tests\test_phase1_ingestion_inspection.py .......                        [ 15%]
tests\test_phase2.py ...............                                     [ 21%]
tests\test_phase3.py ........                                            [ 24%]
tests\test_phase4.py ............................                        [ 34%]
tests\test_phase5.py ....................                                [ 42%]
tests\test_phase6.py ...................                                 [ 49%]
tests\test_phase7.py ................................................... [ 68%]
tests\test_phase8.py ...................                                 [ 77%]
tests\test_progress.py ...s...                                           [ 80%]
tests\test_registration_correctness_foundation.py .......                [ 83%]
tests\test_registration_geometry_canvas.py .......                       [ 85%]
tests\test_registration_quality.py ................                      [ 91%]
tests\test_scalability_large_batch.py .....                              [ 93%]
tests\test_url_ingestion.py .................                            [100%]

================= 266 passed, 1 skipped in 205.32s (0:03:25) ==================
```

Frontend production build check:
```
vite v7.3.6 building client environment for production...
✓ 1778 modules transformed.
dist/public/index.html                   1.38 kB │ gzip:   0.55 kB
dist/public/assets/index-DIK0Js8H.css  122.52 kB │ gzip:  20.71 kB
dist/public/assets/index-BfLigQwM.js   539.58 kB │ gzip: 155.99 kB
✓ built in 6.71s
```

Backend build check:
```
artifacts\api-server\dist\index.mjs                   1.4mb
Done in 537ms
```

---

## 7. Phase 1 Completion Sign-off

Phase 1 — Image Ingestion + Scientific Data Inspection is fully implemented, verified, and integrated. All code changes have been tested in-place without regressions.

**Status: READY FOR PHASE 2 UPON USER INSTRUCTION.**
