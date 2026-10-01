# SELENE-REG — SIH 26166

**Multi-modal lunar image correspondence & registration prototype**

SELENE-REG is a local FastAPI + OpenCV application for pairwise lunar image
registration and relative multi-image registration. It is designed around
the SIH 26166 workflow: preprocessing → feature correspondence → geometric
verification → relative placement → measurable registration evidence.

## What is implemented

### Pair Registration
- SIFT / ORB / AKAZE feature detection
- percentile normalization + CLAHE/detail preprocessing
- Lowe-ratio matching
- spatially distributed match selection
- RANSAC homography estimation
- optional ECC intensity refinement
- RMSE, inlier count, inlier ratio and 4×4 spatial coverage
- registered product + correspondence visualization

### Multi-Image Map Builder
- 2–12 image collection
- thumbnail candidate screening
- cached image/feature/pair evidence
- incremental reuse when an image set is repeated or expanded
- registration graph with accepted/rejected relationships
- largest connected component placement
- relative image-space mosaic with feathered blending
- real inlier-point overlay
- independent footprint and match-point controls
- pair inspector + spatial 4×4 grid
- processing/cache diagnostics
- sensor metadata summary
- graph / CSV / JSON scientific-report exports
- expanded evidence metrics: P90/max residual, residual spread, spatial entropy/uniformity, transform conditioning and evidence score
- pair-level SSIM / PSNR / NMI when a registered raster is produced
- residual-aware inlier-point records (`dx`, `dy`, error)
- registered-mosaic 3D explorer that consumes the actual generated mosaic texture and supports orbit/zoom/top/reset controls plus image-derived relative relief

### Robustness Lab
A controlled synthetic stress-test endpoint evaluates the current pipeline under
known rotation, scale, illumination, blur and noise perturbations.

**Important:** these are software robustness tests. They are not evidence of
real Chandrayaan-2 ↔ LROC/SELENE cross-mission accuracy.

## Scientific scope

The multi-image output is intentionally labelled:

**Relative Registered Lunar Mosaic**

It is not claimed to be a georeferenced lunar map unless mission control points,
navigation/attitude metadata, a reference coordinate system and an appropriate
map projection are supplied.

No latitude/longitude or spacecraft metadata is fabricated by the application.

## Running on Windows

From PowerShell:

```powershell
Set-Location "D:\Branches\SIH26166\SELENE-REG-SIH26166"
backend\venv\Scripts\python.exe serve.py
```

Open:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

## Validation

The packaged source was validated with:

```text
12 tests passed
JavaScript syntax check passed
FastAPI health endpoint passed
FastAPI documentation endpoint passed
Synthetic robustness API smoke test passed (7/7 controlled cases)
```

## Project boundaries

The application currently uses raster uploads (PNG/JPEG/BMP/TIFF/WebP).
Mission-specific PDS/ISIS ingestion should be added with a validated reader
before treating native mission archives as directly ingestible.

External OneDrive/cloud ingestion is intentionally not enabled by default:
authenticated connectors should be added rather than allowing arbitrary remote
URL fetching.

GPU diagnostics report only acceleration that the installed runtime actually
exposes. The current classical OpenCV pipeline retains a CPU fallback.

## Presentation language

Use:
- "relative image-space registration"
- "registered correspondences"
- "RANSAC inliers"
- "measured RMSE / inlier ratio / spatial coverage"
- "synthetic robustness stress test"

Avoid claiming:
- georeferenced map coordinates without control data
- cross-mission accuracy without mission-data validation
- sub-pixel truth from ECC alone
- GPU acceleration unless the actual runtime uses it
- automated geological feature identification unless separately validated


## SELENE-REG X enhancement boundary

The 3D explorer is deliberately a **relative registered terrain visualization** when the job has no valid lunar control/georeferencing metadata. It uses the actual generated mosaic as its texture and can derive a relative visual relief field from image intensity; it does not invent latitude/longitude or claim a scientific DEM. Latitude/longitude and elevation become eligible only when validated georeferencing/DEM inputs are supplied.

The baseline pair and multi-image engine remains the source of truth. Advanced learned matchers, native mission PDS/ISIS ingestion, DEM-backed orthorectification and georeferenced GeoTIFF export remain data/model-dependent and are not represented as completed capabilities without the required assets.
