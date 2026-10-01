# SELENE-REG — Lunar Image Correspondence & Registration

Presentation-ready prototype backend for **SIH 26166: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)**.

## What this version does

The API implements a practical classical baseline:

1. Image ingestion
2. Grayscale conversion + percentile normalization
3. CLAHE / illumination normalization
4. SIFT / ORB / AKAZE feature extraction
5. Lowe-ratio descriptor matching
6. Optional spatial distribution constraint (4×4 grid)
7. RANSAC homography estimation
8. Optional ECC intensity refinement
9. Warp source into reference coordinates
10. RMSE, mean/median reprojection error, inlier count, inlier ratio
11. Source/reference spatial coverage
12. Registered image + correspondence visualization

### Important scientific limitation

This is a strong **prototype baseline**, not a claim that the SIH requirement is already solved at sub-pixel accuracy for every cross-modal case.

Native Chandrayaan-2/LROC scientific products can be PDS/binary products with XML labels. This API intentionally accepts standard raster images (PNG/JPEG/TIFF/WEBP/BMP) so the web prototype remains reliable. A mission-specific PDS/ISIS ingestion module should be added when the actual downloaded products are available.

## Run

### Windows PowerShell

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open:

- Web app: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

The frontend is served by the backend when using the included `main.py` + static mount below.

## Serving the frontend

The simplest demo setup is to run from the project root with:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Then copy `frontend/` to `backend/static/` OR use the optional static mount in `main.py`.

For the zero-configuration presentation setup, use the included `serve.py` in the project root:

```powershell
python serve.py
```

## API

`POST /api/register`

Multipart form fields:

- `source`: source/moving image
- `reference`: fixed/reference image
- `detector`: `sift`, `orb`, `akaze`
- `ratio`: Lowe ratio, default `0.72`
- `ransac_threshold`: pixels, default `3.0`
- `illumination_normalization`: boolean
- `spatial_distribution`: boolean
- `ecc_refinement`: boolean
- `max_features`: default `8000`

The response contains:

- homography
- inlier points
- RMSE
- inlier count
- inlier ratio
- spatial coverage
- registered image URL
- correspondence visualization URL

## Recommended real-data demo

Use one of the following representative pairings:

- Chandrayaan-2 OHRC ↔ LROC NAC
- Chandrayaan-2 TMC-2 ↔ LROC WAC
- Chandrayaan-2 IIRS ↔ LROC WAC

The SIH statement points to the Chandrayaan-2 data explorer and LROC/QuickMap/reference imagery.

## Next research upgrade

For the full SIH solution, the next modules should be:

- RIFT / phase-congruency descriptors for illumination robustness
- multi-scale coarse-to-fine registration
- learned matcher such as LoFTR / SuperGlue
- sensor-aware preprocessing for OHRC/TMC-2/IIRS
- sub-pixel correspondence refinement
- stronger spatial-uniformity optimization
- DEM/orthorectification-aware geometric models when metadata is available
- benchmark evaluation across sun-angle and scale gaps
