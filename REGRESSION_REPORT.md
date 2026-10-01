# SELENE-REG X — Regression Report

## Verification run
- Python test suite: **15 passed**
- Frontend JavaScript syntax: **passed**
- FastAPI `/health`: **200**
- Frontend `/`: **200**
- FastAPI `/docs`: **200**
- `/api/demo`: **200**
- Pair registration synthetic smoke: **200**
- Synthetic robustness endpoint: **200, 7/7 cases passed**

## Performance check
Using the preserved classical engine on a deterministic 5-image, 240×240 synthetic translated collection with SIFT and `max_features=2000`:
- first run: approximately **0.225 s**, 10 new pair evaluations, 5 images placed
- immediate repeat: approximately **0.061 s**, 0 new pairs, 10 reused
- third repeat: approximately **0.046 s**, 0 new pairs, 10 reused

A separate 480×480 textured collection measured approximately 1.51 s first-run and 0.18 s repeat-run before the X enhancements; this confirms the cache remains the dominant incremental speed path.

## Regression outcomes
- Pair registration behavior preserved.
- Multi-image graph and relative mosaic tests preserved.
- Feature/pair cache architecture preserved.
- Footprint/match-point overlay data preserved.
- Existing 12 baseline tests remain green; 3 additional master-quality tests were added.
- Synthetic robustness endpoint remains operational in the actual `serve.py` application.
- `serve.py` now includes the evaluation router; this fixes an integration gap where the frontend robustness control was not exposed by the packaged presentation server.

## New verified behavior
- P90/max residual metrics.
- residual standard deviation.
- 8×8 spatial entropy/uniformity and largest-cluster fraction.
- transform conditioning.
- descriptive evidence score.
- lightweight SSIM/PSNR/NMI for pair runs that generate a registered raster.
- residual-aware inlier point serialization (`dx`, `dy`, error).
- 3D explorer contract: actual generated mosaic URL is passed to the WebGL texture loader.
- 3D viewer explicitly labels the terrain as relative when georeferencing is absent.

## Not yet claimed as complete
- native PDS4/ISIS ingestion
- learned SuperPoint/LightGlue/LoFTR production path
- RIFT/phase-congruency production matcher
- validated DEM-backed terrain
- scientifically georeferenced lunar lat/lon
- mission-dataset cross-mission accuracy
- sub-pixel scientific validation against external ground truth

These remain evidence/data-dependent work items rather than mocked capabilities.
