# SELENE-REG X — Feature Preservation Checklist

## Baseline functionality
- [x] Pair registration route
- [x] SIFT
- [x] ORB
- [x] AKAZE
- [x] percentile normalization
- [x] CLAHE/detail preprocessing
- [x] Lowe ratio matching
- [x] spatial 4×4 selection
- [x] RANSAC homography
- [x] optional ECC refinement
- [x] RMSE/inlier/inlier-ratio/coverage
- [x] registered product
- [x] correspondence visualization
- [x] 2–12 image collection
- [x] thumbnail screening
- [x] feature cache
- [x] pair-evidence cache
- [x] incremental reuse
- [x] registration graph
- [x] relative mosaic
- [x] footprints overlay
- [x] match-point overlay
- [x] pair inspector
- [x] graph/CSV/report JSON exports
- [x] synthetic robustness lab
- [x] truthful CPU diagnostics

## Master-product additions
- [ ] richer metadata/PDS4 ingestion
- [ ] illumination lab
- [ ] scale analysis
- [ ] model selection (similarity/affine/homography)
- [ ] expanded quality engine
- [ ] explicit confidence gate
- [ ] residual analysis
- [ ] manual control points
- [ ] synthetic known-transform validation
- [ ] cycle consistency
- [ ] actual mosaic-to-3D workflow
- [ ] lunar cursor lat/lon
- [ ] DEM-backed terrain when real DEM is supplied
- [ ] mission report PDF
- [ ] PS-26166 audit evidence page
- [ ] premium mission-control visual system

## Rules
A box is checked only after implementation and an executable regression test demonstrate it. Unsupported mission data remains explicitly labeled as unavailable.
