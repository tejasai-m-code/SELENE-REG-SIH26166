# SELENE-REG X — Baseline Audit
Generated after extracting `SELENE-REG-SIH26166-FINAL.zip` before source modifications.

## Baseline inventory
- Archive files: 42
- Frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css` (vanilla HTML/CSS/JS)
- Backend: FastAPI + OpenCV + NumPy
- Pair pipeline: preprocessing → SIFT/ORB/AKAZE → ratio matching → spatial filtering → RANSAC homography → optional ECC → metrics
- Multi-image pipeline: thumbnail screening → feature cache → pair cache → graph → relative placement → feathered mosaic
- Evaluation: controlled synthetic robustness endpoint
- Tests: 4 test modules, 12 tests

## Verified baseline
| Check | Result |
|---|---|
| Automated tests | 12 passed |
| JavaScript syntax | passed (`node --check frontend/app.js`) |
| FastAPI health | 200 via in-process API smoke test |
| Frontend route | 200 |
| OpenAPI/docs | 200 |
| Pair API synthetic smoke | 200; real homography/inliers/RMSE returned |
| Multi-image API synthetic smoke | 200; 3 images placed; real mosaic returned |
| Mosaic inline data | real generated PNG + footprint/match-point metadata |
| Cache reuse | verified: repeat 3-image run reused 3/3 pair evidences |
| CPU/GPU diagnostics | CPU path reported truthfully |
| Baseline code changes before audit | none |

## Synthetic API evidence
Pair smoke case:
- 240×240 translated synthetic image
- 35 candidate matches
- 35 inliers
- inlier ratio 1.0
- RMSE ≈ 0.375 px
- source coverage 18.75%

Multi-image smoke case:
- 3 images
- 3 candidate pairs
- 3 accepted
- 3 placed
- generated mosaic: 250×245
- 188 projected inlier points in mosaic metadata
- first run: 3 cache misses
- repeat run: 3 cache hits, 0 new pair computations

## Performance observations
A five-image, 240×240 synthetic collection using the existing pair engine measured approximately:
- first run: 0.17 s wall time, 10 new pairs
- repeat run: 0.045 s wall time, 10 reused pairs

A larger 480×480 synthetic texture case measured approximately:
- first run: 1.51 s, 10 new pairs
- repeat run: 0.18 s, 10 reused pairs

The exact historical 0.369/0.227/0.301-second Phase-2 figures supplied in the project brief are retained as regression targets, not re-created claims.

## Existing feature preservation targets
- Pair registration
- 2–12 image multi-registration
- thumbnail candidate screening
- feature cache
- pair-evidence cache
- incremental reuse
- graph placement
- relative mosaic
- footprints overlay
- match-point overlay
- pair inspector
- spatial 4×4 evidence
- CSV/JSON exports
- synthetic robustness lab
- truthful CPU diagnostics

## Known baseline gaps discovered before enhancement
- no integrated 3D explorer consuming the generated mosaic
- no lunar cursor coordinate engine
- no PDS4/XML ingestion layer
- no model-selection engine beyond homography
- limited quality metrics
- no formal confidence gate explanation beyond pair acceptance thresholds
- no mission-report PDF
- no dedicated illumination/scale analysis UI
- `serve.py` does not mount the evaluation router even though `backend/main.py` does
- backend dependency file is minimal and contains no learned matcher / geospatial stack
- metadata is user-supplied rather than extracted from mission labels

These are enhancement targets, not reasons to replace the baseline.
