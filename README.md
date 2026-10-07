# SELENE-REG: Multi-Modal Lunar Image Registration

**SIH Problem Statement 26166**: *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).*

---

## 1. Project Purpose & Scope

**SELENE-REG** is a high-precision, non-learned classical computer vision pipeline designed for registering and mosaicking planetary orbital rasters under severe radiometric, geometric, and illumination disparities.

### Core Capabilities
- **Illumination Robustness**: Handles extreme lunar shadow patterns, incidence angle variations, and contrast inversions via morphological structural representations, Multiscale Retinex, and CLAHE.
- **Multiscale Correspondence**: Overcomes scale and rotation differences using SIFT/ORB feature pyramids and Lowe's ratio test.
- **Spatial Distribution Enforcing**: Prevents correspondence clustering on single crater rims via adaptive spatial grid binning.
- **Robust Geometry**: Rejects false matches with RANSAC-verified Homography and Affine models.
- **Subpixel Refinement**: Refines alignment beyond integer pixels using Enhanced Correlation Coefficient (ECC), Lucas–Kanade optical flow, Fourier Phase Correlation, and Quadratic Peak fitting.
- **Multi-Image Mosaicking**: Builds pairwise registration graphs, automatically determines reference frames via maximum degree centrality, and blends rasters using distance-transform feathered seams.
- **Polygon Footprints & Geospatial Layer**: Computes exact polygon boundaries, bounding boxes, intersection areas, and IoU overlap ratios.
- **Rigorous Metadata Provenance**: Tracks data status (`KNOWN`, `REFERENCE_PROFILE`, `NOT_PROVIDED`, `UNAVAILABLE`) and provenance (`MEASURED`, `ESTIMATED`, `REFERENCE_PROFILE`) without inventing flight telemetry.

---

## 2. Architecture Overview

The system operates as a 12-stage sequential pipeline:

```
Input Rasters → Metadata Classification → Radiometric Normalization →
Structural Representation → Multiscale Feature Detection → Matching →
Spatial Keypoint Filtering → RANSAC Homography → Subpixel Refinement →
Acceptance Quality Gate → Multi-Image Graph & Placement → Footprint & Mosaic Export
```

For detailed mathematical specifications and stage parameters, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 3. Quickstart & Installation

### Prerequisites
- **Python**: Version 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- **Node.js**: Version 18+ (tested on Node 20, 24 with pnpm or npm)

### Python Dependencies Installation
Install the computer vision engine dependencies:

```bash
pip install -r pyproject.toml
# Or install directly:
pip install numpy opencv-python-headless pytest
```

---

## 4. Launching the System

### One-Command Unified Startup (Windows / Linux / macOS)
The fastest, most reliable way to start both the Python-backed API server (Port 5000) and the Vite frontend (Port 5173):

```powershell
# In Windows PowerShell:
node scripts/launch.mjs
```

Or using pnpm:
```powershell
pnpm.cmd dev
```

This starts:
- **Backend API**: `http://localhost:5000` (auto-builds if needed, hosts `/api/register` and `/api/ingest-url`)
- **Frontend Workstation**: `http://localhost:5173` (proxies `/api` requests to port 5000)

---

## 5. Input Workflows & Demonstration

The scientific workstation accepts imagery through three standardized input paths:

### A. Local Image Upload
- Click or drag & drop images into the **Moving source raster** or **Fixed reference raster** dropzones.
- **Supported Formats**: PNG, JPEG, TIFF/GeoTIFF, BMP, WEBP.
- Original image bytes are preserved and decoded directly through the OpenCV scientific engine.

### B. Remote Public HTTPS URL Ingestion
- Switch to the **URL / Drive** tab in the raster dropzone.
- Paste a direct public image URL (e.g. `https://example.com/lunar_crater.png`) and click **Load**.
- The server validates URL syntax, enforces strict SSRF checks, streams and verifies image bytes with OpenCV, generates a preview thumbnail, and populates the input.
- Click **Run correspondence** to process with the identical scientific registration pipeline.

### C. Public Google Drive Links
- Paste any standard public Google Drive sharing URL into the **URL / Drive** tab:
  - `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
  - `https://drive.google.com/open?id=FILE_ID`
- The system safely extracts the file ID, normalizes the direct download URL, and processes the image.
- **Important Requirement**: The Google Drive file MUST be set to *"Anyone with the link can view/download"*. Private, login-gated, or domain-restricted files are rejected with clear actionable recovery instructions.

### D. Instant Verification (Demo Pair)
- Click the **Demo Pair** button in the *Assemble the pair* card header.
- Automatically loads the verified synthetic Chandrayaan-2 pair (`synthetic_source.png` and `synthetic_reference.png`).
- Click **Run correspondence** to run the complete feature matching, RANSAC, and subpixel refinement pipeline.

---

## 6. Automated Testing & Verification

The repository contains an exhaustive automated test suite with **193/193 passing tests**:

```powershell
# In Windows PowerShell:
$env:PYTHONPATH="artifacts/api-server/python"
python -m unittest discover -s tests -p "test_*.py"
```

### Test Suite Structure
- `tests/test_phase1.py`: Baseline detector & matching verification (12 tests)
- `tests/test_phase2.py`: Illumination representation & CLAHE (15 tests)
- `tests/test_phase3.py`: Subpixel refinement (ECC, LK, Phase) (8 tests)
- `tests/test_phase4.py`: Cross-modal proxy & gradient invariance (28 tests)
- `tests/test_phase5.py`: Multi-image registration graph & mosaic quality (20 tests)
- `tests/test_phase6.py`: Metadata provenance, status & polygon footprints (19 tests)
- `tests/test_phase7.py`: End-to-end integration & contract verification (57 tests)
- `tests/test_phase8.py`: Quantitative benchmarks, stress testing & reproducibility (18 tests)
- `tests/test_url_ingestion.py`: Remote URL ingestion, SSRF protection & Google Drive (15 tests)

---

## 7. Security Hardening & Limitations

### SSRF (Server-Side Request Forgery) Defense
The remote URL ingestion service enforces multi-layered SSRF defenses:
- **Scheme Restriction**: Only `http://` and `https://` are permitted. Schemes such as `file://`, `ftp://`, `data:`, `javascript:` are rejected immediately.
- **Loopback & Private Network Blocking**: Rejects `localhost`, `127.0.0.0/8`, `::1`, RFC 1918 private IPv4 (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), IPv6 ULA (`fc00::/7`), and internal domain names (`.local`, `.internal`, `.lan`).
- **Cloud Metadata Protection**: Explicitly blocks `169.254.169.254` and all link-local spaces (`169.254.0.0/16`, `fe80::/10`).
- **DNS Re-Validation & Safe Redirects**: Resolves DNS before connection and re-validates resolved IPs at every redirect hop (maximum 5 redirects).
- **Resource Constraints**: Enforces 10-second connect timeout, 15-second read timeout, and an 80 MB maximum streaming file size limit.
- **Strict Byte Sniffing**: Validates that downloaded bytes are genuine decodable raster images (rejecting HTML error/login pages and JSON).

### Known Boundaries
- **Private Cloud Storage**: Accessing private Google Drive, Dropbox, or AWS S3 files requiring user authentication/OAuth is deliberately not supported. Users must make shared files publicly accessible or download them locally first.
- **Synthetic Ground Truth**: Quantitative benchmarks are evaluated on controlled, procedurally generated lunar surfaces.
- **Flight Data Validation Pending**: No uncalibrated Level-1/Level-2 Chandrayaan-2 PDS4 rasters or physical SPICE ephemerides were bundled or fabricated.
- **Relative Coordinate Space**: The mosaic engine computes relative image-space placements; absolute lunar georeferencing requires external SPICE kernels and DEM models.
- **No Claim of Flight Endorsement**: The system is an academic / hackathon technical prototype and does not claim formal certification or endorsement by ISRO.

See [docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md) and [docs/BENCHMARK.md](docs/BENCHMARK.md) for complete details.
