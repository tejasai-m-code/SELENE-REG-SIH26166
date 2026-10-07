# SELENE-REG Architecture Specification

## Overview

**SELENE-REG** is a multi-modal, illumination-robust lunar raster registration and relative mosaicking pipeline developed for SIH Problem Statement 26166 (*Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images: OHRC, TMC-2, and IIRS*).

The engine executes a classical, deterministic, non-learned computer vision pipeline with strict metadata provenance tracking and quantitative subpixel refinement.

---

## 12-Stage Pipeline Flow

```
[ INPUT RASTERS & METADATA ]
              │
              ▼
    1. Metadata Ingestion & Provenance Classification
              │
              ▼
    2. Radiometric Normalization (Safe Stretch / Calibration)
              │
              ▼
    3. Structural Representation (Gradient / CLAHE / Retinex)
              │
              ▼
    4. Multi-Scale Feature Detection (SIFT / ORB / AKAZE)
              │
              ▼
    5. Feature Description & Ratio-Test Matching
              │
              ▼
    6. Spatial Keypoint Distribution & Uniformity Filtering
              │
              ▼
    7. Robust Geometric Estimation (RANSAC Homography / Affine)
              │
              ▼
    8. Multi-Method Subpixel Refinement (ECC / Lucas–Kanade / Phase)
              │
              ▼
    9. Acceptance & Quality Gate Verification (PASS / REVIEW / FAIL)
              │
              ▼
   10. Multi-Image Graph Construction & Reference Selection
              │
              ▼
   11. Global Coordinate Propagation & Mosaic Feather Blending
              │
              ▼
   12. Geospatial Polygon Footprint & Metadata Ledger Export
```

---

## Detailed Stage Descriptions

### Stage 1: Metadata Ingestion & Provenance Classification
- **Input**: Raw raster bytes (PNG, TIFF, JP2, etc.) and optional label dictionaries.
- **Processing**:
  - Classifies metadata into explicit status categories: `KNOWN`, `REFERENCE_PROFILE`, `NOT_PROVIDED`, or `UNAVAILABLE`.
  - Stamps provenance: `MEASURED` (from user/file headers), `ESTIMATED` (inferred from sensor naming), `REFERENCE_PROFILE` (default sensor catalog value), or `UNAVAILABLE`.
  - Prevents hallucinated flight telemetry; unknown attributes remain strictly `UNKNOWN`.

### Stage 2: Radiometric Normalization
- **Purpose**: Map high-dynamic-range (10-bit to 16-bit) and 8-bit lunar imagery into stable computational representations without introducing saturation or clipping artifacts.
- **Modes**:
  - `safe_normalization`: Robust 1%–99% percentile stretch avoiding hot pixels and deep shadow clipping.
  - `metadata_calibration`: Converts raw DN to top-of-atmosphere radiance ($L = \text{gain} \times \text{DN} + \text{offset}$) when calibrated coefficients are supplied.

### Stage 3: Illumination-Robust Representations
- **Purpose**: Decouple surface albedo and crater topography from dynamic solar phase angles and extreme shadows.
- **Available Representations**:
  - `structural`: Morphological edge gradients invariant to uniform scale/contrast shifts.
  - `gradient`: Sobel gradient magnitude capturing morphological rim boundaries across differing sun angles.
  - `clahe`: Contrast-Limited Adaptive Histogram Equalization highlighting low-contrast crater interiors.
  - `retinex`: Multiscale Retinex estimating reflectance components under non-uniform illumination fields.
  - `percentile` / `raw`: Preserved radiometric intensity.

### Stage 4: Multi-Scale Feature Detection
- **Detectors**:
  - SIFT (Scale-Invariant Feature Transform): Primary detector for scale and rotation changes.
  - ORB (Oriented FAST and Rotated BRIEF): High-speed binary alternative.
  - AKAZE: Non-linear scale space detector.
- Features are extracted across octaves to support resolution discrepancies.

### Stage 5: Correspondence Matching
- L2 or Hamming distance nearest-neighbor search with Lowe's ratio test (default $\text{ratio} = 0.72 - 0.75$) to eliminate ambiguous matches.

### Stage 6: Spatial Keypoint Filtering & Distribution
- **Grid-based Binning**: Prevents keypoint clustering on single high-contrast crater rims by partitioning the image into an adaptive spatial grid.
- **Metrics Computed**:
  - Spatial Coverage: Fraction of spatial bins containing active correspondences.
  - Uniformity Index: Entropy/Gini ratio of spatial distribution across bins.

### Stage 7: Robust Geometric Estimation
- **Estimators**: RANSAC (RANdom SAmple Consensus) with MAGAC / USAC fallback.
- **Models**:
  - Homography ($3 \times 3$ projective transform): Full planar surface projection.
  - Affine ($2 \times 3$ affine transform): Constrained 6-DOF transform for small-field pairs.
- Inliers are isolated and geometric degeneracy is detected (collinear points, negative determinants).

### Stage 8: Subpixel Refinement
- Applied to inlier correspondences or globally to refine image alignment beyond integer pixel bounds:
  - **ECC (Enhanced Correlation Coefficient)**: Maximizes correlation coefficient between warped templates.
  - **Lucas–Kanade (Pyramidal Optical Flow)**: Iterative gradient-based displacement refinement.
  - **Phase Correlation**: Fourier-domain subpixel peak interpolation.
  - **Quadratic Peak Interpolation**: Parabolic fit around discrete correlation extrema.
- Reports initial RMSE, refined RMSE, and delta improvements.

### Stage 9: Acceptance & Quality Gate
- Evaluates output against strict criteria:
  - `PASS`: High inlier count ($\ge 15$), low RMSE ($< 2.5$ px), good spatial coverage ($\ge 0.40$), and non-degenerate geometry.
  - `PASS_WITH_WARNING`: Lower inlier count ($6 - 14$) or marginal coverage.
  - `REVIEW`: Insufficient overlap or borderline geometric stability.
  - `FAIL`: Zero inliers, degenerate geometry, or disjoint fields.

### Stage 10: Multi-Image Graph Construction & Reference Selection
- Builds an undirected pairwise registration graph $G = (V, E)$ where vertices are rasters and edge weights reflect inlier counts and inverse RMSE.
- Computes degree centrality and minimum spanning tree (MST) to choose the optimal reference image (minimizing cumulative registration drift).

### Stage 11: Global Coordinate Propagation & Mosaic Blending
- Chains pairwise homographies $H_{i \to \text{ref}} = H_{k \to \text{ref}} \cdot H_{i \to k}$ to place all images on a common mosaic canvas.
- Blends overlapping regions using distance-transform feathered seam masks to avoid visible boundary steps.

### Stage 12: Geospatial Polygon Footprint & Metadata Export
- Projects image bounding boxes through homographies into polygon footprints.
- Computes exact polygon intersections, unions, and IoU overlap ratios.
- Emits structured JSON containing full provenance, transform matrices, subpixel metrics, and visual artifacts.
