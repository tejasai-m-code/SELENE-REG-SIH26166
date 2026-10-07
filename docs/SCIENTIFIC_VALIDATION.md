# Scientific Validation Ledger & Claim Boundaries

## Problem Statement Context
**SIH Problem Statement 26166**: *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS).*

This document establishes the verified scientific boundary of **SELENE-REG**. It delineates what has been empirically validated through reproducible quantitative testing versus what remains pending or requires real mission ground-truth.

---

## 1. Validated Capabilities (Empirically Proven)

The following capabilities have been tested and verified across **177 automated test suites** in controlled, synthetic, and proxy environments:

### A. Controlled Synthetic Pairwise Registration
- Verified on deterministic synthetic lunar surfaces featuring simulated crater topology, ring rims, and high-frequency surface noise.
- Homography and Affine model estimation achieves sub-pixel corner transfer error against synthetic ground-truth:
  - Translation Corner Error: $< 0.05$ px (Test verified: $0.0064$ px).
  - Affine Corner Error: $< 0.50$ px (Test verified: $0.1778$ px).
  - Perspective Corner Error: $< 0.50$ px (Test verified: $0.1318$ px).

### B. Fractional-Pixel Subpixel Refinement
- Validated on synthetic sub-pixel displacements ($0.25$ px, $0.50$ px, $0.75$ px):
  - ECC refinement reliably reduces correspondence RMSE by $30\% - 90\%$ on uncorrupted images.
  - Lucas–Kanade optical flow refines feature coordinates from initial integer detections to $< 0.15$ px residual error.
  - Phase correlation accurately identifies fractional shifts in Fourier frequency domain.

### C. Illumination & Perturbation Robustness
- **Brightness Shifts**: Inlier recovery preserved across $\pm 30$ DN offsets ($> 450$ inliers).
- **Contrast Shifts**: Tested and validated from $0.5\times$ to $1.5\times$ dynamic range stretch.
- **Nonlinear Gamma**: Tested and validated under gamma $\gamma \in [0.5, 2.0]$.
- **Local Shadows & Sun Direction**: Robust gradient and CLAHE representations recover $> 400$ inliers under directional illumination gradients and artificial crater shadow masks.
- **Cross-Modal Inversion Proxy**: Under complete contrast inversion (simulating emissivity vs. reflectance polarity), standard intensity matching yields 0 inliers, whereas structural gradient representation recovers $480$ inliers.

### D. Scale & Rotation Robustness
- Scale variation: Validated in range $[0.67\times, 1.50\times]$ with multi-scale SIFT pyramid.
- Rotation variation: Validated across full planar rotations $\theta \in [-30^\circ, +30^\circ]$.

### E. Partial Overlap Boundaries
- Validated down to $20\%$ spatial overlap with active inliers ($> 100$ inliers).
- Graceful degradation: Under extreme minimal overlap ($< 5\%$) or disjoint scenes ($0\%$), system safely triggers `FAIL` status without crashes.

### F. Multi-Image Graph & Relative Mosaicking
- Validated on 3-image, 5-image, and 8-image synthetic strips:
  - Global placement: $100\%$ placement rate ($3/3$, $5/5$, $8/8$).
  - Reference image selection via maximum spanning tree and degree centrality.
  - Pairwise transform chaining error $< 0.05$ px across 3 images.
  - Feathered multi-band blending eliminates seam discontinuities.

### G. Safe Failure & Degeneracy UX
- Zero-feature images (blank, constant grey, saturated white) safely fail with structured diagnostics (`status: FAIL`, $0$ inliers).
- Extreme blur and uncorrelated noise safely abort with clear evaluator messages.
- Non-deceptive UI: failed registrations are never rendered as successful.

---

## 2. Unvalidated Domains (Pending Mission Validation)

To ensure scientific honesty and avoid over-claiming, the following areas are explicitly documented as **NOT VALIDATED** on real mission flight hardware:

| Domain | Current Status | Why It Cannot Be Claimed |
|---|---|---|
| **Native Chandrayaan-2 Flight Data** | **Pending** | No uncalibrated Level-1/Level-2 Chandrayaan-2 PDS4 rasters were provided or bundled in the repository. |
| **Physical Sun-Angle Invariance** | **Proxy Validated Only** | Validated on mathematical illumination gradients and shadow proxies. True lunar surface scattering (Hapke photometric function, extreme grazing angles $> 80^\circ$) requires calibrated PDS-4 datasets with SPICE ephemerides. |
| **Native OHRC / TMC-2 / IIRS Cross-Registration** | **Architecture Ready, Uncalibrated** | Resolution differences ($0.25$ m OHRC vs. $5.0$ m TMC-2 vs. $80$ m IIRS) span a $320\times$ scale ratio. While GSD resampler architecture exists, flight cross-matching across these sensors requires orbital ground truth. |
| **Absolute Lunar Geolocation** | **Relative Only** | SELENE-REG registers in **relative image space** and relative mosaic coordinates. Absolute latitude/longitude coordinates require SPICE C-kernels, SPK orbit kernels, and high-resolution DEMs (GLD100/SLDEM2015), which are outside the scope of this self-contained package. |
| **ISRO / Flight Validation** | **None** | No certification, formal audit, or flight endorsement by ISRO, SAC, or mission operations is claimed or implied. |

---

## 3. Recommended Evaluator Testing Guidance

Evaluators can verify the reproducibility of all claims by executing:

1. **Deterministic Test Suite**:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
2. **Synthetic Demonstration Data**:
   ```bash
   python scripts/generate_demo_data.py
   python artifacts/api-server/python/worker.py --mode pair --source demo_data/synthetic_source.png --reference demo_data/synthetic_reference.png --out-dir demo_output --settings-json '{"detector":"sift"}'
   ```
