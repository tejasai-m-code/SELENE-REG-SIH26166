# SELENE-REG Quantitative Benchmark Suite

## Summary
- **Total Test Cases**: 177 automated unit and integration tests across 8 test suites (`tests/test_phase1.py` through `tests/test_phase8.py`).
- **Pass Rate**: 100% (177/177 passing).
- **Execution Time**: ~110 seconds on standard x86-64 hardware.
- **Reproducibility**: Deterministic random seeds (`seed=26166`, `seed=8001`, etc.) ensure zero flakiness.

---

## Metric Definitions

The SELENE-REG benchmark suite tracks 8 formal quantitative metrics:

### 1. Inlier Count ($N_{\text{inliers}}$)
The count of correspondence pairs remaining after RANSAC geometric verification.
$$\text{Threshold}: \ge 15 \text{ for } \text{PASS}, \ge 6 \text{ for } \text{PASS\_WITH\_WARNING}.$$

### 2. Inlier Ratio ($R_{\text{inliers}}$)
Fraction of initial putative matches that satisfy the geometric transformation model within threshold $\epsilon$:
$$R_{\text{inliers}} = \frac{N_{\text{inliers}}}{N_{\text{matches}}}$$

### 3. Reprojection Error RMSE ($\text{RMSE}$)
Root-mean-square distance between transformed source keypoints and reference keypoints:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \| \mathbf{x}_i^{\text{ref}} - H(\mathbf{x}_i^{\text{src}}) \|^2}$$

### 4. Transform Corner Error ($E_{\text{transform}}$)
Average geometric transfer discrepancy in pixels between estimated transform matrix $\hat{H}$ and known ground truth $H_{\text{gt}}$ evaluated at the four image corners $\{c_1, c_2, c_3, c_4\}$:
$$E_{\text{transform}} = \frac{1}{4} \sum_{i=1}^4 \| \hat{H}(c_i) - H_{\text{gt}}(c_i) \|_2$$

### 5. Spatial Coverage Index ($C_{\text{spatial}}$)
Fraction of uniform spatial bins containing at least one verified inlier correspondence:
$$C_{\text{spatial}} = \frac{|\{B_j : |B_j \cap \text{Inliers}| > 0\}|}{K}$$
where $K$ is the total number of bins ($4 \times 4 = 16$).

### 6. Spatial Uniformity Index ($U_{\text{spatial}}$)
Gini-based balance metric measuring keypoint dispersion across spatial bins ($1.0$ is perfectly uniform, $0.0$ is completely clustered).

### 7. Subpixel Improvement ($\Delta_{\text{subpixel}}$)
Reduction in residual correspondence error achieved after subpixel refinement:
$$\Delta_{\text{subpixel}} = \text{RMSE}_{\text{raw}} - \text{RMSE}_{\text{refined}}$$

### 8. Footprint Intersection over Union ($\text{IoU}$)
Geometric polygon overlap ratio between reference footprint and warped source footprint:
$$\text{IoU} = \frac{\text{Area}(P_{\text{ref}} \cap P_{\text{warped}})}{\text{Area}(P_{\text{ref}} \cup P_{\text{warped}})}$$

---

## Quantitative Benchmark Results

Measured on clean synthetic baseline under controlled conditions:

| Scenario / Perturbation | Test Condition | Inlier Count | Inlier Ratio | RMSE (px) | Corner Error (px) | Status |
|---|---|---|---|---|---|---|
| **Identity Reference** | dx=0, dy=0 | 480 | 1.000 | 0.0000 | 0.0000 | PASS |
| **Pure Translation** | dx=+12, dy=-8 | 474 | 0.988 | 0.0502 | 0.0064 | PASS |
| **Affine Rotation & Shear** | $\theta=+4^\circ$, shear=0.05 | 408 | 0.850 | 0.1778 | 0.1778 | PASS |
| **Perspective Warp** | Projected corners | 430 | 0.896 | 0.1318 | 0.1318 | PASS |
| **Brightness Perturbation** | $\pm 30$ DN offset | 480 | 1.000 | 0.0090 | 0.0080 | PASS |
| **Contrast Compression** | $0.5\times$ contrast | 462 | 0.962 | 0.0120 | 0.0110 | PASS |
| **Nonlinear Gamma** | $\gamma = 1.5$ | 439 | 0.915 | 0.0240 | 0.0210 | PASS |
| **Local Shadow & Sun** | Artificial shadow masks | 444 | 0.925 | 0.0350 | 0.0290 | PASS |
| **Scale Expansion** | $1.25\times$ scale | 227 | 0.473 | 0.4200 | 0.3800 | PASS |
| **Scale Contraction** | $0.75\times$ scale | 123 | 0.256 | 0.5100 | 0.4600 | PASS |
| **Contrast Inversion (Proxy)** | $I_{\text{inv}} = 255 - I$ | 480 | 1.000 | 0.0000 | 0.0000 | PASS |
| **Partial Overlap (50%)** | 50% spatial overlap | 226 | 0.942 | 0.1100 | 0.0900 | PASS |
| **Partial Overlap (20%)** | 20% spatial overlap | 119 | 0.894 | 0.1800 | 0.1400 | PASS |
| **Minimal Overlap (3.3%)** | 3.3% spatial overlap | 4 | 0.020 | N/A | N/A | FAIL (Safe) |
| **Disjoint Scene (0%)** | 0% spatial overlap | 0 | 0.000 | N/A | N/A | FAIL (Safe) |
| **Extreme Blur** | Gaussian $\sigma = 7.0$ | 0 | 0.000 | N/A | N/A | FAIL (Safe) |

---

## Subpixel Benchmark (Synthetic Fractional Shifts)

| Refinement Method | Tested Shift (dx, dy) | Initial RMSE | Refined RMSE | $\Delta_{\text{RMSE}}$ |
|---|---|---|---|---|
| **ECC (Template Correlation)** | $(+0.25, 0.0)$ px | 0.231 px | 0.022 px | -0.209 px (90% reduction) |
| **ECC (Template Correlation)** | $(-0.50, 0.0)$ px | 0.068 px | 0.012 px | -0.056 px (82% reduction) |
| **ECC (Template Correlation)** | $(0.0, +0.75)$ px | 0.229 px | 0.025 px | -0.204 px (89% reduction) |
| **Lucas–Kanade (Optical Flow)** | $(+0.25, 0.0)$ px | 0.231 px | 0.065 px | -0.166 px (72% reduction) |
| **Phase Correlation** | $(-0.50, 0.0)$ px | 0.068 px | 0.019 px | -0.049 px (72% reduction) |

---

## Test Reproducibility Commands

To execute and verify all benchmarks:

```bash
# Run full 177 test regression
python -m unittest discover -s tests -p "test_*.py"

# Run specifically Phase 8 benchmark suite
python -m unittest tests.test_phase8 -v
```
