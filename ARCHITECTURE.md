# SELENE-REG Architecture

```text
Browser Dashboard
       |
       v
FastAPI /api/register
       |
       +--> Image ingestion
       |
       +--> Illumination normalization
       |
       +--> SIFT / ORB / AKAZE
       |
       +--> Ratio-test matching
       |
       +--> Spatial distribution constraint
       |
       +--> RANSAC homography
       |
       +--> ECC refinement
       |
       +--> Warp to reference
       |
       +--> RMSE / inliers / coverage
       |
       +--> Registered product + match visualization
```

## SIH requirement mapping

| Requirement | Prototype module |
|---|---|
| Correspondence points | SIFT/ORB/AKAZE + ratio test |
| Illumination variation | CLAHE + percentile normalization |
| Viewpoint variation | Homography + RANSAC |
| Scale variation | SIFT multi-scale detector |
| Uniform distribution | 4×4 spatial cell selection |
| Registered product | warpPerspective |
| RMSE | reprojection error |
| Inlier count | RANSAC mask |
| Inlier ratio | RANSAC inlier fraction |
| Sub-pixel direction | ECC refinement + future corner/learned refinement |

## Future SIH-grade research path

`PDS ingestion → sensor-aware normalization → coarse-to-fine scale search → RIFT/phase congruency → LoFTR/SuperGlue → spatial optimization → RANSAC/USAC → ECC/corner refinement → uncertainty-aware metrics`

## Multi-image map-builder extension

```text
                         SELENE-REG
                              |
                +-------------+-------------+
                |                           |
        Pair Registration            Multi-Image Builder
                |                           |
        Source + Reference         Image collection (2–12)
                |                           |
        Existing pair engine     Thumbnail candidate screening
                |                           |
        Pair metrics/output      Reusable pairwise registration
                                            |
                                  Accepted/rejected graph edges
                                            |
                                Maximum-confidence placement tree
                                            |
                            Non-tree consistency diagnostics
                                            |
                          Feathered relative image-space mosaic
```

The pairwise engine remains the source of truth for preprocessing, feature matching, RANSAC, ECC refinement, match visualizations, and metrics. Multi-image processing adds orchestration only; it does not claim global bundle adjustment, semantic overlap detection, or geographic registration. The output is therefore labeled **Relative Registered Lunar Mosaic** unless control/reference data is added in a future georeferencing layer.


## SELENE-REG X evidence layer

The established pair engine now exposes additional measured evidence without changing its fast multi-image cache path:

`correspondences → geometric residuals → P90/max/std → spatial entropy → conditioning → evidence score`

Pair API runs that generate a registered raster also compute lightweight downsampled SSIM, PSNR and histogram-based NMI. Multi-image registration continues to avoid unnecessary full-resolution raster metrics on cached pair evidence so incremental performance is preserved.

## Registered mosaic → 3D contract

`multi-registration result → generated mosaic PNG → WebGL2 texture → displaced relative terrain patch`

The 3D viewer is coupled to the actual job output. If no georeference/control metadata exists, its surface readout is explicitly relative X/Y/relief rather than fabricated lunar latitude/longitude.
