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
