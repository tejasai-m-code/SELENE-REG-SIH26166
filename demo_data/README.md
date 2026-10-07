# Synthetic Demonstration Data

> **IMPORTANT SCIENTIFIC PROVENANCE NOTICE**
> 
> The images in this directory are **strictly synthetic demonstration proxies**.
> They were procedurally generated using deterministic random seeds (NumPy & OpenCV) to simulate lunar crater morphology, surface roughness, and directional illumination.
> 
> **THESE IMAGES ARE NOT NATIVE CHANDRAYAAN-2 FLIGHT PRODUCTS.**
> 
> No physical lunar coordinates, SPICE telemetry, DEM elevations, or ISRO flight validation are associated with these demo assets.

## Included Files

| Filename | Dimensions | Purpose | Simulated Conditions |
|---|---|---|---|
| `synthetic_source.png` | 400 × 400 | Moving raster for pairwise demo | Synthetic cratered terrain (seed 26166) |
| `synthetic_reference.png` | 400 × 400 | Fixed raster for pairwise demo | Warped (+4.0 deg, dx=+14px, dy=-10px) with +15% linear illumination gradient |
| `synthetic_mosaic_1.png` | 400 × 350 | Multi-image strip frame 1 | Left segment of continuous synthetic surface |
| `synthetic_mosaic_2.png` | 400 × 350 | Multi-image strip frame 2 | Center segment (~62% overlap with frame 1) |
| `synthetic_mosaic_3.png` | 400 × 350 | Multi-image strip frame 3 | Right segment (~62% overlap with frame 2) |

## Reproducibility

To regenerate these exact files from clean code:

```bash
python scripts/generate_demo_data.py
```
