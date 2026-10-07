"""Generate deterministic synthetic demo datasets for reproducible SIH evaluation.

All images generated here are procedurally simulated lunar crater surfaces.
They are NOT native Chandrayaan-2 flight data.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def make_synthetic_surface(seed: int = 26166, width: int = 400, height: int = 400) -> np.ndarray:
    """Generate a reproducible, textured synthetic lunar surface with simulated craters."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 170, (height, width), dtype=np.uint8)
    # Add primary impact craters
    for _ in range(20):
        cx = int(rng.integers(25, width - 25))
        cy = int(rng.integers(25, height - 25))
        radius = int(rng.integers(10, 48))
        intensity = int(rng.integers(35, 230))
        cv2.circle(base, (cx, cy), radius, intensity, -1)
        # Ring rim
        cv2.circle(base, (cx, cy), radius, int(rng.integers(20, 55)), max(1, radius // 5))
        # Inner shadow crescent simulating directional sunlight
        cv2.ellipse(base, (cx - radius // 4, cy), (radius // 2, radius // 3), 45, 0, 180, int(rng.integers(10, 40)), -1)
    # Add subtle high-frequency texture
    noise = rng.normal(0, 5.0, base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    demo_dir = Path("demo_data")
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("Generating deterministic synthetic demo images in demo_data/ ...")

    # 1. Pairwise Demonstration Images (400x400)
    # Source image: moving raster
    source = make_synthetic_surface(seed=26166, width=400, height=400)
    source_path = demo_dir / "synthetic_source.png"
    cv2.imwrite(str(source_path), source)

    # Reference image: transformed with Euclidean shift + rotation + illumination perturbation
    theta = np.radians(4.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = 14.0, -10.0
    M = np.float32([
        [cos_t, -sin_t, dx + (1 - cos_t) * 200 + sin_t * 200],
        [sin_t,  cos_t, dy - sin_t * 200 + (1 - cos_t) * 200]
    ])
    reference = cv2.warpAffine(source, M, (400, 400), borderMode=cv2.BORDER_REFLECT)
    # Apply controlled illumination gradient
    y_coords, x_coords = np.mgrid[0:400, 0:400].astype(np.float32)
    illum_gradient = 1.0 + 0.15 * (x_coords / 400.0 - 0.5)
    reference = np.clip(reference.astype(np.float32) * illum_gradient, 0, 255).astype(np.uint8)
    ref_path = demo_dir / "synthetic_reference.png"
    cv2.imwrite(str(ref_path), reference)

    # 2. Multi-Image Strip Demonstration Images (3 overlapping frames)
    # Create a wider continuous ground terrain (900x350)
    panorama = make_synthetic_surface(seed=98765, width=900, height=350)
    # Frame 1: x in [0, 400]
    frame1 = panorama[:, 0:400]
    # Frame 2: x in [250, 650] (62.5% overlap with frame 1)
    frame2 = panorama[:, 250:650]
    # Frame 3: x in [500, 900] (62.5% overlap with frame 2)
    frame3 = panorama[:, 500:900]

    cv2.imwrite(str(demo_dir / "synthetic_mosaic_1.png"), frame1)
    cv2.imwrite(str(demo_dir / "synthetic_mosaic_2.png"), frame2)
    cv2.imwrite(str(demo_dir / "synthetic_mosaic_3.png"), frame3)

    # 3. Create README explaining provenance and validation boundaries
    readme_content = """# Synthetic Demonstration Data

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
"""
    (demo_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"Generated 5 demo images and README.md in {demo_dir.resolve()}")


if __name__ == "__main__":
    main()
