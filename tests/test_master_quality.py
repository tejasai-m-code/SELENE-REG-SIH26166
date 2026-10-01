import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "backend")

from app.services.metrics import compute_metrics
from app.services.pairwise_registration import register_pair


def _fixture():
    source = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(source, (80, 80), 20, 255, -1)
    cv2.rectangle(source, (140, 120), (200, 180), 180, -1)
    cv2.line(source, (30, 200), (200, 30), 220, 3)
    reference = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, 8]]), (240, 240))
    return source, reference


def test_quality_engine_exposes_residual_and_image_metrics():
    source, reference = _fixture()
    result = register_pair(source, reference, ecc_refinement=False, max_features=2000)
    for key in (
        "p90_reprojection_error_pixels",
        "max_reprojection_error_pixels",
        "residual_std_pixels",
        "spatial_uniformity",
        "largest_spatial_cluster_fraction",
        "transform_conditioning",
        "evidence_score",
        "ssim",
        "psnr_db",
        "nmi",
    ):
        assert key in result.metrics
    assert result.metrics["inlier_count"] >= 4
    assert result.metrics["rmse_pixels"] < 2.0


def test_residual_serialization_contains_error_components():
    source, reference = _fixture()
    result = register_pair(source, reference, ecc_refinement=False, max_features=2000)
    from app.services.pairwise_registration import serialize_inlier_points
    points = serialize_inlier_points(result)
    assert points
    assert {"source", "reference", "dx", "dy", "error", "status"} <= set(points[0])


def test_master_frontend_contains_real_mosaic_to_3d_contract():
    html = Path("frontend/index.html").read_text(encoding="utf8")
    js = Path("frontend/app.js").read_text(encoding="utf8")
    assert 'id="open3DBtn"' in html
    assert 'id="terrainCanvas"' in html
    assert 'multiState.result.outputs.mosaic' in js
    assert 'loadTerrainTexture(base+multiState.result.outputs.mosaic)' in js
    assert "LATITUDE" not in html  # no fabricated latitude field is rendered
