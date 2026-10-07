import requests
import json
from pathlib import Path

def run_e2e_verification():
    url = 'http://localhost:5000/api/register'
    source_path = Path('tests/fixtures/moon100.png')
    ref_path = Path('tests/fixtures/moon99.png')
    assert source_path.exists(), f"Missing {source_path}"
    assert ref_path.exists(), f"Missing {ref_path}"

    with open(source_path, 'rb') as f_src, open(ref_path, 'rb') as f_ref:
        files = {
            'source': ('moon100.png', f_src, 'image/png'),
            'reference': ('moon99.png', f_ref, 'image/png'),
        }
        data = {
            'detector': 'sift',
            'representation': 'auto',
            'geometric_model': 'homography',
            'illumination_normalization': 'true',
            'spatial_distribution': 'true',
            'ratio': '0.75',
            'ransac_threshold': '3.0',
            'max_features': '6000',
            'execution_mode': 'hybrid',
        }
        res = requests.post(url, files=files, data=data, timeout=60)

    print('HTTP Status:', res.status_code)
    assert res.ok, f"Request failed: {res.text}"

    payload = res.json()
    metrics = payload.get('metrics', {})
    corrs = payload.get('correspondences', [])
    inliers = [c for c in corrs if c.get('is_inlier') is True and c.get('status') == 'INLIER']
    outliers = [c for c in corrs if c.get('is_inlier') is False or c.get('status') == 'OUTLIER']

    print(f"Total Matches: {len(corrs)}")
    print(f"Record Inliers: {len(inliers)}")
    print(f"Record Outliers: {len(outliers)}")
    print(f"Metrics inlier_count: {metrics.get('inlier_count')}")
    print(f"Metrics inliers: {metrics.get('inliers')}")
    print(f"Metrics n_inliers: {metrics.get('n_inliers')}")
    print(f"Metrics inlier_ratio: {metrics.get('inlier_ratio')}")
    print(f"Metrics rmse_pixels: {metrics.get('rmse_pixels')}")
    gate_info = payload.get('inlier_investigation', {}).get('quality_gate', {})
    print(f"Quality Gate Status: {gate_info.get('status')}")
    print(f"Case Classification: {gate_info.get('case_classification')}")

    # Assert exact internal consistency
    assert len(inliers) == metrics.get('inlier_count'), f"Discrepancy: {len(inliers)} != {metrics.get('inlier_count')}"
    assert len(inliers) == metrics.get('inliers')
    assert len(inliers) == metrics.get('n_inliers')
    assert len(outliers) == metrics.get('outliers')
    assert len(corrs) == metrics.get('matches')
    assert len(corrs) == metrics.get('match_count')
    assert round(len(inliers) / len(corrs), 4) == round(metrics.get('inlier_ratio'), 4)

    # Simulate scientific package export
    verified_points = [
        {
            "index": c["index"],
            "source_xy": c["source"],
            "reference_xy": c["reference"],
            "residual_pixels": c.get("residual") or 0.0,
            "is_inlier": c["is_inlier"],
            "status": c["status"],
        }
        for c in corrs
    ]
    verified_inliers_count = len([p for p in verified_points if p["is_inlier"]])
    candidate_matches_count = len(verified_points)
    inlier_ratio = round(verified_inliers_count / candidate_matches_count, 4)

    pkg = {
        "schema_version": "1.0.0-scientific-lunar",
        "provenance": {"system": "SELENE-REG-X", "job_id": payload.get("job_id")},
        "correspondences": {
            "detector": "SIFT",
            "candidate_matches_count": candidate_matches_count,
            "verified_inliers_count": verified_inliers_count,
            "inlier_ratio": inlier_ratio,
            "spatial_coverage_ratio": metrics.get("source_spatial_coverage"),
            "verified_points": verified_points,
        },
        "geometric_registration": {
            "model": "homography",
            "rmse_pixels": metrics.get("rmse_pixels"),
        },
        "scientific_validation": {
            "quality_status": gate_info.get("status"),
            "case_classification": gate_info.get("case_classification"),
        }
    }

    pkg_path = Path("scratch/e2e_verified_scientific_package.json")
    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    pkg_path.write_text(json.dumps(pkg, indent=2))

    # Read back and independently verify
    pkg_read = json.loads(pkg_path.read_text())
    read_inliers = sum(1 for p in pkg_read["correspondences"]["verified_points"] if p["is_inlier"] is True and p["status"] == "INLIER")
    read_outliers = sum(1 for p in pkg_read["correspondences"]["verified_points"] if p["is_inlier"] is False or p["status"] == "OUTLIER")

    assert pkg_read["correspondences"]["verified_inliers_count"] == read_inliers
    assert pkg_read["correspondences"]["verified_inliers_count"] == metrics.get("inlier_count")
    assert pkg_read["correspondences"]["candidate_matches_count"] == len(corrs)
    assert read_inliers + read_outliers == len(corrs)

    print("\n>>> SCIENTIFIC PACKAGE VERIFIED: 100% INTERNAL CONSISTENCY CONFIRMED! <<<")

if __name__ == '__main__':
    run_e2e_verification()
