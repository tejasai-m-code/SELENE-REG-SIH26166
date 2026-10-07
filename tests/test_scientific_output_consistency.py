import json
import pytest
import numpy as np

from app.services.pairwise_registration import serialize_correspondences, PairwiseRegistrationResult
from app.services.metrics import compute_metrics
from app.services.feature_matching import get_detector_capabilities


def test_inlier_count_and_ratio_consistency():
    """
    Test 1 & Test 2: Inlier count and ratio consistency
    Construct a registration result containing known inliers and outliers.
    Verify:
      summary count == serialized inlier count
      inlier_ratio == verified_inliers_count / total_matches
    """
    # 50 matches total: 43 inliers, 7 outliers
    n_total = 50
    n_inliers = 43
    source_pts = np.random.uniform(50, 400, (n_total, 2))
    ref_pts = source_pts.copy()
    
    # 43 inliers with near-zero error, 7 outliers with large error
    inlier_mask = np.zeros(n_total, dtype=bool)
    inlier_mask[:n_inliers] = True
    ref_pts[n_inliers:] += np.random.uniform(30, 80, (n_total - n_inliers, 2))

    H = np.eye(3)
    metrics = compute_metrics(
        source_pts,
        ref_pts,
        H,
        inlier_mask,
        source_shape=(512, 512),
        reference_shape=(512, 512),
    )

    # In metrics, inlier_count and inliers must agree
    assert metrics["inlier_count"] == n_inliers
    assert np.isclose(metrics["inlier_ratio"], n_inliers / n_total)

    result = PairwiseRegistrationResult(
        homography=H,
        inlier_mask=inlier_mask,
        metrics=metrics,
        registered=None,
        match_visualization=None,
        source_points=source_pts,
        reference_points=ref_pts,
        final_reference_points=ref_pts,
        raw_match_count=n_total,
        post_distribution_match_count=n_total,
        ecc_used=False,
        ecc_correlation=None,
        detector="SIFT",
        registration_status="PASS",
        success=True,
        raw_rmse=0.0,
        refined_rmse=0.0,
        rmse_improvement=None,
        processing_time_seconds=0.1,
    )

    serialized_corrs = serialize_correspondences(result)
    assert len(serialized_corrs) == n_total

    # Count records explicitly classified as inlier
    inlier_records = [c for c in serialized_corrs if c["is_inlier"] is True and c["status"] == "INLIER"]
    outlier_records = [c for c in serialized_corrs if c["is_inlier"] is False and c["status"] == "OUTLIER"]

    assert len(inlier_records) == n_inliers
    assert len(outlier_records) == (n_total - n_inliers)
    assert len(inlier_records) == metrics["inlier_count"]


def test_scientific_package_consistency_and_roundtrip(tmp_path):
    """
    Test 3 & Test 4: JSON Round-trip and Frontend/API consistency
    Simulate generating scientific package, round-tripping through JSON,
    and verifying zero discrepancy between correspondence records and summary header.
    """
    n_total = 60
    n_inliers = 52
    source_pts = np.random.uniform(10, 500, (n_total, 2))
    ref_pts = source_pts.copy()
    inlier_mask = np.zeros(n_total, dtype=bool)
    inlier_mask[:n_inliers] = True
    ref_pts[n_inliers:] += np.random.uniform(25, 75, (n_total - n_inliers, 2))

    H = np.eye(3)
    metrics = compute_metrics(
        source_pts,
        ref_pts,
        H,
        inlier_mask,
        source_shape=(512, 512),
        reference_shape=(512, 512),
    )
    metrics["inliers"] = n_inliers
    metrics["inlier_count"] = n_inliers
    metrics["match_count"] = n_total
    metrics["matches"] = n_total

    result = PairwiseRegistrationResult(
        homography=H,
        inlier_mask=inlier_mask,
        metrics=metrics,
        registered=None,
        match_visualization=None,
        source_points=source_pts,
        reference_points=ref_pts,
        final_reference_points=ref_pts,
        raw_match_count=n_total,
        post_distribution_match_count=n_total,
        ecc_used=False,
        ecc_correlation=None,
        detector="SIFT",
        registration_status="PASS",
        success=True,
        raw_rmse=0.0,
        refined_rmse=0.0,
        rmse_improvement=None,
        processing_time_seconds=0.1,
    )

    serialized_corrs = serialize_correspondences(result)

    # Simulate scientific package generation logic
    verified_points = [
        {
            "index": c["index"],
            "source_xy": c["source"],
            "reference_xy": c["reference"],
            "residual_pixels": c.get("residual") or 0.0,
            "is_inlier": c["is_inlier"],
            "status": c["status"],
        }
        for c in serialized_corrs
    ]

    verified_inliers_count = len([p for p in verified_points if p["is_inlier"]])
    candidate_matches_count = len(verified_points)
    inlier_ratio = round(verified_inliers_count / candidate_matches_count, 4)

    package = {
        "schema_version": "1.0.0-scientific-lunar",
        "provenance": {"system": "SELENE-REG-X", "job_id": "job-test-roundtrip"},
        "correspondences": {
            "detector": "SIFT",
            "candidate_matches_count": candidate_matches_count,
            "verified_inliers_count": verified_inliers_count,
            "inlier_ratio": inlier_ratio,
            "verified_points": verified_points,
        },
        "geometric_registration": {
            "model": "homography",
            "rmse_pixels": metrics["rmse_pixels"],
        },
    }

    # Write and re-read from disk
    pkg_file = tmp_path / "scientific_package.json"
    pkg_file.write_text(json.dumps(package, indent=2))

    read_back = json.loads(pkg_file.read_text())
    read_corr_summary = read_back["correspondences"]
    read_points = read_corr_summary["verified_points"]

    # Re-calculate directly from points
    recalculated_inliers = sum(1 for p in read_points if p["is_inlier"] is True and p["status"] == "INLIER")
    recalculated_outliers = sum(1 for p in read_points if p["is_inlier"] is False or p["status"] == "OUTLIER")

    assert read_corr_summary["verified_inliers_count"] == recalculated_inliers
    assert read_corr_summary["verified_inliers_count"] == n_inliers
    assert read_corr_summary["candidate_matches_count"] == len(read_points)
    assert recalculated_inliers + recalculated_outliers == len(read_points)
    assert np.isclose(read_corr_summary["inlier_ratio"], recalculated_inliers / len(read_points), atol=1e-4)


def test_failed_registration_clean_zero_state():
    """
    Test 5: Failed registration clean zero-state
    Verify that a failed/zero-inlier registration does not inherit stale metrics
    or report conflicting numbers.
    """
    n_total = 30
    source_pts = np.random.uniform(10, 500, (n_total, 2))
    ref_pts = np.random.uniform(10, 500, (n_total, 2))  # completely random, zero inliers
    inlier_mask = np.zeros(n_total, dtype=bool)

    metrics = compute_metrics(
        source_pts,
        ref_pts,
        np.eye(3),
        inlier_mask,
        source_shape=(512, 512),
        reference_shape=(512, 512),
    )

    assert metrics["inlier_count"] == 0
    assert metrics["inlier_ratio"] == 0.0

    result = PairwiseRegistrationResult(
        homography=np.eye(3),
        inlier_mask=inlier_mask,
        metrics=metrics,
        registered=None,
        match_visualization=None,
        source_points=source_pts,
        reference_points=ref_pts,
        final_reference_points=ref_pts,
        raw_match_count=n_total,
        post_distribution_match_count=n_total,
        ecc_used=False,
        ecc_correlation=None,
        detector="SIFT",
        registration_status="FAIL",
        success=False,
        raw_rmse=None,
        refined_rmse=None,
        rmse_improvement=None,
        processing_time_seconds=0.05,
    )

    serialized = serialize_correspondences(result)
    assert len(serialized) == n_total
    inliers = [c for c in serialized if c["is_inlier"]]
    assert len(inliers) == 0
    assert metrics["inlier_count"] == 0


def test_learned_matching_capabilities_matrix():
    """
    Verify learned matching capabilities are reported accurately and honestly.
    No fake inference or simulated weights allowed.
    """
    caps = get_detector_capabilities()
    
    # Classical matchers
    assert caps["sift"]["status"] == "AVAILABLE"
    assert caps["orb"]["status"] == "AVAILABLE"
    # AKAZE status strictly reflects whether cv2 has AKAZE_create
    assert caps["akaze"]["status"] in ("AVAILABLE", "NOT AVAILABLE")

    # Learned matchers must honestly report NOT AVAILABLE with required reason
    assert caps["superpoint"]["status"] == "NOT AVAILABLE"
    assert caps["superpoint"]["substatus"] == "MODEL REQUIRED"
    assert "checkpoint" in caps["superpoint"]["reason"].lower() or ".pt" in caps["superpoint"]["reason"].lower()

    if caps["loftr"]["status"] == "NOT AVAILABLE":
        assert caps["loftr"]["substatus"] in ("MODEL REQUIRED", "DEPENDENCY MISSING")
    else:
        assert caps["loftr"]["status"] == "AVAILABLE"
        assert caps["loftr"]["substatus"] == "AVAILABLE"

    assert caps["lightglue"]["status"] == "NOT AVAILABLE"
    assert caps["lightglue"]["substatus"] == "DEPENDENCY MISSING"
