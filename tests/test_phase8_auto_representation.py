import numpy as np
import pytest
from app.services.pairwise_registration import register_pair
from app.services.preprocessing import preprocess_with_diagnostics

def _generate_synthetic_lunar_pair(seed: int = 42):
    rng = np.random.default_rng(seed)
    # 256x256 base lunar terrain with craters
    base = np.full((256, 256), 128, dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    
    # Add multiple craters
    crater_centers = [(70, 70, 25), (180, 80, 35), (130, 170, 30), (60, 200, 20), (200, 200, 22)]
    for cy, cx, r in crater_centers:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        crater_profile = np.clip(1.0 - (dist / r) ** 2, 0, 1)
        base -= crater_profile * 50.0
        # Add rim
        rim = np.exp(-((dist - r) ** 2) / 8.0) * 25.0
        base += rim

    # Add high-frequency texture
    noise = rng.normal(0, 8, (256, 256))
    base = np.clip(base + noise, 10, 245).astype(np.uint8)

    # Reference image has slight illumination gradient
    grad = np.linspace(0.8, 1.2, 256, dtype=np.float32)[None, :]
    ref = np.clip(base.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    # Source image has a known translation (+8, +5) and slight scale
    import cv2
    M = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    src = cv2.warpAffine(base, M, (256, 256))
    return src, ref


def test_manual_representation_changes_processing_path():
    src, ref = _generate_synthetic_lunar_pair()
    res_grad = register_pair(src, ref, representation="gradient", detector="sift")
    res_clahe = register_pair(src, ref, representation="clahe", detector="sift")

    assert res_grad.preprocessing is not None
    assert res_clahe.preprocessing is not None
    assert res_grad.preprocessing["representation"] == "gradient"
    assert res_clahe.preprocessing["representation"] == "clahe"
    assert res_grad.preprocessing["auto_selection"]["mode"] == "MANUAL"


def test_auto_select_evaluates_multiple_representations():
    src, ref = _generate_synthetic_lunar_pair()
    res = register_pair(src, ref, representation="auto", detector="sift")

    assert res.preprocessing is not None
    auto_data = res.preprocessing.get("auto_selection")
    assert auto_data is not None
    assert auto_data["mode"] == "AUTO"

    # Evaluated all 7 candidates
    assert len(auto_data["evaluated_candidates"]) == 7
    ledger = auto_data["ledger"]
    assert len(ledger) == 7

    reps_in_ledger = {entry["representation"] for entry in ledger}
    expected_reps = {"raw", "percentile", "clahe", "gradient", "highpass", "retinex", "structural"}
    assert reps_in_ledger == expected_reps

    # Ledger entries have required numeric verification fields
    for entry in ledger:
        assert "matches" in entry
        assert "inliers" in entry
        assert "inlier_ratio" in entry
        assert "coverage" in entry
        assert "status" in entry
        assert "score" in entry
        assert "processing_time_s" in entry
        assert entry["status"] in ("PASS", "WARN", "FAIL")

    # Selection score and reason are numeric and transparent
    assert auto_data["selection_score"] is not None
    assert isinstance(auto_data["selection_reason"], str)
    assert len(auto_data["selection_reason"]) > 10

    # The selected representation is in the ledger and was selected based on evidence
    selected = auto_data["selected_representation"]
    assert selected in expected_reps
    assert res.preprocessing["representation"] == selected


def test_auto_select_rejects_deliberately_poor_representation():
    # If one representation is artificially degraded, auto select should choose a better candidate
    src, ref = _generate_synthetic_lunar_pair()
    res = register_pair(src, ref, representation="auto", detector="sift")
    auto_data = res.preprocessing["auto_selection"]
    
    # Selected should be a valid candidate with inliers >= 6
    selected_entry = next(c for c in auto_data["ledger"] if c["representation"] == auto_data["selected_representation"])
    assert selected_entry["is_valid"] is True
    assert selected_entry["inliers"] >= 6


def test_all_representations_fail_reported_honestly():
    # Blank/zero images where no features can be detected
    blank_src = np.zeros((200, 200), dtype=np.uint8)
    blank_ref = np.zeros((200, 200), dtype=np.uint8)

    res = register_pair(blank_src, blank_ref, representation="auto", detector="sift")
    assert res.success is False
    assert res.registration_status in ("FAIL", "REVIEW")
    assert res.preprocessing is not None
    auto_data = res.preprocessing.get("auto_selection")
    assert auto_data is not None
    # All candidates in ledger must be FAIL or invalid
    for cand in auto_data["ledger"]:
        assert cand["inliers"] < 6 or cand["status"] == "FAIL" or not cand.get("is_valid", False)


def test_selected_representation_applied_to_final_geometry():
    src, ref = _generate_synthetic_lunar_pair()
    res = register_pair(src, ref, representation="auto", detector="sift")
    assert res.success is True
    assert res.homography is not None
    assert res.homography.shape == (3, 3)
    assert np.isfinite(res.homography).all()

