import numpy as np
import cv2

from app.services.cross_modal_matching import (
    fuse_representations,
    serialize_match_payload,
    CandidateTransform
)
from app.services.pairwise_registration import PairwiseRegistrationResult

def test_spatial_cell():
    print("--- 4. SPATIAL CELL TEST ---")
    w, h = 800, 600
    grid = (4, 5) # 4 rows, 5 cols
    
    # Points: top-left, top-right, bottom-left, bottom-right, center
    # Top-Left: (0, 0)
    # Top-Right: (799, 0)
    # Bottom-Left: (0, 599)
    # Bottom-Right: (799, 599)
    # Center: (400, 300)
    
    pts = np.array([
        [[0, 0]],
        [[799, 0]],
        [[0, 599]],
        [[799, 599]],
        [[400, 300]]
    ], dtype=np.float32)
    
    cand = CandidateTransform(
        representation_source="rep1", representation_reference="rep2",
        detector="sift", descriptor="sift", matching_method="ecc",
        candidate_count=5, good_matches=5, inlier_count=5, inlier_ratio=1.0,
        reprojection_error=0.0, confidence="MODERATE", failure_reason=None,
        result=PairwiseRegistrationResult(
            homography=np.eye(3, dtype=np.float32), 
            inlier_mask=np.array([True]*5), metrics={}, registered=None,
            match_visualization=np.zeros((1,1)), 
            source_points=pts, reference_points=pts,
            raw_match_count=5, post_distribution_match_count=5, ecc_used=False, ecc_correlation=None, detector="SIFT",
            raw_matches=None, ratios=None
        )
    )
    
    payload = serialize_match_payload(cand, source_shape=(h, w), spatial_grid=grid)
    
    assert payload[0]['spatial_cell'] == {'row': 0, 'col': 0, 'configuration': '4x5'}
    assert payload[1]['spatial_cell'] == {'row': 0, 'col': 4, 'configuration': '4x5'}
    assert payload[2]['spatial_cell'] == {'row': 3, 'col': 0, 'configuration': '4x5'}
    assert payload[3]['spatial_cell'] == {'row': 3, 'col': 4, 'configuration': '4x5'}
    
    # 400 is exactly in the middle of 800. Cells: 0-159 (col 0), 160-319 (col 1), 320-479 (col 2). So 400 is col 2.
    # 300 is exactly in the middle of 600. Cells: 0-149 (row 0), 150-299 (row 1), 300-449 (row 2). So 300 is row 2.
    assert payload[4]['spatial_cell'] == {'row': 2, 'col': 2, 'configuration': '4x5'}
    print("Spatial Cell Test PASS")


def test_fusion_geometry():
    print("--- 3. FUSION TEST ---")
    w, h = 640, 480
    
    cand1 = CandidateTransform(
        representation_source="rep1", representation_reference="rep2",
        detector="sift", descriptor="sift", matching_method="ecc",
        candidate_count=100, good_matches=50, inlier_count=50, inlier_ratio=1.0,
        reprojection_error=1.0, confidence="MODERATE", failure_reason=None,
        result=PairwiseRegistrationResult(
            homography=np.float32([[1, 0, 50], [0, 1, 50], [0, 0, 1]]), inlier_mask=np.array([]), metrics={}, registered=None,
            match_visualization=np.zeros((1,1)), source_points=np.array([]), reference_points=np.array([]),
            raw_match_count=0, post_distribution_match_count=0, ecc_used=False, ecc_correlation=None, detector="SIFT"
        )
    )
    
    # Case A: Minor shift, same geometry
    cand2_agree = CandidateTransform(
        representation_source="rep3", representation_reference="rep4",
        detector="sift", descriptor="sift", matching_method="ecc",
        candidate_count=100, good_matches=48, inlier_count=48, inlier_ratio=1.0,
        reprojection_error=1.0, confidence="MODERATE", failure_reason=None,
        result=PairwiseRegistrationResult(
            homography=np.float32([[1, 0, 50.1], [0, 1, 49.9], [0, 0, 1]]), 
            inlier_mask=np.array([]), metrics={}, registered=None, match_visualization=np.zeros((1,1)), 
            source_points=np.array([]), reference_points=np.array([]), raw_match_count=0, 
            post_distribution_match_count=0, ecc_used=False, ecc_correlation=None, detector="SIFT"
        )
    )
    
    fused_agree = fuse_representations([cand1, cand2_agree], source_shape=(h, w))
    assert fused_agree.confidence == "HIGH (Multi-Modal Agreement)"
    print("Fusion Geometry Case A PASS")

    # Case B: Large drift
    cand2_disagree = CandidateTransform(
        representation_source="rep3", representation_reference="rep4",
        detector="sift", descriptor="sift", matching_method="ecc",
        candidate_count=100, good_matches=48, inlier_count=48, inlier_ratio=1.0,
        reprojection_error=1.0, confidence="MODERATE", failure_reason=None,
        result=PairwiseRegistrationResult(
            homography=np.float32([[1, 0, 150], [0, 1, -50], [0, 0, 1]]), 
            inlier_mask=np.array([]), metrics={}, registered=None, match_visualization=np.zeros((1,1)), 
            source_points=np.array([]), reference_points=np.array([]), raw_match_count=0, 
            post_distribution_match_count=0, ecc_used=False, ecc_correlation=None, detector="SIFT"
        )
    )
    fused_disagree = fuse_representations([cand1, cand2_disagree], source_shape=(h, w))
    assert fused_disagree.failure_reason == "Representations strongly disagree geometrically (Control point drift)"
    print("Fusion Geometry Case B PASS")

if __name__ == "__main__":
    test_spatial_cell()
    test_fusion_geometry()
