import numpy as np
import cv2

from app.services.representation import build_representations
from app.services.cross_modal_matching import (
    execute_cross_modal_matching,
    fuse_representations,
    inspect_deep_matching_capability,
    serialize_match_payload,
    CandidateTransform
)
from app.services.pairwise_registration import PairwiseRegistrationResult
from app.services.radiometry import RadiometricConfig
from app.utils.image_utils import ScientificRaster, ProductMetadata

def test_phase5():
    print("Testing Phase 5 Cross-Modal Matching Corrections...\n")

    print("--- 6. CROSS-MODAL SYNTHETIC TEST ---")
    np.random.seed(42)
    src_img = (np.random.rand(400, 400) * 255).astype(np.uint8)
    src_img = cv2.GaussianBlur(src_img, (15, 15), 5)
    src_img = cv2.normalize(src_img, None, 0, 255, cv2.NORM_MINMAX)

    for i in range(5):
        for j in range(5):
            cv2.circle(src_img, (i*80+40, j*80+40), 15, (i+j)*20, -1)
            cv2.rectangle(src_img, (i*80+10, j*80+10), (i*80+30, j*80+30), 255 - (i+j)*20, -1)

    ref_img = 255 - src_img.copy()
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    ref_img = cv2.warpAffine(ref_img, M, (400, 400))
    
    raster_src = ScientificRaster(data=src_img.astype(np.float32), metadata=ProductMetadata())
    raster_ref = ScientificRaster(data=ref_img.astype(np.float32), metadata=ProductMetadata())
    
    src_reps = build_representations(raster_src, RadiometricConfig())
    ref_reps = build_representations(raster_ref, RadiometricConfig())
    
    candidates = execute_cross_modal_matching(src_reps, ref_reps, levels=1)
    
    grad_cand = next((c for c in candidates if c.representation_source == "gradient_magnitude"), None)
    
    if grad_cand and grad_cand.result and grad_cand.result.homography is not None:
        H = grad_cand.result.homography
        measured_tx = H[0, 2]
        measured_ty = H[1, 2]
        print(f"Known transform: tx=50, ty=50 (with intensity inversion)")
        print(f"Recovered via gradient representation: tx={measured_tx:.2f}, ty={measured_ty:.2f}")
    else:
        print("Gradient representation failed to recover transform")

    print("\n--- 7. REPRESENTATION FUSION (AGREEMENT vs DISAGREEMENT) ---")
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
    fused_agree = fuse_representations([cand1, cand2_agree])
    print(f"CASE A (Agreeing matrices): Confidence={fused_agree.confidence}, FailureReason={fused_agree.failure_reason}")
    
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
    fused_disagree = fuse_representations([cand1, cand2_disagree])
    print(f"CASE B (Disagreeing matrices): Confidence={fused_disagree.confidence}, FailureReason={fused_disagree.failure_reason}")

    print("\n--- 11. PHASE 4 REGRESSION ---")
    if grad_cand and grad_cand.scale_provenance:
        print(f"Scale Provenance Path: {grad_cand.scale_provenance.initialization_path}")
        print(f"Scale Ratio Found: {grad_cand.scale_provenance.scale_ratio}")
        
    print("\nAll targeted checks executed.")

if __name__ == "__main__":
    test_phase5()
