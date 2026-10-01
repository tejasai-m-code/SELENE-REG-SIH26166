import numpy as np
import cv2

from app.services.subpixel import (
    taylor_refinement, lk_refinement, ecc_refinement, phase_correlation_refinement,
    local_correlation_peak, compute_subpixel_consensus, run_subpixel_refinement
)

def test_subpixel_ground_truth():
    print("Testing Phase 7 Subpixel Refinement...")
    
    # Create an image with sufficient texture and features for all methods
    np.random.seed(42)
    src = (np.random.rand(200, 200) * 255).astype(np.uint8)
    src = cv2.GaussianBlur(src, (3, 3), 1.0)
    
    # Add strong geometric edges
    for i in range(5):
        for j in range(5):
            cv2.rectangle(src, (i*30+10, j*30+10), (i*30+20, j*30+20), 255, -1)
            cv2.circle(src, (i*30+25, j*30+25), 5, 0, -1)
            
    # Ground truth subpixel shift
    gt_dx = 0.35
    gt_dy = -0.45
    
    ref = np.zeros_like(src)
    M = np.float32([[1, 0, gt_dx], [0, 1, gt_dy]])
    
    # cv2.INTER_LINEAR allows fractional shifts to be approximated
    ref = cv2.warpAffine(src, M, (200, 200), flags=cv2.INTER_LINEAR)
    
    print(f"\nGround Truth Shift: dx={gt_dx}, dy={gt_dy}")
    
    print("\n--- Individual Method Results ---")
    res_taylor = taylor_refinement(src, ref)
    print(f"Taylor: dx={res_taylor.dx:.4f}, dy={res_taylor.dy:.4f} | Status: {res_taylor.status}")
    
    res_lk = lk_refinement(src, ref)
    print(f"Lucas-Kanade: dx={res_lk.dx:.4f}, dy={res_lk.dy:.4f} | Status: {res_lk.status}")
    
    res_ecc = ecc_refinement(src, ref)
    print(f"ECC: dx={res_ecc.dx:.4f}, dy={res_ecc.dy:.4f} | Status: {res_ecc.status}")
    
    res_phase = phase_correlation_refinement(src, ref)
    print(f"Phase Correlation: dx={res_phase.dx:.4f}, dy={res_phase.dy:.4f} | Status: {res_phase.status}")
    
    res_peak = local_correlation_peak(src, ref)
    print(f"Local Peak: dx={res_peak.dx:.4f}, dy={res_peak.dy:.4f} | Status: {res_peak.status}")
    
    print("\n--- Subpixel Consensus ---")
    consensus = run_subpixel_refinement(src, ref)
    print(f"Consensus dx={consensus.dx:.4f}, dy={consensus.dy:.4f} | Confidence: {consensus.confidence}")
    print(f"Active Methods: {consensus.active_methods}")
    
    # Check that consensus is within 0.1 of ground truth
    assert abs(consensus.dx - gt_dx) < 0.1, f"Consensus dx {consensus.dx} too far from GT {gt_dx}"
    assert abs(consensus.dy - gt_dy) < 0.1, f"Consensus dy {consensus.dy} too far from GT {gt_dy}"
    print("Consensus meets subpixel ground-truth accuracy requirements.")
    
if __name__ == "__main__":
    test_subpixel_ground_truth()
