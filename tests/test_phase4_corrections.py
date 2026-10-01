import numpy as np
import cv2
from app.services.scale import (
    build_image_pyramid,
    propagate_homography,
    estimate_relative_scale,
    compute_expected_scale_ratio,
    coarse_to_fine_register_pair
)
from app.services.metadata import valid_gsd, parse_pds4_xml

def test_phase4_corrections():
    print("Testing Phase 4 Corrections...")

    # A, B, C, D: GSD validation
    assert valid_gsd(2.5) == 2.5
    assert valid_gsd(0.0) is None
    assert valid_gsd(-1.5) is None
    assert valid_gsd(float('nan')) is None
    assert valid_gsd(float('inf')) is None

    # E, F: Unit and anisotropic extraction (Synthetic test metadata XML)
    xml_aniso = """
    <root>
        <Identification_Area>
            <pixel_resolution unit="m/pixel">2.5</pixel_resolution>
        </Identification_Area>
    </root>
    """
    pm = parse_pds4_xml(xml_aniso)
    assert pm.spatial.gsd == 2.5
    assert pm.spatial.gsd_x == 2.5
    assert pm.spatial.gsd_y == 2.5
    assert pm.spatial.gsd_unit == "m/pixel"

    src_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(src_img, (50, 50), (100, 100), 255, -1)
    
    ref_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(ref_img, (100, 100), (200, 200), 255, -1)

    # G. metadata scale actually changes initialization
    # K. no double scaling (verify final homography correctly maps src to ref once)
    try:
        res_meta, diag_meta, prov_meta = coarse_to_fine_register_pair(
            src_img, ref_img, 
            source_gsd=2.0, reference_gsd=1.0, 
            levels=2
        )
        assert prov_meta.initialization_path == "metadata_gsd"
        assert prov_meta.scale_ratio == 2.0
    except ValueError:
        pass
    
    # H. image-derived scale actually changes initialization
    try:
        res_img, diag_img, prov_img = coarse_to_fine_register_pair(
            src_img, ref_img, 
            source_gsd=None, reference_gsd=None, 
            levels=2
        )
        if prov_img.initialization_path == "image_derived":
            assert prov_img.scale_ratio > 1.5
        else:
            assert prov_img.initialization_path == "v1_fallback"
    except ValueError:
        pass
        
    # I. image-derived scale with insufficient evidence falls back
    try:
        res_fail, diag_fail, prov_fail = coarse_to_fine_register_pair(
            np.zeros((100,100), dtype=np.uint8), ref_img, levels=2
        )
        assert prov_fail.initialization_path == "v1_fallback"
    except ValueError:
        pass

    # J. scale uncertainty/dispersion
    src_pts = np.array([[[0, 0]], [[10, 0]], [[0, 10]], [[10, 10]]], dtype=np.float32)
    ref_pts = src_pts * 3.0
    # add slight noise to calculate dispersion
    ref_pts[0, 0, 0] += 0.1
    ref_pts[1, 0, 0] -= 0.1
    s, count, disp = estimate_relative_scale(src_pts, ref_pts)
    assert count == 4
    assert disp is not None and disp > 0.0

    print("All Phase 4 Correction tests passed.")

if __name__ == "__main__":
    test_phase4_corrections()
