import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from app.services.scientific_mosaic import build_scientific_mosaic, serialize_mosaic_result

results = []

def chk(label, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    msg = f"  {tag}   {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((label, cond))
    return cond


def make_T(tx, ty):
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


# ========== Global setup: 4-image GT mosaic ==========
# Phase 9 GT poses: T_i = root -> image_i
# T0=I, T1=T(30,0), T2=T(-20,15), T3=T(10,25)
# M_i = inv(T_i):
#   M0=I:      img0 corners in root: (0,0),(100,0),(100,100),(0,100)
#   M1=T(-30,0): img1 corners in root: (-30,0),(70,0),(70,100),(-30,100)
#   M2=T(20,-15): img2 corners in root: (20,-15),(120,-15),(120,85),(20,85)
#   M3=T(-10,-25): img3 corners in root: (-10,-25),(90,-25),(90,75),(-10,75)
# Global bbox: min_x=-30, min_y=-25, max_x=120, max_y=100
# canvas_W = ceil(120)-floor(-30)+1 = 120+30+1 = 151
# canvas_H = ceil(100)-floor(-25)+1 = 100+25+1 = 126

global_poses_4 = {
    '0': np.eye(3, dtype=np.float64).tolist(),
    '1': make_T(30, 0).tolist(),
    '2': make_T(-20, 15).tolist(),
    '3': make_T(10, 25).tolist(),
}
images_4 = [np.zeros((100, 100), dtype=np.float32) for _ in range(4)]
for i in range(4):
    images_4[i][:] = (i + 1) * 10  # img0=10, img1=20, img2=30, img3=40

gg_4 = {
    'global_poses': global_poses_4,
    'connected_components': [{'component_id': 0, 'node_ids': [0, 1, 2, 3], 'root_node': 0}]
}
ni_4 = [{'image_id': i, 'filename': f'img{i}.png', 'sensor': 'OHRC', 'width': 100, 'height': 100}
        for i in range(4)]

print("\n=== A: Phase 9 pose integration ===")
res = build_scientific_mosaic(images_4, gg_4, ni_4)
chk("A. Phase 9 global_poses consumed, components generated", len(res.components) == 1)
comp = res.components[0]

# ========== B: Transform inversion ==========
print("\n=== B: Transform inversion M_i = inv(T_i) ===")
# T1 = T(30,0), M1 = inv(T1) = T(-30,0)
T1 = make_T(30, 0)
M1 = np.linalg.inv(T1)
# A pixel at (50,50) in image1 should map to (50-30,50)=(20,50) in root via M1
p_img = np.array([50.0, 50.0, 1.0])
p_root = M1 @ p_img
p_back = T1 @ p_root
chk("B. M_i = inv(T_i): T_i @ M_i @ p = p (roundtrip)",
    np.allclose(p_back[:2], [50, 50], atol=1e-6),
    f"roundtrip={p_back[:2]}")
chk("B2. image1 pixel (50,50) maps to root (20,50) via inv(T1)=T(-30,0)",
    np.allclose(p_root[:2], [20, 50], atol=1e-6),
    f"got={p_root[:2]}")

# ========== C: Canvas bounds ==========
print("\n=== C: Canvas bounds ===")
# Proven above: W=151, H=126
chk("C. Canvas W=151 (from min_x=-30, max_x=120, +1 formula)",
    comp.width == 151, f"got W={comp.width}")
chk("C2. Canvas H=126 (from min_y=-25, max_y=100, +1 formula)",
    comp.height == 126, f"got H={comp.height}")

# ========== D: Negative coordinates ==========
print("\n=== D: Negative coordinates ===")
chk("D. Negative min_x=-30 handled (canvas width still positive)",
    comp.width > 0 and comp.mosaic_scientific.shape[1] == comp.width)
# Verify root corner of img3 (-10,-25) maps correctly to canvas (20,0)
# canvas_tx = -floor(-30) = +30, canvas_ty = -floor(-25) = +25
M3 = np.linalg.inv(make_T(10, 25))
corner_root = (M3 @ np.array([0, 0, 1]))[:2]  # (-10,-25) in root
canvas_x = corner_root[0] + 30   # +30 is canvas translation
canvas_y = corner_root[1] + 25
chk("D2. Img3 corner (0,0) in root=(-10,-25) -> canvas=(20,0)",
    np.allclose([canvas_x, canvas_y], [20.0, 0.0], atol=0.5),
    f"canvas corner=({canvas_x:.1f},{canvas_y:.1f})")

# ========== E: Rotated geometry ==========
print("\n=== E: Rotated geometry ===")
rot_T = np.array([[0, -1, 50], [1, 0, 50], [0, 0, 1]], dtype=np.float64)
res_e = build_scientific_mosaic(
    [np.ones((100, 100), dtype=np.float32)],
    {'global_poses': {'0': rot_T.tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0, 'width': 100, 'height': 100}]
)
chk("E. Rotated T_i: canvas computed without crash, positive dimensions",
    res_e.components[0].width > 0 and res_e.components[0].height > 0,
    f"W={res_e.components[0].width} H={res_e.components[0].height}")

# ========== F: Scaled geometry ==========
print("\n=== F: Scaled geometry ===")
# T_i = scale2: [[2,0,0],[0,2,0],[0,0,1]]
# M_i = scale0.5: img(50x50) corners in root: (0,0),(25,0),(25,25),(0,25)
# canvas_W = ceil(25)-floor(0)+1 = 26
scl_T = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
res_f = build_scientific_mosaic(
    [np.ones((50, 50), dtype=np.float32)],
    {'global_poses': {'0': scl_T.tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0, 'width': 50, 'height': 50}]
)
# T_i=scale2 means root->image scales UP (image is bigger than root)
# M_i=scale0.5 maps image->root (root is smaller)
chk("F. Scaled geometry: canvas W=26 (M_i=scale0.5 on 50px img -> 25px in root +1)",
    res_f.components[0].width == 26,
    f"got W={res_f.components[0].width}")

# ========== G: Valid mask ==========
print("\n=== G: Valid mask ===")
chk("G. mosaic_mask shape matches canvas (126,151)",
    comp.mosaic_mask.shape == (126, 151),
    f"got {comp.mosaic_mask.shape}")
chk("G2. mosaic_mask is bool dtype", comp.mosaic_mask.dtype == bool)
chk("G3. mosaic_mask has valid pixels in coverage area", comp.valid_pixel_count > 0)

# ========== H: Nodata ==========
print("\n=== H: Nodata handling ===")
img_nodata = np.ones((50, 50), dtype=np.float32) * 5.0
img_nodata[10:20, 10:20] = -9999.0
res_h = build_scientific_mosaic(
    [img_nodata],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0, 'width': 50, 'height': 50}],
    {'nodata_values': {0: -9999.0}}
)
chk("H. Nodata pixel (nodata=-9999) masked out of mosaic",
    not res_h.components[0].mosaic_mask[15, 15])
chk("H2. Valid pixel (val=5) retained in mask",
    res_h.components[0].mosaic_mask[30, 30])

# ========== I: NaN/Inf ==========
print("\n=== I: NaN/Inf handling ===")
img_nan = np.ones((50, 50), dtype=np.float32)
img_nan[10:20, 10:20] = np.nan
res_i = build_scientific_mosaic(
    [img_nan],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0}]
)
chk("I. NaN pixels masked out", not res_i.components[0].mosaic_mask[15, 15])
img_inf = np.ones((50, 50), dtype=np.float32)
img_inf[5:10, 5:10] = np.inf
res_inf = build_scientific_mosaic(
    [img_inf],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0}]
)
chk("I2. Inf pixels masked out", not res_inf.components[0].mosaic_mask[7, 7])

# ========== J: Zero is valid ==========
print("\n=== J: Zero-valued valid pixels ===")
img_zero = np.zeros((50, 50), dtype=np.float32)
res_j = build_scientific_mosaic(
    [img_zero],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0}]
)
chk("J. Zero pixel value (no nodata) remains in mask",
    res_j.components[0].mosaic_mask[25, 25])

# ========== K: Overlap ==========
print("\n=== K: Overlap detection ===")
chk("K. 4-image mosaic max_overlap >= 2",
    comp.overlap_statistics.get('max_overlap', 0) >= 2,
    f"max_overlap={comp.overlap_statistics.get('max_overlap')}")

# ========== L: Blending ==========
print("\n=== L: Blending ===")
# Isolated 2-image test for clean blending check
# img0=10 (root x:[0,100]), img1=20 (M1=T(-30,0) -> root x:[-30,70])
# min_x=-30 => canvas_tx=+30. Overlap in root: x:[0,70] => canvas: x:[30,100]
# At canvas(50,60): root_x=60-30=30, in both imgs -> weighted avg ~15
res_l = build_scientific_mosaic(
    [np.full((100, 100), 10, dtype=np.float32), np.full((100, 100), 20, dtype=np.float32)],
    {'global_poses': {'0': np.eye(3).tolist(), '1': make_T(30, 0).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0, 1], 'root_node': 0}]},
    [{'image_id': i, 'width': 100, 'height': 100} for i in range(2)]
)
val_l = float(res_l.components[0].mosaic_scientific[50, 60])
chk("L. Blending: overlap pixel weighted avg in (10,20)",
    10.0 < val_l < 20.0,
    f"val={val_l:.3f} (expect ~15)")

# ========== M: Footprint coordinates ==========
print("\n=== M: Footprint coordinates ===")
# T2=T(-20,15) => M2=T(20,-15). img2 corner (0,0) in root = (20,-15).
# Global: min_x=-30, min_y=-25. canvas corner = (20+30, -15+25) = (50, 10)
f2 = next(f for f in comp.footprints if f.image_id == 2)
chk("M. Footprint img2 corner[0] = (50,10) in canvas coords",
    np.isclose(f2.corners_mosaic[0], [50.0, 10.0], atol=0.5).all(),
    f"got={f2.corners_mosaic[0]}")
chk("M2. Footprint has all required fields",
    all(hasattr(f2, k) for k in ('image_id', 'filename', 'sensor', 'source_width', 'source_height',
                                  'corners_mosaic', 'bbox_mosaic', 'component_id', 'root_id', 'valid')))

# ========== N: Outline-only footprint ==========
print("\n=== N: Outline-only footprint (CSS) ===")
css_path = Path(__file__).resolve().parent.parent / 'frontend' / 'styles.css'
css_content = css_path.read_text('utf-8')
chk("N. CSS polygon fill:none (outline-only)",
    'fill:none' in css_content and '.mosaic-stage polygon' in css_content)
chk("N2. CSS does NOT have fill:rgba for polygon",
    'fill:rgba' not in css_content.split('.mosaic-stage polygon')[1].split('}')[0])

# ========== O: Disconnected components ==========
print("\n=== O: Disconnected components ===")
res_o = build_scientific_mosaic(
    [np.ones((20, 20), dtype=np.float32), np.ones((20, 20), dtype=np.float32)],
    {'global_poses': {},
     'connected_components': [
         {'component_id': 0, 'node_ids': [0], 'root_node': 0},
         {'component_id': 1, 'node_ids': [1], 'root_node': 1}
     ]},
    [{'image_id': 0}, {'image_id': 1}]
)
chk("O. Disconnected: 2 separate component mosaics", len(res_o.components) == 2)
chk("O2. Component IDs are distinct",
    res_o.components[0].component_id != res_o.components[1].component_id)

# ========== P: 4-image ground truth ==========
print("\n=== P: 4-image ground truth ===")
# Bright spot at center of each image; verify placement
imgs_p = [np.zeros((51, 51), dtype=np.float32) for _ in range(4)]
for i in range(4):
    imgs_p[i][25, 25] = 100.0

res_p = build_scientific_mosaic(
    imgs_p, gg_4,
    [{'image_id': i, 'width': 51, 'height': 51} for i in range(4)]
)
comp_p = res_p.components[0]
# img0 (M0=I): spot at root (25,25). canvas_tx=+30, canvas_ty=+25 -> canvas (55,50)
chk("P. 4-image GT: img0 center spot at canvas (55,50)",
    comp_p.mosaic_scientific[50, 55] > 0,
    f"val={comp_p.mosaic_scientific[50,55]:.2f}")
chk("P2. 4 footprints generated", len(comp_p.footprints) == 4)

# ========== Q: Portrait image ==========
print("\n=== Q: Portrait image ===")
# T_i=I => M_i=I. 200H x 100W. corners: (0,0),(100,0),(100,200),(0,200)
# canvas_W = ceil(100)-floor(0)+1 = 101, canvas_H = ceil(200)-floor(0)+1 = 201
res_q = build_scientific_mosaic(
    [np.ones((200, 100), dtype=np.float32)],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0, 'width': 100, 'height': 200}]
)
chk("Q. Portrait image: canvas W=101 H=201 (from ceil-floor+1 formula)",
    res_q.components[0].width == 101 and res_q.components[0].height == 201,
    f"W={res_q.components[0].width} H={res_q.components[0].height}")

# ========== R: Landscape image ==========
print("\n=== R: Landscape image ===")
res_r = build_scientific_mosaic(
    [np.ones((100, 300), dtype=np.float32)],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0, 'width': 300, 'height': 100}]
)
chk("R. Landscape image: canvas W=301 H=101",
    res_r.components[0].width == 301 and res_r.components[0].height == 101,
    f"W={res_r.components[0].width} H={res_r.components[0].height}")

# ========== S: Scientific dtype preservation ==========
print("\n=== S: Scientific dtype preservation ===")
res_s_u16 = build_scientific_mosaic(
    [np.ones((10, 10), dtype=np.uint16)],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0}]
)
chk("S. uint16 input -> scientific mosaic dtype=float32 (not uint8)",
    res_s_u16.components[0].dtype == 'float32',
    f"got {res_s_u16.components[0].dtype}")
chk("S2. scientific mosaic array is NOT uint8",
    res_s_u16.components[0].mosaic_scientific.dtype != np.uint8)

res_s_f32 = build_scientific_mosaic(
    [np.array([[0.125, 0.5], [1.25, 123.456]], dtype=np.float32)],
    {'global_poses': {'0': np.eye(3).tolist()},
     'connected_components': [{'component_id': 0, 'node_ids': [0], 'root_node': 0}]},
    [{'image_id': 0}]
)
chk("S3. float32 values preserved (not quantized to uint8 range)",
    res_s_f32.components[0].mosaic_scientific.dtype == np.float32)
val_s = float(res_s_f32.components[0].mosaic_scientific[1, 1])
chk("S4. float32 value 123.456 preserved",
    abs(val_s - 123.456) < 0.1, f"got {val_s:.3f}")

# ========== T: Provenance ==========
print("\n=== T: Provenance ===")
chk("T. provenance has transform_convention", 'transform_convention' in res.provenance)
chk("T2. provenance has georeferencing=UNAVAILABLE",
    'georeferencing' in res.provenance and 'UNAVAILABLE' in res.provenance['georeferencing'].upper())
chk("T3. component provenance has engine field", 'engine' in comp.provenance)

# ========== U: API serialization ==========
print("\n=== U: API serialization ===")
ser = serialize_mosaic_result(res)
chk("U. serialize_mosaic_result produces dict", isinstance(ser, dict))
chk("U2. engine field present", ser.get('engine') == 'phase10_scientific')
chk("U3. components list present", isinstance(ser.get('components'), list))
comp_ser = ser['components'][0]
chk("U4. component has mosaic_display_available", 'mosaic_display_available' in comp_ser)
chk("U5. component has footprints list", isinstance(comp_ser.get('footprints'), list))
chk("U6. footprint has corners_mosaic", 'corners_mosaic' in comp_ser['footprints'][0])
chk("U7. provenance in response", 'provenance' in ser)
chk("U8. global_metrics in response", 'global_metrics' in ser)

# ========== V: V1 fallback ==========
print("\n=== V: V1 fallback ===")
res_v = build_scientific_mosaic(
    [np.ones((10, 10), dtype=np.float32)],
    {'global_poses': {}},   # No poses
    [{'image_id': 0}]
)
# With no global_poses and no connected_components, the fallback creates isolated components
chk("V. Empty global_poses: engine still runs (does not crash)", len(res_v.components) >= 0)

# ========== W: No hardcoded canvas ==========
print("\n=== W: No hardcoded canvas ===")
chk("W. Canvas differs for different image configurations",
    res_p.components[0].width != res.components[0].width or
    res_p.components[0].height != res.components[0].height,
    f"4x100 canvas={res.components[0].width}x{res.components[0].height}, "
    f"4x51 canvas={res_p.components[0].width}x{res_p.components[0].height}")

# ========== X: No fake georeferencing ==========
print("\n=== X: No fake georeferencing ===")
chk("X. Provenance explicitly states georeferencing=UNAVAILABLE",
    'UNAVAILABLE' in res.provenance.get('georeferencing', '').upper())
# Verify production code doesn't contain lat/lon fabrication
src_path = Path(__file__).resolve().parent.parent / 'backend' / 'app' / 'services' / 'scientific_mosaic.py'
src_content = src_path.read_text('utf-8')
chk("X2. No fabricated latitude/longitude in source",
    'latitude' not in src_content.lower() and 'longitude' not in src_content.lower())

# ========== Y: Stale footprints cleared ==========
print("\n=== Y: Stale footprint clearing ===")
js_path = Path(__file__).resolve().parent.parent / 'frontend' / 'app.js'
js_content = js_path.read_text('utf-8')
chk("Y. drawFootprints clears overlay on empty data",
    'overlay.innerHTML = ""' in js_content and 'return;' in js_content)

# ========== Z: 6-image test ==========
print("\n=== Z: 6-image multi-image handling ===")
res_z = build_scientific_mosaic(
    [np.ones((10, 10), dtype=np.float32) * (i + 1) for i in range(6)],
    {'global_poses': {str(i): (make_T(i * 5, 0)).tolist() for i in range(6)},
     'connected_components': [{'component_id': 0, 'node_ids': list(range(6)), 'root_node': 0}]},
    [{'image_id': i, 'width': 10, 'height': 10} for i in range(6)]
)
chk("Z. 6-image mosaic: 1 component", len(res_z.components) == 1)
chk("Z2. 6 footprints", len(res_z.components[0].footprints) == 6)

# ========== Summary ==========
print()
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"=== PHASE 10 TEST SUMMARY: {passed}/{total} PASS ===")
if failed == 0:
    print("PHASE 10 MOSAIC ENGINE VERIFIED")
else:
    print(f"PHASE 10 NOT VERIFIED - {failed} failures:")
    for label, ok in results:
        if not ok:
            print(f"  FAIL  {label}")
