import numpy as np, sys
sys.path.insert(0,'backend')
from app.services.scientific_mosaic import build_scientific_mosaic

def make_T(tx,ty):
    H=np.eye(3,dtype=np.float64); H[0,2]=tx; H[1,2]=ty; return H

# L: 2-image overlap check
gp = {'0': np.eye(3).tolist(), '1': make_T(30,0).tolist()}
images = [np.full((100,100),10,dtype=np.float32), np.full((100,100),20,dtype=np.float32)]
gg = {'global_poses': gp, 'connected_components': [{'component_id':0,'node_ids':[0,1],'root_node':0}]}
ni = [{'image_id':i,'width':100,'height':100} for i in range(2)]
res = build_scientific_mosaic(images, gg, ni)
comp = res.components[0]
print(f'L 2-img canvas: W={comp.width} H={comp.height}')
maxov = comp.overlap_statistics.get('max_overlap', 0)
print(f'overlap_max={maxov}')
# T1=T(30,0) => M1=T(-30,0): img1 corners in root x:[-30,70]
# min_x=-30, canvas_tx=+30
# Overlap region in root: x:[0,70] => canvas: x:[30,100]
# At canvas(50,60): root_x=60-30=30, root_y=50 -- in both img0 and img1
val = comp.mosaic_scientific[50,60]
print(f'canvas[50,60] val={val:.4f}  (expect ~15 = avg(10,20))')
# Check the middle of overlap
# Print first few nonzero values to understand
print(f'mosaic[50, 30:70] = {comp.mosaic_scientific[50,30:40]}')

# M: footprint corners 
gp4 = {'0': np.eye(3).tolist(), '1': make_T(30,0).tolist(), '2': make_T(-20,15).tolist(), '3': make_T(10,25).tolist()}
images4 = [np.zeros((100,100),dtype=np.float32) for _ in range(4)]
gg4 = {'global_poses': gp4, 'connected_components': [{'component_id':0,'node_ids':[0,1,2,3],'root_node':0}]}
ni4 = [{'image_id':i,'width':100,'height':100} for i in range(4)]
res4 = build_scientific_mosaic(images4, gg4, ni4)
comp4 = res4.components[0]
f2 = next(f for f in comp4.footprints if f.image_id==2)
print(f'M: footprint 2 corners_mosaic[0] = {f2.corners_mosaic[0]}')
# T2=T(-20,15) => M2=T(20,-15)
# img2 corner (0,0) in root: M2@(0,0) = (20,-15)
# min_x=-30 (from img1), min_y=-25 (from img3)
# canvas_x=20-(-30)=50, canvas_y=-15-(-25)=10
print(f'Expected footprint[2] corner: (50.0, 10.0)')
print(f'Test checked [0,15] which would be if min_x=-20,min_y=0')

# B: transform inversion proof
T1 = make_T(30,0)
M1 = np.linalg.inv(T1)
p_img = np.array([50.0, 50.0, 1.0])
p_root = M1 @ p_img
print(f'B: img1 pixel (50,50) in root via inv(T1)=T(-30,0): ({p_root[0]:.1f},{p_root[1]:.1f})')
p_back = T1 @ p_root
print(f'B: root (20,50) via T1=T(30,0): ({p_back[0]:.1f},{p_back[1]:.1f}) should be (50,50)')
