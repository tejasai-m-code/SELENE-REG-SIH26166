import cv2
import numpy as np
from dataclasses import dataclass, field
import time

@dataclass
class MosaicFootprint:
    image_id: int
    filename: str
    sensor: str
    source_width: int
    source_height: int
    corners_mosaic: list
    bbox_mosaic: list
    component_id: int
    root_id: int
    valid: bool

@dataclass
class ScientificMosaicComponent:
    component_id: int
    root_node: int
    width: int
    height: int
    dtype: str
    channels: int
    footprints: list[MosaicFootprint]
    overlap_statistics: dict
    provenance: dict
    diagnostics: dict
    valid_pixel_count: int
    invalid_pixel_count: int
    coverage_ratio: float
    mosaic_scientific: np.ndarray = None
    mosaic_display: np.ndarray = None
    mosaic_mask: np.ndarray = None

@dataclass
class ScientificMosaicResult:
    engine: str = 'phase10_scientific'
    components: list[ScientificMosaicComponent] = field(default_factory=list)
    global_metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return None
    valid_mask = np.isfinite(img)
    if not valid_mask.any():
        return np.zeros(img.shape, dtype=np.uint8)
    
    valid_pixels = img[valid_mask]
    min_val = np.min(valid_pixels)
    max_val = np.max(valid_pixels)
    
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
        
    norm_img = ((img - min_val) / (max_val - min_val) * 255.0)
    norm_img[~valid_mask] = 0
    return norm_img.astype(np.uint8)

def build_scientific_mosaic(
    images: list,
    global_graph_data: dict,
    nodes_info: list,
    settings: dict = None,
) -> ScientificMosaicResult:
    if settings is None:
        settings = {}
    
    nodata_values = settings.get('nodata_values', {})
    max_canvas_pixels = settings.get('max_canvas_pixels', 80_000_000)
    blend_mode = settings.get('blend_mode', 'weighted_average')
    
    result = ScientificMosaicResult(
        engine='phase10_scientific',
        global_metrics={'process_time_ms': 0},
        provenance={
            'transform_convention': 'T_i = root -> image_i, M_i = inv(T_i) for warping',
            'georeferencing': 'UNAVAILABLE',
            'description': 'Phase 10 scientific mosaic engine'
        }
    )
    
    start_time = time.time()
    
    # Check if we have global poses
    global_poses = global_graph_data.get('global_poses', {})
    
    components_data = global_graph_data.get('connected_components', [])
    
    # If no components but we have nodes, create a fallback component
    if not components_data and nodes_info:
        for idx, node in enumerate(nodes_info):
            # Treat each as a separate component if no components defined
            components_data.append({
                'component_id': idx,
                'node_ids': [idx],
                'root_node': idx
            })
    
    for comp in components_data:
        comp_id = comp.get('component_id', 0)
        node_ids = comp.get('node_ids', [])
        root_node = comp.get('root_node', -1)
        
        if not node_ids:
            continue
            
        if root_node == -1:
            root_node = node_ids[0]
            
        # Determine output dtype (upcast to float32 minimum)
        first_img = images[node_ids[0]]
        channels = 1 if first_img.ndim == 2 else first_img.shape[2]
        
        # Determine canvas bounds
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        
        image_transforms = {} # node_id -> M_i
        valid_nodes = []
        
        for node_id in node_ids:
            if node_id >= len(images):
                continue
                
            img = images[node_id]
            h, w = img.shape[:2]
            
            # Get T_i
            T_i = None
            if str(node_id) in global_poses:
                T_i = np.array(global_poses[str(node_id)])
            elif node_id == root_node:
                T_i = np.eye(3, dtype=np.float64)
                
            if T_i is None:
                continue
                
            try:
                M_i = np.linalg.inv(T_i)
            except np.linalg.LinAlgError:
                continue
                
            image_transforms[node_id] = M_i
            valid_nodes.append(node_id)
            
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32).reshape(1, -1, 2)
            
            # transform corners to root frame
            corners_root = cv2.perspectiveTransform(corners, M_i.astype(np.float32))[0]
            
            c_min_x, c_min_y = corners_root.min(axis=0)
            c_max_x, c_max_y = corners_root.max(axis=0)
            
            min_x = min(min_x, c_min_x)
            min_y = min(min_y, c_min_y)
            max_x = max(max_x, c_max_x)
            max_y = max(max_y, c_max_y)
            
        if not valid_nodes:
            continue
            
        canvas_W = int(np.ceil(max_x) - np.floor(min_x) + 1)
        canvas_H = int(np.ceil(max_y) - np.floor(min_y) + 1)

        # Canvas safety limit: raise explicitly rather than silently distorting geometry.
        # This is a resource guard, NOT an artificial scientific limit.
        # Caller may use settings['max_canvas_pixels'] to raise the limit.
        if canvas_W * canvas_H > max_canvas_pixels:
            raise ValueError(
                f"Phase 10: computed canvas {canvas_W}x{canvas_H}={canvas_W*canvas_H:,} px "
                f"exceeds safety limit {max_canvas_pixels:,} px for component {comp_id}. "
                f"Set settings['max_canvas_pixels'] to a higher value if this is expected. "
                f"The limit exists to prevent OOM, not as a scientific restriction."
            )
            
        canvas_translation = np.array([
            [1, 0, -np.floor(min_x)],
            [0, 1, -np.floor(min_y)],
            [0, 0, 1]
        ], dtype=np.float64)
        
        out_shape = (canvas_H, canvas_W, channels) if channels > 1 else (canvas_H, canvas_W)
        accum_num = np.zeros(out_shape, dtype=np.float64)
        accum_den = np.zeros(out_shape, dtype=np.float64)
        overlap_map = np.zeros(out_shape[:2], dtype=np.uint8)
        
        footprints = []
        
        for node_id in valid_nodes:
            img = images[node_id]
            M_i = image_transforms[node_id]
            final_H = canvas_translation @ M_i
            
            h, w = img.shape[:2]
            
            nodata = nodata_values.get(node_id, nodata_values.get(str(node_id)))
            source_mask = np.ones(img.shape[:2], dtype=bool)
            
            if nodata is not None:
                if img.ndim == 3:
                    source_mask &= ~np.isclose(img, nodata, equal_nan=False).all(axis=-1)
                else:
                    source_mask &= ~np.isclose(img, nodata, equal_nan=False)
                    
            if np.issubdtype(img.dtype, np.floating):
                if img.ndim == 3:
                    source_mask &= np.isfinite(img).all(axis=-1)
                else:
                    source_mask &= np.isfinite(img)
                    
            # Warping image
            # using default INTER_LINEAR, or INTER_NEAREST for integer/mask
            warped_img = cv2.warpPerspective(
                img, final_H, (canvas_W, canvas_H), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            warped_mask = cv2.warpPerspective(
                source_mask.astype(np.uint8), final_H, (canvas_W, canvas_H),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            ) > 0
            
            feather = cv2.GaussianBlur(warped_mask.astype(np.float32), (0, 0), sigmaX=8, sigmaY=8) * warped_mask
            
            if channels > 1:
                accum_num += warped_img.astype(np.float64) * feather[..., None]
                accum_den += feather[..., None]
            else:
                accum_num += warped_img.astype(np.float64) * feather
                accum_den += feather
                
            overlap_map += warped_mask.astype(np.uint8)
            
            # Compute footprints
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32).reshape(1, -1, 2)
            corners_canvas = cv2.perspectiveTransform(corners, final_H.astype(np.float32))[0]
            
            c_min_x, c_min_y = corners_canvas.min(axis=0)
            c_max_x, c_max_y = corners_canvas.max(axis=0)
            
            node_info = nodes_info[node_id] if node_id < len(nodes_info) else {}
            
            footprints.append(MosaicFootprint(
                image_id=node_id,
                filename=node_info.get('filename', f'image_{node_id}'),
                sensor=node_info.get('sensor', 'Unknown'),
                source_width=w,
                source_height=h,
                corners_mosaic=corners_canvas.tolist(),
                bbox_mosaic=[float(c_min_x), float(c_min_y), float(c_max_x), float(c_max_y)],
                component_id=comp_id,
                root_id=root_node,
                valid=True
            ))
            
        mosaic_scientific = (accum_num / np.maximum(accum_den, 1e-12)).astype(np.float32)
        mosaic_mask = accum_den[..., 0] > 1e-12 if channels > 1 else accum_den > 1e-12
        mosaic_display = normalize_to_uint8(mosaic_scientific)
        
        valid_pixels = int(np.sum(mosaic_mask))
        total_pixels = canvas_W * canvas_H
        
        comp_obj = ScientificMosaicComponent(
            component_id=comp_id,
            root_node=root_node,
            width=canvas_W,
            height=canvas_H,
            dtype=str(mosaic_scientific.dtype),
            channels=channels,
            footprints=footprints,
            overlap_statistics={'max_overlap': int(overlap_map.max()) if overlap_map.size > 0 else 0},
            provenance={'engine': 'phase10'},
            diagnostics={},
            valid_pixel_count=valid_pixels,
            invalid_pixel_count=total_pixels - valid_pixels,
            coverage_ratio=valid_pixels / total_pixels if total_pixels > 0 else 0,
            mosaic_scientific=mosaic_scientific,
            mosaic_display=mosaic_display,
            mosaic_mask=mosaic_mask
        )
        
        result.components.append(comp_obj)
        
    result.global_metrics['process_time_ms'] = (time.time() - start_time) * 1000
    
    return result

def serialize_mosaic_result(result: ScientificMosaicResult) -> dict:
    return {
        'engine': result.engine,
        'global_metrics': result.global_metrics,
        'provenance': result.provenance,
        'components': [
            {
                'component_id': c.component_id,
                'root_node': c.root_node,
                'width': c.width,
                'height': c.height,
                'dtype': c.dtype,
                'channels': c.channels,
                'footprints': [
                    {
                        'image_id': f.image_id,
                        'filename': f.filename,
                        'sensor': f.sensor,
                        'source_width': f.source_width,
                        'source_height': f.source_height,
                        'corners_mosaic': f.corners_mosaic,
                        'bbox_mosaic': f.bbox_mosaic,
                        'component_id': f.component_id,
                        'root_id': f.root_id,
                        'valid': f.valid
                    }
                    for f in c.footprints
                ],
                'overlap_statistics': c.overlap_statistics,
                'provenance': c.provenance,
                'diagnostics': c.diagnostics,
                'valid_pixel_count': c.valid_pixel_count,
                'invalid_pixel_count': c.invalid_pixel_count,
                'coverage_ratio': c.coverage_ratio,
                'mosaic_display_available': c.mosaic_display is not None
            }
            for c in result.components
        ]
    }
