from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import xml.etree.ElementTree as ET

@dataclass
class ProductIdentity:
    filename: Optional[str] = None
    product_id: Optional[str] = None
    logical_identifier: Optional[str] = None
    version_id: Optional[str] = None
    title: Optional[str] = None
    mission: Optional[str] = None
    spacecraft: Optional[str] = None
    instrument: Optional[str] = None
    sensor: Optional[str] = None
    product_type: Optional[str] = None
    processing_level: Optional[str] = None

@dataclass
class RasterInfo:
    width: Optional[int] = None
    height: Optional[int] = None
    dtype: Optional[str] = None
    bit_depth: Optional[int] = None
    number_of_bands: Optional[int] = None
    channel_count: Optional[int] = None
    sample_format: Optional[str] = None
    band_names: List[str] = field(default_factory=list)

@dataclass
class SpectralInfo:
    band_index: int
    wavelength: Optional[float] = None
    bandwidth: Optional[float] = None
    units: Optional[str] = None
    spectral_name: Optional[str] = None
    valid_range: Optional[tuple] = None
    calibration_information: Optional[str] = None

@dataclass
class AcquisitionInfo:
    date_time: Optional[str] = None
    stop_date_time: Optional[str] = None
    observation_duration: Optional[float] = None
    sun_azimuth: Optional[float] = None
    sun_elevation: Optional[float] = None
    incidence_angle: Optional[float] = None
    emission_angle: Optional[float] = None
    phase_angle: Optional[float] = None

@dataclass
class SpatialInfo:
    gsd: Optional[float] = None
    gsd_x: Optional[float] = None
    gsd_y: Optional[float] = None
    gsd_unit: Optional[str] = None
    crs: Optional[str] = None
    projection: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    geotransform: Optional[List[float]] = None
    georeferencing_status: str = "UNAVAILABLE"

@dataclass
class RadiometricInfo:
    scale_factor: Optional[float] = None
    offset: Optional[float] = None
    units: Optional[str] = None
    valid_range: Optional[tuple] = None
    fill_value: Optional[float] = None

@dataclass
class ProvenanceInfo:
    metadata_source: str
    parser_used: str
    extraction_status: str
    warnings: List[str] = field(default_factory=list)
    unavailable_fields: List[str] = field(default_factory=list)
    evidence_hierarchy: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ProductMetadata:
    identity: ProductIdentity = field(default_factory=ProductIdentity)
    raster: RasterInfo = field(default_factory=RasterInfo)
    spectral: List[SpectralInfo] = field(default_factory=list)
    acquisition: AcquisitionInfo = field(default_factory=AcquisitionInfo)
    spatial: SpatialInfo = field(default_factory=SpatialInfo)
    radiometric: RadiometricInfo = field(default_factory=RadiometricInfo)
    provenance: ProvenanceInfo = field(default_factory=lambda: ProvenanceInfo("Unknown", "None", "MISSING"))
    metadata_conflict: bool = False

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)

def safe_float(val: str) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except ValueError:
        return None

def safe_int(val: str) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except ValueError:
        return None

def strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def valid_gsd(val: Optional[float]) -> Optional[float]:
    import numpy as np
    if val is not None and np.isfinite(val) and val > 0:
        return float(val)
    return None

def parse_pds4_xml(xml_content: str) -> ProductMetadata:
    warnings = []
    missing = []
    
    pm = ProductMetadata(
        provenance=ProvenanceInfo(
            metadata_source="PDS4 XML",
            parser_used="xml.etree.ElementTree",
            extraction_status="PARTIAL"
        )
    )
    
    if not xml_content or not xml_content.strip():
        pm.provenance.extraction_status = "ERROR"
        pm.provenance.warnings.append("Empty PDS4 XML content.")
        return pm

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        pm.provenance.extraction_status = "ERROR"
        pm.provenance.warnings.append(f"Malformed XML: {e}")
        return pm

    def find_text(node, target_tag):
        if node is None: return None
        for child in node.iter():
            if strip_ns(child.tag) == target_tag:
                return child.text.strip() if child.text else None
        return None
        
    ident_area = None
    for child in root.iter():
        if strip_ns(child.tag) == "Identification_Area":
            ident_area = child
            break
            
    if ident_area is not None:
        pm.identity.logical_identifier = find_text(ident_area, "logical_identifier")
        pm.identity.product_id = pm.identity.logical_identifier
        pm.identity.version_id = find_text(ident_area, "version_id")
        pm.identity.title = find_text(ident_area, "title")
        pm.identity.product_type = find_text(ident_area, "product_class")
        
        gsd_val = find_text(ident_area, "pixel_resolution")
        if gsd_val:
            try:
                pm.spatial.gsd = float(gsd_val)
                pm.spatial.gsd_x = float(gsd_val)
                pm.spatial.gsd_y = float(gsd_val)
            except ValueError:
                pass
            for child in ident_area.iter():
                if strip_ns(child.tag) == "pixel_resolution":
                    if "unit" in child.attrib:
                        pm.spatial.gsd_unit = child.attrib["unit"]
                    break

    for child in root.iter():
        tag = strip_ns(child.tag)
        if tag in ("Array_2D_Image", "Array_3D_Image", "Array_3D_Spectrum"):
            axes = find_text(child, "axes")
            if axes:
                pm.raster.number_of_bands = 1
                if tag == "Array_3D_Image" or tag == "Array_3D_Spectrum":
                    pm.raster.number_of_bands = safe_int(axes)
            
            for element in child.iter():
                etag = strip_ns(element.tag)
                if etag == "Axis_Array":
                    axis_name = find_text(element, "axis_name")
                    elements = find_text(element, "elements")
                    if axis_name == "Line": pm.raster.height = safe_int(elements)
                    if axis_name == "Sample": pm.raster.width = safe_int(elements)
                    if axis_name == "Band": 
                        pm.raster.number_of_bands = safe_int(elements)
                        pm.raster.channel_count = safe_int(elements)
                if etag == "Element_Array":
                    pm.raster.dtype = find_text(element, "data_type")
            break


    obs_area = None
    for child in root.iter():
        if strip_ns(child.tag) == "Observation_Area":
            obs_area = child
            break
            
    if obs_area is not None:
        time_coords = None
        for child in obs_area.iter():
            if strip_ns(child.tag) == "Time_Coordinates":
                time_coords = child
                break
        if time_coords is not None:
            pm.acquisition.date_time = find_text(time_coords, "start_date_time")
            pm.acquisition.stop_date_time = find_text(time_coords, "stop_date_time")
        else:
            pm.acquisition.date_time = find_text(obs_area, "start_date_time")
            
        investigation = None
        for child in obs_area.iter():
            if strip_ns(child.tag) == "Investigation_Area":
                investigation = child
                break
        if investigation is not None:
            pm.identity.mission = find_text(investigation, "name")
            
        obs_sys = None
        for child in obs_area.iter():
            if strip_ns(child.tag) == "Observing_System":
                obs_sys = child
                break
        if obs_sys is not None:
            sys_comp = find_text(obs_sys, "name")
            if sys_comp:
                sys_comp_upper = sys_comp.upper()
                if "TMC" in sys_comp_upper or "TERRAIN MAPPING" in sys_comp_upper: pm.identity.sensor = "TMC"
                elif "OHRC" in sys_comp_upper or "HIGH RESOLUTION CAMERA" in sys_comp_upper: pm.identity.sensor = "OHRC"
                elif "IIRS" in sys_comp_upper or "INFRARED SPECTROMETER" in sys_comp_upper: pm.identity.sensor = "IIRS"
                else: pm.identity.sensor = sys_comp

    # Georeferencing is unavailable by default, check for Cartography
    carto = False
    map_proj = False
    valid_georef = False
    proj_name = None
    
    for child in root.iter():
        tag = strip_ns(child.tag)
        if tag == "Cartography":
            carto = True
        if tag in ("Spatial_Reference_Information", "Map_Projection"):
            map_proj = True
            pn = find_text(child, "map_projection_name")
            if pn:
                proj_name = pn
                valid_georef = True

    if valid_georef:
        pm.spatial.georeferencing_status = "AVAILABLE"
        pm.spatial.crs = proj_name
    elif carto or map_proj:
        pm.spatial.georeferencing_status = "PARTIAL"
    else:
        pm.spatial.georeferencing_status = "UNAVAILABLE"

    if not pm.identity.product_id: missing.append("product_id")
    if not pm.identity.mission: missing.append("mission")
    if not pm.acquisition.date_time: missing.append("start_date_time")

    if pm.identity.sensor == "IIRS":
        pm.provenance.warnings.append("ENVIRONMENT-BLOCKED — Spectral cube parser unavailable. Cannot slice IIRS cube without proper rasterio/gdal/envi dependencies.")
        pm.provenance.extraction_status = "PARTIAL"

    pm.provenance.unavailable_fields = missing
    if pm.provenance.extraction_status != "PARTIAL":
        pm.provenance.extraction_status = "SUCCESS" if not missing else "PARTIAL"
    
    return pm

def extract_geotiff_metadata(data: bytes) -> ProductMetadata:
    pm = ProductMetadata(
        provenance=ProvenanceInfo(
            metadata_source="GeoTIFF",
            parser_used="rasterio/tifffile",
            extraction_status="ERROR"
        )
    )
    pm.provenance.warnings.append("ENVIRONMENT-BLOCKED — GDAL/osgeo unavailable")
    pm.provenance.unavailable_fields = ["crs", "transform", "bounds", "pixel_size"]
    pm.spatial.georeferencing_status = "ENVIRONMENT-BLOCKED"
    return pm

def infer_sensor_from_string(text: str) -> Optional[str]:
    name = text.upper()
    if "OHRC" in name or "CH2_OHC" in name or "HIGH RESOLUTION CAMERA" in name: return "OHRC"
    if "TMC" in name or "TERRAIN MAPPING" in name: return "TMC"
    if "IIRS" in name or "INFRARED SPECTROMETER" in name: return "IIRS"
    if "LROC" in name or "M1" in name: return "LROC"
    return None

def build_product_metadata(
    filename: str, 
    image_data: bytes, 
    pds4_xml: Optional[str] = None,
    directory_name: str = "",
    user_confirmed_sensor: str = "",
    label_content: str = ""
) -> ProductMetadata:
    evidence = []
    pm = ProductMetadata()
    
    sensors_found = []

    # 1. PDS4
    if pds4_xml:
        pm = parse_pds4_xml(pds4_xml)
        if pm.identity.sensor:
            evidence.append({"sensor": pm.identity.sensor, "source": "PDS4", "status": "explicit", "precedence": 1})
            sensors_found.append(pm.identity.sensor)

    # 2. GeoTIFF
    if filename.lower().endswith((".tif", ".tiff")):
        gtiff_pm = extract_geotiff_metadata(image_data)
        if gtiff_pm.spatial.georeferencing_status != "UNAVAILABLE":
            pm.spatial.georeferencing_status = gtiff_pm.spatial.georeferencing_status
        for w in gtiff_pm.provenance.warnings:
            if w not in pm.provenance.warnings:
                pm.provenance.warnings.append(w)
        if gtiff_pm.identity.sensor:
            evidence.append({"sensor": gtiff_pm.identity.sensor, "source": "GeoTIFF", "status": "explicit", "precedence": 2})
            sensors_found.append(gtiff_pm.identity.sensor)

    # 3. Label
    if label_content:
        lbl_inferred = infer_sensor_from_string(label_content)
        if lbl_inferred:
            evidence.append({"sensor": lbl_inferred, "source": "label", "status": "inferred", "precedence": 3})
            sensors_found.append(lbl_inferred)

    # 4. Filename
    if filename:
        fn_inferred = infer_sensor_from_string(filename)
        if fn_inferred:
            evidence.append({"sensor": fn_inferred, "source": "filename", "status": "inferred", "precedence": 4})
            sensors_found.append(fn_inferred)
            
    # 5. Directory
    if directory_name:
        dir_inferred = infer_sensor_from_string(directory_name)
        if dir_inferred:
            evidence.append({"sensor": dir_inferred, "source": "directory", "status": "inferred", "precedence": 5})
            sensors_found.append(dir_inferred)
            
    # 6. User confirmation (explicit override)
    user_override = False
    if user_confirmed_sensor:
        evidence.append({"sensor": user_confirmed_sensor, "source": "user confirmation", "status": "explicit", "precedence": 0, "is_override": True})
        sensors_found.append(user_confirmed_sensor)
        user_override = True

    # Automatic evidence hierarchy: PDS4(1) -> GeoTIFF(2) -> Label(3) -> Filename(4) -> Directory(5)
    # User Confirmation is treated as an explicit override mechanism, separate from metadata-derived evidence.
    if evidence:
        evidence.sort(key=lambda x: x["precedence"])
        
        # Determine automatic sensor (ignoring user override for conflict detection)
        auto_evidence = [e for e in evidence if not e.get("is_override")]
        
        if user_override:
            pm.identity.sensor = user_confirmed_sensor
            pm.provenance.metadata_source = "User Confirmation"
            pm.provenance.extraction_status = "EXPLICIT"
        elif auto_evidence:
            pm.identity.sensor = auto_evidence[0]["sensor"]
            if pm.provenance.extraction_status == "MISSING":
                pm.provenance.metadata_source = auto_evidence[0]["source"]
                pm.provenance.extraction_status = auto_evidence[0]["status"].upper()
        else:
            pm.identity.sensor = "UNKNOWN"
            
        # Detect conflicts among automatic evidence
        if len(set(e["sensor"] for e in auto_evidence)) > 1:
            pm.metadata_conflict = True
    else:
        pm.identity.sensor = "UNKNOWN"

    if pm.metadata_conflict:
        distinct = set(e["sensor"] for e in evidence if not e.get("is_override"))
        pm.provenance.warnings.append(f"Sensor identification conflict: {list(distinct)}")

    pm.identity.filename = filename
    pm.provenance.evidence_hierarchy = evidence
    
    return pm
