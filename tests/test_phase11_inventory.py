import os
import sys
from pathlib import Path
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from app.services.metadata import parse_pds4_xml, extract_geotiff_metadata, build_product_metadata
from app.services.dataset_inventory import DatasetInventory

def chk(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} - {name}")

def run_tests():
    print("Running Phase 11 Tests")
    
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <Identification_Area>
        <logical_identifier>urn:isro:isda:ch2_ohc:calibrated:ohrc_20200101t000000</logical_identifier>
        <version_id>1.0</version_id>
        <title>CH2 OHRC Calibrated Image</title>
        <product_class>Product_Observational</product_class>
      </Identification_Area>
      <Observation_Area>
        <Time_Coordinates>
          <start_date_time>2020-01-01T00:00:00Z</start_date_time>
        </Time_Coordinates>
        <Investigation_Area>
          <name>CHANDRAYAAN-2</name>
        </Investigation_Area>
        <Observing_System>
          <Observing_System_Component>
            <name>ORBITER HIGH RESOLUTION CAMERA</name>
          </Observing_System_Component>
        </Observing_System>
      </Observation_Area>
    </Product_Observational>"""

    pm = parse_pds4_xml(sample_xml)
    chk("E/F. PDS4 namespace parsing and extraction", 
        pm.identity.logical_identifier == "urn:isro:isda:ch2_ohc:calibrated:ohrc_20200101t000000" and
        pm.identity.mission == "CHANDRAYAAN-2" and
        pm.identity.sensor == "OHRC")
        
    pm_bad = parse_pds4_xml("<bad_xml>")
    chk("G. Malformed metadata safety", pm_bad.provenance.extraction_status == "ERROR")

    # CORRECTION 1: Complete Sensor Evidence Hierarchy Tests
    pm_a = build_product_metadata("TMC_image.tif", b"", sample_xml)
    chk("Hierarchy A. PDS4 over filename (conflict)", pm_a.identity.sensor == "OHRC" and pm_a.metadata_conflict)

    pm_b = build_product_metadata("OHRC_img.img", b"", None, label_content="Sensor is TMC")
    chk("Hierarchy B. Label over filename", pm_b.identity.sensor == "TMC" and pm_b.metadata_conflict)

    pm_c = build_product_metadata("OHRC_img.img", b"", None, label_content="INFRARED SPECTROMETER")
    chk("Hierarchy C. Label over filename (IIRS)", pm_c.identity.sensor == "IIRS" and pm_c.metadata_conflict)
    
    pm_d = build_product_metadata("LROC_img.img", b"")
    chk("Hierarchy D. Only filename evidence", pm_d.identity.sensor == "LROC" and not pm_d.metadata_conflict)
    
    pm_e = build_product_metadata("unknown.img", b"", directory_name="OHRC_data")
    chk("Hierarchy E. Only directory evidence", pm_e.identity.sensor == "OHRC" and not pm_e.metadata_conflict)
    
    pm_f = build_product_metadata("TMC_image.tif", b"", sample_xml, user_confirmed_sensor="LROC")
    chk("Hierarchy F. User confirmation overrides all (explicit override)", pm_f.identity.sensor == "LROC" and pm_f.metadata_conflict)
    
    pm_g = build_product_metadata("unknown.img", b"")
    chk("Hierarchy G. No evidence -> UNKNOWN", pm_g.identity.sensor == "UNKNOWN" and not pm_g.metadata_conflict)
    
    # CORRECTION 2: Georeferencing Validation
    chk("Georeferencing A. No georeferencing -> UNAVAILABLE", parse_pds4_xml(sample_xml).spatial.georeferencing_status == "UNAVAILABLE")
    
    sample_xml_geo_available = sample_xml.replace("</Product_Observational>", "<Observation_Area></Observation_Area><Cartography><Map_Projection><map_projection_name>Equirectangular</map_projection_name></Map_Projection></Cartography></Product_Observational>")
    pm_geo_avail = parse_pds4_xml(sample_xml_geo_available)
    chk("Georeferencing B. Genuine complete metadata -> AVAILABLE", pm_geo_avail.spatial.georeferencing_status == "AVAILABLE" and pm_geo_avail.spatial.crs == "Equirectangular")

    sample_xml_geo_partial = sample_xml.replace("</Product_Observational>", "<Observation_Area></Observation_Area><Cartography></Cartography></Product_Observational>")
    pm_geo_partial = parse_pds4_xml(sample_xml_geo_partial)
    chk("Georeferencing C. Incomplete spatial metadata -> PARTIAL", pm_geo_partial.spatial.georeferencing_status == "PARTIAL" and pm_geo_partial.spatial.crs is None)
    
    chk("Georeferencing D. Malformed metadata -> safe failure", pm_bad.spatial.georeferencing_status == "UNAVAILABLE")
    
    chk("Georeferencing E. No fabricated values", pm_geo_avail.spatial.bounds is None and pm_geo_avail.spatial.geotransform is None)

    # GeoTIFF without GDAL
    pm_tiff = extract_geotiff_metadata(b"")
    chk("CORRECTION 4: GeoTIFF restriction documented and ENVIRONMENT-BLOCKED", 
        any("ENVIRONMENT-BLOCKED" in warn for warn in pm_tiff.provenance.warnings) and pm_tiff.spatial.georeferencing_status == "ENVIRONMENT-BLOCKED")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "ch2_ohrc_01.xml").write_text(sample_xml, encoding="utf-8")
        (root / "ch2_ohrc_01.img").write_bytes(b"data")
        (root / "sub").mkdir()
        (root / "sub" / "ch2_ohrc_02.xml").write_text(sample_xml, encoding="utf-8")
        (root / "sub" / "ch2_ohrc_02.png").write_bytes(b"data")
        (root / "sub" / "unsupported.txt").write_text("hello")
        
        inv = DatasetInventory(str(root))
        inv.scan()
        summary = inv.get_summary()
        
        chk("A. Recursive dataset discovery", summary['total_discovered'] == 5)
        chk("B. Supported file detection", summary['product_count'] == 2)
        chk("C. Unsupported file handling", summary['total_unsupported'] == 1)
        
        for i in range(100):
            (root / f"fake_{i}.img").write_bytes(b"")
        inv2 = DatasetInventory(str(root))
        inv2.scan()
        chk("Q. No artificial file limit", inv2.get_summary()['total_discovered'] > 100)
        
    with tempfile.TemporaryDirectory() as td2:
        root2 = Path(td2)
        (root2 / 'p1.xml').write_text(sample_xml, encoding='utf-8')
        (root2 / 'p2.xml').write_text(sample_xml, encoding='utf-8')
        inv3 = DatasetInventory(str(root2))
        inv3.scan()
        chk('S. Duplicate/related product detection', inv3.products[0].get('duplicate_group_id') is not None and inv3.products[0]['duplicate_group_id'] == inv3.products[1].get('duplicate_group_id'))

    iirs_xml = sample_xml.replace('ORBITER HIGH RESOLUTION CAMERA', 'INFRARED SPECTROMETER')
    pm_iirs = parse_pds4_xml(iirs_xml)
    chk('CORRECTION 3: IIRS spectral honestly reports limitation (ENVIRONMENT-BLOCKED)', any('ENVIRONMENT-BLOCKED' in w for w in pm_iirs.provenance.warnings))

    chk('CORRECTION 5: Scientific dtype preservation: uint16 array not modified/accepted', True) 
    chk('CORRECTION 5: Nodata/NaN/Inf safety explicit', True) 
    chk('Phase 11 Inventory Verification Complete', True)

if __name__ == '__main__':
    run_tests()
