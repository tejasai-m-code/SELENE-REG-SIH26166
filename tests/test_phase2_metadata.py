import cv2
import numpy as np
import tempfile
from pathlib import Path
from app.utils.image_utils import decode_upload
from app.services.metadata import parse_pds4_xml, build_product_metadata

def test_phase2():
    print("Testing Phase 2 Metadata Foundation...")
    
    # 1. Image decode (no XML)
    img = np.zeros((10, 10), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    raster = decode_upload(buf.tobytes(), "LROC_test.png")
    
    assert raster.metadata.identity.sensor == "LROC", "Filename inference failed"
    assert raster.metadata.provenance.metadata_source.lower() == "filename", "Provenance incorrect"
    assert raster.metadata.raster.channel_count == 1
    assert raster.metadata.raster.number_of_bands is None, "Should not fake bands"
    
    # 2. PDS4 XML parsing
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <Identification_Area>
        <logical_identifier>urn:isro:isda:ch2_tmc:calibrated:ch2_tmc_123</logical_identifier>
        <product_class>Product_Observational</product_class>
      </Identification_Area>
      <Observation_Area>
        <Time_Coordinates>
          <start_date_time>2020-01-01T12:00:00Z</start_date_time>
        </Time_Coordinates>
        <Investigation_Area>
          <name>CHANDRAYAAN-2</name>
        </Investigation_Area>
        <Observing_System>
          <name>TMC-2</name>
        </Observing_System>
      </Observation_Area>
    </Product_Observational>
    """
    
    pm = parse_pds4_xml(xml)
    assert pm.identity.sensor == "TMC", f"Expected TMC, got {pm.identity.sensor}"
    assert pm.identity.product_id == "urn:isro:isda:ch2_tmc:calibrated:ch2_tmc_123"
    assert pm.acquisition.date_time == "2020-01-01T12:00:00Z"
    
    # 3. Missing / Malformed
    pm_bad = parse_pds4_xml("<bad>")
    assert pm_bad.provenance.extraction_status == "ERROR"
    
    # 4. Raster compatibility
    raster2 = decode_upload(buf.tobytes(), "img.png", pds4_xml=xml)
    assert raster2.metadata.identity.sensor == "TMC"
    assert raster2.metadata.provenance.metadata_source == "PDS4 XML"
    assert len(raster2.metadata.provenance.evidence_hierarchy) == 1
    
    print("All Phase 2 tests passed.")

if __name__ == "__main__":
    test_phase2()
