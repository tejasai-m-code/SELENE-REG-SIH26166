import os
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.scientific_report import ScientificReportGenerator

def test_reporting():
    metrics_path = "phase12_metrics.json"
    output_dir = "reports"
    
    gen = ScientificReportGenerator(metrics_path, output_dir)
    
    # 1. Report object generation (JSON)
    data, json_path = gen.generate_json_report()
    assert os.path.exists(json_path), "JSON report not generated"
    
    # 2. Required sections exist
    assert "metadata" in data
    assert "architecture" in data
    assert "subpixel_summary" in data
    assert "cross_representation" in data
    assert "global_graph" in data
    assert "mosaic" in data
    assert "limitations" in data
    assert "regression_status" in data
    
    # 3. Phase 12 metrics are sourced correctly, checking corrected values
    # The corrected stats should NOT have 0.927 median. They should be ~0.029 median.
    assert data["subpixel_summary"]["consensus_median_px"] < 0.1, f"Stale median value detected: {data['subpixel_summary']['consensus_median_px']}"
    assert data["subpixel_summary"]["consensus_rmse_px"] < 0.1, f"Stale RMSE value detected: {data['subpixel_summary']['consensus_rmse_px']}"
    
    # 4. Synthetic / real distinction is preserved
    assert data["metadata"]["real_mission_data_validation"] == "NOT EXECUTED"
    
    # 5. PDF generation
    pdf_path = gen.generate_pdf_report(data)
    assert os.path.exists(pdf_path), "PDF report not generated"
    assert os.path.getsize(pdf_path) > 100, "PDF file is suspiciously small or empty"
    
    print("PASS: Phase 13 JSON and PDF reports generated successfully and metrics verified.")

if __name__ == "__main__":
    test_reporting()
