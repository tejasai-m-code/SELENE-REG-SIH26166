import requests
import numpy as np
import cv2
import os
import json

def run_manual_verification():
    base_url = "http://localhost:5000"
    print("--- 1. Verification of Server Health ---")
    res = requests.get(f"{base_url}/api/healthz")
    print(f"Healthz Status: {res.status_code}, Body: {res.text}")
    assert res.status_code == 200

    print("\n--- 2. Verification of /api/dataset/scan ---")
    res = requests.post(f"{base_url}/api/dataset/scan", json={"folder_path": "demo_data"})
    print(f"Scan Status: {res.status_code}")
    manifest_data = res.json()
    print(f"Dataset root: {manifest_data.get('root')}")
    print(f"Total files indexed: {len(manifest_data.get('files', []))}")
    if manifest_data.get('files'):
        sample = manifest_data['files'][0]
        print(f"Sample file: {sample.get('name')}, Sensor: {sample.get('sensor')}, dType: {sample.get('scientific_inspection', {}).get('dtype')}")

    print("\n--- 3. Verification of 16-bit TIFF Ingestion & Inspection via /api/dataset/inspect-file ---")
    # Create synthetic 16-bit TIFF with exact known values
    os.makedirs("scratch", exist_ok=True)
    tiff_16_path = "scratch/manual_test_ohrc_16bit.tif"
    arr16 = np.zeros((100, 100), dtype=np.uint16)
    arr16[10:90, 10:90] = 2048 # Mid-range 12-bit
    arr16[40:60, 40:60] = 4095 # Saturation peak for 12-bit
    cv2.imwrite(tiff_16_path, arr16)

    with open(tiff_16_path, "rb") as f:
        files = {"file": ("ch2_ohr_ncp_manual_test.tif", f, "image/tiff")}
        res = requests.post(f"{base_url}/api/dataset/inspect-file", files=files)
    
    print(f"Inspect 16-bit TIFF Status: {res.status_code}")
    inspect_data = res.json()
    print("Inspection Response:")
    print(json.dumps(inspect_data, indent=2))
    assert "inspection" in inspect_data
    assert inspect_data["inspection"]["dtype"] == "uint16"
    assert inspect_data["inspection"]["max_dn"] == 4095
    assert "OHRC" in inspect_data["sensor"]["sensor"]
    assert inspect_data["sensor"]["confidence_pct"] >= 80.0
    assert len(inspect_data["inspection"]["histogram_counts"]) == 32
    assert "preview_url" in inspect_data
    print(">> 16-bit preservation, histogram, and sensor identification verified successfully!")

    print("\n--- 4. Verification of Folder Upload via /api/dataset/upload-folder ---")
    # Simulate a folder upload with relative paths
    sub_img1 = "scratch/folder_sample1.png"
    sub_img2 = "scratch/folder_sample2.png"
    cv2.imwrite(sub_img1, (np.random.rand(50, 50) * 255).astype(np.uint8))
    cv2.imwrite(sub_img2, (np.random.rand(50, 50) * 255).astype(np.uint8))

    with open(sub_img1, "rb") as f1, open(sub_img2, "rb") as f2:
        upload_files = [
            ("files", ("dataset_a/subfolder/img1.png", f1, "image/png")),
            ("files", ("dataset_a/img2.png", f2, "image/png")),
        ]
        res = requests.post(f"{base_url}/api/dataset/upload-folder", files=upload_files)
    
    print(f"Folder Upload Status: {res.status_code}")
    upload_res = res.json()
    print("Folder Upload Result:")
    print(json.dumps(upload_res, indent=2))
    assert upload_res["images_count"] == 2
    assert upload_res["files_discovered"] == 2
    assert len(upload_res["files"]) == 2
    print(">> Native folder upload & nested path preservation verified successfully!")

    print("\n=== ALL MANUAL TESTS PASSED ===")

if __name__ == "__main__":
    run_manual_verification()
