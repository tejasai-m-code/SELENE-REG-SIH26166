import json
import urllib.request
from pathlib import Path

# Form boundary for multipart/form-data
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"


def make_multipart_body(fields: dict, files: dict) -> tuple[bytes, str]:
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type


def main():
    print("Testing End-to-End API at http://localhost:5000/api/register")

    # 1. Health check
    req = urllib.request.Request("http://localhost:5000/api/healthz")
    with urllib.request.urlopen(req) as resp:
        print("Health status:", resp.status, json.loads(resp.read().decode("utf-8")))

    # 2. Test exact problematic moon pair
    moon100_bytes = Path("tests/fixtures/moon100.png").read_bytes()
    moon99_bytes = Path("tests/fixtures/moon99.png").read_bytes()

    fields = {
        "detector": "sift",
        "ratio": "0.75",
        "representation": "raw",
        "max_features": "6000",
        "refinement_methods": "taylor,phase,ecc,quadratic",
    }
    files = {
        "source": ("moon100.png", moon100_bytes, "image/png"),
        "reference": ("moon99.png", moon99_bytes, "image/png"),
    }
    body, ct = make_multipart_body(fields, files)

    req = urllib.request.Request(
        "http://localhost:5000/api/register",
        data=body,
        headers={"Content-Type": ct},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))

    print("\n--- PROBLEMATIC PAIR RESULT ---")
    print("Registration status:", data.get("registration_status"))
    print("Success:", data.get("success"))
    print("Inliers count:", data.get("metrics", {}).get("inliers"))
    print("RMSE:", data.get("metrics", {}).get("rmse_pixels"))
    print("Outputs:", data.get("outputs"))

    decomp = data.get("transform_decomposition") or {}
    print("\n--- GEOMETRY TELEMETRY ---")
    print("Projected size:", f"{decomp.get('projected_width')} x {decomp.get('projected_height')} px")
    print("Reference size:", f"{decomp.get('reference_width')} x {decomp.get('reference_height')} px")
    print("Scale ratio:", decomp.get("scale_ratio"))
    print("Rotation degrees:", decomp.get("rotation_degrees"))
    print("Perspective distortion:", decomp.get("perspective_distortion"))
    print("Canvas expansion factor:", decomp.get("canvas_expansion_factor"))
    print("Overlap IoU:", decomp.get("iou"))
    print("Is plausible:", decomp.get("is_plausible"))
    print("Issues:", decomp.get("issues"))

    assert data.get("registration_status") in ("PASS", "PASS_WITH_WARNING"), "Registration must pass!"
    assert decomp.get("is_plausible") is True, "Transform must be plausible!"
    assert decomp.get("scale_ratio") is not None and abs(decomp.get("scale_ratio") - 1.0) < 0.15, "Scale must be isotropic!"
    assert decomp.get("projected_width", 0) > 400, "Projected width must not be collapsed!"

    # 3. Verify match visualization download endpoint
    vis_url = data["outputs"]["match_visualization"]
    assert vis_url, "match_visualization output must exist!"
    full_vis_url = f"http://localhost:5000{vis_url}"
    with urllib.request.urlopen(full_vis_url) as resp:
        assert resp.status == 200
        vis_bytes = resp.read()
        print(f"\nMatch visualization downloaded successfully: {len(vis_bytes)} bytes")
        assert len(vis_bytes) > 50000, "Visualization must be full resolution, not empty thumbnail"

    # 4. Verify registered image download endpoint
    reg_url = data["outputs"]["registered_image"]
    assert reg_url, "registered_image output must exist!"
    full_reg_url = f"http://localhost:5000{reg_url}"
    with urllib.request.urlopen(full_reg_url) as resp:
        assert resp.status == 200
        reg_bytes = resp.read()
        print(f"Registered image downloaded successfully: {len(reg_bytes)} bytes")

    # 5. Check correspondences
    corrs = data.get("correspondences") or []
    print(f"Total serialized correspondences: {len(corrs)}")
    if corrs:
        sample = corrs[0]
        print(f"Sample correspondence #1: Source={sample.get('source')}, Ref={sample.get('reference')}, Residual={sample.get('residual')} px, Status={sample.get('status')}")

    # 6. Test wrong pair rejection
    print("\n--- TESTING WRONG PAIR REJECTION ---")
    import cv2
    import numpy as np
    blank_img = np.zeros((200, 200), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".png", blank_img)
    blank_bytes = blank_buf.tobytes()
    files_wrong = {
        "source": ("moon100.png", moon100_bytes, "image/png"),
        "reference": ("blank.png", blank_bytes, "image/png"),
    }
    body_wrong, ct_wrong = make_multipart_body(fields, files_wrong)
    req_wrong = urllib.request.Request(
        "http://localhost:5000/api/register",
        data=body_wrong,
        headers={"Content-Type": ct_wrong},
        method="POST",
    )
    with urllib.request.urlopen(req_wrong) as resp:
        wrong_data = json.loads(resp.read().decode("utf-8"))
    print("Wrong pair status:", wrong_data.get("registration_status"))
    print("Wrong pair success:", wrong_data.get("success"))
    print("Wrong pair registered_image:", wrong_data.get("outputs", {}).get("registered_image"))
    assert wrong_data.get("registration_status") in ("FAIL", "REVIEW")
    assert wrong_data.get("success") is False
    assert wrong_data.get("outputs", {}).get("registered_image") is None, "Must NOT generate registered image for failed pair!"

    print("\nALL HTTP END-TO-END TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    main()
