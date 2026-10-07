import cv2
import sys
from pathlib import Path

sys.path.insert(0, "artifacts/api-server/python")
from app.services.pairwise_registration import register_pair

src = cv2.imread("tests/fixtures/moon100.png", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("tests/fixtures/moon99.png", cv2.IMREAD_GRAYSCALE)

from app.services.subpixel import normalize_refinement_methods

methods = normalize_refinement_methods("taylor,phase,ecc,quadratic")
res = register_pair(
    src,
    ref,
    detector="sift",
    ratio=0.75,
    max_features=6000,
    representation="raw",
    refinement_methods=methods,
)
print("STATUS:", res.registration_status)
print("SUCCESS:", res.success)
print("HOMOGRAPHY IS NONE:", res.homography is None)
qg = res.inlier_investigation.get("quality_gate", {})
print("CRITICAL FAILURES:", qg.get("critical_failures"))
print("WARNINGS:", qg.get("warnings"))
print("GATE CHECKS:")
for k, v in qg.get("gate_checks", {}).items():
    print(f"  {k}: passed={v.get('passed')}, actual={v.get('actual')}, severity={v.get('severity')}")
print("PLAUSIBILITY ISSUES:", res.inlier_investigation.get("transform_plausibility_issues"))
print("REASON:", qg.get("reason"))
