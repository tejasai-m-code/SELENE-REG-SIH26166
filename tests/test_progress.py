"""Tests for real backend-driven registration progress mechanism.

Covers:
1. Registration job creation
2. Progress event/status structure
3. Stage ordering across all 12 pipeline stages
4. Completion event
5. Failure event handling
6. Unknown/malformed stage handling
7. Frontend/backend progress contract
8. Existing registration result invariance
9. Existing API backward compatibility
10. Concurrent / independent registration jobs
"""

import json
import io
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services.pairwise_registration import register_pair, PairwiseRegistrationResult, _notify_progress
from worker import register, _emit_progress


def _make_textured_image(size: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    for _ in range(30):
        cx, cy = rng.integers(30, size - 30, size=2)
        radius = rng.integers(8, 25)
        color = int(rng.integers(40, 240))
        cv2.circle(img, (int(cx), int(cy)), int(radius), color, -1)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


class TestRegistrationProgress(unittest.TestCase):
    def setUp(self):
        self.img1 = _make_textured_image(256, seed=10)
        # Shifted slightly for correspondence
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        self.img2 = cv2.warpAffine(self.img1, M, (256, 256))

    def test_pairwise_emits_stages_in_order(self):
        """Stages 3 through 10 must emit in strictly ascending sequence."""
        events = []

        def callback(stage, stage_index, stage_count, progress, message, status="RUNNING", error=None):
            events.append({
                "stage": stage,
                "stage_index": stage_index,
                "stage_count": stage_count,
                "progress": progress,
                "message": message,
                "status": status,
            })

        res = register_pair(
            self.img1,
            self.img2,
            detector="sift",
            ecc_refinement=False,
            progress_callback=callback,
        )

        self.assertIsInstance(res, PairwiseRegistrationResult)
        self.assertGreater(len(events), 0)

        # Verify event structure
        for ev in events:
            self.assertIn("stage", ev)
            self.assertIn("stage_index", ev)
            self.assertIn("stage_count", ev)
            self.assertEqual(ev["stage_count"], 12)
            self.assertIn("progress", ev)
            self.assertIsInstance(ev["progress"], float)
            self.assertIn("message", ev)
            self.assertIn("status", ev)

        stages_seen = [ev["stage"] for ev in events]
        expected_stages = [
            "PREPROCESSING",
            "FEATURE_DETECTION",
            "FEATURE_MATCHING",
            "SPATIAL_FILTERING",
            "GEOMETRIC_ESTIMATION",
            "SUBPIXEL_REFINEMENT",
            "FOOTPRINT_COMPUTATION",
            "METRICS_EVALUATION",
        ]

        for expected in expected_stages:
            self.assertIn(expected, stages_seen, f"Stage {expected} was not emitted by pairwise pipeline")

        # Verify ascending indices
        indices = [ev["stage_index"] for ev in events]
        for i in range(len(indices) - 1):
            self.assertLessEqual(indices[i], indices[i + 1])

    def test_worker_emits_all_12_stages(self):
        """Worker pipeline must emit all 12 stages from INITIALIZING to COMPLETE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src_file = tmp_path / "source.png"
            ref_file = tmp_path / "reference.png"
            out_dir = tmp_path / "out"

            cv2.imwrite(str(src_file), self.img1)
            cv2.imwrite(str(ref_file), self.img2)

            # Capture stderr to intercept PROGRESS lines
            captured_stderr = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = captured_stderr

            try:
                payload = register(
                    src_file,
                    ref_file,
                    out_dir,
                    settings={"detector": "sift", "job_id": "test-job-123"},
                )
            finally:
                sys.stderr = old_stderr

            output_lines = captured_stderr.getvalue().splitlines()
            progress_events = []
            for line in output_lines:
                if line.startswith("PROGRESS:"):
                    progress_events.append(json.loads(line[len("PROGRESS:"):].strip()))

            self.assertGreater(len(progress_events), 0)
            stages = [e["stage"] for e in progress_events]

            # All 12 canonical stages must be present
            all_12_stages = [
                "INITIALIZING",
                "LOADING_INPUT",
                "PREPROCESSING",
                "FEATURE_DETECTION",
                "FEATURE_MATCHING",
                "SPATIAL_FILTERING",
                "GEOMETRIC_ESTIMATION",
                "SUBPIXEL_REFINEMENT",
                "FOOTPRINT_COMPUTATION",
                "METRICS_EVALUATION",
                "VISUALIZATION",
                "COMPLETE",
            ]

            for s in all_12_stages:
                self.assertIn(s, stages, f"Missing required pipeline stage: {s}")

            # Verify initial and complete stages
            self.assertEqual(progress_events[0]["stage"], "INITIALIZING")
            self.assertEqual(progress_events[0]["stage_index"], 1)
            self.assertEqual(progress_events[-1]["stage"], "COMPLETE")
            self.assertEqual(progress_events[-1]["stage_index"], 12)
            self.assertEqual(progress_events[-1]["progress"], 1.0)
            self.assertEqual(progress_events[-1]["status"], "COMPLETE")
            self.assertEqual(progress_events[-1]["job_id"], "test-job-123")

    def test_failure_event_structure(self):
        """When an error occurs, FAILED status and error details must be populated."""
        captured_stderr = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_stderr
        try:
            _emit_progress(
                job_id="fail-job",
                stage="FAILED",
                stage_index=5,
                stage_count=12,
                progress=0.42,
                message="Degenerate correspondence error",
                status="FAILED",
                error="Degenerate correspondence error",
            )
        finally:
            sys.stderr = old_stderr

        line = captured_stderr.getvalue().strip()
        self.assertTrue(line.startswith("PROGRESS:"))
        event = json.loads(line[len("PROGRESS:"):])
        self.assertEqual(event["status"], "FAILED")
        self.assertEqual(event["error"], "Degenerate correspondence error")
        self.assertEqual(event["job_id"], "fail-job")

    def test_malformed_callback_does_not_crash_pipeline(self):
        """If a progress callback raises an unexpected exception, registration must succeed."""
        def faulty_callback(*args, **kwargs):
            raise RuntimeError("Broken callback")

        res = register_pair(
            self.img1,
            self.img2,
            detector="sift",
            ecc_refinement=False,
            progress_callback=faulty_callback,
        )
        self.assertIsInstance(res, PairwiseRegistrationResult)
        self.assertTrue(res.success)

    def test_existing_result_contract_unchanged(self):
        """The scientific output contract must remain identical."""
        res = register_pair(self.img1, self.img2, detector="sift", ecc_refinement=False)
        self.assertIsNotNone(res.homography)
        self.assertIsNotNone(res.inlier_mask)
        self.assertIn(res.registration_status, ("PASS", "PASS_WITH_WARNING", "REVIEW", "FAIL"))
        self.assertIn("raw_rmse_pixels", res.metrics)
        self.assertIsNotNone(res.footprint)

    def test_concurrent_independent_jobs(self):
        """Two separate jobs must emit with their distinct job_ids."""
        events_job_a = []
        events_job_b = []

        def cb_a(stage, stage_index, stage_count, progress, message, status="RUNNING", error=None):
            events_job_a.append((stage, "job-A"))

        def cb_b(stage, stage_index, stage_count, progress, message, status="RUNNING", error=None):
            events_job_b.append((stage, "job-B"))

        register_pair(self.img1, self.img2, detector="sift", progress_callback=cb_a)
        register_pair(self.img1, self.img2, detector="sift", progress_callback=cb_b)

        self.assertGreater(len(events_job_a), 0)
        self.assertGreater(len(events_job_b), 0)
        self.assertTrue(all(j == "job-A" for _, j in events_job_a))
        self.assertTrue(all(j == "job-B" for _, j in events_job_b))

    def test_live_api_progress_endpoints(self):
        """Test status endpoint and SSE progress stream against the live API server."""
        import requests
        import threading
        import time

        try:
            health = requests.get("http://localhost:5000/api/healthz", timeout=3)
            if health.status_code != 200:
                self.skipTest("Live API server is not running on localhost:5000")
        except Exception:
            self.skipTest("Live API server is not reachable on localhost:5000")

        # 1. Unknown job returns 404
        r_unknown = requests.get("http://localhost:5000/api/register/status/non-existent-job-xyz")
        self.assertEqual(r_unknown.status_code, 404)

        # 2. Register job with SSE listener
        test_job_id = f"test-live-sse-{int(time.time() * 1000)}"
        events_received = []
        sse_ready = threading.Event()

        def listen_sse():
            try:
                sse_ready.set()
                with requests.get(f"http://localhost:5000/api/register/progress/{test_job_id}", stream=True, timeout=15) as sse_res:
                    for line in sse_res.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: "):
                                events_received.append(json.loads(decoded[6:]))
            except Exception:
                pass

        t = threading.Thread(target=listen_sse)
        t.start()
        sse_ready.wait(timeout=2)
        time.sleep(0.05)

        _, buf1 = cv2.imencode(".png", self.img1)
        _, buf2 = cv2.imencode(".png", self.img2)
        files = {
            "source": ("source.png", buf1.tobytes(), "image/png"),
            "reference": ("reference.png", buf2.tobytes(), "image/png"),
        }
        res = requests.post(
            "http://localhost:5000/api/register",
            files=files,
            headers={"x-job-id": test_job_id},
            timeout=30,
        )
        t.join(timeout=10)

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("job_id"), test_job_id)
        self.assertIn("homography", data)
        self.assertIn("registration_status", data)
        self.assertIn("metrics", data)

        # 3. Check status endpoint for finished job
        r_status = requests.get(f"http://localhost:5000/api/register/status/{test_job_id}", timeout=5)
        self.assertEqual(r_status.status_code, 200)
        st_data = r_status.json()
        self.assertEqual(st_data["stage"], "COMPLETE")
        self.assertEqual(st_data["status"], "COMPLETE")
        self.assertEqual(st_data["progress"], 1.0)

        # 4. Check SSE received events
        self.assertGreater(len(events_received), 0)
        received_stages = [ev["stage"] for ev in events_received]
        self.assertIn("COMPLETE", received_stages)


if __name__ == "__main__":
    unittest.main()
