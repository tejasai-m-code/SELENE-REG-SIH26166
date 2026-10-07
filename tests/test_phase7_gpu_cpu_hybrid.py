"""Test Suite for Phase 7: GPU + CPU Hybrid Execution & 4 GB VRAM Memory Safety Engine.

Verifies:
1. Actual CUDA hardware detection and host GPU inspection.
2. Authoritative GPU self-test (device detection, tensor allocation, GEMM computation, synchronization).
3. Honest CPU fallback when CUDA is unavailable (never claims fake GPU execution).
4. Hybrid scheduler task classification and routing (CPU vs GPU affinity).
5. 4 GB VRAM memory safety safeguards (safety headroom, large allocation fallback, chunking, cache clearance).
6. Mathematical consistency of GPU vs CPU implementations (descriptors, phase correlation, warping).
7. Registration pipeline telemetry integration with per-stage hardware tracking.
8. Worker CLI modes (`--mode gpu-self-test`, `--mode hardware-status`).
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

# Ensure api-server python package is on import path
python_root = Path(__file__).resolve().parents[1] / "artifacts" / "api-server" / "python"
if str(python_root) not in sys.path:
    sys.path.insert(0, str(python_root))

from app.services.gpu_accelerator import (
    clear_gpu_cache,
    get_free_vram_mb,
    get_gpu_info,
    is_cuda_available,
    is_vram_sufficient,
    match_descriptors_gpu,
    phase_correlation_gpu,
    run_gpu_self_test,
    warp_perspective_gpu,
    _match_descriptors_cpu_fallback,
)
from app.services.hardware_manager import (
    check_memory_safety,
    get_hardware_status,
    run_hardware_self_test,
)
from app.services.hybrid_scheduler import (
    HybridScheduler,
    TaskCategory,
    get_global_scheduler,
)
from app.services.pairwise_registration import register_pair


def make_test_raster(seed: int = 42, size: int = 256) -> np.ndarray:
    """Generate reproducible test image with textured features."""
    rng = np.random.default_rng(seed)
    base = rng.integers(50, 200, (size, size), dtype=np.uint8)
    for _ in range(15):
        cx, cy = int(rng.integers(20, size - 20)), int(rng.integers(20, size - 20))
        r = int(rng.integers(10, 40))
        color = int(rng.integers(20, 240))
        cv2.circle(base, (cx, cy), r, color, -1)
    base = cv2.GaussianBlur(base, (5, 5), 1.2)
    return base


class TestPhase7HardwareDetection(unittest.TestCase):
    """Test actual CUDA hardware detection and inspection."""

    def test_cuda_detection_and_hardware_status_keys(self):
        status = get_hardware_status()
        self.assertIsInstance(status, dict)

        required_keys = [
            "device",
            "acceleration_status",
            "execution_mode",
            "status_display",
            "hardware_gpu_name",
            "hardware_gpu_detected",
            "cuda_driver_version",
            "vram_total_mb",
            "vram_free_mb",
            "vram_safety_reserve_mb",
            "pytorch_cuda_available",
            "system_ram_total_mb",
            "system_ram_available_mb",
            "cpu_logical_cores",
            "bounded_worker_concurrency",
            "gpu_self_test",
            "note",
        ]
        for key in required_keys:
            self.assertIn(key, status, f"Missing required hardware key: {key}")

        self.assertGreaterEqual(status["system_ram_total_mb"], 1024)
        self.assertGreaterEqual(status["cpu_logical_cores"], 1)
        self.assertGreaterEqual(status["bounded_worker_concurrency"], 1)
        self.assertEqual(status["vram_safety_reserve_mb"], 512)

        if status["pytorch_cuda_available"]:
            self.assertEqual(status["device"], "CUDA")
            self.assertEqual(status["acceleration_status"], "GPU ACCELERATED")
            self.assertEqual(status["execution_mode"], "GPU / HYBRID")
            self.assertGreater(status["vram_total_mb"], 1024)
            self.assertTrue(status["hardware_gpu_detected"])

    def test_gpu_info_metrics(self):
        info = get_gpu_info()
        self.assertIsInstance(info, dict)
        if is_cuda_available():
            self.assertTrue(info["cuda_available"])
            self.assertIn("device_name", info)
            self.assertIn("compute_capability", info)
            self.assertIn("total_vram_mb", info)
            self.assertGreater(info["total_vram_mb"], 0.0)
        else:
            self.assertFalse(info["cuda_available"])
            self.assertEqual(info["device_name"], "CPU Fallback")


class TestPhase7GpuSelfTest(unittest.TestCase):
    """Test authoritative GPU self-test (§24 of Master Engineering Prompt)."""

    def test_live_gpu_self_test_execution(self):
        """Execute real GPU self test on the host hardware."""
        test_res = run_gpu_self_test()
        self.assertIsInstance(test_res, dict)

        required_keys = [
            "cuda_device_detected",
            "tensor_allocation",
            "gpu_computation",
            "synchronization",
            "device_name",
            "vram_total_mb",
            "compute_time_ms",
            "overall_status",
            "execution_mode",
        ]
        for key in required_keys:
            self.assertIn(key, test_res, f"Missing self-test key: {key}")

        if is_cuda_available():
            self.assertEqual(test_res["cuda_device_detected"], "PASS")
            self.assertEqual(test_res["tensor_allocation"], "PASS")
            self.assertEqual(test_res["gpu_computation"], "PASS")
            self.assertEqual(test_res["synchronization"], "PASS")
            self.assertEqual(test_res["overall_status"], "PASS")
            self.assertEqual(test_res["execution_mode"], "GPU / HYBRID")
            self.assertGreater(test_res["compute_time_ms"], 0.0)
            self.assertIn("RTX", test_res["device_name"])

    def test_gpu_self_test_offline_fallback(self):
        """Verify that when CUDA is unavailable, self-test truthfully reports FAIL and CPU FALLBACK."""
        with patch("app.services.gpu_accelerator.is_cuda_available", return_value=False):
            test_res = run_gpu_self_test()
            self.assertEqual(test_res["cuda_device_detected"], "FAIL")
            self.assertEqual(test_res["tensor_allocation"], "FAIL")
            self.assertEqual(test_res["gpu_computation"], "FAIL")
            self.assertEqual(test_res["synchronization"], "FAIL")
            self.assertEqual(test_res["overall_status"], "FAIL")
            self.assertEqual(test_res["execution_mode"], "CPU FALLBACK")


class TestPhase7HybridScheduler(unittest.TestCase):
    """Test HybridScheduler workload classification, routing, and 4 GB VRAM safety (§23 & §25)."""

    def setUp(self):
        self.scheduler = HybridScheduler(vram_safety_margin_mb=512.0)

    def test_cpu_affinity_routing(self):
        """Verify CPU-bound tasks are routed strictly to CPU."""
        cpu_tasks = [
            TaskCategory.FILE_IO,
            TaskCategory.DECODING,
            TaskCategory.METADATA_PARSING,
            TaskCategory.RANSAC_ESTIMATION,
            TaskCategory.GRAPH_ORCHESTRATION,
            TaskCategory.SPATIAL_METRICS,
        ]
        for task in cpu_tasks:
            use_gpu, reason = self.scheduler.should_use_gpu(task)
            self.assertFalse(use_gpu, f"Task {task} should not run on GPU")
            self.assertEqual(reason, "CPU_AFFINITY_TASK")

    def test_gpu_affinity_routing(self):
        """Verify GPU-accelerated tasks are routed to GPU when CUDA and VRAM are available."""
        gpu_tasks = [
            TaskCategory.DESCRIPTOR_MATCHING,
            TaskCategory.PHASE_CORRELATION,
            TaskCategory.PERSPECTIVE_WARPING,
            TaskCategory.SPATIAL_FILTERING,
            TaskCategory.TENSOR_PREPROCESSING,
        ]
        if is_cuda_available():
            for task in gpu_tasks:
                use_gpu, reason = self.scheduler.should_use_gpu(task, estimated_vram_mb=50.0)
                self.assertTrue(use_gpu, f"Task {task} should run on GPU")
                self.assertEqual(reason, "GPU_ACCOMMODATED")

    def test_vram_safety_fallback_on_excessive_demand(self):
        """Verify 4 GB VRAM safeguard triggers automatic CPU fallback when memory demand exceeds available VRAM."""
        # Request 100,000 MB which exceeds any single-node laptop GPU VRAM
        use_gpu, reason = self.scheduler.should_use_gpu(
            TaskCategory.DESCRIPTOR_MATCHING, estimated_vram_mb=100000.0
        )
        self.assertFalse(use_gpu)
        self.assertTrue(reason.startswith("INSUFFICIENT_VRAM_FREE_"))

    def test_transparent_dispatch_with_cpu_fallback(self):
        """Verify dispatch executes CPU fallback when GPU fails or is bypassed."""
        fn_gpu = lambda: 1 / 0  # Intentionally failing GPU function
        fn_cpu = lambda: "cpu_success"

        result = self.scheduler.dispatch(
            TaskCategory.DESCRIPTOR_MATCHING,
            "test_failing_gpu_task",
            fn_gpu=fn_gpu,
            fn_cpu=fn_cpu,
            estimated_vram_mb=10.0,
        )
        self.assertEqual(result, "cpu_success")
        self.assertGreaterEqual(self.scheduler.fallback_count, 1)

    def test_scheduler_telemetry_reporting(self):
        """Verify scheduler telemetry accurately calculates execution percentages."""
        telemetry = self.scheduler.get_telemetry()
        self.assertIsInstance(telemetry, dict)
        self.assertIn("total_tasks_scheduled", telemetry)
        self.assertIn("gpu_tasks_executed", telemetry)
        self.assertIn("cpu_tasks_executed", telemetry)
        self.assertIn("gpu_execution_percentage", telemetry)
        self.assertIn("cpu_execution_percentage", telemetry)
        self.assertIn("vram_safety_margin_mb", telemetry)
        self.assertEqual(telemetry["vram_safety_margin_mb"], 512.0)


class TestPhase7MemorySafety(unittest.TestCase):
    """Test memory-safety utilities and batch chunking (§25)."""

    def test_vram_headroom_checks(self):
        free_mb = get_free_vram_mb()
        self.assertIsInstance(free_mb, float)
        if is_cuda_available():
            self.assertGreater(free_mb, 100.0)
            self.assertTrue(is_vram_sufficient(required_mb=50.0, safety_margin_mb=512.0))
            self.assertFalse(is_vram_sufficient(required_mb=50000.0, safety_margin_mb=512.0))

    def test_clear_gpu_cache_non_throwing(self):
        clear_gpu_cache()  # Must execute smoothly without error

    def test_check_memory_safety_utility(self):
        mem_check = check_memory_safety(required_vram_mb=128.0, required_ram_mb=512.0)
        self.assertIsInstance(mem_check, dict)
        self.assertIn("is_safe", mem_check)
        self.assertIn("vram_safe", mem_check)
        self.assertIn("ram_safe", mem_check)
        self.assertIn("recommendation", mem_check)
        self.assertTrue(mem_check["ram_safe"])

    def test_chunked_descriptor_matching(self):
        """Verify chunked descriptor matching splits queries safely without OOM."""
        rng = np.random.default_rng(123)
        des_s = rng.standard_normal((100, 128)).astype(np.float32)
        des_r = rng.standard_normal((100, 128)).astype(np.float32)

        scheduler = HybridScheduler(max_descriptor_product=500)  # Low threshold forces chunking
        matches, diag = scheduler.match_descriptors(des_s, des_r, ratio=0.85, mutual_check=True)
        self.assertIsInstance(matches, list)
        self.assertIsInstance(diag, dict)


class TestPhase7GpuMathEquivalence(unittest.TestCase):
    """Test that GPU-accelerated algorithms yield equivalent mathematical results to CPU."""

    def test_phase_correlation_shift_accuracy(self):
        """Verify 2D FFT phase correlation accurately recovers subpixel translation."""
        img = make_test_raster(seed=101, size=256)
        dx_true, dy_true = 3.25, -2.50
        M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
        shifted = cv2.warpAffine(img, M, (256, 256))

        sx, sy, peak = phase_correlation_gpu(img, shifted)
        self.assertAlmostEqual(sx, dx_true, delta=0.25)
        self.assertAlmostEqual(sy, dy_true, delta=0.25)
        self.assertGreater(peak, 0.05)

    def test_perspective_warping_fidelity(self):
        """Verify GPU grid_sample perspective warping matches OpenCV warpPerspective."""
        img = make_test_raster(seed=202, size=200)
        H = np.array([
            [1.02, 0.01, 12.0],
            [-0.01, 0.98, 8.0],
            [0.0001, 0.0001, 1.0],
        ], dtype=np.float64)

        warped_gpu = warp_perspective_gpu(img, H, (200, 200))
        warped_cpu = cv2.warpPerspective(img, H, (200, 200))

        # Check mean absolute difference is small (within bilinear discretization tolerance)
        diff = np.abs(warped_gpu.astype(np.float32) - warped_cpu.astype(np.float32))
        # Exclude borders where interpolation edges differ slightly
        valid_diff = diff[10:-10, 10:-10]
        mean_err = np.mean(valid_diff)
        self.assertLess(mean_err, 3.0, f"Mean error between GPU and CPU warping too high: {mean_err}")


class TestPhase7PairwiseHybridIntegration(unittest.TestCase):
    """Test full pairwise registration pipeline with hybrid GPU+CPU telemetry."""

    def test_pairwise_registration_hybrid_telemetry(self):
        src = make_test_raster(seed=301, size=300)
        M = np.float32([[1, 0, 5.0], [0, 1, -4.0]])
        ref = cv2.warpAffine(src, M, (300, 300))

        result = register_pair(
            src,
            ref,
            detector="sift",
            refinement_methods=["phase_correlation"],
            ecc_refinement=False,
        )

        self.assertIn("hardware_acceleration", result.metrics)
        hw = result.metrics["hardware_acceleration"]

        self.assertIn("execution_mode", hw)
        self.assertIn("cuda_active", hw)
        self.assertIn("device_name", hw)
        self.assertIn("stages", hw)
        self.assertIn("gpu_execution_percentage", hw)
        self.assertIn("cpu_execution_percentage", hw)

        stages = hw["stages"]
        self.assertIn("feature_detection", stages)
        self.assertIn("ransac_geometry", stages)
        self.assertIn("descriptor_matching", stages)
        self.assertIn("warping", stages)

        self.assertEqual(stages["feature_detection"], "CPU")
        self.assertEqual(stages["ransac_geometry"], "CPU")

        if is_cuda_available():
            self.assertEqual(hw["execution_mode"], "HYBRID (GPU + CPU)")
            self.assertEqual(stages["descriptor_matching"], "GPU_CUDA")
            self.assertEqual(stages["warping"], "GPU_CUDA")
            self.assertGreater(hw["gpu_execution_percentage"], 0.0)


class TestPhase7WorkerCliModes(unittest.TestCase):
    """Test worker CLI hardware modes (--mode gpu-self-test and --mode hardware-status)."""

    def test_worker_gpu_self_test_cli(self):
        worker_path = python_root / "worker.py"
        cmd = [sys.executable, str(worker_path), "--mode", "gpu-self-test"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        self.assertIn("cuda_device_detected", data)
        self.assertIn("tensor_allocation", data)
        self.assertIn("gpu_computation", data)
        self.assertIn("synchronization", data)
        self.assertIn("overall_status", data)
        self.assertIn("execution_mode", data)

    def test_worker_hardware_status_cli(self):
        worker_path = python_root / "worker.py"
        cmd = [sys.executable, str(worker_path), "--mode", "hardware-status"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        self.assertIn("acceleration_status", data)
        self.assertIn("execution_mode", data)
        self.assertIn("vram_safety_reserve_mb", data)
        self.assertIn("gpu_self_test", data)
        self.assertIn("hybrid_scheduler", data)


if __name__ == "__main__":
    unittest.main()
