"""Unit tests for hardware detection, bounded concurrency, and honest acceleration reporting."""

from __future__ import annotations

import unittest
from app.services.hardware_manager import (
    get_hardware_status,
    measure_execution_benchmark,
)


class TestHardwareManager(unittest.TestCase):
    def test_hardware_status_contract(self):
        """Hardware status must return complete fields without error."""
        status = get_hardware_status()
        self.assertIsInstance(status, dict)

        required_keys = [
            "device",
            "acceleration_status",
            "status_display",
            "hardware_gpu_name",
            "cuda_driver_version",
            "vram_total_mb",
            "opencv_cuda_devices",
            "pytorch_cuda_available",
            "bounded_worker_concurrency",
            "system_ram_total_mb",
            "system_ram_available_mb",
            "cpu_logical_cores",
        ]
        for key in required_keys:
            self.assertIn(key, status, f"Missing required hardware key: {key}")

    def test_honesty_principle_no_fake_cuda(self):
        """If PyTorch and OpenCV CUDA builds are not active, status must report CPU Fallback."""
        status = get_hardware_status()
        if not status["pytorch_cuda_available"] and status["opencv_cuda_devices"] == 0:
            self.assertEqual(status["device"], "CPU")
            self.assertEqual(status["acceleration_status"], "CPU FALLBACK")
            self.assertIn("CPU Fallback", status["status_display"])
            # Host GPU name may be detected via nvidia-smi, but active execution must remain CPU
            if status["hardware_gpu_name"]:
                self.assertIn(status["hardware_gpu_name"], status["status_display"])

    def test_bounded_concurrency(self):
        """Worker concurrency must be bounded to prevent OOM on large lunar rasters."""
        status = get_hardware_status()
        workers = status["bounded_worker_concurrency"]

        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, status["cpu_logical_cores"])

    def test_benchmark_reporting_no_fake_speedup(self):
        """Benchmark reporting must only calculate speedup when both CPU and GPU were measured."""
        # Case 1: CPU only measured -> no speedup claimed
        cpu_only = measure_execution_benchmark(cpu_seconds=2.5, gpu_seconds=None)
        self.assertIsNone(cpu_only["measured_speedup_factor"])
        self.assertEqual(cpu_only["status"], "CPU ONLY MEASURED")

        # Case 2: Both measured -> real speedup ratio
        both = measure_execution_benchmark(cpu_seconds=4.0, gpu_seconds=1.0)
        self.assertEqual(both["measured_speedup_factor"], 4.0)
        self.assertEqual(both["status"], "ACCELERATION MEASURED")


if __name__ == "__main__":
    unittest.main()
