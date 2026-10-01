import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, "backend")

from app.services.evaluation import run_synthetic_robustness


class PhaseFourEvaluationTests(unittest.TestCase):
    def test_synthetic_evaluation_returns_all_controlled_cases(self):
        image = np.zeros((220, 220), dtype=np.uint8)
        cv2.circle(image, (70, 70), 25, 255, -1)
        cv2.rectangle(image, (120, 120), (190, 180), 180, -1)
        cv2.line(image, (20, 200), (200, 20), 220, 2)
        result = run_synthetic_robustness(image, {"ecc_refinement": False, "max_features": 1500})
        self.assertEqual(result["total_cases"], 7)
        self.assertEqual(len(result["results"]), 7)
        self.assertEqual(result["evaluation_type"], "synthetic_robustness_stress_test")
        self.assertIn("scientific_note", result)
        self.assertTrue(all("scenario" in row for row in result["results"]))


if __name__ == "__main__":
    unittest.main()
