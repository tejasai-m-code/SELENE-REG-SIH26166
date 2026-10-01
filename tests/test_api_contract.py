import sys
import unittest

sys.path.insert(0, "backend")

import serve


class ApiContractTests(unittest.TestCase):
    def test_existing_pair_and_new_multi_routes_are_registered(self):
        paths = set(serve.app.openapi()["paths"])
        self.assertIn("/api/register", paths)
        self.assertIn("/api/multi-registration/register", paths)
        self.assertIn("/api/multi-registration/{job_id}/graph", paths)
        self.assertIn("/api/multi-registration/{job_id}/export/metrics.csv", paths)


if __name__ == "__main__":
    unittest.main()
