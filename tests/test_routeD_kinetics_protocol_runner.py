from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_routeD_kinetics_protocol import validate_result


class TestRouteDKineticsProtocolRunner(unittest.TestCase):
    def test_validate_result_requires_metric_contract_when_frozen(self):
        payload = {
            "samples": 2,
            "per_sample": [{"sample": 0}, {"sample": 1}],
            "metric_implementation_sha256": "metric-hash",
            "metric_coordinate_contract": "x * width, y * height",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(payload))
            self.assertTrue(
                validate_result(
                    path,
                    2,
                    expected_metric_hash="metric-hash",
                    expected_metric_contract="x * width, y * height",
                )
            )
            self.assertFalse(
                validate_result(
                    path,
                    2,
                    expected_metric_hash="old-metric-hash",
                    expected_metric_contract="x * width, y * height",
                )
            )
            self.assertFalse(
                validate_result(
                    path,
                    2,
                    expected_metric_hash="metric-hash",
                    expected_metric_contract="x * (width - 1), y * (height - 1)",
                )
            )

    def test_validate_result_rejects_partial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps({"samples": 2, "per_sample": [{}]}))
            self.assertFalse(validate_result(path, 2))


if __name__ == "__main__":
    unittest.main()
