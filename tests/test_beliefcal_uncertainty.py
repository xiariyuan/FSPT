import unittest

import torch

from projects.mmp_tracker.mmp_tracker.calibration_metrics import calibration_report
from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    DiagonalGaussianUncertaintyHead,
)


class TestBeliefCalUncertainty(unittest.TestCase):
    def test_head_only_predicts_variance(self):
        head = DiagonalGaussianUncertaintyHead(9)
        out = head(torch.zeros(2, 3, 9))
        self.assertEqual(tuple(out["variance"].shape), (2, 3, 2))
        self.assertTrue(torch.isfinite(out["variance"]).all())

    def test_coverage_report(self):
        errors = torch.zeros(100, 2)
        variance = torch.ones(100, 2)
        report = calibration_report(errors, variance)
        self.assertAlmostEqual(report["coverage_95"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
