import unittest

import torch

from projects.mmp_tracker.mmp_tracker.calibration_metrics import calibration_report
from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    apply_scalar_variance_calibration,
    fit_scalar_variance_calibration,
)
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

    def test_shared_variance_scale_has_closed_form_solution(self):
        errors = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        variance = torch.ones_like(errors)
        state = fit_scalar_variance_calibration(errors, variance)
        self.assertAlmostEqual(state["scale"], 2.0, places=6)
        calibrated = apply_scalar_variance_calibration(variance, state)
        self.assertTrue(torch.allclose(calibrated, torch.full_like(variance, 2.0)))
        self.assertLessEqual(
            state["calibration_nll_after"],
            state["calibration_nll_before"],
        )

    def test_variance_scale_respects_preregistered_bounds(self):
        errors = torch.full((4, 2), 1e6)
        variance = torch.full((4, 2), 1e-8)
        state = fit_scalar_variance_calibration(errors, variance)
        calibrated = apply_scalar_variance_calibration(variance, state)
        self.assertGreaterEqual(float(calibrated.min()), 0.25**2)
        self.assertLessEqual(float(calibrated.max()), 256.0**2)


if __name__ == "__main__":
    unittest.main()
