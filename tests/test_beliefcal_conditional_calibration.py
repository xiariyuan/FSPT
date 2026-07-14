import unittest

import torch

from projects.mmp_tracker.mmp_tracker.calibration_metrics import gaussian_nll
from projects.mmp_tracker.mmp_tracker.conditional_calibration import (
    apply_conditional_shared_scale_calibration,
    build_conditional_calibration_inputs,
    fit_conditional_shared_scale_calibration,
    select_conditional_candidate,
)
from projects.mmp_tracker.mmp_tracker.beliefcal_runner import synthetic_beliefcal_cache


class TestBeliefCalConditionalCalibration(unittest.TestCase):
    def test_fixed_input_contract_is_28d(self):
        cache = synthetic_beliefcal_cache(64, seed=1)
        raw_variance = torch.ones(64, 2)
        inputs = build_conditional_calibration_inputs(cache, raw_variance)
        self.assertEqual(tuple(inputs.shape), (64, 28))

    def test_conditional_affine_fits_feature_dependent_scale(self):
        count = 512
        cache = synthetic_beliefcal_cache(count, seed=2)
        driver = torch.linspace(-1.0, 1.0, count)
        cache["features"].zero_()
        cache["features"][:, 0] = driver
        cache["features"][:, 13:] = 1.0
        true_variance = torch.exp(1.5 * driver)
        cache["errors_px"] = true_variance.sqrt().unsqueeze(-1).expand(-1, 2).clone()
        raw_variance = torch.ones(count, 2)

        state = fit_conditional_shared_scale_calibration(
            cache,
            raw_variance,
            initial_scale=1.0,
            seed=17,
            steps=250,
            learning_rate=0.02,
        )
        calibrated = apply_conditional_shared_scale_calibration(
            cache, raw_variance, state
        )
        scalar_nll = gaussian_nll(cache["errors_px"], raw_variance).mean()
        conditional_nll = gaussian_nll(cache["errors_px"], calibrated).mean()
        self.assertLess(float(conditional_nll), float(scalar_nll) - 0.05)
        self.assertLessEqual(state["best_calibration_nll"], state["initial_calibration_nll"])

    def test_application_respects_variance_bounds(self):
        cache = synthetic_beliefcal_cache(64, seed=3)
        raw_variance = torch.full((64, 2), 1e-12)
        state = fit_conditional_shared_scale_calibration(
            cache,
            raw_variance,
            initial_scale=64.0,
            seed=17,
            steps=2,
        )
        calibrated = apply_conditional_shared_scale_calibration(
            cache, raw_variance, state
        )
        self.assertGreaterEqual(float(calibrated.min()), 0.25**2)
        self.assertLessEqual(float(calibrated.max()), 256.0**2)

    def test_selection_rule_requires_one_percent(self):
        rejected = select_conditional_candidate(10.0, 9.95)
        accepted = select_conditional_candidate(10.0, 9.89)
        self.assertEqual(rejected["selected"], "shared_scalar")
        self.assertEqual(accepted["selected"], "conditional_affine")


if __name__ == "__main__":
    unittest.main()
