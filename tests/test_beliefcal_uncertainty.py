import unittest

import torch

from projects.mmp_tracker.mmp_tracker.calibration_metrics import calibration_report
from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    apply_scalar_variance_calibration,
    baseline_support_diagnostics,
    fit_scalar_variance_calibration,
    predict_learned_variance,
    resolve_uncertainty_feature_selection,
    synthetic_beliefcal_cache,
    train_uncertainty_head,
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

    def test_apply_scale_clamps_before_scaling(self):
        variance = torch.full((2, 2), 1e-8)
        state = {
            "scale": 4.0,
            "min_std_px": 0.25,
            "max_std_px": 256.0,
        }
        calibrated = apply_scalar_variance_calibration(variance, state)
        self.assertTrue(
            torch.allclose(calibrated, torch.full_like(variance, 4.0 * 0.25**2))
        )

    def test_support_diagnostics_detect_degenerate_groups(self):
        cache = synthetic_beliefcal_cache(64, seed=5)
        cache["pred_visibility"] = torch.ones(64)
        cache["predicted_occluded_duration"] = torch.zeros(64, dtype=torch.long)
        diagnostics = baseline_support_diagnostics(cache)
        self.assertEqual(diagnostics["visibility_group_counts"], {"1": 64})
        self.assertEqual(diagnostics["duration_group_counts"], {"0": 64})
        self.assertEqual(len(diagnostics["warnings"]), 2)

    def test_drop_inert_feature_profile_is_fixed_and_auditable(self):
        indices, names = resolve_uncertainty_feature_selection("drop_inert_mmp")
        self.assertEqual(len(indices), 20)
        self.assertNotIn("selected_global", names)
        self.assertNotIn("active", names)
        self.assertNotIn("predicted_occluded_duration", names)
        self.assertNotIn("selected_global_available", names)

    def test_pruned_state_predicts_from_full_cache(self):
        train = synthetic_beliefcal_cache(128, seed=11)
        validation = synthetic_beliefcal_cache(64, seed=12, sample_offset=1000)
        state = train_uncertainty_head(
            train,
            validation,
            seed=17,
            hidden_dim=8,
            epochs=1,
            batch_size=64,
            feature_profile="drop_inert_mmp",
        )
        variance = predict_learned_variance(validation, state)
        self.assertEqual(tuple(variance.shape), (64, 2))
        self.assertEqual(state["input_dim"], 20)

    def test_variance_scale_respects_preregistered_bounds(self):
        errors = torch.full((4, 2), 1e6)
        variance = torch.full((4, 2), 1e-8)
        state = fit_scalar_variance_calibration(errors, variance)
        calibrated = apply_scalar_variance_calibration(variance, state)
        self.assertGreaterEqual(float(calibrated.min()), 0.25**2)
        self.assertLessEqual(float(calibrated.max()), 256.0**2)


if __name__ == "__main__":
    unittest.main()
