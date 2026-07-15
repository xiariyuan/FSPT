import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_utility_regression import (
    build_utility_targets,
    conformal_lower_bound,
    conformal_upper_residual_quantile,
)


class RouteDUtilityTargetTest(unittest.TestCase):
    def test_exact_utility_and_p1_targets(self):
        examples = {
            "best_global_index": torch.tensor([1, 2, 1]),
            "true_utility": torch.tensor(
                [
                    [0.2, 0.8, 0.0],
                    [0.6, 0.4, 0.8],
                    [0.4, 0.2, 0.0],
                ]
            ),
            "errors": torch.tensor(
                [
                    [2.0, 0.5, 4.0],
                    [0.5, 2.0, 0.8],
                    [0.7, 1.5, 3.0],
                ]
            ),
        }
        utility, p1 = build_utility_targets(examples)
        self.assertTrue(torch.allclose(utility, torch.tensor([0.6, 0.2, -0.2])))
        self.assertTrue(torch.equal(p1, torch.tensor([1.0, 0.0, -1.0])))

    def test_conformal_lower_bound_uses_upper_residual(self):
        prediction = torch.tensor([0.2, 0.4, 0.6, 0.8])
        target = torch.tensor([0.1, 0.5, 0.2, 0.7])
        quantile = conformal_upper_residual_quantile(
            prediction, target, alpha=0.25
        )
        lower = conformal_lower_bound(prediction, quantile)
        self.assertTrue(torch.isfinite(lower).all())
        self.assertEqual(lower.shape, prediction.shape)
        self.assertGreaterEqual(quantile, 0.1)

    def test_target_shape_validation(self):
        with self.assertRaises(ValueError):
            build_utility_targets(
                {
                    "best_global_index": torch.tensor([1]),
                    "true_utility": torch.zeros(1, 2),
                    "errors": torch.zeros(1, 3),
                }
            )


if __name__ == "__main__":
    unittest.main()
