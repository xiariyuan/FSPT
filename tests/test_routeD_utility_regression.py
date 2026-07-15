import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_utility_regression import (
    RouteDUtilityRegressor,
    utility_regression_loss,
)


class RouteDUtilityRegressionTest(unittest.TestCase):
    def test_model_output_shape(self):
        model = RouteDUtilityRegressor(7, hidden_dim=12)
        output = model(torch.randn(5, 7))
        self.assertEqual(output.shape, (5, 2))

    def test_loss_is_finite_and_prefers_exact_prediction(self):
        utility = torch.tensor([-0.4, 0.0, 0.2, 0.6])
        p1 = torch.tensor([-1.0, 0.0, 0.0, 1.0])
        exact = torch.stack([utility, p1], dim=-1)
        wrong = torch.zeros_like(exact)
        exact_loss = utility_regression_loss(exact, utility, p1)
        wrong_loss = utility_regression_loss(wrong, utility, p1)
        self.assertTrue(torch.isfinite(exact_loss))
        self.assertLess(float(exact_loss), float(wrong_loss))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            utility_regression_loss(
                torch.zeros(3, 1), torch.zeros(3), torch.zeros(3)
            )


if __name__ == "__main__":
    unittest.main()
