import unittest

import torch

from scripts.analyze_beliefcal_prediction_diagnostics import (
    equal_count_bin_indices,
    risk_decile_table,
)


class TestBeliefCalPredictionDiagnostics(unittest.TestCase):
    def test_equal_count_bins_cover_each_row_once(self):
        score = torch.tensor([3.0, 1.0, 2.0, 6.0, 5.0, 4.0])
        bins = equal_count_bin_indices(score, bins=3)
        joined = torch.cat(bins)
        self.assertEqual(len(bins), 3)
        self.assertEqual(sorted(joined.tolist()), list(range(6)))
        self.assertEqual([len(index) for index in bins], [2, 2, 2])

    def test_risk_deciles_follow_uncertainty_order(self):
        errors = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        variance = torch.tensor([[1.0, 1.0], [4.0, 4.0], [9.0, 9.0], [16.0, 16.0]])
        rows = risk_decile_table(errors, variance, bins=2)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]["score_mean"], rows[1]["score_mean"])
        self.assertLess(rows[0]["error_mean_px"], rows[1]["error_mean_px"])


if __name__ == "__main__":
    unittest.main()
