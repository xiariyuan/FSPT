import unittest

import torch

from projects.mmp_tracker.mmp_tracker.multi_hypothesis_belief import (
    belief_from_logits,
    belief_from_probabilities,
    conservative_collapse,
    diagnose_belief,
)


class TestMultiHypothesisBelief(unittest.TestCase):
    def test_probability_normalization_respects_invalid_candidates(self):
        points = torch.tensor([[[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]]])
        probabilities = torch.tensor([[2.0, 8.0, 100.0]])
        valid = torch.tensor([[True, True, False]])
        belief = belief_from_probabilities(points, probabilities, valid)
        self.assertTrue(torch.allclose(belief.weights, torch.tensor([[0.2, 0.8, 0.0]])))

    def test_empty_rows_are_well_defined(self):
        points = torch.zeros(2, 3, 2)
        logits = torch.zeros(2, 3)
        valid = torch.tensor([[False, False, False], [True, False, False]])
        belief = belief_from_logits(points, logits, valid)
        diagnostics = diagnose_belief(belief)
        self.assertEqual(float(belief.weights[0].sum()), 0.0)
        self.assertEqual(float(diagnostics.effective_hypotheses[0]), 0.0)
        self.assertEqual(float(belief.weights[1, 0]), 1.0)

    def test_multimodal_belief_does_not_collapse(self):
        points = torch.tensor([[[0.1, 0.1], [0.9, 0.9]]])
        belief = belief_from_probabilities(points, torch.tensor([[0.5, 0.5]]))
        decision = conservative_collapse(belief)
        self.assertFalse(bool(decision.collapse_mask.item()))
        self.assertAlmostEqual(float(decision.diagnostics.effective_hypotheses.item()), 2.0, places=5)

    def test_concentrated_belief_collapses_to_map_not_mean(self):
        points = torch.tensor([[[0.1, 0.1], [0.9, 0.9]]])
        belief = belief_from_probabilities(points, torch.tensor([[0.95, 0.05]]))
        decision = conservative_collapse(belief)
        self.assertTrue(bool(decision.collapse_mask.item()))
        self.assertTrue(torch.allclose(decision.selected_points, torch.tensor([[0.1, 0.1]])))
        self.assertFalse(
            torch.allclose(
                decision.selected_points,
                decision.diagnostics.expected_points,
            )
        )


if __name__ == "__main__":
    unittest.main()
