import tempfile
import unittest
from pathlib import Path

import torch

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import MultiThresholdHypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_multithreshold_training import (
    MultiThresholdTrainingConfig,
    aggregate_threshold_utility,
    multithreshold_training_loss,
    select_multithreshold_candidate,
    threshold_hit_targets,
)
from projects.mmp_tracker.mmp_tracker.routeD_selector import RouteDRiskSelector


class TestRouteDMultiThresholdTraining(unittest.TestCase):
    def test_model_and_targets_shape(self):
        model = MultiThresholdHypothesisScorer(12, 16, 5)
        output = model(torch.randn(2, 3, 12))
        self.assertEqual(output.shape, (2, 3, 5))
        targets = threshold_hit_targets(
            torch.tensor([[0.5, 3.0, 20.0]]), (1, 2, 4, 8, 16)
        )
        self.assertTrue(torch.equal(targets[0, 0], torch.ones(5)))
        self.assertTrue(torch.equal(targets[0, 2], torch.zeros(5)))

    def test_loss_prefers_correct_threshold_profile(self):
        errors = torch.tensor([[0.5, 20.0]])
        valid = torch.ones_like(errors, dtype=torch.bool)
        config = MultiThresholdTrainingConfig()
        good = torch.tensor([[[5.0] * 5, [-5.0] * 5]])
        bad = -good
        self.assertLess(
            float(multithreshold_training_loss(good, errors, valid, config)),
            float(multithreshold_training_loss(bad, errors, valid, config)),
        )

    def test_selector_uses_mean_threshold_utility(self):
        logits = torch.tensor([[
            [-4.0, -4.0, -4.0, -4.0, -4.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        index, probability, _, utility = select_multithreshold_candidate(
            logits, valid, gate_threshold=0.5, gate_temperature=0.1
        )
        self.assertEqual(int(index.item()), 1)
        self.assertGreater(float(probability.item()), 0.5)
        self.assertGreater(float(utility[0, 1]), float(utility[0, 0]))

    def test_inference_adapter_loads_multithreshold_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.pt"
            model = MultiThresholdHypothesisScorer(12, 8, 5)
            torch.save({
                "kind": "routeD_multithreshold_utility_scorer",
                "feature_dim": 12,
                "threshold_count": 5,
                "thresholds_px": [1, 2, 4, 8, 16],
                "config": {"hidden_dim": 8, "gate_temperature": 0.1},
                "model_state": model.state_dict(),
                "feature_mean": torch.zeros(12),
                "feature_std": torch.ones(12),
            }, path)
            selector = RouteDRiskSelector(path, threshold=0.5)
            output = selector.select(
                torch.randn(2, 3, 12), torch.randn(2, 3, 2)
            )
            self.assertEqual(output["threshold_probabilities"].shape, (2, 3, 5))
            self.assertEqual(output["points"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
