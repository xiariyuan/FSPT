import unittest

import torch

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import (
    HYPOTHESIS_FEATURE_NAMES,
    HypothesisScorer,
    build_hypothesis_features,
    hypothesis_ranking_loss,
    oracle_candidate_target,
)


class TestHypothesisScorer(unittest.TestCase):
    def test_feature_shape_and_source_features(self):
        points = torch.zeros(2, 3, 2)
        features = build_hypothesis_features(
            points,
            torch.ones(2, 3),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.ones(2),
        )
        self.assertEqual(features.shape, (2, 3, len(HYPOTHESIS_FEATURE_NAMES)))
        self.assertTrue(torch.equal(features[:, 0, 6], torch.zeros(2)))
        self.assertTrue(torch.equal(features[:, 1:, 6], torch.ones(2, 2)))
        self.assertEqual(HypothesisScorer()(features).shape, (2, 3))

    def test_oracle_target_keeps_local_rows(self):
        candidates = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        target, improves = oracle_candidate_target(candidates, torch.tensor([[0.1, 0.1]]))
        self.assertEqual(int(target.item()), 0)
        self.assertFalse(bool(improves.item()))

    def test_ranking_loss_is_finite(self):
        loss = hypothesis_ranking_loss(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([True]),
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
