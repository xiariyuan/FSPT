import unittest

import torch

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import (
    HypothesisScorer,
    build_hypothesis_features,
    hypothesis_ranking_loss,
)


class TestHypothesisScorer(unittest.TestCase):
    def test_feature_shape(self):
        points = torch.zeros(2, 3, 2)
        features = build_hypothesis_features(
            points,
            torch.ones(2, 3),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.ones(2),
        )
        self.assertEqual(features.shape, (2, 3, 12))
        self.assertEqual(HypothesisScorer()(features).shape, (2, 3))

    def test_ranking_loss_is_finite(self):
        loss = hypothesis_ranking_loss(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([True]),
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
