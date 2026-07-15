import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_hierarchical_prior import (
    FRAME_PRIOR_FEATURE_NAMES,
    aggregate_frame_mean_target,
    build_causal_frame_prior_features,
    frame_prior_loss,
)


class RouteDHierarchicalPriorTest(unittest.TestCase):
    def _build(self, total):
        rows = len(total)
        return build_causal_frame_prior_features(
            sample_id=torch.tensor([0, 0, 0, 0, 1, 1]),
            frame_id=torch.tensor([1, 1, 2, 3, 1, 2]),
            predicted_p1_margin=torch.linspace(-0.1, 0.1, rows),
            predicted_coarse_gain=torch.linspace(0.0, 0.2, rows),
            predicted_total_gain=torch.tensor(total, dtype=torch.float32),
            global_distance_to_local=torch.linspace(0.1, 0.6, rows),
            candidate_entropy=torch.linspace(0.5, 1.0, rows),
            previous_confidence=torch.linspace(0.2, 0.7, rows),
        )

    def test_shape_mapping_and_target_aggregation(self):
        result = self._build([1, 3, 5, 7, 2, 4])
        self.assertEqual(result["features"].shape, (5, len(FRAME_PRIOR_FEATURE_NAMES)))
        self.assertTrue(torch.equal(result["row_to_frame"], torch.tensor([0, 0, 1, 2, 3, 4])))
        target = aggregate_frame_mean_target(
            torch.tensor([1.0, 3.0, 5.0, 7.0, 2.0, 4.0]),
            result["row_to_frame"],
            5,
        )
        self.assertTrue(torch.allclose(target, torch.tensor([2.0, 5.0, 7.0, 2.0, 4.0])))

    def test_future_change_does_not_modify_past_frames(self):
        first = self._build([1, 3, 5, 7, 2, 4])
        second = self._build([1, 3, 5, 700, 2, 400])
        self.assertTrue(torch.allclose(first["features"][:2], second["features"][:2]))
        self.assertTrue(torch.allclose(first["features"][3:4], second["features"][3:4]))
        self.assertFalse(torch.allclose(first["features"][2], second["features"][2]))
        self.assertFalse(torch.allclose(first["features"][4], second["features"][4]))

    def test_loss_prefers_exact_prediction(self):
        target = torch.tensor([-0.2, 0.0, 0.3])
        self.assertLess(
            float(frame_prior_loss(target, target)),
            float(frame_prior_loss(torch.zeros_like(target), target)),
        )


if __name__ == "__main__":
    unittest.main()
