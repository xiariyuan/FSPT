import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_context_calibration import (
    CAUSAL_VIDEO_CONTEXT_NAMES,
    build_causal_video_context,
)


class RouteDCausalContextTest(unittest.TestCase):
    def _build(self, action):
        rows = len(action)
        return build_causal_video_context(
            sample_id=torch.tensor([0, 0, 0, 0, 1, 1]),
            frame_id=torch.tensor([1, 1, 2, 3, 1, 2]),
            action_logit=torch.tensor(action, dtype=torch.float32),
            predicted_p1_margin=torch.linspace(-0.1, 0.1, rows),
            predicted_coarse_gain=torch.linspace(0.0, 0.2, rows),
            predicted_total_gain=torch.linspace(-0.05, 0.05, rows),
            global_distance_to_local=torch.linspace(0.1, 0.6, rows),
            candidate_entropy=torch.linspace(0.5, 1.0, rows),
            previous_confidence=torch.linspace(0.2, 0.7, rows),
        )

    def test_shape_and_shared_frame_context(self):
        context = self._build([1, 3, 5, 7, 2, 4])
        self.assertEqual(context.shape, (6, len(CAUSAL_VIDEO_CONTEXT_NAMES)))
        self.assertTrue(torch.allclose(context[0], context[1]))
        self.assertAlmostEqual(float(context[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(context[2, 1]), 3.0, places=6)

    def test_future_changes_do_not_modify_past_context(self):
        first = self._build([1, 3, 5, 7, 2, 4])
        second = self._build([1, 3, 5, 700, 2, 400])
        self.assertTrue(torch.allclose(first[:3], second[:3]))
        self.assertTrue(torch.allclose(first[4:5], second[4:5]))
        self.assertFalse(torch.allclose(first[3], second[3]))
        self.assertFalse(torch.allclose(first[5], second[5]))


if __name__ == "__main__":
    unittest.main()
