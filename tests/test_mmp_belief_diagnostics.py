import copy
import unittest

import torch

from projects.mmp_tracker.mmp_tracker import MMPTracker, MMPTrackerConfig


class TestMMPBeliefDiagnostics(unittest.TestCase):
    def _config(self, enabled: bool) -> MMPTrackerConfig:
        config = MMPTrackerConfig()
        config.variant = "localglobal_topk"
        config.encoder.base_dim = 8
        config.encoder.out_dim = 16
        config.posterior_fusion.hidden_dim = 16
        config.global_relocator.topk = 2
        config.global_relocator.memory_size = 2
        config.local_matcher.radius = 1
        config.tracking.global_refine_radius = 1
        config.tracking.enable_multi_hypothesis_diagnostics = enabled
        return config

    def _inputs(self):
        torch.manual_seed(7)
        video = torch.randn(1, 3, 3, 32, 32)
        query_points = torch.tensor([[[0.0, 0.25, 0.25], [1.0, 0.70, 0.60]]])
        return video, query_points

    def test_diagnostics_are_opt_in_and_normalized(self):
        video, query_points = self._inputs()
        model = MMPTracker(self._config(True)).eval()
        with torch.no_grad():
            _, _, info = model(video, query_points, return_info=True)
        self.assertEqual(info["belief_points"].shape, (1, 2, 3, 3, 2))
        self.assertEqual(info["belief_weights"].shape, (1, 2, 3, 3))
        active = info["active_mask"]
        totals = info["belief_weights"].sum(dim=-1)
        self.assertTrue(torch.allclose(totals[active], torch.ones_like(totals[active])))
        self.assertTrue(torch.equal(info["belief_valid_mask"].any(dim=-1), active))
        self.assertIn("belief_normalized_entropy", info)
        self.assertIn("belief_map_points", info)
        self.assertIn("belief_expected_points", info)

    def test_enabling_diagnostics_does_not_change_tracker_outputs(self):
        video, query_points = self._inputs()
        disabled = MMPTracker(self._config(False)).eval()
        enabled = MMPTracker(self._config(True)).eval()
        enabled.load_state_dict(copy.deepcopy(disabled.state_dict()), strict=True)
        with torch.no_grad():
            tracks_off, visibility_off, info_off = disabled(
                video, query_points, return_info=True
            )
            tracks_on, visibility_on, info_on = enabled(
                video, query_points, return_info=True
            )
        self.assertNotIn("belief_weights", info_off)
        self.assertIn("belief_weights", info_on)
        self.assertTrue(torch.allclose(tracks_off, tracks_on, atol=0.0, rtol=0.0))
        self.assertTrue(
            torch.allclose(visibility_off, visibility_on, atol=0.0, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(
                info_off["confidence"], info_on["confidence"], atol=0.0, rtol=0.0
            )
        )


if __name__ == "__main__":
    unittest.main()
