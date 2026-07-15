import unittest

import torch

from projects.mmp_tracker.mmp_tracker import MMPTracker, MMPTrackerConfig


class _AlwaysGlobalSelector:
    def select(self, features, candidate_points, valid_mask, **kwargs):
        shape = candidate_points.shape[:-2]
        index = torch.ones(shape, dtype=torch.long, device=candidate_points.device)
        points = candidate_points[..., 1, :]
        probability = torch.ones(shape, device=candidate_points.device)
        threshold_probabilities = torch.full(
            (*candidate_points.shape[:-1], 5),
            0.75,
            device=candidate_points.device,
            dtype=candidate_points.dtype,
        )
        diagnostics = {
            "p1_margin": torch.full_like(probability, 0.1),
            "coarse_gain": torch.full_like(probability, 0.2),
            "total_gain": torch.full_like(probability, 0.15),
        }
        return {
            "points": points,
            "index": index,
            "raw_index": index,
            "gate_probability": probability,
            "global_index": index,
            "threshold_probabilities": threshold_probabilities,
            "profile_diagnostics": diagnostics,
            "fusion_alpha": probability,
        }


class TestRouteDClosedLoop(unittest.TestCase):
    def _config(self):
        config = MMPTrackerConfig()
        config.variant = "localglobal"
        config.encoder.base_dim = 8
        config.encoder.out_dim = 16
        config.posterior_fusion.hidden_dim = 16
        config.global_relocator.topk = 2
        config.global_relocator.memory_size = 2
        config.local_matcher.radius = 1
        config.tracking.global_refine_radius = 1
        return config

    def test_previous_confidence_is_causal_pre_frame_state(self):
        torch.manual_seed(3)
        model = MMPTracker(self._config()).eval()
        video = torch.randn(1, 3, 3, 32, 32)
        query = torch.tensor([[[0.0, 0.4, 0.4]]])
        with torch.no_grad():
            _, _, info = model(video, query, return_info=True)
        previous = info["hypothesis_previous_confidence"]
        state_confidence = info["state_confidence"]
        self.assertTrue(torch.allclose(previous[..., 0], torch.ones_like(previous[..., 0])))
        self.assertTrue(torch.allclose(previous[..., 1], state_confidence[..., 0]))

    def test_routeD_selected_point_becomes_next_prior(self):
        torch.manual_seed(5)
        model = MMPTracker(self._config()).eval()
        model._routeD_selector = _AlwaysGlobalSelector()
        model._routeD_policy = {
            "p1_tolerance": 0.03,
            "min_coarse_gain": 0.0,
            "min_total_gain": -0.05,
            "update_memory": True,
        }
        video = torch.randn(1, 4, 3, 32, 32)
        query = torch.tensor([[[0.0, 0.35, 0.45]]])
        with torch.no_grad():
            tracks, _, info = model(video, query, return_info=True)
        self.assertFalse(bool(info["routeD_selected_global_mask"][..., 0].item()))
        self.assertTrue(bool(info["routeD_selected_global_mask"][..., 1].item()))
        self.assertTrue(
            torch.allclose(info["prior_points"][..., 2, :], tracks[..., 1, :])
        )
        self.assertTrue(info["routeD_closed_loop"])


if __name__ == "__main__":
    unittest.main()
