import unittest

import torch

from projects.mmp_tracker.mmp_tracker import (
    MMPLossWeights,
    MMPTracker,
    MMPTrackerConfig,
    MMPTrackingLoss,
)


def _selector_inputs():
    batch, points, time, candidates = 1, 2, 3, 3
    pred_tracks = torch.zeros(batch, points, time, 2)
    pred_visibility = torch.full((batch, points, time), 0.8)
    gt_tracks = torch.zeros(batch, points, time, 2)
    gt_visibility = torch.ones(batch, points, time)
    candidate_points = torch.zeros(batch, points, time, candidates, 2)
    candidate_points[..., 0, :] = 1.0
    return pred_tracks, pred_visibility, gt_tracks, gt_visibility, candidate_points


class TestMMPCandidateRouting(unittest.TestCase):
    def test_flat_candidate_routing_ignores_empty_rank_logits(self):
        pred_tracks, pred_visibility, gt_tracks, gt_visibility, candidate_points = _selector_inputs()
        info = {
            "candidate_routing_mode": "flat",
            "candidate_points": candidate_points,
            "candidate_logits": torch.zeros(*candidate_points.shape[:-1]),
            "candidate_gate_probability": torch.full(candidate_points.shape[:-2], 0.5),
            "candidate_global_rank_logits": torch.zeros(*candidate_points.shape[:-2], 0),
            "active_mask": torch.ones(candidate_points.shape[:-2], dtype=torch.bool),
        }
        criterion = MMPTrackingLoss(MMPLossWeights(selector=1.0, visibility=0.0))

        losses = criterion(
            pred_tracks,
            pred_visibility,
            gt_tracks,
            gt_visibility,
            info=info,
        )

        self.assertTrue(torch.isfinite(losses["selector"]).item())
        self.assertNotIn("selector_gate_raw", losses)
        self.assertNotIn("selector_rank_raw", losses)

    def test_flat_candidate_routing_does_not_use_stale_rank_logits(self):
        pred_tracks, pred_visibility, gt_tracks, gt_visibility, candidate_points = _selector_inputs()
        info = {
            "candidate_routing_mode": "flat",
            "candidate_points": candidate_points,
            "candidate_logits": torch.zeros(*candidate_points.shape[:-1]),
            "candidate_gate_probability": torch.full(candidate_points.shape[:-2], 0.5),
            "candidate_global_rank_logits": torch.zeros(
                *candidate_points.shape[:-2], candidate_points.shape[-2] - 1
            ),
            "active_mask": torch.ones(candidate_points.shape[:-2], dtype=torch.bool),
        }
        criterion = MMPTrackingLoss(MMPLossWeights(selector=1.0, visibility=0.0))

        losses = criterion(
            pred_tracks,
            pred_visibility,
            gt_tracks,
            gt_visibility,
            info=info,
        )

        self.assertTrue(torch.isfinite(losses["selector"]).item())
        self.assertNotIn("selector_gate_raw", losses)
        self.assertNotIn("selector_rank_raw", losses)

    def test_two_stage_candidate_routing_rejects_empty_rank_logits(self):
        pred_tracks, pred_visibility, gt_tracks, gt_visibility, candidate_points = _selector_inputs()
        info = {
            "candidate_routing_mode": "two_stage",
            "candidate_points": candidate_points,
            "candidate_logits": torch.zeros(*candidate_points.shape[:-1]),
            "candidate_gate_probability": torch.full(candidate_points.shape[:-2], 0.5),
            "candidate_global_rank_logits": torch.zeros(*candidate_points.shape[:-2], 0),
            "active_mask": torch.ones(candidate_points.shape[:-2], dtype=torch.bool),
        }
        criterion = MMPTrackingLoss(MMPLossWeights(selector=1.0, visibility=0.0))

        with self.assertRaisesRegex(ValueError, "two-stage candidate routing requires"):
            criterion(
                pred_tracks,
                pred_visibility,
                gt_tracks,
                gt_visibility,
                info=info,
            )

    def test_mmp_tracker_rejects_unknown_candidate_routing_mode(self):
        config = MMPTrackerConfig()
        config.tracking.candidate_routing_mode = "unknown_mode"

        with self.assertRaisesRegex(ValueError, "Unsupported candidate routing mode"):
            MMPTracker(config)


if __name__ == "__main__":
    unittest.main()
