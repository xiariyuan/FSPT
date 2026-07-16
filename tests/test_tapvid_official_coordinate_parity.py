from __future__ import annotations

import unittest

import numpy as np
import torch

from datasets.metrics import compute_tapvid_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official


class TestTapVidOfficialCoordinateParity(unittest.TestCase):
    def test_strict_one_pixel_boundary_uses_full_raster_size(self):
        # Internal tensors use normalized [y, x].  An x error of 1.001 / 256 is
        # outside the official strict 1-pixel threshold.  Multiplying by 255 would
        # incorrectly shrink it below one pixel and make this test fail.
        gt = torch.zeros((1, 3, 2), dtype=torch.float32)
        pred = gt.clone()
        pred[:, 1:, 1] = 1.001 / 256.0
        visible = torch.ones((1, 3), dtype=torch.bool)
        query = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        metrics = compute_tapvid_metrics(
            pred,
            gt,
            visible,
            visible,
            query,
            resolution=(256, 256),
            query_mode="first",
        )
        self.assertEqual(metrics["pts_within_1"], 0.0)
        self.assertEqual(metrics["pts_within_2"], 1.0)

    def test_wrapper_matches_official_raster_inputs(self):
        rng = np.random.default_rng(17)
        n, t = 7, 13
        height, width = 192, 320

        gt_yx = rng.uniform(-0.02, 1.02, size=(n, t, 2)).astype(np.float32)
        pred_yx = gt_yx + rng.normal(0.0, 0.012, size=(n, t, 2)).astype(np.float32)
        gt_visible = rng.random((n, t)) > 0.25
        pred_visible = rng.random((n, t)) > 0.30
        # Guarantee a valid visible point after every first-query frame.
        gt_visible[:, -1] = True
        query_frames = rng.integers(0, t - 1, size=n)
        query = np.zeros((n, 3), dtype=np.float32)
        query[:, 0] = query_frames
        query[:, 1:] = gt_yx[np.arange(n), query_frames]

        scale_xy = np.asarray([width, height], dtype=np.float32)
        gt_xy_px = gt_yx[..., [1, 0]] * scale_xy
        pred_xy_px = pred_yx[..., [1, 0]] * scale_xy
        query_px = query.copy()
        query_px[:, 1] *= height
        query_px[:, 2] *= width

        for mode in ("first", "strided"):
            wrapped = compute_tapvid_metrics(
                torch.from_numpy(pred_yx),
                torch.from_numpy(gt_yx),
                torch.from_numpy(pred_visible),
                torch.from_numpy(gt_visible),
                torch.from_numpy(query),
                resolution=(height, width),
                query_mode=mode,
            )
            official = compute_tapvid_metrics_official(
                query_points=query_px[None],
                gt_occluded=(~gt_visible)[None],
                gt_tracks=gt_xy_px[None],
                pred_occluded=(~pred_visible)[None],
                pred_tracks=pred_xy_px[None],
                query_mode=mode,
            )
            for key, value in official.items():
                expected = float(np.asarray(value).reshape(-1)[0])
                self.assertAlmostEqual(wrapped[key], expected, places=7, msg=f"{mode}:{key}")


if __name__ == "__main__":
    unittest.main()
