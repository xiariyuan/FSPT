import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    extract_routeD_candidate_cache_batch,
    merge_routeD_candidate_cache_batches,
    validate_routeD_candidate_cache,
)


class TestRouteDCandidateCache(unittest.TestCase):
    def _inputs(self):
        candidate_points = torch.tensor(
            [[[[[0.0, 0.0], [0.5, 0.5]], [[0.1, 0.1], [0.8, 0.8]], [[0.2, 0.2], [0.3, 0.3]]]]]
        )
        info = {
            "hypothesis_candidate_points": candidate_points,
            "hypothesis_candidate_quality": torch.full((1, 1, 3, 2), 0.5),
            "hypothesis_candidate_entropy": torch.full((1, 1, 3, 2), 0.2),
            "hypothesis_previous_points": torch.zeros(1, 1, 3, 2),
            "hypothesis_previous_confidence": torch.ones(1, 1, 3),
        }
        target = torch.tensor([[[[0.0, 0.0], [0.75, 0.75], [0.3, 0.3]]]])
        occluded = torch.tensor([[[False, True, False]]])
        query = torch.tensor([[[0.0, 0.0, 0.0]]])
        return info, target, occluded, query

    def test_extract_excludes_query_and_preserves_visibility(self):
        info, target, occluded, query = self._inputs()
        cache = extract_routeD_candidate_cache_batch(
            info,
            target,
            occluded,
            query,
            video_height=101,
            video_width=101,
            global_gain_margin_px=1.0,
        )
        validate_routeD_candidate_cache(cache)
        self.assertEqual(cache["features"].shape, (2, 2, 12))
        self.assertEqual(cache["visible"].tolist(), [False, True])
        self.assertEqual(cache["oracle_index"].tolist(), [1, 1])
        self.assertTrue(cache["global_improves_local"].all())
        self.assertEqual(cache["frame_id"].tolist(), [1, 2])

    def test_merge_preserves_rows(self):
        info, target, occluded, query = self._inputs()
        cache = extract_routeD_candidate_cache_batch(
            info, target, occluded, query, video_height=64, video_width=64
        )
        merged = merge_routeD_candidate_cache_batches([cache, cache])
        self.assertEqual(merged["features"].shape[0], 4)
        validate_routeD_candidate_cache(merged)


if __name__ == "__main__":
    unittest.main()
