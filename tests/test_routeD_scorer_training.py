import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    ROUTED_CACHE_REQUIRED_KEYS,
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    RouteDCandidateCacheDataset,
    ScorerTrainingConfig,
    split_routeD_candidate_cache_by_sample,
    train_hypothesis_scorer,
)


def synthetic_cache(samples=6, rows_per_sample=24, candidates=3):
    rows = samples * rows_per_sample
    sample_id = torch.arange(samples).repeat_interleave(rows_per_sample)
    target = torch.arange(rows) % candidates
    features = torch.randn(rows, candidates, 12) * 0.05
    features[..., 0] = 0.0
    features[torch.arange(rows), target, 0] = 3.0
    candidate_points = torch.zeros(rows, candidates, 2)
    candidate_quality = torch.zeros(rows, candidates)
    candidate_entropy = torch.zeros(rows, candidates)
    valid = torch.ones(rows, candidates, dtype=torch.bool)
    local_error = torch.full((rows,), 10.0)
    oracle_error = torch.full((rows,), 1.0)
    cache = {
        "features": features,
        "candidate_points": candidate_points,
        "candidate_quality": candidate_quality,
        "candidate_entropy": candidate_entropy,
        "candidate_valid_mask": valid,
        "candidate_error_px": torch.where(
            torch.arange(candidates).view(1, candidates) == target.view(rows, 1),
            torch.ones(rows, candidates),
            torch.full((rows, candidates), 10.0),
        ),
        "gt_points": torch.zeros(rows, 2),
        "oracle_index": target.long(),
        "visible": (torch.arange(rows) % 2 == 0),
        "local_error_px": local_error,
        "oracle_error_px": oracle_error,
        "oracle_gain_px": local_error - oracle_error,
        "global_improves_local": target > 0,
        "sample_id": sample_id.long(),
        "point_id": torch.arange(rows).long(),
        "frame_id": torch.ones(rows, dtype=torch.long),
        "query_frame": torch.zeros(rows, dtype=torch.long),
        "format_version": torch.tensor(1, dtype=torch.long),
    }
    validate_routeD_candidate_cache(cache)
    return cache


class TestRouteDScorerTraining(unittest.TestCase):
    def test_dataset_contract(self):
        cache = synthetic_cache(samples=2)
        dataset = RouteDCandidateCacheDataset(cache)
        self.assertEqual(len(dataset), 48)
        self.assertEqual(dataset[0]["features"].shape, (3, 12))

    def test_sample_split_is_disjoint(self):
        cache = synthetic_cache()
        train, validation, split = split_routeD_candidate_cache_by_sample(
            cache, validation_fraction=0.33, seed=17
        )
        self.assertFalse(set(split.train_sample_ids) & set(split.validation_sample_ids))
        self.assertFalse(
            set(train["sample_id"].tolist()) & set(validation["sample_id"].tolist())
        )
        self.assertEqual(train["features"].shape[0] + validation["features"].shape[0], cache["features"].shape[0])

    def test_training_learns_synthetic_oracle_signal(self):
        cache = synthetic_cache(samples=8, rows_per_sample=30)
        train, validation, _ = split_routeD_candidate_cache_by_sample(
            cache, validation_fraction=0.25, seed=3
        )
        bundle = train_hypothesis_scorer(
            train,
            validation,
            ScorerTrainingConfig(
                hidden_dim=24,
                epochs=12,
                batch_size=64,
                learning_rate=5e-3,
                seed=3,
                patience=4,
            ),
        )
        self.assertGreater(bundle["validation_metrics"]["top1_accuracy"], 0.95)
        self.assertLess(bundle["validation_metrics"]["cross_entropy"], 0.2)
        self.assertFalse(bundle["paper_claim_eligible"])


if __name__ == "__main__":
    unittest.main()
