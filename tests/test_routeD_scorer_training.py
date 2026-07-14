import unittest

import torch

from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    ROUTED_CACHE_REQUIRED_KEYS,
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    RouteDCandidateCacheDataset,
    ScorerTrainingConfig,
    compute_feature_normalization,
    normalize_hypothesis_features,
    routeD_training_loss,
    scorer_model_selection_value,
    select_routeD_candidate,
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


    def test_feature_normalization_is_train_only_and_finite(self):
        cache = synthetic_cache(samples=3)
        mean, std = compute_feature_normalization(cache)
        normalized = normalize_hypothesis_features(cache["features"], mean, std)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(mean.shape, (12,))
        self.assertEqual(std.shape, (12,))

    def test_gain_weighted_loss_is_finite(self):
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        target = torch.tensor([0, 1])
        errors = torch.tensor([[1.0, 5.0], [8.0, 1.0]])
        gain = torch.tensor([0.0, 7.0])
        loss = routeD_training_loss(
            logits,
            target,
            errors,
            gain,
            ScorerTrainingConfig(loss_mode="gain_regret", regret_loss_weight=0.5),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


    def test_gain_mode_selects_by_exact_error(self):
        metric, value = scorer_model_selection_value(
            {"cross_entropy": 1.0, "mean_selected_error_px": 8.0},
            ScorerTrainingConfig(loss_mode="gain_pairwise"),
        )
        self.assertEqual(metric, "mean_selected_error_px")
        self.assertEqual(value, 8.0)

    def test_pairwise_loss_penalizes_local_over_hard_global(self):
        target = torch.tensor([1])
        errors = torch.tensor([[10.0, 1.0]])
        gain = torch.tensor([9.0])
        config = ScorerTrainingConfig(
            loss_mode="gain_pairwise",
            pairwise_loss_weight=1.0,
            pairwise_margin=1.0,
            hard_positive_min_gain_px=3.0,
        )
        bad = routeD_training_loss(
            torch.tensor([[2.0, 0.0]]), target, errors, gain, config
        )
        good = routeD_training_loss(
            torch.tensor([[0.0, 2.0]]), target, errors, gain, config
        )
        self.assertGreater(float(bad), float(good))


    def test_risk_gate_keeps_local_when_global_is_not_confident(self):
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        prediction, probability, _ = select_routeD_candidate(
            logits, valid, selection_mode="risk_gate", gate_threshold=0.5
        )
        self.assertEqual(int(prediction.item()), 0)
        self.assertLess(float(probability.item()), 0.5)

    def test_risk_gate_selects_confident_global(self):
        logits = torch.tensor([[0.0, 2.0, 1.0]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        prediction, probability, _ = select_routeD_candidate(
            logits, valid, selection_mode="risk_gate", gate_threshold=0.5
        )
        self.assertEqual(int(prediction.item()), 1)
        self.assertGreater(float(probability.item()), 0.5)

    def test_risk_aware_loss_penalizes_harmful_global_preference(self):
        target = torch.tensor([0, 1])
        errors = torch.tensor([[1.0, 8.0, 9.0], [10.0, 1.0, 3.0]])
        gain = torch.tensor([0.0, 9.0])
        valid = torch.ones_like(errors, dtype=torch.bool)
        config = ScorerTrainingConfig(
            loss_mode="risk_aware",
            gate_min_gain_px=3.0,
            hard_negative_weight=4.0,
        )
        bad = routeD_training_loss(
            torch.tensor([[0.0, 3.0, 2.0], [0.0, 3.0, 1.0]]),
            target,
            errors,
            gain,
            config,
            candidate_valid_mask=valid,
        )
        good = routeD_training_loss(
            torch.tensor([[3.0, 0.0, -1.0], [0.0, 3.0, 1.0]]),
            target,
            errors,
            gain,
            config,
            candidate_valid_mask=valid,
        )
        self.assertGreater(float(bad), float(good))

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
