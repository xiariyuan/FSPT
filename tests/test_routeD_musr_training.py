from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    MUSRTrainingRows,
    compute_feature_normalization,
    normalize_inputs,
    paired_video_bootstrap_ci,
    visible_post_query_mask,
)


def test_visible_post_query_mask_excludes_query_prequery_and_occlusion():
    tensors = {
        "gt_occluded": torch.tensor(
            [[False, False, True, False], [False, False, False, False]]
        ),
        "query_points_tyx": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    mask = visible_post_query_mask(tensors)
    assert torch.equal(
        mask,
        torch.tensor([[False, False, False, True], [False, True, True, True]]),
    )


def test_normalization_uses_only_valid_candidates_and_zeroes_invalid():
    features = torch.tensor(
        [
            [[1.0, 2.0], [1000.0, 1000.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    valid = torch.tensor([[True, False], [True, True]])
    rows = MUSRTrainingRows(
        candidate_features=features,
        candidate_coords_px=torch.zeros(2, 2, 2),
        candidate_valid_mask=valid,
        state_features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        source_ids=torch.zeros(2, 2, dtype=torch.long),
        gt_coords_px=torch.zeros(2, 2),
        video_row_ids=torch.zeros(2, dtype=torch.long),
        source_rows=2,
        sidecar_paths=(),
    )
    norm = compute_feature_normalization(rows)
    normalized_features, normalized_state = normalize_inputs(
        features, valid, rows.state_features, norm
    )
    assert torch.equal(normalized_features[0, 1], torch.zeros(2))
    assert torch.allclose(normalized_features[valid].mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized_state.mean(dim=0), torch.zeros(2), atol=1e-6)


def test_paired_bootstrap_is_deterministic_and_positive():
    values = [1.0, 2.0, 3.0, 4.0]
    first = paired_video_bootstrap_ci(values, seed=7, samples=1000)
    second = paired_video_bootstrap_ci(values, seed=7, samples=1000)
    assert first == second
    assert first["lower"] > 0.0
    assert first["mean"] == 2.5


def test_state_dict_hash_is_stable_and_parameter_sensitive():
    from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

    state = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0, 3.0])}
    first = state_dict_sha256(state)
    second = state_dict_sha256({"a": state["a"].clone(), "b": state["b"].clone()})
    assert first == second
    changed = {"a": torch.tensor([1.0, 4.0]), "b": state["b"]}
    assert state_dict_sha256(changed) != first


def test_selector_loss_prefers_native_on_utility_tie_and_backpropagates():
    from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import (
        MultiHypothesisStateRecoveryNetwork,
        RecoveryNetworkConfig,
    )
    from projects.mmp_tracker.mmp_tracker.routeD_musr_training import selector_pretraining_loss

    cfg = RecoveryNetworkConfig(
        candidate_feature_dim=4,
        state_feature_dim=3,
        hidden_dim=16,
        num_attention_heads=4,
        num_attention_layers=1,
        num_candidate_sources=4,
        dropout=0.0,
    )
    model = MultiHypothesisStateRecoveryNetwork(cfg)
    features = torch.randn(2, 1, 3, 4)
    coords = torch.tensor(
        [[[[10.0, 10.0], [10.0, 10.0], [30.0, 30.0]]],
         [[[20.0, 20.0], [21.0, 20.0], [40.0, 40.0]]]]
    )
    valid = torch.ones(2, 1, 3, dtype=torch.bool)
    state = torch.randn(2, 1, 3)
    sources = torch.tensor([[[0, 1, 1]], [[0, 1, 1]]])
    gt = torch.tensor([[[10.0, 10.0]], [[20.0, 20.0]]])
    output = model(features, coords, valid, state, sources)
    losses = selector_pretraining_loss(output, coords, valid, gt, cfg)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    assert model.threshold_head[-1].weight.grad is not None
    # At native-safe initialization all scores tie, so native candidate 0 wins.
    assert torch.equal(
        output['selected_candidate_index'],
        torch.zeros_like(output['selected_candidate_index']),
    )
