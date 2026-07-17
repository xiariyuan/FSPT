from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    SelectorLossConfig,
    selector_pretraining_loss,
)
from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import RecoveryNetworkConfig
from projects.mmp_tracker.mmp_tracker.routeD_temporal_selector import (
    CausalSetEvidenceTemporalSelector,
    TemporalSelectorConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_training import (
    build_temporal_batch,
    target_indices,
)


def _configs():
    network = RecoveryNetworkConfig(
        candidate_feature_dim=8,
        state_feature_dim=5,
        hidden_dim=32,
        num_attention_heads=4,
        num_attention_layers=2,
        num_candidate_sources=8,
        dropout=0.0,
    )
    temporal = TemporalSelectorConfig(
        window_frames=4,
        temporal_layers=2,
        temporal_heads=4,
        dropout=0.0,
    )
    return network, temporal


def _batch(batch=2, frames=4, candidates=5):
    torch.manual_seed(11)
    features = torch.randn(batch, frames, candidates, 8)
    coords = torch.randn(batch, frames, candidates, 2) * 4 + 32
    valid = torch.ones(batch, frames, candidates, dtype=torch.bool)
    frame_valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    valid[0, 0] = False
    state = torch.randn(batch, frames, 5)
    source = torch.arange(candidates).view(1, 1, candidates).expand(batch, frames, -1)
    source = source.clone()
    source[0, 0] = 0
    return features, coords, valid, state, source, frame_valid


def test_zero_step_temporal_selector_is_exact_native():
    network, temporal = _configs()
    model = CausalSetEvidenceTemporalSelector(network, temporal).eval()
    batch = _batch()
    output = model(*batch)
    assert torch.equal(output["selected_candidate_index"], torch.zeros(2, 1, dtype=torch.long))
    assert torch.equal(output["selected_coord_px"][:, 0], batch[1][:, -1, 0])
    probability = output["threshold_probabilities"]
    assert torch.all(probability[..., 1:] >= probability[..., :-1])


def test_padded_history_content_is_ignored():
    network, temporal = _configs()
    model = CausalSetEvidenceTemporalSelector(network, temporal).eval()
    first = list(_batch())
    second = [value.clone() for value in first]
    second[0][0, 0] = torch.randn_like(second[0][0, 0]) * 100
    second[1][0, 0] = torch.randn_like(second[1][0, 0]) * 100
    second[3][0, 0] = torch.randn_like(second[3][0, 0]) * 100
    left = model(*first)
    right = model(*second)
    assert torch.equal(left["candidate_score"], right["candidate_score"])


def test_past_non_native_permutation_is_invariant():
    network, temporal = _configs()
    model = CausalSetEvidenceTemporalSelector(network, temporal).eval()
    original = list(_batch())
    permuted = [value.clone() for value in original]
    permutation = torch.tensor([0, 3, 1, 4, 2])
    for index in (0, 1, 2, 4):
        permuted[index][:, :-1] = permuted[index][:, :-1, permutation]
    left = model(*original)
    right = model(*permuted)
    assert torch.allclose(left["candidate_score"], right["candidate_score"], atol=2e-6, rtol=2e-6)


def test_current_non_native_permutation_is_equivariant():
    network, temporal = _configs()
    model = CausalSetEvidenceTemporalSelector(network, temporal).eval()
    original = list(_batch())
    permutation = torch.tensor([0, 3, 1, 4, 2])
    inverse = torch.argsort(permutation)
    permuted = [value.clone() for value in original]
    for index in (0, 1, 2, 4):
        permuted[index][:, -1] = permuted[index][:, -1, permutation]
    left = model(*original)
    right = model(*permuted)
    assert torch.allclose(
        left["candidate_score"], right["candidate_score"][:, :, inverse], atol=2e-6, rtol=2e-6
    )


def test_temporal_selector_reaches_temporal_encoder_after_safe_head_warmup():
    network, temporal = _configs()
    model = CausalSetEvidenceTemporalSelector(network, temporal)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batch = _batch()
    gt = batch[1][:, -1, 2] + torch.tensor([0.5, -0.25])

    # Native-safe output heads are exactly zero at initialization. The first
    # optimizer step must therefore update the heads without leaking arbitrary
    # candidate preferences into zero-step inference.
    output = model(*batch)
    first_loss = selector_pretraining_loss(
        output,
        batch[1][:, -1][:, None],
        batch[2][:, -1][:, None],
        gt[:, None],
        network,
        SelectorLossConfig(),
    )["loss"]
    optimizer.zero_grad(set_to_none=True)
    first_loss.backward()
    head_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.threshold_head.parameters()
        if parameter.grad is not None
    )
    assert head_gradient > 0.0
    optimizer.step()

    # Once the output heads are non-zero, gradients must reach the causal
    # temporal encoder on the next ordinary training step.
    output = model(*batch)
    second_loss = selector_pretraining_loss(
        output,
        batch[1][:, -1][:, None],
        batch[2][:, -1][:, None],
        gt[:, None],
        network,
        SelectorLossConfig(),
    )["loss"]
    optimizer.zero_grad(set_to_none=True)
    second_loss.backward()
    temporal_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.temporal_encoder.parameters()
        if parameter.grad is not None
    )
    assert temporal_gradient > 0.0


def test_temporal_batch_uses_no_future_and_masks_pre_query():
    points, frames, candidates = 2, 6, 3
    tensors = {
        "candidate_features": torch.arange(points * frames * candidates * 8, dtype=torch.float32).reshape(points, frames, candidates, 8),
        "candidate_coords_xy_px": torch.arange(points * frames * candidates * 2, dtype=torch.float32).reshape(points, frames, candidates, 2),
        "candidate_valid_mask": torch.ones(points, frames, candidates, dtype=torch.bool),
        "state_features": torch.arange(points * frames * 5, dtype=torch.float32).reshape(points, frames, 5),
        "source_ids": torch.zeros(points, frames, candidates, dtype=torch.long),
        "query_points_tyx": torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "gt_tracks_yx": torch.zeros(points, frames, 2),
        "gt_occluded": torch.zeros(points, frames, dtype=torch.bool),
    }
    indices = torch.tensor([[0, 3], [1, 5]])
    batch = build_temporal_batch(tensors, indices, window_frames=4)
    assert batch.frame_valid_mask.tolist() == [[False, False, True, True], [True, True, True, True]]
    assert torch.equal(batch.candidate_features[0, -1], tensors["candidate_features"][0, 3])
    assert not torch.equal(batch.candidate_features[0, -1], tensors["candidate_features"][0, 4])
    assert torch.equal(batch.candidate_features[0, :2], torch.zeros_like(batch.candidate_features[0, :2]))


def test_target_indices_training_excludes_occluded_but_eval_does_not():
    tensors = {
        "gt_occluded": torch.tensor([[False, False, True, False]]),
        "query_points_tyx": torch.tensor([[0.0, 0.0, 0.0]]),
    }
    training = target_indices(tensors, training=True)
    evaluation = target_indices(tensors, training=False)
    assert training[:, 1].tolist() == [1, 3]
    assert evaluation[:, 1].tolist() == [1, 2, 3]
