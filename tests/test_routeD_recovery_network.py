from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import (
    MultiHypothesisStateRecoveryNetwork,
    RecoveryLossConfig,
    RecoveryNetworkConfig,
    recovery_training_loss,
)


def _config(**overrides) -> RecoveryNetworkConfig:
    values = dict(
        candidate_feature_dim=10,
        state_feature_dim=7,
        hidden_dim=32,
        num_attention_heads=4,
        num_attention_layers=2,
        num_candidate_sources=8,
        dropout=0.0,
        max_coordinate_update_px=6.0,
    )
    values.update(overrides)
    return RecoveryNetworkConfig(**values)


def _batch(config: RecoveryNetworkConfig, *, batch=2, points=3, candidates=5):
    torch.manual_seed(13)
    features = torch.randn(batch, points, candidates, config.candidate_feature_dim)
    coords = torch.randn(batch, points, candidates, 2) * 12.0 + 64.0
    valid = torch.ones(batch, points, candidates, dtype=torch.bool)
    valid[0, 0, -1] = False
    state = torch.randn(batch, points, config.state_feature_dim)
    sources = torch.arange(candidates).view(1, 1, candidates).expand(batch, points, -1)
    return features, coords, valid, state, sources


def test_forward_contract_monotonicity_masking_and_trust_region():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config).eval()
    features, coords, valid, state, sources = _batch(config)
    outputs = model(features, coords, valid, state, sources)

    assert outputs["threshold_probabilities"].shape == (2, 3, 5, 5)
    assert outputs["candidate_probability"].shape == (2, 3, 5)
    assert outputs["state_write_strength"].shape == (2, 3, 4)
    probabilities = outputs["threshold_probabilities"]
    assert torch.all(probabilities[..., 1:] >= probabilities[..., :-1])
    assert outputs["candidate_probability"][0, 0, -1].item() == 0.0
    assert torch.allclose(
        outputs["candidate_probability"].sum(dim=-1), torch.ones(2, 3), atol=1e-6
    )
    update = outputs["updated_coord_px"] - outputs["native_coord_px"]
    assert torch.all(
        torch.linalg.vector_norm(update, dim=-1)
        <= config.max_coordinate_update_px + 1e-5
    )
    assert torch.all(outputs["state_write_strength"] >= 0.0)
    assert torch.all(outputs["state_write_strength"] <= 1.0)


def test_non_native_candidate_permutation_equivariance():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config).eval()
    features, coords, valid, state, sources = _batch(config)
    baseline = model(features, coords, valid, state, sources)

    permutation = torch.tensor([0, 3, 1, 4, 2])
    inverse = torch.argsort(permutation)
    permuted = model(
        features[:, :, permutation],
        coords[:, :, permutation],
        valid[:, :, permutation],
        state,
        sources[:, :, permutation],
    )
    remapped_probability = permuted["candidate_probability"][:, :, inverse]
    assert torch.allclose(
        baseline["candidate_probability"], remapped_probability, atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        baseline["updated_coord_px"], permuted["updated_coord_px"], atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        baseline["abstention_probability"],
        permuted["abstention_probability"],
        atol=2e-6,
        rtol=2e-6,
    )


def test_only_native_candidate_is_exact_coordinate_parity():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config).eval()
    features, coords, valid, state, sources = _batch(config)
    valid[..., 1:] = False
    outputs = model(features, coords, valid, state, sources, use_hard_selection=True)
    assert torch.equal(outputs["selected_candidate_index"], torch.zeros_like(outputs["selected_candidate_index"]))
    assert torch.allclose(outputs["updated_coord_px"], coords[..., 0, :], atol=1e-7)


def test_native_candidate_must_be_valid():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config)
    features, coords, valid, state, sources = _batch(config)
    valid[0, 1, 0] = False
    try:
        model(features, coords, valid, state, sources)
    except ValueError as exc:
        assert "native continuation" in str(exc)
    else:
        raise AssertionError("invalid native candidate should be rejected")


def test_composite_loss_is_finite_and_backpropagates():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config)
    features, coords, valid, state, sources = _batch(config)
    gt = coords[..., 0, :] + torch.tensor([1.5, -0.75])
    outputs = model(features, coords, valid, state, sources)
    losses = recovery_training_loss(
        outputs,
        coords,
        valid,
        gt,
        config,
        RecoveryLossConfig(tail_weight=0.1),
    )
    assert set(losses) == {
        "loss",
        "threshold_bce",
        "ranking",
        "catastrophe",
        "coordinate",
        "no_harm",
        "abstention",
        "tail",
        "mean_updated_error_px",
        "mean_native_error_px",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    gradient_norm = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0


def test_hard_selection_is_deterministic_in_eval_mode():
    config = _config()
    model = MultiHypothesisStateRecoveryNetwork(config).eval()
    batch = _batch(config)
    first = model(*batch, use_hard_selection=True)
    second = model(*batch, use_hard_selection=True)
    assert torch.equal(first["selected_candidate_index"], second["selected_candidate_index"])
    assert torch.equal(first["updated_coord_px"], second["updated_coord_px"])
