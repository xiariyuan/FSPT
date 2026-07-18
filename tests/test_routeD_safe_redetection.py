from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection import (
    RedetectionStepState,
    SafeRedetectionConfig,
    SafeRedetectionModel,
    apply_future_writeback,
    build_redetection_targets,
    initialize_appearance_memory,
    initialize_confirmation_state,
    make_future_writeback,
    safe_redetection_dense_proposal_loss,
    safe_redetection_frame_loss,
    update_appearance_memory,
    update_confirmation,
)


def _config() -> SafeRedetectionConfig:
    return SafeRedetectionConfig(
        feature_dim=8,
        memory_slots=4,
        proposal_topk=2,
        proposal_hidden_dim=8,
        comparator_hidden_dim=32,
        comparator_heads=4,
        comparator_layers=1,
        input_height=16,
        input_width=16,
        confirmation_frames=2,
        max_future_writeback_step_px=4.0,
    )


def _state(config: SafeRedetectionConfig, batch: int = 2):
    torch.manual_seed(3)
    query = torch.randn(batch, config.feature_dim)
    memory = initialize_appearance_memory(query, 0, config)
    confirmation = initialize_confirmation_state(batch, device=query.device)
    return RedetectionStepState(memory, confirmation)


def test_zero_step_is_exact_native_for_coordinate_visibility_and_writeback():
    config = _config()
    model = SafeRedetectionModel(config).eval()
    state = _state(config)
    fmap = torch.randn(2, config.feature_dim, 4, 4)
    native = torch.tensor([[4.0, 5.0], [10.0, 8.0]])
    visibility = torch.tensor([0.2, 0.9])
    confidence = torch.tensor([0.3, 0.95])
    output, _ = model.step(
        frame_index=1,
        feature_map=fmap,
        native_coord_xy_px=native,
        native_visibility_probability=visibility,
        native_confidence_probability=confidence,
        occlusion_age=torch.tensor([4, 0]),
        state=state,
    )
    assert torch.equal(output["candidate_coords_xy_px"][:, 0], native)
    assert torch.equal(output["selected_candidate_index"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(output["selected_coord_xy_px"], native)
    assert torch.equal(output["selected_visibility_probability"], visibility)
    assert not output["confirmed_recovery"].any()
    assert not output["future_writeback"].allowed.any()


def test_query_independence_under_other_query_perturbation():
    config = _config()
    model = SafeRedetectionModel(config).eval()
    state_a = _state(config)
    state_b = RedetectionStepState(
        memory=type(state_a.memory)(
            anchors=state_a.memory.anchors.clone(),
            valid=state_a.memory.valid.clone(),
            reliability=state_a.memory.reliability.clone(),
            frame_index=state_a.memory.frame_index.clone(),
            next_episodic_slot=state_a.memory.next_episodic_slot.clone(),
        ),
        confirmation=state_a.confirmation,
    )
    fmap_a = torch.randn(2, config.feature_dim, 4, 4)
    fmap_b = fmap_a.clone()
    fmap_b[1] = torch.randn_like(fmap_b[1]) * 100
    native_a = torch.tensor([[4.0, 5.0], [10.0, 8.0]])
    native_b = native_a.clone(); native_b[1] = torch.tensor([0.0, 15.0])
    kwargs = dict(
        frame_index=1,
        native_visibility_probability=torch.tensor([0.4, 0.8]),
        native_confidence_probability=torch.tensor([0.5, 0.9]),
        occlusion_age=torch.tensor([3, 0]),
    )
    out_a, _ = model.step(feature_map=fmap_a, native_coord_xy_px=native_a, state=state_a, **kwargs)
    out_b, _ = model.step(feature_map=fmap_b, native_coord_xy_px=native_b, state=state_b, **kwargs)
    for key in ("candidate_coords_xy_px", "candidate_score", "selected_coord_xy_px"):
        assert torch.equal(out_a[key][0], out_b[key][0])


def test_memory_updates_only_when_reliable_or_confirmed():
    config = _config()
    state = _state(config).memory
    observed = torch.randn(2, config.feature_dim)
    unchanged = update_appearance_memory(
        state,
        observed,
        frame_index=3,
        visibility_probability=torch.tensor([0.1, 0.9]),
        confidence_probability=torch.tensor([0.9, 0.1]),
        anchor_consistency=torch.tensor([0.9, 0.9]),
        confirmed_recovery=torch.tensor([False, False]),
        config=config,
    )
    assert torch.equal(unchanged.anchors, state.anchors)
    updated = update_appearance_memory(
        state,
        observed,
        frame_index=3,
        visibility_probability=torch.tensor([0.1, 0.9]),
        confidence_probability=torch.tensor([0.1, 0.9]),
        anchor_consistency=torch.tensor([0.1, 0.9]),
        confirmed_recovery=torch.tensor([True, False]),
        config=config,
    )
    assert updated.valid[:, 1].all()
    assert not torch.equal(updated.anchors[:, 1], state.anchors[:, 1])


def test_confirmation_requires_two_consistent_frames():
    config = _config()
    state = initialize_confirmation_state(1, device="cpu")
    index = torch.tensor([2])
    coord = torch.tensor([[12.0, 7.0]])
    state, confirmed = update_confirmation(state, index, coord, torch.tensor([True]), config)
    assert not confirmed.item()
    state, confirmed = update_confirmation(state, index, coord + 0.5, torch.tensor([True]), config)
    assert confirmed.item()


def test_writeback_is_bounded_and_future_only():
    config = _config()
    native = torch.tensor([[0.0, 0.0]])
    selected = torch.tensor([[12.0, 0.0]])
    writeback = make_future_writeback(
        frame_index=5,
        native_coord_xy_px=native,
        selected_coord_xy_px=selected,
        selected_visibility_probability=torch.tensor([0.9]),
        native_confidence_probability=torch.tensor([0.2]),
        selected_writeback_probability=torch.tensor([0.9]),
        confirmed=torch.tensor([True]),
        config=config,
    )
    current = apply_future_writeback(
        native, torch.tensor([0.1]), torch.tensor([0.2]), writeback, target_frame=5
    )
    future = apply_future_writeback(
        native, torch.tensor([0.1]), torch.tensor([0.2]), writeback, target_frame=6
    )
    assert torch.equal(current[0], native)
    assert torch.allclose(future[0], torch.tensor([[4.0, 0.0]]))
    assert future[1].item() == torch.tensor(0.9).item()


def test_event_targets_mark_record_breaking_long_reappearance():
    visible = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 1]], dtype=torch.bool)
    targets = build_redetection_targets(visible, torch.tensor([0]))
    assert torch.where(targets["eligible_reappearance_mask"][0])[0].tolist() == [2, 6]
    assert targets["occlusion_duration"][0, 6].item() == 2


def test_dense_and_comparator_losses_are_finite_and_backpropagate():
    config = _config()
    model = SafeRedetectionModel(config)
    state = _state(config)
    fmap = torch.randn(2, config.feature_dim, 4, 4)
    native = torch.tensor([[4.0, 5.0], [10.0, 8.0]])
    output, _ = model.step(
        frame_index=1,
        feature_map=fmap,
        native_coord_xy_px=native,
        native_visibility_probability=torch.tensor([0.2, 0.9]),
        native_confidence_probability=torch.tensor([0.3, 0.95]),
        occlusion_age=torch.tensor([4, 0]),
        state=state,
    )
    gt = torch.tensor([[12.0, 8.0], [10.0, 8.0]])
    dense = safe_redetection_dense_proposal_loss(
        output["proposal_score"], output["visibility_evidence_map"], gt,
        torch.tensor([True, True]), torch.tensor([True, True]), config,
    )
    local = safe_redetection_frame_loss(
        output,
        output["candidate_coords_xy_px"].detach(),
        output["candidate_valid_mask"],
        gt,
        torch.tensor([True, True]),
        torch.tensor([True, False]),
        torch.tensor([True, True]),
        config,
    )
    total = dense["loss"] + local["loss"]
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.proposal.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.comparator.parameters())
