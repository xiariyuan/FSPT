import torch

from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    build_dense_proposal_targets,
    extract_proposal_candidates,
    stable_spatial_topk_nms,
)


def _config() -> CMCPConfig:
    return CMCPConfig(
        feature_dim=4,
        hidden_channels=8,
        proposal_topk=3,
        input_height=32,
        input_width=32,
    )


def test_zero_step_proposal_is_exact_native():
    config = _config()
    model = CausalMultiMemoryProposalGenerator(config).eval()
    corr = torch.randn(2, 4, 3, 7, 9)
    motion = torch.rand(2, 4, 1, 7, 9)
    valid = torch.ones(2, 4, dtype=torch.bool)
    output = model.forward_sequence(corr, motion, valid)
    native = torch.tensor([[8.0, 9.0], [12.0, 15.0]])
    proposals = extract_proposal_candidates(
        output["proposal_score"][:, -1],
        native,
        output["native_logit"][:, -1],
        config,
    )
    assert torch.equal(
        proposals["selected_candidate_index"], torch.zeros(2, dtype=torch.long)
    )
    assert torch.equal(proposals["selected_coord_xy_px"], native)


def test_stable_nms_uses_row_major_ties_and_suppression():
    score = torch.zeros(1, 4, 5)
    index, selected_score, valid = stable_spatial_topk_nms(
        score, topk=4, radius_cells=1
    )
    assert valid.all()
    assert index.tolist() == [[0, 2, 4, 10]]
    assert torch.equal(selected_score, torch.zeros_like(selected_score))


def test_correlation_builder_masks_prequery_and_is_future_causal():
    torch.manual_seed(7)
    fmaps = torch.randn(5, 4, 6, 8)
    native = torch.tensor(
        [
            [[5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]],
            [[12.0, 4.0], [13.0, 5.0], [14.0, 6.0], [15.0, 7.0], [16.0, 8.0]],
        ]
    )
    queries = torch.tensor([[2.0, 8.0 / 31.0, 7.0 / 31.0], [1.0, 5.0 / 31.0, 13.0 / 31.0]])
    first = build_causal_multi_memory_correlations(
        fmaps,
        native,
        queries,
        input_height=32,
        input_width=32,
    )
    changed = fmaps.clone()
    changed[4] = torch.randn_like(changed[4]) * 100.0
    second = build_causal_multi_memory_correlations(
        changed,
        native,
        queries,
        input_height=32,
        input_width=32,
    )
    correlation, motion, valid = first
    assert not valid[0, :2].any()
    assert not correlation[0, :2].any()
    assert not motion[0, :2].any()
    assert torch.equal(first[0][:, :4], second[0][:, :4])
    assert torch.equal(first[1][:, :4], second[1][:, :4])


def test_sequence_model_is_causal():
    torch.manual_seed(11)
    model = CausalMultiMemoryProposalGenerator(_config()).eval()
    corr = torch.randn(1, 5, 3, 6, 7)
    motion = torch.rand(1, 5, 1, 6, 7)
    valid = torch.ones(1, 5, dtype=torch.bool)
    first = model.forward_sequence(corr, motion, valid)
    changed = corr.clone()
    changed[:, 4] = torch.randn_like(changed[:, 4]) * 100.0
    second = model.forward_sequence(changed, motion, valid)
    for key in first:
        assert torch.equal(first[key][:, :4], second[key][:, :4])


def test_invalid_frame_preserves_recurrent_state():
    model = CausalMultiMemoryProposalGenerator(_config()).eval()
    corr = torch.randn(1, 3, 3, 5, 6)
    motion = torch.rand(1, 3, 1, 5, 6)
    valid = torch.tensor([[True, False, True]])
    first_output, first_state = model.step(corr[:, 0], motion[:, 0])
    invalid_output, invalid_state = model.step(
        corr[:, 1], motion[:, 1], first_state, frame_valid=valid[:, 1]
    )
    assert torch.equal(first_state.hidden, invalid_state.hidden)
    assert torch.equal(first_state.previous_evidence, invalid_state.previous_evidence)
    assert not invalid_output["proposal_score"].any()


def test_dense_targets_are_bounded_and_native_fallback_on_exact_gt():
    gt = torch.tensor([[8.0, 10.0], [20.0, 15.0]])
    native = gt.clone()
    target = build_dense_proposal_targets(
        gt,
        native,
        feature_height=8,
        feature_width=8,
        input_height=32,
        input_width=32,
    )
    assert target["utility_target"].min() >= 0
    assert target["utility_target"].max() <= 1
    assert target["gaussian_target"].min() >= 0
    assert target["gaussian_target"].max() <= 1
    assert target["native_fallback_target"].bool().all()


def test_two_optimizer_steps_reach_recurrent_core():
    torch.manual_seed(13)
    config = _config()
    model = CausalMultiMemoryProposalGenerator(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    corr = torch.randn(2, 3, 3, 5, 6)
    motion = torch.rand(2, 3, 1, 5, 6)
    valid = torch.ones(2, 3, dtype=torch.bool)
    for _ in range(2):
        output = model.forward_sequence(corr, motion, valid)
        loss = output["proposal_score"].square().mean() + output["native_logit"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.recurrent.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0
