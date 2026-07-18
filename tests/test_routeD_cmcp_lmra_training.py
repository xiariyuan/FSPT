from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LateMetricResidualAdapter,
    LMRAConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training import (
    LMRAJointLossConfig,
    train_lmra_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    PairwiseSafetyLossConfig,
    StaticTokenNormalization,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import (
    CMCPLossConfig,
    CMCPVideoBundle,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _synthetic_bundle() -> CMCPVideoBundle:
    torch.manual_seed(7)
    points, frames = 2, 3
    feature_maps = torch.randn(frames, 128, 4, 4)
    native = torch.tensor(
        [
            [[4.0, 4.0], [5.0, 4.5], [6.0, 5.0]],
            [[10.0, 9.0], [9.5, 9.0], [9.0, 8.5]],
        ]
    )
    gt_xy = native.clone()
    gt_xy[0, 1:] += torch.tensor([2.0, 0.0])
    gt_xy[1, 1:] += torch.tensor([-1.0, 1.0])
    query_points = torch.tensor(
        [
            [0.0, native[0, 0, 1] / 15.0, native[0, 0, 0] / 15.0],
            [0.0, native[1, 0, 1] / 15.0, native[1, 0, 0] / 15.0],
        ]
    )
    tensors = {
        "native_coords_xy_px": native,
        "query_points_tyx": query_points,
        "gt_tracks_yx": gt_xy[..., [1, 0]] / 15.0,
        "gt_occluded": torch.zeros(points, frames, dtype=torch.bool),
        "native_visibility_probability": torch.full((points, frames), 0.9),
        "native_confidence_probability": torch.full((points, frames), 0.8),
        "native_joint_probability": torch.full((points, frames), 0.72),
    }
    return CMCPVideoBundle(
        source_index=0,
        video_name="synthetic",
        feature_maps=feature_maps,
        tensors=tensors,
        feature_sidecar="synthetic_feature.pt",
        base_sidecar="synthetic_base.pt",
    )


def _modules():
    adapter = LateMetricResidualAdapter(LMRAConfig())
    cmcp = CausalMultiMemoryProposalGenerator(
        CMCPConfig(input_height=16, input_width=16)
    )
    comparator = CMCPLocalPairwiseSafetyComparator(
        CMCPLocalSafetyConfig(dropout=0.0)
    )
    # Formal P0i starts from learned P0g/P0h heads. Give the synthetic test the
    # same non-degenerate gradient condition while retaining the real modules.
    with torch.no_grad():
        cmcp.utility_head.weight.fill_(0.01)
        cmcp.risk_head.weight.fill_(0.01)
        comparator.preference_head.weight.fill_(0.01)
        comparator.abstention_head.weight.fill_(0.01)
    return adapter, cmcp, comparator


def test_joint_epoch_updates_once_per_video_and_routes_gradients(monkeypatch):
    bundle = _synthetic_bundle()
    monkeypatch.setattr(
        "projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training.load_complete_feature_index",
        lambda *args, **kwargs: {"videos": [{}]},
    )
    monkeypatch.setattr(
        "projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training.load_cmcp_video",
        lambda row: bundle,
    )
    adapter, cmcp, comparator = _modules()
    optimizer = CountingAdamW(
        list(adapter.parameters()) + list(cmcp.parameters()) + list(comparator.parameters())
    )
    normalization = StaticTokenNormalization(
        mean=torch.zeros(84), std=torch.ones(84)
    )
    metrics = train_lmra_epoch(
        adapter,
        cmcp,
        comparator,
        "unused.json",
        normalization,
        CMCPLossConfig(),
        PairwiseSafetyLossConfig(),
        LMRAJointLossConfig(),
        optimizer,
        device="cpu",
        point_batch_size=1,
        generator=torch.Generator().manual_seed(17),
    )
    assert optimizer.step_calls == 1
    assert metrics["videos"] == 1.0
    assert metrics["supervised_rows"] == 4.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in cmcp.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in comparator.parameters()
    )
    assert bundle.feature_maps.requires_grad is False
    assert bundle.tensors["native_coords_xy_px"].requires_grad is False


def test_memory_sampling_is_stop_gradient_but_current_map_is_trainable():
    adapter, cmcp, _ = _modules()
    frozen = torch.randn(3, 128, 4, 4)
    adapted = adapter(frozen)
    adapted.retain_grad()
    native = torch.tensor([[[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    query = torch.tensor([[0.0, 4.0 / 15.0, 4.0 / 15.0]])
    from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
        build_causal_multi_memory_correlations,
    )

    correlation, motion, valid = build_causal_multi_memory_correlations(
        adapted,
        native,
        query,
        input_height=16,
        input_width=16,
        detach_sampled_memories=True,
    )
    output = cmcp.forward_sequence(correlation, motion, valid)
    output["proposal_score"].sum().backward()
    assert adapted.grad is not None
    assert adapted.grad.abs().sum() > 0
    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0


def _run_variant_epoch(monkeypatch, *, train_adapter: bool, train_cmcp: bool):
    bundle = _synthetic_bundle()
    monkeypatch.setattr(
        "projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training.load_complete_feature_index",
        lambda *args, **kwargs: {"videos": [{}]},
    )
    monkeypatch.setattr(
        "projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training.load_cmcp_video",
        lambda row: bundle,
    )
    adapter, cmcp, comparator = _modules()
    for parameter in adapter.parameters():
        parameter.requires_grad_(train_adapter)
    for parameter in cmcp.parameters():
        parameter.requires_grad_(train_cmcp)
    trainable = [
        parameter
        for module in (adapter, cmcp, comparator)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = CountingAdamW(trainable)
    initial = {
        "adapter": {key: value.detach().clone() for key, value in adapter.state_dict().items()},
        "cmcp": {key: value.detach().clone() for key, value in cmcp.state_dict().items()},
    }
    train_lmra_epoch(
        adapter,
        cmcp,
        comparator,
        "unused.json",
        StaticTokenNormalization(mean=torch.zeros(84), std=torch.ones(84)),
        CMCPLossConfig(),
        PairwiseSafetyLossConfig(),
        LMRAJointLossConfig(),
        optimizer,
        device="cpu",
        point_batch_size=1,
        generator=torch.Generator().manual_seed(17),
        train_adapter=train_adapter,
        train_cmcp=train_cmcp,
        train_comparator=True,
    )
    return adapter, cmcp, comparator, initial


def test_variant_b_freezes_identity_adapter(monkeypatch):
    adapter, cmcp, comparator, initial = _run_variant_epoch(
        monkeypatch, train_adapter=False, train_cmcp=True
    )
    assert all(
        torch.equal(value, initial["adapter"][key])
        for key, value in adapter.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in cmcp.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in comparator.parameters()
    )


def test_variant_c_freezes_formal_cmcp(monkeypatch):
    adapter, cmcp, comparator, initial = _run_variant_epoch(
        monkeypatch, train_adapter=True, train_cmcp=False
    )
    assert all(
        torch.equal(value, initial["cmcp"][key])
        for key, value in cmcp.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in cmcp.parameters())
    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in comparator.parameters()
    )
