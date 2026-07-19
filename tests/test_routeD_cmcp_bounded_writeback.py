from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_bounded_writeback import (
    BoundedWritebackConfig,
    bounded_coordinate_writeback,
    initialize_variant_c_online_state,
    normalize_adapted_feature_maps,
    overlap_write_eligible,
    variant_c_online_step,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRAConfig,
    LateMetricResidualAdapter,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    StaticTokenNormalization,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    extract_proposal_candidates,
)


def test_bounded_writeback_is_native_safe_and_norm_clipped():
    native = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    selected = torch.tensor([[30.0, 10.0], [23.0, 24.0], [40.0, 40.0]])
    index = torch.tensor([1, 1, 0])
    eligible = torch.tensor([True, False, True])
    write, applied, norm = bounded_coordinate_writeback(
        native,
        selected,
        index,
        eligible,
        BoundedWritebackConfig(max_step_px=8.0),
    )
    assert applied.tolist() == [True, False, False]
    assert torch.allclose(write[0], torch.tensor([18.0, 10.0]))
    assert torch.equal(write[1:], native[1:])
    assert norm.max().item() <= 8.0 + 1e-6


def test_overlap_eligibility_is_exact_second_half():
    rows = [
        overlap_write_eligible(
            frame, chunk_start=8, step=8, point_count=2, device="cpu"
        )[0].item()
        for frame in (15, 16, 23, 24)
    ]
    assert rows == [False, True, True, False]


def test_framewise_variant_c_matches_formal_correlation_construction():
    torch.manual_seed(3)
    points, frames, height, width = 2, 4, 4, 4
    frozen = torch.randn(frames, 128, height, width)
    adapter = LateMetricResidualAdapter(LMRAConfig())
    cmcp = CausalMultiMemoryProposalGenerator(
        CMCPConfig(input_height=16, input_width=16, hidden_channels=64)
    )
    comparator = CMCPLocalPairwiseSafetyComparator(
        CMCPLocalSafetyConfig(hidden_dim=16, attention_heads=4, attention_layers=1, dropout=0.0)
    )
    adapter.eval(); cmcp.eval(); comparator.eval()
    native = torch.tensor(
        [
            [[3.0, 3.0], [4.0, 3.5], [5.0, 4.0], [6.0, 4.5]],
            [[9.0, 9.0], [8.5, 9.0], [8.0, 8.5], [7.5, 8.0]],
        ]
    )
    queries = torch.tensor([[0.0, 3.0 / 15.0, 3.0 / 15.0], [1.0, 9.0 / 15.0, 9.0 / 15.0]])
    vis = torch.full((points, frames), 0.8)
    conf = torch.full((points, frames), 0.7)
    norm = StaticTokenNormalization(mean=torch.zeros(84), std=torch.ones(84))
    adapted_raw = adapter(frozen)
    adapted = normalize_adapted_feature_maps(adapter, frozen)
    online = initialize_variant_c_online_state(
        adapted, queries, input_height=16, input_width=16
    )
    online_candidates = []
    online_valid = []
    with torch.no_grad():
        for frame in range(frames):
            output, online = variant_c_online_step(
                frame_index=frame,
                normalized_feature_map=adapted[frame],
                native_coord_xy_px=native[:, frame],
                native_visibility_probability=vis[:, frame],
                native_confidence_probability=conf[:, frame],
                state=online,
                cmcp=cmcp,
                comparator=comparator,
                normalization=norm,
            )
            online_candidates.append(output["candidate_coords_xy_px"])
            online_valid.append(output["candidate_valid_mask"])

    corr, motion, valid = build_causal_multi_memory_correlations(
        adapted_raw,
        native,
        queries,
        input_height=16,
        input_width=16,
        ema_alpha=cmcp.config.ema_alpha,
        motion_sigma_cells=cmcp.config.motion_sigma_cells,
    )
    state = None
    formal_candidates = []
    formal_valid = []
    with torch.no_grad():
        for frame in range(frames):
            dense, state = cmcp.step(corr[:, frame], motion[:, frame], state, frame_valid=valid[:, frame])
            proposal = extract_proposal_candidates(
                dense["proposal_score"], native[:, frame], dense["native_logit"], cmcp.config
            )
            mask = proposal["candidate_valid_mask"].clone()
            mask[:, 1:] &= valid[:, frame, None]
            formal_candidates.append(proposal["candidate_coords_xy_px"])
            formal_valid.append(mask)
    assert torch.equal(torch.stack(online_candidates, 1), torch.stack(formal_candidates, 1))
    assert torch.equal(torch.stack(online_valid, 1), torch.stack(formal_valid, 1))
