"""Causal variant-C inference and bounded coordinate-only writeback primitives.

This module does not load data or mutate a tracker.  It exposes the exact
framewise P0j-C proposal/comparator path and a separately testable state-write
operator.  Runtime scripts are responsible for applying the write only inside
the CoTracker overlap that is consumed by the next online window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .cotracker3_stage0_adapter import sample_feature_at_xy
from .routeD_cmcp_pairwise_cache import inject_dynamic_summary
from .routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    build_cmcp_local_candidate_tokens,
)
from .routeD_cmcp_pairwise_training import (
    StaticTokenNormalization,
    _update_summary,
    normalize_static_tokens,
)
from .routeD_multi_memory_proposal import (
    CMCPConfig,
    CMCPState,
    CausalMultiMemoryProposalGenerator,
    build_native_motion_prior,
    extract_proposal_candidates,
)


@dataclass(frozen=True)
class BoundedWritebackConfig:
    max_step_px: float = 8.0
    input_height: int = 256
    input_width: int = 256

    def __post_init__(self) -> None:
        if self.max_step_px <= 0.0:
            raise ValueError("max_step_px must be positive")
        if self.input_height <= 1 or self.input_width <= 1:
            raise ValueError("input raster must exceed one pixel")


class VariantCOnlineState(NamedTuple):
    query_support: torch.Tensor
    previous_support: torch.Tensor
    ema_support: torch.Tensor
    query_frames: torch.Tensor
    cmcp_state: CMCPState | None
    decision_summary: torch.Tensor


def normalize_adapted_feature_maps(
    adapter: torch.nn.Module,
    frozen_feature_maps: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen LMRA and the formal post-adapter L2 normalization."""
    if frozen_feature_maps.ndim != 4:
        raise ValueError("feature maps must have shape (T,D,H,W)")
    return F.normalize(adapter(frozen_feature_maps.float()), dim=1, eps=1.0e-12)


def initialize_variant_c_online_state(
    normalized_feature_maps: torch.Tensor,
    query_points_tyx: torch.Tensor,
    *,
    input_height: int = 256,
    input_width: int = 256,
) -> VariantCOnlineState:
    """Initialize query/previous/EMA memories exactly as formal P0j-C."""
    if normalized_feature_maps.ndim != 4:
        raise ValueError("feature maps must have shape (T,D,H,W)")
    if query_points_tyx.ndim != 2 or query_points_tyx.shape[-1] != 3:
        raise ValueError("query_points_tyx must have shape (N,3)")
    frames = normalized_feature_maps.shape[0]
    queries = query_points_tyx.to(
        device=normalized_feature_maps.device,
        dtype=normalized_feature_maps.dtype,
    )
    query_frames = queries[:, 0].round().long().clamp(0, frames - 1)
    query_xy = torch.stack(
        [
            queries[:, 2] * float(input_width - 1),
            queries[:, 1] * float(input_height - 1),
        ],
        dim=-1,
    )
    support = torch.stack(
        [
            sample_feature_at_xy(
                normalized_feature_maps[int(query_frames[index].item())],
                query_xy[index],
                input_height=input_height,
                input_width=input_width,
            )
            for index in range(query_points_tyx.shape[0])
        ]
    )
    support = F.normalize(support.float(), dim=-1, eps=1.0e-12)
    return VariantCOnlineState(
        query_support=support,
        previous_support=support.clone(),
        ema_support=support.clone(),
        query_frames=query_frames,
        cmcp_state=None,
        decision_summary=torch.zeros(
            query_points_tyx.shape[0],
            4,
            device=normalized_feature_maps.device,
            dtype=normalized_feature_maps.dtype,
        ),
    )


def variant_c_online_step(
    *,
    frame_index: int,
    normalized_feature_map: torch.Tensor,
    native_coord_xy_px: torch.Tensor,
    native_visibility_probability: torch.Tensor,
    native_confidence_probability: torch.Tensor,
    state: VariantCOnlineState,
    cmcp: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    normalization: StaticTokenNormalization,
) -> tuple[dict[str, torch.Tensor], VariantCOnlineState]:
    """Run one causal P0j-C frame for a fixed point batch.

    The memory update uses the pre-writeback native feature.  The caller may
    mutate the external tracker coordinate state only after this function
    returns.
    """
    if normalized_feature_map.ndim != 3:
        raise ValueError("normalized_feature_map must have shape (D,H,W)")
    if native_coord_xy_px.ndim != 2 or native_coord_xy_px.shape[-1] != 2:
        raise ValueError("native coordinates must have shape (N,2)")
    batch = native_coord_xy_px.shape[0]
    for name, value in {
        "native_visibility_probability": native_visibility_probability,
        "native_confidence_probability": native_confidence_probability,
    }.items():
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape (N,)")
    if state.query_support.shape[0] != batch:
        raise ValueError("state batch mismatch")

    # ``normalized_feature_map`` already equals the single post-LMRA
    # normalization used by formal P0j-C.  Re-normalizing here changes dense
    # correlation ties and can move otherwise unused proposal peaks.
    fmap = normalized_feature_map.float()
    native = native_coord_xy_px.to(device=fmap.device, dtype=fmap.dtype)
    active = torch.full_like(state.query_frames, int(frame_index)) >= state.query_frames
    current_native = sample_feature_at_xy(
        fmap,
        native,
        input_height=cmcp.config.input_height,
        input_width=cmcp.config.input_width,
    )
    current_native = F.normalize(current_native.float(), dim=-1, eps=1.0e-12)
    memories = torch.stack(
        [state.query_support, state.previous_support, state.ema_support], dim=1
    )
    correlation = torch.einsum("nmd,dhw->nmhw", memories, fmap)
    correlation = torch.where(
        active[:, None, None, None], correlation, torch.zeros_like(correlation)
    )
    motion = build_native_motion_prior(
        native,
        feature_height=fmap.shape[-2],
        feature_width=fmap.shape[-1],
        input_height=cmcp.config.input_height,
        input_width=cmcp.config.input_width,
        sigma_cells=cmcp.config.motion_sigma_cells,
    )
    motion = torch.where(active[:, None, None, None], motion, torch.zeros_like(motion))
    dense, cmcp_state = cmcp.step(
        correlation,
        motion,
        state.cmcp_state,
        frame_valid=active,
    )
    proposal = extract_proposal_candidates(
        dense["proposal_score"],
        native,
        dense["native_logit"],
        cmcp.config,
    )
    candidate_valid = proposal["candidate_valid_mask"].clone()
    candidate_valid[:, 1:] &= active[:, None]
    zeros = torch.zeros(batch, 4, device=fmap.device, dtype=fmap.dtype)
    raw_token = build_cmcp_local_candidate_tokens(
        hidden_map=dense["hidden_map"],
        recurrent_input=dense["recurrent_input"],
        utility_logit=dense["utility_logit"],
        risk_logit=dense["risk_logit"],
        proposal_score=dense["proposal_score"],
        candidate_coords_xy_px=proposal["candidate_coords_xy_px"],
        candidate_scores=proposal["candidate_scores"],
        candidate_valid_mask=candidate_valid,
        native_visibility_probability=native_visibility_probability.to(fmap),
        native_confidence_probability=native_confidence_probability.to(fmap),
        native_joint_probability=(
            native_visibility_probability * native_confidence_probability
        ).to(fmap),
        previous_decision_summary=zeros,
        input_height=cmcp.config.input_height,
        input_width=cmcp.config.input_width,
    )
    normalized_token = normalize_static_tokens(raw_token, candidate_valid, normalization)
    dynamic_token = inject_dynamic_summary(
        normalized_token, state.decision_summary.to(normalized_token)
    )
    compared = comparator(
        dynamic_token,
        candidate_valid,
        proposal["candidate_coords_xy_px"],
    )
    summary = _update_summary(
        compared,
        proposal["candidate_coords_xy_px"],
        active,
        state.decision_summary.to(compared["selected_coord_xy_px"]),
    )

    active_feature = active[:, None]
    previous = torch.where(active_feature, current_native, state.previous_support)
    ema = float(cmcp.config.ema_alpha) * state.ema_support + (
        1.0 - float(cmcp.config.ema_alpha)
    ) * current_native
    ema = F.normalize(ema, dim=-1, eps=1.0e-12)
    ema = torch.where(active_feature, ema, state.ema_support)
    new_state = VariantCOnlineState(
        query_support=state.query_support,
        previous_support=previous,
        ema_support=ema,
        query_frames=state.query_frames,
        cmcp_state=cmcp_state,
        decision_summary=summary,
    )
    output = {
        **compared,
        "candidate_coords_xy_px": proposal["candidate_coords_xy_px"],
        "candidate_scores": proposal["candidate_scores"],
        "candidate_valid_mask": candidate_valid,
        "frame_valid": active,
        "decision_summary": summary,
        "pre_writeback_native_feature": current_native,
    }
    return output, new_state


def bounded_coordinate_writeback(
    native_coord_xy_px: torch.Tensor,
    selected_coord_xy_px: torch.Tensor,
    selected_candidate_index: torch.Tensor,
    eligible: torch.Tensor,
    config: BoundedWritebackConfig = BoundedWritebackConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded write coordinates, applied mask, and write norms."""
    if native_coord_xy_px.shape != selected_coord_xy_px.shape:
        raise ValueError("native/selected coordinate shape mismatch")
    if native_coord_xy_px.ndim != 2 or native_coord_xy_px.shape[-1] != 2:
        raise ValueError("coordinates must have shape (N,2)")
    if selected_candidate_index.shape != native_coord_xy_px.shape[:1]:
        raise ValueError("selected_candidate_index shape mismatch")
    if eligible.shape != selected_candidate_index.shape:
        raise ValueError("eligible shape mismatch")
    delta = selected_coord_xy_px - native_coord_xy_px
    norm = torch.linalg.vector_norm(delta, dim=-1)
    scale = torch.clamp(
        float(config.max_step_px) / norm.clamp_min(1.0e-12), max=1.0
    )
    bounded = native_coord_xy_px + delta * scale[:, None]
    bounded = torch.stack(
        [
            bounded[:, 0].clamp(0.0, float(config.input_width - 1)),
            bounded[:, 1].clamp(0.0, float(config.input_height - 1)),
        ],
        dim=-1,
    )
    applied = eligible.bool() & (selected_candidate_index > 0)
    write = torch.where(applied[:, None], bounded, native_coord_xy_px)
    write_norm = torch.linalg.vector_norm(write - native_coord_xy_px, dim=-1)
    return write, applied, write_norm


def overlap_write_eligible(
    frame_index: int,
    *,
    chunk_start: int,
    step: int,
    point_count: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Eligibility for the second half of the current 2*step online window."""
    if step <= 0 or point_count <= 0:
        raise ValueError("step and point_count must be positive")
    allowed = chunk_start + step <= int(frame_index) < chunk_start + 2 * step
    return torch.full((point_count,), allowed, dtype=torch.bool, device=device)
