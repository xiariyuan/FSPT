"""Risk-controlled long-occlusion re-detection for frozen point trackers.

This module is backbone-independent at the decision level. A backbone adapter
provides one feature map and its native coordinate/visibility/confidence for each
query. The module maintains causal appearance anchors, proposes global matches,
requires temporal confirmation, preserves candidate 0 as an exact native
fallback, jointly restores coordinate and visibility, and emits a bounded update
that is valid only for future tracker windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .routeD_multi_memory_proposal import (
    build_native_motion_prior,
    stable_spatial_topk_nms,
)

SAFE_REDETECTION_SCHEMA_VERSION = "routeD_safe_redetection_v0"


@dataclass(frozen=True)
class SafeRedetectionConfig:
    feature_dim: int = 128
    memory_slots: int = 4
    proposal_topk: int = 5
    proposal_hidden_dim: int = 32
    comparator_hidden_dim: int = 96
    comparator_heads: int = 4
    comparator_layers: int = 2
    input_height: int = 256
    input_width: int = 256
    nms_radius_cells: int = 1
    motion_sigma_cells: float = 2.0
    reliable_visibility_min: float = 0.80
    reliable_confidence_min: float = 0.80
    reliable_consistency_min: float = 0.70
    episodic_novelty_cosine_max: float = 0.95
    last_reliable_ema: float = 0.90
    confirmation_frames: int = 2
    confirmation_radius_px: float = 8.0
    max_future_writeback_step_px: float = 32.0
    risk_weight: float = 1.0
    preference_weight: float = 0.25
    initial_abstention_bias: float = 8.0
    initial_reappearance_bias: float = -8.0
    initial_writeback_bias: float = -8.0
    utility_thresholds_px: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    utility_weights: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.memory_slots < 3:
            raise ValueError("feature_dim must be positive and memory_slots >= 3")
        if self.proposal_topk <= 0 or self.confirmation_frames < 1:
            raise ValueError("proposal_topk and confirmation_frames must be positive")
        if len(self.utility_thresholds_px) != len(self.utility_weights):
            raise ValueError("utility threshold/weight length mismatch")
        if not 0.0 <= self.last_reliable_ema < 1.0:
            raise ValueError("last_reliable_ema must be in [0,1)")


@dataclass(frozen=True)
class AppearanceMemoryState:
    anchors: torch.Tensor  # B,M,D
    valid: torch.Tensor  # B,M
    reliability: torch.Tensor  # B,M
    frame_index: torch.Tensor  # B,M
    next_episodic_slot: torch.Tensor  # B


@dataclass(frozen=True)
class ConfirmationState:
    pending_coord_xy_px: torch.Tensor  # B,2
    pending_candidate_index: torch.Tensor  # B
    pending_count: torch.Tensor  # B
    pending_valid: torch.Tensor  # B


@dataclass(frozen=True)
class FutureWriteback:
    coordinate_xy_px: torch.Tensor
    visibility_probability: torch.Tensor
    confidence_probability: torch.Tensor
    allowed: torch.Tensor
    effective_from_frame: torch.Tensor


@dataclass(frozen=True)
class RedetectionStepState:
    memory: AppearanceMemoryState
    confirmation: ConfirmationState


def initialize_appearance_memory(
    query_feature: torch.Tensor,
    query_frame: torch.Tensor | int,
    config: SafeRedetectionConfig,
) -> AppearanceMemoryState:
    if query_feature.ndim != 2 or query_feature.shape[-1] != config.feature_dim:
        raise ValueError("query_feature must have shape (B,D)")
    batch = query_feature.shape[0]
    device = query_feature.device
    anchors = torch.zeros(
        batch, config.memory_slots, config.feature_dim,
        device=device, dtype=query_feature.dtype,
    )
    valid = torch.zeros(batch, config.memory_slots, dtype=torch.bool, device=device)
    reliability = torch.zeros(batch, config.memory_slots, device=device, dtype=query_feature.dtype)
    frame_index = torch.full(
        (batch, config.memory_slots), -1, device=device, dtype=torch.long
    )
    normalized = F.normalize(query_feature.float(), dim=-1, eps=1e-12).to(query_feature.dtype)
    anchors[:, 0] = normalized
    valid[:, 0] = True
    reliability[:, 0] = 1.0
    query_frame_tensor = torch.as_tensor(query_frame, device=device, dtype=torch.long)
    if query_frame_tensor.ndim == 0:
        query_frame_tensor = query_frame_tensor.expand(batch)
    if query_frame_tensor.shape != (batch,):
        raise ValueError("query_frame must be scalar or shape (B,)")
    frame_index[:, 0] = query_frame_tensor
    return AppearanceMemoryState(
        anchors=anchors,
        valid=valid,
        reliability=reliability,
        frame_index=frame_index,
        next_episodic_slot=torch.full((batch,), 2, device=device, dtype=torch.long),
    )


def initialize_confirmation_state(
    batch: int, *, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> ConfirmationState:
    return ConfirmationState(
        pending_coord_xy_px=torch.zeros(batch, 2, device=device, dtype=dtype),
        pending_candidate_index=torch.zeros(batch, device=device, dtype=torch.long),
        pending_count=torch.zeros(batch, device=device, dtype=torch.long),
        pending_valid=torch.zeros(batch, device=device, dtype=torch.bool),
    )


def _clone_memory(state: AppearanceMemoryState) -> AppearanceMemoryState:
    return AppearanceMemoryState(
        anchors=state.anchors.clone(),
        valid=state.valid.clone(),
        reliability=state.reliability.clone(),
        frame_index=state.frame_index.clone(),
        next_episodic_slot=state.next_episodic_slot.clone(),
    )


def update_appearance_memory(
    state: AppearanceMemoryState,
    observed_feature: torch.Tensor,
    *,
    frame_index: int,
    visibility_probability: torch.Tensor,
    confidence_probability: torch.Tensor,
    anchor_consistency: torch.Tensor,
    confirmed_recovery: torch.Tensor,
    config: SafeRedetectionConfig,
) -> AppearanceMemoryState:
    """Update last-reliable and episodic anchors without GT or future frames."""
    if observed_feature.shape != state.anchors[:, 0].shape:
        raise ValueError("observed_feature shape mismatch")
    batch = observed_feature.shape[0]
    for tensor in (visibility_probability, confidence_probability, anchor_consistency, confirmed_recovery):
        if tensor.shape != (batch,):
            raise ValueError("memory update controls must have shape (B,)")
    normalized = F.normalize(observed_feature.float(), dim=-1, eps=1e-12).to(
        observed_feature.dtype
    )
    reliable_native = (
        (visibility_probability >= config.reliable_visibility_min)
        & (confidence_probability >= config.reliable_confidence_min)
        & (anchor_consistency >= config.reliable_consistency_min)
    )
    allowed = reliable_native | confirmed_recovery.bool()
    output = _clone_memory(state)
    alpha = float(config.last_reliable_ema)
    for index in range(batch):
        if not bool(allowed[index]):
            continue
        if bool(output.valid[index, 1]):
            mixed = alpha * output.anchors[index, 1] + (1.0 - alpha) * normalized[index]
            output.anchors[index, 1] = F.normalize(mixed.float(), dim=-1, eps=1e-12).to(mixed.dtype)
        else:
            output.anchors[index, 1] = normalized[index]
            output.valid[index, 1] = True
        output.reliability[index, 1] = torch.where(
            confirmed_recovery[index],
            output.reliability.new_tensor(0.9),
            (visibility_probability[index] * confidence_probability[index]).clamp(0, 1),
        )
        output.frame_index[index, 1] = int(frame_index)

        valid_anchor = output.anchors[index, output.valid[index]]
        maximum_cosine = torch.max(valid_anchor @ normalized[index])
        if float(maximum_cosine.item()) > config.episodic_novelty_cosine_max:
            continue
        slot = int(output.next_episodic_slot[index].item())
        output.anchors[index, slot] = normalized[index]
        output.valid[index, slot] = True
        output.reliability[index, slot] = output.reliability[index, 1]
        output.frame_index[index, slot] = int(frame_index)
        next_slot = slot + 1
        if next_slot >= config.memory_slots:
            next_slot = 2
        output.next_episodic_slot[index] = next_slot
    return output


def build_multi_anchor_correlations(
    feature_map: torch.Tensor,
    memory: AppearanceMemoryState,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_map.ndim != 4 or feature_map.shape[1] != memory.anchors.shape[-1]:
        raise ValueError("feature_map must have shape (B,D,H,W)")
    if feature_map.shape[0] != memory.anchors.shape[0]:
        raise ValueError("batch mismatch")
    fmap = F.normalize(feature_map.float(), dim=1, eps=1e-12)
    anchors = F.normalize(memory.anchors.float(), dim=-1, eps=1e-12)
    correlation = torch.einsum("bmd,bdhw->bmhw", anchors, fmap)
    correlation = torch.where(
        memory.valid[:, :, None, None], correlation, torch.zeros_like(correlation)
    )
    masked = correlation.masked_fill(~memory.valid[:, :, None, None], float("-inf"))
    aggregate = masked.max(dim=1, keepdim=True).values
    return correlation, aggregate


def _normalized_grid(coords_xy_px: torch.Tensor, config: SafeRedetectionConfig) -> torch.Tensor:
    x = coords_xy_px[..., 0] / float(max(config.input_width - 1, 1))
    y = coords_xy_px[..., 1] / float(max(config.input_height - 1, 1))
    return torch.stack((2.0 * x - 1.0, 2.0 * y - 1.0), dim=-1)


def sample_batched_map(
    feature_map: torch.Tensor,
    coords_xy_px: torch.Tensor,
    config: SafeRedetectionConfig,
) -> torch.Tensor:
    """Sample BxCxHxW at BxKx2 input-raster coordinates, returning BxKxC."""
    if feature_map.ndim != 4 or coords_xy_px.ndim != 3:
        raise ValueError("expected feature_map BxCxHxW and coords BxKx2")
    grid = _normalized_grid(coords_xy_px, config).unsqueeze(2)
    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


class MultiAnchorProposalGenerator(nn.Module):
    """Global proposal score = strongest anchor match + learned residual."""

    def __init__(self, config: SafeRedetectionConfig) -> None:
        super().__init__()
        self.config = config
        input_channels = config.memory_slots + 1
        hidden = config.proposal_hidden_dim
        self.residual = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.visibility_evidence = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.visibility_evidence[-1].weight)
        nn.init.zeros_(self.visibility_evidence[-1].bias)

    def forward(
        self,
        feature_map: torch.Tensor,
        memory: AppearanceMemoryState,
        native_coord_xy_px: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        correlation, aggregate = build_multi_anchor_correlations(feature_map, memory)
        motion = build_native_motion_prior(
            native_coord_xy_px,
            feature_height=feature_map.shape[-2],
            feature_width=feature_map.shape[-1],
            input_height=self.config.input_height,
            input_width=self.config.input_width,
            sigma_cells=self.config.motion_sigma_cells,
        )
        evidence = torch.cat((correlation, motion), dim=1)
        proposal_score = aggregate + self.residual(evidence)
        visibility_evidence = self.visibility_evidence(evidence)
        indices, peak_scores, peak_valid = stable_spatial_topk_nms(
            proposal_score[:, 0],
            topk=self.config.proposal_topk,
            radius_cells=self.config.nms_radius_cells,
        )
        width = proposal_score.shape[-1]
        height = proposal_score.shape[-2]
        peak_y = torch.div(indices, width, rounding_mode="floor")
        peak_x = indices % width
        peak_coords = torch.stack(
            (
                peak_x.to(proposal_score.dtype) / float(max(width - 1, 1)) * (self.config.input_width - 1),
                peak_y.to(proposal_score.dtype) / float(max(height - 1, 1)) * (self.config.input_height - 1),
            ),
            dim=-1,
        )
        candidate_coords = torch.cat((native_coord_xy_px[:, None], peak_coords), dim=1)
        candidate_valid = torch.cat(
            (
                torch.ones(native_coord_xy_px.shape[0], 1, dtype=torch.bool, device=feature_map.device),
                peak_valid,
            ),
            dim=1,
        )
        candidate_scores = torch.cat(
            (
                sample_batched_map(proposal_score, native_coord_xy_px[:, None], self.config)[..., 0],
                peak_scores,
            ),
            dim=1,
        )
        sampled_corr = sample_batched_map(correlation, candidate_coords, self.config)
        candidate_source = sampled_corr.argmax(dim=-1)
        return {
            "correlation_maps": correlation,
            "aggregate_correlation": aggregate,
            "proposal_score": proposal_score,
            "visibility_evidence_map": visibility_evidence,
            "candidate_coords_xy_px": candidate_coords,
            "candidate_valid_mask": candidate_valid,
            "candidate_scores": candidate_scores,
            "candidate_source_index": candidate_source,
            "candidate_anchor_correlations": sampled_corr,
            "candidate_features": sample_batched_map(
                F.normalize(feature_map.float(), dim=1, eps=1e-12),
                candidate_coords,
                self.config,
            ),
            "candidate_visibility_evidence": sample_batched_map(
                visibility_evidence, candidate_coords, self.config
            )[..., 0],
        }


def redetection_token_dim(config: SafeRedetectionConfig) -> int:
    # feature + anchor correlations + reliability + valid flags + source one-hot
    # + score + visibility evidence + displacement(3) + rank + native stats(2)
    # + occlusion age + confirmation summary(2)
    return config.feature_dim + 4 * config.memory_slots + 11


def build_redetection_tokens(
    proposal: Mapping[str, torch.Tensor],
    memory: AppearanceMemoryState,
    native_visibility_probability: torch.Tensor,
    native_confidence_probability: torch.Tensor,
    occlusion_age: torch.Tensor,
    confirmation: ConfirmationState,
    config: SafeRedetectionConfig,
) -> torch.Tensor:
    coords = proposal["candidate_coords_xy_px"]
    batch, candidates = coords.shape[:2]
    native = coords[:, :1]
    displacement = coords - native
    dx = displacement[..., 0] / float(max(config.input_width - 1, 1))
    dy = displacement[..., 1] / float(max(config.input_height - 1, 1))
    displacement_token = torch.stack((dx, dy, torch.sqrt(dx.square() + dy.square())), dim=-1)
    rank = torch.arange(candidates, device=coords.device, dtype=coords.dtype)
    rank = (rank / float(max(candidates - 1, 1))).view(1, candidates, 1).expand(batch, -1, -1)
    reliability = memory.reliability[:, None].expand(-1, candidates, -1)
    memory_valid = memory.valid[:, None].expand(-1, candidates, -1).to(coords.dtype)
    source = F.one_hot(
        proposal["candidate_source_index"].clamp(0, config.memory_slots - 1),
        num_classes=config.memory_slots,
    ).to(coords.dtype)
    native_stats = torch.stack(
        (native_visibility_probability, native_confidence_probability), dim=-1
    )[:, None].expand(-1, candidates, -1)
    age = torch.log1p(occlusion_age.float()).clamp(max=8.0) / 8.0
    age = age[:, None, None].expand(-1, candidates, -1)
    pending_distance = torch.linalg.norm(
        coords - confirmation.pending_coord_xy_px[:, None], dim=-1
    ) / float(max(config.input_width, config.input_height))
    confirmation_summary = torch.stack(
        (
            confirmation.pending_count.float() / float(config.confirmation_frames),
            confirmation.pending_valid.float(),
        ),
        dim=-1,
    )[:, None].expand(-1, candidates, -1)
    token = torch.cat(
        (
            proposal["candidate_features"].detach(),
            proposal["candidate_anchor_correlations"].detach(),
            reliability,
            memory_valid,
            source,
            proposal["candidate_scores"].detach().unsqueeze(-1),
            proposal["candidate_visibility_evidence"].detach().unsqueeze(-1),
            displacement_token,
            rank,
            native_stats,
            age,
            confirmation_summary + torch.stack((pending_distance, torch.zeros_like(pending_distance)), dim=-1),
        ),
        dim=-1,
    )
    if token.shape[-1] != redetection_token_dim(config):
        raise RuntimeError(f"redetection token dimension drift: {token.shape[-1]}")
    return torch.where(
        proposal["candidate_valid_mask"].unsqueeze(-1), token, torch.zeros_like(token)
    )


class SafeRedetectionComparator(nn.Module):
    def __init__(self, config: SafeRedetectionConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.comparator_hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(redetection_token_dim(config), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.comparator_heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set = nn.TransformerEncoder(layer, num_layers=config.comparator_layers)
        count = len(config.utility_thresholds_px)
        self.utility_head = nn.Linear(hidden, count)
        self.risk_head = nn.Linear(hidden, 1)
        self.preference_head = nn.Linear(hidden, 1)
        self.reappearance_head = nn.Linear(hidden, 1)
        self.visibility_head = nn.Linear(hidden, 1)
        self.writeback_head = nn.Linear(hidden, 1)
        self.abstention_head = nn.Linear(hidden, 1)
        self._initialize_native_safe()

    def _initialize_native_safe(self) -> None:
        for head in (
            self.utility_head,
            self.risk_head,
            self.preference_head,
            self.visibility_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head, bias in (
            (self.reappearance_head, self.config.initial_reappearance_bias),
            (self.writeback_head, self.config.initial_writeback_bias),
            (self.abstention_head, self.config.initial_abstention_bias),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, bias)

    def forward(
        self,
        tokens: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        candidate_coords_xy_px: torch.Tensor,
        native_visibility_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(tokens)
        encoded = self.candidate_set(encoded, src_key_padding_mask=~candidate_valid_mask)
        encoded = torch.where(candidate_valid_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        raw_utility = self.utility_head(encoded)
        base = raw_utility[..., :1]
        increments = F.softplus(raw_utility[..., 1:])
        rows = [base]
        running = base
        for index in range(increments.shape[-1]):
            running = running + increments[..., index : index + 1]
            rows.append(running)
        utility_logits = torch.cat(rows, dim=-1)
        utility_probability = torch.sigmoid(utility_logits)
        weights = torch.tensor(
            self.config.utility_weights, device=tokens.device, dtype=tokens.dtype
        )
        weights = weights / weights.sum()
        weighted_utility = (utility_probability * weights).sum(dim=-1)
        risk_probability = torch.sigmoid(self.risk_head(encoded).squeeze(-1))
        preference_logit = self.preference_head(encoded).squeeze(-1)
        score = weighted_utility - self.config.risk_weight * risk_probability
        score = score + self.config.preference_weight * preference_logit
        score = score.masked_fill(~candidate_valid_mask, float("-inf"))
        proposed_index = score.argmax(dim=-1)
        batch = tokens.shape[0]
        gather = proposed_index[:, None]
        reappearance_probability = torch.sigmoid(self.reappearance_head(encoded).squeeze(-1))
        candidate_visibility_probability = torch.sigmoid(self.visibility_head(encoded).squeeze(-1))
        writeback_probability = torch.sigmoid(self.writeback_head(encoded).squeeze(-1))
        abstention_probability = torch.sigmoid(self.abstention_head(encoded[:, 0]).squeeze(-1))
        proposed_risk = risk_probability.gather(1, gather).squeeze(1)
        proposed_reappearance = reappearance_probability.gather(1, gather).squeeze(1)
        proposal_ready = (
            (proposed_index > 0)
            & (abstention_probability < 0.5)
            & (proposed_risk <= 0.5)
            & (proposed_reappearance >= 0.5)
        )
        native_safe_index = torch.where(proposal_ready, proposed_index, torch.zeros_like(proposed_index))
        safe_coord = candidate_coords_xy_px.gather(
            1, native_safe_index[:, None, None].expand(batch, 1, 2)
        ).squeeze(1)
        proposed_visibility = candidate_visibility_probability.gather(1, gather).squeeze(1)
        safe_visibility = torch.where(
            proposal_ready, proposed_visibility, native_visibility_probability
        )
        return {
            "utility_logits": utility_logits,
            "utility_probability": utility_probability,
            "risk_probability": risk_probability,
            "preference_logit": preference_logit,
            "candidate_score": score,
            "reappearance_probability": reappearance_probability,
            "candidate_visibility_probability": candidate_visibility_probability,
            "writeback_probability": writeback_probability,
            "abstention_probability": abstention_probability,
            "proposed_candidate_index": proposed_index,
            "proposal_ready": proposal_ready,
            "native_safe_candidate_index": native_safe_index,
            "native_safe_coord_xy_px": safe_coord,
            "native_safe_visibility_probability": safe_visibility,
        }


def update_confirmation(
    state: ConfirmationState,
    proposed_index: torch.Tensor,
    proposed_coord_xy_px: torch.Tensor,
    proposal_ready: torch.Tensor,
    config: SafeRedetectionConfig,
) -> tuple[ConfirmationState, torch.Tensor]:
    same = (
        state.pending_valid
        & proposal_ready
        & (state.pending_candidate_index == proposed_index)
        & (
            torch.linalg.norm(state.pending_coord_xy_px - proposed_coord_xy_px, dim=-1)
            <= config.confirmation_radius_px
        )
    )
    count = torch.where(
        proposal_ready,
        torch.where(same, state.pending_count + 1, torch.ones_like(state.pending_count)),
        torch.zeros_like(state.pending_count),
    )
    valid = proposal_ready.bool()
    coord = torch.where(valid[:, None], proposed_coord_xy_px, torch.zeros_like(proposed_coord_xy_px))
    index = torch.where(valid, proposed_index, torch.zeros_like(proposed_index))
    output = ConfirmationState(coord, index, count, valid)
    confirmed = valid & (count >= config.confirmation_frames)
    return output, confirmed


def make_future_writeback(
    *,
    frame_index: int,
    native_coord_xy_px: torch.Tensor,
    selected_coord_xy_px: torch.Tensor,
    selected_visibility_probability: torch.Tensor,
    native_confidence_probability: torch.Tensor,
    selected_writeback_probability: torch.Tensor,
    confirmed: torch.Tensor,
    config: SafeRedetectionConfig,
) -> FutureWriteback:
    delta = selected_coord_xy_px - native_coord_xy_px
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(
        float(config.max_future_writeback_step_px) / norm.clamp_min(1e-12), max=1.0
    )
    bounded = native_coord_xy_px + delta * scale
    allowed = confirmed & (selected_writeback_probability >= 0.5)
    effective = torch.full(
        (native_coord_xy_px.shape[0],),
        int(frame_index) + 1,
        dtype=torch.long,
        device=native_coord_xy_px.device,
    )
    return FutureWriteback(
        coordinate_xy_px=torch.where(allowed[:, None], bounded, native_coord_xy_px),
        visibility_probability=torch.where(
            allowed, selected_visibility_probability, selected_visibility_probability.new_zeros(())
        ),
        confidence_probability=torch.where(
            allowed,
            torch.maximum(native_confidence_probability, selected_writeback_probability),
            native_confidence_probability.new_zeros(()),
        ),
        allowed=allowed,
        effective_from_frame=effective,
    )


def apply_future_writeback(
    native_coord_xy_px: torch.Tensor,
    native_visibility_probability: torch.Tensor,
    native_confidence_probability: torch.Tensor,
    writeback: FutureWriteback,
    *,
    target_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply writeback only when evaluating a strictly future frame."""
    eligible = writeback.allowed & (int(target_frame) >= writeback.effective_from_frame)
    coord = torch.where(eligible[:, None], writeback.coordinate_xy_px, native_coord_xy_px)
    visibility = torch.where(
        eligible, writeback.visibility_probability, native_visibility_probability
    )
    confidence = torch.where(
        eligible, writeback.confidence_probability, native_confidence_probability
    )
    return coord, visibility, confidence


class SafeRedetectionModel(nn.Module):
    """One causal step of the full Route-D safe re-detection idea."""

    def __init__(self, config: SafeRedetectionConfig = SafeRedetectionConfig()) -> None:
        super().__init__()
        self.config = config
        self.proposal = MultiAnchorProposalGenerator(config)
        self.comparator = SafeRedetectionComparator(config)

    def step(
        self,
        *,
        frame_index: int,
        feature_map: torch.Tensor,
        native_coord_xy_px: torch.Tensor,
        native_visibility_probability: torch.Tensor,
        native_confidence_probability: torch.Tensor,
        occlusion_age: torch.Tensor,
        state: RedetectionStepState,
    ) -> tuple[dict[str, Any], RedetectionStepState]:
        proposal = self.proposal(feature_map, state.memory, native_coord_xy_px)
        tokens = build_redetection_tokens(
            proposal,
            state.memory,
            native_visibility_probability,
            native_confidence_probability,
            occlusion_age,
            state.confirmation,
            self.config,
        )
        compared = self.comparator(
            tokens,
            proposal["candidate_valid_mask"],
            proposal["candidate_coords_xy_px"],
            native_visibility_probability,
        )
        proposed_index = compared["proposed_candidate_index"]
        proposed_coord = proposal["candidate_coords_xy_px"].gather(
            1, proposed_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        confirmation, confirmed = update_confirmation(
            state.confirmation,
            proposed_index,
            proposed_coord,
            compared["proposal_ready"],
            self.config,
        )
        selected_index = torch.where(confirmed, proposed_index, torch.zeros_like(proposed_index))
        selected_coord = proposal["candidate_coords_xy_px"].gather(
            1, selected_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        candidate_visibility = compared["candidate_visibility_probability"].gather(
            1, proposed_index[:, None]
        ).squeeze(1)
        selected_visibility = torch.where(
            confirmed, candidate_visibility, native_visibility_probability
        )
        selected_writeback = compared["writeback_probability"].gather(
            1, proposed_index[:, None]
        ).squeeze(1)
        writeback = make_future_writeback(
            frame_index=frame_index,
            native_coord_xy_px=native_coord_xy_px,
            selected_coord_xy_px=selected_coord,
            selected_visibility_probability=selected_visibility,
            native_confidence_probability=native_confidence_probability,
            selected_writeback_probability=selected_writeback,
            confirmed=confirmed,
            config=self.config,
        )

        normalized_map = F.normalize(feature_map.float(), dim=1, eps=1e-12)
        native_feature = sample_batched_map(
            normalized_map, native_coord_xy_px[:, None], self.config
        )[:, 0]
        selected_feature = sample_batched_map(
            normalized_map, selected_coord[:, None], self.config
        )[:, 0]
        native_corr = sample_batched_map(
            proposal["correlation_maps"], native_coord_xy_px[:, None], self.config
        )[:, 0]
        consistency = native_corr.masked_fill(~state.memory.valid, float("-inf")).max(dim=-1).values
        observed = torch.where(confirmed[:, None], selected_feature, native_feature)
        updated_memory = update_appearance_memory(
            state.memory,
            observed,
            frame_index=frame_index,
            visibility_probability=torch.where(
                confirmed, selected_visibility, native_visibility_probability
            ),
            confidence_probability=torch.where(
                confirmed,
                torch.maximum(native_confidence_probability, selected_writeback),
                native_confidence_probability,
            ),
            anchor_consistency=torch.where(confirmed, torch.ones_like(consistency), consistency),
            confirmed_recovery=confirmed,
            config=self.config,
        )
        output: dict[str, Any] = {
            **proposal,
            **compared,
            "tokens": tokens,
            "confirmed_recovery": confirmed,
            "selected_candidate_index": selected_index,
            "selected_coord_xy_px": selected_coord,
            "selected_visibility_probability": selected_visibility,
            "future_writeback": writeback,
            "native_anchor_consistency": consistency,
        }
        return output, RedetectionStepState(updated_memory, confirmation)


def build_redetection_targets(
    gt_visible: torch.Tensor,
    query_frame: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build causal event labels for coordinate, visibility and reappearance loss."""
    if gt_visible.ndim != 2:
        raise ValueError("gt_visible must have shape (B,T)")
    batch, frames = gt_visible.shape
    query = query_frame.round().long().clamp(0, frames - 1)
    frame = torch.arange(frames, device=gt_visible.device)[None]
    active = frame >= query[:, None]
    visible = gt_visible.bool() & active
    duration = torch.zeros(batch, frames, dtype=torch.long, device=gt_visible.device)
    for index in range(1, frames):
        duration[:, index] = torch.where(
            ~gt_visible[:, index - 1], duration[:, index - 1] + 1, 0
        )
    reappearance = torch.zeros_like(visible)
    reappearance[:, 1:] = visible[:, 1:] & ~gt_visible[:, :-1]
    eligible = torch.zeros_like(visible)
    for row in range(batch):
        maximum = -1
        for index in torch.where(reappearance[row])[0].tolist():
            current = int(duration[row, index].item())
            if current > maximum:
                eligible[row, index] = True
                maximum = current
    return {
        "active_mask": active,
        "visible_mask": visible,
        "occluded_mask": active & ~gt_visible.bool(),
        "reappearance_mask": reappearance,
        "eligible_reappearance_mask": eligible,
        "occlusion_duration": duration,
        "long_occlusion_reappearance_mask": eligible & (duration >= 16),
    }


def safe_redetection_dense_proposal_loss(
    proposal_score: torch.Tensor,
    visibility_evidence_map: torch.Tensor,
    gt_coord_xy_px: torch.Tensor,
    gt_visible: torch.Tensor,
    supervised: torch.Tensor,
    config: SafeRedetectionConfig,
    *,
    target_sigma_cells: float = 1.5,
) -> dict[str, torch.Tensor]:
    """Dense differentiable supervision before hard top-K discretization."""
    if proposal_score.shape != visibility_evidence_map.shape or proposal_score.ndim != 4:
        raise ValueError("proposal and visibility maps must share shape (B,1,H,W)")
    batch, _, height, width = proposal_score.shape
    if gt_coord_xy_px.shape != (batch, 2):
        raise ValueError("gt_coord_xy_px must have shape (B,2)")
    x = gt_coord_xy_px[:, 0] / float(max(config.input_width - 1, 1)) * float(width - 1)
    y = gt_coord_xy_px[:, 1] / float(max(config.input_height - 1, 1)) * float(height - 1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=proposal_score.device, dtype=proposal_score.dtype),
        torch.arange(width, device=proposal_score.device, dtype=proposal_score.dtype),
        indexing="ij",
    )
    distance2 = (grid_x[None] - x[:, None, None]).square() + (
        grid_y[None] - y[:, None, None]
    ).square()
    heatmap = torch.exp(-0.5 * distance2 / float(target_sigma_cells**2))
    heatmap = heatmap * gt_visible[:, None, None].to(heatmap.dtype)
    active = supervised.float().view(batch, 1, 1)
    proposal_loss = F.binary_cross_entropy_with_logits(
        proposal_score[:, 0], heatmap, reduction="none"
    )
    visibility_target = gt_visible.float().view(batch, 1, 1).expand(-1, height, width)
    visibility_loss = F.binary_cross_entropy_with_logits(
        visibility_evidence_map[:, 0], visibility_target, reduction="none"
    )
    denominator = (active.sum() * height * width).clamp_min(1.0)
    proposal_loss = (proposal_loss * active).sum() / denominator
    visibility_loss = (visibility_loss * active).sum() / denominator
    return {
        "loss": proposal_loss + visibility_loss,
        "proposal_heatmap_loss": proposal_loss,
        "dense_visibility_loss": visibility_loss,
    }


@dataclass(frozen=True)
class SafeRedetectionLossConfig:
    utility_weight: float = 1.0
    risk_weight: float = 1.0
    preference_weight: float = 0.5
    visibility_weight: float = 1.0
    reappearance_weight: float = 2.0
    abstention_weight: float = 0.5
    writeback_weight: float = 0.5
    false_reacquisition_weight: float = 1.0


def safe_redetection_frame_loss(
    output: Mapping[str, torch.Tensor],
    candidate_coords_xy_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    gt_coord_xy_px: torch.Tensor,
    gt_visible: torch.Tensor,
    reappearance_target: torch.Tensor,
    supervised: torch.Tensor,
    config: SafeRedetectionConfig,
    loss_config: SafeRedetectionLossConfig = SafeRedetectionLossConfig(),
) -> dict[str, torch.Tensor]:
    error = torch.linalg.norm(
        candidate_coords_xy_px - gt_coord_xy_px[:, None], dim=-1
    )
    thresholds = torch.tensor(
        config.utility_thresholds_px, device=error.device, dtype=error.dtype
    )
    utility_target = (error[..., None] <= thresholds).to(error.dtype)
    utility_loss = F.binary_cross_entropy_with_logits(
        output["utility_logits"], utility_target, reduction="none"
    ).mean(dim=-1)
    utility_weights = torch.tensor(
        config.utility_weights, device=error.device, dtype=error.dtype
    )
    utility_weights = utility_weights / utility_weights.sum()
    scalar_utility = (utility_target * utility_weights).sum(dim=-1)
    native_utility = scalar_utility[:, :1]
    harmful_target = (scalar_utility + 1e-6 < native_utility).to(error.dtype)
    risk_loss = F.binary_cross_entropy(
        output["risk_probability"], harmful_target, reduction="none"
    )
    best = scalar_utility.masked_fill(~candidate_valid_mask, float("-inf")).argmax(dim=-1)
    preference_loss = F.cross_entropy(
        output["candidate_score"], best, reduction="none"
    )
    candidate_visible_target = (
        gt_visible[:, None] & (error <= config.utility_thresholds_px[-1])
    ).to(error.dtype)
    visibility_loss = F.binary_cross_entropy(
        output["candidate_visibility_probability"],
        candidate_visible_target,
        reduction="none",
    )
    better_than_native = scalar_utility > native_utility + 1e-6
    reappearance_candidate_target = (
        reappearance_target[:, None] & better_than_native & candidate_valid_mask
    ).to(error.dtype)
    reappearance_loss = F.binary_cross_entropy(
        output["reappearance_probability"],
        reappearance_candidate_target,
        reduction="none",
    )
    any_better = better_than_native.any(dim=-1)
    abstention_target = (~any_better).to(error.dtype)
    abstention_loss = F.binary_cross_entropy(
        output["abstention_probability"], abstention_target, reduction="none"
    )
    writeback_target = (
        reappearance_target[:, None]
        & better_than_native
        & candidate_visible_target.bool()
    ).to(error.dtype)
    writeback_loss = F.binary_cross_entropy(
        output["writeback_probability"], writeback_target, reduction="none"
    )
    false_reacquisition = torch.where(
        gt_visible[:, None],
        torch.zeros_like(output["candidate_visibility_probability"]),
        output["candidate_visibility_probability"],
    )
    mask = supervised.float()
    candidate_mask = candidate_valid_mask.float() * mask[:, None]
    visible_candidate_mask = candidate_mask * gt_visible.float()[:, None]
    candidate_denominator = candidate_mask.sum().clamp_min(1.0)
    visible_candidate_denominator = visible_candidate_mask.sum().clamp_min(1.0)
    row_denominator = mask.sum().clamp_min(1.0)
    visible_row_mask = mask * gt_visible.float()
    visible_row_denominator = visible_row_mask.sum().clamp_min(1.0)
    utility_value = (utility_loss * visible_candidate_mask).sum() / visible_candidate_denominator
    risk_value = (risk_loss * visible_candidate_mask).sum() / visible_candidate_denominator
    preference_value = (preference_loss * visible_row_mask).sum() / visible_row_denominator
    visibility_value = (visibility_loss * candidate_mask).sum() / candidate_denominator
    reappearance_value = (reappearance_loss * candidate_mask).sum() / candidate_denominator
    abstention_value = (abstention_loss * mask).sum() / row_denominator
    writeback_value = (writeback_loss * candidate_mask).sum() / candidate_denominator
    false_reacquisition_value = (false_reacquisition * candidate_mask).sum() / candidate_denominator
    total = (
        loss_config.utility_weight * utility_value
        + loss_config.risk_weight * risk_value
        + loss_config.preference_weight * preference_value
        + loss_config.visibility_weight * visibility_value
        + loss_config.reappearance_weight * reappearance_value
        + loss_config.abstention_weight * abstention_value
        + loss_config.writeback_weight * writeback_value
        + loss_config.false_reacquisition_weight * false_reacquisition_value
    )
    return {
        "loss": total,
        "utility_loss": utility_value,
        "risk_loss": risk_value,
        "preference_loss": preference_value,
        "visibility_loss": visibility_value,
        "reappearance_loss": reappearance_value,
        "abstention_loss": abstention_value,
        "writeback_loss": writeback_value,
        "false_reacquisition_loss": false_reacquisition_value,
    }
