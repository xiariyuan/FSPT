"""Causal multi-memory dense proposal generation for Route-D CMCP.

The frozen CoTracker3 backbone supplies normalized feature maps and the native
trajectory.  This module learns before candidate discretization: it fuses query,
previous-native, and causal-EMA correlation fields with a native motion prior,
then extracts native plus stable spatial proposal peaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from .cotracker3_stage0_adapter import sample_feature_at_xy


CMCP_SCHEMA_VERSION = "routeD_cmcp_multi_memory_proposal_v0"
CMCP_MEMORY_COUNT = 3
CMCP_PREVIOUS_EVIDENCE_FRAMES = 2
CMCP_PAIRWISE_DIFFERENCE_COUNT = 3


@dataclass(frozen=True)
class CMCPConfig:
    feature_dim: int = 128
    hidden_channels: int = 64
    proposal_topk: int = 5
    nms_radius_cells: int = 1
    ema_alpha: float = 0.9
    motion_sigma_cells: float = 2.0
    risk_weight: float = 0.75
    native_logit_bias: float = 2.0
    input_height: int = 256
    input_width: int = 256

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.hidden_channels <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if self.proposal_topk <= 0:
            raise ValueError("proposal_topk must be positive")
        if self.nms_radius_cells < 0:
            raise ValueError("nms_radius_cells must be non-negative")
        if not 0.0 <= self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in [0,1)")
        if self.motion_sigma_cells <= 0.0:
            raise ValueError("motion_sigma_cells must be positive")
        if self.input_height <= 1 or self.input_width <= 1:
            raise ValueError("input raster must exceed one pixel")


class CMCPState(NamedTuple):
    hidden: torch.Tensor
    previous_evidence: torch.Tensor


class ConvGRUCell(nn.Module):
    """Minimal spatial GRU cell with same-resolution hidden state."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        combined = input_channels + hidden_channels
        self.gates = nn.Conv2d(combined, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(combined, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([value, hidden], dim=1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([value, reset * hidden], dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class CausalMultiMemoryProposalGenerator(nn.Module):
    """Causal dense proposal head with exact native-safe initialization."""

    def __init__(self, config: CMCPConfig = CMCPConfig()) -> None:
        super().__init__()
        self.config = config
        input_channels = (
            CMCP_MEMORY_COUNT
            + CMCP_PAIRWISE_DIFFERENCE_COUNT
            + 1
            + CMCP_PREVIOUS_EVIDENCE_FRAMES
        )
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, config.hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.recurrent = ConvGRUCell(config.hidden_channels, config.hidden_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(config.hidden_channels, config.hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.utility_head = nn.Conv2d(config.hidden_channels, 1, 1)
        self.risk_head = nn.Conv2d(config.hidden_channels, 1, 1)
        self.native_head = nn.Linear(config.hidden_channels, 1)
        self._initialize_native_safe()

    def _initialize_native_safe(self) -> None:
        for head in (self.utility_head, self.risk_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.native_head.weight)
        nn.init.constant_(self.native_head.bias, self.config.native_logit_bias)

    def initial_state(
        self,
        batch_size: int,
        feature_height: int,
        feature_width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CMCPState:
        hidden = torch.zeros(
            batch_size,
            self.config.hidden_channels,
            feature_height,
            feature_width,
            device=device,
            dtype=dtype,
        )
        previous = torch.zeros(
            batch_size,
            CMCP_PREVIOUS_EVIDENCE_FRAMES,
            feature_height,
            feature_width,
            device=device,
            dtype=dtype,
        )
        return CMCPState(hidden, previous)

    def step(
        self,
        correlation_maps: torch.Tensor,
        motion_prior: torch.Tensor,
        state: CMCPState | None = None,
        *,
        frame_valid: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], CMCPState]:
        if correlation_maps.ndim != 4 or correlation_maps.shape[1] != 3:
            raise ValueError("correlation_maps must have shape (B,3,H,W)")
        if motion_prior.shape != correlation_maps.shape[:1] + (1,) + correlation_maps.shape[2:]:
            raise ValueError("motion_prior must have shape (B,1,H,W)")
        batch, _, height, width = correlation_maps.shape
        if state is None:
            state = self.initial_state(
                batch,
                height,
                width,
                device=correlation_maps.device,
                dtype=correlation_maps.dtype,
            )
        if state.hidden.shape != (
            batch,
            self.config.hidden_channels,
            height,
            width,
        ):
            raise ValueError("hidden-state shape mismatch")
        if state.previous_evidence.shape != (
            batch,
            CMCP_PREVIOUS_EVIDENCE_FRAMES,
            height,
            width,
        ):
            raise ValueError("previous-evidence shape mismatch")
        if frame_valid is None:
            frame_valid = torch.ones(batch, dtype=torch.bool, device=correlation_maps.device)
        if frame_valid.shape != (batch,):
            raise ValueError("frame_valid must have shape (B,)")

        pairwise_differences = torch.stack(
            [
                correlation_maps[:, 0] - correlation_maps[:, 1],
                correlation_maps[:, 0] - correlation_maps[:, 2],
                correlation_maps[:, 1] - correlation_maps[:, 2],
            ],
            dim=1,
        )
        recurrent_input = torch.cat(
            [
                correlation_maps,
                pairwise_differences,
                motion_prior,
                state.previous_evidence,
            ],
            dim=1,
        )
        projected = self.input_projection(recurrent_input)
        candidate_hidden = self.recurrent(projected, state.hidden)
        valid4 = frame_valid[:, None, None, None]
        hidden = torch.where(valid4, candidate_hidden, state.hidden)
        decoded = self.decoder(hidden)
        utility_logit = self.utility_head(decoded)
        risk_logit = self.risk_head(decoded)
        proposal_score = utility_logit - float(self.config.risk_weight) * torch.sigmoid(
            risk_logit
        )
        pooled = hidden.mean(dim=(-2, -1))
        native_logit = self.native_head(pooled).squeeze(-1)

        zero_map = torch.zeros_like(proposal_score)
        utility_logit = torch.where(valid4, utility_logit, zero_map)
        risk_logit = torch.where(valid4, risk_logit, zero_map)
        proposal_score = torch.where(valid4, proposal_score, zero_map)
        native_logit = torch.where(
            frame_valid,
            native_logit,
            torch.full_like(native_logit, float(self.config.native_logit_bias)),
        )
        shifted = torch.cat(
            [state.previous_evidence[:, 1:], proposal_score], dim=1
        )
        previous = torch.where(valid4, shifted, state.previous_evidence)
        output = {
            "utility_logit": utility_logit,
            "risk_logit": risk_logit,
            "proposal_score": proposal_score,
            "native_logit": native_logit,
            # Read-only local evidence for the frozen P0h comparator. These
            # additions do not change the CMCP state dict or proposal outputs.
            "hidden_map": hidden,
            "recurrent_input": recurrent_input,
        }
        return output, CMCPState(hidden, previous)

    def forward_sequence(
        self,
        correlation_maps: torch.Tensor,
        motion_prior: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if correlation_maps.ndim != 5 or correlation_maps.shape[2] != 3:
            raise ValueError("correlation_maps must have shape (B,T,3,H,W)")
        if motion_prior.shape != correlation_maps.shape[:2] + (1,) + correlation_maps.shape[3:]:
            raise ValueError("motion_prior must have shape (B,T,1,H,W)")
        if frame_valid.shape != correlation_maps.shape[:2]:
            raise ValueError("frame_valid must have shape (B,T)")
        state = None
        collected: dict[str, list[torch.Tensor]] = {
            "utility_logit": [],
            "risk_logit": [],
            "proposal_score": [],
            "native_logit": [],
        }
        for frame_index in range(correlation_maps.shape[1]):
            output, state = self.step(
                correlation_maps[:, frame_index],
                motion_prior[:, frame_index],
                state,
                frame_valid=frame_valid[:, frame_index],
            )
            for key in collected:
                collected[key].append(output[key])
        return {
            key: torch.stack(values, dim=1) for key, values in collected.items()
        }


def _feature_xy_from_input_xy(
    xy_px: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    input_height: int,
    input_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = xy_px[..., 0] / float(max(input_width - 1, 1)) * float(
        max(feature_width - 1, 1)
    )
    y = xy_px[..., 1] / float(max(input_height - 1, 1)) * float(
        max(feature_height - 1, 1)
    )
    return x, y


def build_native_motion_prior(
    native_xy_px: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    input_height: int,
    input_width: int,
    sigma_cells: float,
) -> torch.Tensor:
    if native_xy_px.ndim != 2 or native_xy_px.shape[-1] != 2:
        raise ValueError("native_xy_px must have shape (N,2)")
    x, y = _feature_xy_from_input_xy(
        native_xy_px,
        feature_height=feature_height,
        feature_width=feature_width,
        input_height=input_height,
        input_width=input_width,
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feature_height, device=native_xy_px.device, dtype=native_xy_px.dtype),
        torch.arange(feature_width, device=native_xy_px.device, dtype=native_xy_px.dtype),
        indexing="ij",
    )
    distance2 = (
        (grid_x[None] - x[:, None, None]).square()
        + (grid_y[None] - y[:, None, None]).square()
    )
    return torch.exp(-0.5 * distance2 / float(sigma_cells**2)).unsqueeze(1)


def build_causal_multi_memory_correlations(
    fmaps: torch.Tensor,
    native_coords_xy_px: torch.Tensor,
    query_points_tyx: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    ema_alpha: float = 0.9,
    motion_sigma_cells: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build query/previous/EMA correlation fields without future information."""
    if fmaps.ndim != 4:
        raise ValueError("fmaps must have shape (T,D,H,W)")
    if native_coords_xy_px.ndim != 3 or native_coords_xy_px.shape[-1] != 2:
        raise ValueError("native_coords_xy_px must have shape (N,T,2)")
    if query_points_tyx.shape != (native_coords_xy_px.shape[0], 3):
        raise ValueError("query_points_tyx shape mismatch")
    if native_coords_xy_px.shape[1] != fmaps.shape[0]:
        raise ValueError("frame count mismatch")
    if not 0.0 <= ema_alpha < 1.0:
        raise ValueError("ema_alpha must be in [0,1)")

    fmaps = F.normalize(fmaps.float(), dim=1, eps=1.0e-12)
    native = native_coords_xy_px.to(device=fmaps.device, dtype=fmaps.dtype)
    queries = query_points_tyx.to(device=fmaps.device, dtype=fmaps.dtype)
    point_count, frame_count = native.shape[:2]
    _, channels, feature_height, feature_width = fmaps.shape
    query_frames = queries[:, 0].round().long().clamp(0, frame_count - 1)
    query_xy = torch.stack(
        [queries[:, 2] * float(input_width - 1), queries[:, 1] * float(input_height - 1)],
        dim=-1,
    )
    query_support = torch.stack(
        [
            sample_feature_at_xy(
                fmaps[int(query_frames[index].item())],
                query_xy[index],
                input_height=input_height,
                input_width=input_width,
            )
            for index in range(point_count)
        ]
    )
    query_support = F.normalize(query_support.float(), dim=-1, eps=1.0e-12)
    previous_support = query_support.clone()
    ema_support = query_support.clone()

    correlations = fmaps.new_zeros(
        point_count, frame_count, CMCP_MEMORY_COUNT, feature_height, feature_width
    )
    motion = fmaps.new_zeros(point_count, frame_count, 1, feature_height, feature_width)
    valid = torch.zeros(point_count, frame_count, dtype=torch.bool, device=fmaps.device)

    for frame_index in range(frame_count):
        active = frame_index >= query_frames
        current_map = fmaps[frame_index]
        current_native = sample_feature_at_xy(
            current_map,
            native[:, frame_index],
            input_height=input_height,
            input_width=input_width,
        )
        current_native = F.normalize(current_native.float(), dim=-1, eps=1.0e-12)
        memory = torch.stack([query_support, previous_support, ema_support], dim=1)
        frame_corr = torch.einsum("nmd,dhw->nmhw", memory, current_map)
        frame_motion = build_native_motion_prior(
            native[:, frame_index],
            feature_height=feature_height,
            feature_width=feature_width,
            input_height=input_height,
            input_width=input_width,
            sigma_cells=motion_sigma_cells,
        )
        correlations[:, frame_index] = torch.where(
            active[:, None, None, None], frame_corr, torch.zeros_like(frame_corr)
        )
        motion[:, frame_index] = torch.where(
            active[:, None, None, None], frame_motion, torch.zeros_like(frame_motion)
        )
        valid[:, frame_index] = active
        active_feature = active[:, None]
        previous_support = torch.where(active_feature, current_native, previous_support)
        updated_ema = float(ema_alpha) * ema_support + (1.0 - float(ema_alpha)) * current_native
        updated_ema = F.normalize(updated_ema, dim=-1, eps=1.0e-12)
        ema_support = torch.where(active_feature, updated_ema, ema_support)
    return correlations, motion, valid


def stable_spatial_topk_nms(
    score_map: torch.Tensor,
    *,
    topk: int,
    radius_cells: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable row-major top-K with square NMS suppression.

    ``torch.argmax`` returns the first flattened maximum, preserving the frozen
    row-major tie rule without sorting the entire dense map.
    """
    if score_map.ndim != 3:
        raise ValueError("score_map must have shape (B,H,W)")
    if topk <= 0 or radius_cells < 0:
        raise ValueError("invalid topk or radius")
    batch, height, width = score_map.shape
    work = score_map.clone()
    selected_indices = torch.zeros(batch, topk, dtype=torch.long, device=score_map.device)
    selected_scores = torch.full(
        (batch, topk), float("-inf"), device=score_map.device, dtype=score_map.dtype
    )
    selected_valid = torch.zeros(batch, topk, dtype=torch.bool, device=score_map.device)
    for rank in range(topk):
        flat = work.reshape(batch, -1)
        index = flat.argmax(dim=1)
        value = flat.gather(1, index[:, None]).squeeze(1)
        valid = torch.isfinite(value)
        selected_indices[:, rank] = index
        selected_scores[:, rank] = value
        selected_valid[:, rank] = valid
        for batch_index in range(batch):
            if not bool(valid[batch_index]):
                continue
            flat_index = int(index[batch_index].item())
            y = flat_index // width
            x = flat_index % width
            ymin = max(0, y - radius_cells)
            ymax = min(height, y + radius_cells + 1)
            xmin = max(0, x - radius_cells)
            xmax = min(width, x + radius_cells + 1)
            work[batch_index, ymin:ymax, xmin:xmax] = float("-inf")
    return selected_indices, selected_scores, selected_valid


def extract_proposal_candidates(
    proposal_score: torch.Tensor,
    native_xy_px: torch.Tensor,
    native_logit: torch.Tensor,
    config: CMCPConfig,
) -> dict[str, torch.Tensor]:
    if proposal_score.ndim != 4 or proposal_score.shape[1] != 1:
        raise ValueError("proposal_score must have shape (B,1,H,W)")
    if native_xy_px.shape != (proposal_score.shape[0], 2):
        raise ValueError("native_xy_px must have shape (B,2)")
    if native_logit.shape != (proposal_score.shape[0],):
        raise ValueError("native_logit must have shape (B,)")
    batch, _, height, width = proposal_score.shape
    indices, peak_scores, peak_valid = stable_spatial_topk_nms(
        proposal_score[:, 0],
        topk=config.proposal_topk,
        radius_cells=config.nms_radius_cells,
    )
    peak_y = torch.div(indices, width, rounding_mode="floor")
    peak_x = indices % width
    x_px = peak_x.to(proposal_score.dtype) / float(max(width - 1, 1)) * float(
        config.input_width - 1
    )
    y_px = peak_y.to(proposal_score.dtype) / float(max(height - 1, 1)) * float(
        config.input_height - 1
    )
    peak_xy = torch.stack([x_px, y_px], dim=-1)
    coords = torch.cat([native_xy_px[:, None], peak_xy], dim=1)
    scores = torch.cat([native_logit[:, None], peak_scores], dim=1)
    valid = torch.cat(
        [torch.ones(batch, 1, dtype=torch.bool, device=proposal_score.device), peak_valid],
        dim=1,
    )
    masked_scores = scores.masked_fill(~valid, float("-inf"))
    selected_index = masked_scores.argmax(dim=1)
    selected_coord = coords.gather(
        1, selected_index[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    return {
        "candidate_coords_xy_px": coords,
        "candidate_scores": scores,
        "candidate_valid_mask": valid,
        "selected_candidate_index": selected_index,
        "selected_coord_xy_px": selected_coord,
        "peak_flat_indices": indices,
    }


def build_dense_proposal_targets(
    gt_xy_px: torch.Tensor,
    native_xy_px: torch.Tensor,
    *,
    feature_height: int,
    feature_width: int,
    input_height: int,
    input_width: int,
    thresholds_px: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
    utility_weights: tuple[float, ...] = (0.28, 0.24, 0.20, 0.16, 0.12),
    gaussian_sigma_px: float = 4.0,
    catastrophe_threshold_px: float = 16.0,
) -> dict[str, torch.Tensor]:
    if gt_xy_px.ndim != 2 or gt_xy_px.shape[-1] != 2:
        raise ValueError("gt_xy_px must have shape (B,2)")
    if native_xy_px.shape != gt_xy_px.shape:
        raise ValueError("native_xy_px shape mismatch")
    if len(thresholds_px) != len(utility_weights):
        raise ValueError("threshold/utility length mismatch")
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feature_height, device=gt_xy_px.device, dtype=gt_xy_px.dtype),
        torch.arange(feature_width, device=gt_xy_px.device, dtype=gt_xy_px.dtype),
        indexing="ij",
    )
    grid_x_px = grid_x / float(max(feature_width - 1, 1)) * float(input_width - 1)
    grid_y_px = grid_y / float(max(feature_height - 1, 1)) * float(input_height - 1)
    dx = grid_x_px[None] - gt_xy_px[:, 0, None, None]
    dy = grid_y_px[None] - gt_xy_px[:, 1, None, None]
    distance = torch.sqrt(dx.square() + dy.square())
    thresholds = torch.tensor(thresholds_px, device=gt_xy_px.device, dtype=gt_xy_px.dtype)
    weights = torch.tensor(utility_weights, device=gt_xy_px.device, dtype=gt_xy_px.dtype)
    weights = weights / weights.sum()
    utility = (
        (distance[..., None] <= thresholds).to(gt_xy_px.dtype) * weights
    ).sum(dim=-1)
    gaussian = torch.exp(-0.5 * distance.square() / float(gaussian_sigma_px**2))
    risk = (distance >= float(catastrophe_threshold_px)).to(gt_xy_px.dtype)
    native_distance = torch.linalg.vector_norm(native_xy_px - gt_xy_px, dim=-1)
    native_utility = (
        (native_distance[:, None] <= thresholds).to(gt_xy_px.dtype) * weights
    ).sum(dim=-1)
    fallback = native_utility >= utility.flatten(1).max(dim=1).values - 1.0e-6
    return {
        "utility_target": utility.unsqueeze(1),
        "gaussian_target": gaussian.unsqueeze(1),
        "risk_target": risk.unsqueeze(1),
        "native_fallback_target": fallback.to(gt_xy_px.dtype),
    }
