"""Causal shortlist features for Route-D Gate 3C1D top-1 selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
)
from .routeD_temporal_identity_selector import (
    QueryClosureIdentitySelectorConfig,
    candidate_relative_zscore,
    select_query_closure_identity_candidates,
)

TOP1_FEATURE_SCHEMA_VERSION: Final = (
    "routeD_query_closure_shortlist_top1_features_gate3c1d_v0"
)


@dataclass(frozen=True)
class Top1FeatureConfig:
    shortlist_size: int = 9
    temporal_summary_channels: tuple[int, ...] = (0, 1, 2, 3, 6, 7, 8)
    zscore_epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.shortlist_size != 9:
            raise ValueError("Gate 3C1D shortlist size must remain nine")
        if self.zscore_epsilon <= 0.0:
            raise ValueError("top-1 zscore epsilon must be positive")
        if any(
            index < 0 or index >= len(TEMPORAL_FEATURE_CHANNELS)
            for index in self.temporal_summary_channels
        ):
            raise ValueError("top-1 temporal summary channel is invalid")


def _candidate_relative_feature_zscore(
    value: torch.Tensor, valid_mask: torch.Tensor, epsilon: float
) -> torch.Tensor:
    if value.ndim != 3 or valid_mask.shape != value.shape[:2]:
        raise ValueError("candidate feature zscore shape mismatch")
    valid = valid_mask.bool()
    mask = valid[..., None].to(value.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=1, keepdim=True) / count
    variance = ((value - mean).square() * mask).sum(dim=1, keepdim=True) / count
    output = (value - mean) / variance.sqrt().clamp_min(float(epsilon))
    return output.masked_fill(~valid[..., None], 0.0)


def build_full_bank_top1_features(
    *,
    temporal_features: torch.Tensor,
    static_features: torch.Tensor,
    query_frames: torch.Tensor,
    valid_mask: torch.Tensor,
    config: Top1FeatureConfig = Top1FeatureConfig(),
) -> torch.Tensor:
    """Build 98 causal raw-plus-relative candidate features for all 129 candidates."""
    if temporal_features.ndim != 4:
        raise ValueError("temporal features must have shape [R,K,T,C]")
    rows, candidates, frames, channels = temporal_features.shape
    if channels != len(TEMPORAL_FEATURE_CHANNELS):
        raise ValueError("top-1 temporal channel drift")
    if static_features.shape != (rows, candidates, len(STATIC_FEATURE_CHANNELS)):
        raise ValueError("top-1 static feature shape drift")
    if query_frames.shape != (rows,) or valid_mask.shape != (rows, candidates):
        raise ValueError("top-1 query/mask shape drift")
    if rows == 0:
        return temporal_features.new_empty((0, candidates, 98))
    if torch.any(query_frames < 0) or torch.any(query_frames >= frames):
        raise ValueError("top-1 query frame is outside observed frames")

    gather_index = query_frames[:, None, None].expand(rows, candidates, 1)
    query_identity = torch.gather(
        temporal_features[..., TEMPORAL_FEATURE_CHANNELS.index("query_descriptor_cosine")],
        2,
        gather_index,
    ).squeeze(2)
    previous_identity = torch.gather(
        temporal_features[
            ..., TEMPORAL_FEATURE_CHANNELS.index("descriptor_previous_frame_cosine")
        ],
        2,
        gather_index,
    ).squeeze(2)
    query_fraction = (
        query_frames.float() / float(max(frames - 1, 1))
    )[:, None].expand(rows, candidates)

    parts = [
        static_features.float(),
        query_identity[..., None],
        previous_identity[..., None],
        query_fraction[..., None],
    ]
    for channel in config.temporal_summary_channels:
        value = temporal_features[..., int(channel)].float()
        parts.extend(
            [
                value.mean(dim=2, keepdim=True),
                value.amin(dim=2, keepdim=True),
                value.amax(dim=2, keepdim=True),
                value.std(dim=2, unbiased=False, keepdim=True),
            ]
        )
    velocity = temporal_features[..., 6:8].float()
    acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
    jerk = acceleration[:, :, 1:] - acceleration[:, :, :-1]
    parts.extend(
        [
            velocity.norm(dim=-1).sum(dim=2, keepdim=True),
            acceleration.norm(dim=-1).mean(dim=2, keepdim=True),
            acceleration.norm(dim=-1).amax(dim=2, keepdim=True),
            jerk.norm(dim=-1).mean(dim=2, keepdim=True),
        ]
    )
    raw = torch.cat(parts, dim=2)
    if raw.shape[-1] != 49:
        raise ValueError("top-1 raw feature dimension drift")
    relative = _candidate_relative_feature_zscore(
        raw, valid_mask, config.zscore_epsilon
    )
    output = torch.cat([raw, relative], dim=2)
    if output.shape[-1] != 98 or not torch.isfinite(output[valid_mask]).all():
        raise ValueError("top-1 full-bank feature drift")
    return output.masked_fill(~valid_mask[..., None], 0.0)


def build_shortlist_top1_features(
    *,
    temporal_features: torch.Tensor,
    static_features: torch.Tensor,
    query_frames: torch.Tensor,
    valid_mask: torch.Tensor,
    selector_config: QueryClosureIdentitySelectorConfig = (
        QueryClosureIdentitySelectorConfig()
    ),
    feature_config: Top1FeatureConfig = Top1FeatureConfig(),
) -> dict[str, torch.Tensor]:
    """Return frozen shortlist indices and 102-D causal top-1 candidate features."""
    shortlist = select_query_closure_identity_candidates(
        temporal_features=temporal_features,
        static_features=static_features,
        query_frames=query_frames,
        valid_mask=valid_mask,
        config=selector_config,
    )
    selected = shortlist["selected_indices"]
    rows, shortlist_size = selected.shape
    if shortlist_size != feature_config.shortlist_size:
        raise ValueError("top-1 shortlist size drift")
    full = build_full_bank_top1_features(
        temporal_features=temporal_features,
        static_features=static_features,
        query_frames=query_frames,
        valid_mask=valid_mask,
        config=feature_config,
    )
    gather = selected[..., None].expand(rows, shortlist_size, full.shape[-1])
    candidate = full.gather(1, gather)
    score = shortlist["score"].gather(1, selected)
    score_z = candidate_relative_zscore(
        score, torch.ones_like(score, dtype=torch.bool), epsilon=feature_config.zscore_epsilon
    )
    position = torch.arange(
        shortlist_size, device=candidate.device, dtype=candidate.dtype
    )[None, :, None].expand(rows, shortlist_size, 1) / float(shortlist_size - 1)
    native = torch.zeros(rows, shortlist_size, 1, device=candidate.device, dtype=candidate.dtype)
    if rows:
        native[:, 0] = 1.0
    output = torch.cat(
        [candidate, score[..., None], score_z[..., None], position, native], dim=2
    )
    if output.shape != (rows, shortlist_size, 102):
        raise ValueError("top-1 shortlist feature dimension drift")
    if rows and not torch.isfinite(output).all():
        raise ValueError("top-1 shortlist features are non-finite")
    return {
        "selected_indices": selected,
        "base_score": score,
        "candidate_features": output,
    }
