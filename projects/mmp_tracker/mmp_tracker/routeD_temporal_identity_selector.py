"""Deterministic query-closure identity selector for Route-D Gate 3C1B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
)

SELECTOR_SCHEMA_VERSION: Final = (
    "routeD_query_closure_identity_selector_gate3c1b_v0"
)


@dataclass(frozen=True)
class QueryClosureIdentitySelectorConfig:
    """Frozen analytic shortlist rule.

    Candidate zero is always retained. Scores are used only to choose eight
    non-native candidates from the frozen 128-candidate M1 bank.
    """

    cycle_weight: float = 2.0
    query_frame_identity_weight: float = 1.0
    mean_identity_weight: float = 1.0
    minimum_identity_weight: float = 1.0
    retained_nonnative: int = 8
    zscore_epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        weights = (
            self.cycle_weight,
            self.query_frame_identity_weight,
            self.mean_identity_weight,
            self.minimum_identity_weight,
        )
        if any(value <= 0.0 for value in weights):
            raise ValueError("all closure-identity weights must be positive")
        if self.retained_nonnative <= 0:
            raise ValueError("retained_nonnative must be positive")
        if self.zscore_epsilon <= 0.0:
            raise ValueError("zscore_epsilon must be positive")


def candidate_relative_zscore(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Normalize each candidate list using only valid candidates."""
    if value.ndim != 2:
        raise ValueError("value must have shape [rows,candidates]")
    if valid_mask.shape != value.shape:
        raise ValueError("valid-mask shape mismatch")
    valid = valid_mask.bool()
    if not torch.all(valid.any(dim=1)):
        raise ValueError("every row must contain at least one valid candidate")
    mask = valid.to(value.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (value * mask).sum(dim=1, keepdim=True) / count
    variance = ((value - mean).square() * mask).sum(dim=1, keepdim=True) / count
    output = (value - mean) / variance.sqrt().clamp_min(float(epsilon))
    return output.masked_fill(~valid, 0.0)


def query_closure_identity_score(
    *,
    temporal_features: torch.Tensor,
    static_features: torch.Tensor,
    query_frames: torch.Tensor,
    valid_mask: torch.Tensor,
    config: QueryClosureIdentitySelectorConfig = (
        QueryClosureIdentitySelectorConfig()
    ),
) -> dict[str, torch.Tensor]:
    """Compute the frozen causal score without accepting teacher labels.

    The score is

    ``-2 * z(cycle_error) + z(identity_at_query) + z(identity_mean)
    + z(identity_minimum)``

    under the default frozen configuration. Normalization is performed within
    each candidate list and ignores invalid candidates.
    """
    if temporal_features.ndim != 4:
        raise ValueError("temporal features must have shape [R,K,T,C]")
    rows, candidates, frames, temporal_channels = temporal_features.shape
    if temporal_channels != len(TEMPORAL_FEATURE_CHANNELS):
        raise ValueError("temporal feature-channel mismatch")
    if static_features.shape != (
        rows,
        candidates,
        len(STATIC_FEATURE_CHANNELS),
    ):
        raise ValueError("static feature shape mismatch")
    if query_frames.shape != (rows,):
        raise ValueError("query-frame shape mismatch")
    if valid_mask.shape != (rows, candidates):
        raise ValueError("valid-mask shape mismatch")
    if rows == 0:
        empty = temporal_features.new_empty((0, candidates))
        return {
            "score": empty,
            "cycle_z": empty,
            "query_frame_identity_z": empty,
            "mean_identity_z": empty,
            "minimum_identity_z": empty,
        }
    if torch.any(query_frames < 0) or torch.any(query_frames >= frames):
        raise ValueError("query frame is outside observed frames")
    valid = valid_mask.bool()
    if not torch.all(valid[:, 0]):
        raise ValueError("mandatory native candidate must be valid")
    if torch.any(valid[:, 1:].sum(dim=1) < config.retained_nonnative):
        raise ValueError("insufficient valid non-native candidates")

    query_identity_index = TEMPORAL_FEATURE_CHANNELS.index(
        "query_descriptor_cosine"
    )
    cycle_index = STATIC_FEATURE_CHANNELS.index(
        "reverse_cycle_error_at_query_normalized"
    )
    mean_identity_index = STATIC_FEATURE_CHANNELS.index(
        "mean_query_descriptor_cosine"
    )
    minimum_identity_index = STATIC_FEATURE_CHANNELS.index(
        "minimum_query_descriptor_cosine"
    )

    gather_index = query_frames[:, None, None].expand(rows, candidates, 1)
    identity_at_query = torch.gather(
        temporal_features[..., query_identity_index],
        dim=2,
        index=gather_index,
    ).squeeze(2)
    cycle = static_features[..., cycle_index]
    identity_mean = static_features[..., mean_identity_index]
    identity_minimum = static_features[..., minimum_identity_index]

    cycle_z = candidate_relative_zscore(
        cycle, valid, epsilon=config.zscore_epsilon
    )
    query_z = candidate_relative_zscore(
        identity_at_query, valid, epsilon=config.zscore_epsilon
    )
    mean_z = candidate_relative_zscore(
        identity_mean, valid, epsilon=config.zscore_epsilon
    )
    minimum_z = candidate_relative_zscore(
        identity_minimum, valid, epsilon=config.zscore_epsilon
    )
    score = (
        -float(config.cycle_weight) * cycle_z
        + float(config.query_frame_identity_weight) * query_z
        + float(config.mean_identity_weight) * mean_z
        + float(config.minimum_identity_weight) * minimum_z
    )
    score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
    if not torch.isfinite(score[valid]).all():
        raise ValueError("non-finite selector score")
    return {
        "score": score,
        "cycle_z": cycle_z,
        "query_frame_identity_z": query_z,
        "mean_identity_z": mean_z,
        "minimum_identity_z": minimum_z,
    }


def retain_native_plus_nonnative(
    score: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    retained_nonnative: int = 8,
) -> torch.Tensor:
    """Return stable candidate indices [native, eight scored non-native]."""
    if score.ndim != 2 or valid_mask.shape != score.shape:
        raise ValueError("score and valid mask must have shape [R,K]")
    rows, candidates = score.shape
    if candidates <= retained_nonnative:
        raise ValueError("candidate list is too short")
    valid = valid_mask.bool()
    if rows and not torch.all(valid[:, 0]):
        raise ValueError("mandatory native candidate must be valid")
    if rows and torch.any(valid[:, 1:].sum(dim=1) < retained_nonnative):
        raise ValueError("insufficient valid non-native candidates")
    if rows == 0:
        return torch.empty(
            0,
            retained_nonnative + 1,
            dtype=torch.long,
            device=score.device,
        )
    nonnative_score = score[:, 1:].masked_fill(
        ~valid[:, 1:], torch.finfo(score.dtype).min
    )
    # Stable sort makes lower frozen M1 rank the deterministic tie breaker.
    order = torch.argsort(
        nonnative_score, dim=1, descending=True, stable=True
    )[:, :retained_nonnative]
    native = torch.zeros(rows, 1, dtype=torch.long, device=score.device)
    return torch.cat([native, order + 1], dim=1)


def select_query_closure_identity_candidates(
    *,
    temporal_features: torch.Tensor,
    static_features: torch.Tensor,
    query_frames: torch.Tensor,
    valid_mask: torch.Tensor,
    config: QueryClosureIdentitySelectorConfig = (
        QueryClosureIdentitySelectorConfig()
    ),
) -> dict[str, torch.Tensor]:
    """Score and retain the frozen native-plus-eight shortlist."""
    result = query_closure_identity_score(
        temporal_features=temporal_features,
        static_features=static_features,
        query_frames=query_frames,
        valid_mask=valid_mask,
        config=config,
    )
    result["selected_indices"] = retain_native_plus_nonnative(
        result["score"],
        valid_mask,
        retained_nonnative=config.retained_nonnative,
    )
    return result
