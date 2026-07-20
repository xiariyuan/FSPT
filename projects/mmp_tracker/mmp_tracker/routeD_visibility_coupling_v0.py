"""Causal post-writeback visibility features for Route-D Gate 3C1G0."""
from __future__ import annotations

from typing import Final, Mapping

import numpy as np
import torch

VISIBILITY_FEATURE_SCHEMA_VERSION: Final = (
    "routeD_post_writeback_visibility_features_gate3c1g0_v0"
)

_BASE_CHANNELS = (
    "frame_fraction",
    "native_x",
    "native_y",
    "modified_x",
    "modified_y",
    "coordinate_delta_x",
    "coordinate_delta_y",
    "coordinate_delta_norm",
    "native_speed",
    "modified_speed",
    "native_acceleration",
    "modified_acceleration",
    "modified_distance_from_commit",
    "modified_border_margin",
    "native_visibility_probability",
    "native_confidence_probability",
    "native_joint_probability",
    "modified_visibility_probability",
    "modified_confidence_probability",
    "modified_joint_probability",
    "visibility_probability_delta",
    "confidence_probability_delta",
    "joint_probability_delta",
    "native_visible_binary",
    "modified_visible_binary",
    "dino_modified_query_cosine",
    "dino_native_query_cosine",
    "dino_query_cosine_delta",
    "dino_modified_previous_cosine",
    "dino_native_previous_cosine",
    "dino_modified_commit_cosine",
    "dino_native_commit_cosine",
    "dino_modified_native_cosine",
    "dino_modified_query_running_mean",
    "dino_modified_query_running_minimum",
    "entry_probability",
    "commit_native_joint_probability",
    "selected_support_probability",
    "predicted_value_normalized",
    "predicted_harm_probability",
    "selected_slot_normalized",
    "output_candidate_index_normalized",
)

_COTRACKER_PER_LEVEL = (
    "modified_commit_cosine",
    "native_commit_cosine",
    "modified_native_commit_cosine",
    "modified_native_current_cosine",
    "modified_previous_cosine",
    "native_previous_cosine",
)

VISIBILITY_FEATURE_CHANNELS: Final = _BASE_CHANNELS + tuple(
    f"cotracker_level{level}_{name}"
    for level in range(4)
    for name in _COTRACKER_PER_LEVEL
)


def assemble_visibility_features(
    channels: Mapping[str, torch.Tensor | np.ndarray],
    *,
    rows: int,
    frames: int,
) -> torch.Tensor:
    """Stack the frozen 66-D causal feature schema into [rows,frames,channels]."""
    expected = set(VISIBILITY_FEATURE_CHANNELS)
    actual = set(channels)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"visibility feature channel drift: missing={missing}, extra={extra}")
    values = []
    for name in VISIBILITY_FEATURE_CHANNELS:
        value = channels[name]
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
        else:
            raise ValueError(f"visibility feature channel is not tensor-like: {name}")
        tensor = tensor.float()
        if tuple(tensor.shape) != (int(rows), int(frames)):
            raise ValueError(
                f"visibility feature channel shape drift: {name}={tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"visibility feature channel is non-finite: {name}")
        values.append(tensor)
    output = torch.stack(values, dim=-1).contiguous()
    if output.shape != (int(rows), int(frames), len(VISIBILITY_FEATURE_CHANNELS)):
        raise ValueError("visibility feature output shape drift")
    return output


def flatten_visibility_rows(
    features: torch.Tensor,
    *,
    source_index: int,
    point_indices: torch.Tensor,
    frame_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Flatten [A,F,D] features with deterministic source/point/frame identities."""
    if features.ndim != 3:
        raise ValueError("visibility features must have shape [actions,frames,dim]")
    actions, frames, dim = features.shape
    if dim != len(VISIBILITY_FEATURE_CHANNELS):
        raise ValueError("visibility feature dimension drift")
    if point_indices.shape != (actions,) or frame_indices.shape != (frames,):
        raise ValueError("visibility identity shape drift")
    return {
        "features": features.reshape(actions * frames, dim).contiguous(),
        "source_indices": torch.full(
            (actions * frames,), int(source_index), dtype=torch.long
        ),
        "point_indices": point_indices[:, None]
        .expand(actions, frames)
        .reshape(-1)
        .long()
        .contiguous(),
        "frame_indices": frame_indices[None]
        .expand(actions, frames)
        .reshape(-1)
        .long()
        .contiguous(),
    }
