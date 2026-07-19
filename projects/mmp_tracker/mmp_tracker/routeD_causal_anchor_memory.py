"""Causal immutable-anchor extraction for Route-D relocalization."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .routeD_counterfactual_state_restorer import input_xy_to_model_xy


CAUSAL_ANCHOR_SCHEMA_VERSION = "routeD_causal_anchor_memory_interface_v0"


def extract_immutable_query_anchor_memory(
    model: Any,
    observed_feature_pyramid: Sequence[torch.Tensor],
    query_frames: torch.Tensor,
    query_coordinates_input_xy: torch.Tensor,
    *,
    input_height: int = 256,
    input_width: int = 256,
    support_radius: int = 3,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Extract four-level anchor memory at each point's original query event.

    The feature sequence is physically truncated after the latest requested
    query frame before ``get_track_feat`` is called. This makes the no-future
    contract structural rather than relying on the callee to ignore later
    frames. The returned tensors use native CoTracker layouts
    ``[1,1,N,C]`` and ``[1,(2r+1)^2,N,C]``.
    """
    if query_frames.ndim != 1:
        raise ValueError("query frames must have shape [N]")
    if (
        query_coordinates_input_xy.ndim != 2
        or query_coordinates_input_xy.shape != (query_frames.numel(), 2)
    ):
        raise ValueError("query coordinates must have shape [N,2]")
    if query_frames.numel() == 0:
        raise ValueError("at least one query anchor is required")
    if query_frames.dtype.is_floating_point:
        rounded = query_frames.round()
        if not torch.equal(query_frames, rounded):
            raise ValueError("query frames must be integer-valued")
        frames = rounded.long()
    else:
        frames = query_frames.long()
    if int(frames.min()) < 0:
        raise ValueError("query frames must be non-negative")
    expected_levels = int(model.corr_levels)
    if len(observed_feature_pyramid) != expected_levels:
        raise ValueError("feature-pyramid level mismatch")
    latest_query = int(frames.max())
    prefix_end = latest_query + 1
    for feature in observed_feature_pyramid:
        if feature.ndim != 5 or feature.shape[0] != 1:
            raise ValueError("feature levels must have shape [1,T,C,H,W]")
        if feature.shape[1] < prefix_end:
            raise ValueError("feature sequence ends before a query frame")

    model_height, model_width = map(int, model.model_resolution)
    coordinates = query_coordinates_input_xy.to(
        device=observed_feature_pyramid[0].device,
        dtype=observed_feature_pyramid[0].dtype,
    )
    model_xy = input_xy_to_model_xy(
        coordinates,
        input_height=int(input_height),
        input_width=int(input_width),
        model_height=model_height,
        model_width=model_width,
    )
    queried_frames = frames.to(model_xy.device)[None]
    track_features: list[torch.Tensor] = []
    track_supports: list[torch.Tensor] = []
    for level, feature in enumerate(observed_feature_pyramid):
        causal_prefix = feature[:, :prefix_end]
        track_feature, track_support = model.get_track_feat(
            causal_prefix,
            queried_frames,
            model_xy[None] / float(int(model.stride) * (2**level)),
            support_radius=int(support_radius),
        )
        expected_support = (2 * int(support_radius) + 1) ** 2
        if track_feature.shape[:3] != (1, 1, frames.numel()):
            raise RuntimeError("unexpected immutable track-feature shape")
        if track_support.shape[:3] != (1, expected_support, frames.numel()):
            raise RuntimeError("unexpected immutable track-support shape")
        track_features.append(track_feature)
        track_supports.append(track_support)
    return tuple(track_features), tuple(track_supports)


def select_anchor_rows(
    values: Sequence[torch.Tensor], point_rows: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Select point rows from native-layout anchor memory without reordering tokens."""
    rows = point_rows.long()
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("anchor row selection must be a non-empty vector")
    output = []
    for value in values:
        if value.ndim != 4:
            raise ValueError("anchor values must use native four-dimensional layout")
        if int(rows.min()) < 0 or int(rows.max()) >= value.shape[2]:
            raise ValueError("anchor row outside point dimension")
        output.append(value[:, :, rows.to(value.device)].detach().clone())
    return tuple(output)
