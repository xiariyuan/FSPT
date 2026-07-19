"""Causal candidate-conditioned temporal identity feature utilities."""
from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F


TEMPORAL_IDENTITY_FEATURE_SCHEMA_VERSION: Final = (
    "routeD_candidate_temporal_identity_features_v0"
)

TEMPORAL_FEATURE_CHANNELS: Final = (
    "query_descriptor_cosine",
    "descriptor_previous_frame_cosine",
    "reverse_visibility_probability",
    "reverse_confidence_probability",
    "track_x_normalized",
    "track_y_normalized",
    "track_dx_normalized",
    "track_dy_normalized",
    "distance_from_native_track_normalized",
)

STATIC_FEATURE_CHANNELS: Final = (
    "m1_score_z",
    "rank_fraction",
    "is_native",
    "is_valid",
    "commit_x_normalized",
    "commit_y_normalized",
    "commit_offset_from_native_x_normalized",
    "commit_offset_from_native_y_normalized",
    "reverse_cycle_error_at_query_normalized",
    "mean_query_descriptor_cosine",
    "minimum_query_descriptor_cosine",
    "mean_reverse_visibility_probability",
    "mean_reverse_confidence_probability",
    "mean_descriptor_previous_frame_cosine",
)


def sample_spatiotemporal_descriptors(
    feature_video: torch.Tensor,
    tracklets_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Sample a frozen feature video along candidate tracklets.

    feature_video is [T,C,Hf,Wf], tracklets_xy is [K,T,2], and the
    returned unit-normalized descriptor tensor is [K,T,C].
    """
    if feature_video.ndim != 4:
        raise ValueError("feature video must have shape [T,C,H,W]")
    if tracklets_xy.ndim != 3 or tracklets_xy.shape[-1] != 2:
        raise ValueError("tracklets must have shape [K,T,2]")
    if feature_video.shape[0] != tracklets_xy.shape[1]:
        raise ValueError("feature-video and tracklet time dimensions differ")
    if int(input_height) <= 0 or int(input_width) <= 0:
        raise ValueError("input raster must be positive")

    coordinates = tracklets_xy.to(
        device=feature_video.device, dtype=feature_video.dtype
    )
    output = []
    for frame in range(feature_video.shape[0]):
        grid = coordinates[:, frame].clone()
        # Pixel-center conversion for align_corners=False.
        grid[:, 0] = 2.0 * (grid[:, 0] + 0.5) / float(input_width) - 1.0
        grid[:, 1] = 2.0 * (grid[:, 1] + 0.5) / float(input_height) - 1.0
        sampled = F.grid_sample(
            feature_video[frame : frame + 1],
            grid[None, :, None],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].T
        output.append(sampled)
    value = torch.stack(output, dim=1)
    return F.normalize(value.float(), p=2, dim=-1, eps=1.0e-8)


def _valid_zscore(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(value, dtype=torch.float32)
    local = value[valid].float()
    if local.numel() == 0:
        return output
    output[valid] = (local - local.mean()) / local.std(
        unbiased=False
    ).clamp_min(1.0e-6)
    return output


def assemble_candidate_temporal_identity_features(
    *,
    descriptor_sequence: torch.Tensor,
    query_descriptor: torch.Tensor,
    tracklets_xy: torch.Tensor,
    visibility_probability: torch.Tensor,
    confidence_probability: torch.Tensor,
    candidate_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    query_frame: int,
    query_coordinate_xy: torch.Tensor,
    input_height: int,
    input_width: int,
) -> dict[str, torch.Tensor]:
    """Assemble fixed causal features for one candidate list.

    Candidate zero is the mandatory native proposal. No teacher coordinate or
    future frame is accepted by this interface.
    """
    if descriptor_sequence.ndim != 3:
        raise ValueError("descriptor sequence must have shape [K,T,C]")
    candidates, frames, _ = descriptor_sequence.shape
    expected_kt = (candidates, frames)
    if tracklets_xy.shape != (candidates, frames, 2):
        raise ValueError("tracklet shape mismatch")
    if visibility_probability.shape != expected_kt:
        raise ValueError("visibility shape mismatch")
    if confidence_probability.shape != expected_kt:
        raise ValueError("confidence shape mismatch")
    if candidate_scores.shape != (candidates,):
        raise ValueError("candidate-score shape mismatch")
    if valid_mask.shape != (candidates,):
        raise ValueError("candidate-mask shape mismatch")
    if query_descriptor.shape != (descriptor_sequence.shape[-1],):
        raise ValueError("query-descriptor shape mismatch")
    if query_coordinate_xy.shape != (2,):
        raise ValueError("query coordinate must have shape [2]")
    if not 0 <= int(query_frame) < frames:
        raise ValueError("query frame is outside the observed sequence")
    if candidates == 0:
        raise ValueError("candidate list is empty")

    descriptor = F.normalize(
        descriptor_sequence.float(), p=2, dim=-1, eps=1.0e-8
    )
    query = F.normalize(query_descriptor.float(), p=2, dim=-1, eps=1.0e-8)
    query_cosine = (descriptor * query[None, None]).sum(dim=-1)
    previous_cosine = torch.ones_like(query_cosine)
    if frames > 1:
        previous_cosine[:, 1:] = (
            descriptor[:, 1:] * descriptor[:, :-1]
        ).sum(dim=-1)

    scale = tracklets_xy.new_tensor(
        [float(input_width), float(input_height)]
    )
    normalized_track = tracklets_xy.float() / scale
    normalized_delta = torch.zeros_like(normalized_track)
    if frames > 1:
        normalized_delta[:, 1:] = (
            tracklets_xy[:, 1:].float() - tracklets_xy[:, :-1].float()
        ) / scale
    distance_from_native = torch.linalg.vector_norm(
        (tracklets_xy.float() - tracklets_xy[0:1].float()) / scale,
        dim=-1,
    )
    temporal = torch.stack(
        [
            query_cosine,
            previous_cosine,
            visibility_probability.float(),
            confidence_probability.float(),
            normalized_track[..., 0],
            normalized_track[..., 1],
            normalized_delta[..., 0],
            normalized_delta[..., 1],
            distance_from_native,
        ],
        dim=-1,
    )

    commit = normalized_track[:, -1]
    commit_offset = commit - commit[0:1]
    cycle_error = torch.linalg.vector_norm(
        (
            tracklets_xy[:, int(query_frame)].float()
            - query_coordinate_xy.float()[None]
        )
        / scale,
        dim=-1,
    )
    rank = torch.arange(
        candidates, device=descriptor.device, dtype=torch.float32
    ) / float(max(candidates - 1, 1))
    native = torch.zeros(candidates, device=descriptor.device)
    native[0] = 1.0
    valid = valid_mask.to(device=descriptor.device, dtype=torch.bool)
    static = torch.stack(
        [
            _valid_zscore(candidate_scores.to(descriptor.device), valid),
            rank,
            native,
            valid.float(),
            commit[:, 0],
            commit[:, 1],
            commit_offset[:, 0],
            commit_offset[:, 1],
            cycle_error,
            query_cosine.mean(dim=1),
            query_cosine.min(dim=1).values,
            visibility_probability.float().mean(dim=1),
            confidence_probability.float().mean(dim=1),
            previous_cosine.mean(dim=1),
        ],
        dim=-1,
    )
    temporal = temporal.masked_fill(~valid[:, None, None], 0.0)
    static = static.masked_fill(~valid[:, None], 0.0)
    static[:, STATIC_FEATURE_CHANNELS.index("is_valid")] = valid.float()
    if not torch.isfinite(temporal).all() or not torch.isfinite(static).all():
        raise ValueError("non-finite temporal identity feature")
    return {
        "temporal_features": temporal,
        "static_features": static,
        "valid_mask": valid,
    }
