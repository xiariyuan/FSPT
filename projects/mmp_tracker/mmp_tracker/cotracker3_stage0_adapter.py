"""Causal CoTracker3-online adapter primitives for Route-D MUSR stage 0.

The adapter keeps candidate generation independent from ground truth. Ground
truth is accepted only by the oracle-audit helpers after candidate tensors have
already been frozen. Coordinates are represented as input-raster ``[x, y]``
pixels unless a function explicitly states otherwise.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


CANDIDATE_FEATURE_DIM = 64
STATE_FEATURE_DIM = 32
NATIVE_SOURCE_ID = 0
LOCAL_CORRELATION_SOURCE_ID = 1

CANDIDATE_STRUCTURED_FEATURE_NAMES = (
    "support_correlation",
    "score_gap_to_local_peak",
    "local_peak",
    "local_peak_margin",
    "local_entropy",
    "normalized_rank",
    "is_native",
    "delta_x_over_radius",
    "delta_y_over_radius",
    "delta_norm_over_radius",
    "native_visibility_probability",
    "native_confidence_probability",
    "native_joint_probability",
    "motion_x_over_radius",
    "motion_y_over_radius",
    "motion_norm_over_radius",
)
STATE_STRUCTURED_FEATURE_NAMES = (
    "native_x_normalized",
    "native_y_normalized",
    "previous_x_normalized",
    "previous_y_normalized",
    "velocity_x_over_radius",
    "velocity_y_over_radius",
    "velocity_norm_over_radius",
    "acceleration_x_over_radius",
    "acceleration_y_over_radius",
    "acceleration_norm_over_radius",
    "visibility_probability",
    "confidence_probability",
    "joint_probability",
    "frame_progress",
    "query_age_normalized",
    "is_query_frame",
)


@dataclass(frozen=True)
class LocalSearchResult:
    """A deterministic local-correlation search result for one point/frame."""

    candidate_xy_px: torch.Tensor
    candidate_features: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_valid: torch.Tensor
    source_ids: torch.Tensor
    peak: float
    margin: float
    entropy: float
    search_cell_count: int


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("utf-8")
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_first_visible_queries(
    target_points_yx: np.ndarray,
    occluded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create official first-visible queries from normalized ``[y, x]`` tracks.

    Returns query points ``[t, y, x]``, filtered tracks, and filtered occlusion.
    Tracks that are never visible are removed.
    """
    tracks = np.asarray(target_points_yx, dtype=np.float32)
    occ = np.asarray(occluded, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("target_points_yx must have shape (N,T,2)")
    if occ.shape != tracks.shape[:2]:
        raise ValueError("occluded must have shape (N,T)")
    keep = (~occ).any(axis=1)
    tracks = tracks[keep]
    occ = occ[keep]
    if tracks.shape[0] == 0:
        raise ValueError("sample has no visible tracks")
    first_t = np.argmax(~occ, axis=1).astype(np.int64)
    point_index = np.arange(tracks.shape[0], dtype=np.int64)
    first_yx = tracks[point_index, first_t]
    queries = np.concatenate(
        [first_t[:, None].astype(np.float32), first_yx.astype(np.float32)], axis=1
    )
    return queries, tracks, occ


def normalize_feature_maps(fmaps: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature maps with shape ``(T,D,H,W)``."""
    if fmaps.ndim != 4:
        raise ValueError("fmaps must have shape (T,D,H,W)")
    return F.normalize(fmaps.float(), dim=1, eps=1.0e-12)


def _normalized_grid_from_xy(
    xy_px: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    x = xy_px[..., 0] / float(max(input_width - 1, 1))
    y = xy_px[..., 1] / float(max(input_height - 1, 1))
    return torch.stack([2.0 * x - 1.0, 2.0 * y - 1.0], dim=-1)


def sample_feature_at_xy(
    fmap: torch.Tensor,
    xy_px: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Bilinearly sample a ``(D,Hf,Wf)`` feature map at input-raster xy."""
    if fmap.ndim != 3:
        raise ValueError("fmap must have shape (D,Hf,Wf)")
    if xy_px.shape[-1] != 2:
        raise ValueError("xy_px must end in dimension 2")
    original_shape = xy_px.shape[:-1]
    flat_xy = xy_px.reshape(-1, 2).to(device=fmap.device, dtype=fmap.dtype)
    grid = _normalized_grid_from_xy(
        flat_xy, input_height=input_height, input_width=input_width
    ).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        fmap.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled[0, :, :, 0].transpose(0, 1)
    return F.normalize(sampled, dim=-1, eps=1.0e-12).reshape(*original_shape, fmap.shape[0])


def channel_bin_means(vector: torch.Tensor, bins: int = 16) -> torch.Tensor:
    """Deterministically pool a feature vector into equal contiguous channel bins."""
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if bins <= 0:
        raise ValueError("bins must be positive")
    chunks = torch.tensor_split(vector, bins)
    return torch.stack(
        [chunk.mean() if chunk.numel() else vector.new_zeros(()) for chunk in chunks]
    )


def _normalized_entropy(scores: torch.Tensor, temperature: float = 10.0) -> float:
    if scores.numel() <= 1:
        return 0.0
    probability = torch.softmax(scores.float() * float(temperature), dim=0)
    entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum()
    return float((entropy / math.log(scores.numel())).item())


def build_candidate_feature(
    *,
    support_feature: torch.Tensor,
    candidate_feature: torch.Tensor,
    candidate_score: float,
    local_peak: float,
    local_margin: float,
    local_entropy: float,
    rank: int,
    candidate_count: int,
    is_native: bool,
    candidate_xy_px: torch.Tensor,
    native_xy_px: torch.Tensor,
    previous_native_xy_px: torch.Tensor,
    native_visibility_probability: float,
    native_confidence_probability: float,
    radius_px: float,
) -> torch.Tensor:
    """Build the fixed 64-dimensional stage-0 candidate feature contract."""
    support = F.normalize(support_feature.float(), dim=0, eps=1.0e-12)
    candidate = F.normalize(candidate_feature.float(), dim=0, eps=1.0e-12)
    radius = max(float(radius_px), 1.0e-6)
    delta = candidate_xy_px.float() - native_xy_px.float()
    motion = native_xy_px.float() - previous_native_xy_px.float()
    structured = torch.tensor(
        [
            float(candidate_score),
            float(local_peak - candidate_score),
            float(local_peak),
            float(local_margin),
            float(local_entropy),
            float(rank) / float(max(candidate_count - 1, 1)),
            1.0 if is_native else 0.0,
            float(delta[0].item()) / radius,
            float(delta[1].item()) / radius,
            float(torch.linalg.vector_norm(delta).item()) / radius,
            float(native_visibility_probability),
            float(native_confidence_probability),
            float(native_visibility_probability * native_confidence_probability),
            float(motion[0].item()) / radius,
            float(motion[1].item()) / radius,
            float(torch.linalg.vector_norm(motion).item()) / radius,
        ],
        dtype=torch.float32,
        device=candidate_feature.device,
    )
    feature = torch.cat(
        [
            structured,
            channel_bin_means(support, 16),
            channel_bin_means(candidate, 16),
            channel_bin_means(support * candidate, 16),
        ]
    )
    if feature.numel() != CANDIDATE_FEATURE_DIM:
        raise RuntimeError(f"candidate feature dim drift: {feature.numel()}")
    return feature


def build_state_feature(
    *,
    native_xy_px: torch.Tensor,
    previous_native_xy_px: torch.Tensor,
    previous_previous_native_xy_px: torch.Tensor,
    visibility_probability: float,
    confidence_probability: float,
    frame_index: int,
    frame_count: int,
    query_frame: int,
    online_track_feature: torch.Tensor,
    input_height: int,
    input_width: int,
    radius_px: float,
) -> torch.Tensor:
    """Build the fixed 32-dimensional native-state feature contract."""
    current = native_xy_px.float()
    previous = previous_native_xy_px.float()
    previous_previous = previous_previous_native_xy_px.float()
    velocity = current - previous
    previous_velocity = previous - previous_previous
    acceleration = velocity - previous_velocity
    radius = max(float(radius_px), 1.0e-6)
    age_denom = float(max(frame_count - 1, 1))
    structured = torch.tensor(
        [
            float(current[0].item()) / float(max(input_width - 1, 1)),
            float(current[1].item()) / float(max(input_height - 1, 1)),
            float(previous[0].item()) / float(max(input_width - 1, 1)),
            float(previous[1].item()) / float(max(input_height - 1, 1)),
            float(velocity[0].item()) / radius,
            float(velocity[1].item()) / radius,
            float(torch.linalg.vector_norm(velocity).item()) / radius,
            float(acceleration[0].item()) / radius,
            float(acceleration[1].item()) / radius,
            float(torch.linalg.vector_norm(acceleration).item()) / radius,
            float(visibility_probability),
            float(confidence_probability),
            float(visibility_probability * confidence_probability),
            float(frame_index) / age_denom,
            float(max(frame_index - query_frame, 0)) / age_denom,
            1.0 if int(frame_index) == int(query_frame) else 0.0,
        ],
        dtype=torch.float32,
        device=online_track_feature.device,
    )
    feature = torch.cat(
        [structured, channel_bin_means(online_track_feature.float().flatten(), 16)]
    )
    if feature.numel() != STATE_FEATURE_DIM:
        raise RuntimeError(f"state feature dim drift: {feature.numel()}")
    return feature


def local_correlation_candidates(
    *,
    fmap: torch.Tensor,
    support_feature: torch.Tensor,
    native_xy_px: torch.Tensor,
    previous_native_xy_px: torch.Tensor,
    native_visibility_probability: float,
    native_confidence_probability: float,
    input_height: int,
    input_width: int,
    radius_px: float,
    topk: int,
) -> LocalSearchResult:
    """Generate native + deterministic local-correlation top-K candidates.

    The search is a square input-raster trust region around the native point.
    Similarity is cosine correlation against the query-frame support feature.
    Ties are resolved by ascending row-major feature-cell index through stable
    descending argsort.
    """
    if fmap.ndim != 3:
        raise ValueError("fmap must have shape (D,Hf,Wf)")
    if topk <= 0:
        raise ValueError("topk must be positive")
    device = fmap.device
    native = native_xy_px.to(device=device, dtype=torch.float32)
    previous = previous_native_xy_px.to(device=device, dtype=torch.float32)
    support = F.normalize(support_feature.to(device=device).float(), dim=0, eps=1.0e-12)
    _, feature_height, feature_width = fmap.shape

    native_feature = sample_feature_at_xy(
        fmap,
        native,
        input_height=input_height,
        input_width=input_width,
    )
    native_score = float(torch.dot(native_feature.float(), support).item())

    native_fx = float(native[0].item()) / float(max(input_width - 1, 1)) * float(
        max(feature_width - 1, 1)
    )
    native_fy = float(native[1].item()) / float(max(input_height - 1, 1)) * float(
        max(feature_height - 1, 1)
    )
    radius_fx = float(radius_px) / float(max(input_width - 1, 1)) * float(
        max(feature_width - 1, 1)
    )
    radius_fy = float(radius_px) / float(max(input_height - 1, 1)) * float(
        max(feature_height - 1, 1)
    )
    xmin = max(0, int(math.ceil(native_fx - radius_fx)))
    xmax = min(feature_width - 1, int(math.floor(native_fx + radius_fx)))
    ymin = max(0, int(math.ceil(native_fy - radius_fy)))
    ymax = min(feature_height - 1, int(math.floor(native_fy + radius_fy)))
    if xmin > xmax or ymin > ymax:
        xmin = xmax = int(round(max(0.0, min(feature_width - 1.0, native_fx))))
        ymin = ymax = int(round(max(0.0, min(feature_height - 1.0, native_fy))))

    patch = fmap[:, ymin : ymax + 1, xmin : xmax + 1].float()
    patch = F.normalize(patch, dim=0, eps=1.0e-12)
    scores = torch.einsum("dhw,d->hw", patch, support).reshape(-1)
    order = torch.argsort(scores, descending=True, stable=True)
    selected_count = min(int(topk), int(order.numel()))
    selected = order[:selected_count]
    sorted_scores = scores[order]
    peak = float(sorted_scores[0].item()) if sorted_scores.numel() else native_score
    margin = (
        float((sorted_scores[0] - sorted_scores[1]).item())
        if sorted_scores.numel() > 1
        else 0.0
    )
    entropy = _normalized_entropy(scores)

    total_candidates = 1 + int(topk)
    coords = native.new_zeros((total_candidates, 2))
    features = fmap.new_zeros((total_candidates, CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    candidate_scores = fmap.new_full((total_candidates,), float("-inf"), dtype=torch.float32)
    valid = torch.zeros(total_candidates, dtype=torch.bool, device=device)
    source_ids = torch.full(
        (total_candidates,), LOCAL_CORRELATION_SOURCE_ID, dtype=torch.long, device=device
    )

    coords[0] = native
    candidate_scores[0] = native_score
    valid[0] = True
    source_ids[0] = NATIVE_SOURCE_ID
    features[0] = build_candidate_feature(
        support_feature=support,
        candidate_feature=native_feature,
        candidate_score=native_score,
        local_peak=peak,
        local_margin=margin,
        local_entropy=entropy,
        rank=0,
        candidate_count=total_candidates,
        is_native=True,
        candidate_xy_px=native,
        native_xy_px=native,
        previous_native_xy_px=previous,
        native_visibility_probability=native_visibility_probability,
        native_confidence_probability=native_confidence_probability,
        radius_px=radius_px,
    )

    patch_width = xmax - xmin + 1
    for output_index, flat_index in enumerate(selected.tolist(), start=1):
        local_y = int(flat_index) // patch_width
        local_x = int(flat_index) % patch_width
        feature_x = xmin + local_x
        feature_y = ymin + local_y
        x_px = float(feature_x) / float(max(feature_width - 1, 1)) * float(
            max(input_width - 1, 1)
        )
        y_px = float(feature_y) / float(max(feature_height - 1, 1)) * float(
            max(input_height - 1, 1)
        )
        candidate_xy = native.new_tensor([x_px, y_px])
        candidate_feature = patch[:, local_y, local_x]
        score = float(scores[flat_index].item())
        coords[output_index] = candidate_xy
        candidate_scores[output_index] = score
        valid[output_index] = True
        features[output_index] = build_candidate_feature(
            support_feature=support,
            candidate_feature=candidate_feature,
            candidate_score=score,
            local_peak=peak,
            local_margin=margin,
            local_entropy=entropy,
            rank=output_index,
            candidate_count=total_candidates,
            is_native=False,
            candidate_xy_px=candidate_xy,
            native_xy_px=native,
            previous_native_xy_px=previous,
            native_visibility_probability=native_visibility_probability,
            native_confidence_probability=native_confidence_probability,
            radius_px=radius_px,
        )

    return LocalSearchResult(
        candidate_xy_px=coords,
        candidate_features=features,
        candidate_scores=candidate_scores,
        candidate_valid=valid,
        source_ids=source_ids,
        peak=peak,
        margin=margin,
        entropy=entropy,
        search_cell_count=int(scores.numel()),
    )


def oracle_candidate_indices(
    candidate_xy_px: torch.Tensor,
    candidate_valid: torch.Tensor,
    gt_xy_px: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the GT-nearest valid candidate for post-generation oracle audit."""
    if candidate_xy_px.shape[:-1] != candidate_valid.shape:
        raise ValueError("candidate_valid shape mismatch")
    if gt_xy_px.shape != candidate_xy_px.shape[:-2] + (2,):
        raise ValueError("gt_xy_px shape mismatch")
    error = torch.linalg.vector_norm(candidate_xy_px - gt_xy_px.unsqueeze(-2), dim=-1)
    masked = error.masked_fill(~candidate_valid, float("inf"))
    best_error, best_index = masked.min(dim=-1)
    native_error = error[..., 0]
    return best_index, best_error, native_error


def gather_candidate_coordinates(
    candidate_xy_px: torch.Tensor,
    candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Gather candidate coordinates along the penultimate candidate dimension."""
    if candidate_index.shape != candidate_xy_px.shape[:-2]:
        raise ValueError("candidate_index shape mismatch")
    return candidate_xy_px.gather(
        -2, candidate_index.unsqueeze(-1).unsqueeze(-1).expand(*candidate_index.shape, 1, 2)
    ).squeeze(-2)
