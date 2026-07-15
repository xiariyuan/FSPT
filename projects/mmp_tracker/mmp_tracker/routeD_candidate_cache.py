"""Frozen candidate-cache contract for Route-D hypothesis scoring.

The cache contains candidate-level features and oracle assignments extracted
from a frozen MMP checkpoint.  It does not contain video tensors and does not
modify tracker outputs.  Coordinates are normalized ``[y, x]``; diagnostic
errors are stored in pixels using the source video resolution.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch

from .hypothesis_scorer import build_hypothesis_features


ROUTED_CACHE_FORMAT_VERSION = 2
ROUTED_CACHE_REQUIRED_KEYS = (
    "features",
    "candidate_points",
    "candidate_quality",
    "candidate_entropy",
    "candidate_valid_mask",
    "candidate_error_px",
    "gt_points",
    "oracle_index",
    "visible",
    "local_error_px",
    "oracle_error_px",
    "oracle_gain_px",
    "global_improves_local",
    "sample_id",
    "point_id",
    "frame_id",
    "query_frame",
)


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape={tuple(tensor.shape)}, expected={expected}")


def extract_routeD_candidate_cache_batch(
    info: Mapping[str, torch.Tensor],
    target_points: torch.Tensor,
    occluded: torch.Tensor,
    query_points: torch.Tensor,
    *,
    video_height: int,
    video_width: int,
    sample_ids: torch.Tensor | None = None,
    global_gain_margin_px: float = 1.0,
    exclude_query_frame: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract flattened candidate rows from one model batch.

    Eligible rows are active frames with finite GT/candidates. Query frames are
    excluded by default to avoid trivial self-observations. Both visible and
    occluded rows are retained and explicitly labelled.
    """
    candidate_points = info["hypothesis_candidate_points"]
    candidate_quality = info["hypothesis_candidate_quality"]
    candidate_entropy = info["hypothesis_candidate_entropy"]
    previous_points = info["hypothesis_previous_points"]
    previous_confidence = info["hypothesis_previous_confidence"]

    if candidate_points.ndim != 5 or candidate_points.shape[-1] != 2:
        raise ValueError("hypothesis_candidate_points must have shape (B,N,T,K,2)")
    batch, points, time, candidates, _ = candidate_points.shape
    _require_shape("candidate_quality", candidate_quality, (batch, points, time, candidates))
    _require_shape("candidate_entropy", candidate_entropy, (batch, points, time, candidates))
    _require_shape("previous_points", previous_points, (batch, points, time, 2))
    _require_shape("previous_confidence", previous_confidence, (batch, points, time))
    _require_shape("target_points", target_points, (batch, points, time, 2))
    _require_shape("occluded", occluded, (batch, points, time))
    _require_shape("query_points", query_points, (batch, points, 3))

    if video_height <= 1 or video_width <= 1:
        raise ValueError("video dimensions must exceed one pixel")
    if global_gain_margin_px < 0:
        raise ValueError("global_gain_margin_px must be non-negative")

    local_points = candidate_points[..., 0, :]
    features = build_hypothesis_features(
        candidate_points,
        candidate_quality,
        local_points,
        previous_points,
        previous_confidence,
        candidate_entropy,
    )

    candidate_valid_mask = torch.isfinite(candidate_points).all(dim=-1)
    finite_row = candidate_valid_mask.all(dim=-1)
    finite_row = finite_row & torch.isfinite(target_points).all(dim=-1)
    query_frame = query_points[..., 0].round().long().clamp(0, time - 1)
    frame_id = torch.arange(time, device=query_points.device).view(1, 1, time)
    active = frame_id >= query_frame.unsqueeze(-1)
    if exclude_query_frame:
        active = frame_id > query_frame.unsqueeze(-1)
    eligible = active & finite_row

    scale = torch.tensor(
        [float(video_height - 1), float(video_width - 1)],
        device=candidate_points.device,
        dtype=candidate_points.dtype,
    )
    errors_px = torch.norm(
        (candidate_points - target_points.unsqueeze(-2)) * scale,
        dim=-1,
    )
    oracle_error_px, oracle_index = errors_px.min(dim=-1)
    local_error_px = errors_px[..., 0]
    oracle_gain_px = local_error_px - oracle_error_px
    global_improves_local = (
        (oracle_index > 0) & (oracle_gain_px >= float(global_gain_margin_px))
    )

    if sample_ids is None:
        sample_ids = torch.arange(batch, device=query_points.device, dtype=torch.long)
    else:
        sample_ids = sample_ids.to(device=query_points.device, dtype=torch.long)
        _require_shape("sample_ids", sample_ids, (batch,))
    sample_grid = sample_ids.view(batch, 1, 1).expand(batch, points, time)
    point_grid = torch.arange(points, device=query_points.device).view(1, points, 1)
    point_grid = point_grid.expand(batch, points, time)
    frame_grid = frame_id.expand(batch, points, time)
    query_grid = query_frame.unsqueeze(-1).expand(batch, points, time)

    def select(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[eligible].detach().cpu()

    return {
        "format_version": torch.tensor(ROUTED_CACHE_FORMAT_VERSION, dtype=torch.long),
        "features": select(features),
        "candidate_points": select(candidate_points),
        "candidate_quality": select(candidate_quality),
        "candidate_entropy": select(candidate_entropy),
        "candidate_valid_mask": select(candidate_valid_mask),
        "candidate_error_px": select(errors_px),
        "gt_points": select(target_points),
        "oracle_index": select(oracle_index).long(),
        "visible": select(~occluded.bool()).bool(),
        "local_error_px": select(local_error_px),
        "oracle_error_px": select(oracle_error_px),
        "oracle_gain_px": select(oracle_gain_px),
        "global_improves_local": select(global_improves_local).bool(),
        "sample_id": select(sample_grid).long(),
        "point_id": select(point_grid).long(),
        "frame_id": select(frame_grid).long(),
        "query_frame": select(query_grid).long(),
    }


def require_causal_routeD_candidate_cache(
    cache: Mapping[str, torch.Tensor],
) -> None:
    """Reject legacy caches whose previous confidence was not causal."""
    validate_routeD_candidate_cache(cache)
    version = cache.get("format_version")
    if not isinstance(version, torch.Tensor) or version.ndim != 0:
        raise ValueError("Route-D cache format_version must be a scalar tensor")
    if int(version.item()) < 2:
        raise ValueError(
            "Route-D cache format version 2+ is required for causal online parity; "
            "legacy version 1 stored post-update confidence."
        )


def validate_routeD_candidate_cache(cache: Mapping[str, torch.Tensor]) -> None:
    missing = [key for key in ROUTED_CACHE_REQUIRED_KEYS if key not in cache]
    if missing:
        raise ValueError(f"Route-D candidate cache missing keys: {missing}")
    rows = int(cache["oracle_index"].shape[0])
    for key in ROUTED_CACHE_REQUIRED_KEYS:
        tensor = cache[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cache[{key!r}] must be a tensor")
        if tensor.ndim == 0 or int(tensor.shape[0]) != rows:
            raise ValueError(f"cache[{key!r}] has inconsistent row count")
    if cache["features"].ndim != 3:
        raise ValueError("features must have shape (R,K,F)")
    if cache["candidate_points"].shape[:2] != cache["features"].shape[:2]:
        raise ValueError("candidate_points and features candidate dimensions differ")
    if cache["candidate_points"].shape[-1] != 2:
        raise ValueError("candidate_points must end in xy dimension 2")
    if cache["candidate_error_px"].shape != cache["features"].shape[:2]:
        raise ValueError("candidate_error_px must have shape (R,K)")
    if cache["candidate_valid_mask"].dtype is not torch.bool:
        raise TypeError("candidate_valid_mask must be boolean")
    if rows:
        k = cache["features"].shape[1]
        if int(cache["oracle_index"].min()) < 0 or int(cache["oracle_index"].max()) >= k:
            raise ValueError("oracle_index is outside candidate range")


def merge_routeD_candidate_cache_batches(
    batches: Iterable[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    batches = list(batches)
    if not batches:
        raise ValueError("At least one candidate-cache batch is required")
    for batch in batches:
        validate_routeD_candidate_cache(batch)
    merged: Dict[str, torch.Tensor] = {}
    for key in ROUTED_CACHE_REQUIRED_KEYS:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    merged["format_version"] = torch.tensor(ROUTED_CACHE_FORMAT_VERSION, dtype=torch.long)
    validate_routeD_candidate_cache(merged)
    return merged
