"""Counterfactual CoTracker3 online-state restoration primitives.

This module contains no learned parameters. It provides a strict clone/restore
contract for the state consumed by the next online window, deterministic
composite corruption, and future-state comparison utilities used by Gate 0.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .cotracker3_stage0_adapter import tensor_sha256


GATE0_SCHEMA_VERSION = "routeD_counterfactual_state_restoration_gate0_v0"


def _clone_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _clone_optional_sequence(
    values: Iterable[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(_clone_optional(value) for value in values)


@dataclass(frozen=True)
class CoTrackerOnlineStateSnapshot:
    predictor_n: int
    predictor_queries: torch.Tensor
    online_ind: int
    online_track_feat: tuple[torch.Tensor | None, ...]
    online_track_support: tuple[torch.Tensor | None, ...]
    online_coords_predicted: torch.Tensor
    online_vis_predicted: torch.Tensor
    online_conf_predicted: torch.Tensor


def snapshot_cotracker_online_state(predictor: Any) -> CoTrackerOnlineStateSnapshot:
    """Deep-clone every state field needed by the next CoTracker3 window."""
    model = predictor.model
    queries = getattr(predictor, "queries", None)
    if not isinstance(queries, torch.Tensor):
        raise ValueError("predictor queries are not initialized")
    required = {
        "online_coords_predicted": getattr(model, "online_coords_predicted", None),
        "online_vis_predicted": getattr(model, "online_vis_predicted", None),
        "online_conf_predicted": getattr(model, "online_conf_predicted", None),
    }
    if any(not isinstance(value, torch.Tensor) for value in required.values()):
        raise ValueError("CoTracker online prediction state is incomplete")
    track_feat = getattr(model, "online_track_feat", None)
    track_support = getattr(model, "online_track_support", None)
    if not isinstance(track_feat, (list, tuple)) or not isinstance(
        track_support, (list, tuple)
    ):
        raise ValueError("CoTracker online track memories are incomplete")
    if len(track_feat) != len(track_support) or not track_feat:
        raise ValueError("CoTracker online memory level mismatch")
    return CoTrackerOnlineStateSnapshot(
        predictor_n=int(predictor.N),
        predictor_queries=queries.detach().clone(),
        online_ind=int(model.online_ind),
        online_track_feat=_clone_optional_sequence(track_feat),
        online_track_support=_clone_optional_sequence(track_support),
        online_coords_predicted=required["online_coords_predicted"].detach().clone(),
        online_vis_predicted=required["online_vis_predicted"].detach().clone(),
        online_conf_predicted=required["online_conf_predicted"].detach().clone(),
    )


def restore_cotracker_online_state(
    predictor: Any, snapshot: CoTrackerOnlineStateSnapshot
) -> None:
    """Restore a snapshot without sharing storage with the frozen copy."""
    predictor.N = int(snapshot.predictor_n)
    predictor.queries = snapshot.predictor_queries.detach().clone()
    model = predictor.model
    model.online_ind = int(snapshot.online_ind)
    model.online_track_feat = [
        _clone_optional(value) for value in snapshot.online_track_feat
    ]
    model.online_track_support = [
        _clone_optional(value) for value in snapshot.online_track_support
    ]
    model.online_coords_predicted = snapshot.online_coords_predicted.detach().clone()
    model.online_vis_predicted = snapshot.online_vis_predicted.detach().clone()
    model.online_conf_predicted = snapshot.online_conf_predicted.detach().clone()


def snapshot_tensor_hashes(
    snapshot: CoTrackerOnlineStateSnapshot,
) -> dict[str, Any]:
    def rows(values: tuple[torch.Tensor | None, ...]) -> list[str | None]:
        return [None if value is None else tensor_sha256(value) for value in values]

    return {
        "predictor_n": int(snapshot.predictor_n),
        "predictor_queries": tensor_sha256(snapshot.predictor_queries),
        "online_ind": int(snapshot.online_ind),
        "online_track_feat": rows(snapshot.online_track_feat),
        "online_track_support": rows(snapshot.online_track_support),
        "online_coords_predicted": tensor_sha256(snapshot.online_coords_predicted),
        "online_vis_predicted": tensor_sha256(snapshot.online_vis_predicted),
        "online_conf_predicted": tensor_sha256(snapshot.online_conf_predicted),
    }


def snapshots_exact(
    left: CoTrackerOnlineStateSnapshot, right: CoTrackerOnlineStateSnapshot
) -> bool:
    if left.predictor_n != right.predictor_n or left.online_ind != right.online_ind:
        return False
    if not torch.equal(left.predictor_queries, right.predictor_queries):
        return False
    for left_values, right_values in (
        (left.online_track_feat, right.online_track_feat),
        (left.online_track_support, right.online_track_support),
    ):
        if len(left_values) != len(right_values):
            return False
        for lhs, rhs in zip(left_values, right_values):
            if lhs is None or rhs is None:
                if lhs is not rhs:
                    return False
            elif not torch.equal(lhs, rhs):
                return False
    return bool(
        torch.equal(left.online_coords_predicted, right.online_coords_predicted)
        and torch.equal(left.online_vis_predicted, right.online_vis_predicted)
        and torch.equal(left.online_conf_predicted, right.online_conf_predicted)
    )


def _channel_order(seed: int, level: int, channels: int) -> list[int]:
    rows = []
    for channel in range(channels):
        digest = hashlib.sha256(f"{seed}|{level}|{channel}".encode("utf-8")).digest()
        rows.append((digest, channel))
    return [channel for _, channel in sorted(rows)]


def composite_corrupt_snapshot(
    snapshot: CoTrackerOnlineStateSnapshot,
    *,
    input_height: int,
    input_width: int,
    interp_height: int,
    interp_width: int,
    overlap_start: int,
    overlap_end_inclusive: int,
    active_before_frame: int,
    coordinate_shift_input_xy_px: tuple[float, float],
    visibility_logit_delta: float,
    confidence_logit_delta: float,
    support_channel_keep_fraction: float,
    seed: int,
) -> CoTrackerOnlineStateSnapshot:
    """Apply the preregistered deterministic composite corruption."""
    if not 0.0 < support_channel_keep_fraction <= 1.0:
        raise ValueError("support keep fraction must be in (0,1]")
    coords = snapshot.online_coords_predicted.detach().clone()
    visibility = snapshot.online_vis_predicted.detach().clone()
    confidence = snapshot.online_conf_predicted.detach().clone()
    if coords.ndim != 4 or coords.shape[0] != 1 or coords.shape[-1] != 2:
        raise ValueError("unexpected CoTracker coordinate-state shape")
    if visibility.shape != coords.shape[:-1] or confidence.shape != coords.shape[:-1]:
        raise ValueError("CoTracker visibility/confidence-state shape mismatch")
    query_frames = snapshot.predictor_queries[0, :, 0].round().long()
    point_count = int(coords.shape[2])
    if query_frames.numel() != point_count:
        raise ValueError("query/state point-count mismatch")
    active = query_frames < int(active_before_frame)
    start = int(overlap_start)
    end = int(overlap_end_inclusive) + 1
    if start < 0 or end > coords.shape[1] or start >= end:
        raise ValueError("overlap range is outside online state")
    shift = coords.new_tensor(
        [
            float(coordinate_shift_input_xy_px[0])
            * float(interp_width - 1)
            / float(max(input_width - 1, 1)),
            float(coordinate_shift_input_xy_px[1])
            * float(interp_height - 1)
            / float(max(input_height - 1, 1)),
        ]
    )
    active4 = active.view(1, 1, point_count, 1)
    frame4 = torch.zeros(
        1, coords.shape[1], point_count, 1, dtype=torch.bool, device=coords.device
    )
    frame4[:, start:end] = active4
    coords = torch.where(frame4, coords + shift.view(1, 1, 1, 2), coords)
    active3 = frame4[..., 0]
    visibility = torch.where(
        active3, visibility + float(visibility_logit_delta), visibility
    )
    confidence = torch.where(
        active3, confidence + float(confidence_logit_delta), confidence
    )

    corrupted_support: list[torch.Tensor | None] = []
    for level, support in enumerate(snapshot.online_track_support):
        if support is None:
            corrupted_support.append(None)
            continue
        value = support.detach().clone()
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != point_count:
            raise ValueError("unexpected CoTracker track-support shape")
        channels = int(value.shape[-1])
        keep_count = max(1, int(round(channels * support_channel_keep_fraction)))
        keep = _channel_order(seed, level, channels)[:keep_count]
        channel_mask = torch.zeros(channels, dtype=torch.bool, device=value.device)
        channel_mask[keep] = True
        point_channel_mask = active[:, None] & (~channel_mask[None, :])
        value[0] = value[0].masked_fill(point_channel_mask[None, :, :], 0.0)
        corrupted_support.append(value)

    return CoTrackerOnlineStateSnapshot(
        predictor_n=snapshot.predictor_n,
        predictor_queries=snapshot.predictor_queries.detach().clone(),
        online_ind=snapshot.online_ind,
        online_track_feat=_clone_optional_sequence(snapshot.online_track_feat),
        online_track_support=tuple(corrupted_support),
        online_coords_predicted=coords,
        online_vis_predicted=visibility,
        online_conf_predicted=confidence,
    )


def restore_coordinates_only(
    corrupted: CoTrackerOnlineStateSnapshot,
    clean: CoTrackerOnlineStateSnapshot,
) -> CoTrackerOnlineStateSnapshot:
    if corrupted.online_coords_predicted.shape != clean.online_coords_predicted.shape:
        raise ValueError("coordinate state shape mismatch")
    return CoTrackerOnlineStateSnapshot(
        predictor_n=corrupted.predictor_n,
        predictor_queries=corrupted.predictor_queries.detach().clone(),
        online_ind=corrupted.online_ind,
        online_track_feat=_clone_optional_sequence(corrupted.online_track_feat),
        online_track_support=_clone_optional_sequence(corrupted.online_track_support),
        online_coords_predicted=clean.online_coords_predicted.detach().clone(),
        online_vis_predicted=corrupted.online_vis_predicted.detach().clone(),
        online_conf_predicted=corrupted.online_conf_predicted.detach().clone(),
    )


def future_difference(
    clean: CoTrackerOnlineStateSnapshot,
    other: CoTrackerOnlineStateSnapshot,
    *,
    future_start: int,
    future_end_inclusive: int,
    input_height: int,
    input_width: int,
    interp_height: int,
    interp_width: int,
) -> dict[str, float | int]:
    """Compare finalized future outputs on active post-query rows."""
    if clean.online_coords_predicted.shape != other.online_coords_predicted.shape:
        raise ValueError("future coordinate shape mismatch")
    start = int(future_start)
    end = int(future_end_inclusive) + 1
    coords_clean = clean.online_coords_predicted[0, start:end]
    coords_other = other.online_coords_predicted[0, start:end]
    query_frames = clean.predictor_queries[0, :, 0].round().long()
    frame_ids = torch.arange(start, end, device=query_frames.device)[:, None]
    active = frame_ids >= query_frames[None]
    scale = coords_clean.new_tensor(
        [
            float(input_width - 1) / float(max(interp_width - 1, 1)),
            float(input_height - 1) / float(max(interp_height - 1, 1)),
        ]
    )
    l2 = torch.linalg.vector_norm((coords_other - coords_clean) * scale, dim=-1)
    selected = l2[active]
    if selected.numel() == 0:
        raise ValueError("future comparison has no active rows")
    vis_clean = torch.sigmoid(clean.online_vis_predicted[0, start:end])
    vis_other = torch.sigmoid(other.online_vis_predicted[0, start:end])
    conf_clean = torch.sigmoid(clean.online_conf_predicted[0, start:end])
    conf_other = torch.sigmoid(other.online_conf_predicted[0, start:end])
    vis_diff = (vis_other - vis_clean).abs()[active]
    conf_diff = (conf_other - conf_clean).abs()[active]
    return {
        "rows": int(selected.numel()),
        "coordinate_mean_l2_px": float(selected.mean().item()),
        "coordinate_max_l2_px": float(selected.max().item()),
        "coordinate_fraction_above_1px": float((selected > 1.0).float().mean().item()),
        "visibility_mean_abs_probability_difference": float(vis_diff.mean().item()),
        "visibility_max_abs_probability_difference": float(vis_diff.max().item()),
        "confidence_mean_abs_probability_difference": float(conf_diff.mean().item()),
        "confidence_max_abs_probability_difference": float(conf_diff.max().item()),
    }
