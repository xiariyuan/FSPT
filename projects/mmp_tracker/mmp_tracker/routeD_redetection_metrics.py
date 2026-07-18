"""Official-compatible re-detection metrics and Route-D safety diagnostics.

The AJ_RD implementation follows the public TAPNext++ definition: eligible
reappearance events are record-breaking invisibility durations for each track;
for every eligible event, Jaccard is evaluated from its reappearance frame to
the end of the sequence; results are averaged over distance thresholds and
minimum invisibility durations while ignoring unavailable buckets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

OFFICIAL_AJRD_DISTANCE_THRESHOLDS = (1, 2, 4, 8, 16)
OFFICIAL_AJRD_MIN_DURATIONS = (1, 4, 16, 64, 256)


@dataclass(frozen=True)
class RedetectionEvent:
    batch_index: int
    frame_index: int
    point_index: int
    invisibility_duration: int


def count_consecutive_invisibility(gt_visible: torch.Tensor) -> torch.Tensor:
    """Count invisible frames immediately preceding every timestep."""
    if gt_visible.ndim != 3:
        raise ValueError("gt_visible must have shape (B,T,N)")
    visible = gt_visible.bool()
    output = torch.zeros_like(visible, dtype=torch.long)
    for frame in range(1, visible.shape[1]):
        output[:, frame] = torch.where(
            ~visible[:, frame - 1], output[:, frame - 1] + 1, 0
        )
    return output


def eligible_redetection_events(gt_visible: torch.Tensor) -> tuple[RedetectionEvent, ...]:
    """Return record-breaking events in official flattened tensor order."""
    visible = gt_visible.bool()
    if visible.ndim != 3:
        raise ValueError("gt_visible must have shape (B,T,N)")
    reappearance = torch.zeros_like(visible)
    reappearance[:, 1:] = visible[:, 1:] & ~visible[:, :-1]
    duration = count_consecutive_invisibility(visible)
    indices = torch.where(reappearance)
    candidates = [
        RedetectionEvent(
            int(indices[0][index].item()),
            int(indices[1][index].item()),
            int(indices[2][index].item()),
            int(duration[indices[0][index], indices[1][index], indices[2][index]].item()),
        )
        for index in range(indices[0].numel())
    ]
    by_track: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for index, event in enumerate(candidates):
        by_track.setdefault((event.batch_index, event.point_index), []).append(
            (event.frame_index, event.invisibility_duration, index)
        )
    eligible = [False] * len(candidates)
    for rows in by_track.values():
        maximum = -1
        for _, current, index in sorted(rows, key=lambda row: row[0]):
            if current > maximum:
                eligible[index] = True
                maximum = current
    return tuple(event for index, event in enumerate(candidates) if eligible[index])


def segment_jaccard(
    pred_track: torch.Tensor,
    pred_visible: torch.Tensor,
    gt_track: torch.Tensor,
    gt_visible: torch.Tensor,
    distance_threshold: float,
) -> torch.Tensor:
    """TAP-Vid Jaccard on one post-reappearance track segment."""
    within = torch.linalg.norm(pred_track - gt_track, dim=-1) <= float(
        distance_threshold
    )
    gt_vis = gt_visible.bool()
    pred_vis = pred_visible.bool()
    true_positive = torch.sum(within & gt_vis & pred_vis)
    false_positive = torch.sum((~gt_vis & pred_vis) | (~within & pred_vis))
    denominator = torch.sum(gt_vis) + false_positive
    if int(denominator.item()) == 0:
        return pred_track.new_tensor(float("nan"))
    return true_positive.float() / denominator


def compute_official_aj_rd(
    pred_tracks: torch.Tensor,
    pred_visible: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visible: torch.Tensor,
    *,
    distance_thresholds: Sequence[int] = OFFICIAL_AJRD_DISTANCE_THRESHOLDS,
    min_durations: Sequence[int] = OFFICIAL_AJRD_MIN_DURATIONS,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Compute official TAPNext++ AJ_RD on pixel-space tracks.

    Inputs use shape ``(B,T,N,2)``. Coordinates must already be in the pixel
    space in which the 1/2/4/8/16 thresholds are interpreted.
    """
    if pred_tracks.shape != gt_tracks.shape or pred_tracks.ndim != 4:
        raise ValueError("track tensors must share shape (B,T,N,2)")
    if pred_tracks.shape[-1] != 2:
        raise ValueError("last track dimension must be xy")
    if pred_visible.shape != pred_tracks.shape[:-1] or gt_visible.shape != pred_visible.shape:
        raise ValueError("visibility tensors must have shape (B,T,N)")
    events = eligible_redetection_events(gt_visible)
    metrics: dict[str, Any] = {}
    raw_by_distance: dict[int, torch.Tensor] = {}
    durations = torch.tensor(
        [event.invisibility_duration for event in events],
        dtype=torch.long,
        device=pred_tracks.device,
    )
    for distance in distance_thresholds:
        values = []
        for event in events:
            values.append(
                segment_jaccard(
                    pred_tracks[event.batch_index, event.frame_index :, event.point_index],
                    pred_visible[event.batch_index, event.frame_index :, event.point_index],
                    gt_tracks[event.batch_index, event.frame_index :, event.point_index],
                    gt_visible[event.batch_index, event.frame_index :, event.point_index],
                    distance,
                )
            )
        raw_by_distance[int(distance)] = (
            torch.stack(values) if values else pred_tracks.new_empty((0,))
        )

    duration_means: list[float] = []
    for minimum in min_durations:
        mask = durations >= int(minimum)
        threshold_means: list[float] = []
        for distance in distance_thresholds:
            values = raw_by_distance[int(distance)][mask]
            values = values[~torch.isnan(values)]
            value = float(values.mean().item()) if values.numel() else float("nan")
            metrics[f"AJ_RD_D{int(distance)}_dmin{int(minimum)}"] = value
            if not np.isnan(value):
                threshold_means.append(value)
        aggregate = float(np.mean(threshold_means)) if threshold_means else float("nan")
        metrics[f"AJ_RD_dmin{int(minimum)}"] = aggregate
        if not np.isnan(aggregate):
            duration_means.append(aggregate)
    metrics["AJ_RD"] = float(np.mean(duration_means)) if duration_means else float("nan")
    metrics["eligible_event_count"] = len(events)
    metrics["eligible_event_count_by_dmin"] = {
        str(int(minimum)): int((durations >= int(minimum)).sum().item())
        for minimum in min_durations
    }
    if include_raw:
        metrics["raw_events"] = [event.__dict__ for event in events]
        metrics["raw_by_distance"] = raw_by_distance
    return metrics


def recovery_latency_and_safety(
    pred_tracks: torch.Tensor,
    pred_visible: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visible: torch.Tensor,
    *,
    recovery_distance_px: float = 4.0,
    selected_non_native: torch.Tensor | None = None,
    native_tracks: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Report latency, false reacquisition and harmful intervention diagnostics."""
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError("predicted and GT tracks must have equal shape")
    visible = gt_visible.bool()
    predicted_visible = pred_visible.bool()
    events = eligible_redetection_events(visible)
    latencies: list[int] = []
    recovered = 0
    for event in events:
        error = torch.linalg.vector_norm(
            pred_tracks[event.batch_index, event.frame_index :, event.point_index]
            - gt_tracks[event.batch_index, event.frame_index :, event.point_index],
            dim=-1,
        )
        valid = (
            predicted_visible[event.batch_index, event.frame_index :, event.point_index]
            & visible[event.batch_index, event.frame_index :, event.point_index]
            & (error <= float(recovery_distance_px))
        )
        indices = torch.where(valid)[0]
        if indices.numel():
            recovered += 1
            latencies.append(int(indices[0].item()))
    false_reacquisition = predicted_visible & ~visible
    result: dict[str, float | int] = {
        "eligible_events": len(events),
        "recovered_events": recovered,
        "recovery_rate": recovered / max(len(events), 1),
        "mean_recovery_latency_frames": float(np.mean(latencies)) if latencies else float("nan"),
        "median_recovery_latency_frames": float(np.median(latencies)) if latencies else float("nan"),
        "false_reacquisition_rate": float(false_reacquisition.float().mean().item()),
    }
    if selected_non_native is not None:
        if selected_non_native.shape != visible.shape:
            raise ValueError("selected_non_native must have shape (B,T,N)")
        intervention = selected_non_native.bool()
        result["intervention_rate"] = float(intervention.float().mean().item())
        if native_tracks is not None:
            pred_error = torch.linalg.vector_norm(pred_tracks - gt_tracks, dim=-1)
            native_error = torch.linalg.vector_norm(native_tracks - gt_tracks, dim=-1)
            harmful = intervention & visible & (pred_error > native_error + 1e-6)
            result["harmful_intervention_rate"] = float(
                harmful.sum().item() / max(int((intervention & visible).sum().item()), 1)
            )
    return result
