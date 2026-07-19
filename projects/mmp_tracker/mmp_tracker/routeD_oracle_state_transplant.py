"""Oracle fresh-query state transplantation for fit-only Gate 1."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from .routeD_counterfactual_state_restoration import CoTrackerOnlineStateSnapshot


GATE1_SCHEMA_VERSION = "routeD_oracle_state_transplant_gate1_v0"
THRESHOLDS_PX = (1.0, 2.0, 4.0, 8.0, 16.0)


def _clone_optional_sequence(
    values: Iterable[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if value is None else value.detach().clone() for value in values)


def transplant_fresh_state(
    native: CoTrackerOnlineStateSnapshot,
    fresh: CoTrackerOnlineStateSnapshot,
    *,
    selected_point_indices: torch.Tensor,
    oracle_query_frames: torch.Tensor,
    overlap_end_inclusive: int,
    copy_probability: bool,
    copy_track_memory: bool,
) -> CoTrackerOnlineStateSnapshot:
    """Copy selected fresh-query state into original point slots.

    Original predictor queries and query times remain native. Fresh temporal state
    is copied only from each oracle query frame through the end of the overlap.
    """
    selected = selected_point_indices.long().detach().cpu()
    oracle_frames = oracle_query_frames.long().detach().cpu()
    if selected.ndim != 1 or oracle_frames.shape != selected.shape:
        raise ValueError("selected indices and oracle frames must be one-dimensional")
    if fresh.predictor_n != int(selected.numel()):
        raise ValueError("fresh state point count must match selected points")
    if native.online_ind != fresh.online_ind:
        raise ValueError("native/fresh commit index mismatch")
    coords = native.online_coords_predicted.detach().clone()
    visibility = native.online_vis_predicted.detach().clone()
    confidence = native.online_conf_predicted.detach().clone()
    end = int(overlap_end_inclusive) + 1
    if end > coords.shape[1] or end > fresh.online_coords_predicted.shape[1]:
        raise ValueError("overlap end exceeds state frame count")
    for fresh_index, (native_index, start) in enumerate(
        zip(selected.tolist(), oracle_frames.tolist())
    ):
        if start < 0 or start >= end:
            raise ValueError("oracle query frame outside transplant overlap")
        coords[:, start:end, native_index] = fresh.online_coords_predicted[
            :, start:end, fresh_index
        ]
        if copy_probability:
            visibility[:, start:end, native_index] = fresh.online_vis_predicted[
                :, start:end, fresh_index
            ]
            confidence[:, start:end, native_index] = fresh.online_conf_predicted[
                :, start:end, fresh_index
            ]

    track_feat = _clone_optional_sequence(native.online_track_feat)
    track_support = _clone_optional_sequence(native.online_track_support)
    if copy_track_memory:
        feat_rows: list[torch.Tensor | None] = []
        support_rows: list[torch.Tensor | None] = []
        if len(track_feat) != len(fresh.online_track_feat) or len(track_support) != len(
            fresh.online_track_support
        ):
            raise ValueError("native/fresh memory level mismatch")
        for native_value, fresh_value in zip(track_feat, fresh.online_track_feat):
            if native_value is None or fresh_value is None:
                if native_value is not fresh_value:
                    raise ValueError("native/fresh track-feature availability mismatch")
                feat_rows.append(None)
                continue
            value = native_value.detach().clone()
            if value.ndim != 4 or value.shape[2] != native.predictor_n:
                raise ValueError("unexpected native track-feature shape")
            if fresh_value.shape[2] != fresh.predictor_n:
                raise ValueError("unexpected fresh track-feature shape")
            selected_device = selected.to(value.device)
            value[:, :, selected_device] = fresh_value.detach().to(value.device)[:, :, :]
            feat_rows.append(value)
        for native_value, fresh_value in zip(track_support, fresh.online_track_support):
            if native_value is None or fresh_value is None:
                if native_value is not fresh_value:
                    raise ValueError("native/fresh track-support availability mismatch")
                support_rows.append(None)
                continue
            value = native_value.detach().clone()
            if value.ndim != 4 or value.shape[2] != native.predictor_n:
                raise ValueError("unexpected native track-support shape")
            if fresh_value.shape[2] != fresh.predictor_n:
                raise ValueError("unexpected fresh track-support shape")
            selected_device = selected.to(value.device)
            value[:, :, selected_device] = fresh_value.detach().to(value.device)[:, :, :]
            support_rows.append(value)
        track_feat = tuple(feat_rows)
        track_support = tuple(support_rows)

    return CoTrackerOnlineStateSnapshot(
        predictor_n=native.predictor_n,
        predictor_queries=native.predictor_queries.detach().clone(),
        online_ind=native.online_ind,
        online_track_feat=track_feat,
        online_track_support=track_support,
        online_coords_predicted=coords,
        online_vis_predicted=visibility,
        online_conf_predicted=confidence,
    )


def select_failure_points(
    *,
    native_coords_xy_px: torch.Tensor,
    gt_tracks_yx: torch.Tensor,
    gt_occluded: torch.Tensor,
    original_query_frames: torch.Tensor,
    overlap_start: int,
    overlap_end_inclusive: int,
    future_start: int,
    future_end_inclusive: int,
    min_overlap_visible_frames: int,
    min_future_visible_frames: int,
    min_native_future_mean_error_px: float,
    per_video_cap: int,
) -> dict[str, torch.Tensor]:
    """Apply the preregistered natural-failure selection rule."""
    native = native_coords_xy_px.float().cpu()
    gt_xy = gt_tracks_yx[..., [1, 0]].float().cpu() * 255.0
    occluded = gt_occluded.bool().cpu()
    query = original_query_frames.round().long().cpu()
    if native.shape != gt_xy.shape:
        raise ValueError("native/GT track shape mismatch")
    overlap = slice(int(overlap_start), int(overlap_end_inclusive) + 1)
    future = slice(int(future_start), int(future_end_inclusive) + 1)
    overlap_visible = (~occluded[:, overlap]).sum(dim=1)
    future_visible = ~occluded[:, future]
    future_visible_count = future_visible.sum(dim=1)
    error = torch.linalg.vector_norm(native[:, future] - gt_xy[:, future], dim=-1)
    masked_sum = (error * future_visible.float()).sum(dim=1)
    mean_error = masked_sum / future_visible_count.clamp_min(1)
    eligible = (
        (query < int(overlap_start))
        & (overlap_visible >= int(min_overlap_visible_frames))
        & (future_visible_count >= int(min_future_visible_frames))
        & (mean_error >= float(min_native_future_mean_error_px))
    )
    indices = torch.where(eligible)[0]
    order = sorted(indices.tolist(), key=lambda index: (-float(mean_error[index]), index))
    selected = torch.tensor(order[: int(per_video_cap)], dtype=torch.long)
    latest_visible = torch.full_like(selected, -1)
    for ordinal, point_index in enumerate(selected.tolist()):
        visible_frames = torch.where(~occluded[point_index, overlap])[0]
        latest_visible[ordinal] = int(overlap_start) + int(visible_frames[-1].item())
    return {
        "selected_point_indices": selected,
        "oracle_query_frames": latest_visible,
        "native_future_mean_error_px": mean_error[selected],
        "eligible_mask": eligible,
        "future_visible_count": future_visible_count[selected],
    }


def point_future_metrics(
    *,
    final_coords_xy_model: torch.Tensor,
    selected_point_indices: torch.Tensor,
    gt_tracks_yx: torch.Tensor,
    gt_occluded: torch.Tensor,
    future_start: int,
    future_end_inclusive: int,
    interp_height: int,
    interp_width: int,
    input_height: int = 256,
    input_width: int = 256,
) -> list[dict[str, float | int]]:
    coords = final_coords_xy_model[0].float().cpu().clone()
    coords[..., 0] *= float(input_width - 1) / float(max(interp_width - 1, 1))
    coords[..., 1] *= float(input_height - 1) / float(max(interp_height - 1, 1))
    gt_xy = gt_tracks_yx[..., [1, 0]].float().cpu() * 255.0
    occluded = gt_occluded.bool().cpu()
    start, end = int(future_start), int(future_end_inclusive) + 1
    rows: list[dict[str, float | int]] = []
    thresholds = torch.tensor(THRESHOLDS_PX)
    for point_index in selected_point_indices.long().tolist():
        visible = ~occluded[point_index, start:end]
        if not visible.any():
            raise ValueError("selected point has no visible future rows")
        error = torch.linalg.vector_norm(
            coords[start:end, point_index] - gt_xy[point_index, start:end], dim=-1
        )[visible]
        utility = (error[:, None] <= thresholds[None]).float().mean().item()
        rows.append(
            {
                "point_index": int(point_index),
                "visible_rows": int(error.numel()),
                "mean_l2_error_px": float(error.mean().item()),
                "severe_16px_rate": float((error > 16.0).float().mean().item()),
                "threshold_utility": float(utility),
            }
        )
    return rows


def support_delta_svd_energy(
    native: CoTrackerOnlineStateSnapshot,
    fresh: CoTrackerOnlineStateSnapshot,
    *,
    selected_point_indices: torch.Tensor,
    ranks: Iterable[int],
) -> dict[str, Any]:
    """Report low-rank energy of fresh-minus-native support matrices."""
    selected = selected_point_indices.long().cpu()
    rank_values = tuple(int(rank) for rank in ranks)
    levels = []
    pooled_numerator = {rank: 0.0 for rank in rank_values}
    pooled_denominator = 0.0
    for level, (native_support, fresh_support) in enumerate(
        zip(native.online_track_support, fresh.online_track_support)
    ):
        if native_support is None or fresh_support is None:
            raise ValueError("support SVD requires all memory levels")
        per_point = []
        for fresh_index, native_index in enumerate(selected.tolist()):
            delta = (
                fresh_support[0, :, fresh_index].float().cpu()
                - native_support[0, :, native_index].float().cpu()
            )
            singular = torch.linalg.svdvals(delta)
            energy = singular.square()
            denominator = float(energy.sum().item())
            fractions = {}
            for rank in rank_values:
                numerator = float(energy[: min(rank, energy.numel())].sum().item())
                fractions[str(rank)] = 1.0 if denominator <= 1e-12 else numerator / denominator
                pooled_numerator[rank] += numerator
            pooled_denominator += denominator
            per_point.append(
                {
                    "point_index": int(native_index),
                    "delta_frobenius": float(torch.linalg.vector_norm(delta).item()),
                    "rank_energy": fractions,
                }
            )
        levels.append({"level": level, "points": per_point})
    return {
        "levels": levels,
        "pooled_total_energy": pooled_denominator,
        "pooled_rank_energy_numerator": {
            str(rank): pooled_numerator[rank] for rank in rank_values
        },
        "pooled_rank_energy": {
            str(rank): (
                1.0
                if pooled_denominator <= 1e-12
                else pooled_numerator[rank] / pooled_denominator
            )
            for rank in rank_values
        },
    }


def aggregate_variant_metrics(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("variant metrics must be non-empty")
    return {
        "points": len(rows),
        "mean_l2_error_px": float(np.mean([row["mean_l2_error_px"] for row in rows])),
        "severe_16px_rate": float(np.mean([row["severe_16px_rate"] for row in rows])),
        "threshold_utility": float(np.mean([row["threshold_utility"] for row in rows])),
    }
