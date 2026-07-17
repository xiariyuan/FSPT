"""Training and evaluation for the causal Route-D temporal selector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn

from datasets.metrics import compute_tapvid_metrics

from .routeD_musr_training import (
    FeatureNormalization,
    _load_verified_artifact,
    _tracks_from_xy,
    _visible_error_stats,
    load_cache_index,
    normalize_inputs,
    paired_video_bootstrap_ci,
    selector_pretraining_loss,
    SelectorLossConfig,
    visible_post_query_mask,
)
from .routeD_recovery_network import RecoveryNetworkConfig
from .routeD_temporal_selector import CausalSetEvidenceTemporalSelector


@dataclass(frozen=True)
class TemporalBatch:
    candidate_features: torch.Tensor
    candidate_coords_px: torch.Tensor
    candidate_valid_mask: torch.Tensor
    state_features: torch.Tensor
    source_ids: torch.Tensor
    frame_valid_mask: torch.Tensor
    gt_coords_px: torch.Tensor
    point_indices: torch.Tensor
    frame_indices: torch.Tensor


def streaming_feature_normalization(
    cache_index_path: str | Path,
    *,
    expected_partition: str,
) -> tuple[FeatureNormalization, dict[str, Any]]:
    index = load_cache_index(cache_index_path, expected_partition=expected_partition)
    candidate_sum = candidate_sq = state_sum = state_sq = None
    candidate_count = 0
    state_count = 0
    for row in index["videos"]:
        tensors = _load_verified_artifact(row)["tensors"]
        mask = visible_post_query_mask(tensors)
        candidate = tensors["candidate_features"][mask].double()
        valid = tensors["candidate_valid_mask"][mask]
        candidate = candidate[valid]
        state = tensors["state_features"][mask].double()
        if candidate_sum is None:
            candidate_sum = torch.zeros(candidate.shape[-1], dtype=torch.float64)
            candidate_sq = torch.zeros_like(candidate_sum)
            state_sum = torch.zeros(state.shape[-1], dtype=torch.float64)
            state_sq = torch.zeros_like(state_sum)
        candidate_sum += candidate.sum(0)
        candidate_sq += candidate.square().sum(0)
        candidate_count += int(candidate.shape[0])
        state_sum += state.sum(0)
        state_sq += state.square().sum(0)
        state_count += int(state.shape[0])
    if candidate_count == 0 or state_count == 0:
        raise ValueError("empty fit normalization rows")
    candidate_mean = candidate_sum / candidate_count
    candidate_std = (candidate_sq / candidate_count - candidate_mean.square()).clamp_min(0).sqrt()
    state_mean = state_sum / state_count
    state_std = (state_sq / state_count - state_mean.square()).clamp_min(0).sqrt()
    return FeatureNormalization(
        candidate_mean.float(),
        candidate_std.float().clamp_min(1.0e-5),
        state_mean.float(),
        state_std.float().clamp_min(1.0e-5),
    ), index


def target_indices(tensors: Mapping[str, torch.Tensor], *, training: bool) -> torch.Tensor:
    points, frames = tensors["gt_occluded"].shape
    frame = torch.arange(frames).view(1, frames)
    query = tensors["query_points_tyx"][:, 0].round().long().view(points, 1)
    mask = frame > query
    if training:
        mask = mask & (~tensors["gt_occluded"])
    return torch.nonzero(mask, as_tuple=False)


def build_temporal_batch(
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    window_frames: int,
    raster: int = 256,
) -> TemporalBatch:
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("indices must contain point/frame pairs")
    point = indices[:, 0].long()
    current = indices[:, 1].long()
    offsets = torch.arange(-(window_frames - 1), 1).view(1, window_frames)
    raw_frame = current[:, None] + offsets
    query = tensors["query_points_tyx"][point, 0].round().long()
    frame_valid = (raw_frame >= query[:, None]) & (raw_frame >= 0)
    frame = raw_frame.clamp(0, tensors["gt_occluded"].shape[1] - 1)
    point_grid = point[:, None].expand_as(frame)

    candidate_features = tensors["candidate_features"][point_grid, frame].float()
    candidate_coords = tensors["candidate_coords_xy_px"][point_grid, frame].float()
    candidate_valid = tensors["candidate_valid_mask"][point_grid, frame].bool()
    state_features = tensors["state_features"][point_grid, frame].float()
    source_ids = tensors["source_ids"][point_grid, frame].long()
    candidate_features = torch.where(
        frame_valid[..., None, None], candidate_features, torch.zeros_like(candidate_features)
    )
    candidate_coords = torch.where(
        frame_valid[..., None, None], candidate_coords, torch.zeros_like(candidate_coords)
    )
    candidate_valid = candidate_valid & frame_valid[..., None]
    state_features = torch.where(
        frame_valid[..., None], state_features, torch.zeros_like(state_features)
    )
    source_ids = torch.where(
        frame_valid[..., None], source_ids, torch.zeros_like(source_ids)
    )
    gt = tensors["gt_tracks_yx"][point, current][..., [1, 0]].float() * float(raster - 1)
    return TemporalBatch(
        candidate_features,
        candidate_coords,
        candidate_valid,
        state_features,
        source_ids,
        frame_valid,
        gt,
        point,
        current,
    )


def _normalize_batch(batch: TemporalBatch, normalization: FeatureNormalization) -> TemporalBatch:
    shape = batch.candidate_features.shape
    flat_features = batch.candidate_features.reshape(-1, shape[-2], shape[-1])
    flat_valid = batch.candidate_valid_mask.reshape(-1, shape[-2])
    flat_state = batch.state_features.reshape(-1, batch.state_features.shape[-1])
    flat_features, flat_state = normalize_inputs(
        flat_features, flat_valid, flat_state, normalization
    )
    return TemporalBatch(
        flat_features.reshape(shape),
        batch.candidate_coords_px,
        batch.candidate_valid_mask,
        flat_state.reshape_as(batch.state_features),
        batch.source_ids,
        batch.frame_valid_mask,
        batch.gt_coords_px,
        batch.point_indices,
        batch.frame_indices,
    )


def iter_training_batches(
    cache_index_path: str | Path,
    normalization: FeatureNormalization,
    *,
    batch_size: int,
    window_frames: int,
    generator: torch.Generator,
) -> Iterator[TemporalBatch]:
    index = load_cache_index(cache_index_path, expected_partition="fit")
    video_order = torch.randperm(len(index["videos"]), generator=generator)
    for video_index in video_order.tolist():
        tensors = _load_verified_artifact(index["videos"][video_index])["tensors"]
        indices = target_indices(tensors, training=True)
        order = torch.randperm(indices.shape[0], generator=generator)
        indices = indices[order]
        for start in range(0, indices.shape[0], batch_size):
            yield _normalize_batch(
                build_temporal_batch(
                    tensors,
                    indices[start : start + batch_size],
                    window_frames=window_frames,
                ),
                normalization,
            )


def train_temporal_selector_one_epoch(
    model: CausalSetEvidenceTemporalSelector,
    cache_index_path: str | Path,
    normalization: FeatureNormalization,
    optimizer: torch.optim.Optimizer,
    network_config: RecoveryNetworkConfig,
    selector_config: SelectorLossConfig,
    *,
    device: str,
    batch_size: int,
    generator: torch.Generator,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in iter_training_batches(
        cache_index_path,
        normalization,
        batch_size=batch_size,
        window_frames=model.temporal_config.window_frames,
        generator=generator,
    ):
        output = model(
            batch.candidate_features.to(device),
            batch.candidate_coords_px.to(device),
            batch.candidate_valid_mask.to(device),
            batch.state_features.to(device),
            batch.source_ids.to(device),
            batch.frame_valid_mask.to(device),
        )
        current_coords = batch.candidate_coords_px[:, -1].to(device)
        current_valid = batch.candidate_valid_mask[:, -1].to(device)
        losses = selector_pretraining_loss(
            output,
            current_coords[:, None],
            current_valid[:, None],
            batch.gt_coords_px.to(device)[:, None],
            network_config,
            selector_config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("non-finite temporal selector loss")
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        rows = int(batch.gt_coords_px.shape[0])
        count += rows
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * rows
    return {key: value / max(count, 1) for key, value in totals.items()}


def _predict_video(
    model: CausalSetEvidenceTemporalSelector,
    tensors: Mapping[str, torch.Tensor],
    normalization: FeatureNormalization,
    *,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected_coords = tensors["native_coords_xy_px"].clone()
    indices = target_indices(tensors, training=False)
    selections = []
    all_indices = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.shape[0], batch_size):
            local_indices = indices[start : start + batch_size]
            batch = _normalize_batch(
                build_temporal_batch(
                    tensors,
                    local_indices,
                    window_frames=model.temporal_config.window_frames,
                ),
                normalization,
            )
            output = model(
                batch.candidate_features.to(device),
                batch.candidate_coords_px.to(device),
                batch.candidate_valid_mask.to(device),
                batch.state_features.to(device),
                batch.source_ids.to(device),
                batch.frame_valid_mask.to(device),
            )
            selection = output["selected_candidate_index"][:, 0].cpu()
            current_coords = batch.candidate_coords_px[:, -1]
            selected = current_coords.gather(
                1, selection[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            selected_coords[batch.point_indices, batch.frame_indices] = selected
            selections.append(selection)
            all_indices.append(local_indices)
    selection = torch.cat(selections)
    all_indices = torch.cat(all_indices)
    eval_mask = ~tensors["gt_occluded"][all_indices[:, 0], all_indices[:, 1]]
    eval_selection = selection[eval_mask]
    point = all_indices[eval_mask, 0]
    frame = all_indices[eval_mask, 1]
    coords = tensors["candidate_coords_xy_px"][point, frame]
    valid = tensors["candidate_valid_mask"][point, frame]
    gt = tensors["gt_tracks_yx"][point, frame][..., [1, 0]] * 255.0
    error = torch.linalg.vector_norm(coords - gt[:, None], dim=-1).masked_fill(~valid, float("inf"))
    thresholds = torch.tensor(model.config.thresholds_px)
    weights = torch.tensor(model.config.threshold_utility_weights)
    weights = weights / weights.sum()
    utility = ((error[..., None] <= thresholds) * weights).sum(-1).masked_fill(~valid, -1.0)
    native_utility = utility[:, 0]
    selected_utility = utility.gather(1, eval_selection[:, None]).squeeze(1)
    oracle = utility.argmax(-1)
    beneficial = utility[:, 1:].max(-1).values > native_utility
    selected_global = eval_selection > 0
    return selected_coords, {
        "rows": int(eval_selection.numel()),
        "selected_non_native": int(selected_global.sum()),
        "oracle_match": int((eval_selection == oracle).sum()),
        "utility_regret_sum": float((utility.max(-1).values - selected_utility).sum()),
        "harmful_selected": int((selected_global & (selected_utility < native_utility)).sum()),
        "beneficial_rows": int(beneficial.sum()),
        "beneficial_selected": int((beneficial & selected_global).sum()),
    }


def evaluate_temporal_selector(
    model: CausalSetEvidenceTemporalSelector,
    cache_index_path: str | Path,
    normalization: FeatureNormalization,
    *,
    device: str,
    batch_size: int = 512,
    raster: int = 256,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 3701,
) -> dict[str, Any]:
    index = load_cache_index(cache_index_path, expected_partition="model_validation")
    native_tracks = []
    selected_tracks = []
    gt_tracks = []
    visibility_rows = []
    gt_visibility_rows = []
    queries = []
    per_video = []
    behavior_counts = {
        "rows": 0,
        "selected_non_native": 0,
        "oracle_match": 0,
        "utility_regret_sum": 0.0,
        "harmful_selected": 0,
        "beneficial_rows": 0,
        "beneficial_selected": 0,
    }
    for row in index["videos"]:
        tensors = _load_verified_artifact(row)["tensors"]
        selected_xy, behavior = _predict_video(
            model, tensors, normalization, device=device, batch_size=batch_size
        )
        for key in behavior_counts:
            behavior_counts[key] += behavior[key]
        native_xy = tensors["native_coords_xy_px"]
        native_yx = _tracks_from_xy(native_xy, raster)
        selected_yx = _tracks_from_xy(selected_xy, raster)
        visibility = tensors["native_visibility"]
        gt_visibility = ~tensors["gt_occluded"]
        native_metrics = compute_tapvid_metrics(
            native_yx,
            tensors["gt_tracks_yx"],
            visibility,
            gt_visibility,
            tensors["query_points_tyx"],
            resolution=raster,
            query_mode="first",
        )
        selected_metrics = compute_tapvid_metrics(
            selected_yx,
            tensors["gt_tracks_yx"],
            visibility,
            gt_visibility,
            tensors["query_points_tyx"],
            resolution=raster,
            query_mode="first",
        )
        native_error = _visible_error_stats(native_xy, tensors, raster=raster)
        selected_error = _visible_error_stats(selected_xy, tensors, raster=raster)
        per_video.append(
            {
                "source_index": int(row["source_index"]),
                "video_name": row["sample_identity"]["video_name"],
                "AJ_gain_points": 100.0
                * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
                "delta_gain_points": 100.0
                * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
                "native_error": native_error,
                "selected_error": selected_error,
                "behavior": behavior,
            }
        )
        native_tracks.append(native_yx)
        selected_tracks.append(selected_yx)
        gt_tracks.append(tensors["gt_tracks_yx"])
        visibility_rows.append(visibility)
        gt_visibility_rows.append(gt_visibility)
        queries.append(tensors["query_points_tyx"])
    native_metrics = compute_tapvid_metrics(
        torch.cat(native_tracks),
        torch.cat(gt_tracks),
        torch.cat(visibility_rows),
        torch.cat(gt_visibility_rows),
        torch.cat(queries),
        resolution=raster,
        query_mode="first",
    )
    selected_metrics = compute_tapvid_metrics(
        torch.cat(selected_tracks),
        torch.cat(gt_tracks),
        torch.cat(visibility_rows),
        torch.cat(gt_visibility_rows),
        torch.cat(queries),
        resolution=raster,
        query_mode="first",
    )
    aj = [row["AJ_gain_points"] for row in per_video]
    delta = [row["delta_gain_points"] for row in per_video]
    native_severe = np.average(
        [row["native_error"]["severe_16px_rate"] for row in per_video],
        weights=[row["native_error"]["rows"] for row in per_video],
    )
    selected_severe = np.average(
        [row["selected_error"]["severe_16px_rate"] for row in per_video],
        weights=[row["selected_error"]["rows"] for row in per_video],
    )
    rows = max(int(behavior_counts["rows"]), 1)
    beneficial = max(int(behavior_counts["beneficial_rows"]), 1)
    behavior = {
        "selected_non_native_rate": behavior_counts["selected_non_native"] / rows,
        "oracle_utility_match_rate": behavior_counts["oracle_match"] / rows,
        "mean_utility_regret": behavior_counts["utility_regret_sum"] / rows,
        "harmful_global_selection_rate": behavior_counts["harmful_selected"] / rows,
        "beneficial_global_recall": behavior_counts["beneficial_selected"] / beneficial,
    }
    return {
        "partition": "model_validation",
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "selected_metrics": selected_metrics,
        "gain_points": {
            "AJ": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0
            * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0 * (float(selected_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "paired_video_AJ_gain_CI": paired_video_bootstrap_ci(
            aj, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "paired_video_delta_gain_CI": paired_video_bootstrap_ci(
            delta, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        "severe_16px_rate": {
            "native": float(native_severe),
            "selected": float(selected_severe),
            "delta": float(selected_severe - native_severe),
        },
        "behavior": behavior,
        "per_video": per_video,
        "cache_index_sha256": index["_index_sha256"],
    }
