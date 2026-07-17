"""Fit-only training and proposal-only evaluation for Route-D CMCP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from datasets.metrics import compute_tapvid_metrics

from .routeD_cmcp_feature_cache import (
    load_complete_feature_index,
    verify_feature_cache_artifact,
)
from .routeD_kubric_cache import file_sha256
from .routeD_musr_training import (
    _tracks_from_xy,
    _visible_error_stats,
    paired_video_bootstrap_ci,
    visible_post_query_mask,
)
from .routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    build_dense_proposal_targets,
    extract_proposal_candidates,
)


@dataclass(frozen=True)
class CMCPLossConfig:
    focal_weight: float = 1.0
    utility_weight: float = 0.5
    risk_weight: float = 0.25
    native_fallback_weight: float = 1.0
    no_harm_weight: float = 1.0
    temporal_consistency_weight: float = 0.05
    focal_gamma: float = 2.0
    focal_negative_beta: float = 4.0
    ranking_margin: float = 0.25
    gaussian_sigma_px: float = 4.0
    catastrophe_threshold_px: float = 16.0
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class CMCPVideoBundle:
    source_index: int
    video_name: str
    feature_maps: torch.Tensor
    tensors: Mapping[str, torch.Tensor]
    feature_sidecar: str
    base_sidecar: str


def load_cmcp_video(row: Mapping[str, Any]) -> CMCPVideoBundle:
    feature_path = Path(str(row["sidecar"])).resolve()
    if file_sha256(feature_path) != str(row["sidecar_sha256"]):
        raise ValueError(f"feature sidecar hash mismatch: {feature_path}")
    feature_artifact = verify_feature_cache_artifact(
        feature_path,
        expected_protocol_sha256=str(row["protocol_sha256"]),
        expected_base_sidecar_sha256=str(row["base_sidecar_sha256"]),
    )
    base_path = Path(str(row["base_sidecar"])).resolve()
    if file_sha256(base_path) != str(row["base_sidecar_sha256"]):
        raise ValueError(f"base sidecar hash mismatch: {base_path}")
    base_artifact = torch.load(base_path, map_location="cpu", weights_only=False)
    tensors = base_artifact["tensors"]
    if not torch.equal(
        tensors["candidate_coords_xy_px"][..., 0, :],
        tensors["native_coords_xy_px"],
    ):
        raise ValueError("base candidate-0/native parity drift")
    if feature_artifact["native_state_hashes"] != row["native_state_hashes"]:
        raise ValueError("feature-cache native-state provenance mismatch")
    return CMCPVideoBundle(
        source_index=int(row["source_index"]),
        video_name=str(row["sample_identity"]["video_name"]),
        feature_maps=feature_artifact["feature_maps_f16"],
        tensors=tensors,
        feature_sidecar=str(feature_path),
        base_sidecar=str(base_path),
    )


def _balanced_dense_focal_loss(
    logits: torch.Tensor,
    gaussian_target: torch.Tensor,
    supervised: torch.Tensor,
    *,
    gamma: float,
    negative_beta: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, gaussian_target, reduction="none")
    positive_weight = gaussian_target
    negative_weight = (1.0 - gaussian_target).pow(float(negative_beta))
    positive = positive_weight * (1.0 - probability).pow(float(gamma)) * bce
    negative = negative_weight * probability.pow(float(gamma)) * bce
    positive = positive.flatten(-2).sum(dim=-1) / positive_weight.flatten(-2).sum(dim=-1).clamp_min(1.0)
    negative = negative.flatten(-2).mean(dim=-1)
    per_frame = (positive + negative).squeeze(-1)
    mask = supervised.to(per_frame.dtype)
    return (per_frame * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_dense_mean(value: torch.Tensor, supervised: torch.Tensor) -> torch.Tensor:
    per_frame = value.flatten(-2).mean(dim=-1).squeeze(-1)
    mask = supervised.to(per_frame.dtype)
    return (per_frame * mask).sum() / mask.sum().clamp_min(1.0)


def _warp_previous_by_native_motion(
    previous: torch.Tensor,
    previous_native_xy: torch.Tensor,
    current_native_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    batch, _, height, width = previous.shape
    base_y, base_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=previous.device, dtype=previous.dtype),
        torch.linspace(-1.0, 1.0, width, device=previous.device, dtype=previous.dtype),
        indexing="ij",
    )
    delta = current_native_xy - previous_native_xy
    delta_x = 2.0 * delta[:, 0] / float(max(input_width - 1, 1))
    delta_y = 2.0 * delta[:, 1] / float(max(input_height - 1, 1))
    grid = torch.stack(
        [
            base_x[None].expand(batch, -1, -1) - delta_x[:, None, None],
            base_y[None].expand(batch, -1, -1) - delta_y[:, None, None],
        ],
        dim=-1,
    )
    return F.grid_sample(
        previous,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def cmcp_proposal_loss(
    output: Mapping[str, torch.Tensor],
    native_xy_px: torch.Tensor,
    gt_xy_px: torch.Tensor,
    visible_post_query: torch.Tensor,
    config: CMCPConfig,
    loss_config: CMCPLossConfig,
) -> dict[str, torch.Tensor]:
    if native_xy_px.shape != gt_xy_px.shape or native_xy_px.ndim != 3:
        raise ValueError("native/gt coordinates must have shape (B,T,2)")
    batch, frames = native_xy_px.shape[:2]
    score = output["proposal_score"]
    if score.shape[:2] != (batch, frames):
        raise ValueError("proposal output sequence shape mismatch")
    height, width = score.shape[-2:]
    target = build_dense_proposal_targets(
        gt_xy_px.reshape(-1, 2),
        native_xy_px.reshape(-1, 2),
        feature_height=height,
        feature_width=width,
        input_height=config.input_height,
        input_width=config.input_width,
        gaussian_sigma_px=loss_config.gaussian_sigma_px,
        catastrophe_threshold_px=loss_config.catastrophe_threshold_px,
    )
    shaped = {
        key: value.reshape(batch, frames, *value.shape[1:])
        for key, value in target.items()
    }
    focal = _balanced_dense_focal_loss(
        output["utility_logit"],
        shaped["gaussian_target"],
        visible_post_query,
        gamma=loss_config.focal_gamma,
        negative_beta=loss_config.focal_negative_beta,
    )
    utility_bce = F.binary_cross_entropy_with_logits(
        output["utility_logit"], shaped["utility_target"], reduction="none"
    )
    utility_scale = 1.0 + 4.0 * shaped["utility_target"]
    utility = _masked_dense_mean(utility_bce * utility_scale, visible_post_query)
    risk_bce = F.binary_cross_entropy_with_logits(
        output["risk_logit"], shaped["risk_target"], reduction="none"
    )
    risk = _masked_dense_mean(risk_bce, visible_post_query)
    fallback_target = shaped["native_fallback_target"].reshape(batch, frames)
    fallback_bce = F.binary_cross_entropy_with_logits(
        output["native_logit"], fallback_target, reduction="none"
    )
    supervised = visible_post_query.to(fallback_bce.dtype)
    fallback = (fallback_bce * supervised).sum() / supervised.sum().clamp_min(1.0)

    best_proposal = output["proposal_score"].flatten(-2).max(dim=-1).values.squeeze(-1)
    signed = torch.where(fallback_target > 0.5, 1.0, -1.0)
    ranking = F.softplus(
        float(loss_config.ranking_margin)
        - signed * (output["native_logit"] - best_proposal)
    )
    no_harm = (ranking * supervised).sum() / supervised.sum().clamp_min(1.0)

    if frames > 1:
        current_probability = torch.sigmoid(output["proposal_score"][:, 1:]).reshape(-1, 1, height, width)
        # Stop-gradient causal teacher: the previous proposal map is a fixed
        # temporal target, preventing mutual collapse and avoiding the
        # nondeterministic CUDA grid-sampler backward path.
        previous_probability = torch.sigmoid(
            output["proposal_score"][:, :-1]
        ).detach().reshape(-1, 1, height, width)
        warped = _warp_previous_by_native_motion(
            previous_probability,
            native_xy_px[:, :-1].reshape(-1, 2),
            native_xy_px[:, 1:].reshape(-1, 2),
            input_height=config.input_height,
            input_width=config.input_width,
        ).reshape(batch, frames - 1, 1, height, width)
        temporal_frame = (current_probability.reshape(batch, frames - 1, 1, height, width) - warped).abs().flatten(-2).mean(dim=-1).squeeze(-1)
        temporal_mask = (visible_post_query[:, 1:] & visible_post_query[:, :-1]).to(temporal_frame.dtype)
        temporal = (temporal_frame * temporal_mask).sum() / temporal_mask.sum().clamp_min(1.0)
    else:
        temporal = score.new_zeros(())

    total = (
        float(loss_config.focal_weight) * focal
        + float(loss_config.utility_weight) * utility
        + float(loss_config.risk_weight) * risk
        + float(loss_config.native_fallback_weight) * fallback
        + float(loss_config.no_harm_weight) * no_harm
        + float(loss_config.temporal_consistency_weight) * temporal
    )
    return {
        "loss": total,
        "focal": focal,
        "utility": utility,
        "risk": risk,
        "native_fallback": fallback,
        "no_harm": no_harm,
        "temporal_consistency": temporal,
    }


def _utility_from_error(error_px: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(
        (1.0, 2.0, 4.0, 8.0, 16.0), device=error_px.device, dtype=error_px.dtype
    )
    weights = torch.tensor(
        (0.28, 0.24, 0.20, 0.16, 0.12), device=error_px.device, dtype=error_px.dtype
    )
    return ((error_px[..., None] <= thresholds) * weights).sum(dim=-1)


def predict_cmcp_video(
    model: CausalMultiMemoryProposalGenerator,
    bundle: CMCPVideoBundle,
    config: CMCPConfig,
    *,
    device: str,
    point_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    tensors = bundle.tensors
    fmaps = bundle.feature_maps.to(device=device, dtype=torch.float32)
    native = tensors["native_coords_xy_px"].float()
    queries = tensors["query_points_tyx"].float()
    point_count, frames = native.shape[:2]
    all_coords = []
    all_scores = []
    all_valid = []
    all_selected = []
    model.eval()
    with torch.no_grad():
        for start in range(0, point_count, int(point_batch_size)):
            end = min(point_count, start + int(point_batch_size))
            correlation, motion, frame_valid = build_causal_multi_memory_correlations(
                fmaps,
                native[start:end].to(device),
                queries[start:end].to(device),
                input_height=config.input_height,
                input_width=config.input_width,
                ema_alpha=config.ema_alpha,
                motion_sigma_cells=config.motion_sigma_cells,
            )
            output = model.forward_sequence(correlation, motion, frame_valid)
            coords_frames = []
            scores_frames = []
            valid_frames = []
            selected_frames = []
            for frame_index in range(frames):
                proposal = extract_proposal_candidates(
                    output["proposal_score"][:, frame_index],
                    native[start:end, frame_index].to(device),
                    output["native_logit"][:, frame_index],
                    config,
                )
                active = frame_valid[:, frame_index]
                proposal["candidate_valid_mask"][:, 1:] &= active[:, None]
                scores = proposal["candidate_scores"].masked_fill(
                    ~proposal["candidate_valid_mask"], float("-inf")
                )
                selected = scores.argmax(dim=1)
                coords_frames.append(proposal["candidate_coords_xy_px"].cpu())
                scores_frames.append(scores.cpu())
                valid_frames.append(proposal["candidate_valid_mask"].cpu())
                selected_frames.append(selected.cpu())
            all_coords.append(torch.stack(coords_frames, dim=1))
            all_scores.append(torch.stack(scores_frames, dim=1))
            all_valid.append(torch.stack(valid_frames, dim=1))
            all_selected.append(torch.stack(selected_frames, dim=1))
    candidate_coords = torch.cat(all_coords, dim=0)
    candidate_scores = torch.cat(all_scores, dim=0)
    candidate_valid = torch.cat(all_valid, dim=0)
    selected_index = torch.cat(all_selected, dim=0)
    selected_xy = candidate_coords.gather(
        2, selected_index[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    gt_xy = tensors["gt_tracks_yx"][..., [1, 0]].float() * float(config.input_width - 1)
    error = torch.linalg.vector_norm(candidate_coords - gt_xy[..., None, :], dim=-1)
    error = error.masked_fill(~candidate_valid, float("inf"))
    oracle_index = error.argmin(dim=-1)
    oracle_xy = candidate_coords.gather(
        2, oracle_index[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    mask = visible_post_query_mask(tensors)
    selected_error = torch.linalg.vector_norm(selected_xy - gt_xy, dim=-1)
    native_error = torch.linalg.vector_norm(native - gt_xy, dim=-1)
    selected_utility = _utility_from_error(selected_error)
    native_utility = _utility_from_error(native_error)
    oracle_utility = _utility_from_error(error.min(dim=-1).values)
    selected_non_native = selected_index > 0
    harmful = selected_non_native & (selected_utility + 1.0e-6 < native_utility)
    beneficial_available = oracle_utility > native_utility + 1.0e-6
    beneficial_selected = selected_non_native & (selected_utility > native_utility + 1.0e-6)
    rows = int(mask.sum().item())
    behavior = {
        "rows": rows,
        "selected_non_native_rate": float(selected_non_native[mask].float().mean().item()),
        "harmful_non_native_rate": float(harmful[mask].float().mean().item()),
        "beneficial_candidate_available_rate": float(beneficial_available[mask].float().mean().item()),
        "beneficial_candidate_recall": float(
            beneficial_selected[mask].sum().item() / max(int(beneficial_available[mask].sum().item()), 1)
        ),
    }
    return {
        "candidate_coords_xy_px": candidate_coords,
        "candidate_scores": candidate_scores,
        "candidate_valid_mask": candidate_valid,
        "selected_candidate_index": selected_index,
        "selected_coords_xy_px": selected_xy,
        "oracle_candidate_index": oracle_index,
        "oracle_coords_xy_px": oracle_xy,
    }, behavior


def evaluate_cmcp_index(
    model: CausalMultiMemoryProposalGenerator,
    index_path: str | Path,
    config: CMCPConfig,
    *,
    expected_partition: str,
    device: str,
    point_batch_size: int,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 1701,
    max_videos: int = 0,
) -> dict[str, Any]:
    index = load_complete_feature_index(index_path, expected_partition=expected_partition)
    native_tracks: list[torch.Tensor] = []
    selected_tracks: list[torch.Tensor] = []
    oracle_tracks: list[torch.Tensor] = []
    gt_tracks: list[torch.Tensor] = []
    pred_visibility: list[torch.Tensor] = []
    gt_visibility: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    per_video: list[dict[str, Any]] = []
    behavior_totals = {"rows": 0, "selected_non_native": 0.0, "harmful": 0.0, "beneficial_available": 0.0, "beneficial_selected": 0.0}
    rows_to_evaluate = index["videos"][: int(max_videos)] if max_videos > 0 else index["videos"]
    for row in rows_to_evaluate:
        bundle = load_cmcp_video(row)
        prediction, behavior = predict_cmcp_video(
            model, bundle, config, device=device, point_batch_size=point_batch_size
        )
        tensors = bundle.tensors
        native_xy = tensors["native_coords_xy_px"].float()
        selected_xy = prediction["selected_coords_xy_px"]
        oracle_xy = prediction["oracle_coords_xy_px"]
        visibility = tensors["native_visibility"]
        gt_vis = ~tensors["gt_occluded"]
        metric_args = (tensors["gt_tracks_yx"], visibility, gt_vis, tensors["query_points_tyx"])
        native_metrics = compute_tapvid_metrics(_tracks_from_xy(native_xy, 256), *metric_args, resolution=256, query_mode="first")
        selected_metrics = compute_tapvid_metrics(_tracks_from_xy(selected_xy, 256), *metric_args, resolution=256, query_mode="first")
        oracle_metrics = compute_tapvid_metrics(_tracks_from_xy(oracle_xy, 256), *metric_args, resolution=256, query_mode="first")
        native_error = _visible_error_stats(native_xy, tensors, raster=256)
        selected_error = _visible_error_stats(selected_xy, tensors, raster=256)
        oracle_error = _visible_error_stats(oracle_xy, tensors, raster=256)
        per_video.append({
            "source_index": bundle.source_index,
            "video_name": bundle.video_name,
            "native_AJ": float(native_metrics["AJ"]),
            "selected_AJ": float(selected_metrics["AJ"]),
            "oracle_AJ": float(oracle_metrics["AJ"]),
            "selected_AJ_gain_points": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "oracle_AJ_gain_points": 100.0 * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
            "native_delta_average": float(native_metrics["<avg"]),
            "selected_delta_average": float(selected_metrics["<avg"]),
            "oracle_delta_average": float(oracle_metrics["<avg"]),
            "selected_delta_gain_points": 100.0 * (float(selected_metrics["<avg"])-float(native_metrics["<avg"])),
            "oracle_delta_gain_points": 100.0 * (float(oracle_metrics["<avg"])-float(native_metrics["<avg"])),
            "native_error": native_error,
            "selected_error": selected_error,
            "oracle_error": oracle_error,
            "behavior": behavior,
        })
        rows = behavior["rows"]
        behavior_totals["rows"] += rows
        behavior_totals["selected_non_native"] += behavior["selected_non_native_rate"] * rows
        behavior_totals["harmful"] += behavior["harmful_non_native_rate"] * rows
        behavior_totals["beneficial_available"] += behavior["beneficial_candidate_available_rate"] * rows
        behavior_totals["beneficial_selected"] += behavior["beneficial_candidate_recall"] * behavior["beneficial_candidate_available_rate"] * rows
        native_tracks.append(_tracks_from_xy(native_xy, 256))
        selected_tracks.append(_tracks_from_xy(selected_xy, 256))
        oracle_tracks.append(_tracks_from_xy(oracle_xy, 256))
        gt_tracks.append(tensors["gt_tracks_yx"])
        pred_visibility.append(visibility)
        gt_visibility.append(gt_vis)
        queries.append(tensors["query_points_tyx"])
    pooled_args = (
        torch.cat(gt_tracks), torch.cat(pred_visibility), torch.cat(gt_visibility), torch.cat(queries)
    )
    native_metrics = compute_tapvid_metrics(torch.cat(native_tracks), *pooled_args, resolution=256, query_mode="first")
    selected_metrics = compute_tapvid_metrics(torch.cat(selected_tracks), *pooled_args, resolution=256, query_mode="first")
    oracle_metrics = compute_tapvid_metrics(torch.cat(oracle_tracks), *pooled_args, resolution=256, query_mode="first")
    selected_aj = [row["selected_AJ_gain_points"] for row in per_video]
    oracle_aj = [row["oracle_AJ_gain_points"] for row in per_video]
    selected_delta = [row["selected_delta_gain_points"] for row in per_video]
    rows = max(int(behavior_totals["rows"]), 1)
    native_severe = np.average([r["native_error"]["severe_16px_rate"] for r in per_video], weights=[r["native_error"]["rows"] for r in per_video])
    selected_severe = np.average([r["selected_error"]["severe_16px_rate"] for r in per_video], weights=[r["selected_error"]["rows"] for r in per_video])
    oracle_severe = np.average([r["oracle_error"]["severe_16px_rate"] for r in per_video], weights=[r["oracle_error"]["rows"] for r in per_video])
    return {
        "partition": expected_partition,
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "selected_metrics": selected_metrics,
        "oracle_metrics": oracle_metrics,
        "selected_gain_points": {
            "AJ": 100.0 * (float(selected_metrics["AJ"])-float(native_metrics["AJ"])),
            "delta_average": 100.0 * (float(selected_metrics["<avg"])-float(native_metrics["<avg"])),
            "OA": 100.0 * (float(selected_metrics["OA"])-float(native_metrics["OA"])),
        },
        "oracle_gain_points": {
            "AJ": 100.0 * (float(oracle_metrics["AJ"])-float(native_metrics["AJ"])),
            "delta_average": 100.0 * (float(oracle_metrics["<avg"])-float(native_metrics["<avg"])),
            "OA": 100.0 * (float(oracle_metrics["OA"])-float(native_metrics["OA"])),
        },
        "paired_video_selected_AJ_gain_CI": paired_video_bootstrap_ci(selected_aj, seed=bootstrap_seed, samples=bootstrap_samples),
        "paired_video_oracle_AJ_gain_CI": paired_video_bootstrap_ci(oracle_aj, seed=bootstrap_seed+1, samples=bootstrap_samples),
        "paired_video_selected_delta_gain_CI": paired_video_bootstrap_ci(selected_delta, seed=bootstrap_seed+2, samples=bootstrap_samples),
        "severe_16px_rate": {
            "native": float(native_severe),
            "selected": float(selected_severe),
            "oracle": float(oracle_severe),
            "selected_delta": float(selected_severe-native_severe),
            "oracle_delta": float(oracle_severe-native_severe),
        },
        "behavior": {
            "rows": int(behavior_totals["rows"]),
            "selected_non_native_rate": behavior_totals["selected_non_native"] / rows,
            "harmful_non_native_rate": behavior_totals["harmful"] / rows,
            "beneficial_candidate_available_rate": behavior_totals["beneficial_available"] / rows,
            "beneficial_candidate_recall": behavior_totals["beneficial_selected"] / max(behavior_totals["beneficial_available"], 1.0),
        },
        "per_video": per_video,
        "cache_index_sha256": index["_index_sha256"],
    }


def train_cmcp_epoch(
    model: CausalMultiMemoryProposalGenerator,
    feature_index_path: str | Path,
    config: CMCPConfig,
    loss_config: CMCPLossConfig,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    point_batch_size: int,
    generator: torch.Generator,
    max_videos: int = 0,
) -> dict[str, float]:
    index = load_complete_feature_index(feature_index_path, expected_partition="fit")
    video_order = torch.randperm(len(index["videos"]), generator=generator).tolist()
    if max_videos > 0:
        video_order = video_order[: int(max_videos)]
    totals: dict[str, float] = {}
    supervised_rows = 0
    model.train()
    for video_id in video_order:
        bundle = load_cmcp_video(index["videos"][video_id])
        fmaps = bundle.feature_maps.to(device=device, dtype=torch.float32)
        tensors = bundle.tensors
        native = tensors["native_coords_xy_px"].float()
        queries = tensors["query_points_tyx"].float()
        gt_xy = tensors["gt_tracks_yx"][..., [1, 0]].float() * 255.0
        visible_mask = visible_post_query_mask(tensors)
        point_order = torch.randperm(native.shape[0], generator=generator)
        for start in range(0, native.shape[0], int(point_batch_size)):
            point_index = point_order[start:start+int(point_batch_size)]
            correlation, motion, frame_valid = build_causal_multi_memory_correlations(
                fmaps,
                native[point_index].to(device),
                queries[point_index].to(device),
                input_height=config.input_height,
                input_width=config.input_width,
                ema_alpha=config.ema_alpha,
                motion_sigma_cells=config.motion_sigma_cells,
            )
            output = model.forward_sequence(correlation, motion, frame_valid)
            losses = cmcp_proposal_loss(
                output,
                native[point_index].to(device),
                gt_xy[point_index].to(device),
                visible_mask[point_index].to(device),
                config,
                loss_config,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("non-finite CMCP loss")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(loss_config.grad_clip_norm))
            optimizer.step()
            rows = int(visible_mask[point_index].sum().item())
            supervised_rows += rows
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * rows
        del fmaps
    return {key: value / max(supervised_rows, 1) for key, value in totals.items()} | {
        "supervised_rows": float(supervised_rows),
        "videos": float(len(video_order)),
    }
