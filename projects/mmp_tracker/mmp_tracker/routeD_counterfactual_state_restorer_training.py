"""Training and future-rollout evaluation helpers for Route-D CSRR Gate 2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .routeD_counterfactual_state_restoration import CoTrackerOnlineStateSnapshot
from .routeD_counterfactual_state_restorer_cache import (
    snapshot_from_cache_dict,
    verify_csrr_cache_artifact,
)


@dataclass(frozen=True)
class CSRRBatchLossConfig:
    failure_heatmap_focal: float = 1.0
    failure_coordinate_smooth_l1: float = 2.0
    teacher_track_feature_cosine: float = 0.5
    teacher_support_cosine: float = 1.0
    teacher_visibility_mse: float = 0.25
    teacher_confidence_mse: float = 0.25
    apply_noop_bce: float = 1.0
    clean_coordinate_displacement: float = 2.0
    clean_memory_change: float = 1.0
    coordinate_smooth_l1_beta_px: float = 4.0
    target_heatmap_sigma_grid_cells: float = 1.5


def load_csrr_cache_video(
    row: Mapping[str, Any],
    *,
    expected_partition: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    return verify_csrr_cache_artifact(
        row["sidecar"],
        expected_partition=expected_partition,
        expected_source_index=int(row["source_index"]),
        expected_config_sha256=expected_config_sha256,
    )


def snapshot_to_device(
    snapshot: CoTrackerOnlineStateSnapshot, device: torch.device | str
) -> CoTrackerOnlineStateSnapshot:
    return CoTrackerOnlineStateSnapshot(
        predictor_n=snapshot.predictor_n,
        predictor_queries=snapshot.predictor_queries.to(device),
        online_ind=snapshot.online_ind,
        online_track_feat=tuple(
            None if value is None else value.to(device) for value in snapshot.online_track_feat
        ),
        online_track_support=tuple(
            None if value is None else value.to(device)
            for value in snapshot.online_track_support
        ),
        online_coords_predicted=snapshot.online_coords_predicted.to(device),
        online_vis_predicted=snapshot.online_vis_predicted.to(device),
        online_conf_predicted=snapshot.online_conf_predicted.to(device),
    )


def exact_snapshot_from_artifact(
    artifact: Mapping[str, Any], device: torch.device | str
) -> CoTrackerOnlineStateSnapshot:
    return snapshot_to_device(snapshot_from_cache_dict(artifact["exact_native_state"]), device)


def gaussian_heatmap_targets(
    normalized_xy: torch.Tensor,
    *,
    height: int = 64,
    width: int = 64,
    sigma: float = 1.5,
) -> torch.Tensor:
    if normalized_xy.ndim != 2 or normalized_xy.shape[-1] != 2:
        raise ValueError("normalized_xy must have shape [B,2]")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=normalized_xy.device, dtype=normalized_xy.dtype),
        torch.arange(width, device=normalized_xy.device, dtype=normalized_xy.dtype),
        indexing="ij",
    )
    center_x = normalized_xy[:, 0].clamp(0, 1) * float(width - 1)
    center_y = normalized_xy[:, 1].clamp(0, 1) * float(height - 1)
    distance_squared = (
        (xx[None] - center_x[:, None, None]) ** 2
        + (yy[None] - center_y[:, None, None]) ** 2
    )
    return torch.exp(-distance_squared / (2.0 * float(sigma) ** 2))


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    positive = targets
    negative = 1.0 - targets
    positive_loss = -positive * (1.0 - probability).square() * torch.log(
        probability.clamp_min(1.0e-8)
    )
    negative_weight = negative.pow(4)
    negative_loss = -negative_weight * probability.square() * torch.log(
        (1.0 - probability).clamp_min(1.0e-8)
    )
    return (positive_loss + negative_loss).mean()


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    return 1.0 - F.cosine_similarity(left_flat, right_flat, dim=-1)


def csrr_batch_loss(
    *,
    output: Mapping[str, torch.Tensor],
    reextracted_track_features: Sequence[torch.Tensor],
    reextracted_track_supports: Sequence[torch.Tensor],
    batch: Mapping[str, Any],
    config: CSRRBatchLossConfig,
) -> dict[str, torch.Tensor]:
    target = batch["apply_target"].float()
    failure = target > 0.5
    clean = ~failure
    predicted_xy = output["predicted_coordinates_xy"].float()
    teacher_normalized = batch["teacher_commit_coordinates_normalized_xy"].float()
    native_normalized = batch["native_commit_coordinates_normalized_xy"].float()
    teacher_xy = teacher_normalized * 255.0
    native_xy = native_normalized * 255.0
    zero = predicted_xy.sum() * 0.0

    if failure.any():
        heatmap_target = gaussian_heatmap_targets(
            teacher_normalized[failure],
            height=output["fused_logits"].shape[-2],
            width=output["fused_logits"].shape[-1],
            sigma=config.target_heatmap_sigma_grid_cells,
        )
        heatmap = focal_heatmap_loss(output["fused_logits"][failure], heatmap_target)
        coordinate = F.smooth_l1_loss(
            predicted_xy[failure],
            teacher_xy[failure],
            beta=config.coordinate_smooth_l1_beta_px,
        )
    else:
        heatmap = zero
        coordinate = zero

    feature_distances = []
    support_distances = []
    clean_memory_distances = []
    for level in range(4):
        predicted_feat = reextracted_track_features[level][:, 0, 0]
        predicted_support = reextracted_track_supports[level][:, :, 0]
        teacher_feat = batch["teacher_track_feat"][level][:, 0].float()
        teacher_support = batch["teacher_track_support"][level].float()
        native_feat = batch["native_track_feat"][level][:, 0].float()
        native_support = batch["native_track_support"][level].float()
        if failure.any():
            feature_distances.append(
                cosine_distance(predicted_feat[failure], teacher_feat[failure]).mean()
            )
            support_distances.append(
                cosine_distance(predicted_support[failure], teacher_support[failure]).mean()
            )
        if clean.any():
            clean_memory_distances.append(
                0.5
                * (
                    cosine_distance(predicted_feat[clean], native_feat[clean]).mean()
                    + cosine_distance(
                        predicted_support[clean], native_support[clean]
                    ).mean()
                )
            )
    feature_loss = torch.stack(feature_distances).mean() if feature_distances else zero
    support_loss = torch.stack(support_distances).mean() if support_distances else zero
    clean_memory = (
        torch.stack(clean_memory_distances).mean() if clean_memory_distances else zero
    )

    native_vis = batch["native_visibility_probability"].float()
    native_conf = batch["native_confidence_probability"].float()
    teacher_vis = batch["teacher_visibility_probability"].float()
    teacher_conf = batch["teacher_confidence_probability"].float()
    predicted_vis = (native_vis + output["visibility_residual"]).clamp(0.0, 1.0)
    predicted_conf = (native_conf + output["confidence_residual"]).clamp(0.0, 1.0)
    visibility = F.mse_loss(predicted_vis, teacher_vis)
    confidence = F.mse_loss(predicted_conf, teacher_conf)
    apply = F.binary_cross_entropy_with_logits(output["apply_logit"], target)
    clean_coordinate = (
        F.smooth_l1_loss(
            predicted_xy[clean],
            native_xy[clean],
            beta=config.coordinate_smooth_l1_beta_px,
        )
        if clean.any()
        else zero
    )
    total = (
        config.failure_heatmap_focal * heatmap
        + config.failure_coordinate_smooth_l1 * coordinate
        + config.teacher_track_feature_cosine * feature_loss
        + config.teacher_support_cosine * support_loss
        + config.teacher_visibility_mse * visibility
        + config.teacher_confidence_mse * confidence
        + config.apply_noop_bce * apply
        + config.clean_coordinate_displacement * clean_coordinate
        + config.clean_memory_change * clean_memory
    )
    return {
        "loss": total,
        "failure_heatmap_focal": heatmap,
        "failure_coordinate_smooth_l1": coordinate,
        "teacher_track_feature_cosine": feature_loss,
        "teacher_support_cosine": support_loss,
        "teacher_visibility_mse": visibility,
        "teacher_confidence_mse": confidence,
        "apply_noop_bce": apply,
        "clean_coordinate_displacement": clean_coordinate,
        "clean_memory_change": clean_memory,
    }


def point_future_rows(
    coordinates_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    visible: torch.Tensor,
) -> list[dict[str, float]]:
    coordinates = coordinates_xy.float().cpu()
    gt = gt_xy.float().cpu()
    mask = visible.bool().cpu()
    if coordinates.shape != gt.shape or mask.shape != coordinates.shape[:-1]:
        raise ValueError("future metric tensor shape mismatch")
    rows = []
    thresholds = (1.0, 2.0, 4.0, 8.0, 16.0)
    for index in range(coordinates.shape[0]):
        active = mask[index]
        if not active.any():
            raise ValueError("future row has no visible frame")
        error = torch.linalg.vector_norm(
            coordinates[index, active] - gt[index, active], dim=-1
        )
        rows.append(
            {
                "mean_l2_error_px": float(error.mean().item()),
                "severe_16px_rate": float((error > 16.0).float().mean().item()),
                "threshold_utility": float(
                    torch.stack([(error <= threshold).float().mean() for threshold in thresholds])
                    .mean()
                    .item()
                ),
            }
        )
    return rows


def aggregate_point_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate empty future rows")
    return {
        "points": len(rows),
        "mean_l2_error_px": float(
            sum(float(row["mean_l2_error_px"]) for row in rows) / len(rows)
        ),
        "severe_16px_rate": float(
            sum(float(row["severe_16px_rate"]) for row in rows) / len(rows)
        ),
        "threshold_utility": float(
            sum(float(row["threshold_utility"]) for row in rows) / len(rows)
        ),
    }
