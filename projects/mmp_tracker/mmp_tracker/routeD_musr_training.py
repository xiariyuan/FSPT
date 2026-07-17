"""Training and frozen-partition evaluation utilities for Route-D MUSR."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from datasets.metrics import compute_tapvid_metrics

from .routeD_kubric_cache import VIDEO_SIDECAR_SCHEMA_VERSION, file_sha256
from .routeD_recovery_network import (
    MultiHypothesisStateRecoveryNetwork,
    RecoveryLossConfig,
    RecoveryNetworkConfig,
    recovery_training_loss,
)


@dataclass(frozen=True)
class FeatureNormalization:
    candidate_mean: torch.Tensor
    candidate_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def to_serializable(self) -> dict[str, list[float]]:
        return {
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
        }

    @classmethod
    def from_serializable(cls, payload: Mapping[str, Sequence[float]]) -> "FeatureNormalization":
        return cls(
            candidate_mean=torch.tensor(payload["candidate_mean"], dtype=torch.float32),
            candidate_std=torch.tensor(payload["candidate_std"], dtype=torch.float32),
            state_mean=torch.tensor(payload["state_mean"], dtype=torch.float32),
            state_std=torch.tensor(payload["state_std"], dtype=torch.float32),
        )


@dataclass(frozen=True)
class MUSRTrainingRows:
    candidate_features: torch.Tensor
    candidate_coords_px: torch.Tensor
    candidate_valid_mask: torch.Tensor
    state_features: torch.Tensor
    source_ids: torch.Tensor
    gt_coords_px: torch.Tensor
    video_row_ids: torch.Tensor
    source_rows: int
    sidecar_paths: tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.candidate_features.shape[0])


def load_cache_index(path: str | Path, *, expected_partition: str | None = None) -> dict[str, Any]:
    index_path = Path(path).resolve()
    index = json.loads(index_path.read_text())
    if not index.get("complete"):
        raise ValueError(f"cache partition is incomplete: {index_path}")
    if expected_partition is not None and index.get("partition") != expected_partition:
        raise ValueError(
            f"expected partition {expected_partition}, got {index.get('partition')}"
        )
    expected = index.get("expected_source_indices")
    completed = index.get("completed_source_indices")
    if expected != completed or int(index.get("expected_count", -1)) != int(
        index.get("completed_count", -2)
    ):
        raise ValueError("cache index membership/completion mismatch")
    index["_index_path"] = str(index_path)
    index["_index_sha256"] = file_sha256(index_path)
    return index


def _load_verified_artifact(row: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = Path(str(row["sidecar"]))
    if file_sha256(sidecar) != str(row["sidecar_sha256"]):
        raise ValueError(f"sidecar hash mismatch: {sidecar}")
    artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != VIDEO_SIDECAR_SCHEMA_VERSION:
        raise ValueError(f"unexpected sidecar schema: {sidecar}")
    tensors = artifact["tensors"]
    if not torch.equal(
        tensors["candidate_coords_xy_px"][..., 0, :],
        tensors["native_coords_xy_px"],
    ):
        raise ValueError(f"native candidate parity drift: {sidecar}")
    return artifact


def visible_post_query_mask(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    points, frames = tensors["gt_occluded"].shape
    frame_index = torch.arange(frames).view(1, frames)
    query_frame = tensors["query_points_tyx"][:, 0].round().long().view(points, 1)
    return (~tensors["gt_occluded"]) & (frame_index > query_frame)


def load_training_rows(
    cache_index_path: str | Path,
    *,
    expected_partition: str,
    raster: int = 256,
) -> tuple[MUSRTrainingRows, dict[str, Any]]:
    index = load_cache_index(cache_index_path, expected_partition=expected_partition)
    feature_rows = []
    coord_rows = []
    valid_rows = []
    state_rows = []
    source_rows = []
    gt_rows = []
    video_rows = []
    sidecars = []
    source_row_count = 0

    for video_row_id, row in enumerate(index["videos"]):
        artifact = _load_verified_artifact(row)
        tensors = artifact["tensors"]
        mask = visible_post_query_mask(tensors)
        source_row_count += int(mask.numel())
        gt_xy_px = tensors["gt_tracks_yx"][..., [1, 0]] * float(raster - 1)
        feature_rows.append(tensors["candidate_features"][mask].float())
        coord_rows.append(tensors["candidate_coords_xy_px"][mask].float())
        valid_rows.append(tensors["candidate_valid_mask"][mask].bool())
        state_rows.append(tensors["state_features"][mask].float())
        source_rows.append(tensors["source_ids"][mask].long())
        gt_rows.append(gt_xy_px[mask].float())
        video_rows.append(
            torch.full((int(mask.sum().item()),), video_row_id, dtype=torch.long)
        )
        sidecars.append(str(row["sidecar"]))

    bundle = MUSRTrainingRows(
        candidate_features=torch.cat(feature_rows, dim=0),
        candidate_coords_px=torch.cat(coord_rows, dim=0),
        candidate_valid_mask=torch.cat(valid_rows, dim=0),
        state_features=torch.cat(state_rows, dim=0),
        source_ids=torch.cat(source_rows, dim=0),
        gt_coords_px=torch.cat(gt_rows, dim=0),
        video_row_ids=torch.cat(video_rows, dim=0),
        source_rows=source_row_count,
        sidecar_paths=tuple(sidecars),
    )
    return bundle, index


def compute_feature_normalization(rows: MUSRTrainingRows) -> FeatureNormalization:
    valid_features = rows.candidate_features[rows.candidate_valid_mask]
    if valid_features.numel() == 0:
        raise ValueError("no valid candidate features")
    candidate_mean = valid_features.double().mean(dim=0).float()
    candidate_std = valid_features.double().std(dim=0, unbiased=False).float().clamp_min(1.0e-5)
    state_mean = rows.state_features.double().mean(dim=0).float()
    state_std = rows.state_features.double().std(dim=0, unbiased=False).float().clamp_min(1.0e-5)
    return FeatureNormalization(candidate_mean, candidate_std, state_mean, state_std)


def normalize_inputs(
    candidate_features: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    state_features: torch.Tensor,
    normalization: FeatureNormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_mean = normalization.candidate_mean.to(
        device=candidate_features.device, dtype=candidate_features.dtype
    )
    candidate_std = normalization.candidate_std.to(
        device=candidate_features.device, dtype=candidate_features.dtype
    )
    state_mean = normalization.state_mean.to(
        device=state_features.device, dtype=state_features.dtype
    )
    state_std = normalization.state_std.to(
        device=state_features.device, dtype=state_features.dtype
    )
    normalized_candidates = (candidate_features - candidate_mean) / candidate_std
    normalized_candidates = torch.where(
        candidate_valid_mask.unsqueeze(-1),
        normalized_candidates,
        torch.zeros_like(normalized_candidates),
    )
    normalized_state = (state_features - state_mean) / state_std
    return normalized_candidates, normalized_state


def paired_video_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 1701,
    samples: int = 5000,
    confidence: float = 0.95,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("paired bootstrap requires a non-empty 1D vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _tracks_from_xy(coords_xy_px: torch.Tensor, raster: int) -> torch.Tensor:
    scale = float(raster - 1)
    return torch.stack(
        [coords_xy_px[..., 1] / scale, coords_xy_px[..., 0] / scale], dim=-1
    )


def _visible_error_stats(
    coords_xy_px: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    *,
    raster: int,
) -> dict[str, float]:
    gt_xy_px = tensors["gt_tracks_yx"][..., [1, 0]] * float(raster - 1)
    mask = visible_post_query_mask(tensors)
    error = torch.linalg.vector_norm(coords_xy_px - gt_xy_px, dim=-1)[mask]
    return {
        "mean_error_px": float(error.mean().item()),
        "median_error_px": float(error.median().item()),
        "p95_error_px": float(torch.quantile(error, 0.95).item()),
        "severe_16px_rate": float((error >= 16.0).float().mean().item()),
        "rows": int(error.numel()),
    }


def predict_sidecar(
    model: MultiHypothesisStateRecoveryNetwork,
    tensors: Mapping[str, torch.Tensor],
    normalization: FeatureNormalization,
    *,
    device: str,
    batch_size: int,
    abstention_threshold: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = tensors["candidate_features"].reshape(
        -1, tensors["candidate_features"].shape[-2], tensors["candidate_features"].shape[-1]
    )
    coords = tensors["candidate_coords_xy_px"].reshape(-1, tensors["candidate_coords_xy_px"].shape[-2], 2)
    valid = tensors["candidate_valid_mask"].reshape(-1, tensors["candidate_valid_mask"].shape[-1])
    state = tensors["state_features"].reshape(-1, tensors["state_features"].shape[-1])
    source_ids = tensors["source_ids"].reshape(-1, tensors["source_ids"].shape[-1])
    updated_chunks = []
    selected_global = 0
    abstention_sum = 0.0
    write_sum = 0.0
    rows = int(features.shape[0])

    model.eval()
    with torch.no_grad():
        for start in range(0, rows, int(batch_size)):
            end = min(rows, start + int(batch_size))
            f = features[start:end].to(device)
            c = coords[start:end].to(device)
            v = valid[start:end].to(device)
            s = state[start:end].to(device)
            ids = source_ids[start:end].to(device)
            f, s = normalize_inputs(f, v, s, normalization)
            output = model(
                f.unsqueeze(1),
                c.unsqueeze(1),
                v.unsqueeze(1),
                s.unsqueeze(1),
                ids.unsqueeze(1),
                use_hard_selection=True,
            )
            updated = output["updated_coord_px"].squeeze(1)
            abstention = output["abstention_probability"].squeeze(1)
            if abstention_threshold is not None:
                native = output["native_coord_px"].squeeze(1)
                updated = torch.where(
                    (abstention >= float(abstention_threshold)).unsqueeze(-1),
                    native,
                    updated,
                )
            updated_chunks.append(updated.cpu())
            selected_global += int((output["selected_candidate_index"].squeeze(1) > 0).sum().item())
            abstention_sum += float(abstention.sum().item())
            write_sum += float(output["state_write_strength"][..., 0].sum().item())
    shape = tensors["native_coords_xy_px"].shape
    return torch.cat(updated_chunks, dim=0).reshape(shape), {
        "selected_non_native_rate": selected_global / max(rows, 1),
        "mean_abstention_probability": abstention_sum / max(rows, 1),
        "mean_coordinate_write_strength": write_sum / max(rows, 1),
    }


def evaluate_model_on_cache_index(
    model: MultiHypothesisStateRecoveryNetwork,
    cache_index_path: str | Path,
    normalization: FeatureNormalization,
    *,
    expected_partition: str,
    device: str,
    batch_size: int = 2048,
    raster: int = 256,
    abstention_threshold: float | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 1701,
) -> dict[str, Any]:
    index = load_cache_index(cache_index_path, expected_partition=expected_partition)
    native_tracks = []
    learned_tracks = []
    gt_tracks = []
    pred_visibility = []
    gt_visibility = []
    queries = []
    per_video = []

    for row in index["videos"]:
        artifact = _load_verified_artifact(row)
        tensors = artifact["tensors"]
        learned_xy, behavior = predict_sidecar(
            model,
            tensors,
            normalization,
            device=device,
            batch_size=batch_size,
            abstention_threshold=abstention_threshold,
        )
        native_xy = tensors["native_coords_xy_px"]
        native_yx = _tracks_from_xy(native_xy, raster)
        learned_yx = _tracks_from_xy(learned_xy, raster)
        visibility = tensors["native_visibility"]
        gt_vis = ~tensors["gt_occluded"]
        native_metrics = compute_tapvid_metrics(
            native_yx,
            tensors["gt_tracks_yx"],
            visibility,
            gt_vis,
            tensors["query_points_tyx"],
            resolution=raster,
            query_mode="first",
        )
        learned_metrics = compute_tapvid_metrics(
            learned_yx,
            tensors["gt_tracks_yx"],
            visibility,
            gt_vis,
            tensors["query_points_tyx"],
            resolution=raster,
            query_mode="first",
        )
        native_error = _visible_error_stats(native_xy, tensors, raster=raster)
        learned_error = _visible_error_stats(learned_xy, tensors, raster=raster)
        per_video.append(
            {
                "source_index": int(row["source_index"]),
                "video_name": row["sample_identity"]["video_name"],
                "native_AJ": float(native_metrics["AJ"]),
                "learned_AJ": float(learned_metrics["AJ"]),
                "AJ_gain_points": 100.0
                * (float(learned_metrics["AJ"]) - float(native_metrics["AJ"])),
                "native_delta_average": float(native_metrics["<avg"]),
                "learned_delta_average": float(learned_metrics["<avg"]),
                "delta_gain_points": 100.0
                * (float(learned_metrics["<avg"]) - float(native_metrics["<avg"])),
                "native_error": native_error,
                "learned_error": learned_error,
                "behavior": behavior,
            }
        )
        native_tracks.append(native_yx)
        learned_tracks.append(learned_yx)
        gt_tracks.append(tensors["gt_tracks_yx"])
        pred_visibility.append(visibility)
        gt_visibility.append(gt_vis)
        queries.append(tensors["query_points_tyx"])

    native_metrics = compute_tapvid_metrics(
        torch.cat(native_tracks, dim=0),
        torch.cat(gt_tracks, dim=0),
        torch.cat(pred_visibility, dim=0),
        torch.cat(gt_visibility, dim=0),
        torch.cat(queries, dim=0),
        resolution=raster,
        query_mode="first",
    )
    learned_metrics = compute_tapvid_metrics(
        torch.cat(learned_tracks, dim=0),
        torch.cat(gt_tracks, dim=0),
        torch.cat(pred_visibility, dim=0),
        torch.cat(gt_visibility, dim=0),
        torch.cat(queries, dim=0),
        resolution=raster,
        query_mode="first",
    )
    aj_values = [row["AJ_gain_points"] for row in per_video]
    delta_values = [row["delta_gain_points"] for row in per_video]
    native_severe = np.average(
        [row["native_error"]["severe_16px_rate"] for row in per_video],
        weights=[row["native_error"]["rows"] for row in per_video],
    )
    learned_severe = np.average(
        [row["learned_error"]["severe_16px_rate"] for row in per_video],
        weights=[row["learned_error"]["rows"] for row in per_video],
    )
    return {
        "partition": expected_partition,
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "learned_metrics": learned_metrics,
        "gain_points": {
            "AJ": 100.0 * (float(learned_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0
            * (float(learned_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0 * (float(learned_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "paired_video_AJ_gain_CI": paired_video_bootstrap_ci(
            aj_values, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "paired_video_delta_gain_CI": paired_video_bootstrap_ci(
            delta_values, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        "severe_16px_rate": {
            "native": float(native_severe),
            "learned": float(learned_severe),
            "delta": float(learned_severe - native_severe),
        },
        "per_video": per_video,
        "cache_index_sha256": index["_index_sha256"],
        "abstention_threshold": abstention_threshold,
    }


def train_one_epoch(
    model: MultiHypothesisStateRecoveryNetwork,
    rows: MUSRTrainingRows,
    normalization: FeatureNormalization,
    optimizer: torch.optim.Optimizer,
    network_config: RecoveryNetworkConfig,
    loss_config: RecoveryLossConfig,
    *,
    device: str,
    batch_size: int,
    generator: torch.Generator,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(rows.rows, generator=generator)
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, rows.rows, int(batch_size)):
        index = permutation[start : start + int(batch_size)]
        features = rows.candidate_features[index].to(device)
        coords = rows.candidate_coords_px[index].to(device)
        valid = rows.candidate_valid_mask[index].to(device)
        state = rows.state_features[index].to(device)
        source_ids = rows.source_ids[index].to(device)
        gt = rows.gt_coords_px[index].to(device)
        features, state = normalize_inputs(features, valid, state, normalization)
        output = model(
            features.unsqueeze(1),
            coords.unsqueeze(1),
            valid.unsqueeze(1),
            state.unsqueeze(1),
            source_ids.unsqueeze(1),
            use_hard_selection=False,
            use_straight_through_selection=True,
        )
        losses = recovery_training_loss(
            output,
            coords.unsqueeze(1),
            valid.unsqueeze(1),
            gt.unsqueeze(1),
            network_config,
            loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("non-finite MUSR training loss")
        nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        batch_rows = int(index.numel())
        count += batch_rows
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_rows
    return {key: value / max(count, 1) for key, value in totals.items()}


def clone_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SelectorLossConfig:
    threshold_bce_weight: float = 1.0
    utility_ranking_weight: float = 1.0
    catastrophe_weight: float = 0.5
    native_gate_weight: float = 1.0
    harmful_global_weight: float = 4.0
    gate_temperature: float = 0.25
    catastrophe_threshold_px: float = 16.0


def selector_pretraining_loss(
    outputs: Mapping[str, torch.Tensor],
    candidate_coords_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    gt_coords_px: torch.Tensor,
    network_config: RecoveryNetworkConfig,
    config: SelectorLossConfig = SelectorLossConfig(),
) -> dict[str, torch.Tensor]:
    """Utility-aligned selector objective with native fallback on utility ties."""
    if candidate_coords_px.ndim != 4:
        raise ValueError("candidate_coords_px must have shape (B,P,K,2)")
    error = torch.linalg.vector_norm(
        candidate_coords_px - gt_coords_px.unsqueeze(-2), dim=-1
    )
    thresholds = torch.tensor(
        network_config.thresholds_px,
        device=error.device,
        dtype=error.dtype,
    )
    targets = (error.unsqueeze(-1) <= thresholds).to(error.dtype)
    valid_weight = candidate_valid_mask.unsqueeze(-1).to(error.dtype)
    threshold_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["threshold_logits"], targets, reduction="none"
    )
    threshold_loss = (threshold_bce * valid_weight).sum() / (
        valid_weight.sum().clamp_min(1.0) * targets.shape[-1]
    )

    utility_weights = torch.tensor(
        network_config.threshold_utility_weights,
        device=error.device,
        dtype=error.dtype,
    )
    utility_weights = utility_weights / utility_weights.sum()
    true_utility = (targets * utility_weights).sum(dim=-1)
    masked_utility = true_utility.masked_fill(~candidate_valid_mask, -1.0)
    # torch.argmax deterministically gives candidate 0 on utility ties.
    best_candidate = masked_utility.argmax(dim=-1)
    ranking_loss = torch.nn.functional.cross_entropy(
        outputs["candidate_score"].reshape(-1, candidate_coords_px.shape[-2]),
        best_candidate.reshape(-1),
    )

    catastrophe_target = (error >= float(config.catastrophe_threshold_px)).to(
        outputs["catastrophe_logit"].dtype
    )
    catastrophe_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["catastrophe_logit"], catastrophe_target, reduction="none"
    )
    candidate_weight = candidate_valid_mask.to(catastrophe_bce.dtype)
    catastrophe_loss = (catastrophe_bce * candidate_weight).sum() / candidate_weight.sum().clamp_min(1.0)

    native_utility = true_utility[..., 0]
    global_valid = candidate_valid_mask[..., 1:]
    global_true_utility = true_utility[..., 1:].masked_fill(~global_valid, -1.0)
    best_global_true = global_true_utility.max(dim=-1).values
    beneficial_global = best_global_true > native_utility
    score = outputs["candidate_score"]
    best_global_score = score[..., 1:].max(dim=-1).values
    gate_logit = (best_global_score - score[..., 0]) / float(config.gate_temperature)
    gate_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        gate_logit, beneficial_global.to(gate_logit.dtype), reduction="none"
    )
    harmful = best_global_true < native_utility
    gate_weight = torch.where(
        harmful,
        torch.full_like(gate_bce, float(config.harmful_global_weight)),
        torch.ones_like(gate_bce),
    )
    gate_loss = (gate_bce * gate_weight).sum() / gate_weight.sum().clamp_min(1.0)

    total = (
        float(config.threshold_bce_weight) * threshold_loss
        + float(config.utility_ranking_weight) * ranking_loss
        + float(config.catastrophe_weight) * catastrophe_loss
        + float(config.native_gate_weight) * gate_loss
    )
    with torch.no_grad():
        prediction = outputs["candidate_score"].argmax(dim=-1)
        accuracy = (prediction == best_candidate).float().mean()
        selected_utility = true_utility.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
        utility_regret = (masked_utility.max(dim=-1).values - selected_utility).mean()
        harmful_selected = ((prediction > 0) & (selected_utility < native_utility)).float().mean()
    return {
        "loss": total,
        "threshold_bce": threshold_loss,
        "utility_ranking": ranking_loss,
        "catastrophe": catastrophe_loss,
        "native_gate": gate_loss,
        "selector_accuracy": accuracy,
        "mean_utility_regret": utility_regret,
        "harmful_global_selection_rate": harmful_selected,
    }


def train_selector_one_epoch(
    model: MultiHypothesisStateRecoveryNetwork,
    rows: MUSRTrainingRows,
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
    permutation = torch.randperm(rows.rows, generator=generator)
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, rows.rows, int(batch_size)):
        index = permutation[start : start + int(batch_size)]
        features = rows.candidate_features[index].to(device)
        coords = rows.candidate_coords_px[index].to(device)
        valid = rows.candidate_valid_mask[index].to(device)
        state = rows.state_features[index].to(device)
        source_ids = rows.source_ids[index].to(device)
        gt = rows.gt_coords_px[index].to(device)
        features, state = normalize_inputs(features, valid, state, normalization)
        output = model(
            features.unsqueeze(1),
            coords.unsqueeze(1),
            valid.unsqueeze(1),
            state.unsqueeze(1),
            source_ids.unsqueeze(1),
        )
        losses = selector_pretraining_loss(
            output,
            coords.unsqueeze(1),
            valid.unsqueeze(1),
            gt.unsqueeze(1),
            network_config,
            selector_config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("non-finite selector loss")
        nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        batch_rows = int(index.numel())
        count += batch_rows
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_rows
    return {key: value / max(count, 1) for key, value in totals.items()}


def predict_raw_selector_sidecar(
    model: MultiHypothesisStateRecoveryNetwork,
    tensors: Mapping[str, torch.Tensor],
    normalization: FeatureNormalization,
    *,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = tensors["candidate_features"].reshape(
        -1, tensors["candidate_features"].shape[-2], tensors["candidate_features"].shape[-1]
    )
    coords = tensors["candidate_coords_xy_px"].reshape(-1, tensors["candidate_coords_xy_px"].shape[-2], 2)
    valid = tensors["candidate_valid_mask"].reshape(-1, tensors["candidate_valid_mask"].shape[-1])
    state = tensors["state_features"].reshape(-1, tensors["state_features"].shape[-1])
    source_ids = tensors["source_ids"].reshape(-1, tensors["source_ids"].shape[-1])
    selected_coords = []
    selected_indices = []
    oracle_indices = []
    utility_regrets = []
    harmful = 0
    eval_rows = 0
    gt_xy = tensors["gt_tracks_yx"][..., [1, 0]] * 255.0
    gt_flat = gt_xy.reshape(-1, 2)
    eval_mask = visible_post_query_mask(tensors).reshape(-1)

    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], int(batch_size)):
            end = min(features.shape[0], start + int(batch_size))
            f = features[start:end].to(device)
            c = coords[start:end].to(device)
            v = valid[start:end].to(device)
            s = state[start:end].to(device)
            ids = source_ids[start:end].to(device)
            f, s = normalize_inputs(f, v, s, normalization)
            output = model(
                f.unsqueeze(1),
                c.unsqueeze(1),
                v.unsqueeze(1),
                s.unsqueeze(1),
                ids.unsqueeze(1),
                use_hard_selection=True,
            )
            selection = output["selected_candidate_index"].squeeze(1)
            selected = c.gather(1, selection[:, None, None].expand(-1, 1, 2)).squeeze(1)
            selected_coords.append(selected.cpu())
            selected_indices.append(selection.cpu())

            local_eval = eval_mask[start:end].to(device)
            if local_eval.any():
                gt = gt_flat[start:end].to(device)
                error = torch.linalg.vector_norm(c - gt.unsqueeze(1), dim=-1).masked_fill(~v, float("inf"))
                thresholds = torch.tensor(
                    model.config.thresholds_px, device=device, dtype=error.dtype
                )
                target = (error.unsqueeze(-1) <= thresholds).to(error.dtype)
                weights = torch.tensor(
                    model.config.threshold_utility_weights,
                    device=device,
                    dtype=error.dtype,
                )
                weights = weights / weights.sum()
                utility = (target * weights).sum(dim=-1).masked_fill(~v, -1.0)
                oracle = utility.argmax(dim=-1)
                selected_utility = utility.gather(1, selection[:, None]).squeeze(1)
                native_utility = utility[:, 0]
                regret = utility.max(dim=-1).values - selected_utility
                oracle_indices.append(oracle[local_eval].cpu())
                utility_regrets.append(regret[local_eval].cpu())
                harmful += int(
                    ((selection[local_eval] > 0) & (selected_utility[local_eval] < native_utility[local_eval])).sum().item()
                )
                eval_rows += int(local_eval.sum().item())

    selected_coord = torch.cat(selected_coords, dim=0).reshape_as(tensors["native_coords_xy_px"])
    selection = torch.cat(selected_indices, dim=0).reshape(tensors["native_visibility"].shape)
    selected_eval = selection.reshape(-1)[eval_mask]
    oracle_eval = torch.cat(oracle_indices) if oracle_indices else torch.empty(0, dtype=torch.long)
    regret_eval = torch.cat(utility_regrets) if utility_regrets else torch.empty(0)
    return selected_coord, {
        "selected_non_native_rate": float((selected_eval > 0).float().mean().item()),
        "oracle_utility_match_rate": float((selected_eval == oracle_eval).float().mean().item()),
        "mean_utility_regret": float(regret_eval.mean().item()),
        "harmful_global_selection_rate": harmful / max(eval_rows, 1),
    }


def evaluate_raw_selector_on_cache_index(
    model: MultiHypothesisStateRecoveryNetwork,
    cache_index_path: str | Path,
    normalization: FeatureNormalization,
    *,
    expected_partition: str,
    device: str,
    batch_size: int = 2048,
    raster: int = 256,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 2701,
) -> dict[str, Any]:
    index = load_cache_index(cache_index_path, expected_partition=expected_partition)
    native_tracks = []
    selected_tracks = []
    gt_tracks = []
    pred_visibility = []
    gt_visibility = []
    queries = []
    per_video = []
    for row in index["videos"]:
        artifact = _load_verified_artifact(row)
        tensors = artifact["tensors"]
        selected_xy, behavior = predict_raw_selector_sidecar(
            model, tensors, normalization, device=device, batch_size=batch_size
        )
        native_xy = tensors["native_coords_xy_px"]
        native_yx = _tracks_from_xy(native_xy, raster)
        selected_yx = _tracks_from_xy(selected_xy, raster)
        visibility = tensors["native_visibility"]
        gt_vis = ~tensors["gt_occluded"]
        native_metrics = compute_tapvid_metrics(
            native_yx, tensors["gt_tracks_yx"], visibility, gt_vis,
            tensors["query_points_tyx"], resolution=raster, query_mode="first"
        )
        selected_metrics = compute_tapvid_metrics(
            selected_yx, tensors["gt_tracks_yx"], visibility, gt_vis,
            tensors["query_points_tyx"], resolution=raster, query_mode="first"
        )
        native_error = _visible_error_stats(native_xy, tensors, raster=raster)
        selected_error = _visible_error_stats(selected_xy, tensors, raster=raster)
        per_video.append({
            "source_index": int(row["source_index"]),
            "video_name": row["sample_identity"]["video_name"],
            "AJ_gain_points": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_gain_points": 100.0 * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
            "native_error": native_error,
            "selected_error": selected_error,
            "behavior": behavior,
        })
        native_tracks.append(native_yx); selected_tracks.append(selected_yx)
        gt_tracks.append(tensors["gt_tracks_yx"]); pred_visibility.append(visibility)
        gt_visibility.append(gt_vis); queries.append(tensors["query_points_tyx"])
    native_metrics = compute_tapvid_metrics(
        torch.cat(native_tracks), torch.cat(gt_tracks), torch.cat(pred_visibility),
        torch.cat(gt_visibility), torch.cat(queries), resolution=raster, query_mode="first"
    )
    selected_metrics = compute_tapvid_metrics(
        torch.cat(selected_tracks), torch.cat(gt_tracks), torch.cat(pred_visibility),
        torch.cat(gt_visibility), torch.cat(queries), resolution=raster, query_mode="first"
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
    behavior_keys = list(per_video[0]["behavior"])
    behavior = {
        key: float(np.mean([row["behavior"][key] for row in per_video]))
        for key in behavior_keys
    }
    return {
        "partition": expected_partition,
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "selected_metrics": selected_metrics,
        "gain_points": {
            "AJ": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0 * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
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
