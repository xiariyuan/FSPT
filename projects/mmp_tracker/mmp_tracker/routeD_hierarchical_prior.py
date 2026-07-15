"""Causal frame-level recovery prior for Route-D hierarchical gating."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


FRAME_PRIOR_FEATURE_NAMES = (
    "frame_mean_p1_margin",
    "frame_std_p1_margin",
    "frame_mean_coarse_gain",
    "frame_std_coarse_gain",
    "frame_mean_total_gain",
    "frame_std_total_gain",
    "frame_q90_total_gain",
    "frame_mean_global_distance",
    "frame_std_global_distance",
    "frame_mean_candidate_entropy",
    "frame_mean_previous_confidence",
    "prefix_mean_p1_margin",
    "prefix_mean_coarse_gain",
    "prefix_mean_total_gain",
    "normalized_frame_id",
    "log_active_rows",
)


def _finite_1d(name: str, value: torch.Tensor, rows: int) -> torch.Tensor:
    if value.ndim != 1 or int(value.numel()) != rows:
        raise ValueError(f"{name} must be one-dimensional with {rows} rows")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def build_causal_frame_prior_features(
    *,
    sample_id: torch.Tensor,
    frame_id: torch.Tensor,
    predicted_p1_margin: torch.Tensor,
    predicted_coarse_gain: torch.Tensor,
    predicted_total_gain: torch.Tensor,
    global_distance_to_local: torch.Tensor,
    candidate_entropy: torch.Tensor,
    previous_confidence: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return one feature row per frame and a row-to-frame mapping.

    Current-frame statistics and prefix statistics use only the current and
    preceding frames from the same video. No ground truth or future frame is
    consumed.
    """
    if sample_id.ndim != 1 or frame_id.ndim != 1:
        raise ValueError("sample_id and frame_id must be one-dimensional")
    if sample_id.shape != frame_id.shape:
        raise ValueError("sample_id and frame_id must have identical shape")
    rows = int(sample_id.numel())
    sample_id = sample_id.long().cpu()
    frame_id = frame_id.long().cpu()
    p1 = _finite_1d("predicted_p1_margin", predicted_p1_margin, rows).cpu()
    coarse = _finite_1d("predicted_coarse_gain", predicted_coarse_gain, rows).cpu()
    total = _finite_1d("predicted_total_gain", predicted_total_gain, rows).cpu()
    distance = _finite_1d(
        "global_distance_to_local", global_distance_to_local, rows
    ).cpu()
    entropy = _finite_1d("candidate_entropy", candidate_entropy, rows).cpu()
    confidence = _finite_1d(
        "previous_confidence", previous_confidence, rows
    ).cpu()

    if rows == 0:
        return {
            "features": torch.empty(0, len(FRAME_PRIOR_FEATURE_NAMES)),
            "sample_id": torch.empty(0, dtype=torch.long),
            "frame_id": torch.empty(0, dtype=torch.long),
            "row_to_frame": torch.empty(0, dtype=torch.long),
        }

    frame_features = []
    frame_samples = []
    frame_numbers = []
    row_to_frame = torch.empty(rows, dtype=torch.long)
    frame_index = 0
    for sample in torch.unique(sample_id, sorted=True):
        sample_mask = sample_id == sample
        cumulative_p1 = 0.0
        cumulative_coarse = 0.0
        cumulative_total = 0.0
        cumulative_rows = 0
        for frame in torch.unique(frame_id[sample_mask], sorted=True):
            mask = sample_mask & (frame_id == frame)
            count = int(mask.sum())
            values_p1 = p1[mask]
            values_coarse = coarse[mask]
            values_total = total[mask]
            values_distance = distance[mask]
            values_entropy = entropy[mask]
            values_confidence = confidence[mask]
            cumulative_p1 += float(values_p1.sum())
            cumulative_coarse += float(values_coarse.sum())
            cumulative_total += float(values_total.sum())
            cumulative_rows += count
            denom = float(max(cumulative_rows, 1))
            frame_value = float(frame)
            feature = torch.tensor(
                [
                    float(values_p1.mean()),
                    float(values_p1.std(unbiased=False)),
                    float(values_coarse.mean()),
                    float(values_coarse.std(unbiased=False)),
                    float(values_total.mean()),
                    float(values_total.std(unbiased=False)),
                    float(torch.quantile(values_total, 0.9)),
                    float(values_distance.mean()),
                    float(values_distance.std(unbiased=False)),
                    float(values_entropy.mean()),
                    float(values_confidence.mean()),
                    cumulative_p1 / denom,
                    cumulative_coarse / denom,
                    cumulative_total / denom,
                    frame_value / (frame_value + 10.0),
                    float(torch.log1p(torch.tensor(float(count)))),
                ],
                dtype=torch.float32,
            )
            frame_features.append(feature)
            frame_samples.append(int(sample))
            frame_numbers.append(int(frame))
            row_to_frame[mask] = frame_index
            frame_index += 1
    result = {
        "features": torch.stack(frame_features),
        "sample_id": torch.tensor(frame_samples, dtype=torch.long),
        "frame_id": torch.tensor(frame_numbers, dtype=torch.long),
        "row_to_frame": row_to_frame,
    }
    if not torch.isfinite(result["features"]).all():
        raise RuntimeError("Built non-finite frame prior features")
    return result


def aggregate_frame_mean_target(
    row_target: torch.Tensor,
    row_to_frame: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    row_target = row_target.float().flatten().cpu()
    row_to_frame = row_to_frame.long().flatten().cpu()
    if row_target.shape != row_to_frame.shape:
        raise ValueError("row_target and row_to_frame must have identical shape")
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if row_target.numel() == 0:
        return torch.empty(frame_count, dtype=torch.float32)
    if int(row_to_frame.min()) < 0 or int(row_to_frame.max()) >= frame_count:
        raise ValueError("row_to_frame index outside frame range")
    sums = torch.zeros(frame_count, dtype=torch.float32)
    counts = torch.zeros(frame_count, dtype=torch.float32)
    sums.scatter_add_(0, row_to_frame, row_target)
    counts.scatter_add_(0, row_to_frame, torch.ones_like(row_target))
    if (counts == 0).any():
        raise ValueError("Every frame must receive at least one row")
    return sums / counts


class RouteDFramePriorRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected frame prior dim {self.input_dim}, got {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class FramePriorConfig:
    hidden_dim: int = 48
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 15
    seed: int = 17
    magnitude_weight: float = 4.0
    sign_loss_weight: float = 0.5


def _normalize(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) / std.clamp_min(1.0e-6)


def frame_prior_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    magnitude_weight: float = 4.0,
    sign_loss_weight: float = 0.5,
) -> torch.Tensor:
    prediction = prediction.float().flatten()
    target = target.float().flatten()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    importance = 1.0 + float(magnitude_weight) * target.abs()
    regression = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=0.05
    )
    sign = F.binary_cross_entropy_with_logits(
        prediction / 0.05,
        (target > 0.0).float(),
        reduction="none",
    )
    return (importance * (regression + float(sign_loss_weight) * sign)).mean()


def frame_prior_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.detach().float().flatten().cpu()
    target = target.detach().float().flatten().cpu()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    predicted_positive = prediction > 0.0
    target_positive = target > 0.0
    true_positive = int((predicted_positive & target_positive).sum())
    return {
        "frames": int(target.numel()),
        "target_mean": float(target.mean()),
        "target_positive_rate": float(target_positive.float().mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_positive_rate": float(predicted_positive.float().mean()),
        "mae": float((prediction - target).abs().mean()),
        "rmse": float((prediction - target).square().mean().sqrt()),
        "pearson": float(
            (centered_prediction * centered_target).sum() / denominator
        ),
        "positive_precision": true_positive / max(int(predicted_positive.sum()), 1),
        "positive_recall": true_positive / max(int(target_positive.sum()), 1),
    }


def train_frame_prior(
    fit_features: torch.Tensor,
    fit_target: torch.Tensor,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    *,
    config: FramePriorConfig,
    device: torch.device,
):
    torch.manual_seed(int(config.seed))
    mean = fit_features.float().mean(dim=0)
    std = fit_features.float().std(dim=0, unbiased=False).clamp_min(1.0e-6)
    fit_x = _normalize(fit_features.float(), mean, std)
    validation_x = _normalize(validation_features.float(), mean, std)
    loader = DataLoader(
        TensorDataset(fit_x, fit_target.float()),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )
    model = RouteDFramePriorRegressor(
        fit_x.shape[-1], hidden_dim=int(config.hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(int(config.epochs)):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            prediction = model(features)
            loss = frame_prior_loss(
                prediction,
                target,
                magnitude_weight=config.magnitude_weight,
                sign_loss_weight=config.sign_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(target.numel())
            total_loss += float(loss) * rows
            total_rows += rows
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x.to(device)).cpu()
        validation_mae = float(
            (validation_prediction - validation_target.float()).abs().mean()
        )
        validation_sign = F.binary_cross_entropy_with_logits(
            validation_prediction / 0.05,
            (validation_target.float() > 0.0).float(),
        )
        selection_value = validation_mae + 0.25 * float(validation_sign)
        history.append(
            {
                "epoch": epoch + 1,
                "fit_loss": total_loss / max(total_rows, 1),
                "validation_selection_value": selection_value,
                "validation": frame_prior_metrics(
                    validation_prediction, validation_target
                ),
            }
        )
        if selection_value < best_value - 1.0e-8:
            best_value = selection_value
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    model.load_state_dict(best_state, strict=True)
    return {
        "model": model,
        "feature_mean": mean,
        "feature_std": std,
        "best_epoch": best_epoch,
        "history": history,
    }


def infer_frame_prior(
    model: RouteDFramePriorRegressor,
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), max(1, int(batch_size))):
            batch = _normalize(
                features[start : start + batch_size].float(), mean, std
            ).to(device)
            outputs.append(model(batch).cpu())
    return torch.cat(outputs) if outputs else torch.empty(0)
