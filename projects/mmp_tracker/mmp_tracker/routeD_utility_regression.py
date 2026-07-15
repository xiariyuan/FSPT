"""Expected-utility regression primitives for Route-D action selection."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RouteDUtilityRegressor(nn.Module):
    """Predict expected threshold-utility gain and 1-pixel gain."""

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected utility input dim {self.input_dim}, got {features.shape[-1]}"
            )
        return self.network(features)


@dataclass(frozen=True)
class UtilityRegressionConfig:
    hidden_dim: int = 96
    epochs: int = 60
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 17
    nonzero_weight: float = 2.0
    magnitude_weight: float = 4.0
    p1_loss_weight: float = 1.0
    sign_loss_weight: float = 0.5


def normalize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) / std.clamp_min(1.0e-6)


def utility_regression_loss(
    prediction: torch.Tensor,
    utility_target: torch.Tensor,
    p1_target: torch.Tensor,
    *,
    nonzero_weight: float = 2.0,
    magnitude_weight: float = 4.0,
    p1_loss_weight: float = 1.0,
    sign_loss_weight: float = 0.5,
) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape[-1] != 2:
        raise ValueError("prediction must have shape (B, 2)")
    if prediction.shape[0] != utility_target.numel():
        raise ValueError("utility_target row count mismatch")
    if p1_target.shape != utility_target.shape:
        raise ValueError("p1_target must match utility_target")
    utility_prediction = prediction[:, 0]
    p1_prediction = prediction[:, 1]
    utility_target = utility_target.float()
    p1_target = p1_target.float()
    importance = 1.0 + float(nonzero_weight) * (utility_target != 0.0).float()
    importance = importance + float(magnitude_weight) * utility_target.abs()
    utility_loss = F.smooth_l1_loss(
        utility_prediction, utility_target, reduction="none", beta=0.1
    )
    p1_loss = F.smooth_l1_loss(
        p1_prediction, p1_target, reduction="none", beta=0.25
    )
    sign_target = (utility_target > 0.0).float()
    sign_loss = F.binary_cross_entropy_with_logits(
        utility_prediction / 0.1,
        sign_target,
        reduction="none",
    )
    return (
        importance
        * (
            utility_loss
            + float(p1_loss_weight) * p1_loss
            + float(sign_loss_weight) * sign_loss
        )
    ).mean()


def regression_metrics(
    prediction: torch.Tensor,
    utility_target: torch.Tensor,
    p1_target: torch.Tensor,
) -> Dict[str, float]:
    prediction = prediction.detach().float().cpu()
    utility_target = utility_target.detach().float().cpu()
    p1_target = p1_target.detach().float().cpu()
    utility_prediction = prediction[:, 0]
    p1_prediction = prediction[:, 1]
    centered_prediction = utility_prediction - utility_prediction.mean()
    centered_target = utility_target - utility_target.mean()
    denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    utility_pearson = float(
        (centered_prediction * centered_target).sum() / denominator
    )
    sign_prediction = utility_prediction > 0.0
    sign_target = utility_target > 0.0
    true_positive = int((sign_prediction & sign_target).sum())
    predicted_positive = int(sign_prediction.sum())
    actual_positive = int(sign_target.sum())
    return {
        "rows": int(utility_target.numel()),
        "utility_mae": float((utility_prediction - utility_target).abs().mean()),
        "utility_rmse": float(
            (utility_prediction - utility_target).square().mean().sqrt()
        ),
        "utility_pearson": utility_pearson,
        "p1_mae": float((p1_prediction - p1_target).abs().mean()),
        "predicted_positive_rate": float(sign_prediction.float().mean()),
        "target_positive_rate": float(sign_target.float().mean()),
        "positive_precision": true_positive / max(predicted_positive, 1),
        "positive_recall": true_positive / max(actual_positive, 1),
    }


def train_utility_regressor(
    fit_features: torch.Tensor,
    fit_utility: torch.Tensor,
    fit_p1: torch.Tensor,
    validation_features: torch.Tensor,
    validation_utility: torch.Tensor,
    validation_p1: torch.Tensor,
    *,
    config: UtilityRegressionConfig,
    device: torch.device,
):
    torch.manual_seed(int(config.seed))
    feature_mean = fit_features.float().mean(dim=0)
    feature_std = fit_features.float().std(dim=0, unbiased=False).clamp_min(1.0e-6)
    fit_x = normalize_features(fit_features.float(), feature_mean, feature_std)
    validation_x = normalize_features(
        validation_features.float(), feature_mean, feature_std
    )
    dataset = TensorDataset(fit_x, fit_utility.float(), fit_p1.float())
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )
    model = RouteDUtilityRegressor(
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
        loss_sum = 0.0
        rows = 0
        for features, utility, p1 in loader:
            features = features.to(device)
            utility = utility.to(device)
            p1 = p1.to(device)
            prediction = model(features)
            loss = utility_regression_loss(
                prediction,
                utility,
                p1,
                nonzero_weight=config.nonzero_weight,
                magnitude_weight=config.magnitude_weight,
                p1_loss_weight=config.p1_loss_weight,
                sign_loss_weight=config.sign_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = int(utility.numel())
            rows += batch_rows
            loss_sum += float(loss.item()) * batch_rows
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x.to(device)).cpu()
        validation_mae = float(
            (
                validation_prediction[:, 0]
                - validation_utility.float().cpu()
            ).abs().mean()
        )
        validation_sign = F.binary_cross_entropy_with_logits(
            validation_prediction[:, 0] / 0.1,
            (validation_utility.float().cpu() > 0.0).float(),
        )
        selection_value = validation_mae + 0.25 * float(validation_sign)
        row = {
            "epoch": epoch + 1,
            "fit_loss": loss_sum / max(rows, 1),
            "validation_selection_value": selection_value,
            "validation": regression_metrics(
                validation_prediction,
                validation_utility,
                validation_p1,
            ),
        }
        history.append(row)
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
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "best_epoch": best_epoch,
        "history": history,
    }


def infer_utility_regressor(
    model: RouteDUtilityRegressor,
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), max(1, int(batch_size))):
            batch = features[start : start + batch_size].float()
            batch = normalize_features(batch, feature_mean, feature_std).to(device)
            outputs.append(model(batch).cpu())
    return torch.cat(outputs, dim=0)
