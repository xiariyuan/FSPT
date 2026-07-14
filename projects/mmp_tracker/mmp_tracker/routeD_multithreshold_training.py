"""Explicit multi-threshold utility training for Route-D candidate selection."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .hypothesis_scorer import HYPOTHESIS_FEATURE_DIM, MultiThresholdHypothesisScorer
from .routeD_candidate_cache import validate_routeD_candidate_cache
from .routeD_scorer_training import (
    RouteDCandidateCacheDataset,
    compute_feature_normalization,
    normalize_hypothesis_features,
)


@dataclass(frozen=True)
class MultiThresholdTrainingConfig:
    thresholds_px: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    hidden_dim: int = 128
    epochs: int = 20
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 17
    device: str = "cpu"
    patience: int = 5
    normalize_features: bool = True
    bce_loss_weight: float = 1.0
    monotonic_loss_weight: float = 0.25
    gate_loss_weight: float = 1.0
    rank_loss_weight: float = 0.5
    utility_gate_margin: float = 0.2
    harmful_negative_weight: float = 4.0
    gate_temperature: float = 0.1
    gate_selection_threshold: float = 0.5
    selection_harmful_penalty: float = 2.0


def threshold_hit_targets(
    candidate_error_px: torch.Tensor,
    thresholds_px: Sequence[float],
) -> torch.Tensor:
    thresholds = tuple(float(value) for value in thresholds_px)
    if not thresholds or any(value <= 0.0 for value in thresholds):
        raise ValueError("thresholds_px must be positive and non-empty")
    threshold_tensor = torch.tensor(
        thresholds,
        device=candidate_error_px.device,
        dtype=candidate_error_px.dtype,
    )
    return (candidate_error_px.unsqueeze(-1) <= threshold_tensor).to(
        candidate_error_px.dtype
    )


def aggregate_threshold_utility(threshold_logits: torch.Tensor) -> torch.Tensor:
    if threshold_logits.ndim < 2:
        raise ValueError("threshold logits must include candidate and threshold dimensions")
    return torch.sigmoid(threshold_logits).mean(dim=-1)


def select_multithreshold_candidate(
    threshold_logits: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    gate_threshold: float = 0.5,
    gate_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    utility = aggregate_threshold_utility(threshold_logits)
    if utility.shape != candidate_valid_mask.shape:
        raise ValueError("candidate_valid_mask must match candidate utility shape")
    masked = utility.masked_fill(~candidate_valid_mask, -1.0)
    if masked.shape[-1] < 2:
        local = torch.zeros(masked.shape[:-1], dtype=torch.long, device=masked.device)
        probability = torch.zeros_like(local, dtype=utility.dtype)
        return local, probability, local, utility
    global_utility = masked[..., 1:]
    best_global_utility, best_global_relative = global_utility.max(dim=-1)
    best_global = best_global_relative + 1
    has_global = candidate_valid_mask[..., 1:].any(dim=-1)
    temperature = max(float(gate_temperature), 1.0e-6)
    gate_probability = torch.sigmoid(
        (best_global_utility - masked[..., 0]) / temperature
    )
    choose_global = has_global & (gate_probability >= float(gate_threshold))
    prediction = torch.where(choose_global, best_global, torch.zeros_like(best_global))
    gate_probability = torch.where(
        has_global, gate_probability, torch.zeros_like(gate_probability)
    )
    return prediction, gate_probability, best_global, utility


def select_multithreshold_profile_candidate(
    threshold_logits: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    p1_tolerance: float = 0.0,
    min_coarse_gain: float = 0.0,
    min_total_gain: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select a global candidate while explicitly protecting the 1-px head.

    The global candidate is ranked by the mean predicted correctness over all
    thresholds except the first/smallest one. It is accepted only if its
    predicted small-threshold probability is not worse than local beyond the
    configured tolerance and its predicted coarse/total utility gains pass the
    configured margins.
    """
    if threshold_logits.ndim < 3:
        raise ValueError("threshold_logits must have shape (..., K, T)")
    probabilities = torch.sigmoid(threshold_logits)
    if probabilities.shape[:-1] != candidate_valid_mask.shape:
        raise ValueError("candidate_valid_mask must match candidate dimensions")
    if probabilities.shape[-2] < 2:
        local = torch.zeros(
            probabilities.shape[:-2], dtype=torch.long, device=probabilities.device
        )
        zeros = torch.zeros_like(local, dtype=probabilities.dtype)
        return local, {
            "probabilities": probabilities,
            "best_global_index": local,
            "p1_margin": zeros,
            "coarse_gain": zeros,
            "total_gain": zeros,
            "eligible": torch.zeros_like(local, dtype=torch.bool),
        }

    local_profile = probabilities[..., 0, :]
    global_profiles = probabilities[..., 1:, :]
    global_valid = candidate_valid_mask[..., 1:]
    has_global = global_valid.any(dim=-1)
    if probabilities.shape[-1] > 1:
        local_coarse = local_profile[..., 1:].mean(dim=-1)
        global_coarse = global_profiles[..., 1:].mean(dim=-1)
    else:
        local_coarse = local_profile.mean(dim=-1)
        global_coarse = global_profiles.mean(dim=-1)
    global_coarse = global_coarse.masked_fill(~global_valid, -1.0)
    best_global_coarse, best_global_relative = global_coarse.max(dim=-1)
    best_global_index = best_global_relative + 1
    gather_index = best_global_relative.unsqueeze(-1).unsqueeze(-1).expand(
        *best_global_relative.shape, 1, probabilities.shape[-1]
    )
    best_global_profile = global_profiles.gather(-2, gather_index).squeeze(-2)
    p1_margin = best_global_profile[..., 0] - local_profile[..., 0]
    coarse_gain = best_global_coarse - local_coarse
    total_gain = best_global_profile.mean(dim=-1) - local_profile.mean(dim=-1)
    eligible = (
        has_global
        & (p1_margin >= -float(p1_tolerance))
        & (coarse_gain >= float(min_coarse_gain))
        & (total_gain >= float(min_total_gain))
    )
    prediction = torch.where(
        eligible, best_global_index, torch.zeros_like(best_global_index)
    )
    return prediction, {
        "probabilities": probabilities,
        "best_global_index": best_global_index,
        "best_global_profile": best_global_profile,
        "local_profile": local_profile,
        "p1_margin": p1_margin,
        "coarse_gain": coarse_gain,
        "total_gain": total_gain,
        "eligible": eligible,
    }


def multithreshold_training_loss(
    threshold_logits: torch.Tensor,
    candidate_error_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: MultiThresholdTrainingConfig,
) -> torch.Tensor:
    if threshold_logits.shape[:-1] != candidate_error_px.shape:
        raise ValueError("threshold logits must have shape (B,K,T)")
    if candidate_error_px.shape != candidate_valid_mask.shape:
        raise ValueError("candidate validity must match candidate errors")
    if threshold_logits.shape[-1] != len(config.thresholds_px):
        raise ValueError("threshold head count does not match thresholds_px")

    targets = threshold_hit_targets(candidate_error_px, config.thresholds_px)
    valid_weight = candidate_valid_mask.unsqueeze(-1).to(threshold_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        threshold_logits, targets, reduction="none"
    )
    bce_objective = (bce * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    probabilities = torch.sigmoid(threshold_logits)
    monotonic_violation = F.relu(
        probabilities[..., :-1] - probabilities[..., 1:]
    )
    monotonic_weight = candidate_valid_mask.unsqueeze(-1).to(probabilities.dtype)
    monotonic_objective = (
        monotonic_violation * monotonic_weight
    ).sum() / monotonic_weight.sum().clamp_min(1.0)

    true_utility = targets.mean(dim=-1)
    predicted_utility = probabilities.mean(dim=-1)
    global_valid = candidate_valid_mask[..., 1:]
    has_global = global_valid.any(dim=-1)
    true_global = true_utility[..., 1:].masked_fill(~global_valid, -1.0)
    best_true_global, best_true_relative = true_global.max(dim=-1)
    true_gain = best_true_global - true_utility[..., 0]
    gate_target = has_global & (
        true_gain >= float(config.utility_gate_margin)
    )

    predicted_global = predicted_utility[..., 1:].masked_fill(
        ~global_valid, -1.0
    )
    best_predicted_global = predicted_global.max(dim=-1).values
    temperature = max(float(config.gate_temperature), 1.0e-6)
    gate_logit = (
        best_predicted_global - predicted_utility[..., 0]
    ) / temperature
    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logit, gate_target.to(gate_logit.dtype), reduction="none"
    )
    gate_weight = torch.ones_like(gate_loss)
    harmful = has_global & (true_gain < 0.0)
    gate_weight = torch.where(
        harmful,
        gate_weight * float(config.harmful_negative_weight),
        gate_weight,
    )
    gate_weight = torch.where(
        gate_target,
        gate_weight * (1.0 + true_gain.clamp_min(0.0) * len(config.thresholds_px)),
        gate_weight,
    )
    gate_objective = (
        gate_loss * gate_weight
    ).sum() / gate_weight.sum().clamp_min(1.0)

    rank_objective = threshold_logits.sum() * 0.0
    if gate_target.any():
        global_rank_logits = predicted_utility[..., 1:] / temperature
        rank_loss = F.cross_entropy(
            global_rank_logits[gate_target],
            best_true_relative[gate_target],
            reduction="none",
        )
        rank_weight = true_gain[gate_target].clamp_min(1.0e-6)
        rank_objective = (
            rank_loss * rank_weight
        ).sum() / rank_weight.sum().clamp_min(1.0)

    return (
        float(config.bce_loss_weight) * bce_objective
        + float(config.monotonic_loss_weight) * monotonic_objective
        + float(config.gate_loss_weight) * gate_objective
        + float(config.rank_loss_weight) * rank_objective
    )


def evaluate_multithreshold_scorer(
    model: MultiThresholdHypothesisScorer,
    cache: Mapping[str, torch.Tensor],
    *,
    config: MultiThresholdTrainingConfig,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> Dict[str, float]:
    validate_routeD_candidate_cache(cache)
    loader = DataLoader(
        RouteDCandidateCacheDataset(cache),
        batch_size=max(1, int(config.batch_size)),
        shuffle=False,
    )
    model = model.to(config.device).eval()
    total_rows = 0
    selected_error_sum = local_error_sum = oracle_error_sum = 0.0
    selected_utility_sum = local_utility_sum = oracle_utility_sum = 0.0
    global_selected = harmful_global = beneficial_global = beneficial_selected = 0
    threshold_selected = torch.zeros(len(config.thresholds_px), dtype=torch.float64)
    threshold_local = torch.zeros_like(threshold_selected)
    threshold_oracle = torch.zeros_like(threshold_selected)
    bce_sum = 0.0
    monotonic_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(config.device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, feature_mean, feature_std
            )
            valid = batch["candidate_valid_mask"].to(config.device, dtype=torch.bool)
            errors = batch["candidate_error_px"].to(config.device, dtype=torch.float32)
            logits = model(features)
            targets = threshold_hit_targets(errors, config.thresholds_px)
            prediction, _, _, predicted_utility = select_multithreshold_candidate(
                logits,
                valid,
                gate_threshold=config.gate_selection_threshold,
                gate_temperature=config.gate_temperature,
            )
            selected_error = errors.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
            local_error = errors[..., 0]
            oracle_error = errors.masked_fill(~valid, float("inf")).min(dim=-1).values
            true_utility = targets.mean(dim=-1)
            selected_utility = true_utility.gather(
                -1, prediction.unsqueeze(-1)
            ).squeeze(-1)
            local_utility = true_utility[..., 0]
            oracle_utility = true_utility.masked_fill(~valid, -1.0).max(dim=-1).values
            selected_hits = targets.gather(
                -2,
                prediction.unsqueeze(-1).unsqueeze(-1).expand(
                    *prediction.shape, 1, targets.shape[-1]
                ),
            ).squeeze(-2)
            local_hits = targets[..., 0, :]
            oracle_hits = targets.masked_fill(
                ~valid.unsqueeze(-1), 0.0
            ).max(dim=-2).values

            rows = int(prediction.numel())
            total_rows += rows
            selected_error_sum += float(selected_error.sum().item())
            local_error_sum += float(local_error.sum().item())
            oracle_error_sum += float(oracle_error.sum().item())
            selected_utility_sum += float(selected_utility.sum().item())
            local_utility_sum += float(local_utility.sum().item())
            oracle_utility_sum += float(oracle_utility.sum().item())
            threshold_selected += selected_hits.sum(dim=0).cpu().double()
            threshold_local += local_hits.sum(dim=0).cpu().double()
            threshold_oracle += oracle_hits.sum(dim=0).cpu().double()
            selected_global = prediction > 0
            global_selected += int(selected_global.sum().item())
            harmful_global += int(
                (selected_global & (selected_utility < local_utility)).sum().item()
            )
            global_true_utility = true_utility[..., 1:].masked_fill(
                ~valid[..., 1:], -1.0
            ).max(dim=-1).values
            beneficial = global_true_utility >= (
                local_utility + float(config.utility_gate_margin)
            )
            beneficial_global += int(beneficial.sum().item())
            beneficial_selected += int((beneficial & selected_global).sum().item())
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            bce_sum += float(
                (bce * valid.unsqueeze(-1)).sum().item()
            )
            monotonic_sum += float(
                (
                    F.relu(torch.sigmoid(logits[..., :-1]) - torch.sigmoid(logits[..., 1:]))
                    * valid.unsqueeze(-1)
                ).sum().item()
            )

    denominator = max(total_rows, 1)
    threshold_metrics = {}
    for index, threshold in enumerate(config.thresholds_px):
        label = f"delta_{int(threshold) if float(threshold).is_integer() else threshold}"
        threshold_metrics[f"selected_{label}"] = float(threshold_selected[index] / denominator)
        threshold_metrics[f"local_{label}"] = float(threshold_local[index] / denominator)
        threshold_metrics[f"oracle_{label}"] = float(threshold_oracle[index] / denominator)
    selected_utility = selected_utility_sum / denominator
    local_utility = local_utility_sum / denominator
    oracle_utility = oracle_utility_sum / denominator
    harmful_rate = harmful_global / max(global_selected, 1)
    metrics = {
        "rows": float(total_rows),
        "global_selection_rate": global_selected / denominator,
        "mean_selected_error_px": selected_error_sum / denominator,
        "mean_local_error_px": local_error_sum / denominator,
        "mean_oracle_error_px": oracle_error_sum / denominator,
        "mean_gain_over_local_px": (local_error_sum - selected_error_sum) / denominator,
        "mean_selected_threshold_utility": selected_utility,
        "mean_local_threshold_utility": local_utility,
        "mean_oracle_threshold_utility": oracle_utility,
        "mean_gain_over_local_threshold_utility": selected_utility - local_utility,
        "mean_regret_to_oracle_threshold_utility": oracle_utility - selected_utility,
        "threshold_utility_beneficial_global_rate": beneficial_global / denominator,
        "threshold_utility_beneficial_recall": beneficial_selected / max(beneficial_global, 1),
        "threshold_utility_harmful_global_rate": harmful_rate,
        "threshold_bce": bce_sum / max(denominator * len(config.thresholds_px), 1),
        "monotonic_violation": monotonic_sum / max(
            denominator * max(len(config.thresholds_px) - 1, 1), 1
        ),
    }
    metrics["selection_objective"] = (
        metrics["mean_regret_to_oracle_threshold_utility"]
        + float(config.selection_harmful_penalty) * harmful_rate
    )
    metrics.update(threshold_metrics)
    return metrics


def train_multithreshold_scorer(
    train_cache: Mapping[str, torch.Tensor],
    validation_cache: Mapping[str, torch.Tensor],
    config: MultiThresholdTrainingConfig | None = None,
) -> Dict[str, object]:
    config = config or MultiThresholdTrainingConfig()
    validate_routeD_candidate_cache(train_cache)
    validate_routeD_candidate_cache(validation_cache)
    torch.manual_seed(int(config.seed))
    feature_mean = feature_std = None
    if config.normalize_features:
        feature_mean, feature_std = compute_feature_normalization(train_cache)
    model = MultiThresholdHypothesisScorer(
        HYPOTHESIS_FEATURE_DIM,
        int(config.hidden_dim),
        len(config.thresholds_px),
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loader = DataLoader(
        RouteDCandidateCacheDataset(train_cache),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
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
        for batch in loader:
            features = batch["features"].to(config.device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, feature_mean, feature_std
            )
            valid = batch["candidate_valid_mask"].to(config.device, dtype=torch.bool)
            errors = batch["candidate_error_px"].to(config.device, dtype=torch.float32)
            loss = multithreshold_training_loss(model(features), errors, valid, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_rows = int(errors.shape[0])
            rows += batch_rows
            loss_sum += float(loss.item()) * batch_rows
        train_metrics = evaluate_multithreshold_scorer(
            model, train_cache, config=config,
            feature_mean=feature_mean, feature_std=feature_std,
        )
        validation_metrics = evaluate_multithreshold_scorer(
            model, validation_cache, config=config,
            feature_mean=feature_mean, feature_std=feature_std,
        )
        history.append({
            "epoch": epoch + 1,
            "optimization_loss": loss_sum / max(rows, 1),
            "train": train_metrics,
            "validation": validation_metrics,
        })
        value = float(validation_metrics["selection_objective"])
        if value < best_value - 1.0e-8:
            best_value = value
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    model.load_state_dict(best_state, strict=True)
    final_train = evaluate_multithreshold_scorer(
        model, train_cache, config=config,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    final_validation = evaluate_multithreshold_scorer(
        model, validation_cache, config=config,
        feature_mean=feature_mean, feature_std=feature_std,
    )
    return {
        "format_version": 1,
        "kind": "routeD_multithreshold_utility_scorer",
        "config": asdict(config),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "feature_mean": feature_mean.detach().cpu() if feature_mean is not None else None,
        "feature_std": feature_std.detach().cpu() if feature_std is not None else None,
        "feature_dim": HYPOTHESIS_FEATURE_DIM,
        "thresholds_px": list(config.thresholds_px),
        "threshold_count": len(config.thresholds_px),
        "candidate_count": int(train_cache["features"].shape[1]),
        "train_rows": int(train_cache["features"].shape[0]),
        "validation_rows": int(validation_cache["features"].shape[0]),
        "history": history,
        "best_epoch": best_epoch,
        "model_selection_metric": "selection_objective",
        "model_selection_value": best_value,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
    }
