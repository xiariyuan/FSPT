"""Training utilities for the frozen Route-D hypothesis scorer.

This module trains only :class:`HypothesisScorer` from precomputed candidate
caches. The MMP tracker remains frozen. Splits are performed by ``sample_id``
to prevent rows from the same video/sample leaking across train and validation.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .hypothesis_scorer import HYPOTHESIS_FEATURE_DIM, HypothesisScorer
from .routeD_candidate_cache import (
    ROUTED_CACHE_REQUIRED_KEYS,
    validate_routeD_candidate_cache,
)


class RouteDCandidateCacheDataset(Dataset):
    """Row-wise dataset backed by an in-memory Route-D candidate cache."""

    def __init__(self, cache: Mapping[str, torch.Tensor]):
        validate_routeD_candidate_cache(cache)
        self.cache = {key: cache[key] for key in ROUTED_CACHE_REQUIRED_KEYS}

    def __len__(self) -> int:
        return int(self.cache["oracle_index"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.cache.items()}


@dataclass(frozen=True)
class ScorerTrainingConfig:
    hidden_dim: int = 128
    epochs: int = 20
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 17
    device: str = "cpu"
    patience: int = 5
    normalize_features: bool = True
    loss_mode: str = "cross_entropy"
    gain_weight_alpha: float = 1.0
    max_gain_weight: float = 6.0
    regret_loss_weight: float = 0.0
    pairwise_loss_weight: float = 0.0
    pairwise_margin: float = 0.5
    hard_positive_min_gain_px: float = 3.0
    model_selection_metric: str = "auto"
    selection_mode: str = "auto"
    gate_loss_weight: float = 1.0
    global_rank_loss_weight: float = 1.0
    gate_min_gain_px: float = 3.0
    gate_selection_threshold: float = 0.5
    gate_temperature: float = 1.0
    hard_negative_weight: float = 3.0
    utility_thresholds_px: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    utility_loss_weight: float = 1.0
    utility_gate_loss_weight: float = 1.0
    utility_rank_loss_weight: float = 0.5
    utility_harmful_selection_penalty_weight: float = 1.0
    utility_gate_margin: float = 0.2
    utility_hard_negative_weight: float = 4.0


@dataclass(frozen=True)
class SampleSplit:
    train_sample_ids: tuple[int, ...]
    validation_sample_ids: tuple[int, ...]


@dataclass(frozen=True)
class ThreeWaySampleSplit:
    fit_sample_ids: tuple[int, ...]
    model_validation_sample_ids: tuple[int, ...]
    calibration_sample_ids: tuple[int, ...]


def subset_routeD_candidate_cache(
    cache: Mapping[str, torch.Tensor], row_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    validate_routeD_candidate_cache(cache)
    row_mask = row_mask.to(dtype=torch.bool, device=cache["sample_id"].device)
    if row_mask.shape != cache["sample_id"].shape:
        raise ValueError("row_mask must match cache row count")
    subset = {key: cache[key][row_mask].clone() for key in ROUTED_CACHE_REQUIRED_KEYS}
    subset["format_version"] = cache.get(
        "format_version", torch.tensor(1, dtype=torch.long)
    ).clone()
    validate_routeD_candidate_cache(subset)
    return subset


def split_routeD_candidate_cache_by_sample(
    cache: Mapping[str, torch.Tensor],
    *,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], SampleSplit]:
    """Split by sample/video identity rather than individual rows."""
    validate_routeD_candidate_cache(cache)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    sample_ids = sorted(int(value) for value in torch.unique(cache["sample_id"]).tolist())
    if len(sample_ids) < 2:
        raise ValueError("At least two distinct sample_id values are required")
    rng = random.Random(int(seed))
    rng.shuffle(sample_ids)
    validation_count = max(1, int(round(len(sample_ids) * validation_fraction)))
    validation_count = min(validation_count, len(sample_ids) - 1)
    validation_ids = tuple(sorted(sample_ids[:validation_count]))
    train_ids = tuple(sorted(sample_ids[validation_count:]))

    sample_tensor = cache["sample_id"]
    train_mask = torch.zeros_like(sample_tensor, dtype=torch.bool)
    validation_mask = torch.zeros_like(sample_tensor, dtype=torch.bool)
    for sample_id in train_ids:
        train_mask |= sample_tensor == sample_id
    for sample_id in validation_ids:
        validation_mask |= sample_tensor == sample_id
    if (train_mask & validation_mask).any() or not (train_mask | validation_mask).all():
        raise RuntimeError("Invalid sample split masks")
    return (
        subset_routeD_candidate_cache(cache, train_mask),
        subset_routeD_candidate_cache(cache, validation_mask),
        SampleSplit(train_ids, validation_ids),
    )


def split_routeD_candidate_cache_three_way_by_sample(
    cache: Mapping[str, torch.Tensor],
    *,
    model_validation_fraction: float = 10.0 / 64.0,
    calibration_fraction: float = 13.0 / 64.0,
    seed: int = 17,
) -> tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    ThreeWaySampleSplit,
]:
    """Create disjoint fit, model-validation, and calibration partitions.

    Fractions are converted to deterministic sample counts after one seeded
    shuffle. At least one sample is assigned to each held-out partition and at
    least one remains for fitting. Rows from a sample never cross partitions.
    """
    validate_routeD_candidate_cache(cache)
    if not 0.0 < model_validation_fraction < 1.0:
        raise ValueError("model_validation_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if model_validation_fraction + calibration_fraction >= 1.0:
        raise ValueError(
            "model-validation and calibration fractions must sum to less than 1"
        )

    sample_ids = sorted(
        int(value) for value in torch.unique(cache["sample_id"]).tolist()
    )
    if len(sample_ids) < 3:
        raise ValueError("At least three distinct sample_id values are required")
    rng = random.Random(int(seed))
    rng.shuffle(sample_ids)

    model_validation_count = max(
        1, int(round(len(sample_ids) * float(model_validation_fraction)))
    )
    calibration_count = max(
        1, int(round(len(sample_ids) * float(calibration_fraction)))
    )
    max_held_out = len(sample_ids) - 1
    if model_validation_count + calibration_count > max_held_out:
        overflow = model_validation_count + calibration_count - max_held_out
        reduce_calibration = min(overflow, calibration_count - 1)
        calibration_count -= reduce_calibration
        overflow -= reduce_calibration
        if overflow:
            model_validation_count -= overflow
    if model_validation_count < 1 or calibration_count < 1:
        raise RuntimeError("Unable to allocate non-empty held-out partitions")

    model_validation_ids = tuple(
        sorted(sample_ids[:model_validation_count])
    )
    calibration_start = model_validation_count
    calibration_stop = calibration_start + calibration_count
    calibration_ids = tuple(sorted(sample_ids[calibration_start:calibration_stop]))
    fit_ids = tuple(sorted(sample_ids[calibration_stop:]))

    partitions = [set(fit_ids), set(model_validation_ids), set(calibration_ids)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Three-way sample partitions overlap")
    if set().union(*partitions) != set(sample_ids):
        raise RuntimeError("Three-way sample partitions do not cover all samples")

    sample_tensor = cache["sample_id"]

    def subset(sample_id_values: tuple[int, ...]) -> Dict[str, torch.Tensor]:
        mask = torch.zeros_like(sample_tensor, dtype=torch.bool)
        for sample_id in sample_id_values:
            mask |= sample_tensor == sample_id
        return subset_routeD_candidate_cache(cache, mask)

    return (
        subset(fit_ids),
        subset(model_validation_ids),
        subset(calibration_ids),
        ThreeWaySampleSplit(
            fit_sample_ids=fit_ids,
            model_validation_sample_ids=model_validation_ids,
            calibration_sample_ids=calibration_ids,
        ),
    )


def compute_feature_normalization(
    cache: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only feature statistics over rows and candidates."""
    validate_routeD_candidate_cache(cache)
    features = cache["features"].float()
    mean = features.mean(dim=(0, 1))
    std = features.std(dim=(0, 1), unbiased=False)
    std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
    return mean, std


def normalize_hypothesis_features(
    features: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return features
    mean = mean.to(device=features.device, dtype=features.dtype)
    std = std.to(device=features.device, dtype=features.dtype)
    if mean.shape != (features.shape[-1],) or std.shape != mean.shape:
        raise ValueError("Feature normalization statistics have invalid shape")
    return (features - mean) / std


def resolve_selection_mode(config: ScorerTrainingConfig) -> str:
    mode = config.selection_mode
    if mode == "auto":
        mode = (
            "risk_gate"
            if config.loss_mode in {"risk_aware", "threshold_utility"}
            else "argmax"
        )
    if mode not in {"argmax", "risk_gate"}:
        raise ValueError(f"Unsupported selection_mode: {mode}")
    return mode


def select_routeD_candidate(
    logits: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    selection_mode: str = "argmax",
    gate_threshold: float = 0.5,
    gate_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select local or best global candidate and return gate probability."""
    if logits.shape != candidate_valid_mask.shape:
        raise ValueError("candidate_valid_mask must match logits")
    if logits.shape[-1] < 2:
        prediction = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
        probability = torch.zeros_like(prediction, dtype=logits.dtype)
        return prediction, probability, prediction
    if selection_mode == "argmax":
        prediction = logits.argmax(dim=-1)
        best_global = logits[..., 1:].argmax(dim=-1) + 1
        probability = (prediction > 0).to(logits.dtype)
        return prediction, probability, best_global
    if selection_mode != "risk_gate":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    global_logits = logits[..., 1:]
    global_valid = candidate_valid_mask[..., 1:]
    has_global = global_valid.any(dim=-1)
    best_global_score, best_global_relative = global_logits.max(dim=-1)
    best_global = best_global_relative + 1
    local_score = logits[..., 0]
    temperature = max(float(gate_temperature), 1.0e-6)
    gate_probability = torch.sigmoid((best_global_score - local_score) / temperature)
    choose_global = has_global & (gate_probability >= float(gate_threshold))
    prediction = torch.where(choose_global, best_global, torch.zeros_like(best_global))
    gate_probability = torch.where(has_global, gate_probability, torch.zeros_like(gate_probability))
    return prediction, gate_probability, best_global


def _risk_aware_loss(
    logits: torch.Tensor,
    candidate_error_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: ScorerTrainingConfig,
) -> torch.Tensor:
    """Two-stage objective: safe global gate plus ranking among global modes."""
    if logits.shape != candidate_error_px.shape or logits.shape != candidate_valid_mask.shape:
        raise ValueError("risk-aware tensors must share shape (B,K)")
    if logits.shape[-1] < 2:
        return logits.sum() * 0.0
    global_valid = candidate_valid_mask[..., 1:]
    has_global = global_valid.any(dim=-1)
    global_error = candidate_error_px[..., 1:].masked_fill(~global_valid, float("inf"))
    best_global_error, best_global_relative = global_error.min(dim=-1)
    local_error = candidate_error_px[..., 0]
    global_gain = local_error - best_global_error
    gate_target = has_global & (global_gain >= float(config.gate_min_gain_px))

    global_logits = logits[..., 1:]
    best_global_score = global_logits.max(dim=-1).values
    local_score = logits[..., 0]
    temperature = max(float(config.gate_temperature), 1.0e-6)
    gate_logit = (best_global_score - local_score) / temperature
    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logit, gate_target.to(logits.dtype), reduction="none"
    )

    gate_weight = torch.ones_like(gate_loss)
    positive = gate_target
    if positive.any():
        positive_gain = torch.log1p(global_gain.clamp_min(0.0))
        normalizer = positive_gain[positive].mean().clamp_min(1.0e-6)
        gate_weight = torch.where(
            positive,
            1.0 + positive_gain / normalizer,
            gate_weight,
        )
    harmful_negative = has_global & (global_gain <= 0.0)
    gate_weight = torch.where(
        harmful_negative,
        gate_weight * float(config.hard_negative_weight),
        gate_weight,
    )
    gate_objective = (gate_loss * gate_weight).sum() / gate_weight.sum().clamp_min(1.0)

    rank_objective = logits.sum() * 0.0
    if positive.any():
        rank_loss = F.cross_entropy(
            global_logits[positive], best_global_relative[positive], reduction="none"
        )
        rank_weight = torch.log1p(global_gain[positive].clamp_min(0.0))
        rank_objective = (rank_loss * rank_weight).sum() / rank_weight.sum().clamp_min(1.0)
    return (
        float(config.gate_loss_weight) * gate_objective
        + float(config.global_rank_loss_weight) * rank_objective
    )


def candidate_threshold_utility(
    candidate_error_px: torch.Tensor,
    thresholds_px: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Return TAP-style localization utility averaged over pixel thresholds."""
    thresholds = tuple(float(value) for value in thresholds_px)
    if not thresholds or any(value <= 0.0 for value in thresholds):
        raise ValueError("utility thresholds must be positive and non-empty")
    threshold_tensor = torch.tensor(
        thresholds,
        device=candidate_error_px.device,
        dtype=candidate_error_px.dtype,
    )
    return (
        candidate_error_px.unsqueeze(-1) <= threshold_tensor
    ).to(candidate_error_px.dtype).mean(dim=-1)


def _threshold_utility_loss(
    logits: torch.Tensor,
    candidate_error_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: ScorerTrainingConfig,
) -> torch.Tensor:
    """Learn candidate multi-threshold utility and a safe global-vs-local gate."""
    if logits.shape != candidate_error_px.shape or logits.shape != candidate_valid_mask.shape:
        raise ValueError("threshold-utility tensors must share shape (B,K)")
    utility = candidate_threshold_utility(
        candidate_error_px, config.utility_thresholds_px
    )
    valid_weight = candidate_valid_mask.to(logits.dtype)
    candidate_loss = F.binary_cross_entropy_with_logits(
        logits, utility, reduction="none"
    )
    candidate_objective = (
        candidate_loss * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)

    if logits.shape[-1] < 2:
        return float(config.utility_loss_weight) * candidate_objective

    global_valid = candidate_valid_mask[..., 1:]
    has_global = global_valid.any(dim=-1)
    global_utility = utility[..., 1:].masked_fill(~global_valid, -1.0)
    best_global_utility, best_global_relative = global_utility.max(dim=-1)
    local_utility = utility[..., 0]
    utility_gain = best_global_utility - local_utility
    gate_target = has_global & (
        utility_gain >= float(config.utility_gate_margin)
    )

    best_global_score = logits[..., 1:].masked_fill(
        ~global_valid, -1.0e9
    ).max(dim=-1).values
    gate_logit = best_global_score - logits[..., 0]
    gate_loss = F.binary_cross_entropy_with_logits(
        gate_logit, gate_target.to(logits.dtype), reduction="none"
    )
    gate_weight = torch.ones_like(gate_loss)
    harmful = has_global & (utility_gain < 0.0)
    gate_weight = torch.where(
        harmful,
        gate_weight * float(config.utility_hard_negative_weight),
        gate_weight,
    )
    positive_weight = 1.0 + utility_gain.clamp_min(0.0) * len(
        config.utility_thresholds_px
    )
    gate_weight = torch.where(gate_target, gate_weight * positive_weight, gate_weight)
    gate_objective = (
        gate_loss * gate_weight
    ).sum() / gate_weight.sum().clamp_min(1.0)

    rank_objective = logits.sum() * 0.0
    if gate_target.any():
        rank_loss = F.cross_entropy(
            logits[..., 1:][gate_target],
            best_global_relative[gate_target],
            reduction="none",
        )
        rank_weight = utility_gain[gate_target].clamp_min(1.0e-6)
        rank_objective = (
            rank_loss * rank_weight
        ).sum() / rank_weight.sum().clamp_min(1.0)

    return (
        float(config.utility_loss_weight) * candidate_objective
        + float(config.utility_gate_loss_weight) * gate_objective
        + float(config.utility_rank_loss_weight) * rank_objective
    )


def routeD_training_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_error_px: torch.Tensor,
    oracle_gain_px: torch.Tensor,
    config: ScorerTrainingConfig,
    candidate_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute CE, gain-weighted CE, or gain-weighted CE plus regret."""
    if config.loss_mode not in {
        "cross_entropy",
        "gain_weighted",
        "gain_regret",
        "gain_pairwise",
        "risk_aware",
        "threshold_utility",
    }:
        raise ValueError(f"Unsupported loss_mode: {config.loss_mode}")
    if config.loss_mode == "threshold_utility":
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones_like(logits, dtype=torch.bool)
        return _threshold_utility_loss(
            logits, candidate_error_px, candidate_valid_mask, config
        )
    if config.loss_mode == "risk_aware":
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones_like(logits, dtype=torch.bool)
        return _risk_aware_loss(
            logits, candidate_error_px, candidate_valid_mask, config
        )
    per_row_ce = F.cross_entropy(logits, target, reduction="none")
    weights = torch.ones_like(per_row_ce)
    if config.loss_mode in {"gain_weighted", "gain_regret", "gain_pairwise"}:
        positive = torch.log1p(oracle_gain_px.clamp_min(0.0))
        normalizer = positive[positive > 0].mean().clamp_min(1.0e-6)
        scaled = (positive / normalizer).clamp(max=float(config.max_gain_weight))
        weights = 1.0 + float(config.gain_weight_alpha) * scaled
    loss = (per_row_ce * weights).sum() / weights.sum().clamp_min(1.0)
    if config.loss_mode == "gain_pairwise" or config.pairwise_loss_weight > 0:
        hard_positive = (target > 0) & (
            oracle_gain_px >= float(config.hard_positive_min_gain_px)
        )
        if hard_positive.any():
            target_score = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            local_score = logits[..., 0]
            pairwise = F.relu(
                float(config.pairwise_margin) - (target_score - local_score)
            )
            pairwise_weight = torch.log1p(oracle_gain_px.clamp_min(0.0))
            pairwise_weight = pairwise_weight * hard_positive.to(pairwise_weight.dtype)
            pairwise_loss = (pairwise * pairwise_weight).sum() / pairwise_weight.sum().clamp_min(1.0)
            loss = loss + float(config.pairwise_loss_weight) * pairwise_loss
    if config.loss_mode == "gain_regret" or config.regret_loss_weight > 0:
        probabilities = torch.softmax(logits, dim=-1)
        oracle_error = candidate_error_px.min(dim=-1, keepdim=True).values
        regret = (candidate_error_px - oracle_error).clamp_min(0.0)
        row_scale = oracle_gain_px.clamp_min(1.0).unsqueeze(-1)
        expected_normalized_regret = (probabilities * (regret / row_scale)).sum(dim=-1)
        loss = loss + float(config.regret_loss_weight) * (
            expected_normalized_regret * weights
        ).sum() / weights.sum().clamp_min(1.0)
    return loss


def _masked_logits(logits: torch.Tensor, candidate_valid_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != candidate_valid_mask.shape:
        raise ValueError("candidate_valid_mask must match logits")
    if not candidate_valid_mask.any(dim=-1).all():
        raise ValueError("Every training row must contain at least one valid candidate")
    return logits.masked_fill(~candidate_valid_mask, -1.0e9)


def evaluate_hypothesis_scorer(
    model: HypothesisScorer,
    cache: Mapping[str, torch.Tensor],
    *,
    batch_size: int = 8192,
    device: str = "cpu",
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    selection_mode: str = "argmax",
    gate_threshold: float = 0.5,
    gate_temperature: float = 1.0,
    gate_min_gain_px: float = 3.0,
) -> Dict[str, float]:
    validate_routeD_candidate_cache(cache)
    dataset = RouteDCandidateCacheDataset(cache)
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False)
    model = model.to(device)
    model.eval()

    total_rows = 0
    total_loss = 0.0
    correct = 0
    selected_error_sum = 0.0
    local_error_sum = 0.0
    oracle_error_sum = 0.0
    global_selected = 0
    visible_rows = 0
    visible_correct = 0
    occluded_rows = 0
    occluded_correct = 0
    beneficial_global_rows = 0
    beneficial_global_selected = 0
    selected_global_beneficial = 0
    selected_global_harmful = 0
    selected_threshold_utility_harmful = 0.0
    selected_threshold_utility_sum = 0.0
    local_threshold_utility_sum = 0.0
    oracle_threshold_utility_sum = 0.0
    threshold_utility_beneficial_rows = 0
    threshold_utility_beneficial_selected = 0
    selected_global_threshold_harmful = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, feature_mean, feature_std
            )
            valid = batch["candidate_valid_mask"].to(device=device, dtype=torch.bool)
            target = batch["oracle_index"].to(device=device, dtype=torch.long)
            logits = _masked_logits(model(features), valid)
            loss = F.cross_entropy(logits, target, reduction="sum")
            prediction, gate_probability, best_global = select_routeD_candidate(
                logits,
                valid,
                selection_mode=selection_mode,
                gate_threshold=gate_threshold,
                gate_temperature=gate_temperature,
            )
            rows = int(target.numel())
            total_rows += rows
            total_loss += float(loss.item())
            is_correct = prediction == target
            correct += int(is_correct.sum().item())
            global_selected += int((prediction > 0).sum().item())

            candidate_points = batch["candidate_points"].to(device=device, dtype=torch.float32)
            gt_points = batch["gt_points"].to(device=device, dtype=torch.float32)
            candidate_error = batch["candidate_error_px"].to(
                device=device, dtype=torch.float32
            )
            local_error = batch["local_error_px"].to(device=device, dtype=torch.float32)
            oracle_error = batch["oracle_error_px"].to(device=device, dtype=torch.float32)
            selected_error = candidate_error.gather(
                -1, prediction.unsqueeze(-1)
            ).squeeze(-1)
            if candidate_points.shape[-1] != 2 or gt_points.shape[-1] != 2:
                raise ValueError("candidate/GT point coordinates must be 2D")
            selected_error_sum += float(selected_error.sum().item())
            local_error_sum += float(local_error.sum().item())
            oracle_error_sum += float(oracle_error.sum().item())
            threshold_utility = candidate_threshold_utility(
                candidate_error, (1.0, 2.0, 4.0, 8.0, 16.0)
            )
            selected_threshold_utility = threshold_utility.gather(
                -1, prediction.unsqueeze(-1)
            ).squeeze(-1)
            local_threshold_utility = threshold_utility[..., 0]
            oracle_threshold_utility = threshold_utility.max(dim=-1).values
            selected_threshold_utility_sum += float(
                selected_threshold_utility.sum().item()
            )
            local_threshold_utility_sum += float(
                local_threshold_utility.sum().item()
            )
            oracle_threshold_utility_sum += float(
                oracle_threshold_utility.sum().item()
            )
            best_global_threshold_utility = threshold_utility[..., 1:].masked_fill(
                ~valid[..., 1:], -1.0
            ).max(dim=-1).values
            threshold_beneficial = (
                best_global_threshold_utility >= local_threshold_utility + 0.2
            )
            threshold_utility_beneficial_rows += int(
                threshold_beneficial.sum().item()
            )
            threshold_utility_beneficial_selected += int(
                (threshold_beneficial & (prediction > 0)).sum().item()
            )
            harmful_selected_mask = (
                (prediction > 0)
                & (selected_threshold_utility < local_threshold_utility)
            )
            selected_global_threshold_harmful += int(
                harmful_selected_mask.sum().item()
            )
            selected_threshold_utility_harmful += float(
                (
                    local_threshold_utility[harmful_selected_mask]
                    - selected_threshold_utility[harmful_selected_mask]
                ).sum().item()
            )
            global_error = candidate_error[..., 1:].masked_fill(
                ~valid[..., 1:], float("inf")
            )
            best_global_error = global_error.min(dim=-1).values
            beneficial_global = (
                best_global_error + float(gate_min_gain_px) <= local_error
            )
            selected_global = prediction > 0
            selected_beneficial = selected_global & (
                selected_error + float(gate_min_gain_px) <= local_error
            )
            selected_harmful = selected_global & (selected_error > local_error)
            beneficial_global_rows += int(beneficial_global.sum().item())
            beneficial_global_selected += int(
                (beneficial_global & selected_global).sum().item()
            )
            selected_global_beneficial += int(selected_beneficial.sum().item())
            selected_global_harmful += int(selected_harmful.sum().item())

            visible = batch["visible"].to(device=device, dtype=torch.bool)
            visible_rows += int(visible.sum().item())
            visible_correct += int((is_correct & visible).sum().item())
            occluded = ~visible
            occluded_rows += int(occluded.sum().item())
            occluded_correct += int((is_correct & occluded).sum().item())

    denominator = max(total_rows, 1)
    mean_selected_error = selected_error_sum / denominator
    mean_local_error = local_error_sum / denominator
    mean_oracle_error = oracle_error_sum / denominator
    mean_selected_threshold_utility = selected_threshold_utility_sum / denominator
    mean_local_threshold_utility = local_threshold_utility_sum / denominator
    mean_oracle_threshold_utility = oracle_threshold_utility_sum / denominator
    return {
        "rows": float(total_rows),
        "cross_entropy": total_loss / denominator,
        "top1_accuracy": correct / denominator,
        "visible_top1_accuracy": visible_correct / max(visible_rows, 1),
        "occluded_top1_accuracy": occluded_correct / max(occluded_rows, 1),
        "global_selection_rate": global_selected / denominator,
        "mean_selected_error_px": mean_selected_error,
        "mean_local_error_px": mean_local_error,
        "mean_oracle_error_px": mean_oracle_error,
        "mean_regret_to_oracle_px": mean_selected_error - mean_oracle_error,
        "mean_gain_over_local_px": mean_local_error - mean_selected_error,
        "beneficial_global_rate": beneficial_global_rows / denominator,
        "beneficial_global_recall": beneficial_global_selected
        / max(beneficial_global_rows, 1),
        "global_selection_precision": selected_global_beneficial
        / max(global_selected, 1),
        "harmful_global_selection_rate": selected_global_harmful
        / max(global_selected, 1),
        "mean_selected_threshold_utility": mean_selected_threshold_utility,
        "mean_local_threshold_utility": mean_local_threshold_utility,
        "mean_oracle_threshold_utility": mean_oracle_threshold_utility,
        "mean_gain_over_local_threshold_utility": (
            mean_selected_threshold_utility - mean_local_threshold_utility
        ),
        "mean_regret_to_oracle_threshold_utility": (
            mean_oracle_threshold_utility - mean_selected_threshold_utility
        ),
        "threshold_utility_beneficial_global_rate": (
            threshold_utility_beneficial_rows / denominator
        ),
        "threshold_utility_beneficial_recall": (
            threshold_utility_beneficial_selected
            / max(threshold_utility_beneficial_rows, 1)
        ),
        "threshold_utility_harmful_global_rate": (
            selected_global_threshold_harmful / max(global_selected, 1)
        ),
        "threshold_utility_harmful_loss": (
            selected_threshold_utility_harmful / denominator
        ),
    }


def scorer_model_selection_value(
    validation_metrics: Mapping[str, float],
    config: ScorerTrainingConfig,
) -> tuple[str, float]:
    metric = config.model_selection_metric
    if metric == "auto":
        metric = (
            "cross_entropy"
            if config.loss_mode == "cross_entropy"
            else (
                "mean_regret_to_oracle_threshold_utility"
                if config.loss_mode == "threshold_utility"
                else "mean_selected_error_px"
            )
        )
    if metric not in {
        "cross_entropy",
        "mean_selected_error_px",
        "mean_regret_to_oracle_threshold_utility",
    }:
        raise ValueError(f"Unsupported model_selection_metric: {metric}")
    return metric, float(validation_metrics[metric])


def train_hypothesis_scorer(
    train_cache: Mapping[str, torch.Tensor],
    validation_cache: Mapping[str, torch.Tensor],
    config: ScorerTrainingConfig | None = None,
) -> Dict[str, object]:
    """Train scorer from frozen caches and return a serializable bundle."""
    config = config or ScorerTrainingConfig()
    validate_routeD_candidate_cache(train_cache)
    validate_routeD_candidate_cache(validation_cache)
    if train_cache["features"].shape[-1] != HYPOTHESIS_FEATURE_DIM:
        raise ValueError("Unexpected Route-D feature dimension")
    if validation_cache["features"].shape[-1] != HYPOTHESIS_FEATURE_DIM:
        raise ValueError("Unexpected validation feature dimension")

    torch.manual_seed(int(config.seed))
    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    if config.normalize_features:
        feature_mean, feature_std = compute_feature_normalization(train_cache)
    selection_mode = resolve_selection_mode(config)
    model = HypothesisScorer(
        feature_dim=HYPOTHESIS_FEATURE_DIM,
        hidden_dim=int(config.hidden_dim),
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    train_loader = DataLoader(
        RouteDCandidateCacheDataset(train_cache),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_selection_value = float("inf")
    best_selection_metric = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(int(config.epochs)):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for batch in train_loader:
            features = batch["features"].to(config.device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, feature_mean, feature_std
            )
            valid = batch["candidate_valid_mask"].to(config.device, dtype=torch.bool)
            target = batch["oracle_index"].to(config.device, dtype=torch.long)
            candidate_error = batch["candidate_error_px"].to(
                config.device, dtype=torch.float32
            )
            oracle_gain = batch["oracle_gain_px"].to(
                config.device, dtype=torch.float32
            )
            logits = _masked_logits(model(features), valid)
            loss = routeD_training_loss(
                logits,
                target,
                candidate_error,
                oracle_gain,
                config,
                candidate_valid_mask=valid,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(target.numel())
            loss_sum += float(loss.item()) * rows
            row_count += rows

        train_metrics = evaluate_hypothesis_scorer(
            model,
            train_cache,
            batch_size=config.batch_size,
            device=config.device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            selection_mode=selection_mode,
            gate_threshold=config.gate_selection_threshold,
            gate_temperature=config.gate_temperature,
            gate_min_gain_px=config.gate_min_gain_px,
        )
        validation_metrics = evaluate_hypothesis_scorer(
            model,
            validation_cache,
            batch_size=config.batch_size,
            device=config.device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            selection_mode=selection_mode,
            gate_threshold=config.gate_selection_threshold,
            gate_temperature=config.gate_temperature,
            gate_min_gain_px=config.gate_min_gain_px,
        )
        epoch_record = {
            "epoch": epoch + 1,
            "optimization_loss": loss_sum / max(row_count, 1),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        selection_metric, selection_value = scorer_model_selection_value(
            validation_metrics, config
        )
        if selection_value < best_selection_value - 1.0e-8:
            best_selection_value = selection_value
            best_selection_metric = selection_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                break

    model.load_state_dict(best_state, strict=True)
    final_train = evaluate_hypothesis_scorer(
        model,
        train_cache,
        batch_size=config.batch_size,
        device=config.device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        selection_mode=selection_mode,
        gate_threshold=config.gate_selection_threshold,
        gate_temperature=config.gate_temperature,
        gate_min_gain_px=config.gate_min_gain_px,
    )
    final_validation = evaluate_hypothesis_scorer(
        model,
        validation_cache,
        batch_size=config.batch_size,
        device=config.device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        selection_mode=selection_mode,
        gate_threshold=config.gate_selection_threshold,
        gate_temperature=config.gate_temperature,
        gate_min_gain_px=config.gate_min_gain_px,
    )
    return {
        "format_version": 1,
        "kind": "routeD_hypothesis_scorer",
        "config": asdict(config),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "feature_mean": feature_mean.detach().cpu() if feature_mean is not None else None,
        "feature_std": feature_std.detach().cpu() if feature_std is not None else None,
        "feature_dim": HYPOTHESIS_FEATURE_DIM,
        "candidate_count": int(train_cache["features"].shape[1]),
        "selection_mode": selection_mode,
        "train_rows": int(train_cache["features"].shape[0]),
        "validation_rows": int(validation_cache["features"].shape[0]),
        "history": history,
        "best_epoch": best_epoch,
        "model_selection_metric": best_selection_metric,
        "model_selection_value": best_selection_value,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
    }
