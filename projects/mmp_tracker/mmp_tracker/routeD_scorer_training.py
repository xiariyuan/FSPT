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


@dataclass(frozen=True)
class SampleSplit:
    train_sample_ids: tuple[int, ...]
    validation_sample_ids: tuple[int, ...]


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

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            valid = batch["candidate_valid_mask"].to(device=device, dtype=torch.bool)
            target = batch["oracle_index"].to(device=device, dtype=torch.long)
            logits = _masked_logits(model(features), valid)
            loss = F.cross_entropy(logits, target, reduction="sum")
            prediction = logits.argmax(dim=-1)
            rows = int(target.numel())
            total_rows += rows
            total_loss += float(loss.item())
            is_correct = prediction == target
            correct += int(is_correct.sum().item())
            global_selected += int((prediction > 0).sum().item())

            candidate_points = batch["candidate_points"].to(device=device, dtype=torch.float32)
            gt_points = batch["gt_points"].to(device=device, dtype=torch.float32)
            # Candidate caches retain normalized coordinates but also store exact
            # local/oracle errors in pixels. Selected error is reconstructed by
            # linear interpolation only for rank evaluation when resolution is
            # unavailable; therefore primary regret uses cached exact bounds.
            local_error = batch["local_error_px"].to(device=device, dtype=torch.float32)
            oracle_error = batch["oracle_error_px"].to(device=device, dtype=torch.float32)
            selected_is_oracle = is_correct.to(torch.float32)
            selected_error = torch.where(
                is_correct,
                oracle_error,
                local_error,
            )
            # candidate_points/gt_points are intentionally touched here to keep
            # shape validation close to evaluation and detect corrupted caches.
            if candidate_points.shape[-1] != 2 or gt_points.shape[-1] != 2:
                raise ValueError("candidate/GT point coordinates must be 2D")
            selected_error_sum += float(selected_error.sum().item())
            local_error_sum += float(local_error.sum().item())
            oracle_error_sum += float(oracle_error.sum().item())

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
    return {
        "rows": float(total_rows),
        "cross_entropy": total_loss / denominator,
        "top1_accuracy": correct / denominator,
        "visible_top1_accuracy": visible_correct / max(visible_rows, 1),
        "occluded_top1_accuracy": occluded_correct / max(occluded_rows, 1),
        "global_selection_rate": global_selected / denominator,
        "mean_selected_proxy_error_px": mean_selected_error,
        "mean_local_error_px": mean_local_error,
        "mean_oracle_error_px": mean_oracle_error,
        "mean_proxy_regret_to_oracle_px": mean_selected_error - mean_oracle_error,
        "mean_proxy_gain_over_local_px": mean_local_error - mean_selected_error,
    }


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
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(int(config.epochs)):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for batch in train_loader:
            features = batch["features"].to(config.device, dtype=torch.float32)
            valid = batch["candidate_valid_mask"].to(config.device, dtype=torch.bool)
            target = batch["oracle_index"].to(config.device, dtype=torch.long)
            logits = _masked_logits(model(features), valid)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows = int(target.numel())
            loss_sum += float(loss.item()) * rows
            row_count += rows

        train_metrics = evaluate_hypothesis_scorer(
            model, train_cache, batch_size=config.batch_size, device=config.device
        )
        validation_metrics = evaluate_hypothesis_scorer(
            model, validation_cache, batch_size=config.batch_size, device=config.device
        )
        epoch_record = {
            "epoch": epoch + 1,
            "optimization_loss": loss_sum / max(row_count, 1),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        validation_loss = float(validation_metrics["cross_entropy"])
        if validation_loss < best_validation_loss - 1.0e-8:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                break

    model.load_state_dict(best_state, strict=True)
    final_train = evaluate_hypothesis_scorer(
        model, train_cache, batch_size=config.batch_size, device=config.device
    )
    final_validation = evaluate_hypothesis_scorer(
        model, validation_cache, batch_size=config.batch_size, device=config.device
    )
    return {
        "format_version": 1,
        "kind": "routeD_hypothesis_scorer",
        "config": asdict(config),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "feature_dim": HYPOTHESIS_FEATURE_DIM,
        "candidate_count": int(train_cache["features"].shape[1]),
        "train_rows": int(train_cache["features"].shape[0]),
        "validation_rows": int(validation_cache["features"].shape[0]),
        "history": history,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
    }
