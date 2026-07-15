#!/usr/bin/env python3
"""Train a lightweight cross-domain Route-D action calibrator.

The MMP tracker, candidate generator, and multi-threshold scorer stay frozen.
The calibrator only predicts whether the frozen scorer's best global candidate
should replace local. All partitions are disjoint by sample/video ID.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import MultiThresholdHypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import require_causal_routeD_candidate_cache
from projects.mmp_tracker.mmp_tracker.routeD_multithreshold_training import (
    select_multithreshold_profile_candidate,
    threshold_hit_targets,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    RouteDCandidateCacheDataset,
    normalize_hypothesis_features,
    subset_routeD_candidate_cache,
)


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 17
    harmful_weight: float = 4.0
    utility_loss_penalty: float = 2.0
    delta1_tolerance: float = 0.0


class DomainActionCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", payload)
    require_causal_routeD_candidate_cache(cache)
    return cache, payload.get("metadata", {}) if isinstance(payload, dict) else {}


def split_sample_ids(sample_ids: torch.Tensor, seed: int):
    ids = sorted(set(int(value) for value in sample_ids.tolist()))
    if len(ids) < 20:
        raise ValueError("At least 20 unique samples are required for four-way calibration")
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    # For DAVIS-30 this gives 10 / 5 / 5 / 10.
    n = len(ids)
    fit_n = max(1, round(n / 3))
    validation_n = max(1, round(n / 6))
    policy_n = max(1, round(n / 6))
    if fit_n + validation_n + policy_n >= n:
        policy_n = max(1, n - fit_n - validation_n - 1)
    return {
        "fit": ids[:fit_n],
        "model_validation": ids[fit_n : fit_n + validation_n],
        "policy_calibration": ids[
            fit_n + validation_n : fit_n + validation_n + policy_n
        ],
        "holdout": ids[fit_n + validation_n + policy_n :],
    }


def subset_by_ids(cache: Mapping[str, torch.Tensor], ids: Sequence[int]):
    mask = torch.zeros_like(cache["sample_id"], dtype=torch.bool)
    for sample_id in ids:
        mask |= cache["sample_id"] == int(sample_id)
    if not mask.any():
        raise ValueError("Requested sample partition is empty")
    return subset_routeD_candidate_cache(cache, mask)


def load_frozen_scorer(bundle, device):
    thresholds = tuple(float(value) for value in bundle["thresholds_px"])
    hidden_dim = int(bundle.get("config", {}).get("hidden_dim", 0))
    if hidden_dim <= 0:
        hidden_dim = int(bundle["model_state"]["network.0.weight"].shape[0])
    model = MultiThresholdHypothesisScorer(
        int(bundle["feature_dim"]), hidden_dim, len(thresholds)
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    return model.to(device).eval(), thresholds


def infer_threshold_logits(model, cache, bundle, device, batch_size):
    loader = DataLoader(
        RouteDCandidateCacheDataset(cache), batch_size=max(1, batch_size), shuffle=False
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, bundle.get("feature_mean"), bundle.get("feature_std")
            )
            outputs.append(model(features).cpu())
    return torch.cat(outputs, dim=0)


def build_action_examples(cache, logits, thresholds):
    probabilities = torch.sigmoid(logits.float())
    valid = cache["candidate_valid_mask"].bool()
    raw = cache["features"].float()
    errors = cache["candidate_error_px"].float()
    targets = threshold_hit_targets(errors, thresholds)
    true_utility = targets.mean(dim=-1)

    local_profile = probabilities[:, 0]
    global_profiles = probabilities[:, 1:]
    global_valid = valid[:, 1:]
    global_coarse = global_profiles[:, :, 1:].mean(dim=-1)
    global_coarse = global_coarse.masked_fill(~global_valid, -1.0)
    best_global_relative = global_coarse.argmax(dim=-1)
    best_global_index = best_global_relative + 1

    gather_profile = best_global_relative[:, None, None].expand(
        -1, 1, probabilities.shape[-1]
    )
    best_global_profile = global_profiles.gather(1, gather_profile).squeeze(1)
    gather_raw = best_global_index[:, None, None].expand(-1, 1, raw.shape[-1])
    best_global_raw = raw.gather(1, gather_raw).squeeze(1)
    local_raw = raw[:, 0]

    best_global_true_utility = true_utility.gather(
        1, best_global_index[:, None]
    ).squeeze(1)
    local_true_utility = true_utility[:, 0]
    true_gain = best_global_true_utility - local_true_utility
    action_target = true_gain > 0.0

    p1_margin = best_global_profile[:, 0] - local_profile[:, 0]
    coarse_gain = best_global_profile[:, 1:].mean(dim=-1) - local_profile[:, 1:].mean(dim=-1)
    total_gain = best_global_profile.mean(dim=-1) - local_profile.mean(dim=-1)
    action_features = torch.cat(
        [
            local_profile,
            best_global_profile,
            best_global_profile - local_profile,
            local_raw,
            best_global_raw,
            best_global_raw - local_raw,
            p1_margin[:, None],
            coarse_gain[:, None],
            total_gain[:, None],
        ],
        dim=-1,
    )
    return {
        "features": action_features,
        "target": action_target,
        "true_gain": true_gain,
        "best_global_index": best_global_index,
        "true_utility": true_utility,
        "errors": errors,
        "valid": valid,
        "logits": logits,
        "predicted_p1_margin": p1_margin,
        "predicted_coarse_gain": coarse_gain,
        "predicted_total_gain": total_gain,
        "best_global_distance_to_local": best_global_raw[:, 2],
        "best_global_quality_gap": best_global_raw[:, 4],
    }


def normalize(values, mean, std):
    return (values - mean) / std.clamp_min(1e-6)


def binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.float().flatten()
    targets = targets.bool().flatten()
    positives = int(targets.sum())
    negatives = int((~targets).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    positive_rank_sum = ranks[targets].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def train_action_model(fit, validation, config: TrainConfig, device):
    mean = fit["features"].mean(dim=0)
    std = fit["features"].std(dim=0, unbiased=False).clamp_min(1e-6)
    fit_x = normalize(fit["features"], mean, std)
    val_x = normalize(validation["features"], mean, std)
    fit_y = fit["target"].float()
    val_y = validation["target"].float()
    fit_gain = fit["true_gain"].float()

    model = DomainActionCalibrator(fit_x.shape[-1], config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    positive = float(fit_y.sum())
    negative = float(fit_y.numel() - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    loader = DataLoader(
        TensorDataset(fit_x, fit_y, fit_gain),
        batch_size=max(1, config.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = rows = 0
        for x, y, gain in loader:
            x, y, gain = x.to(device), y.to(device), gain.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight, reduction="none"
            )
            importance = 1.0 + float(config.harmful_weight) * gain.abs()
            loss = (loss * importance).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(y.numel())
            rows += int(y.numel())
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, val_y.to(device), pos_weight=pos_weight
            )
            val_prob = torch.sigmoid(val_logits).cpu()
        row = {
            "epoch": epoch + 1,
            "fit_loss": total / max(rows, 1),
            "validation_bce": float(val_loss),
            "validation_auc": binary_auc(val_prob, validation["target"]),
        }
        history.append(row)
        if row["validation_bce"] < best_value - 1e-8:
            best_value = row["validation_bce"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, history, best_epoch


def action_probabilities(model, examples, mean, std, device, batch_size):
    loader = DataLoader(examples["features"], batch_size=max(1, batch_size), shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for values in loader:
            outputs.append(
                torch.sigmoid(model(normalize(values, mean, std).to(device))).cpu()
            )
    return torch.cat(outputs)


def evaluate_selection(examples, prediction, thresholds, name):
    errors = examples["errors"]
    true_utility = examples["true_utility"]
    selected_error = errors.gather(1, prediction[:, None]).squeeze(1)
    local_error = errors[:, 0]
    selected_utility = true_utility.gather(1, prediction[:, None]).squeeze(1)
    local_utility = true_utility[:, 0]
    selected_global = prediction > 0
    beneficial_available = examples["true_gain"] > 0
    harmful_loss = (local_utility - selected_utility).clamp_min(0.0)
    beneficial_selected = selected_global & (selected_utility > local_utility)
    harmful_selected = selected_global & (selected_utility < local_utility)
    result = {
        "name": name,
        "rows": int(prediction.numel()),
        "global_selection_rate": float(selected_global.float().mean()),
        "mean_selected_error_px": float(selected_error.mean()),
        "mean_local_error_px": float(local_error.mean()),
        "mean_gain_over_local_px": float((local_error - selected_error).mean()),
        "mean_selected_threshold_utility": float(selected_utility.mean()),
        "mean_local_threshold_utility": float(local_utility.mean()),
        "mean_gain_over_local_threshold_utility": float(
            (selected_utility - local_utility).mean()
        ),
        "mean_harmful_selection_utility_loss": float(harmful_loss.mean()),
        "beneficial_available_rate": float(beneficial_available.float().mean()),
        "beneficial_recall": float(
            beneficial_selected.sum() / beneficial_available.sum().clamp_min(1)
        ),
        "harmful_global_rate": float(
            harmful_selected.sum() / selected_global.sum().clamp_min(1)
        ),
    }
    selected_hits = threshold_hit_targets(selected_error[:, None], thresholds).squeeze(1)
    local_hits = threshold_hit_targets(local_error[:, None], thresholds).squeeze(1)
    for index, threshold in enumerate(thresholds):
        label = int(threshold) if float(threshold).is_integer() else threshold
        result[f"gain_delta_{label}"] = float(
            selected_hits[:, index].mean() - local_hits[:, index].mean()
        )
    return result


def policy_prediction_from_profile(examples, policy):
    prediction, _ = select_multithreshold_profile_candidate(
        examples["logits"],
        examples["valid"],
        p1_tolerance=policy["p1_tolerance"],
        min_coarse_gain=policy["min_coarse_gain"],
        min_total_gain=policy["min_total_gain"],
    )
    return prediction


def calibrate_probability_threshold(examples, probabilities, thresholds, config):
    candidates = []
    for value in [i / 100 for i in range(5, 100, 5)]:
        prediction = torch.where(
            probabilities >= value,
            examples["best_global_index"],
            torch.zeros_like(examples["best_global_index"]),
        )
        row = evaluate_selection(examples, prediction, thresholds, f"threshold_{value:.2f}")
        row["probability_threshold"] = value
        row["objective"] = (
            row["mean_gain_over_local_threshold_utility"]
            - config.utility_loss_penalty * row["mean_harmful_selection_utility_loss"]
        )
        candidates.append(row)
    feasible = [
        row
        for row in candidates
        if row["gain_delta_1"] >= -config.delta1_tolerance
        and row["global_selection_rate"] > 0
    ]
    if not feasible:
        return 1.0, candidates, True
    best = max(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mean_gain_over_local_threshold_utility"],
            row["gain_delta_4"],
            -row["mean_harmful_selection_utility_loss"],
        ),
    )
    return float(best["probability_threshold"]), candidates, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--harmful-weight", type=float, default=4.0)
    parser.add_argument("--utility-loss-penalty", type=float, default=2.0)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        harmful_weight=args.harmful_weight,
        utility_loss_penalty=args.utility_loss_penalty,
        delta1_tolerance=args.delta1_tolerance,
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bundle = torch.load(args.scorer_bundle, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "routeD_multithreshold_utility_scorer":
        raise ValueError("Expected a frozen multi-threshold Route-D scorer bundle")
    cache, metadata = load_cache(args.cache)
    partitions = split_sample_ids(cache["sample_id"], config.seed)
    partition_caches = {name: subset_by_ids(cache, ids) for name, ids in partitions.items()}

    scorer, thresholds = load_frozen_scorer(bundle, device)
    examples = {}
    for name, partition_cache in partition_caches.items():
        logits = infer_threshold_logits(
            scorer, partition_cache, bundle, device, config.batch_size
        )
        examples[name] = build_action_examples(partition_cache, logits, thresholds)

    model, feature_mean, feature_std, history, best_epoch = train_action_model(
        examples["fit"], examples["model_validation"], config, device
    )
    probabilities = {
        name: action_probabilities(
            model, values, feature_mean, feature_std, device, config.batch_size
        )
        for name, values in examples.items()
    }
    frozen_threshold, threshold_rows, fallback = calibrate_probability_threshold(
        examples["policy_calibration"],
        probabilities["policy_calibration"],
        thresholds,
        config,
    )

    old_policy = {
        "p1_tolerance": 0.0,
        "min_coarse_gain": 0.05,
        "min_total_gain": -0.05,
    }
    evaluations = {}
    for partition in ["fit", "model_validation", "policy_calibration", "holdout"]:
        data = examples[partition]
        action_prediction = torch.where(
            probabilities[partition] >= frozen_threshold,
            data["best_global_index"],
            torch.zeros_like(data["best_global_index"]),
        )
        profile_prediction = policy_prediction_from_profile(data, old_policy)
        oracle_gate_prediction = torch.where(
            data["true_gain"] > 0,
            data["best_global_index"],
            torch.zeros_like(data["best_global_index"]),
        )
        all_oracle_prediction = data["true_utility"].argmax(dim=-1)
        evaluations[partition] = {
            "action_calibrator": evaluate_selection(
                data, action_prediction, thresholds, "action_calibrator"
            ),
            "frozen_kubric_profile_gate": evaluate_selection(
                data, profile_prediction, thresholds, "frozen_kubric_profile_gate"
            ),
            "oracle_gate_on_frozen_ranking": evaluate_selection(
                data, oracle_gate_prediction, thresholds, "oracle_gate_on_frozen_ranking"
            ),
            "all_candidate_oracle": evaluate_selection(
                data, all_oracle_prediction, thresholds, "all_candidate_oracle"
            ),
            "action_auc": binary_auc(probabilities[partition], data["target"]),
            "positive_rate": float(data["target"].float().mean()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_bundle = {
        "format_version": 1,
        "kind": "routeD_domain_action_calibrator",
        "config": asdict(config),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(feature_mean.numel()),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "probability_threshold": frozen_threshold,
        "thresholds_px": list(thresholds),
        "sample_partitions": partitions,
        "source_scorer_bundle": str(Path(args.scorer_bundle).resolve()),
        "source_scorer_sha256": sha256(args.scorer_bundle),
        "source_cache": str(Path(args.cache).resolve()),
        "source_cache_sha256": sha256(args.cache),
        "cache_metadata": metadata,
        "best_epoch": best_epoch,
        "history": history,
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
    }
    torch.save(calibrator_bundle, output_dir / "domain_action_calibrator.pt")
    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": "four-way disjoint video split; frozen MMP/candidates/scorer",
        "sample_partitions": partitions,
        "partition_rows": {
            name: int(values["target"].numel()) for name, values in examples.items()
        },
        "config": asdict(config),
        "best_epoch": best_epoch,
        "frozen_probability_threshold": frozen_threshold,
        "fallback_to_local": fallback,
        "threshold_calibration": threshold_rows,
        "evaluations": evaluations,
        "bundle": str((output_dir / "domain_action_calibrator.pt").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    compact = {
        "sample_partitions": partitions,
        "partition_rows": result["partition_rows"],
        "best_epoch": best_epoch,
        "frozen_probability_threshold": frozen_threshold,
        "fallback_to_local": fallback,
        "holdout": evaluations["holdout"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
