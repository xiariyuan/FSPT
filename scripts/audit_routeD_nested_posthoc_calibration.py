#!/usr/bin/env python3
"""Nested video-level post-hoc calibration audit for Route-D action gating.

For each outer fold, all test videos are excluded from action-head fitting,
early stopping, post-hoc probability calibration, and threshold selection.
The MMP tracker, candidate generator, and multi-threshold scorer remain frozen.
This is development-only cross-validation, not untouched benchmark evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_routeD_domain_action_calibrator import (
    TrainConfig,
    binary_auc,
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    normalize,
    policy_prediction_from_profile,
    train_action_model,
)


def subset_examples(examples: Mapping[str, torch.Tensor], mask: torch.Tensor):
    return {key: value[mask] for key, value in examples.items()}


def make_outer_folds(sample_ids: Sequence[int], folds: int, seed: int):
    ids = sorted(set(int(value) for value in sample_ids))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    return [ids[index::folds] for index in range(folds)]


def make_inner_split(train_ids: Sequence[int], seed: int):
    ids = list(int(value) for value in train_ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if len(ids) < 8:
        raise ValueError("Nested split requires at least eight outer-train videos")
    n = len(ids)
    # DAVIS outer train has 24 videos -> 12 / 4 / 4 / 4.
    fit_n = n // 2
    remaining = n - fit_n
    val_n = remaining // 3
    posthoc_n = remaining // 3
    policy_n = remaining - val_n - posthoc_n
    if min(fit_n, val_n, posthoc_n, policy_n) <= 0:
        raise ValueError("Nested split produced an empty partition")
    return {
        "fit": ids[:fit_n],
        "model_validation": ids[fit_n : fit_n + val_n],
        "posthoc_fit": ids[fit_n + val_n : fit_n + val_n + posthoc_n],
        "policy_calibration": ids[fit_n + val_n + posthoc_n :],
    }


def mask_for_ids(sample_id: torch.Tensor, ids: Sequence[int]):
    mask = torch.zeros_like(sample_id, dtype=torch.bool)
    for value in ids:
        mask |= sample_id == int(value)
    return mask


def model_logits(model, examples, mean, std, device, batch_size):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(examples["features"].shape[0]), max(1, batch_size)):
            values = examples["features"][start : start + batch_size]
            outputs.append(model(normalize(values, mean, std).to(device)).cpu())
    return torch.cat(outputs)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.detach().float()
    targets = targets.detach().float()
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    return temperature


def apply_temperature(logits: torch.Tensor, temperature: float):
    return torch.sigmoid(logits.float() / float(temperature))


class ConstantCalibrator:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, values):
        return np.full(np.asarray(values).shape, self.value, dtype=np.float64)


def fit_isotonic(probabilities: torch.Tensor, targets: torch.Tensor):
    x = probabilities.detach().cpu().numpy().astype(np.float64)
    y = targets.detach().cpu().numpy().astype(np.int64)
    if np.unique(y).size < 2:
        return ConstantCalibrator(float(y.mean()))
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(x, y)
    return calibrator


def apply_isotonic(calibrator, probabilities: torch.Tensor):
    values = calibrator.predict(probabilities.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def calibration_metrics(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 10):
    probabilities = probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    targets_f = targets.float()
    brier = float((probabilities - targets_f).square().mean())
    nll = float(F.binary_cross_entropy(probabilities, targets_f))
    ece = 0.0
    bin_rows = []
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(targets_f[mask].mean())
        weight = count / max(int(probabilities.numel()), 1)
        ece += weight * abs(confidence - accuracy)
        bin_rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "rows": count,
                "mean_probability": confidence,
                "positive_rate": accuracy,
            }
        )
    return {
        "rows": int(probabilities.numel()),
        "positive_rate": float(targets_f.mean()),
        "auc": binary_auc(probabilities, targets),
        "brier": brier,
        "nll": nll,
        "ece10": float(ece),
        "reliability_bins": bin_rows,
    }


def prediction_for_threshold(examples, probabilities, threshold):
    if not math.isfinite(float(threshold)):
        return torch.zeros_like(examples["best_global_index"])
    return torch.where(
        probabilities >= float(threshold),
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )


def harmful_counts(examples, prediction):
    true_utility = examples["true_utility"]
    selected = prediction > 0
    selected_utility = true_utility.gather(1, prediction[:, None]).squeeze(1)
    local_utility = true_utility[:, 0]
    harmful = selected & (selected_utility < local_utility)
    return int(harmful.sum()), int(selected.sum())


def clopper_pearson_upper(harmful: int, selected: int, alpha: float):
    if selected <= 0:
        return 1.0
    if harmful >= selected:
        return 1.0
    return float(beta.ppf(1.0 - float(alpha), harmful + 1, selected - harmful))


def select_conformal_policy(examples, probabilities, thresholds_px, args):
    rows = []
    coverages = np.arange(
        float(args.min_coverage),
        float(args.max_coverage) + 0.0001,
        float(args.coverage_step),
    )
    for target_coverage in coverages:
        threshold = float(
            torch.quantile(probabilities.float(), max(0.0, 1.0 - target_coverage))
        )
        prediction = prediction_for_threshold(examples, probabilities, threshold)
        row = evaluate_selection(
            examples, prediction, thresholds_px, f"coverage_{target_coverage:.4f}"
        )
        harmful, selected = harmful_counts(examples, prediction)
        row.update(
            {
                "target_coverage": float(target_coverage),
                "probability_threshold": threshold,
                "harmful_count": harmful,
                "selected_count": selected,
                "harmful_rate_upper_bound": clopper_pearson_upper(
                    harmful, selected, args.risk_alpha
                ),
            }
        )
        row["objective"] = (
            row["mean_gain_over_local_threshold_utility"]
            - float(args.utility_loss_penalty)
            * row["mean_harmful_selection_utility_loss"]
        )
        row["risk_feasible"] = (
            row["gain_delta_1"] >= -float(args.delta1_tolerance)
            and row["mean_gain_over_local_threshold_utility"] > 0.0
            and row["mean_gain_over_local_px"] >= 0.0
            and row["harmful_rate_upper_bound"] <= float(args.max_harmful_rate)
        )
        rows.append(row)
    feasible = [row for row in rows if row["risk_feasible"]]
    if not feasible:
        return float("inf"), None, rows, True
    best = max(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mean_gain_over_local_threshold_utility"],
            row["mean_gain_over_local_px"],
            -row["harmful_rate_upper_bound"],
        ),
    )
    return float(best["probability_threshold"]), best, rows, False


def per_video_rows(examples, prediction, sample_ids, thresholds_px):
    rows = []
    for sample_id in sorted(set(int(value) for value in sample_ids.tolist())):
        mask = sample_ids == sample_id
        sub = subset_examples(examples, mask)
        row = evaluate_selection(
            sub, prediction[mask], thresholds_px, "nested_cv_action_gate"
        )
        row["sample_id"] = sample_id
        rows.append(row)
    return rows


def bootstrap_video_means(rows, metric, seed, resamples=20000):
    values = torch.tensor([float(row[metric]) for row in rows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    n = int(values.numel())
    chunks = []
    for start in range(0, int(resamples), 2000):
        take = min(2000, int(resamples) - start)
        indices = torch.randint(0, n, (take, n), generator=generator)
        chunks.append(values[indices].mean(dim=1))
    distribution = torch.cat(chunks)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(torch.quantile(distribution, 0.025)),
        "ci95_high": float(torch.quantile(distribution, 0.975)),
        "positive_videos": int((values > 1e-12).sum()),
        "negative_videos": int((values < -1e-12).sum()),
        "tie_videos": int((values.abs() <= 1e-12).sum()),
        "videos": n,
        "bootstrap_resamples": int(resamples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--harmful-weight", type=float, default=4.0)
    parser.add_argument("--utility-loss-penalty", type=float, default=2.0)
    parser.add_argument("--min-coverage", type=float, default=0.02)
    parser.add_argument("--max-coverage", type=float, default=0.10)
    parser.add_argument("--coverage-step", type=float, default=0.005)
    parser.add_argument("--max-harmful-rate", type=float, default=0.25)
    parser.add_argument("--risk-alpha", type=float, default=0.05)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scorer_bundle = torch.load(
        args.scorer_bundle, map_location="cpu", weights_only=False
    )
    cache, metadata = load_cache(args.cache)
    scorer, thresholds_px = load_frozen_scorer(scorer_bundle, device)
    scorer_logits = infer_threshold_logits(
        scorer, cache, scorer_bundle, device, args.batch_size
    )
    examples = build_action_examples(cache, scorer_logits, thresholds_px)
    sample_id = cache["sample_id"].long()
    unique_ids = sorted(set(int(value) for value in sample_id.tolist()))
    outer_folds = make_outer_folds(unique_ids, args.folds, args.seed)

    methods = ["raw", "temperature", "isotonic"]
    variants = ["semantic_0p5", "conformal"]
    oof_prediction = {
        method: {
            variant: torch.zeros_like(examples["best_global_index"])
            for variant in variants
        }
        for method in methods
    }
    oof_probability = {
        method: torch.full_like(examples["true_gain"], float("nan"), dtype=torch.float32)
        for method in methods
    }
    assigned = torch.zeros_like(sample_id, dtype=torch.bool)
    fold_reports = []

    for fold_index, test_ids in enumerate(outer_folds):
        train_ids = [value for value in unique_ids if value not in set(test_ids)]
        inner = make_inner_split(train_ids, args.seed + 1000 + fold_index)
        masks = {name: mask_for_ids(sample_id, ids) for name, ids in inner.items()}
        test_mask = mask_for_ids(sample_id, test_ids)
        if (assigned & test_mask).any():
            raise RuntimeError("Outer test folds overlap")
        assigned |= test_mask

        train_config = TrainConfig(
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed + fold_index,
            harmful_weight=args.harmful_weight,
            utility_loss_penalty=args.utility_loss_penalty,
            delta1_tolerance=args.delta1_tolerance,
        )
        fit_examples = subset_examples(examples, masks["fit"])
        val_examples = subset_examples(examples, masks["model_validation"])
        posthoc_examples = subset_examples(examples, masks["posthoc_fit"])
        policy_examples = subset_examples(examples, masks["policy_calibration"])
        test_examples = subset_examples(examples, test_mask)

        model, feature_mean, feature_std, history, best_epoch = train_action_model(
            fit_examples, val_examples, train_config, device
        )
        raw_logits = {
            "posthoc_fit": model_logits(
                model, posthoc_examples, feature_mean, feature_std, device, args.batch_size
            ),
            "policy_calibration": model_logits(
                model, policy_examples, feature_mean, feature_std, device, args.batch_size
            ),
            "test": model_logits(
                model, test_examples, feature_mean, feature_std, device, args.batch_size
            ),
        }
        raw_probability = {
            name: torch.sigmoid(values) for name, values in raw_logits.items()
        }
        temperature = fit_temperature(
            raw_logits["posthoc_fit"], posthoc_examples["target"]
        )
        isotonic = fit_isotonic(
            raw_probability["posthoc_fit"], posthoc_examples["target"]
        )
        probabilities = {
            "raw": raw_probability,
            "temperature": {
                name: apply_temperature(values, temperature)
                for name, values in raw_logits.items()
            },
            "isotonic": {
                name: apply_isotonic(isotonic, values)
                for name, values in raw_probability.items()
            },
        }

        fold_report = {
            "fold": fold_index,
            "outer_test_sample_ids": test_ids,
            "inner_sample_partitions": inner,
            "partition_rows": {
                "fit": int(fit_examples["target"].numel()),
                "model_validation": int(val_examples["target"].numel()),
                "posthoc_fit": int(posthoc_examples["target"].numel()),
                "policy_calibration": int(policy_examples["target"].numel()),
                "test": int(test_examples["target"].numel()),
            },
            "best_epoch": best_epoch,
            "temperature": temperature,
            "methods": {},
        }
        for method in methods:
            policy_prob = probabilities[method]["policy_calibration"]
            test_prob = probabilities[method]["test"]
            threshold, selected_policy, sweep, fallback = select_conformal_policy(
                policy_examples, policy_prob, thresholds_px, args
            )
            semantic_prediction = prediction_for_threshold(
                test_examples, test_prob, 0.5
            )
            conformal_prediction = prediction_for_threshold(
                test_examples, test_prob, threshold
            )
            oof_prediction[method]["semantic_0p5"][test_mask] = semantic_prediction
            oof_prediction[method]["conformal"][test_mask] = conformal_prediction
            oof_probability[method][test_mask] = test_prob
            fold_report["methods"][method] = {
                "posthoc_fit_calibration": calibration_metrics(
                    probabilities[method]["posthoc_fit"], posthoc_examples["target"]
                ),
                "policy_calibration": calibration_metrics(
                    policy_prob, policy_examples["target"]
                ),
                "test_calibration": calibration_metrics(
                    test_prob, test_examples["target"]
                ),
                "conformal_probability_threshold": threshold,
                "conformal_fallback_to_local": fallback,
                "selected_policy_calibration": selected_policy,
                "policy_sweep": sweep,
                "test_semantic_0p5": evaluate_selection(
                    test_examples, semantic_prediction, thresholds_px, "semantic_0p5"
                ),
                "test_conformal": evaluate_selection(
                    test_examples, conformal_prediction, thresholds_px, "conformal"
                ),
            }
        fold_reports.append(fold_report)

    if not assigned.all():
        raise RuntimeError("Some rows were not assigned to an outer test fold")

    aggregate = {}
    per_video = {}
    paired_stats = {}
    for method in methods:
        aggregate[method] = {
            "oof_calibration": calibration_metrics(
                oof_probability[method], examples["target"]
            )
        }
        per_video[method] = {}
        paired_stats[method] = {}
        for variant in variants:
            prediction = oof_prediction[method][variant]
            aggregate[method][variant] = evaluate_selection(
                examples, prediction, thresholds_px, f"{method}_{variant}"
            )
            rows = per_video_rows(examples, prediction, sample_id, thresholds_px)
            per_video[method][variant] = rows
            paired_stats[method][variant] = {
                metric: bootstrap_video_means(
                    rows, metric, args.seed + 100 * methods.index(method) + index
                )
                for index, metric in enumerate(
                    [
                        "mean_gain_over_local_px",
                        "mean_gain_over_local_threshold_utility",
                        "gain_delta_1",
                        "gain_delta_2",
                        "gain_delta_4",
                        "gain_delta_8",
                        "gain_delta_16",
                        "global_selection_rate",
                    ]
                )
            }

    old_policy = {
        "p1_tolerance": 0.0,
        "min_coarse_gain": 0.05,
        "min_total_gain": -0.05,
    }
    old_prediction = policy_prediction_from_profile(examples, old_policy)
    oracle_prediction = torch.where(
        examples["true_gain"] > 0,
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )
    references = {
        "frozen_kubric_profile_gate": evaluate_selection(
            examples, old_prediction, thresholds_px, "frozen_kubric_profile_gate"
        ),
        "oracle_gate_on_frozen_ranking": evaluate_selection(
            examples, oracle_prediction, thresholds_px, "oracle_gate_on_frozen_ranking"
        ),
    }

    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": "5-fold nested video-level CV; outer test excluded from model fit, early stopping, post-hoc calibration, and threshold selection",
        "important_caveat": "DAVIS has already been used for development; these are out-of-fold development estimates, not untouched final-test claims.",
        "config": vars(args),
        "cache_metadata": metadata,
        "outer_folds": outer_folds,
        "fold_reports": fold_reports,
        "aggregate_oof": aggregate,
        "references_full_cache": references,
        "paired_video_bootstrap": paired_stats,
        "per_video_oof": per_video,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    compact = {
        "outer_folds": outer_folds,
        "aggregate_oof": aggregate,
        "references_full_cache": references,
        "paired_video_bootstrap": paired_stats,
        "fold_policy_summary": [
            {
                "fold": row["fold"],
                "outer_test_sample_ids": row["outer_test_sample_ids"],
                "methods": {
                    name: {
                        "threshold": values["conformal_probability_threshold"],
                        "fallback": values["conformal_fallback_to_local"],
                        "test_conformal": values["test_conformal"],
                    }
                    for name, values in row["methods"].items()
                },
            }
            for row in fold_reports
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
