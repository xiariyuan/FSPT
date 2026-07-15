#!/usr/bin/env python3
"""Freeze a coverage-constrained Route-D action threshold and audit holdout.

Threshold selection uses only the policy-calibration video partition. The
holdout partition is evaluated once after the threshold is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_routeD_domain_action_calibrator import (
    DomainActionCalibrator,
    action_probabilities,
    binary_auc,
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    normalize,
    policy_prediction_from_profile,
    subset_by_ids,
)


def quantiles(values: torch.Tensor):
    values = values.float()
    return {
        "min": float(values.min()),
        "p01": float(torch.quantile(values, 0.01)),
        "p05": float(torch.quantile(values, 0.05)),
        "p10": float(torch.quantile(values, 0.10)),
        "p25": float(torch.quantile(values, 0.25)),
        "p50": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def prediction_for_threshold(examples, probabilities, threshold):
    return torch.where(
        probabilities >= float(threshold),
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )


def threshold_sweep(examples, probabilities, thresholds_px, args):
    rows = []
    for step in range(1, 100):
        threshold = step / 100.0
        prediction = prediction_for_threshold(examples, probabilities, threshold)
        row = evaluate_selection(
            examples, prediction, thresholds_px, f"threshold_{threshold:.2f}"
        )
        row["probability_threshold"] = threshold
        row["objective"] = (
            row["mean_gain_over_local_threshold_utility"]
            - float(args.utility_loss_penalty)
            * row["mean_harmful_selection_utility_loss"]
        )
        row["coverage_feasible"] = (
            row["global_selection_rate"] >= float(args.min_coverage)
            and row["global_selection_rate"] <= float(args.max_coverage)
        )
        row["risk_feasible"] = (
            row["gain_delta_1"] >= -float(args.delta1_tolerance)
            and row["mean_gain_over_local_threshold_utility"] > 0.0
            and row["mean_gain_over_local_px"] >= 0.0
            and row["harmful_global_rate"] <= float(args.max_harmful_rate)
        )
        rows.append(row)
    feasible = [
        row for row in rows if row["coverage_feasible"] and row["risk_feasible"]
    ]
    if not feasible:
        raise RuntimeError("No coverage-constrained risk-feasible threshold exists")
    best = max(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mean_gain_over_local_threshold_utility"],
            row["mean_gain_over_local_px"],
            -row["mean_harmful_selection_utility_loss"],
        ),
    )
    return float(best["probability_threshold"]), best, rows


def per_sample_rows(examples, probabilities, sample_ids, threshold, thresholds_px):
    rows = []
    for sample_id in sorted(set(int(value) for value in sample_ids.tolist())):
        mask = sample_ids == sample_id
        sub = {key: value[mask] for key, value in examples.items()}
        pred = prediction_for_threshold(sub, probabilities[mask], threshold)
        row = evaluate_selection(sub, pred, thresholds_px, "action_calibrator")
        row["sample_id"] = sample_id
        row["action_auc"] = binary_auc(probabilities[mask], sub["target"])
        row["positive_rate"] = float(sub["target"].float().mean())
        rows.append(row)
    return rows


def bootstrap_video_means(rows, metric, seed, resamples=20000):
    values = torch.tensor([float(row[metric]) for row in rows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    n = int(values.numel())
    samples = []
    batch = 2000
    remaining = int(resamples)
    while remaining > 0:
        take = min(batch, remaining)
        indices = torch.randint(0, n, (take, n), generator=generator)
        samples.append(values[indices].mean(dim=1))
        remaining -= take
    distribution = torch.cat(samples)
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
    parser.add_argument("--calibrator-bundle", required=True)
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--max-coverage", type=float, default=0.10)
    parser.add_argument("--max-harmful-rate", type=float, default=0.25)
    parser.add_argument("--utility-loss-penalty", type=float, default=2.0)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    action_bundle = torch.load(
        args.calibrator_bundle, map_location="cpu", weights_only=False
    )
    scorer_bundle = torch.load(
        args.scorer_bundle, map_location="cpu", weights_only=False
    )
    cache, metadata = load_cache(args.cache)
    partitions = action_bundle["sample_partitions"]
    required = {"fit", "model_validation", "policy_calibration", "holdout"}
    if set(partitions) != required:
        raise ValueError("Calibrator bundle does not contain the expected four-way split")
    sets = {name: set(int(v) for v in ids) for name, ids in partitions.items()}
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError(f"Partitions overlap: {left} and {right}")

    model = DomainActionCalibrator(
        int(action_bundle["input_dim"]),
        int(action_bundle.get("config", {}).get("hidden_dim", 64)),
    )
    model.load_state_dict(action_bundle["model_state"], strict=True)
    model.to(device).eval()
    feature_mean = action_bundle["feature_mean"].float()
    feature_std = action_bundle["feature_std"].float()
    scorer, thresholds_px = load_frozen_scorer(scorer_bundle, device)

    data = {}
    probabilities = {}
    sample_ids = {}
    for partition in ["policy_calibration", "holdout"]:
        part_cache = subset_by_ids(cache, partitions[partition])
        logits = infer_threshold_logits(
            scorer, part_cache, scorer_bundle, device, args.batch_size
        )
        data[partition] = build_action_examples(part_cache, logits, thresholds_px)
        probabilities[partition] = action_probabilities(
            model,
            data[partition],
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        sample_ids[partition] = part_cache["sample_id"].long()

    threshold, selected_calibration, sweep = threshold_sweep(
        data["policy_calibration"],
        probabilities["policy_calibration"],
        thresholds_px,
        args,
    )

    holdout_prediction = prediction_for_threshold(
        data["holdout"], probabilities["holdout"], threshold
    )
    holdout = evaluate_selection(
        data["holdout"], holdout_prediction, thresholds_px, "action_calibrator"
    )
    holdout["action_auc"] = binary_auc(
        probabilities["holdout"], data["holdout"]["target"]
    )
    holdout["positive_rate"] = float(data["holdout"]["target"].float().mean())

    old_policy = {
        "p1_tolerance": 0.0,
        "min_coarse_gain": 0.05,
        "min_total_gain": -0.05,
    }
    old_prediction = policy_prediction_from_profile(data["holdout"], old_policy)
    old_holdout = evaluate_selection(
        data["holdout"], old_prediction, thresholds_px, "frozen_kubric_profile_gate"
    )
    oracle_prediction = torch.where(
        data["holdout"]["true_gain"] > 0,
        data["holdout"]["best_global_index"],
        torch.zeros_like(data["holdout"]["best_global_index"]),
    )
    oracle_holdout = evaluate_selection(
        data["holdout"],
        oracle_prediction,
        thresholds_px,
        "oracle_gate_on_frozen_ranking",
    )

    per_video = per_sample_rows(
        data["holdout"],
        probabilities["holdout"],
        sample_ids["holdout"],
        threshold,
        thresholds_px,
    )
    stats = {
        metric: bootstrap_video_means(per_video, metric, args.seed + index)
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

    output = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "threshold_selection_partition": "policy_calibration",
        "holdout_used_for_threshold_selection": False,
        "protocol": {
            "min_coverage": args.min_coverage,
            "max_coverage": args.max_coverage,
            "max_harmful_rate": args.max_harmful_rate,
            "utility_loss_penalty": args.utility_loss_penalty,
            "delta1_tolerance": args.delta1_tolerance,
            "seed": args.seed,
        },
        "sample_partitions": partitions,
        "cache_metadata": metadata,
        "probability_distribution": {
            "policy_calibration": quantiles(probabilities["policy_calibration"]),
            "holdout": quantiles(probabilities["holdout"]),
        },
        "frozen_probability_threshold": threshold,
        "selected_policy_calibration": selected_calibration,
        "holdout": holdout,
        "holdout_frozen_kubric_profile_gate": old_holdout,
        "holdout_oracle_gate_on_frozen_ranking": oracle_holdout,
        "paired_video_bootstrap": stats,
        "per_video_holdout": per_video,
        "policy_calibration_sweep": sweep,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "frozen_probability_threshold": threshold,
                "selected_policy_calibration": selected_calibration,
                "probability_distribution": output["probability_distribution"],
                "holdout": holdout,
                "holdout_frozen_kubric_profile_gate": old_holdout,
                "holdout_oracle_gate_on_frozen_ranking": oracle_holdout,
                "paired_video_bootstrap": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
