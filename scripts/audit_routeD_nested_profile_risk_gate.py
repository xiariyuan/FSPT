#!/usr/bin/env python3
"""Nested video-level audit of a safety-aware Route-D profile-risk gate.

Each outer test fold is excluded from action-head fitting, early stopping,
isotonic calibration, and gate-policy selection. The MMP tracker, candidate
generator, and multi-threshold scorer remain frozen. This is development-only
cross-validation because DAVIS has already been used during development.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_routeD_nested_posthoc_calibration import (
    apply_isotonic,
    bootstrap_video_means,
    calibration_metrics,
    clopper_pearson_upper,
    fit_isotonic,
    harmful_counts,
    make_inner_split,
    make_outer_folds,
    mask_for_ids,
    model_logits,
    per_video_rows,
    subset_examples,
)
from scripts.train_routeD_domain_action_calibrator import (
    TrainConfig,
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    policy_prediction_from_profile,
    train_action_model,
)


def predict_profile_risk_gate(
    examples: Mapping[str, torch.Tensor],
    probabilities: torch.Tensor,
    policy: Mapping[str, float],
) -> torch.Tensor:
    if not math.isfinite(float(policy["probability_threshold"])):
        return torch.zeros_like(examples["best_global_index"])
    selected = probabilities >= float(policy["probability_threshold"])
    selected &= examples["predicted_p1_margin"] >= float(policy["min_p1_margin"])
    selected &= examples["predicted_coarse_gain"] >= float(
        policy["min_coarse_gain"]
    )
    selected &= examples["predicted_total_gain"] >= float(
        policy["min_total_gain"]
    )
    return torch.where(
        selected,
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )


def summarize_policy_videos(
    examples: Mapping[str, torch.Tensor],
    prediction: torch.Tensor,
    sample_ids: torch.Tensor,
    thresholds_px: Sequence[float],
):
    rows = per_video_rows(examples, prediction, sample_ids, thresholds_px)
    delta1 = torch.tensor([row["gain_delta_1"] for row in rows], dtype=torch.float64)
    utility = torch.tensor(
        [row["mean_gain_over_local_threshold_utility"] for row in rows],
        dtype=torch.float64,
    )
    pixel = torch.tensor(
        [row["mean_gain_over_local_px"] for row in rows], dtype=torch.float64
    )
    return {
        "videos": len(rows),
        "mean_delta1": float(delta1.mean()),
        "mean_utility_gain": float(utility.mean()),
        "mean_pixel_gain": float(pixel.mean()),
        "negative_delta1_videos": int((delta1 < -1e-12).sum()),
        "negative_utility_videos": int((utility < -1e-12).sum()),
        "positive_utility_videos": int((utility > 1e-12).sum()),
        "per_video": rows,
    }


def candidate_probability_thresholds(
    probabilities: torch.Tensor,
    min_coverage: float,
    max_coverage: float,
    coverage_step: float,
):
    thresholds = set()
    for coverage in np.arange(
        float(min_coverage),
        float(max_coverage) + 1e-9,
        float(coverage_step),
    ):
        threshold = float(torch.quantile(probabilities.float(), 1.0 - coverage))
        thresholds.add(round(threshold, 10))
    return sorted(thresholds, reverse=True)


def select_profile_risk_policy(
    examples: Mapping[str, torch.Tensor],
    probabilities: torch.Tensor,
    sample_ids: torch.Tensor,
    thresholds_px: Sequence[float],
    args,
):
    probability_thresholds = candidate_probability_thresholds(
        probabilities,
        args.min_coverage,
        args.max_coverage,
        args.coverage_step,
    )
    rows = []
    for probability_threshold in probability_thresholds:
        for min_p1_margin in args.p1_margins:
            for min_coarse_gain in args.coarse_gains:
                for min_total_gain in args.total_gains:
                    policy = {
                        "probability_threshold": float(probability_threshold),
                        "min_p1_margin": float(min_p1_margin),
                        "min_coarse_gain": float(min_coarse_gain),
                        "min_total_gain": float(min_total_gain),
                    }
                    prediction = predict_profile_risk_gate(
                        examples, probabilities, policy
                    )
                    metrics = evaluate_selection(
                        examples,
                        prediction,
                        thresholds_px,
                        "profile_risk_policy",
                    )
                    actual_coverage = float(metrics["global_selection_rate"])
                    if not (
                        float(args.min_coverage)
                        <= actual_coverage
                        <= float(args.max_coverage)
                    ):
                        continue
                    harmful, selected = harmful_counts(examples, prediction)
                    video_summary = summarize_policy_videos(
                        examples, prediction, sample_ids, thresholds_px
                    )
                    metrics.update(policy)
                    metrics.update(
                        {
                            "harmful_count": harmful,
                            "selected_count": selected,
                            "harmful_rate_upper_bound": clopper_pearson_upper(
                                harmful, selected, args.risk_alpha
                            ),
                            "video_summary": video_summary,
                        }
                    )
                    metrics["objective"] = (
                        metrics["mean_gain_over_local_threshold_utility"]
                        - float(args.utility_loss_penalty)
                        * metrics["mean_harmful_selection_utility_loss"]
                        + float(args.delta4_reward)
                        * metrics["gain_delta_4"]
                    )
                    max_negative = int(
                        math.floor(
                            float(args.max_negative_video_fraction)
                            * int(video_summary["videos"])
                        )
                    )
                    metrics["risk_feasible"] = (
                        metrics["gain_delta_1"]
                        >= -float(args.delta1_tolerance)
                        and metrics["mean_gain_over_local_threshold_utility"] > 0.0
                        and metrics["mean_gain_over_local_px"] >= 0.0
                        and metrics["harmful_rate_upper_bound"]
                        <= float(args.max_harmful_rate)
                        and video_summary["mean_delta1"]
                        >= -float(args.delta1_tolerance)
                        and video_summary["mean_utility_gain"] > 0.0
                        and video_summary["negative_delta1_videos"] <= max_negative
                        and video_summary["negative_utility_videos"] <= max_negative
                    )
                    rows.append(metrics)
    feasible = [row for row in rows if row["risk_feasible"]]
    if not feasible:
        fallback = {
            "probability_threshold": float("inf"),
            "min_p1_margin": 0.0,
            "min_coarse_gain": 0.0,
            "min_total_gain": 0.0,
        }
        return fallback, None, rows, True
    best = max(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mean_gain_over_local_threshold_utility"],
            row["gain_delta_4"],
            row["mean_gain_over_local_px"],
            -row["harmful_rate_upper_bound"],
        ),
    )
    policy = {
        key: float(best[key])
        for key in [
            "probability_threshold",
            "min_p1_margin",
            "min_coarse_gain",
            "min_total_gain",
        ]
    }
    return policy, best, rows, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-output", default=None)
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
    parser.add_argument("--delta4-reward", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.02)
    parser.add_argument("--max-coverage", type=float, default=0.10)
    parser.add_argument("--coverage-step", type=float, default=0.005)
    parser.add_argument("--max-harmful-rate", type=float, default=0.25)
    parser.add_argument("--risk-alpha", type=float, default=0.05)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    parser.add_argument("--max-negative-video-fraction", type=float, default=0.5)
    parser.add_argument(
        "--p1-margins",
        type=float,
        nargs="+",
        default=[-0.05, -0.02, 0.0, 0.01, 0.02],
    )
    parser.add_argument(
        "--coarse-gains",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--total-gains",
        type=float,
        nargs="+",
        default=[-0.05, -0.02, 0.0, 0.01],
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scorer_bundle = torch.load(
        args.scorer_bundle, map_location="cpu", weights_only=False
    )
    cache, cache_metadata = load_cache(args.cache)
    scorer, thresholds_px = load_frozen_scorer(scorer_bundle, device)
    scorer_logits = infer_threshold_logits(
        scorer, cache, scorer_bundle, device, args.batch_size
    )
    examples = build_action_examples(cache, scorer_logits, thresholds_px)
    sample_id = cache["sample_id"].long()
    unique_ids = sorted(set(int(value) for value in sample_id.tolist()))
    outer_folds = make_outer_folds(unique_ids, args.folds, args.seed)

    oof_probability = torch.full_like(examples["true_gain"], float("nan"))
    oof_prediction = torch.zeros_like(examples["best_global_index"])
    assigned = torch.zeros_like(sample_id, dtype=torch.bool)
    fold_reports = []

    for fold_index, test_ids in enumerate(outer_folds):
        train_ids = [value for value in unique_ids if value not in set(test_ids)]
        inner = make_inner_split(train_ids, args.seed + 1000 + fold_index)
        masks = {name: mask_for_ids(sample_id, ids) for name, ids in inner.items()}
        test_mask = mask_for_ids(sample_id, test_ids)
        if (assigned & test_mask).any():
            raise RuntimeError("Outer folds overlap")
        assigned |= test_mask

        fit_examples = subset_examples(examples, masks["fit"])
        validation_examples = subset_examples(
            examples, masks["model_validation"]
        )
        posthoc_examples = subset_examples(examples, masks["posthoc_fit"])
        policy_examples = subset_examples(
            examples, masks["policy_calibration"]
        )
        test_examples = subset_examples(examples, test_mask)

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
        model, feature_mean, feature_std, history, best_epoch = train_action_model(
            fit_examples, validation_examples, train_config, device
        )
        raw_posthoc = model_logits(
            model,
            posthoc_examples,
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        raw_policy = model_logits(
            model,
            policy_examples,
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        raw_test = model_logits(
            model,
            test_examples,
            feature_mean,
            feature_std,
            device,
            args.batch_size,
        )
        isotonic = fit_isotonic(
            torch.sigmoid(raw_posthoc), posthoc_examples["target"]
        )
        policy_probability = apply_isotonic(isotonic, torch.sigmoid(raw_policy))
        test_probability = apply_isotonic(isotonic, torch.sigmoid(raw_test))

        policy, selected_policy, search_rows, fallback = select_profile_risk_policy(
            policy_examples,
            policy_probability,
            sample_id[masks["policy_calibration"]],
            thresholds_px,
            args,
        )
        test_prediction = predict_profile_risk_gate(
            test_examples, test_probability, policy
        )
        oof_probability[test_mask] = test_probability
        oof_prediction[test_mask] = test_prediction

        fold_reports.append(
            {
                "fold": fold_index,
                "outer_test_sample_ids": test_ids,
                "inner_sample_partitions": inner,
                "partition_rows": {
                    "fit": int(fit_examples["target"].numel()),
                    "model_validation": int(
                        validation_examples["target"].numel()
                    ),
                    "posthoc_fit": int(posthoc_examples["target"].numel()),
                    "policy_calibration": int(
                        policy_examples["target"].numel()
                    ),
                    "test": int(test_examples["target"].numel()),
                },
                "best_epoch": best_epoch,
                "fallback_to_local": fallback,
                "frozen_policy": policy,
                "selected_policy_calibration": selected_policy,
                "feasible_policy_count": int(
                    sum(bool(row["risk_feasible"]) for row in search_rows)
                ),
                "candidate_policy_count": len(search_rows),
                "test_calibration": calibration_metrics(
                    test_probability, test_examples["target"]
                ),
                "test_metrics": evaluate_selection(
                    test_examples,
                    test_prediction,
                    thresholds_px,
                    "nested_profile_risk_gate",
                ),
            }
        )

    if not assigned.all() or not torch.isfinite(oof_probability).all():
        raise RuntimeError("Incomplete OOF assignment")

    aggregate = evaluate_selection(
        examples,
        oof_prediction,
        thresholds_px,
        "nested_profile_risk_gate",
    )
    oof_video_rows = per_video_rows(
        examples, oof_prediction, sample_id, thresholds_px
    )
    metrics_for_stats = [
        "mean_gain_over_local_px",
        "mean_gain_over_local_threshold_utility",
        "gain_delta_1",
        "gain_delta_2",
        "gain_delta_4",
        "gain_delta_8",
        "gain_delta_16",
        "global_selection_rate",
    ]
    paired_stats = {
        metric: bootstrap_video_means(
            oof_video_rows, metric, args.seed + index
        )
        for index, metric in enumerate(metrics_for_stats)
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
            examples,
            old_prediction,
            thresholds_px,
            "frozen_kubric_profile_gate",
        ),
        "oracle_gate_on_frozen_ranking": evaluate_selection(
            examples,
            oracle_prediction,
            thresholds_px,
            "oracle_gate_on_frozen_ranking",
        ),
    }

    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": (
            "5-fold nested video-level CV with disjoint fit, model-validation, "
            "isotonic-calibration, policy-calibration, and outer-test videos"
        ),
        "important_caveat": (
            "DAVIS has already been used for development; OOF results are not "
            "untouched final-test evidence."
        ),
        "config": vars(args),
        "cache_metadata": cache_metadata,
        "outer_folds": outer_folds,
        "fold_reports": fold_reports,
        "oof_calibration": calibration_metrics(
            oof_probability, examples["target"]
        ),
        "aggregate_oof": aggregate,
        "paired_video_bootstrap": paired_stats,
        "per_video_oof": oof_video_rows,
        "references_full_cache": references,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    artifact_path = (
        Path(args.artifact_output)
        if args.artifact_output
        else output.with_suffix(".pt")
    )
    torch.save(
        {
            "kind": "routeD_nested_profile_risk_oof",
            "evidence_tier": "development_diagnostic_only",
            "paper_claim_eligible": False,
            "sample_id": sample_id,
            "point_id": cache["point_id"].long(),
            "frame_id": cache["frame_id"].long(),
            "oof_isotonic_probability": oof_probability,
            "oof_prediction": oof_prediction,
            "outer_folds": outer_folds,
            "fold_policies": [row["frozen_policy"] for row in fold_reports],
        },
        artifact_path,
    )
    print(
        json.dumps(
            {
                "aggregate_oof": aggregate,
                "oof_calibration": result["oof_calibration"],
                "paired_video_bootstrap": paired_stats,
                "references_full_cache": references,
                "fold_summary": [
                    {
                        "fold": row["fold"],
                        "test_ids": row["outer_test_sample_ids"],
                        "fallback": row["fallback_to_local"],
                        "feasible_policy_count": row["feasible_policy_count"],
                        "policy": row["frozen_policy"],
                        "test_metrics": row["test_metrics"],
                    }
                    for row in fold_reports
                ],
                "artifact": str(artifact_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
