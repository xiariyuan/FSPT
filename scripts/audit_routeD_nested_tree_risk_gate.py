#!/usr/bin/env python3
"""Nested video-level audit of a tree-based Route-D action-risk gate.

The frozen MMP tracker, candidate generator, and multi-threshold scorer are not
changed. Each outer test fold is excluded from tree configuration selection,
isotonic calibration, and policy calibration. DAVIS has already been used for
development, so the output is a development diagnostic only.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_routeD_nested_posthoc_calibration import (
    apply_isotonic,
    bootstrap_video_means,
    calibration_metrics,
    fit_isotonic,
    make_inner_split,
    make_outer_folds,
    mask_for_ids,
    per_video_rows,
    subset_examples,
)
from scripts.audit_routeD_nested_profile_risk_gate import (
    predict_profile_risk_gate,
    select_profile_risk_policy,
)
from scripts.train_routeD_domain_action_calibrator import (
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    policy_prediction_from_profile,
)


def parse_max_features(value: str):
    if value in {"sqrt", "log2"}:
        return value
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "numeric max_features must be in the interval (0, 1]"
        )
    return parsed


def to_numpy(features: torch.Tensor) -> np.ndarray:
    values = features.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def probability_metrics(probability: np.ndarray, target: np.ndarray):
    target = target.astype(np.int64, copy=False)
    if np.unique(target).size < 2:
        return {
            "rows": int(target.size),
            "positive_rate": float(target.mean()) if target.size else float("nan"),
            "auc": float("nan"),
            "average_precision": float("nan"),
            "brier": float("nan"),
        }
    return {
        "rows": int(target.size),
        "positive_rate": float(target.mean()),
        "auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
    }


def make_tree(
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features,
    seed: int,
    n_jobs: int,
):
    return ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        class_weight="balanced",
        criterion="entropy",
        bootstrap=False,
        n_jobs=int(n_jobs),
        random_state=int(seed),
    )


def fit_tree(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features,
    seed: int,
    n_jobs: int,
):
    model = make_tree(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        seed=seed,
        n_jobs=n_jobs,
    )
    model.fit(to_numpy(features), target.detach().bool().cpu().numpy().astype(np.int64))
    return model


def predict_probability(model, features: torch.Tensor) -> torch.Tensor:
    probability = model.predict_proba(to_numpy(features))
    if probability.shape[1] != 2:
        raise RuntimeError("Tree action model did not observe both target classes")
    return torch.from_numpy(probability[:, 1]).float()


def select_tree_config(
    fit_examples: Mapping[str, torch.Tensor],
    validation_examples: Mapping[str, torch.Tensor],
    *,
    n_estimators: int,
    min_samples_leaf_values: Sequence[int],
    max_features_values: Sequence[str | float],
    seed: int,
    n_jobs: int,
):
    rows = []
    validation_target = (
        validation_examples["target"].detach().bool().cpu().numpy().astype(np.int64)
    )
    for leaf_index, min_samples_leaf in enumerate(min_samples_leaf_values):
        for feature_index, max_features in enumerate(max_features_values):
            config_seed = int(seed + 101 * leaf_index + 17 * feature_index)
            model = fit_tree(
                fit_examples["features"],
                fit_examples["target"],
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                seed=config_seed,
                n_jobs=n_jobs,
            )
            probability = predict_probability(model, validation_examples["features"])
            metrics = probability_metrics(probability.numpy(), validation_target)
            rows.append(
                {
                    "min_samples_leaf": int(min_samples_leaf),
                    "max_features": max_features,
                    "seed": config_seed,
                    **metrics,
                }
            )
    finite = [
        row
        for row in rows
        if math.isfinite(float(row["average_precision"]))
        and math.isfinite(float(row["auc"]))
    ]
    if not finite:
        raise RuntimeError("No valid ExtraTrees configuration on model-validation split")
    best = max(
        finite,
        key=lambda row: (
            row["average_precision"],
            row["auc"],
            -row["brier"],
            -row["min_samples_leaf"],
        ),
    )
    return best, rows


def concatenate_examples(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
):
    result = {}
    for key in first:
        value = first[key]
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            result[key] = torch.cat([value, second[key]], dim=0)
        else:
            result[key] = value
    return result


def load_reference(path: str | None):
    if not path:
        return None
    reference_path = Path(path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    payload = json.loads(reference_path.read_text())
    return {
        "path": str(reference_path),
        "aggregate_oof": payload.get("aggregate_oof"),
        "oof_calibration": payload.get("oof_calibration"),
        "paired_video_bootstrap": payload.get("paired_video_bootstrap"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-output", default=None)
    parser.add_argument("--mlp-reference", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument(
        "--min-samples-leaf-values", type=int, nargs="+", default=[10, 20, 40]
    )
    parser.add_argument(
        "--max-features-values",
        type=parse_max_features,
        nargs="+",
        default=["sqrt"],
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
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
        "--coarse-gains", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05]
    )
    parser.add_argument(
        "--total-gains", type=float, nargs="+", default=[-0.05, -0.02, 0.0, 0.01]
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
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

    oof_raw_probability = torch.full_like(examples["true_gain"], float("nan"))
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

        selected_config, config_search = select_tree_config(
            fit_examples,
            validation_examples,
            n_estimators=args.n_estimators,
            min_samples_leaf_values=args.min_samples_leaf_values,
            max_features_values=args.max_features_values,
            seed=args.seed + fold_index * 10000,
            n_jobs=args.n_jobs,
        )
        refit_examples = concatenate_examples(fit_examples, validation_examples)
        tree = fit_tree(
            refit_examples["features"],
            refit_examples["target"],
            n_estimators=args.n_estimators,
            min_samples_leaf=int(selected_config["min_samples_leaf"]),
            max_features=selected_config["max_features"],
            seed=args.seed + fold_index,
            n_jobs=args.n_jobs,
        )

        raw_posthoc = predict_probability(tree, posthoc_examples["features"])
        raw_policy = predict_probability(tree, policy_examples["features"])
        raw_test = predict_probability(tree, test_examples["features"])
        isotonic = fit_isotonic(raw_posthoc, posthoc_examples["target"])
        policy_probability = apply_isotonic(isotonic, raw_policy)
        test_probability = apply_isotonic(isotonic, raw_test)

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
        oof_raw_probability[test_mask] = raw_test
        oof_probability[test_mask] = test_probability
        oof_prediction[test_mask] = test_prediction

        test_target_numpy = (
            test_examples["target"].detach().bool().cpu().numpy().astype(np.int64)
        )
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
                "selected_tree_config": selected_config,
                "tree_config_search": config_search,
                "fallback_to_local": fallback,
                "frozen_policy": policy,
                "selected_policy_calibration": selected_policy,
                "feasible_policy_count": int(
                    sum(bool(row["risk_feasible"]) for row in search_rows)
                ),
                "candidate_policy_count": len(search_rows),
                "test_raw_ranking": probability_metrics(
                    raw_test.numpy(), test_target_numpy
                ),
                "test_calibration": calibration_metrics(
                    test_probability, test_examples["target"]
                ),
                "test_metrics": evaluate_selection(
                    test_examples,
                    test_prediction,
                    thresholds_px,
                    "nested_tree_risk_gate",
                ),
            }
        )

    if not assigned.all():
        raise RuntimeError("Incomplete OOF assignment")
    if not torch.isfinite(oof_raw_probability).all() or not torch.isfinite(
        oof_probability
    ).all():
        raise RuntimeError("Non-finite OOF probabilities")

    aggregate = evaluate_selection(
        examples,
        oof_prediction,
        thresholds_px,
        "nested_tree_risk_gate",
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
        "nested_mlp_profile_risk_gate": load_reference(args.mlp_reference),
    }

    raw_target = examples["target"].detach().bool().cpu().numpy().astype(np.int64)
    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": (
            "5-fold nested video-level CV with disjoint tree configuration fit/"
            "validation, refit, isotonic calibration, policy calibration, and "
            "outer-test videos"
        ),
        "important_caveat": (
            "DAVIS has already been used for development; OOF results are not "
            "untouched final-test evidence."
        ),
        "config": vars(args),
        "cache_metadata": cache_metadata,
        "outer_folds": outer_folds,
        "fold_reports": fold_reports,
        "oof_raw_ranking": probability_metrics(
            oof_raw_probability.numpy(), raw_target
        ),
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
            "kind": "routeD_nested_tree_risk_oof",
            "evidence_tier": "development_diagnostic_only",
            "paper_claim_eligible": False,
            "sample_id": sample_id,
            "point_id": cache["point_id"].long(),
            "frame_id": cache["frame_id"].long(),
            "oof_raw_probability": oof_raw_probability,
            "oof_isotonic_probability": oof_probability,
            "oof_prediction": oof_prediction,
            "outer_folds": outer_folds,
            "fold_tree_configs": [
                row["selected_tree_config"] for row in fold_reports
            ],
            "fold_policies": [row["frozen_policy"] for row in fold_reports],
        },
        artifact_path,
    )

    print(
        json.dumps(
            {
                "oof_raw_ranking": result["oof_raw_ranking"],
                "oof_calibration": result["oof_calibration"],
                "aggregate_oof": aggregate,
                "paired_video_bootstrap": paired_stats,
                "fold_summary": [
                    {
                        "fold": row["fold"],
                        "test_ids": row["outer_test_sample_ids"],
                        "selected_tree_config": row["selected_tree_config"],
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
