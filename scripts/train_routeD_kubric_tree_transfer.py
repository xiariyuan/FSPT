#!/usr/bin/env python3
"""Freeze a Kubric-only ExtraTrees Route-D controller and audit transfer.

All tree fitting, tree configuration selection, isotonic calibration, and gate
policy selection use only the causal Kubric train cache. Kubric validation and
DAVIS are evaluated only after the controller is frozen. DAVIS has already been
used during Route-D development, so its result remains a transfer diagnostic,
not untouched paper evidence.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
    fit_isotonic,
    make_inner_split,
    mask_for_ids,
    per_video_rows,
    subset_examples,
)
from scripts.audit_routeD_nested_profile_risk_gate import (
    predict_profile_risk_gate,
    select_profile_risk_policy,
)
from scripts.audit_routeD_nested_tree_risk_gate import (
    concatenate_examples,
    fit_tree,
    parse_max_features,
    predict_probability,
    probability_metrics,
    select_tree_config,
)
from scripts.train_routeD_domain_action_calibrator import (
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    policy_prediction_from_profile,
)


def build_examples(
    cache_path: str | Path,
    scorer,
    scorer_bundle,
    thresholds_px,
    *,
    device: torch.device,
    batch_size: int,
):
    cache, metadata = load_cache(cache_path)
    logits = infer_threshold_logits(
        scorer, cache, scorer_bundle, device, batch_size
    )
    examples = build_action_examples(cache, logits, thresholds_px)
    return cache, metadata, examples


def evaluate_frozen_controller(
    *,
    name: str,
    cache: Mapping[str, torch.Tensor],
    examples: Mapping[str, torch.Tensor],
    tree,
    isotonic,
    policy: Mapping[str, float],
    thresholds_px: Sequence[float],
    seed: int,
):
    raw_probability = predict_probability(tree, examples["features"])
    probability = apply_isotonic(isotonic, raw_probability)
    prediction = predict_profile_risk_gate(examples, probability, policy)
    aggregate = evaluate_selection(examples, prediction, thresholds_px, name)
    video_rows = per_video_rows(
        examples, prediction, cache["sample_id"].long(), thresholds_px
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
        metric: bootstrap_video_means(video_rows, metric, seed + index)
        for index, metric in enumerate(metrics_for_stats)
    }
    target_numpy = (
        examples["target"].detach().bool().cpu().numpy().astype(np.int64)
    )
    return {
        "raw_ranking": probability_metrics(raw_probability.numpy(), target_numpy),
        "calibration": calibration_metrics(probability, examples["target"]),
        "aggregate": aggregate,
        "paired_video_bootstrap": paired_stats,
        "per_video": video_rows,
        "raw_probability": raw_probability,
        "probability": probability,
        "prediction": prediction,
    }


def compact_evaluation(evaluation):
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"raw_probability", "probability", "prediction"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--kubric-validation-cache", required=True)
    parser.add_argument("--davis-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-output", default=None)
    parser.add_argument("--controller-output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
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
    scorer, thresholds_px = load_frozen_scorer(scorer_bundle, device)

    train_cache, train_metadata, train_examples = build_examples(
        args.train_cache,
        scorer,
        scorer_bundle,
        thresholds_px,
        device=device,
        batch_size=args.batch_size,
    )
    train_sample_id = train_cache["sample_id"].long()
    train_ids = sorted(set(int(value) for value in train_sample_id.tolist()))
    partitions = make_inner_split(train_ids, args.seed)
    masks = {
        name: mask_for_ids(train_sample_id, ids) for name, ids in partitions.items()
    }
    split_examples = {
        name: subset_examples(train_examples, mask) for name, mask in masks.items()
    }

    selected_config, config_search = select_tree_config(
        split_examples["fit"],
        split_examples["model_validation"],
        n_estimators=args.n_estimators,
        min_samples_leaf_values=args.min_samples_leaf_values,
        max_features_values=args.max_features_values,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    refit_examples = concatenate_examples(
        split_examples["fit"], split_examples["model_validation"]
    )
    tree = fit_tree(
        refit_examples["features"],
        refit_examples["target"],
        n_estimators=args.n_estimators,
        min_samples_leaf=int(selected_config["min_samples_leaf"]),
        max_features=selected_config["max_features"],
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    raw_posthoc = predict_probability(
        tree, split_examples["posthoc_fit"]["features"]
    )
    raw_policy = predict_probability(
        tree, split_examples["policy_calibration"]["features"]
    )
    isotonic = fit_isotonic(
        raw_posthoc, split_examples["posthoc_fit"]["target"]
    )
    policy_probability = apply_isotonic(isotonic, raw_policy)
    policy, selected_policy, policy_search, fallback = select_profile_risk_policy(
        split_examples["policy_calibration"],
        policy_probability,
        train_sample_id[masks["policy_calibration"]],
        thresholds_px,
        args,
    )

    # The controller is frozen here. The following caches are read only after
    # tree configuration, isotonic mapping, and policy have been fixed.
    validation_cache, validation_metadata, validation_examples = build_examples(
        args.kubric_validation_cache,
        scorer,
        scorer_bundle,
        thresholds_px,
        device=device,
        batch_size=args.batch_size,
    )
    validation_evaluation = evaluate_frozen_controller(
        name="kubric_validation_frozen_tree_transfer",
        cache=validation_cache,
        examples=validation_examples,
        tree=tree,
        isotonic=isotonic,
        policy=policy,
        thresholds_px=thresholds_px,
        seed=args.seed + 100,
    )

    davis_cache, davis_metadata, davis_examples = build_examples(
        args.davis_cache,
        scorer,
        scorer_bundle,
        thresholds_px,
        device=device,
        batch_size=args.batch_size,
    )
    davis_evaluation = evaluate_frozen_controller(
        name="davis_frozen_kubric_tree_transfer",
        cache=davis_cache,
        examples=davis_examples,
        tree=tree,
        isotonic=isotonic,
        policy=policy,
        thresholds_px=thresholds_px,
        seed=args.seed + 200,
    )

    old_policy = {
        "p1_tolerance": 0.0,
        "min_coarse_gain": 0.05,
        "min_total_gain": -0.05,
    }
    validation_old_prediction = policy_prediction_from_profile(
        validation_examples, old_policy
    )
    davis_old_prediction = policy_prediction_from_profile(davis_examples, old_policy)

    result = {
        "evidence_tier": "development_transfer_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": (
            "ExtraTrees configuration selection, tree fitting, isotonic calibration, "
            "and policy calibration use only disjoint partitions of the causal "
            "Kubric train cache. Kubric validation and DAVIS are evaluated only "
            "after controller freeze."
        ),
        "important_caveat": (
            "DAVIS has already been used in Route-D development. This zero-fit "
            "transfer is stronger than DAVIS-fitted OOF evidence but is not an "
            "untouched final-test claim."
        ),
        "config": vars(args),
        "thresholds_px": list(thresholds_px),
        "train_cache_metadata": train_metadata,
        "kubric_validation_cache_metadata": validation_metadata,
        "davis_cache_metadata": davis_metadata,
        "train_partitions": partitions,
        "partition_rows": {
            name: int(values["target"].numel())
            for name, values in split_examples.items()
        },
        "selected_tree_config": selected_config,
        "tree_config_search": config_search,
        "posthoc_calibration": calibration_metrics(
            apply_isotonic(isotonic, raw_posthoc),
            split_examples["posthoc_fit"]["target"],
        ),
        "policy_fallback_to_local": fallback,
        "frozen_policy": policy,
        "selected_policy_calibration": selected_policy,
        "feasible_policy_count": int(
            sum(bool(row["risk_feasible"]) for row in policy_search)
        ),
        "candidate_policy_count": len(policy_search),
        "kubric_validation": compact_evaluation(validation_evaluation),
        "davis_transfer": compact_evaluation(davis_evaluation),
        "references": {
            "kubric_validation_frozen_profile_gate": evaluate_selection(
                validation_examples,
                validation_old_prediction,
                thresholds_px,
                "kubric_validation_frozen_profile_gate",
            ),
            "davis_frozen_profile_gate": evaluate_selection(
                davis_examples,
                davis_old_prediction,
                thresholds_px,
                "davis_frozen_profile_gate",
            ),
        },
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
            "kind": "routeD_kubric_tree_transfer_audit",
            "evidence_tier": result["evidence_tier"],
            "paper_claim_eligible": False,
            "train_partitions": partitions,
            "selected_tree_config": selected_config,
            "frozen_policy": policy,
            "kubric_validation": {
                "sample_id": validation_cache["sample_id"].long(),
                "point_id": validation_cache["point_id"].long(),
                "frame_id": validation_cache["frame_id"].long(),
                "probability": validation_evaluation["probability"],
                "prediction": validation_evaluation["prediction"],
            },
            "davis_transfer": {
                "sample_id": davis_cache["sample_id"].long(),
                "point_id": davis_cache["point_id"].long(),
                "frame_id": davis_cache["frame_id"].long(),
                "probability": davis_evaluation["probability"],
                "prediction": davis_evaluation["prediction"],
            },
        },
        artifact_path,
    )

    controller_path = (
        Path(args.controller_output)
        if args.controller_output
        else output.with_suffix(".pkl")
    )
    with open(controller_path, "wb") as handle:
        pickle.dump(
            {
                "kind": "routeD_kubric_extratrees_controller",
                "evidence_tier": result["evidence_tier"],
                "paper_claim_eligible": False,
                "tree": tree,
                "isotonic": isotonic,
                "policy": policy,
                "selected_tree_config": selected_config,
                "thresholds_px": tuple(thresholds_px),
                "train_partitions": partitions,
                "scorer_bundle": str(args.scorer_bundle),
                "scorer_bundle_payload": scorer_bundle,
                "train_cache": str(args.train_cache),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        json.dumps(
            {
                "selected_tree_config": selected_config,
                "frozen_policy": policy,
                "policy_fallback_to_local": fallback,
                "kubric_validation": validation_evaluation["aggregate"],
                "davis_transfer": davis_evaluation["aggregate"],
                "davis_bootstrap": davis_evaluation["paired_video_bootstrap"],
                "output": str(output),
                "artifact": str(artifact_path),
                "controller": str(controller_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
