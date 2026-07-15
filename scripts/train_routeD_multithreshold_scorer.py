#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import MultiThresholdHypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    require_causal_routeD_candidate_cache,
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_multithreshold_training import (
    MultiThresholdTrainingConfig,
    evaluate_multithreshold_scorer,
    train_multithreshold_scorer,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    split_routeD_candidate_cache_three_way_by_sample,
)


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", payload) if isinstance(payload, dict) else payload
    validate_routeD_candidate_cache(cache)
    require_causal_routeD_candidate_cache(cache)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--evaluation-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-validation-fraction", type=float, default=10.0 / 64.0
    )
    parser.add_argument("--calibration-fraction", type=float, default=13.0 / 64.0)
    parser.add_argument("--thresholds-px", default="1,2,4,8,16")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gate-temperature", type=float, default=0.1)
    parser.add_argument("--gate-selection-threshold", type=float, default=0.5)
    parser.add_argument("--utility-gate-margin", type=float, default=0.2)
    parser.add_argument("--harmful-negative-weight", type=float, default=4.0)
    parser.add_argument("--selection-harmful-penalty", type=float, default=2.0)
    args = parser.parse_args()

    full_train = load_cache(args.train_cache)
    fit_cache, model_validation_cache, calibration_cache, split = (
        split_routeD_candidate_cache_three_way_by_sample(
            full_train,
            model_validation_fraction=args.model_validation_fraction,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
    )
    thresholds = tuple(
        float(value) for value in args.thresholds_px.split(",") if value.strip()
    )
    config = MultiThresholdTrainingConfig(
        thresholds_px=thresholds,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        gate_temperature=args.gate_temperature,
        gate_selection_threshold=args.gate_selection_threshold,
        utility_gate_margin=args.utility_gate_margin,
        harmful_negative_weight=args.harmful_negative_weight,
        selection_harmful_penalty=args.selection_harmful_penalty,
    )
    bundle = train_multithreshold_scorer(
        fit_cache, model_validation_cache, config
    )
    bundle["format_version"] = 2
    bundle["source_train_cache"] = str(Path(args.train_cache).resolve())
    bundle["sample_split"] = {
        "fit_sample_ids": list(split.fit_sample_ids),
        "model_validation_sample_ids": list(split.model_validation_sample_ids),
        "calibration_sample_ids": list(split.calibration_sample_ids),
        "fit_samples": len(split.fit_sample_ids),
        "model_validation_samples": len(split.model_validation_sample_ids),
        "calibration_samples": len(split.calibration_sample_ids),
        "fit_rows": int(fit_cache["features"].shape[0]),
        "model_validation_rows": int(model_validation_cache["features"].shape[0]),
        "calibration_rows": int(calibration_cache["features"].shape[0]),
        "seed": int(args.seed),
        "model_validation_fraction": float(args.model_validation_fraction),
        "calibration_fraction": float(args.calibration_fraction),
    }
    bundle["model_selection_partition"] = "model_validation_sample_ids"
    bundle["calibration_partition"] = "calibration_sample_ids"
    bundle["external_evaluation_cache"] = None
    bundle["external_evaluation_used_for_model_selection"] = False
    bundle["external_evaluation_performed_before_policy_freeze"] = False

    # Optional diagnostic only. The protocol-correct workflow leaves this unset
    # until after independent calibration has frozen the profile gate.
    if args.evaluation_cache:
        evaluation_cache = load_cache(args.evaluation_cache)
        model = MultiThresholdHypothesisScorer(
            bundle["feature_dim"], args.hidden_dim, len(thresholds)
        )
        model.load_state_dict(bundle["model_state"], strict=True)
        bundle["external_evaluation_cache"] = str(
            Path(args.evaluation_cache).resolve()
        )
        bundle["external_evaluation_metrics"] = evaluate_multithreshold_scorer(
            model,
            evaluation_cache,
            config=config,
            feature_mean=bundle["feature_mean"],
            feature_std=bundle["feature_std"],
        )
        bundle["external_evaluation_performed_before_policy_freeze"] = True

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output / "multithreshold_scorer.pt")
    summary = {
        key: value
        for key, value in bundle.items()
        if key not in {"model_state", "feature_mean", "feature_std"}
    }
    summary["feature_mean"] = (
        bundle["feature_mean"].tolist()
        if bundle["feature_mean"] is not None
        else None
    )
    summary["feature_std"] = (
        bundle["feature_std"].tolist()
        if bundle["feature_std"] is not None
        else None
    )
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(output / "multithreshold_scorer.pt")
    print(
        json.dumps(
            {
                "sample_split": bundle["sample_split"],
                "model_validation": bundle["validation_metrics"],
                "external_evaluation": bundle.get("external_evaluation_metrics"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
