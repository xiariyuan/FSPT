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

from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    require_causal_routeD_candidate_cache,
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_multithreshold_training import (
    MultiThresholdTrainingConfig,
    evaluate_multithreshold_scorer,
    train_multithreshold_scorer,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import split_routeD_candidate_cache_by_sample
from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import MultiThresholdHypothesisScorer


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", payload) if isinstance(payload, dict) else payload
    validate_routeD_candidate_cache(cache)
    require_causal_routeD_candidate_cache(cache)
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--evaluation-cache", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--thresholds-px", default="1,2,4,8,16")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gate-temperature", type=float, default=0.1)
    p.add_argument("--gate-selection-threshold", type=float, default=0.5)
    p.add_argument("--utility-gate-margin", type=float, default=0.2)
    p.add_argument("--harmful-negative-weight", type=float, default=4.0)
    p.add_argument("--selection-harmful-penalty", type=float, default=2.0)
    args = p.parse_args()

    full_train = load_cache(args.train_cache)
    train_cache, internal_validation, split = split_routeD_candidate_cache_by_sample(
        full_train,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    thresholds = tuple(float(value) for value in args.thresholds_px.split(",") if value.strip())
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
    bundle = train_multithreshold_scorer(train_cache, internal_validation, config)
    bundle["source_train_cache"] = str(Path(args.train_cache).resolve())
    bundle["sample_split"] = {
        "train_sample_ids": list(split.train_sample_ids),
        "validation_sample_ids": list(split.validation_sample_ids),
    }
    bundle["external_evaluation_cache"] = None
    bundle["external_evaluation_used_for_model_selection"] = False
    if args.evaluation_cache:
        evaluation_cache = load_cache(args.evaluation_cache)
        model = MultiThresholdHypothesisScorer(
            bundle["feature_dim"], args.hidden_dim, len(thresholds)
        )
        model.load_state_dict(bundle["model_state"], strict=True)
        bundle["external_evaluation_cache"] = str(Path(args.evaluation_cache).resolve())
        bundle["external_evaluation_metrics"] = evaluate_multithreshold_scorer(
            model,
            evaluation_cache,
            config=config,
            feature_mean=bundle["feature_mean"],
            feature_std=bundle["feature_std"],
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output / "multithreshold_scorer.pt")
    summary = {
        key: value for key, value in bundle.items()
        if key not in {"model_state", "feature_mean", "feature_std"}
    }
    summary["feature_mean"] = bundle["feature_mean"].tolist() if bundle["feature_mean"] is not None else None
    summary["feature_std"] = bundle["feature_std"].tolist() if bundle["feature_std"] is not None else None
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(output / "multithreshold_scorer.pt")
    print(json.dumps({
        "internal_validation": bundle["validation_metrics"],
        "external_evaluation": bundle.get("external_evaluation_metrics"),
    }, indent=2))


if __name__ == "__main__":
    main()
