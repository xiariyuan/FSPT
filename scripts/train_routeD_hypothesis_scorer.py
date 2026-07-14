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
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    ScorerTrainingConfig,
    split_routeD_candidate_cache_by_sample,
    train_hypothesis_scorer,
)


def load_cache(path: str | Path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cache = payload.get("cache", payload) if isinstance(payload, dict) else payload
    validate_routeD_candidate_cache(cache)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Route-D hypothesis scorer")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cache = load_cache(args.train_cache)
    split_record = None
    if args.validation_cache:
        validation_cache = load_cache(args.validation_cache)
    else:
        train_cache, validation_cache, split = split_routeD_candidate_cache_by_sample(
            train_cache,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        split_record = {
            "train_sample_ids": list(split.train_sample_ids),
            "validation_sample_ids": list(split.validation_sample_ids),
        }

    config = ScorerTrainingConfig(
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
    )
    bundle = train_hypothesis_scorer(train_cache, validation_cache, config)
    bundle["source_train_cache"] = str(Path(args.train_cache).resolve())
    bundle["source_validation_cache"] = (
        str(Path(args.validation_cache).resolve()) if args.validation_cache else None
    )
    bundle["sample_split"] = split_record

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output / "hypothesis_scorer.pt")
    summary = {key: value for key, value in bundle.items() if key != "model_state"}
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(output / "hypothesis_scorer.pt")
    print(output / "metrics.json")
    print(json.dumps(bundle["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
