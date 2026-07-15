#!/usr/bin/env python3
"""Calibrate a small-threshold-protected Route-D policy on independent rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import MultiThresholdHypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import (
    require_causal_routeD_candidate_cache,
    validate_routeD_candidate_cache,
)
from projects.mmp_tracker.mmp_tracker.routeD_multithreshold_training import (
    select_multithreshold_profile_candidate,
    threshold_hit_targets,
)
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import (
    RouteDCandidateCacheDataset,
    normalize_hypothesis_features,
    subset_routeD_candidate_cache,
)


def load_cache(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = payload.get("cache", payload) if isinstance(payload, dict) else payload
    validate_routeD_candidate_cache(cache)
    require_causal_routeD_candidate_cache(cache)
    return cache


def subset_by_sample_ids(cache, sample_ids):
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")
    mask = torch.zeros_like(cache["sample_id"], dtype=torch.bool)
    for sample_id in sample_ids:
        mask |= cache["sample_id"] == int(sample_id)
    if not mask.any():
        raise ValueError("Requested sample IDs are absent from the cache")
    return subset_routeD_candidate_cache(cache, mask)


def infer_logits(model, cache, bundle, device, batch_size):
    loader = DataLoader(
        RouteDCandidateCacheDataset(cache),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )
    outputs = []
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, dtype=torch.float32)
            features = normalize_hypothesis_features(
                features, bundle.get("feature_mean"), bundle.get("feature_std")
            )
            outputs.append(model(features).cpu())
    return torch.cat(outputs, dim=0)


def evaluate_policy(cache, logits, thresholds, policy, harmful_penalty=2.0):
    valid = cache["candidate_valid_mask"].bool()
    prediction, diagnostics = select_multithreshold_profile_candidate(
        logits,
        valid,
        p1_tolerance=policy["p1_tolerance"],
        min_coarse_gain=policy["min_coarse_gain"],
        min_total_gain=policy["min_total_gain"],
    )
    errors = cache["candidate_error_px"].float()
    targets = threshold_hit_targets(errors, thresholds)
    selected_error = errors.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
    local_error = errors[..., 0]
    selected_hits = targets.gather(
        -2,
        prediction.unsqueeze(-1).unsqueeze(-1).expand(
            *prediction.shape, 1, targets.shape[-1]
        ),
    ).squeeze(-2)
    local_hits = targets[..., 0, :]
    selected_utility = selected_hits.mean(dim=-1)
    local_utility = local_hits.mean(dim=-1)
    selected_global = prediction > 0
    utility_loss = (local_utility - selected_utility).clamp_min(0.0)
    harmful = selected_global & (utility_loss > 0.0)
    beneficial = selected_global & (selected_utility > local_utility)
    mean_utility_gain = float((selected_utility - local_utility).mean())
    mean_harmful_utility_loss = float(utility_loss.mean())
    result = {
        **policy,
        "rows": int(prediction.numel()),
        "global_selection_rate": float(selected_global.float().mean()),
        "mean_selected_error_px": float(selected_error.mean()),
        "mean_local_error_px": float(local_error.mean()),
        "mean_gain_over_local_px": float((local_error - selected_error).mean()),
        "mean_selected_threshold_utility": float(selected_utility.mean()),
        "mean_local_threshold_utility": float(local_utility.mean()),
        "mean_gain_over_local_threshold_utility": mean_utility_gain,
        "mean_harmful_selection_utility_loss": mean_harmful_utility_loss,
        "calibration_objective": mean_utility_gain
        - float(harmful_penalty) * mean_harmful_utility_loss,
        "harmful_global_rate": float(
            harmful.sum() / selected_global.sum().clamp_min(1)
        ),
        "beneficial_selected_rate": float(
            beneficial.sum() / selected_global.sum().clamp_min(1)
        ),
        "mean_predicted_p1_margin_selected": (
            float(diagnostics["p1_margin"][selected_global].mean())
            if selected_global.any()
            else 0.0
        ),
        "mean_predicted_coarse_gain_selected": (
            float(diagnostics["coarse_gain"][selected_global].mean())
            if selected_global.any()
            else 0.0
        ),
    }
    for index, threshold in enumerate(thresholds):
        label = int(threshold) if float(threshold).is_integer() else threshold
        selected_value = float(selected_hits[..., index].mean())
        local_value = float(local_hits[..., index].mean())
        result[f"selected_delta_{label}"] = selected_value
        result[f"local_delta_{label}"] = local_value
        result[f"gain_delta_{label}"] = selected_value - local_value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--evaluation-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    parser.add_argument("--harmful-penalty", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "routeD_multithreshold_utility_scorer":
        raise ValueError("bundle must contain a multi-threshold scorer")
    sample_split = bundle.get("sample_split", {})
    required_split_keys = {
        "fit_sample_ids",
        "model_validation_sample_ids",
        "calibration_sample_ids",
    }
    missing = sorted(required_split_keys - set(sample_split))
    if missing:
        raise ValueError(
            "Bundle lacks an independent three-way split; missing keys: "
            + ", ".join(missing)
        )
    split_sets = [
        set(int(value) for value in sample_split[key])
        for key in [
            "fit_sample_ids",
            "model_validation_sample_ids",
            "calibration_sample_ids",
        ]
    ]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Bundle sample partitions overlap")

    thresholds = tuple(float(value) for value in bundle["thresholds_px"])
    hidden_dim = int(bundle.get("config", {}).get("hidden_dim", 0))
    if hidden_dim <= 0:
        hidden_dim = int(bundle["model_state"]["network.0.weight"].shape[0])
    model = MultiThresholdHypothesisScorer(
        int(bundle["feature_dim"]), hidden_dim, len(thresholds)
    )
    model.load_state_dict(bundle["model_state"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_cache = load_cache(args.train_cache)
    calibration_ids = sample_split["calibration_sample_ids"]
    calibration_cache = subset_by_sample_ids(train_cache, calibration_ids)
    calibration_logits = infer_logits(
        model, calibration_cache, bundle, device, args.batch_size
    )

    p1_tolerances = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    coarse_gains = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    total_gains = [-0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.075]
    calibration_results = []
    for p1_tolerance in p1_tolerances:
        for min_coarse_gain in coarse_gains:
            for min_total_gain in total_gains:
                policy = {
                    "p1_tolerance": p1_tolerance,
                    "min_coarse_gain": min_coarse_gain,
                    "min_total_gain": min_total_gain,
                }
                calibration_results.append(
                    evaluate_policy(
                        calibration_cache,
                        calibration_logits,
                        thresholds,
                        policy,
                        harmful_penalty=args.harmful_penalty,
                    )
                )

    baseline = evaluate_policy(
        calibration_cache,
        calibration_logits,
        thresholds,
        {
            "p1_tolerance": 0.0,
            "min_coarse_gain": 2.0,
            "min_total_gain": 2.0,
        },
        harmful_penalty=args.harmful_penalty,
    )
    feasible = [
        row
        for row in calibration_results
        if row["selected_delta_1"]
        >= baseline["local_delta_1"] - float(args.delta1_tolerance)
        and row["mean_gain_over_local_threshold_utility"] > 0.0
        and row["global_selection_rate"] > 0.0
    ]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                row["calibration_objective"],
                row["mean_gain_over_local_threshold_utility"],
                row["gain_delta_4"],
                -row["mean_harmful_selection_utility_loss"],
            ),
        )
        fallback_to_local = False
    else:
        best = baseline
        fallback_to_local = True
    frozen_policy = {
        key: best[key]
        for key in ["p1_tolerance", "min_coarse_gain", "min_total_gain"]
    }

    external = None
    if args.evaluation_cache:
        evaluation_cache = load_cache(args.evaluation_cache)
        evaluation_logits = infer_logits(
            model, evaluation_cache, bundle, device, args.batch_size
        )
        external = evaluate_policy(
            evaluation_cache,
            evaluation_logits,
            thresholds,
            frozen_policy,
            harmful_penalty=args.harmful_penalty,
        )

    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "calibration_scope": "independent_train_cache_calibration_partition",
        "model_validation_used_for_calibration": False,
        "external_evaluation_used_for_selection": False,
        "fit_sample_ids": list(sample_split["fit_sample_ids"]),
        "model_validation_sample_ids": list(
            sample_split["model_validation_sample_ids"]
        ),
        "calibration_sample_ids": list(calibration_ids),
        "thresholds_px": list(thresholds),
        "delta1_tolerance": args.delta1_tolerance,
        "harmful_penalty": args.harmful_penalty,
        "feasible_count": len(feasible),
        "fallback_to_local": fallback_to_local,
        "baseline_calibration": baseline,
        "best_calibration": best,
        "frozen_policy": frozen_policy,
        "external_evaluation": external,
        "calibration_results": calibration_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "calibration_results"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
