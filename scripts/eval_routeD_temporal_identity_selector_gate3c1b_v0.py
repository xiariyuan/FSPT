#!/usr/bin/env python3
"""Evaluate the frozen Gate 3C1B query-closure identity shortlist."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import (
    tensor_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_selector import (
    QueryClosureIdentitySelectorConfig,
    candidate_relative_zscore,
    select_query_closure_identity_candidates,
)

SCHEMA = "routeD_temporal_identity_selector_gate3c1b_v0"
RESULT_SCHEMA = "routeD_temporal_identity_selector_gate3c1b_v0_result"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/routeD_temporal_identity_selector_gate3c1b_v0.yaml"
)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    for path_key, hash_key in (
        ("gate3c1a_result", "gate3c1a_result_sha256"),
        ("gate3c1a_summary", "gate3c1a_summary_sha256"),
        ("gradient_train_cache_index", "gradient_train_cache_index_sha256"),
    ):
        path = Path(parent[path_key])
        if file_sha256(path) != parent[hash_key]:
            raise ValueError(f"Gate 3C1B parent hash drift: {path_key}")
    summary = json.loads(Path(parent["gate3c1a_summary"]).read_text())
    if (
        not summary.get("pass")
        or summary.get("formal_decision")
        != "AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION"
        or any(summary.get("locked_data", {}).values())
    ):
        raise ValueError("Gate 3C1A did not authorize Gate 3C1B")


def _selector_config(config: Mapping[str, Any]) -> QueryClosureIdentitySelectorConfig:
    mechanism = config["mechanism"]
    selection = config["selection"]
    return QueryClosureIdentitySelectorConfig(
        cycle_weight=float(mechanism["cycle_weight"]),
        query_frame_identity_weight=float(
            mechanism["query_frame_identity_weight"]
        ),
        mean_identity_weight=float(mechanism["mean_identity_weight"]),
        minimum_identity_weight=float(
            mechanism["minimum_identity_weight"]
        ),
        retained_nonnative=int(selection["retained_nonnative"]),
        zscore_epsilon=float(mechanism["candidate_relative_zscore_epsilon"]),
    )


def _stable_nonnative(score: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    local = score[:, 1:].masked_fill(
        ~valid[:, 1:], torch.finfo(score.dtype).min
    )
    order = torch.argsort(local, dim=1, descending=True, stable=True)[:, :k]
    return torch.cat(
        [
            torch.zeros(score.shape[0], 1, dtype=torch.long),
            order.cpu() + 1,
        ],
        dim=1,
    )


def _method_selections(
    tensors: Mapping[str, torch.Tensor],
    selector_config: QueryClosureIdentitySelectorConfig,
) -> dict[str, torch.Tensor]:
    valid = tensors["candidate_valid_mask"].bool()
    rows, candidates = valid.shape
    if rows == 0:
        empty = torch.empty(0, selector_config.retained_nonnative + 1, dtype=torch.long)
        return {
            "primary": empty,
            "static_m1": empty,
            "cycle_only": empty,
            "identity_only": empty,
        }
    primary = select_query_closure_identity_candidates(
        temporal_features=tensors["temporal_features"].float(),
        static_features=tensors["static_features"].float(),
        query_frames=tensors["query_frames"].long(),
        valid_mask=valid,
        config=selector_config,
    )
    static = torch.arange(
        selector_config.retained_nonnative + 1, dtype=torch.long
    )[None].expand(rows, -1).clone()

    static_features = tensors["static_features"].float()
    temporal_features = tensors["temporal_features"].float()
    query_frames = tensors["query_frames"].long()
    cycle = static_features[..., 8]
    mean_identity = static_features[..., 9]
    minimum_identity = static_features[..., 10]
    query_at = torch.gather(
        temporal_features[..., 0],
        dim=2,
        index=query_frames[:, None, None].expand(rows, candidates, 1),
    ).squeeze(2)
    cycle_score = -candidate_relative_zscore(
        cycle, valid, epsilon=selector_config.zscore_epsilon
    )
    identity_score = (
        candidate_relative_zscore(
            query_at, valid, epsilon=selector_config.zscore_epsilon
        )
        + candidate_relative_zscore(
            mean_identity, valid, epsilon=selector_config.zscore_epsilon
        )
        + candidate_relative_zscore(
            minimum_identity, valid, epsilon=selector_config.zscore_epsilon
        )
    )
    return {
        "primary": primary["selected_indices"].cpu(),
        "static_m1": static,
        "cycle_only": _stable_nonnative(
            cycle_score, valid, selector_config.retained_nonnative
        ),
        "identity_only": _stable_nonnative(
            identity_score, valid, selector_config.retained_nonnative
        ),
    }


def _method_row_metrics(
    tensors: Mapping[str, torch.Tensor], selected: torch.Tensor
) -> dict[str, torch.Tensor]:
    valid = tensors["candidate_valid_mask"].bool()
    distance = tensors["teacher_candidate_distance_px"].float().clone()
    distance[~valid] = float("inf")
    if selected.shape[0] == 0:
        empty = torch.empty(0)
        return {
            "minimum_distance_px": empty,
            "supported_4px": torch.empty(0, dtype=torch.bool),
            "supported_8px": torch.empty(0, dtype=torch.bool),
            "supported_12px": torch.empty(0, dtype=torch.bool),
        }
    selected_distance = distance.gather(1, selected)
    minimum = selected_distance.min(dim=1).values
    return {
        "minimum_distance_px": minimum,
        "supported_4px": minimum <= 4.0,
        "supported_8px": minimum <= 8.0,
        "supported_12px": minimum <= 12.0,
    }


def _bootstrap_video_gain(
    gains: np.ndarray, *, samples: int, seed: int
) -> dict[str, float]:
    if gains.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        bootstrap[index] = rng.choice(gains, gains.size, replace=True).mean()
    return {
        "mean": float(gains.mean()),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
    }


def _aggregate_method(values: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    minimum = torch.cat([value["minimum_distance_px"] for value in values])
    supported_4 = torch.cat([value["supported_4px"] for value in values])
    supported_8 = torch.cat([value["supported_8px"] for value in values])
    supported_12 = torch.cat([value["supported_12px"] for value in values])
    rows = int(minimum.numel())
    return {
        "rows": rows,
        "supported_4px_rows": int(supported_4.sum()),
        "recall_4px": 0.0 if rows == 0 else float(supported_4.float().mean()),
        "supported_8px_rows": int(supported_8.sum()),
        "recall_8px": 0.0 if rows == 0 else float(supported_8.float().mean()),
        "supported_12px_rows": int(supported_12.sum()),
        "recall_12px": 0.0 if rows == 0 else float(supported_12.float().mean()),
        "mean_minimum_distance_px": (
            0.0 if rows == 0 else float(minimum.mean())
        ),
        "median_minimum_distance_px": (
            0.0 if rows == 0 else float(minimum.median())
        ),
    }


def _expected_sources(config: Mapping[str, Any], partition: str) -> list[int]:
    bounds = config["partitions"][partition]["source_indices"]
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def _read_policy(config: Mapping[str, Any], partition: str) -> dict[str, bool]:
    policy = config["partitions"][partition]["read_state_after_evaluation"]
    return {key: bool(value) for key, value in policy.items()}


def _gate_checks(
    config: Mapping[str, Any],
    partition: str,
    primary: Mapping[str, Any],
    static: Mapping[str, Any],
    video_records: list[Mapping[str, Any]],
    video_ci: Mapping[str, float],
    *,
    exact_replay: bool,
    reference_provided: bool,
) -> dict[str, Any]:
    if partition == "gradient_train":
        return {
            "applicable": False,
            "pass": True,
            "decision": "TRAINING_DIAGNOSTIC_ONLY_PREREGISTER_CHECKPOINT_SELECTION",
        }
    thresholds = config["gates"][partition]
    gains = np.array([record["support_gain_12px"] for record in video_records])
    pooled_gain = float(primary["recall_12px"] - static["recall_12px"])
    median_improvement = float(
        static["median_minimum_distance_px"]
        - primary["median_minimum_distance_px"]
    )
    within8_gain = float(primary["recall_8px"] - static["recall_8px"])
    nonnegative_fraction = (
        0.0 if gains.size == 0 else float((gains >= 0.0).mean())
    )
    checks = {
        "pooled_support_gain": pooled_gain
        >= float(thresholds["minimum_pooled_support_gain_12px"]),
        "absolute_support_recall": primary["recall_12px"]
        >= float(thresholds["minimum_absolute_support_recall_12px"]),
        "paired_video_ci_lower_positive": video_ci["lower"] > 0.0,
        "median_minimum_distance_improvement": median_improvement
        >= float(thresholds["minimum_median_distance_improvement_px"]),
        "within8_recall_gain_positive": within8_gain > 0.0,
        "nonnegative_video_fraction": nonnegative_fraction
        >= float(thresholds["minimum_nonnegative_video_fraction"]),
        "exact_replay": bool(exact_replay),
    }
    passed = all(checks.values())
    if not reference_provided:
        decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
        passed = False
    else:
        decision = thresholds["decision_pass" if passed else "decision_fail"]
    return {
        "applicable": True,
        "checks": checks,
        "pass": passed,
        "pooled_support_gain_12px": pooled_gain,
        "median_minimum_distance_improvement_px": median_improvement,
        "within8_recall_gain": within8_gain,
        "nonnegative_video_fraction": nonnegative_fraction,
        "decision": decision,
    }


def evaluate(
    *,
    config_path: Path,
    cache_index_path: Path,
    partition: str,
    reference_path: Path | None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1B config")
    _validate_parent(config)
    if partition not in config["partitions"]:
        raise ValueError("partition is not preregistered")
    partition_config = config["partitions"][partition]
    expected_cache_path = Path(partition_config["cache_index"]).resolve()
    if cache_index_path != expected_cache_path:
        raise ValueError("cache-index path drift")
    actual_cache_index_sha256 = file_sha256(cache_index_path)
    expected_cache_index_sha256 = partition_config["cache_index_sha256"]
    if expected_cache_index_sha256 != "GENERATED_AFTER_PREREGISTRATION":
        if actual_cache_index_sha256 != expected_cache_index_sha256:
            raise ValueError("cache-index hash drift")
    index = json.loads(cache_index_path.read_text())
    expected_cache_config_sha256 = partition_config["cache_config_sha256"]
    if index.get("config_sha256") != expected_cache_config_sha256:
        raise ValueError("cache-build config hash drift")
    expected_sources = _expected_sources(config, partition)
    if index.get("completed_source_indices") != expected_sources:
        raise ValueError("cache source membership drift")
    if int(index.get("videos", -1)) != len(expected_sources):
        raise ValueError("cache video count drift")
    if index.get("manifest_sha256") != config["manifest_sha256"]:
        raise ValueError("cache manifest hash drift")

    selector_config = _selector_config(config)
    method_values: dict[str, list[dict[str, torch.Tensor]]] = {
        name: []
        for name in ("primary", "static_m1", "cycle_only", "identity_only")
    }
    selected_hashes: dict[str, list[str]] = {
        name: [] for name in method_values
    }
    score_hashes: list[str] = []
    video_records = []
    videos_with_failures = 0
    for row in index["rows"]:
        source_index = int(row["source_index"])
        sidecar = Path(row["sidecar"])
        if file_sha256(sidecar) != row["sidecar_sha256"]:
            raise ValueError(f"sidecar hash drift at {source_index}")
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        if int(payload["source_index"]) != source_index:
            raise ValueError("sidecar source drift")
        if payload["partition"] != partition:
            raise ValueError("sidecar partition drift")
        tensors = payload["tensors"]
        rows = int(tensors["point_indices"].numel())
        selections = _method_selections(tensors, selector_config)
        primary_score = select_query_closure_identity_candidates(
            temporal_features=tensors["temporal_features"].float(),
            static_features=tensors["static_features"].float(),
            query_frames=tensors["query_frames"].long(),
            valid_mask=tensors["candidate_valid_mask"].bool(),
            config=selector_config,
        )["score"]
        score_hashes.append(tensor_sha256(primary_score.contiguous()))
        local_metrics = {}
        for name, selected in selections.items():
            selected_hashes[name].append(tensor_sha256(selected.contiguous()))
            metrics = _method_row_metrics(tensors, selected)
            method_values[name].append(metrics)
            local_metrics[name] = metrics
        if rows:
            videos_with_failures += 1
            primary_hit = local_metrics["primary"]["supported_12px"].float()
            static_hit = local_metrics["static_m1"]["supported_12px"].float()
            primary_min = local_metrics["primary"]["minimum_distance_px"]
            static_min = local_metrics["static_m1"]["minimum_distance_px"]
            video_records.append(
                {
                    "source_index": source_index,
                    "video_name": payload["video_name"],
                    "rows": rows,
                    "primary_recall_12px": float(primary_hit.mean()),
                    "static_recall_12px": float(static_hit.mean()),
                    "support_gain_12px": float(
                        (primary_hit - static_hit).mean()
                    ),
                    "primary_median_minimum_distance_px": float(
                        primary_min.median()
                    ),
                    "static_median_minimum_distance_px": float(
                        static_min.median()
                    ),
                }
            )

    methods = {
        name: _aggregate_method(values) for name, values in method_values.items()
    }
    oracle_supported = int(
        index[
            "training_only_descriptive_support"
            if partition == "gradient_train"
            else "descriptive_support"
        ]["top128_oracle_supported_12px_rows"]
    )
    for value in methods.values():
        value["oracle_conditional_recall_12px"] = (
            0.0
            if oracle_supported == 0
            else value["supported_12px_rows"] / oracle_supported
        )
    bootstrap_config = config["bootstrap"]
    gains = np.array(
        [record["support_gain_12px"] for record in video_records],
        dtype=np.float64,
    )
    video_ci = _bootstrap_video_gain(
        gains,
        samples=int(bootstrap_config["samples"]),
        seed=int(bootstrap_config["seed"]),
    )
    deterministic = {
        "selected_indices_digest": {
            name: canonical_json_sha256(values)
            for name, values in selected_hashes.items()
        },
        "primary_score_digest": canonical_json_sha256(score_hashes),
        "method_metrics_digest": canonical_json_sha256(methods),
        "video_records_digest": canonical_json_sha256(video_records),
    }
    exact_replay = False
    replay_comparison: dict[str, Any] | None = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        reference_deterministic = reference.get("deterministic", {})
        replay_comparison = {
            key: deterministic[key] == reference_deterministic.get(key)
            for key in deterministic
        }
        exact_replay = all(replay_comparison.values())

    gate = _gate_checks(
        config,
        partition,
        methods["primary"],
        methods["static_m1"],
        video_records,
        video_ci,
        exact_replay=(partition == "gradient_train" or exact_replay),
        reference_provided=(reference_path is not None),
    )
    read_state = _read_policy(config, partition)
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "partition": partition,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "cache_index": str(cache_index_path),
        "cache_index_sha256": file_sha256(cache_index_path),
        "source_indices": [expected_sources[0], expected_sources[-1]],
        "videos": len(expected_sources),
        "videos_with_failures": videos_with_failures,
        "failure_rows": methods["primary"]["rows"],
        "selector": {
            "schema_version": "routeD_query_closure_identity_selector_gate3c1b_v0",
            "cycle_weight": selector_config.cycle_weight,
            "query_frame_identity_weight": selector_config.query_frame_identity_weight,
            "mean_identity_weight": selector_config.mean_identity_weight,
            "minimum_identity_weight": selector_config.minimum_identity_weight,
            "mandatory_native": True,
            "retained_nonnative": selector_config.retained_nonnative,
            "trainable_parameters": 0,
        },
        "methods": methods,
        "paired_video_support_gain_12px": video_ci,
        "video_records": video_records,
        "deterministic": deterministic,
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": replay_comparison,
        "exact_replay": exact_replay,
        "gate": gate,
        "read_state": read_state,
        "claim_boundary": config["claim_boundary"],
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-index", required=True)
    parser.add_argument(
        "--partition",
        required=True,
        choices=(
            "gradient_train",
            "checkpoint_selection",
            "fit_only_internal_audit",
        ),
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(
        config_path=Path(args.config).resolve(),
        cache_index_path=Path(args.cache_index).resolve(),
        partition=args.partition,
        reference_path=(
            None if args.reference is None else Path(args.reference).resolve()
        ),
    )
    _atomic_json_save(result, Path(args.output).resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
