"""Pure protocol and gate helpers for Route-D P0l final synthetic holdout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .routeD_kubric_cache import file_sha256

FINAL_HOLDOUT_CACHE_SCHEMA = "routeD_cmcp_final_holdout_video_cache_v0"
FINAL_HOLDOUT_INDEX_SCHEMA = "routeD_cmcp_final_holdout_cache_index_v0"
FINAL_HOLDOUT_RESULT_SCHEMA = "routeD_cmcp_final_holdout_result_v0"


def load_final_holdout_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_final_holdout_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def validate_final_holdout_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment", {}).get("stage") != "P0l":
        raise ValueError("expected P0l final-holdout config")
    partitions = config.get("partitions", {})
    if set(partitions) != {"implementation_smoke", "final_holdout"}:
        raise ValueError("unexpected P0l partitions")
    final_indices = [int(value) for value in partitions["final_holdout"]["indices"]]
    if final_indices != list(range(16, 32)):
        raise ValueError("final holdout membership must be validation 16--31")
    if int(partitions["final_holdout"]["expected_count"]) != 16:
        raise ValueError("final holdout count must be 16")
    if config["calibration_policy"]["calibration_source_read"] is not False:
        raise ValueError("P0l calibration must remain unread")
    if config["calibration_policy"]["posthoc_calibration"] != "none":
        raise ValueError("P0l forbids posthoc calibration")
    if config["frozen_model"]["state_writeback"] != "disabled":
        raise ValueError("P0l is output-only")
    if int(config["frozen_model"]["new_trainable_parameters"]) != 0:
        raise ValueError("P0l cannot add trainable parameters")
    if not config["cache_contract"]["cache_before_metrics"]:
        raise ValueError("P0l requires cache-before-metrics")
    if not config["cache_contract"]["partial_metrics_forbidden"]:
        raise ValueError("P0l forbids partial metrics")
    gates = config["formal_gates"]
    if float(gates["direct_AJ_gain_points_min"]) != 0.5:
        raise ValueError("P0l direct AJ gate drift")
    if int(gates["positive_video_count_min"]) != 12:
        raise ValueError("P0l positive-video gate drift")


def partition_indices(config: Mapping[str, Any], partition: str) -> tuple[int, ...]:
    spec = config["partitions"][partition]
    return tuple(int(value) for value in spec["indices"])


def verify_frozen_files(config: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    model_path = Path(str(config["frozen_model"]["checkpoint"]))
    if not model_path.is_absolute():
        model_path = (Path(config["_config_path"]).parents[1] / model_path).resolve()
    backbone_path = Path(str(config["backbone"]["checkpoint"])).resolve()
    for name, path, expected in (
        ("model", model_path, config["frozen_model"]["checkpoint_sha256"]),
        ("backbone", backbone_path, config["backbone"]["checkpoint_sha256"]),
    ):
        if not path.exists() or file_sha256(path) != str(expected):
            raise ValueError(f"{name} file/hash mismatch")
        verified[name] = {"path": str(path), "sha256": str(expected)}
    for alias, row in config["source_manifests"].items():
        path = Path(str(row["path"])).resolve()
        if not path.exists() or file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"manifest file/hash mismatch: {alias}")
        payload = json.loads(path.read_text())
        if int(payload.get("num_samples", -1)) != int(row["num_samples"]):
            raise ValueError(f"manifest count mismatch: {alias}")
        verified[alias] = {
            "path": str(path),
            "sha256": str(row["sha256"]),
            "num_samples": int(row["num_samples"]),
        }
    return verified


def evaluate_final_holdout_gate(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    exact_replay: bool,
) -> dict[str, Any]:
    gates = config["formal_gates"]
    per_video = metrics["per_video"]
    checks = {
        "complete_16_video_partition": int(metrics["videos"]) == 16 and len(per_video) == 16,
        "native_candidate_parity_all": bool(metrics["native_candidate_parity_all"]),
        "candidate_oracle_AJ_gain_at_least_3": float(metrics["oracle_gain_points"]["AJ"])
        >= float(gates["candidate_oracle_AJ_gain_points_min"]),
        "direct_AJ_gain_at_least_0_5": float(metrics["selected_gain_points"]["AJ"])
        >= float(gates["direct_AJ_gain_points_min"]),
        "paired_direct_AJ_CI_lower_positive": float(
            metrics["paired_video_selected_AJ_gain_CI"]["lower"]
        )
        > float(gates["paired_video_direct_AJ_CI_lower_gt"]),
        "delta_gain_positive": float(metrics["selected_gain_points"]["delta_average"])
        > float(gates["direct_delta_average_gain_gt"]),
        "severe_16px_not_worse": float(metrics["severe_16px_rate"]["selected_delta"])
        <= 0.0,
        "harmful_non_native_rate_at_most_0_01": float(
            metrics["behavior"]["harmful_non_native_rate"]
        )
        <= float(gates["harmful_non_native_rate_max"]),
        "positive_videos_at_least_12": sum(
            float(row["selected_AJ_gain_points"]) > 0.0 for row in per_video
        )
        >= int(gates["positive_video_count_min"]),
        "exact_primary_replay": bool(exact_replay),
    }
    checks["pass"] = all(checks.values())
    checks["decision"] = config["formal_decisions"]["pass" if checks["pass"] else "fail"]
    return checks
