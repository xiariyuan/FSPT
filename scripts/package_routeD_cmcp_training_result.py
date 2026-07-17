#!/usr/bin/env python3
"""Package and verify the formal CMCP proposal-only training result."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    CMCP_FEATURE_INDEX_SCHEMA_VERSION,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _threshold_gains(final: dict[str, Any], key: str) -> dict[str, float]:
    metrics = final[key]
    native = final["native_metrics"]
    return {
        threshold: 100.0 * (float(metrics[threshold]) - float(native[threshold]))
        for threshold in ("<1px", "<2px", "<4px", "<8px", "<16px")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(REPO_ROOT / "outputs/routeD_cmcp_training_20260717/seed17/metrics.json"))
    ap.add_argument("--replay", default=str(REPO_ROOT / "outputs/routeD_cmcp_training_20260717/seed17_replay/metrics.json"))
    ap.add_argument("--fit-index", default=str(REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"))
    ap.add_argument("--validation-index", default=str(REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/model_validation/cache_index.json"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/routeD_cmcp_cotracker3_stage0.yaml"))
    ap.add_argument("--interface-primary", default=str(REPO_ROOT / "outputs/routeD_cmcp_20260717/interface_smoke_v2.json"))
    ap.add_argument("--interface-replay", default=str(REPO_ROOT / "outputs/routeD_cmcp_20260717/interface_smoke_v2_replay.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "docs/generated/ROUTED_CMCP_PROPOSAL_TRAINING_SUMMARY_2026-07-17.json"))
    args = ap.parse_args()

    paths = {name: Path(value).resolve() for name, value in vars(args).items() if name != "output"}
    primary = _read(paths["primary"])
    replay = _read(paths["replay"])
    fit = _read(paths["fit_index"])
    validation = _read(paths["validation_index"])
    interface_primary = _read(paths["interface_primary"])
    interface_replay = _read(paths["interface_replay"])

    exact_fields = (
        "model_state_sha256",
        "best_epoch",
        "config_sha256",
        "protocol_sha256",
        "fit_cache_index_sha256",
        "validation_cache_index_sha256",
    )
    replay_checks = {key: primary[key] == replay[key] for key in exact_fields}
    replay_checks.update({
        "model_config": primary["model_config"] == replay["model_config"],
        "loss_config": primary["loss_config"] == replay["loss_config"],
        "optimizer": primary["optimizer"] == replay["optimizer"],
        "initialization_validation": primary["initialization_validation"] == replay["initialization_validation"],
        "history": primary["history"] == replay["history"],
        "final_validation": primary["final_validation"] == replay["final_validation"],
        "gate": primary["gate"] == replay["gate"],
        "external_data_read": primary["external_data_read"] == replay["external_data_read"],
    })
    if not all(replay_checks.values()):
        raise RuntimeError(f"CMCP seed replay mismatch: {replay_checks}")

    for name, index, expected in (("fit", fit, 48), ("model_validation", validation, 16)):
        if index.get("schema_version") != CMCP_FEATURE_INDEX_SCHEMA_VERSION:
            raise RuntimeError(f"unexpected feature index schema for {name}")
        if not index.get("complete") or int(index.get("completed_count", -1)) != expected:
            raise RuntimeError(f"incomplete CMCP feature cache: {name}")
        if index.get("expected_source_indices") != index.get("completed_source_indices"):
            raise RuntimeError(f"CMCP feature cache membership mismatch: {name}")
        if any(row["integrity"]["ground_truth_used_for_feature_map"] for row in index["videos"]):
            raise RuntimeError(f"GT contamination in feature cache: {name}")

    if primary["smoke_limits"] != {"max_train_videos": 0, "max_validation_videos": 0}:
        raise RuntimeError("primary CMCP result is not full-scale")
    if replay["smoke_limits"] != {"max_train_videos": 0, "max_validation_videos": 0}:
        raise RuntimeError("replay CMCP result is not full-scale")
    if any(bool(value) for value in primary["external_data_read"].values()):
        raise RuntimeError("locked data was read")
    if int(primary["model_config"]["hidden_channels"]) != 64:
        raise RuntimeError("unexpected CMCP hidden width")
    if int(primary["model_config"]["proposal_topk"]) != 5:
        raise RuntimeError("unexpected proposal top-K")
    if interface_primary["trainable_parameters"] != 263747:
        raise RuntimeError("corrected nine-channel CMCP interface parameter count mismatch")
    interface_checks = {
        "independent_tensor_hashes": interface_primary["tensor_hashes"] == interface_replay["tensor_hashes"],
        "native_state_hashes": interface_primary["native_state_hashes"] == interface_replay["native_state_hashes"],
        "first_chunk_replay": interface_primary["deterministic_first_chunk_replay"] == interface_replay["deterministic_first_chunk_replay"],
        "zero_step_native": bool(interface_primary["zero_step_selected_native_all_active"]),
        "candidate_zero_native": bool(interface_primary["candidate_zero_native_parity"]),
    }
    if not all(interface_checks.values()):
        raise RuntimeError(f"corrected CMCP interface mismatch: {interface_checks}")

    final = primary["final_validation"]
    selected_values = [float(row["selected_AJ_gain_points"]) for row in final["per_video"]]
    oracle_values = [float(row["oracle_AJ_gain_points"]) for row in final["per_video"]]
    summary = {
        "schema_version": "routeD_cmcp_proposal_training_summary_v1",
        "date": "2026-07-17",
        "status": "completed_partial_mechanism_success_gate_failure",
        "formal_decision": primary["gate"]["decision"],
        "research_interpretation": "Dense proposal generation succeeds strongly, but native-vs-proposal safety selection fails.",
        "claim_boundary": "Kubric fit/model-validation proposal-only result; not calibration, final-holdout, DAVIS, or Kinetics evidence.",
        "corrected_interface": {
            "recurrent_input_channels": 9,
            "includes_all_three_pairwise_correlation_differences": True,
            "trainable_parameters": interface_primary["trainable_parameters"],
            "config_sha256": primary["config_sha256"],
            "interface_checks": interface_checks,
            "supersedes_parameter_count_262019": True,
        },
        "cache_integrity": {
            "fit": {
                "videos": fit["completed_count"],
                "index_sha256": file_sha256(paths["fit_index"]),
                "payload_sha256": fit["cache_index_payload_sha256"],
            },
            "model_validation": {
                "videos": validation["completed_count"],
                "index_sha256": file_sha256(paths["validation_index"]),
                "payload_sha256": validation["cache_index_payload_sha256"],
            },
        },
        "training": {
            "seed": primary["seed"],
            "best_epoch": primary["best_epoch"],
            "epochs_executed": len(primary["history"]),
            "model_state_sha256": primary["model_state_sha256"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "metrics_sha256": file_sha256(paths["primary"]),
            "exact_seed_replay": all(replay_checks.values()),
            "replay_model_state_sha256": replay["model_state_sha256"],
            "replay_checkpoint_sha256": replay["checkpoint_sha256"],
            "replay_metrics_sha256": file_sha256(paths["replay"]),
            "zero_step_checkpoint_in_model_selection": True,
            "optimizer": primary["optimizer"],
        },
        "best_model_validation": {
            "native_metrics": final["native_metrics"],
            "selected_metrics": final["selected_metrics"],
            "oracle_metrics": final["oracle_metrics"],
            "selected_gain_points": final["selected_gain_points"],
            "oracle_gain_points": final["oracle_gain_points"],
            "selected_threshold_gain_points": _threshold_gains(final, "selected_metrics"),
            "oracle_threshold_gain_points": _threshold_gains(final, "oracle_metrics"),
            "paired_video_selected_AJ_gain_CI": final["paired_video_selected_AJ_gain_CI"],
            "paired_video_oracle_AJ_gain_CI": final["paired_video_oracle_AJ_gain_CI"],
            "paired_video_selected_delta_gain_CI": final["paired_video_selected_delta_gain_CI"],
            "severe_16px_rate": final["severe_16px_rate"],
            "behavior": final["behavior"],
            "selected_per_video": {
                "min": min(selected_values),
                "median": statistics.median(selected_values),
                "mean": statistics.mean(selected_values),
                "max": max(selected_values),
                "positive_videos": sum(value > 0.0 for value in selected_values),
                "negative_videos": sum(value < 0.0 for value in selected_values),
                "videos": len(selected_values),
            },
            "oracle_per_video": {
                "min": min(oracle_values),
                "median": statistics.median(oracle_values),
                "mean": statistics.mean(oracle_values),
                "max": max(oracle_values),
                "positive_videos": sum(value > 0.0 for value in oracle_values),
                "videos": len(oracle_values),
            },
            "per_video": final["per_video"],
        },
        "gate": primary["gate"],
        "diagnosis": {
            "candidate_generation_gate_passes": bool(primary["gate"]["candidate_oracle_AJ_gain_at_least_3"]),
            "direct_AJ_magnitude_gate_passes": bool(primary["gate"]["direct_top1_AJ_gain_at_least_0_5"]),
            "paired_consistency_gate_passes": bool(primary["gate"]["paired_top1_AJ_CI_lower_positive"]),
            "safety_gate_passes": bool(primary["gate"]["harmful_non_native_rate_at_most_0_01"]),
            "candidate_quality_is_current_bottleneck": False,
            "native_vs_peak_local_comparison_is_current_bottleneck": True,
            "global_mean_native_head_is_structurally_insufficient": True,
            "do_not_tune": ["NMS radius", "EMA alpha", "proposal top-K", "score threshold on model-validation"],
        },
        "external_data_read": primary["external_data_read"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "sha256": file_sha256(output),
        "model_state_sha256": primary["model_state_sha256"],
        "selected_AJ_gain_points": final["selected_gain_points"]["AJ"],
        "oracle_AJ_gain_points": final["oracle_gain_points"]["AJ"],
        "harmful_non_native_rate": final["behavior"]["harmful_non_native_rate"],
        "formal_decision": primary["gate"]["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
