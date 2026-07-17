#!/usr/bin/env python3
"""Package and verify the formal P0i LMRA joint-training result."""
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
    return {
        threshold: 100.0 * (
            float(final[key][threshold]) - float(final["native_metrics"][threshold])
        )
        for threshold in ("<1px", "<2px", "<4px", "<8px", "<16px")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        default=str(REPO_ROOT / "outputs/routeD_cmcp_lmra_training_20260717/seed17/metrics.json"),
    )
    parser.add_argument(
        "--replay",
        default=str(REPO_ROOT / "outputs/routeD_cmcp_lmra_training_20260717/seed17_replay/metrics.json"),
    )
    parser.add_argument(
        "--fit-index",
        default=str(REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"),
    )
    parser.add_argument(
        "--validation-index",
        default=str(REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/model_validation/cache_index.json"),
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs/routeD_cmcp_lmra_v0.yaml")
    )
    parser.add_argument(
        "--p0h-summary",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_CMCP_PAIRWISE_SAFETY_TRAINING_SUMMARY_2026-07-17.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "docs/generated/ROUTED_CMCP_LMRA_TRAINING_SUMMARY_2026-07-17.json"
        ),
    )
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in vars(args).items()
        if key != "output"
    }
    primary = _read(paths["primary"])
    replay = _read(paths["replay"])
    fit = _read(paths["fit_index"])
    validation = _read(paths["validation_index"])
    p0h = _read(paths["p0h_summary"])

    fields = (
        "schema_version",
        "seed",
        "best_epoch",
        "adapter_config",
        "cmcp_config",
        "comparator_config",
        "cmcp_loss_config",
        "comparator_loss_config",
        "joint_loss_config",
        "combined_model_state_sha256",
        "adapter_state_sha256",
        "cmcp_state_sha256",
        "comparator_state_sha256",
        "normalization",
        "optimizer",
        "config_sha256",
        "fit_feature_index_sha256",
        "validation_feature_index_sha256",
        "initial_cmcp_checkpoint_sha256",
        "initial_cmcp_model_state_sha256",
        "initial_comparator_checkpoint_sha256",
        "initial_comparator_model_state_sha256",
        "zero_step_formal_p0h_exact",
        "initialization_validation",
        "history",
        "final_validation",
        "gate",
        "external_data_read",
        "smoke_limits",
    )
    replay_checks = {key: primary[key] == replay[key] for key in fields}
    if not all(replay_checks.values()):
        raise RuntimeError(f"P0i replay mismatch: {replay_checks}")
    if primary["checkpoint_sha256"] != replay["checkpoint_sha256"]:
        raise RuntimeError("P0i checkpoint files are not byte-identical")

    for name, index, expected in (
        ("fit", fit, 48),
        ("model_validation", validation, 16),
    ):
        if (
            index.get("schema_version") != CMCP_FEATURE_INDEX_SCHEMA_VERSION
            or not index.get("complete")
            or int(index.get("completed_count", -1)) != expected
        ):
            raise RuntimeError(f"incomplete feature cache: {name}")
    if primary["smoke_limits"] != {
        "max_train_videos": 0,
        "max_validation_videos": 0,
    }:
        raise RuntimeError("primary P0i result is not full scale")
    if replay["smoke_limits"] != primary["smoke_limits"]:
        raise RuntimeError("replay P0i result is not full scale")
    if any(primary["external_data_read"].values()):
        raise RuntimeError("locked data was read")
    if not primary["zero_step_formal_p0h_exact"]:
        raise RuntimeError("zero-step formal P0h equality failed")
    if not primary["gate"]["pass"]:
        raise RuntimeError("formal P0i gate did not pass")

    initial = primary["initialization_validation"]
    p0h_final = p0h["best_model_validation"]
    for key in (
        "native_metrics",
        "selected_metrics",
        "oracle_metrics",
        "selected_gain_points",
        "oracle_gain_points",
        "severe_16px_rate",
        "behavior",
    ):
        if initial[key] != p0h_final[key]:
            raise RuntimeError(f"zero-step P0h metric mismatch: {key}")

    final = primary["final_validation"]
    selected = [float(row["selected_AJ_gain_points"]) for row in final["per_video"]]
    history = [
        {
            "epoch": int(row["epoch"]),
            "checkpoint_improved": bool(row["checkpoint_improved"]),
            "safety_feasible": bool(row["safety_feasible"]),
            "AJ_gain_points": float(row["validation"]["selected_gain_points"]["AJ"]),
            "oracle_AJ_gain_points": float(row["validation"]["oracle_gain_points"]["AJ"]),
            "harmful_non_native_rate": float(
                row["validation"]["behavior"]["harmful_non_native_rate"]
            ),
            "beneficial_candidate_recall": float(
                row["validation"]["behavior"]["beneficial_candidate_recall"]
            ),
            "feature_distortion": float(row["train"]["feature_distortion"]),
        }
        for row in primary["history"]
    ]
    summary = {
        "schema_version": "routeD_cmcp_lmra_training_summary_v0",
        "date": "2026-07-17",
        "status": "completed_pass",
        "formal_decision": primary["gate"]["decision"],
        "research_interpretation": (
            "A zero-initialized late metric adapter plus joint CMCP/comparator training "
            "crosses the preregistered magnitude, consistency, tail, and safety gates "
            "while the native trajectory remains frozen."
        ),
        "claim_boundary": (
            "Kubric fit/model-validation evidence only. This authorizes controlled "
            "strong-backbone MUSR ablation, not calibration, final holdout, DAVIS, "
            "or official Kinetics evaluation."
        ),
        "initial_contract": {
            "zero_step_formal_p0h_exact": True,
            "p0h_selected_gain_points": initial["selected_gain_points"],
            "p0h_behavior": initial["behavior"],
            "native_candidate_parity_all": initial["native_candidate_parity_all"],
            "initial_cmcp_model_state_sha256": primary[
                "initial_cmcp_model_state_sha256"
            ],
            "initial_comparator_model_state_sha256": primary[
                "initial_comparator_model_state_sha256"
            ],
            "config_path": str(paths["config"]),
            "config_sha256": file_sha256(paths["config"]),
        },
        "training": {
            "seed": primary["seed"],
            "best_epoch": primary["best_epoch"],
            "epochs_executed": len(primary["history"]),
            "combined_model_state_sha256": primary["combined_model_state_sha256"],
            "adapter_state_sha256": primary["adapter_state_sha256"],
            "cmcp_state_sha256": primary["cmcp_state_sha256"],
            "comparator_state_sha256": primary["comparator_state_sha256"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "metrics_sha256": file_sha256(paths["primary"]),
            "replay_checkpoint_sha256": replay["checkpoint_sha256"],
            "replay_metrics_sha256": file_sha256(paths["replay"]),
            "exact_seed_replay": True,
            "replay_checks": replay_checks,
            "replay_log_used_as_evidence": False,
            "optimizer": primary["optimizer"],
            "history_summary": history,
        },
        "best_model_validation": {
            "native_metrics": final["native_metrics"],
            "selected_metrics": final["selected_metrics"],
            "oracle_metrics": final["oracle_metrics"],
            "selected_gain_points": final["selected_gain_points"],
            "oracle_gain_points": final["oracle_gain_points"],
            "selected_threshold_gain_points": _threshold_gains(final, "selected_metrics"),
            "oracle_threshold_gain_points": _threshold_gains(final, "oracle_metrics"),
            "paired_video_selected_AJ_gain_CI": final[
                "paired_video_selected_AJ_gain_CI"
            ],
            "paired_video_oracle_AJ_gain_CI": final[
                "paired_video_oracle_AJ_gain_CI"
            ],
            "paired_video_selected_delta_gain_CI": final[
                "paired_video_selected_delta_gain_CI"
            ],
            "severe_16px_rate": final["severe_16px_rate"],
            "behavior": final["behavior"],
            "candidate_coordinate_combined_sha256": final[
                "candidate_coordinate_combined_sha256"
            ],
            "native_candidate_parity_all": final["native_candidate_parity_all"],
            "per_video_summary": {
                "min": min(selected),
                "median": statistics.median(selected),
                "mean": statistics.mean(selected),
                "max": max(selected),
                "positive_videos": sum(value > 0 for value in selected),
                "negative_videos": sum(value < 0 for value in selected),
                "videos": len(selected),
            },
            "per_video": final["per_video"],
        },
        "gate": primary["gate"],
        "diagnosis": {
            "candidate_quality_is_bottleneck": False,
            "safe_recall_improved_over_p0h": (
                final["behavior"]["beneficial_candidate_recall"]
                > initial["behavior"]["beneficial_candidate_recall"]
            ),
            "p0h_AJ_gain_points": initial["selected_gain_points"]["AJ"],
            "p0i_AJ_gain_points": final["selected_gain_points"]["AJ"],
            "unsafe_peak_epoch": 3,
            "unsafe_peak_AJ_gain_points": history[3]["AJ_gain_points"],
            "unsafe_peak_harmful_non_native_rate": history[3][
                "harmful_non_native_rate"
            ],
            "next_route": (
                "preregistered strong-backbone MUSR component and bounded-state-write "
                "ablation on fit/model-validation only"
            ),
        },
        "external_data_read": primary["external_data_read"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "combined_model_state_sha256": primary[
                    "combined_model_state_sha256"
                ],
                "AJ_gain_points": final["selected_gain_points"]["AJ"],
                "AJ_CI": final["paired_video_selected_AJ_gain_CI"],
                "harmful_non_native_rate": final["behavior"][
                    "harmful_non_native_rate"
                ],
                "formal_decision": primary["gate"]["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
