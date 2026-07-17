#!/usr/bin/env python3
"""Validate and package the reproducible MUSR Stage-A selector result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_musr_selector_20260717/seed17/metrics.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_musr_selector_20260717/seed17_replay/metrics.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_MUSR_SELECTOR_STAGEA_SUMMARY_2026-07-17.json"
CACHE_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"


def load_index(partition: str) -> tuple[dict, Path]:
    path = CACHE_ROOT / partition / "cache_index.json"
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise ValueError(f"incomplete cache partition: {partition}")
    return payload, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(DEFAULT_PRIMARY))
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    for key in (
        "model_state_sha256",
        "best_epoch",
        "protocol_sha256",
        "fit_cache_index_sha256",
        "validation_cache_index_sha256",
        "final_validation",
        "gate",
    ):
        if primary[key] != replay[key]:
            raise ValueError(f"seed-17 replay mismatch: {key}")
    if primary["gate"]["pass"]:
        raise ValueError("Stage-A unexpectedly passed")
    if primary["gate"]["decision"] != "STOP_STAGE_B_AND_REDESIGN_CANDIDATE_REPRESENTATION":
        raise ValueError("unexpected Stage-A decision")
    if any(primary["external_data_read"].values()):
        raise ValueError("Stage-A read a locked partition/domain")

    fit, fit_path = load_index("fit")
    validation, validation_path = load_index("model_validation")
    calibration, calibration_path = load_index("calibration")
    final = primary["final_validation"]
    values = [row["AJ_gain_points"] for row in final["per_video"]]
    summary = {
        "schema_version": "routeD_musr_selector_stageA_summary_v1",
        "date": "2026-07-17",
        "status": "FAIL_SELECTOR_GATE_STOP_STATE_WRITE",
        "decision": primary["gate"]["decision"],
        "claim_boundary": (
            "Frozen Kubric fit/model-validation selector feasibility only; "
            "calibration was exported but not read, and final holdout/DAVIS/Kinetics were not read."
        ),
        "training": {
            "stage": "A_utility_aligned_raw_candidate_selector",
            "seed": primary["seed"],
            "best_epoch": primary["best_epoch"],
            "epochs_executed": len(primary["history"]),
            "model_state_sha256": primary["model_state_sha256"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "metrics_file_sha256": file_sha256(primary_path),
            "exact_seed_replay": True,
            "replay_metrics_file_sha256": file_sha256(replay_path),
            "native_safe_initialization": True,
            "state_write_heads_trained": False,
            "selection_supervision": "weighted multi-threshold utility with native tie fallback",
        },
        "cache_partitions": {
            "fit": {
                "videos": fit["completed_count"],
                "oracle_gain_points": fit["aggregate_oracle_audit"]["gain_points"],
                "index_file_sha256": file_sha256(fit_path),
                "index_payload_sha256": fit["cache_index_payload_sha256"],
            },
            "model_validation": {
                "videos": validation["completed_count"],
                "oracle_gain_points": validation["aggregate_oracle_audit"]["gain_points"],
                "index_file_sha256": file_sha256(validation_path),
                "index_payload_sha256": validation["cache_index_payload_sha256"],
            },
            "calibration": {
                "videos": calibration["completed_count"],
                "oracle_gain_points": calibration["aggregate_oracle_audit"]["gain_points"],
                "index_file_sha256": file_sha256(calibration_path),
                "index_payload_sha256": calibration["cache_index_payload_sha256"],
                "read_by_stageA": False,
            },
        },
        "initialization": primary["initialization_validation"],
        "best_model_validation": {
            "native_metrics": final["native_metrics"],
            "selected_metrics": final["selected_metrics"],
            "gain_points": final["gain_points"],
            "paired_video_AJ_gain_CI": final["paired_video_AJ_gain_CI"],
            "paired_video_delta_gain_CI": final["paired_video_delta_gain_CI"],
            "severe_16px_rate": final["severe_16px_rate"],
            "behavior": final["behavior"],
            "per_video_AJ_gain_points": {
                "minimum": min(values),
                "median": sorted(values)[len(values) // 2 - 1 : len(values) // 2 + 1],
                "mean": sum(values) / len(values),
                "maximum": max(values),
                "positive_videos": sum(value > 0.0 for value in values),
                "total_videos": len(values),
            },
            "per_video": final["per_video"],
        },
        "gate": primary["gate"],
        "diagnosis": {
            "candidate_oracle_is_material": True,
            "selector_gain_is_material": False,
            "current_64d_handcrafted_representation_recovers_oracle": False,
            "state_write_training_allowed": False,
            "next_change": (
                "Keep coordinates/candidate generator fixed; replace binned candidate/state "
                "summaries with full CoTracker feature interactions and a local correlation patch."
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
                "model_state_sha256": primary["model_state_sha256"],
                "AJ_gain_points": final["gain_points"]["AJ"],
                "AJ_CI": final["paired_video_AJ_gain_CI"],
                "decision": primary["gate"]["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
