#!/usr/bin/env python3
"""Package and verify the formal MUSR raw-v1 Stage-A result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_raw_representation import (
    RAW_V1_CANDIDATE_FEATURE_DIM,
    RAW_V1_SCHEMA_VERSION,
    RAW_V1_STATE_FEATURE_DIM,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _threshold_gains(final: dict[str, Any]) -> dict[str, float]:
    return {
        key: 100.0 * (
            float(final["selected_metrics"][key]) - float(final["native_metrics"][key])
        )
        for key in ("<1px", "<2px", "<4px", "<8px", "<16px")
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_selector_20260717/seed17/metrics.json"))
    ap.add_argument("--replay", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_selector_20260717/seed17_replay/metrics.json"))
    ap.add_argument("--fit-index", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_cache_20260717/fit/cache_index.json"))
    ap.add_argument("--validation-index", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_cache_20260717/model_validation/cache_index.json"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/routeD_musr_cotracker3_raw_v1.yaml"))
    ap.add_argument("--feature-shift", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_selector_20260717/seed17/feature_shift.json"))
    ap.add_argument("--temporal-diagnosis", default=str(REPO_ROOT / "outputs/routeD_musr_raw_v1_selector_20260717/seed17/temporal_diagnosis.json"))
    ap.add_argument("--structured-summary", default=str(REPO_ROOT / "docs/generated/ROUTED_MUSR_SELECTOR_STAGEA_SUMMARY_2026-07-17.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "docs/generated/ROUTED_MUSR_RAW_V1_STAGEA_SUMMARY_2026-07-17.json"))
    args = ap.parse_args()

    paths = {name: Path(value).resolve() for name, value in vars(args).items() if name != "output"}
    primary = _read(paths["primary"])
    replay = _read(paths["replay"])
    fit = _read(paths["fit_index"])
    validation = _read(paths["validation_index"])
    shift = _read(paths["feature_shift"])
    temporal = _read(paths["temporal_diagnosis"])
    structured = _read(paths["structured_summary"])

    exact_fields = (
        "model_state_sha256",
        "best_epoch",
        "fit_cache_index_sha256",
        "validation_cache_index_sha256",
        "protocol_sha256",
    )
    replay_checks = {key: primary[key] == replay[key] for key in exact_fields}
    replay_checks.update(
        {
            "history": primary["history"] == replay["history"],
            "final_validation": primary["final_validation"] == replay["final_validation"],
            "gate": primary["gate"] == replay["gate"],
        }
    )
    if not all(replay_checks.values()):
        raise RuntimeError(f"raw-v1 seed replay mismatch: {replay_checks}")

    for name, index, expected in (
        ("fit", fit, 48),
        ("model_validation", validation, 16),
    ):
        if not index["complete"] or index["completed_count"] != expected:
            raise RuntimeError(f"incomplete raw-v1 {name} cache")
        if index["representation_schema_version"] != RAW_V1_SCHEMA_VERSION:
            raise RuntimeError(f"representation mismatch in {name}")
        if index["candidate_feature_dim"] != RAW_V1_CANDIDATE_FEATURE_DIM:
            raise RuntimeError(f"candidate dimension mismatch in {name}")
        if index["state_feature_dim"] != RAW_V1_STATE_FEATURE_DIM:
            raise RuntimeError(f"state dimension mismatch in {name}")
        if not all(row["candidate_coordinate_hash_unchanged"] for row in index["videos"]):
            raise RuntimeError(f"candidate coordinate drift in {name}")
        if any(row["integrity"]["ground_truth_used_for_representation"] for row in index["videos"]):
            raise RuntimeError(f"GT contamination in {name}")

    replay_anchors = [
        {"partition": name, "source_index": row["source_index"], "exact": row["deterministic_raw_replay"]["exact"]}
        for name, index in (("fit", fit), ("model_validation", validation))
        for row in index["videos"]
        if row["deterministic_raw_replay"] is not None
    ]
    if len(replay_anchors) != 4 or not all(row["exact"] for row in replay_anchors):
        raise RuntimeError("raw-v1 cache replay anchors are incomplete")

    final = primary["final_validation"]
    per_video = [float(row["AJ_gain_points"]) for row in final["per_video"]]
    summary = {
        "schema_version": "routeD_musr_raw_v1_stageA_summary_v1",
        "date": "2026-07-17",
        "status": "completed_negative_gate",
        "decision": "STOP_RAW_V1_AND_START_CAUSAL_TEMPORAL_REPRESENTATION",
        "claim_boundary": "Raw-v1 is a selector-only Kubric model-validation result; it is not a final-holdout or external benchmark result.",
        "representation": {
            "schema_version": RAW_V1_SCHEMA_VERSION,
            "candidate_feature_dim": RAW_V1_CANDIDATE_FEATURE_DIM,
            "state_feature_dim": RAW_V1_STATE_FEATURE_DIM,
            "candidate_coordinates_changed": False,
            "config_path": str(paths["config"]),
            "config_sha256": file_sha256(paths["config"]),
        },
        "cache_integrity": {
            "fit": {
                "videos": fit["completed_count"],
                "cache_index_sha256": file_sha256(paths["fit_index"]),
                "payload_sha256": fit["cache_index_payload_sha256"],
                "base_cache_index_sha256": fit["base_cache_index_sha256"],
            },
            "model_validation": {
                "videos": validation["completed_count"],
                "cache_index_sha256": file_sha256(paths["validation_index"]),
                "payload_sha256": validation["cache_index_payload_sha256"],
                "base_cache_index_sha256": validation["base_cache_index_sha256"],
            },
            "replay_anchors": replay_anchors,
            "all_candidate_coordinate_hashes_unchanged": True,
        },
        "training": {
            "stage": "selector_only_stage_A",
            "seed": primary["seed"],
            "best_epoch": primary["best_epoch"],
            "epochs_executed": len(primary["history"]),
            "model_state_sha256": primary["model_state_sha256"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "metrics_file_sha256": file_sha256(paths["primary"]),
            "exact_seed_replay": all(replay_checks.values()),
            "replay_model_state_sha256": replay["model_state_sha256"],
            "replay_metrics_file_sha256": file_sha256(paths["replay"]),
            "state_write_heads_trained": False,
        },
        "best_model_validation": {
            "native_metrics": final["native_metrics"],
            "selected_metrics": final["selected_metrics"],
            "gain_points": final["gain_points"],
            "threshold_gain_points": _threshold_gains(final),
            "paired_video_AJ_gain_CI": final["paired_video_AJ_gain_CI"],
            "paired_video_delta_gain_CI": final["paired_video_delta_gain_CI"],
            "severe_16px_rate": final["severe_16px_rate"],
            "behavior": final["behavior"],
            "per_video_AJ_gain_points": {
                "min": min(per_video),
                "median": float(sorted(per_video)[len(per_video) // 2 - 1 : len(per_video) // 2 + 1][0] + sorted(per_video)[len(per_video) // 2]) / 2.0,
                "mean": sum(per_video) / len(per_video),
                "max": max(per_video),
                "positive_videos": sum(value > 0.0 for value in per_video),
                "videos": len(per_video),
            },
            "per_video": final["per_video"],
        },
        "comparison_to_structured_64": {
            "structured_AJ_gain_points": structured["best_model_validation"]["gain_points"]["AJ"],
            "raw_v1_AJ_gain_points": final["gain_points"]["AJ"],
            "difference_points": final["gain_points"]["AJ"] - structured["best_model_validation"]["gain_points"]["AJ"],
            "structured_delta_gain_points": structured["best_model_validation"]["gain_points"]["delta_average"],
            "raw_v1_delta_gain_points": final["gain_points"]["delta_average"],
            "structured_oracle_match_rate": structured["best_model_validation"]["behavior"]["oracle_utility_match_rate"],
            "raw_v1_oracle_match_rate": final["behavior"]["oracle_utility_match_rate"],
        },
        "gate": primary["gate"],
        "feature_shift": shift,
        "temporal_diagnosis": temporal,
        "diagnosis": {
            "simple_fit_validation_covariate_shift_detected": False,
            "single_frame_raw_representation_recovers_oracle_headroom": False,
            "beneficial_events_are_temporally_persistent": True,
            "next_route": "causal rank-invariant temporal evidence accumulator over frozen candidate sets",
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
        "AJ_gain_points": final["gain_points"]["AJ"],
        "AJ_CI": final["paired_video_AJ_gain_CI"],
        "decision": summary["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
