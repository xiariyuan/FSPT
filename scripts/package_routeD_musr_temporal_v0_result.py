#!/usr/bin/env python3
"""Package and verify the formal MUSR temporal-v0 Stage-A result."""
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
        key: 100.0
        * (float(final["selected_metrics"][key]) - float(final["native_metrics"][key]))
        for key in ("<1px", "<2px", "<4px", "<8px", "<16px")
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--primary",
        default=str(
            REPO_ROOT
            / "outputs/routeD_musr_temporal_v0_20260717/seed17/metrics.json"
        ),
    )
    ap.add_argument(
        "--replay",
        default=str(
            REPO_ROOT
            / "outputs/routeD_musr_temporal_v0_20260717/seed17_replay/metrics.json"
        ),
    )
    ap.add_argument(
        "--fit-index",
        default=str(
            REPO_ROOT
            / "outputs/routeD_musr_raw_v1_cache_20260717/fit/cache_index.json"
        ),
    )
    ap.add_argument(
        "--validation-index",
        default=str(
            REPO_ROOT
            / "outputs/routeD_musr_raw_v1_cache_20260717/model_validation/cache_index.json"
        ),
    )
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/routeD_musr_cotracker3_temporal_v0.yaml"),
    )
    ap.add_argument(
        "--structured-summary",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_MUSR_SELECTOR_STAGEA_SUMMARY_2026-07-17.json"
        ),
    )
    ap.add_argument(
        "--raw-summary",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_MUSR_RAW_V1_STAGEA_SUMMARY_2026-07-17.json"
        ),
    )
    ap.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_MUSR_TEMPORAL_V0_STAGEA_SUMMARY_2026-07-17.json"
        ),
    )
    args = ap.parse_args()

    paths = {
        name: Path(value).resolve()
        for name, value in vars(args).items()
        if name != "output"
    }
    primary = _read(paths["primary"])
    replay = _read(paths["replay"])
    fit = _read(paths["fit_index"])
    validation = _read(paths["validation_index"])
    structured = _read(paths["structured_summary"])
    raw = _read(paths["raw_summary"])

    exact_fields = (
        "model_state_sha256",
        "best_epoch",
        "fit_cache_index_sha256",
        "validation_cache_index_sha256",
        "protocol_sha256",
        "config_sha256",
        "representation_schema_version",
    )
    replay_checks = {key: primary[key] == replay[key] for key in exact_fields}
    replay_checks.update(
        {
            "history": primary["history"] == replay["history"],
            "initialization_validation": primary["initialization_validation"]
            == replay["initialization_validation"],
            "final_validation": primary["final_validation"]
            == replay["final_validation"],
            "gate": primary["gate"] == replay["gate"],
            "network_config": primary["network_config"]
            == replay["network_config"],
            "temporal_config": primary["temporal_config"]
            == replay["temporal_config"],
            "normalization": primary["normalization"] == replay["normalization"],
        }
    )
    if not all(replay_checks.values()):
        raise RuntimeError(f"temporal-v0 seed replay mismatch: {replay_checks}")

    for name, index, expected in (
        ("fit", fit, 48),
        ("model_validation", validation, 16),
    ):
        if not index["complete"] or int(index["completed_count"]) != expected:
            raise RuntimeError(f"incomplete raw-v1 {name} cache")
        if index["representation_schema_version"] != RAW_V1_SCHEMA_VERSION:
            raise RuntimeError(f"representation mismatch in {name}")
        if int(index["candidate_feature_dim"]) != RAW_V1_CANDIDATE_FEATURE_DIM:
            raise RuntimeError(f"candidate dimension mismatch in {name}")
        if int(index["state_feature_dim"]) != RAW_V1_STATE_FEATURE_DIM:
            raise RuntimeError(f"state dimension mismatch in {name}")
        if not all(
            bool(row["candidate_coordinate_hash_unchanged"])
            for row in index["videos"]
        ):
            raise RuntimeError(f"candidate coordinate drift in {name}")

    external = primary["external_data_read"]
    if any(bool(value) for value in external.values()):
        raise RuntimeError(f"locked-data contamination: {external}")
    if primary["representation_schema_version"] != RAW_V1_SCHEMA_VERSION:
        raise RuntimeError("temporal result did not use frozen raw-v1 representation")

    final = primary["final_validation"]
    per_video = [float(row["AJ_gain_points"]) for row in final["per_video"]]
    temporal_recall = float(final["behavior"]["beneficial_global_recall"])
    raw_recall = float(
        raw["temporal_diagnosis"]["selector_beneficial_recall"]
    )
    summary = {
        "schema_version": "routeD_musr_temporal_v0_stageA_summary_v1",
        "date": "2026-07-17",
        "status": "completed_negative_gate",
        "decision": "STOP_TEMPORAL_SELECTOR_AND_START_MULTI_MEMORY_PROPOSAL_GENERATION",
        "claim_boundary": (
            "Temporal-v0 is a selector-only Kubric model-validation result. "
            "It is not a final-holdout or external benchmark result."
        ),
        "representation": {
            "candidate_feature_dim": RAW_V1_CANDIDATE_FEATURE_DIM,
            "state_feature_dim": RAW_V1_STATE_FEATURE_DIM,
            "raw_schema_version": RAW_V1_SCHEMA_VERSION,
            "window_frames": primary["temporal_config"]["window_frames"],
            "temporal_layers": primary["temporal_config"]["temporal_layers"],
            "past_candidate_pooling": "rank_invariant_attention",
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
            "replay_checkpoint_sha256": replay["checkpoint_sha256"],
            "replay_metrics_file_sha256": file_sha256(paths["replay"]),
            "checkpoint_file_hash_expected_to_differ_by_output_path": True,
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
                "median": _median(per_video),
                "mean": sum(per_video) / len(per_video),
                "max": max(per_video),
                "positive_videos": sum(value > 0.0 for value in per_video),
                "videos": len(per_video),
            },
            "per_video": final["per_video"],
        },
        "comparison": {
            "coordinate_oracle_AJ_gain_points": 6.8463205369881095,
            "structured_64_AJ_gain_points": structured["best_model_validation"][
                "gain_points"
            ]["AJ"],
            "raw_v1_AJ_gain_points": raw["best_model_validation"]["gain_points"][
                "AJ"
            ],
            "temporal_v0_AJ_gain_points": final["gain_points"]["AJ"],
            "temporal_minus_raw_points": final["gain_points"]["AJ"]
            - raw["best_model_validation"]["gain_points"]["AJ"],
            "temporal_minus_structured_points": final["gain_points"]["AJ"]
            - structured["best_model_validation"]["gain_points"]["AJ"],
            "raw_single_frame_beneficial_recall": raw_recall,
            "temporal_v0_beneficial_recall": temporal_recall,
        },
        "gate": primary["gate"],
        "diagnosis": {
            "oracle_headroom_exists": True,
            "single_frame_representation_sufficient": False,
            "fixed_six_frame_temporal_selector_sufficient": False,
            "temporal_selector_is_safe_but_under_recalls_beneficial_events": True,
            "next_route": (
                "train a causal multi-memory dense correlation proposal generator "
                "on fit before any selector or state-write training"
            ),
        },
        "replay_checks": replay_checks,
        "external_data_read": external,
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
                "beneficial_global_recall": temporal_recall,
                "decision": summary["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
