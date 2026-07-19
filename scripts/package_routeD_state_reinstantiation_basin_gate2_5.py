#!/usr/bin/env python3
"""Package exact primary/replay Route-D state-reinstatement basin runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_state_reinstantiation_basin import (
    BASIN_SCHEMA_VERSION,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_state_reinstantiation_basin_gate2_5_v0.yaml"
DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_state_reinstantiation_basin_gate2_5_20260719/primary.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_state_reinstantiation_basin_gate2_5_20260719/replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_STATE_REINSTATEMENT_BASIN_GATE2_5_SUMMARY_2026-07-19.json"


def reports_exact(primary: Mapping[str, Any], replay: Mapping[str, Any]) -> bool:
    """The runner records no output path or timestamp, so full equality is required."""
    return primary == replay


def _validate_report(report: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if report.get("schema_version") != BASIN_SCHEMA_VERSION:
        raise ValueError("Gate 2.5 report schema mismatch")
    if report.get("status") != "completed":
        raise ValueError("Gate 2.5 report is incomplete")
    if report.get("decision") != config["formal_decisions"]["always"]:
        raise ValueError("Gate 2.5 runner decision drift")
    expected_sources = [int(value) for value in config["cache"]["expected_source_indices"]]
    observed_sources = [int(row["source_index"]) for row in report["videos"]]
    if observed_sources != expected_sources:
        raise ValueError("Gate 2.5 report source membership drift")
    if any(value is not False for value in report["locked_data"].values()):
        raise ValueError("Gate 2.5 report read a locked partition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--primary", default=str(DEFAULT_PRIMARY))
    parser.add_argument("--replay", default=str(DEFAULT_REPLAY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    output = Path(args.output).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != BASIN_SCHEMA_VERSION:
        raise ValueError("unexpected Gate 2.5 packaging config")
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    _validate_report(primary, config)
    _validate_report(replay, config)

    report_exact = reports_exact(primary, replay)
    primary_integrity = primary["center_reextraction_integrity"]
    replay_integrity = replay["center_reextraction_integrity"]
    gates = config["integrity_gates"]
    checks = {
        "primary_replay_report_exact": report_exact,
        "primary_center_track_feature_cosine": float(
            primary_integrity["minimum_track_feature_cosine"]
        )
        >= float(gates["center_reextraction_track_feature_cosine_min"]),
        "primary_center_track_support_cosine": float(
            primary_integrity["minimum_track_support_cosine"]
        )
        >= float(gates["center_reextraction_track_support_cosine_min"]),
        "replay_center_track_feature_cosine": float(
            replay_integrity["minimum_track_feature_cosine"]
        )
        >= float(gates["center_reextraction_track_feature_cosine_min"]),
        "replay_center_track_support_cosine": float(
            replay_integrity["minimum_track_support_cosine"]
        )
        >= float(gates["center_reextraction_track_support_cosine_min"]),
        "config_hash_exact": (
            primary["config_sha256"]
            == replay["config_sha256"]
            == file_sha256(config_path)
        ),
        "cache_index_hash_exact": (
            primary["cache_index_sha256"] == replay["cache_index_sha256"]
        ),
        "cache_tensor_digest_exact": (
            primary["cache_combined_tensor_hash_digest"]
            == replay["cache_combined_tensor_hash_digest"]
        ),
        "locked_data_unread": all(
            value is False for value in primary["locked_data"].values()
        )
        and all(value is False for value in replay["locked_data"].values()),
    }
    passed = all(checks.values())
    decision = config["formal_decisions"][
        "always" if passed else "integrity_fail"
    ]
    summary = {
        "schema_version": "routeD_state_reinstantiation_basin_gate2_5_summary_v0",
        "date": "2026-07-19",
        "status": "completed_valid" if passed else "completed_integrity_fail",
        "formal_decision": decision,
        "pass": passed,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "primary_report": str(primary_path),
        "primary_report_sha256": file_sha256(primary_path),
        "replay_report": str(replay_path),
        "replay_report_sha256": file_sha256(replay_path),
        "cache_index": primary["cache_index"],
        "cache_index_sha256": primary["cache_index_sha256"],
        "cache_combined_tensor_hash_digest": primary[
            "cache_combined_tensor_hash_digest"
        ],
        "failure_points": primary["failure_points"],
        "failure_videos": primary["failure_videos"],
        "center_reextraction_integrity": primary_integrity,
        "aggregate": primary["aggregate"],
        "direction_summary": primary["direction_summary"],
        "downstream_gate_calibration": primary[
            "downstream_gate_calibration"
        ],
        "independent_replay": {
            "full_report_exact": report_exact,
            "exact": report_exact,
        },
        "checks": checks,
        "locked_data": primary["locked_data"],
        "claim_boundary": (
            "Design-exposed mechanistic coordinate-basin calibration only. A valid "
            "audit may set Gate 3A-v1 candidate radii but does not authorize model "
            "validation, selector training, external evaluation, or a paper claim."
        ),
    }
    summary["summary_payload_sha256"] = canonical_json_sha256(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "formal_decision": decision,
                "pass": passed,
                "independent_replay_exact": report_exact,
                "failure_points": primary["failure_points"],
                "failure_videos": primary["failure_videos"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
