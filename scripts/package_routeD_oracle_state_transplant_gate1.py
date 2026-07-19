#!/usr/bin/env python3
"""Package independent oracle state-transplant Gate 1 runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_oracle_state_transplant import (
    GATE1_SCHEMA_VERSION,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_oracle_state_transplant_gate1_v0.yaml"
DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_oracle_state_transplant_gate1_20260719/primary.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_oracle_state_transplant_gate1_20260719/replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_ORACLE_STATE_TRANSPLANT_GATE1_SUMMARY_2026-07-19.json"


def nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(nested_exact(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(nested_exact(lhs, rhs) for lhs, rhs in zip(left, right))
        )
    return left == right


def report_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"sidecar", "sidecar_sha256"}
    }


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
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    if primary.get("schema_version") != GATE1_SCHEMA_VERSION or replay.get(
        "schema_version"
    ) != GATE1_SCHEMA_VERSION:
        raise ValueError("Gate 1 report schema mismatch")
    primary_artifact = torch.load(
        primary["sidecar"], map_location="cpu", weights_only=False
    )
    replay_artifact = torch.load(
        replay["sidecar"], map_location="cpu", weights_only=False
    )
    report_exact = report_view(primary) == report_view(replay)
    artifact_exact = nested_exact(primary_artifact, replay_artifact)
    independent_replay_exact = report_exact and artifact_exact
    checks = {
        **primary["local_checks"],
        "primary_replay_report_exact_excluding_paths": report_exact,
        "primary_replay_all_nested_tensors_and_scalars_exact": artifact_exact,
        "independent_replay_exact": independent_replay_exact,
        "primary_local_pass": bool(primary["local_pass"]),
        "replay_local_pass": bool(replay["local_pass"]),
    }
    passed = all(checks.values())
    decision = config["formal_decisions"]["pass" if passed else "fail"]
    summary = {
        "schema_version": "routeD_oracle_state_transplant_gate1_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if passed else "completed_fail",
        "formal_decision": decision,
        "pass": passed,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "primary_report_sha256": file_sha256(primary_path),
        "replay_report_sha256": file_sha256(replay_path),
        "primary_sidecar_sha256": file_sha256(primary["sidecar"]),
        "replay_sidecar_sha256": file_sha256(replay["sidecar"]),
        "independent_replay": {
            "report_exact_excluding_paths": report_exact,
            "all_nested_tensors_and_scalars_exact": artifact_exact,
            "exact": independent_replay_exact,
        },
        "selected_points": primary["selected_points"],
        "selected_videos": primary["selected_videos"],
        "variant_aggregates": primary["variant_aggregates"],
        "comparisons": primary["comparisons"],
        "support_svd_aggregate": primary["support_svd_aggregate"],
        "video_reports": primary["video_reports"],
        "checks": checks,
        "locked_data": primary["locked_data"],
        "claim_boundary": (
            "Fit-only oracle teacher evidence. A pass authorizes only a separately "
            "preregistered learned restorer protocol; it is not inference-time, "
            "external-transfer, or paper-level evidence."
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
                "decision": decision,
                "pass": passed,
                "independent_replay_exact": independent_replay_exact,
                "selected_points": primary["selected_points"],
                "selected_videos": primary["selected_videos"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
