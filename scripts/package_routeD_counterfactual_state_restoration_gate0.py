#!/usr/bin/env python3
"""Package independent Counterfactual Tracker-State Restoration Gate 0 runs."""
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

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import GATE0_SCHEMA_VERSION
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_counterfactual_state_restoration_gate0_v0.yaml"
DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_counterfactual_state_restoration_gate0_20260719/primary.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_counterfactual_state_restoration_gate0_20260719/replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_COUNTERFACTUAL_STATE_RESTORATION_GATE0_SUMMARY_2026-07-19.json"


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right) and all(
            _nested_exact(lhs, rhs) for lhs, rhs in zip(left, right)
        )
    return left == right


def _report_view(report: dict[str, Any]) -> dict[str, Any]:
    excluded = {"sidecar", "sidecar_sha256"}
    return {key: value for key, value in report.items() if key not in excluded}


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
    if primary.get("schema_version") != GATE0_SCHEMA_VERSION or replay.get("schema_version") != GATE0_SCHEMA_VERSION:
        raise ValueError("Gate 0 report schema mismatch")
    primary_artifact = torch.load(primary["sidecar"], map_location="cpu", weights_only=False)
    replay_artifact = torch.load(replay["sidecar"], map_location="cpu", weights_only=False)
    report_exact = _report_view(primary) == _report_view(replay)
    artifact_exact = _nested_exact(primary_artifact, replay_artifact)
    independent_replay_exact = report_exact and artifact_exact
    local_checks_exact = primary["local_checks"] == replay["local_checks"]
    final_checks = {
        **primary["local_checks"],
        "clean_independent_replay_exact": independent_replay_exact,
        "primary_replay_local_checks_exact": local_checks_exact,
        "primary_local_pass": bool(primary["local_pass"]),
        "replay_local_pass": bool(replay["local_pass"]),
    }
    passed = all(final_checks.values())
    decision = config["formal_decisions"]["pass" if passed else "fail"]
    summary = {
        "schema_version": "routeD_counterfactual_state_restoration_gate0_summary_v0",
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
        "state_shapes": primary["state_shapes"],
        "future_differences_vs_clean": primary["future_differences_vs_clean"],
        "checks": final_checks,
        "sample": primary["sample"],
        "integrity": primary["integrity"],
        "claim_boundary": (
            "Fit-only causal interface evidence. A pass authorizes only a separately "
            "preregistered learned state-restorer experiment; it is not a tracking "
            "performance, external-transfer, or paper claim."
        ),
    }
    summary["summary_payload_sha256"] = canonical_json_sha256(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "sha256": file_sha256(output),
        "decision": decision,
        "pass": passed,
        "independent_replay_exact": independent_replay_exact,
    }, indent=2))


if __name__ == "__main__":
    main()
