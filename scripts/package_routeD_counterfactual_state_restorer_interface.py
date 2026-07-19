#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    primary_path = Path(args.primary).resolve(); replay_path = Path(args.replay).resolve(); output = Path(args.output).resolve()
    primary = json.loads(primary_path.read_text()); replay = json.loads(replay_path.read_text())
    pcmp = {key: value for key, value in primary.items() if key not in {"sidecar", "sidecar_sha256"}}
    rcmp = {key: value for key, value in replay.items() if key not in {"sidecar", "sidecar_sha256"}}
    report_exact = pcmp == rcmp
    part = torch.load(primary_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    rart = torch.load(replay_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    tensor_exact = _exact(part, rart)
    pass_value = bool(primary["pass"] and replay["pass"] and report_exact and tensor_exact)
    summary = {
        "schema_version": "routeD_counterfactual_state_restorer_interface_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if pass_value else "completed_fail",
        "formal_decision": "ALLOW_GATE2_TEACHER_CACHE_BUILD" if pass_value else "STOP_GATE2_BEFORE_TEACHER_CACHE",
        "pass": pass_value,
        "primary_report_sha256": file_sha256(primary_path),
        "replay_report_sha256": file_sha256(replay_path),
        "primary_sidecar_sha256": primary["sidecar_sha256"],
        "replay_sidecar_sha256": replay["sidecar_sha256"],
        "independent_replay": {"report_exact_excluding_paths": report_exact, "all_nested_tensors_exact": tensor_exact, "exact": report_exact and tensor_exact},
        "trainable_parameters": primary["trainable_parameters"],
        "levels": primary["levels"],
        "checks": {**primary["checks"], "independent_replay_exact": report_exact and tensor_exact},
        "claim_boundary": "Training-partition interface evidence only. No learned model, model-validation, holdout, or external claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": file_sha256(output), "decision": summary["formal_decision"], "pass": pass_value, "independent_replay_exact": report_exact and tensor_exact}, indent=2))


if __name__ == "__main__":
    main()
