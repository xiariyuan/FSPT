#!/usr/bin/env python3
"""Package primary/replay results for Route-D CSRR Gate 2."""
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


def nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(nested_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_exact(a, b) for a, b in zip(left, right))
    return left == right


def _normalise_report(report: dict[str, Any]) -> dict[str, Any]:
    ignored = {"checkpoint", "checkpoint_sha256"}
    return {key: value for key, value in report.items() if key not in ignored}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    report_exact = _normalise_report(primary) == _normalise_report(replay)
    primary_checkpoint = torch.load(
        Path(primary["checkpoint"]), map_location="cpu", weights_only=False
    )
    replay_checkpoint = torch.load(
        Path(replay["checkpoint"]), map_location="cpu", weights_only=False
    )
    checkpoint_exact = nested_exact(primary_checkpoint, replay_checkpoint)
    model_state_exact = nested_exact(
        primary_checkpoint["model_state"], replay_checkpoint["model_state"]
    )
    digest_exact = (
        primary["model_state_digest"]
        == replay["model_state_digest"]
        == primary_checkpoint["model_state_digest"]
        == replay_checkpoint["model_state_digest"]
    )
    independent_exact = report_exact and checkpoint_exact and model_state_exact and digest_exact
    local_pass = bool(primary["local_pass"] and replay["local_pass"])
    pass_value = bool(local_pass and independent_exact)
    validation = primary["validation"]
    summary = {
        "schema_version": "routeD_csrr_gate2_training_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if pass_value else "completed_fail",
        "formal_decision": (
            "AUTHORIZE_ONE_FROZEN_MODEL_VALIDATION_STATE_RESTORER_AUDIT"
            if pass_value
            else "STOP_LEARNED_STATE_RESTORER_BEFORE_MODEL_VALIDATION"
        ),
        "pass": pass_value,
        "local_pass": local_pass,
        "config": primary["config"],
        "config_sha256": primary["config_sha256"],
        "seed": primary["seed"],
        "train_cache_index_sha256": primary["train_cache_index_sha256"],
        "validation_cache_index_sha256": primary[
            "validation_cache_index_sha256"
        ],
        "primary_report": str(primary_path),
        "primary_report_sha256": file_sha256(primary_path),
        "replay_report": str(replay_path),
        "replay_report_sha256": file_sha256(replay_path),
        "primary_checkpoint_sha256": primary["checkpoint_sha256"],
        "replay_checkpoint_sha256": replay["checkpoint_sha256"],
        "model_state_digest": primary["model_state_digest"],
        "best_epoch": primary["best_epoch"],
        "trainable_parameters": primary["trainable_parameters"],
        "independent_replay": {
            "report_exact_excluding_output_paths": report_exact,
            "checkpoint_all_nested_tensors_and_metadata_exact": checkpoint_exact,
            "model_state_exact": model_state_exact,
            "model_state_digest_exact": digest_exact,
            "exact": independent_exact,
        },
        "validation": validation,
        "checks": {
            **primary["checks"],
            "independent_replay_exact": independent_exact,
        },
        "integrity": primary["integrity"],
        "claim_boundary": (
            "Fit-only learned restorer result. A pass authorizes one frozen read "
            "of model-validation indices 48-63 only; no calibration, holdout, DAVIS, "
            "Kinetics, or paper claim."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": summary["formal_decision"],
                "pass": pass_value,
                "local_pass": local_pass,
                "independent_replay_exact": independent_exact,
                "best_epoch": primary["best_epoch"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
