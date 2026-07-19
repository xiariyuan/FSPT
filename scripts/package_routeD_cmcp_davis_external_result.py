#!/usr/bin/env python3
"""Verify primary/replay P0m evaluations and package the external decision."""
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

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    DAVIS_EXTERNAL_RESULT_SCHEMA,
    evaluate_davis_external_gate,
    load_davis_external_config,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_davis_external_v0.yaml"
DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_cmcp_davis_external_20260719/evaluation/primary.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_cmcp_davis_external_20260719/evaluation/replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_STRONG_BACKBONE_DAVIS_EXTERNAL_SUMMARY_2026-07-19.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_davis_eval_replay(
    primary: dict[str, Any], replay: dict[str, Any]
) -> dict[str, bool]:
    fields = (
        "schema_version",
        "seed",
        "config_sha256",
        "cache_index_sha256",
        "cache_index_payload_sha256",
        "dataset_sha256",
        "video_order_sha256",
        "checkpoint_sha256",
        "combined_model_state_sha256",
        "adapter_state_sha256",
        "cmcp_state_sha256",
        "comparator_state_sha256",
        "normalization",
        "metrics",
        "model_selection_on_DAVIS",
        "threshold_selection_on_DAVIS",
        "calibration_read",
        "kinetics_read_or_rerun",
        "claim_boundary",
    )
    checks = {key: primary[key] == replay[key] for key in fields}
    if not all(checks.values()):
        raise RuntimeError(f"P0m replay mismatch: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--primary", default=str(DEFAULT_PRIMARY))
    parser.add_argument("--replay", default=str(DEFAULT_REPLAY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    config = load_davis_external_config(args.config)
    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    primary = _read(primary_path)
    replay = _read(replay_path)
    for label, row in (("primary", primary), ("replay", replay)):
        if row["schema_version"] != DAVIS_EXTERNAL_RESULT_SCHEMA:
            raise RuntimeError(f"bad {label} schema")
        if row["config_sha256"] != config["_config_sha256"]:
            raise RuntimeError(f"bad {label} config")
        if (
            row["model_selection_on_DAVIS"]
            or row["threshold_selection_on_DAVIS"]
            or row["calibration_read"]
            or row["kinetics_read_or_rerun"]
        ):
            raise RuntimeError(f"forbidden adaptation/read in {label}")
        if row["claim_boundary"] != config["claim_boundary"]:
            raise RuntimeError(f"claim boundary drift in {label}")
    replay_checks = verify_davis_eval_replay(primary, replay)
    metrics = primary["metrics"]
    gate = evaluate_davis_external_gate(metrics, config, exact_replay=True)
    gains = [float(row["selected_AJ_gain_points"]) for row in metrics["per_video"]]
    summary = {
        "schema_version": "routeD_strong_backbone_davis_external_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if gate["pass"] else "completed_fail",
        "formal_decision": gate["decision"],
        "claim_boundary": config["claim_boundary"],
        "model": {
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "combined_model_state_sha256": primary[
                "combined_model_state_sha256"
            ],
        },
        "cache": {
            "index": primary["cache_index"],
            "index_sha256": primary["cache_index_sha256"],
            "payload_sha256": primary["cache_index_payload_sha256"],
            "dataset_sha256": primary["dataset_sha256"],
            "video_order_sha256": primary["video_order_sha256"],
        },
        "metrics": metrics,
        "gate": gate,
        "per_video_summary": {
            "videos": len(gains),
            "positive": sum(value > 0 for value in gains),
            "negative": sum(value < 0 for value in gains),
            "zero": sum(value == 0 for value in gains),
            "min": min(gains),
            "median": statistics.median(gains),
            "mean": statistics.mean(gains),
            "max": max(gains),
        },
        "replay": {
            "exact": True,
            "checks": replay_checks,
            "primary_sha256": file_sha256(primary_path),
            "replay_sha256": file_sha256(replay_path),
        },
        "calibration_read": False,
        "kinetics_read_or_rerun": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": gate["decision"],
                "AJ_gain_points": metrics["selected_gain_points"]["AJ"],
                "AJ_CI": metrics["paired_video_selected_AJ_gain_CI"],
                "gate_pass": gate["pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
