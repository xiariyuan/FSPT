#!/usr/bin/env python3
"""Qualify exact primary/replay candidate caches before Gate 3A v1 teacher read."""
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


SCHEMA = "routeD_geometry_representation_audit_gate3a_v1"
INDEX_SCHEMA = "routeD_geometry_representation_candidate_index_gate3a_v1"


def nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
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
            and all(nested_exact(a, b) for a, b in zip(left, right))
        )
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--primary-index", required=True)
    parser.add_argument("--replay-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    primary_path = Path(args.primary_index).resolve()
    replay_path = Path(args.replay_index).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3A v1 candidate qualification config")
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    for value in (primary, replay):
        if value.get("schema_version") != INDEX_SCHEMA or not value.get("complete"):
            raise ValueError("Gate 3A v1 candidate index incomplete")
        if value.get("config_sha256") != file_sha256(config_path):
            raise ValueError("Gate 3A v1 candidate index config drift")
    membership_exact = (
        primary["expected_source_indices"]
        == primary["completed_source_indices"]
        == replay["expected_source_indices"]
        == replay["completed_source_indices"]
    )
    digest_exact = (
        primary["combined_candidate_hash_digest"]
        == replay["combined_candidate_hash_digest"]
    )
    row_identity_exact = [
        (int(row["source_index"]), str(row["video_name"]), int(row["failure_points"]))
        for row in primary["rows"]
    ] == [
        (int(row["source_index"]), str(row["video_name"]), int(row["failure_points"]))
        for row in replay["rows"]
    ]
    sidecar_checks = []
    for left_row, right_row in zip(primary["rows"], replay["rows"]):
        left = torch.load(left_row["sidecar"], map_location="cpu", weights_only=False)
        right = torch.load(right_row["sidecar"], map_location="cpu", weights_only=False)
        sidecar_checks.append(
            {
                "source_index": int(left_row["source_index"]),
                "candidate_hash_digest_exact": (
                    left["candidate_hash_digest"]
                    == right["candidate_hash_digest"]
                    == left_row["candidate_hash_digest"]
                    == right_row["candidate_hash_digest"]
                ),
                "all_nested_tensors_and_metadata_exact": nested_exact(left, right),
            }
        )
    sidecars_exact = all(
        row["candidate_hash_digest_exact"]
        and row["all_nested_tensors_and_metadata_exact"]
        for row in sidecar_checks
    )
    checks = {
        "membership_exact": membership_exact,
        "row_identity_exact": row_identity_exact,
        "combined_candidate_hash_digest_exact": digest_exact,
        "all_sidecars_nested_exact": sidecars_exact,
        "candidate_sets_frozen_before_teacher_read": bool(
            primary["integrity"]["candidate_sets_frozen_before_teacher_read"]
        )
        and bool(replay["integrity"]["candidate_sets_frozen_before_teacher_read"]),
        "model_validation_unread": not bool(primary["integrity"]["model_validation_read"])
        and not bool(replay["integrity"]["model_validation_read"]),
        "external_unread": not bool(primary["integrity"]["external_read"])
        and not bool(replay["integrity"]["external_read"]),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "routeD_geometry_candidate_cache_qualification_gate3a_v1",
        "date": "2026-07-19",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "formal_decision": (
            "AUTHORIZE_GATE3A_V1_TEACHER_AUDIT"
            if passed
            else config["formal_decisions"]["integrity_fail"]
        ),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "primary_index": str(primary_path),
        "primary_index_sha256": file_sha256(primary_path),
        "replay_index": str(replay_path),
        "replay_index_sha256": file_sha256(replay_path),
        "combined_candidate_hash_digest": primary[
            "combined_candidate_hash_digest"
        ],
        "checks": checks,
        "sidecar_checks": sidecar_checks,
    }
    summary["summary_payload_sha256"] = canonical_json_sha256(summary)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
