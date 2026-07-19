#!/usr/bin/env python3
"""Package independent replay of the four fixed CSRR cache anchors."""
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
    return {key: value for key, value in report.items() if key not in {"sidecar", "sidecar_sha256"}}


def _normalise_index(index: dict[str, Any]) -> dict[str, Any]:
    ignored_top = {
        "combined_sidecar_sha256",
        "cache_index_payload_sha256",
    }
    result = {key: value for key, value in index.items() if key not in ignored_top and key != "rows"}
    result["rows"] = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "sidecar",
                "sidecar_sha256",
                "report",
                "report_sha256",
            }
        }
        for row in index["rows"]
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-index", required=True)
    parser.add_argument("--replay-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    primary_path = Path(args.primary_index).resolve()
    replay_path = Path(args.replay_index).resolve()
    output = Path(args.output).resolve()
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    if primary["completed_source_indices"] != [8, 31, 32, 47]:
        raise ValueError("unexpected primary smoke anchors")
    if replay["completed_source_indices"] != [8, 31, 32, 47]:
        raise ValueError("unexpected replay smoke anchors")
    index_exact = _normalise_index(primary) == _normalise_index(replay)
    video_rows = []
    all_artifacts_exact = True
    all_reports_exact = True
    for primary_row, replay_row in zip(primary["rows"], replay["rows"]):
        if primary_row["source_index"] != replay_row["source_index"]:
            raise ValueError("smoke source ordering mismatch")
        primary_artifact = torch.load(primary_row["sidecar"], map_location="cpu", weights_only=False)
        replay_artifact = torch.load(replay_row["sidecar"], map_location="cpu", weights_only=False)
        artifact_exact = nested_exact(primary_artifact, replay_artifact)
        primary_report = json.loads(Path(primary_row["report"]).read_text())
        replay_report = json.loads(Path(replay_row["report"]).read_text())
        report_exact = _normalise_report(primary_report) == _normalise_report(replay_report)
        all_artifacts_exact = all_artifacts_exact and artifact_exact
        all_reports_exact = all_reports_exact and report_exact
        video_rows.append(
            {
                "source_index": primary_row["source_index"],
                "artifact_all_nested_tensors_and_metadata_exact": artifact_exact,
                "report_exact_excluding_paths": report_exact,
                "primary_sidecar_sha256": primary_row["sidecar_sha256"],
                "replay_sidecar_sha256": replay_row["sidecar_sha256"],
                "tensor_hash_digest": primary_row["tensor_hash_digest"],
            }
        )
    pass_value = bool(
        primary["complete"]
        and replay["complete"]
        and index_exact
        and all_artifacts_exact
        and all_reports_exact
    )
    summary = {
        "schema_version": "routeD_csrr_teacher_cache_smoke_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if pass_value else "completed_fail",
        "formal_decision": "ALLOW_COMPLETE_GATE2_TEACHER_CACHE_BUILD" if pass_value else "STOP_GATE2_CACHE_BEFORE_COMPLETE_BUILD",
        "pass": pass_value,
        "primary_index": str(primary_path),
        "primary_index_sha256": file_sha256(primary_path),
        "replay_index": str(replay_path),
        "replay_index_sha256": file_sha256(replay_path),
        "independent_replay": {
            "index_exact_excluding_paths_and_serialization_hashes": index_exact,
            "all_artifacts_exact": all_artifacts_exact,
            "all_reports_exact_excluding_paths": all_reports_exact,
            "exact": index_exact and all_artifacts_exact and all_reports_exact,
        },
        "completed_source_indices": primary["completed_source_indices"],
        "total_failure_rows": primary["total_failure_rows"],
        "total_clean_rows": primary["total_clean_rows"],
        "maximum_quantization_absolute_error": primary["maximum_quantization_absolute_error"],
        "minimum_quantization_cosine_similarity": primary["minimum_quantization_cosine_similarity"],
        "combined_tensor_hash_digest": primary["combined_tensor_hash_digest"],
        "videos": video_rows,
        "claim_boundary": "Four-anchor cache interface replay only; no learned model or performance claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": summary["formal_decision"],
                "pass": pass_value,
                "independent_replay_exact": summary["independent_replay"]["exact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
