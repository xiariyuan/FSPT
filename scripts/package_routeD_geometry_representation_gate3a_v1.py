#!/usr/bin/env python3
"""Package exact primary/replay Gate 3A v1 teacher audits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
QUALIFICATION_SCHEMA = "routeD_geometry_candidate_cache_qualification_gate3a_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-qualification", required=True)
    parser.add_argument("--primary-report", required=True)
    parser.add_argument("--replay-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    qualification_path = Path(args.candidate_qualification).resolve()
    primary_path = Path(args.primary_report).resolve()
    replay_path = Path(args.replay_report).resolve()
    output = Path(args.output).resolve()
    config = yaml.safe_load(config_path.read_text())
    qualification = json.loads(qualification_path.read_text())
    primary = json.loads(primary_path.read_text())
    replay = json.loads(replay_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3A v1 config")
    if (
        qualification.get("schema_version") != QUALIFICATION_SCHEMA
        or not bool(qualification.get("pass"))
    ):
        raise ValueError("Gate 3A v1 candidate cache was not qualified")
    for report in (primary, replay):
        if report.get("schema_version") != SCHEMA or report.get("status") != "completed":
            raise ValueError("Gate 3A v1 teacher audit incomplete")
        if report.get("config_sha256") != file_sha256(config_path):
            raise ValueError("Gate 3A v1 report/config drift")
        if report.get("candidate_qualification_sha256") != file_sha256(
            qualification_path
        ):
            raise ValueError("Gate 3A v1 report/qualification drift")
        if report.get("candidate_index_sha256") != qualification.get(
            "primary_index_sha256"
        ):
            raise ValueError("Gate 3A v1 report used an unqualified candidate index")
    full_report_exact = primary == replay
    if full_report_exact:
        formal_decision = primary["formal_decision"]
        selected_representation = primary["selected_representation"]
        status = "completed_pass_integrity"
    else:
        formal_decision = config["formal_decisions"]["integrity_fail"]
        selected_representation = None
        status = "completed_fail_integrity"
    summary = {
        "schema_version": "routeD_geometry_representation_audit_gate3a_v1_summary",
        "date": "2026-07-19",
        "status": status,
        "formal_decision": formal_decision,
        "selected_representation": selected_representation,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "candidate_qualification": str(qualification_path),
        "candidate_qualification_sha256": file_sha256(qualification_path),
        "candidate_cache_replay_exact": bool(qualification["pass"]),
        "teacher_audit_full_report_replay_exact": full_report_exact,
        "primary_report": str(primary_path),
        "primary_report_sha256": file_sha256(primary_path),
        "replay_report": str(replay_path),
        "replay_report_sha256": file_sha256(replay_path),
        "failure_points": int(primary["failure_points"]),
        "failure_videos": int(primary["failure_videos"]),
        "native": primary["native"],
        "representations": primary["representations"],
        "checks": primary["checks"],
        "locked_data": primary["locked_data"],
    }
    summary["summary_payload_sha256"] = canonical_json_sha256(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
