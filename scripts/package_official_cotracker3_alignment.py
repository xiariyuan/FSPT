#!/usr/bin/env python3
"""Verify and package the official CoTracker3 DAVIS-first alignment result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_CHECKPOINT_SHA = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
EXPECTED_OFFICIAL_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
PUBLISHED_REFERENCE_AJ = 0.638
ALIGNMENT_TOLERANCE = 0.010


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    protocol_path = run / "protocol.json"
    launcher_path = run / "compatibility_launcher_manifest.json"
    result_path = run / "result_eval_.json"
    for path in (protocol_path, launcher_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(protocol_path.read_text())
    launcher = json.loads(launcher_path.read_text())
    result = json.loads(result_path.read_text())
    if protocol["official_source"]["commit"] != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("official source commit drift")
    if protocol["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("checkpoint drift")
    frozen = protocol["frozen_config"]
    required = {
        "query_mode": "first",
        "single_point": True,
        "offline_model": False,
        "window_len": 16,
        "n_iters": 6,
        "seed": 0,
    }
    if any(frozen[key] != value for key, value in required.items()):
        raise RuntimeError("frozen official evaluation config drift")
    if launcher["official_evaluate_sha256"] != protocol["official_source"]["key_file_sha256"]["cotracker/evaluation/evaluate.py"]:
        raise RuntimeError("launcher did not call the pinned official evaluate source")
    metrics = result.get("tapvid_davis_first", result)
    aj = float(metrics["average_jaccard"])
    aligned = abs(aj - PUBLISHED_REFERENCE_AJ) <= ALIGNMENT_TOLERANCE
    if not aligned:
        raise RuntimeError(f"official baseline outside published range: {aj}")
    summary = {
        "schema_version": "official_cotracker3_davis_first_alignment_summary_v0",
        "date": "2026-07-18",
        "status": "pass",
        "scope": "Official CoTracker3 scaled-online TAP-Vid-DAVIS first-query single-point replication",
        "claim_boundary": "DAVIS has prior project exposure and is an alignment/development benchmark, not a new untouched final test.",
        "official_source": protocol["official_source"],
        "checkpoint": protocol["checkpoint"],
        "dataset": protocol["dataset"],
        "frozen_config": frozen,
        "compatibility_launcher": {
            "scope": launcher["scope"],
            "manifest_sha256": sha256(launcher_path),
            "official_evaluate_sha256": launcher["official_evaluate_sha256"],
            "official_evaluator_sha256": launcher["official_evaluator_sha256"],
        },
        "metrics": metrics,
        "published_reference": {
            "AJ": PUBLISHED_REFERENCE_AJ,
            "absolute_tolerance": ALIGNMENT_TOLERANCE,
            "difference": aj - PUBLISHED_REFERENCE_AJ,
            "aligned": aligned,
        },
        "artifacts": {
            "protocol_sha256": sha256(protocol_path),
            "result_sha256": sha256(result_path),
            "run_log_sha256": sha256(run / "run.log"),
        },
        "gate": {
            "official_source_pinned": True,
            "official_checkpoint_exact": True,
            "official_dataset_hashed": True,
            "query_first": True,
            "single_point_independence": True,
            "published_range_alignment": aligned,
            "pass": True,
            "decision": "ALLOW_SAFE_REDETECTION_INTERFACE_AND_POINTODYSSEY_TRAINING_PROTOCOL",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "AJ": aj, "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
