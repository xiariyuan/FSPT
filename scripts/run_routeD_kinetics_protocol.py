#!/usr/bin/env python3
"""Run a prepared Route-D Kinetics protocol sequentially and resumably."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def validate_result(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("samples", -1)) == int(expected_rows)
        and len(payload.get("per_sample", [])) == int(expected_rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--evaluator", default="scripts/eval_routeD_risk_selector.py"
    )
    parser.add_argument("--only-shards", type=int, nargs="*", default=None)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    repo = Path(args.repo).resolve()
    evaluator = Path(args.evaluator)
    if not evaluator.is_absolute():
        evaluator = repo / evaluator

    selected = (
        set(int(value) for value in args.only_shards)
        if args.only_shards is not None
        else None
    )
    frozen = protocol["frozen_inputs"]
    evaluation = protocol["evaluation"]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    completed = []
    skipped = []
    for shard in sorted(protocol["shards"], key=lambda row: row["shard_index"]):
        shard_index = int(shard["shard_index"])
        if selected is not None and shard_index not in selected:
            continue
        expected_rows = int(shard["source_num_samples"])
        result_path = Path(shard["expected_result"])
        if validate_result(result_path, expected_rows):
            print(
                json.dumps(
                    {
                        "event": "skip_valid_result",
                        "shard": shard_index,
                        "rows": expected_rows,
                        "result": str(result_path),
                    }
                ),
                flush=True,
            )
            skipped.append(shard_index)
            continue

        if result_path.exists():
            result_path.unlink()
        command = [
            sys.executable,
            str(evaluator),
            "--config",
            frozen["config"]["path"],
            "--checkpoint",
            frozen["checkpoint"]["path"],
            "--selector-bundle",
            frozen["controller"]["path"],
            "--threshold",
            "0.5",
            "--profile-p1-tolerance",
            "0.0",
            "--profile-min-coarse-gain",
            "0.0",
            "--profile-min-total-gain",
            "0.0",
            "--closed-loop",
            "--split",
            "val",
            "--dataset-root",
            str(protocol_path.parent),
            "--annotation-file",
            shard["evaluation_manifest"],
            "--dataset-split",
            evaluation["dataset_split"],
            "--dataset",
            evaluation["dataset"],
            "--query-mode",
            evaluation["query_mode"],
            "--resolution",
            *[str(value) for value in evaluation["input_resolution"]],
            "--metric-resolution",
            *[str(value) for value in evaluation["metric_resolution"]],
            "--limit",
            str(expected_rows),
            "--device",
            "cuda",
            "--baseline-device",
            "cpu",
            "--seed",
            str(evaluation["seed"]),
            "--output",
            str(result_path),
        ]
        print(
            json.dumps(
                {
                    "event": "start_shard",
                    "shard": shard_index,
                    "rows": expected_rows,
                    "command": command,
                }
            ),
            flush=True,
        )
        subprocess.run(command, cwd=repo, env=env, check=True)
        if not validate_result(result_path, expected_rows):
            raise RuntimeError(
                f"Shard {shard_index} finished without a valid {expected_rows}-row result"
            )
        print(
            json.dumps(
                {
                    "event": "complete_shard",
                    "shard": shard_index,
                    "rows": expected_rows,
                    "result": str(result_path),
                }
            ),
            flush=True,
        )
        completed.append(shard_index)

    print(
        json.dumps(
            {
                "event": "protocol_complete",
                "completed_this_run": completed,
                "skipped_existing": skipped,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
