#!/usr/bin/env python3
"""Run a prepared Route-D Kinetics protocol sequentially and resumably."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(
    path: Path,
    expected_rows: int,
    expected_metric_hash: str | None = None,
    expected_metric_contract: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        int(payload.get("samples", -1)) == int(expected_rows)
        and len(payload.get("per_sample", [])) == int(expected_rows)
    )
    if expected_metric_hash is not None:
        valid = valid and payload.get("metric_implementation_sha256") == expected_metric_hash
    if expected_metric_contract is not None:
        valid = valid and payload.get("metric_coordinate_contract") == expected_metric_contract
    return bool(valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--evaluator", default="scripts/eval_routeD_risk_selector.py"
    )
    parser.add_argument("--only-shards", type=int, nargs="*", default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
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
    expected_metric_hash = frozen.get("metric_implementation", {}).get("sha256")
    expected_metric_contract = evaluation.get("normalized_to_raster_contract")
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if current_head != protocol.get("git_head"):
        raise ValueError(
            f"Git HEAD {current_head} does not match protocol {protocol.get('git_head')}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if dirty:
        raise ValueError("Protocol runner requires a clean working tree")
    for name, entry in frozen.items():
        path = Path(entry["path"])
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Frozen input hash mismatch before run: {name}")
    for name, entry in protocol.get("official_protocol_evidence", {}).items():
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            continue
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Official protocol evidence hash mismatch: {name}")
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
        if validate_result(
            result_path,
            expected_rows,
            expected_metric_hash=expected_metric_hash,
            expected_metric_contract=expected_metric_contract,
        ):
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
        completed_attempt = None
        for attempt in range(1, int(args.max_attempts) + 1):
            if result_path.exists() and not validate_result(
                result_path,
                expected_rows,
                expected_metric_hash=expected_metric_hash,
                expected_metric_contract=expected_metric_contract,
            ):
                result_path.unlink()
            print(
                json.dumps(
                    {
                        "event": "start_shard",
                        "shard": shard_index,
                        "rows": expected_rows,
                        "attempt": attempt,
                        "max_attempts": int(args.max_attempts),
                        "command": command,
                    }
                ),
                flush=True,
            )
            completed_process = subprocess.run(command, cwd=repo, env=env, check=False)
            valid = validate_result(
                result_path,
                expected_rows,
                expected_metric_hash=expected_metric_hash,
                expected_metric_contract=expected_metric_contract,
            )
            print(
                json.dumps(
                    {
                        "event": "shard_attempt_finished",
                        "shard": shard_index,
                        "attempt": attempt,
                        "returncode": completed_process.returncode,
                        "valid_result": valid,
                    }
                ),
                flush=True,
            )
            if valid:
                completed_attempt = attempt
                break
            if attempt < int(args.max_attempts):
                time.sleep(float(args.retry_delay_seconds))
        if completed_attempt is None:
            raise RuntimeError(
                f"Shard {shard_index} failed to produce a valid "
                f"{expected_rows}-row result after {args.max_attempts} attempts"
            )
        print(
            json.dumps(
                {
                    "event": "complete_shard",
                    "shard": shard_index,
                    "rows": expected_rows,
                    "result": str(result_path),
                    "attempt": completed_attempt,
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
