#!/usr/bin/env python3
"""Merge independently evaluated Route-D dataset shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    all_rows = []
    shard_audit = []
    expected_global_index = 0
    reference_metadata = None

    for shard in protocol["shards"]:
        result_path = Path(shard["expected_result"])
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        payload = json.loads(result_path.read_text())
        expected_rows = int(shard["source_num_samples"])
        rows = payload.get("per_sample", [])
        if len(rows) != expected_rows:
            raise ValueError(
                f"Shard {shard['shard_index']} has {len(rows)} rows; "
                f"expected {expected_rows}"
            )
        metadata = {
            key: payload.get(key)
            for key in (
                "evaluation_mode",
                "closed_loop_state_updated",
                "independent_baseline_model",
                "baseline_has_routeD_selector",
                "threshold",
                "seed",
                "dataset",
                "query_mode",
                "input_resolution",
                "metric_resolution",
                "fusion_strength",
                "max_switch_distance_px",
                "profile_p1_tolerance",
                "profile_min_coarse_gain",
                "profile_min_total_gain",
            )
        }
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError(
                f"Evaluation metadata mismatch in shard {shard['shard_index']}"
            )

        for local_index, row in enumerate(rows):
            copied = dict(row)
            raw_video_name = str(copied.get("video_name", ""))
            copied["raw_video_name"] = raw_video_name
            copied["video_name"] = (
                f"kinetics_source_s{int(shard['shard_index']):03d}_"
                f"p{local_index:06d}_{raw_video_name}"
            )
            copied["source_shard_index"] = int(shard["shard_index"])
            copied["source_shard_local_index"] = local_index
            copied["sample"] = expected_global_index
            all_rows.append(copied)
            expected_global_index += 1
        shard_audit.append(
            {
                "shard_index": int(shard["shard_index"]),
                "result": str(result_path),
                "result_sha256": sha256(result_path),
                "rows": len(rows),
            }
        )

    expected_total = int(protocol["expected_video_count"])
    if len(all_rows) != expected_total:
        raise ValueError(f"Merged {len(all_rows)} rows; expected {expected_total}")
    video_names = [str(row.get("video_name", "")) for row in all_rows]
    if any(not name for name in video_names):
        raise ValueError("At least one merged row is missing video_name")
    if len(set(video_names)) != len(video_names):
        raise ValueError("Merged video_name values are not unique")

    prediction_names = ("baseline", "local", "routeD_open", "routeD_closed")
    metrics = ("AJ", "OA", "delta_avg")
    aggregate = {
        name: {
            metric: finite_mean(row[name][metric] for row in all_rows)
            for metric in metrics
        }
        for name in prediction_names
    }
    diagnostic_keys = [
        key
        for key in all_rows[0]
        if key
        not in {
            "sample",
            "video_name",
            "source_shard_index",
            "source_shard_local_index",
            *prediction_names,
        }
        and isinstance(all_rows[0][key], (int, float))
    ]
    aggregate_diagnostics = {
        key: finite_mean(row[key] for row in all_rows) for key in diagnostic_keys
    }
    delta = {}
    for label, left, right in (
        ("closed_vs_baseline", "routeD_closed", "baseline"),
        ("open_vs_baseline", "routeD_open", "baseline"),
        ("closed_vs_open", "routeD_closed", "routeD_open"),
    ):
        delta[label] = {
            metric: aggregate[left][metric] - aggregate[right][metric]
            for metric in metrics
        }

    result = {
        "evidence_tier": "predeclared_full_local_dataset_evaluation",
        "paper_claim_eligible": True,
        "claim_boundary": protocol["claim_boundary"],
        **(reference_metadata or {}),
        "dataset_root": protocol["source_root"],
        "samples": len(all_rows),
        "aggregate": aggregate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "delta_routeD_vs_baseline": delta,
        "per_sample": all_rows,
        "shard_audit": shard_audit,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(output),
        "samples": len(all_rows),
        "aggregate": aggregate,
        "delta": delta,
    }, indent=2))


if __name__ == "__main__":
    main()
