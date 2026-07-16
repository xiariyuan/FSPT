#!/usr/bin/env python3
"""Final integrity and claim audit for the full local Kinetics evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    merged_path = Path(args.merged).resolve()
    paired_path = Path(args.paired).resolve()
    protocol = json.loads(protocol_path.read_text())
    merged = json.loads(merged_path.read_text())
    paired = json.loads(paired_path.read_text())

    expected_videos = int(protocol["expected_video_count"])
    require(
        bool(protocol.get("created_before_any_full_dataset_inference")),
        "Protocol was not marked as created before full-dataset inference",
    )
    require(merged.get("samples") == expected_videos, "Merged sample count mismatch")
    require(len(merged.get("per_sample", [])) == expected_videos, "Merged row count mismatch")
    video_names = [str(row.get("video_name", "")) for row in merged["per_sample"]]
    require(all(video_names), "Missing canonical video name")
    require(len(set(video_names)) == expected_videos, "Canonical video names are not unique")
    require(merged.get("protocol_sha256") == sha256(protocol_path), "Protocol hash mismatch")

    frozen_input_audit = {}
    for name, entry in protocol["frozen_inputs"].items():
        path = Path(entry["path"])
        actual = sha256(path)
        frozen_input_audit[name] = {
            "path": str(path),
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual,
            "match": actual == entry["sha256"],
        }
        require(actual == entry["sha256"], f"Frozen input hash mismatch: {name}")

    source_manifest_path = Path(protocol["source_manifest"])
    source_hash_protocol_path = Path(protocol["source_hash_protocol"])
    require(
        sha256(source_manifest_path) == protocol["source_manifest_sha256"],
        "Source manifest hash mismatch",
    )
    require(
        sha256(source_hash_protocol_path) == protocol["source_hash_protocol_sha256"],
        "Source hash protocol mismatch",
    )
    source_hash_protocol = json.loads(source_hash_protocol_path.read_text())
    source_hashes = {
        int(row["source_shard_index"]): row["sha256"]
        for row in source_hash_protocol["source_shards"]
    }

    shard_audit_by_index = {
        int(row["shard_index"]): row for row in merged.get("shard_audit", [])
    }
    require(len(shard_audit_by_index) == len(protocol["shards"]), "Shard audit count mismatch")
    shard_integrity = []
    for shard in protocol["shards"]:
        index = int(shard["shard_index"])
        result_path = Path(shard["expected_result"])
        result_payload = json.loads(result_path.read_text())
        result_hash = sha256(result_path)
        expected_rows = int(shard["source_num_samples"])
        require(result_payload.get("samples") == expected_rows, f"Shard {index} sample count mismatch")
        require(len(result_payload.get("per_sample", [])) == expected_rows, f"Shard {index} row count mismatch")
        require(
            result_hash == shard_audit_by_index[index]["result_sha256"],
            f"Shard {index} result hash mismatch",
        )
        require(
            source_hashes[index] == shard["source_sha256"],
            f"Shard {index} source hash provenance mismatch",
        )
        manifest_path = Path(shard["evaluation_manifest"])
        require(
            sha256(manifest_path) == shard["evaluation_manifest_sha256"],
            f"Shard {index} evaluation manifest hash mismatch",
        )
        shard_integrity.append(
            {
                "shard_index": index,
                "rows": expected_rows,
                "result": str(result_path),
                "result_sha256": result_hash,
                "source_sha256": shard["source_sha256"],
                "evaluation_manifest_sha256": shard["evaluation_manifest_sha256"],
            }
        )

    primary = paired["paired_bootstrap"]["closed_vs_baseline"]
    aj = primary["AJ"]
    delta = primary["delta_avg"]
    require(aj["videos"] > 0 and delta["videos"] > 0, "No finite paired videos")
    require(math.isfinite(float(aj["ci95_low"])), "AJ confidence interval is nonfinite")
    require(math.isfinite(float(delta["ci95_low"])), "delta_avg confidence interval is nonfinite")
    primary_pass = float(aj["ci95_low"]) > 0.0 and float(delta["ci95_low"]) > 0.0

    invalid_rows = paired.get("failure_audit", {}).get("invalid_metric_rows", [])
    invalid_videos = sorted(
        {
            (int(row["sample"]), str(row["video_name"]))
            for row in invalid_rows
        }
    )
    result = {
        "kind": "routeD_full_local_kinetics_final_audit",
        "paper_claim_eligible": bool(primary_pass),
        "primary_pass": bool(primary_pass),
        "claim_boundary": protocol["claim_boundary"],
        "required_wording": (
            "Report this as the complete local 1,144-video TAP-Vid-Kinetics package "
            "evaluation, not as an official benchmark result unless package identity "
            "and official protocol are independently verified."
        ),
        "expected_videos": expected_videos,
        "merged_videos": len(merged["per_sample"]),
        "finite_paired_videos": {
            "AJ": int(aj["videos"]),
            "delta_avg": int(delta["videos"]),
        },
        "invalid_metric_entries": len(invalid_rows),
        "invalid_videos": [
            {"sample": sample, "video_name": video_name}
            for sample, video_name in invalid_videos
        ],
        "invalid_video_count": len(invalid_videos),
        "closed_vs_baseline": {
            "AJ": aj,
            "delta_avg": delta,
        },
        "aggregate": merged["aggregate"],
        "delta_routeD_vs_baseline": merged["delta_routeD_vs_baseline"],
        "metric_validity": merged.get("metric_validity"),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "merged": str(merged_path),
        "merged_sha256": sha256(merged_path),
        "paired": str(paired_path),
        "paired_sha256": sha256(paired_path),
        "frozen_input_audit": frozen_input_audit,
        "shard_integrity": shard_integrity,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
