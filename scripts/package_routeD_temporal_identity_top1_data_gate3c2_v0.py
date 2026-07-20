#!/usr/bin/env python3
"""Qualify the raw-record-disjoint Kubric renewal for Gate 3C1D v1."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256

SCHEMA = "routeD_temporal_identity_top1_data_gate3c2_v0"
MATERIALIZER_SCHEMA = "routeD_kubric_excluding_manifest_materializer_v1"


def manifest_samples(path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    for entry in manifest["shards"]:
        shard = Path(entry["path"])
        if not shard.is_absolute():
            shard = path.parent / shard
        if file_sha256(shard) != entry["sha256"]:
            raise ValueError(f"Gate 3C2 shard hash drift: {shard}")
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        values = list(payload.values()) if isinstance(payload, dict) else payload
        if len(values) != int(entry["num_samples"]):
            raise ValueError("Gate 3C2 shard sample-count drift")
        yield from values


def raw_identities(path: Path) -> list[tuple[str, int]]:
    return [
        (str(sample["source_tfrecord"]), int(sample["source_record_index"]))
        for sample in manifest_samples(path)
    ]


def validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    for key, hash_key in (
        ("result", "result_sha256"),
        ("summary", "summary_sha256"),
        ("checkpoint_replay", "checkpoint_replay_sha256"),
    ):
        if file_sha256(parent[key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C2 parent hash drift: {key}")
    summary = json.loads(Path(parent["summary"]).read_text())
    replay = json.loads(Path(parent["checkpoint_replay"]).read_text())
    summary_without_hash = dict(summary)
    summary_payload = summary_without_hash.pop("summary_payload_sha256", None)
    replay_without_hash = dict(replay)
    replay_payload = replay_without_hash.pop("result_payload_sha256", None)
    if (
        summary_payload != parent["summary_payload_sha256"]
        or summary_payload != canonical_json_sha256(summary_without_hash)
        or replay_payload != canonical_json_sha256(replay_without_hash)
        or summary.get("formal_decision") != parent["required_decision"]
        or not bool(summary.get("exact_replay"))
        or replay.get("gate", {}).get("decision") != parent["required_decision"]
        or not bool(replay.get("exact_replay"))
    ):
        raise ValueError("Gate 3C1D failure did not authorize data renewal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/routeD_temporal_identity_top1_data_gate3c2_v0.yaml",
    )
    parser.add_argument("--renewal-manifest", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C2 data config")
    validate_parent(config)
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C2 implementation hash drift: {name}")

    excluded_path = Path(config["exclusion"]["manifest"]).resolve()
    if file_sha256(excluded_path) != config["exclusion"]["manifest_sha256"]:
        raise ValueError("Gate 3C2 exclusion manifest hash drift")
    dataset_info = Path(config["source"]["dataset_info"]).resolve()
    if file_sha256(dataset_info) != config["source"]["dataset_info_sha256"]:
        raise ValueError("Gate 3C2 dataset_info hash drift")

    renewal_path = Path(
        args.renewal_manifest
        or Path(config["materialization"]["output_dir"]) / "train.index.json"
    ).resolve()
    renewal = json.loads(renewal_path.read_text())
    if renewal.get("schema_version") != MATERIALIZER_SCHEMA:
        raise ValueError("Gate 3C2 renewal manifest schema drift")
    excluded = raw_identities(excluded_path)
    selected = raw_identities(renewal_path)
    overlap = sorted(set(excluded) & set(selected))

    source_hash_checks = []
    for row in renewal.get("source_files", []):
        if int(row.get("records_selected", 0)) <= 0:
            continue
        expected = row.get("sha256")
        actual = file_sha256(row["path"])
        source_hash_checks.append(
            {
                "path": row["path"],
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exact": isinstance(expected, str) and expected == actual,
            }
        )

    expected_samples = int(config["materialization"]["max_samples"])
    expected_shards = expected_samples // int(config["materialization"]["shard_size"])
    partitions = {
        name: list(range(int(bounds[0]), int(bounds[1]) + 1))
        for name, bounds in config["partition"].items()
    }
    union = (
        partitions["checkpoint_selection_v1"]
        + partitions["fit_only_internal_audit_v1"]
        + partitions["model_validation_v1"]
    )
    exclusion_authority = renewal.get("exclude_manifests", [])
    expected_selection = config["expected_selection"]
    expected_first = tuple(expected_selection["first_identity"])
    expected_last = tuple(expected_selection["last_identity"])
    expected_partition_digests = expected_selection["partition_identity_digests"]
    checks = {
        "sample_count_exact": len(selected) == expected_samples == int(renewal.get("num_samples", -1)),
        "shard_count_exact": len(renewal.get("shards", [])) == expected_shards,
        "selected_identities_unique": len(selected) == len(set(selected)),
        "raw_identity_overlap_empty": not overlap,
        "excluded_identity_count_exact": int(renewal.get("excluded_identity_count", -1)) == len(excluded),
        "excluded_identity_digest_exact": renewal.get("excluded_identity_digest")
        == canonical_json_sha256(sorted(excluded)),
        "selected_identity_digest_exact": renewal.get("selected_identity_digest")
        == canonical_json_sha256(selected)
        == expected_selection["selected_identity_digest"],
        "selected_first_identity_exact": bool(selected) and selected[0] == expected_first,
        "selected_last_identity_exact": bool(selected) and selected[-1] == expected_last,
        "exclusion_manifest_authority_exact": len(exclusion_authority) == 1
        and exclusion_authority[0].get("path") == str(excluded_path)
        and exclusion_authority[0].get("sha256") == config["exclusion"]["manifest_sha256"],
        "dataset_info_authority_exact": renewal.get("dataset_info") == str(dataset_info)
        and renewal.get("dataset_info_sha256") == config["source"]["dataset_info_sha256"],
        "available_raw_records_exact": int(renewal.get("available_raw_records", -1))
        == int(config["source"]["available_raw_records"]),
        "all_selected_source_hashes_exact": bool(source_hash_checks)
        and all(row["exact"] for row in source_hash_checks),
        "partition_union_exact": union == list(range(expected_samples)),
        "partition_membership_disjoint": len(union) == len(set(union)),
        "partition_identity_digests_exact": all(
            canonical_json_sha256([selected[index] for index in values])
            == expected_partition_digests[name]
            for name, values in partitions.items()
        ),
        "locked_data_unread": all(value is False for value in config["locked_data"].values()),
    }
    passed = all(checks.values())
    output = {
        "schema_version": "routeD_temporal_identity_top1_data_gate3c2_v0_summary",
        "date": "2026-07-20",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "formal_decision": config["formal_decisions"][
            "data_gate_pass" if passed else "data_gate_fail"
        ],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "renewal_manifest": str(renewal_path),
        "renewal_manifest_sha256": file_sha256(renewal_path),
        "excluded_manifest": str(excluded_path),
        "excluded_manifest_sha256": file_sha256(excluded_path),
        "excluded_identities": len(excluded),
        "selected_identities": len(selected),
        "raw_identity_overlap": overlap,
        "selected_identity_digest": canonical_json_sha256(selected),
        "partitions": {
            name: {
                "start": values[0],
                "end": values[-1],
                "videos": len(values),
                "raw_identity_digest": canonical_json_sha256(
                    [selected[index] for index in values]
                ),
            }
            for name, values in partitions.items()
        },
        "checks": checks,
        "source_hash_checks": source_hash_checks,
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    output["summary_payload_sha256"] = canonical_json_sha256(output)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
