#!/usr/bin/env python3
"""Qualify the third raw-record-disjoint Kubric population for Gate 3C1F1."""
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

SCHEMA = "routeD_temporal_identity_full_population_data_gate3c1f1_v0"
MATERIALIZER_SCHEMA = "routeD_kubric_excluding_manifest_materializer_v1"


def manifest_samples(path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    for entry in manifest["shards"]:
        shard = Path(entry["path"])
        if not shard.is_absolute():
            shard = path.parent / shard
        if file_sha256(shard) != entry["sha256"]:
            raise ValueError(f"Gate 3C1F1 shard hash drift: {shard}")
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        values = list(payload.values()) if isinstance(payload, dict) else payload
        if len(values) != int(entry["num_samples"]):
            raise ValueError("Gate 3C1F1 shard count drift")
        yield from values


def raw_identities(path: Path) -> list[tuple[str, int]]:
    return [
        (str(sample["source_tfrecord"]), int(sample["source_record_index"]))
        for sample in manifest_samples(path)
    ]


def result_payload_sha256(result: Mapping[str, Any]) -> str:
    value = dict(result)
    value.pop("result_payload_sha256", None)
    return canonical_json_sha256(value)


def validate_parent(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["authorized_parent"]
    for key, hash_key in (("result", "result_sha256"), ("replay", "replay_sha256")):
        if file_sha256(parent[key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C1F1 parent hash drift: {key}")
    replay = json.loads(Path(parent["replay"]).read_text())
    if (
        replay.get("result_payload_sha256") != result_payload_sha256(replay)
        or replay.get("result_payload_sha256") != parent["replay_payload_sha256"]
        or not bool(replay.get("exact_replay"))
        or not bool(replay.get("gate", {}).get("pass"))
        or replay.get("gate", {}).get("decision") != parent["required_decision"]
    ):
        raise ValueError("Gate 3C1F0 did not authorize Gate 3C1F1")
    bundle = replay["frozen_bundle_artifact"]
    if (
        bundle["path"] != parent["entry_bundle"]
        or bundle["file_sha256"] != parent["entry_bundle_sha256"]
        or file_sha256(bundle["path"]) != parent["entry_bundle_sha256"]
    ):
        raise ValueError("Gate 3C1F1 entry bundle drift")
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/routeD_temporal_identity_full_population_data_gate3c1f1_v0.yaml",
    )
    parser.add_argument("--renewal-manifest", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1F1 config")
    validate_parent(config)
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1F1 implementation hash drift: {name}")
    dataset_info = Path(config["source"]["dataset_info"])
    if file_sha256(dataset_info) != config["source"]["dataset_info_sha256"]:
        raise ValueError("Gate 3C1F1 dataset_info drift")

    excluded: list[tuple[str, int]] = []
    exclusion_authority = []
    for row in config["exclusions"]:
        path = Path(row["manifest"])
        if file_sha256(path) != row["manifest_sha256"]:
            raise ValueError("Gate 3C1F1 exclusion manifest drift")
        values = raw_identities(path)
        if len(values) != int(row["raw_records"]):
            raise ValueError("Gate 3C1F1 exclusion count drift")
        excluded.extend(values)
        exclusion_authority.append(
            {"path": str(path.resolve()), "sha256": row["manifest_sha256"], "identities_added": len(values)}
        )
    if len(excluded) != len(set(excluded)):
        raise ValueError("Gate 3C1F1 exclusion identities overlap")

    renewal_path = Path(
        args.renewal_manifest
        or Path(config["materialization"]["output_dir"]) / "train.index.json"
    ).resolve()
    renewal = json.loads(renewal_path.read_text())
    if renewal.get("schema_version") != MATERIALIZER_SCHEMA:
        raise ValueError("Gate 3C1F1 materializer schema drift")
    selected = raw_identities(renewal_path)
    expected = config["expected_selection"]
    source_hash_checks = []
    for row in renewal.get("source_files", []):
        if int(row.get("records_selected", 0)) <= 0:
            continue
        actual = file_sha256(row["path"])
        source_hash_checks.append(
            {
                "path": row["path"],
                "expected_sha256": row.get("sha256"),
                "actual_sha256": actual,
                "exact": actual == row.get("sha256"),
            }
        )
    checks = {
        "sample_count_exact": len(selected) == int(config["materialization"]["max_samples"]) == int(renewal.get("num_samples", -1)),
        "shard_count_exact": len(renewal.get("shards", [])) == int(expected["output_shards"]),
        "selected_unique": len(selected) == len(set(selected)),
        "exclusion_unique": len(excluded) == len(set(excluded)),
        "raw_overlap_empty": not (set(selected) & set(excluded)),
        "excluded_count_exact": len(excluded) == int(expected["excluded_raw_records"]) == int(renewal.get("excluded_identity_count", -1)),
        "excluded_digest_exact": canonical_json_sha256(sorted(excluded)) == expected["excluded_identity_digest"] == renewal.get("excluded_identity_digest"),
        "selected_digest_exact": canonical_json_sha256(selected) == expected["selected_identity_digest"] == renewal.get("selected_identity_digest"),
        "first_identity_exact": bool(selected) and list(selected[0]) == expected["first_identity"],
        "last_identity_exact": bool(selected) and list(selected[-1]) == expected["last_identity"],
        "source_file_count_exact": len(source_hash_checks) == int(expected["selected_source_files"]),
        "all_source_hashes_exact": bool(source_hash_checks) and all(row["exact"] for row in source_hash_checks),
        "exclusion_authority_exact": renewal.get("exclude_manifests") == exclusion_authority,
        "dataset_info_exact": renewal.get("dataset_info") == str(dataset_info.resolve()) and renewal.get("dataset_info_sha256") == config["source"]["dataset_info_sha256"],
        "available_raw_records_exact": int(renewal.get("available_raw_records", -1)) == int(config["source"]["available_raw_records"]),
        "locked_data_unread": all(value is False for value in config["locked_data"].values()),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "routeD_temporal_identity_full_population_data_gate3c1f1_v0_summary",
        "date": "2026-07-20",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "formal_decision": config["formal_decisions"]["pass" if passed else "fail"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "renewal_manifest": str(renewal_path),
        "renewal_manifest_sha256": file_sha256(renewal_path),
        "excluded_identities": len(excluded),
        "selected_identities": len(selected),
        "selected_identity_digest": canonical_json_sha256(selected),
        "checks": checks,
        "source_hash_checks": source_hash_checks,
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    result["summary_payload_sha256"] = canonical_json_sha256(result)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
