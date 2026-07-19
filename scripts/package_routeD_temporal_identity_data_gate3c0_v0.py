#!/usr/bin/env python3
"""Qualify the expanded raw-record-disjoint Kubric Gate 3C0 data protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)


SCHEMA = "routeD_temporal_identity_data_gate3c0_v0"


def _manifest_samples(path: Path) -> Iterator[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    for entry in manifest["shards"]:
        shard = Path(entry["path"])
        if not shard.is_absolute():
            shard = path.parent / shard
        if file_sha256(shard) != entry["sha256"]:
            raise ValueError(f"Gate 3C0 output shard hash drift: {shard}")
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        values = list(payload.values()) if isinstance(payload, dict) else payload
        if len(values) != int(entry["num_samples"]):
            raise ValueError("Gate 3C0 output shard sample-count drift")
        yield from values


def _array_sha256(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _sample_identity(sample: dict[str, Any]) -> tuple[str, int]:
    return str(sample["source_tfrecord"]), int(sample["source_record_index"])


def _sample_digest(sample: dict[str, Any]) -> str:
    value = {}
    for key in sorted(sample):
        item = sample[key]
        if isinstance(item, (np.ndarray, list, tuple)):
            value[key] = _array_sha256(item)
        elif isinstance(item, np.generic):
            value[key] = item.item()
        elif item is None or isinstance(item, (str, int, float, bool)):
            value[key] = item
        else:
            raise TypeError(f"unsupported Gate 3C0 sample field {key}: {type(item)}")
    return canonical_json_sha256(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/routeD_temporal_identity_data_gate3c0_v0.yaml",
    )
    parser.add_argument("--expanded-manifest", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C0 config")
    expanded_path = Path(
        args.expanded_manifest
        or (
            Path(config["materialization"]["output_dir"])
            / f"{config['source']['split']}.index.json"
        )
    ).resolve()
    existing_path = Path(
        config["overlap_contract"]["frozen_existing_manifest"]
    ).resolve()
    if file_sha256(existing_path) != config["overlap_contract"][
        "frozen_existing_manifest_sha256"
    ]:
        raise ValueError("Gate 3C0 existing manifest hash drift")
    expanded = json.loads(expanded_path.read_text())
    expected_samples = int(config["materialization"]["max_samples"])
    expected_shards = expected_samples // int(config["materialization"]["shard_size"])
    source_hash_checks = []
    for entry in expanded["source_files"]:
        path = Path(entry["path"])
        expected = entry.get("sha256")
        actual = file_sha256(path)
        source_hash_checks.append(
            {"path": str(path), "expected_sha256": expected, "actual_sha256": actual,
             "exact": isinstance(expected, str) and expected == actual}
        )
    existing_samples = list(_manifest_samples(existing_path))
    if len(existing_samples) != 64:
        raise ValueError("Gate 3C0 frozen existing population is not 64")
    existing_identities = [_sample_identity(sample) for sample in existing_samples]
    existing_digests = [_sample_digest(sample) for sample in existing_samples]
    expanded_identities = []
    expanded_digests = []
    first64_exact = []
    for index, sample in enumerate(_manifest_samples(expanded_path)):
        identity = _sample_identity(sample)
        digest = _sample_digest(sample)
        expanded_identities.append(identity)
        expanded_digests.append(digest)
        if index < 64:
            first64_exact.append(
                identity == existing_identities[index]
                and digest == existing_digests[index]
            )
    development_identities = expanded_identities[64:]
    overlap = sorted(set(existing_identities) & set(development_identities))
    partitions = config["partition"]
    ranges = {
        name: list(range(int(bounds[0]), int(bounds[1]) + 1))
        for name, bounds in partitions.items()
        if name in ("gradient_train", "checkpoint_selection", "fit_only_internal_audit")
    }
    partition_union = ranges["gradient_train"] + ranges["checkpoint_selection"] + ranges[
        "fit_only_internal_audit"
    ]
    checks = {
        "expanded_sample_count_exact": len(expanded_identities) == expected_samples
        == int(expanded.get("num_samples", -1)),
        "expanded_shard_count_exact": len(expanded.get("shards", [])) == expected_shards,
        "all_source_hashes_exact": bool(source_hash_checks)
        and all(row["exact"] for row in source_hash_checks),
        "first_64_sample_tensors_exact": len(first64_exact) == 64
        and all(first64_exact),
        "first_64_raw_identities_exact": expanded_identities[:64]
        == existing_identities,
        "development_raw_disjoint_from_existing_64": not overlap,
        "expanded_raw_identities_unique": len(set(expanded_identities))
        == len(expanded_identities),
        "partition_union_exact_64_511": partition_union == list(range(64, 512)),
        "partition_identities_disjoint": len(set(partition_union))
        == len(partition_union),
        "locked_data_unread": all(
            value is False for value in config["locked_data"].values()
        ),
    }
    passed = all(checks.values())
    output = {
        "schema_version": "routeD_temporal_identity_data_gate3c0_v0_summary",
        "date": "2026-07-19",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "formal_decision": config["formal_decisions"][
            "data_gate_pass" if passed else "data_gate_fail"
        ],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expanded_manifest": str(expanded_path),
        "expanded_manifest_sha256": file_sha256(expanded_path),
        "existing_manifest": str(existing_path),
        "existing_manifest_sha256": file_sha256(existing_path),
        "samples": len(expanded_identities),
        "shards": len(expanded.get("shards", [])),
        "source_files": len(expanded.get("source_files", [])),
        "combined_expanded_sample_digest": canonical_json_sha256(expanded_digests),
        "combined_existing_64_sample_digest": canonical_json_sha256(existing_digests),
        "partitions": {
            name: {
                "start": values[0],
                "end": values[-1],
                "videos": len(values),
                "combined_raw_identity_digest": canonical_json_sha256(
                    [expanded_identities[index] for index in values]
                ),
            }
            for name, values in ranges.items()
        },
        "checks": checks,
        "raw_identity_overlap": overlap,
        "source_hash_checks": source_hash_checks,
        "locked_data": config["locked_data"],
    }
    output["summary_payload_sha256"] = canonical_json_sha256(output)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
