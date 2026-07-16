#!/usr/bin/env python3
"""Build a deterministic, shard-balanced TAP-Vid-Kinetics subset.

The selection protocol is written before any source shard samples are loaded.
Each source shard contributes the same number of samples, chosen with one
seeded draw from each equal-width position bin. The output records source
hashes, source positions, and original video names for exact provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_positions(count: int, samples_per_shard: int, seed: int):
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    if count < samples_per_shard:
        raise ValueError(
            f"shard has {count} samples but {samples_per_shard} were requested"
        )
    rng = random.Random(int(seed))
    positions = []
    for bin_index in range(samples_per_shard):
        start = (bin_index * count) // samples_per_shard
        end = ((bin_index + 1) * count) // samples_per_shard
        if end <= start:
            end = start + 1
        positions.append(rng.randrange(start, end))
    if len(set(positions)) != len(positions):
        raise RuntimeError("stratified selection unexpectedly produced duplicates")
    return positions


def sample_at(container: Any, position: int):
    if isinstance(container, list):
        return container[position], str(position)
    if isinstance(container, dict):
        keys = list(container.keys())
        key = keys[position]
        return container[key], str(key)
    raise TypeError(f"unsupported shard payload type: {type(container)}")


def original_video_name(sample: Any, shard_index: int, position: int):
    if isinstance(sample, dict):
        value = sample.get("video_name") or sample.get("name") or sample.get("id")
        if value is not None and str(value).strip():
            return str(value)
    return f"kinetics_source_s{shard_index:03d}_{position:06d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", default="balanced50.index.json")
    parser.add_argument("--protocol-output", default="balanced50.protocol.json")
    parser.add_argument("--samples-per-shard", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    source_manifest_path = Path(args.source_manifest)
    if not source_manifest_path.is_absolute():
        source_manifest_path = source_root / source_manifest_path
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_root / args.output_manifest
    protocol_path = output_root / args.protocol_output

    source_manifest = json.loads(source_manifest_path.read_text())
    raw_shards = source_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("source manifest must contain a non-empty shards list")

    selection_rows = []
    global_offset = 0
    for shard_index, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            raise ValueError("all source shard entries must be objects")
        relative_path = item.get("path") or item.get("file")
        count = int(item.get("num_samples", 0) or 0)
        if not relative_path or count <= 0:
            raise ValueError(f"invalid source shard entry: {item}")
        positions = selected_positions(
            count,
            args.samples_per_shard,
            args.seed + 1009 * shard_index,
        )
        selection_rows.append(
            {
                "source_shard_index": shard_index,
                "source_path": str(relative_path),
                "source_num_samples": count,
                "source_global_offset": global_offset,
                "selected_positions": positions,
                "selected_global_indices": [global_offset + p for p in positions],
            }
        )
        global_offset += count

    protocol = {
        "kind": "tapvid_kinetics_shard_balanced_subset_protocol",
        "created_before_loading_source_samples": True,
        "source_root": str(source_root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "seed": int(args.seed),
        "samples_per_shard": int(args.samples_per_shard),
        "source_shard_count": len(selection_rows),
        "selected_sample_count": len(selection_rows) * args.samples_per_shard,
        "selection_rule": (
            "For each source shard, partition source positions into equal-width "
            "bins and make one fixed-seed uniform draw per bin."
        ),
        "selection": selection_rows,
        "no_metric_or_label_based_selection": True,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2))

    output_shards = []
    provenance = []
    for row in selection_rows:
        shard_index = int(row["source_shard_index"])
        source_path = Path(row["source_path"])
        if not source_path.is_absolute():
            source_path = source_root / source_path
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        source_hash = sha256_file(source_path)
        with source_path.open("rb") as handle:
            payload = pickle.load(handle)

        selected_samples = []
        for output_position, source_position in enumerate(row["selected_positions"]):
            sample, source_key = sample_at(payload, int(source_position))
            if not isinstance(sample, dict):
                raise TypeError(
                    f"selected source sample must be a dict, got {type(sample)}"
                )
            copied = dict(sample)
            source_name = original_video_name(sample, shard_index, int(source_position))
            stable_name = (
                f"kinetics_balanced_s{shard_index:02d}_"
                f"p{int(source_position):04d}_{source_name}"
            )
            copied["video_name"] = stable_name
            copied["_routeD_source_shard_index"] = shard_index
            copied["_routeD_source_position"] = int(source_position)
            copied["_routeD_source_key"] = source_key
            copied["_routeD_source_video_name"] = source_name
            selected_samples.append(copied)
            provenance.append(
                {
                    "output_shard_index": shard_index,
                    "output_position": output_position,
                    "video_name": stable_name,
                    "source_shard_index": shard_index,
                    "source_path": str(source_path),
                    "source_position": int(source_position),
                    "source_key": source_key,
                    "source_global_index": int(row["source_global_offset"])
                    + int(source_position),
                    "source_video_name": source_name,
                }
            )

        output_name = f"balanced_{shard_index:02d}_of_{len(selection_rows):02d}.pkl"
        output_path = output_root / output_name
        with output_path.open("wb") as handle:
            pickle.dump(selected_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        output_shards.append(
            {
                "path": output_name,
                "num_samples": len(selected_samples),
                "sha256": sha256_file(output_path),
                "source_path": str(source_path),
                "source_sha256": source_hash,
                "source_positions": list(row["selected_positions"]),
            }
        )
        del payload
        del selected_samples

    output_manifest = {
        "split": "routeD_frozen_balanced_evaluation",
        "num_samples": len(provenance),
        "shards": output_shards,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "selection_protocol": str(protocol_path),
        "selection_seed": int(args.seed),
        "samples_per_source_shard": int(args.samples_per_shard),
        "provenance": provenance,
    }
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2))

    protocol["materialization_completed"] = True
    protocol["output_root"] = str(output_root)
    protocol["output_manifest"] = str(output_manifest_path)
    protocol["output_manifest_sha256"] = sha256_file(output_manifest_path)
    protocol["source_shards"] = [
        {
            "source_shard_index": index,
            "path": shard["source_path"],
            "sha256": shard["source_sha256"],
            "selected_positions": shard["source_positions"],
        }
        for index, shard in enumerate(output_shards)
    ]
    protocol_path.write_text(json.dumps(protocol, indent=2))

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "output_manifest": str(output_manifest_path),
                "protocol": str(protocol_path),
                "selected_samples": len(provenance),
                "source_shards": len(selection_rows),
                "output_shards": output_shards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
