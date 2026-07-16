#!/usr/bin/env python3
"""Prepare a predeclared, resumable full TAP-Vid-Kinetics evaluation.

This script does not load video samples or run the tracker. It creates one
single-shard manifest per source shard and records all frozen inputs before the
first inference process is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-hash-protocol", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol-output", default="full1144.protocol.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source_root = Path(args.source_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    hash_protocol_path = Path(args.source_hash_protocol).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    controller = Path(args.controller).resolve()
    config = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(source_manifest.read_text())
    hash_protocol = json.loads(hash_protocol_path.read_text())
    hash_by_index = {
        int(row["source_shard_index"]): row
        for row in hash_protocol["source_shards"]
    }

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Source manifest must contain a non-empty shards list")

    prepared_shards = []
    global_offset = 0
    for shard_index, shard in enumerate(shards):
        relative_path = shard["path"] if isinstance(shard, dict) else shard
        num_samples = int(shard.get("num_samples", 0)) if isinstance(shard, dict) else 0
        source_path = Path(relative_path)
        if not source_path.is_absolute():
            source_path = source_root / source_path
        source_path = source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        prior_hash = hash_by_index.get(shard_index)
        if prior_hash is None:
            raise ValueError(f"Missing prior source hash for shard {shard_index}")
        if Path(prior_hash["path"]).resolve() != source_path:
            raise ValueError(f"Source path mismatch for shard {shard_index}")

        shard_manifest_path = output_root / f"shard_{shard_index:02d}.index.json"
        shard_manifest = {
            "split": manifest.get("split", "train"),
            "num_samples": num_samples,
            "shards": [
                {
                    "path": str(source_path),
                    "num_samples": num_samples,
                }
            ],
        }
        shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2))
        stat = source_path.stat()
        prepared_shards.append(
            {
                "shard_index": shard_index,
                "source_path": str(source_path),
                "source_num_samples": num_samples,
                "global_offset": global_offset,
                "source_sha256": prior_hash["sha256"],
                "source_hash_provenance": str(hash_protocol_path),
                "source_size_bytes": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "evaluation_manifest": str(shard_manifest_path),
                "evaluation_manifest_sha256": sha256(shard_manifest_path),
                "expected_result": str(
                    output_root / f"shard_{shard_index:02d}.result.json"
                ),
            }
        )
        global_offset += num_samples

    declared_total = int(manifest.get("num_samples", global_offset))
    if declared_total != global_offset:
        raise ValueError(
            f"Manifest total {declared_total} does not match shard total {global_offset}"
        )

    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=repo)
    protocol = {
        "kind": "routeD_full_tapvid_kinetics_sharded_protocol",
        "created_before_any_full_dataset_inference": True,
        "created_date": "2026-07-16",
        "dataset_scope": (
            "All 1,144 videos present in the local TAP-Vid-Kinetics "
            "train.index.json package"
        ),
        "claim_boundary": (
            "This is the complete local 1,144-video package. Do not call it an "
            "official benchmark result unless the local package identity and "
            "official evaluation protocol are independently verified."
        ),
        "source_root": str(source_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_hash_protocol": str(hash_protocol_path),
        "source_hash_protocol_sha256": sha256(hash_protocol_path),
        "source_shard_count": len(prepared_shards),
        "expected_video_count": global_offset,
        "git_head": git_value(repo, "rev-parse", "HEAD"),
        "git_branch": git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "frozen_inputs": {
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "controller": {"path": str(controller), "sha256": sha256(controller)},
            "config": {"path": str(config), "sha256": sha256(config)},
        },
        "evaluation": {
            "dataset": "tapvid_kinetics",
            "dataset_split": "train",
            "query_mode": "first",
            "input_resolution": [256, 256],
            "metric_resolution": [256, 256],
            "closed_loop": True,
            "independent_baseline": True,
            "seed": int(args.seed),
            "controller_or_policy_tuning_after_start": False,
            "primary_metrics": ["AJ", "delta_avg"],
            "primary_comparison": "routeD_closed vs independent baseline",
            "primary_success_rule": (
                "Paired-video bootstrap 95% confidence interval lower bounds "
                "must be greater than zero for both AJ and delta_avg."
            ),
        },
        "shards": prepared_shards,
        "merge_rule": (
            "Concatenate all per-video rows in source-shard order, verify the "
            "expected count and unique video names, then compute unweighted "
            "per-video means and paired-video bootstrap intervals."
        ),
    }
    protocol_path = output_root / args.protocol_output
    protocol_path.write_text(json.dumps(protocol, indent=2))
    print(json.dumps({
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "expected_video_count": global_offset,
        "shards": len(prepared_shards),
    }, indent=2))


if __name__ == "__main__":
    main()
