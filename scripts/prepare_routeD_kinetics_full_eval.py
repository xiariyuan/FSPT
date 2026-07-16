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
    parser.add_argument("--metric-implementation", required=True)
    parser.add_argument("--metric-parity-audit", required=True)
    parser.add_argument("--package-identity-audit", required=True)
    parser.add_argument("--official-release-audit", required=True)
    parser.add_argument("--official-metric-source", required=True)
    parser.add_argument("--official-generator-source", required=True)
    parser.add_argument("--supersession-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--protocol-output", default="full1144.officialscale.protocol.json"
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source_root = Path(args.source_root).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    hash_protocol_path = Path(args.source_hash_protocol).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    controller = Path(args.controller).resolve()
    config = Path(args.config).resolve()
    metric_implementation = Path(args.metric_implementation).resolve()
    metric_parity_path = Path(args.metric_parity_audit).resolve()
    package_identity_path = Path(args.package_identity_audit).resolve()
    official_release_path = Path(args.official_release_audit).resolve()
    official_metric_source = Path(args.official_metric_source).resolve()
    official_generator_source = Path(args.official_generator_source).resolve()
    supersession_manifest = Path(args.supersession_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dirty = git_value(repo, "status", "--porcelain")
    if dirty:
        raise ValueError(
            "Corrected-metric protocol must be created from a clean working tree"
        )

    metric_parity = json.loads(metric_parity_path.read_text())
    package_identity = json.loads(package_identity_path.read_text())
    official_release = json.loads(official_release_path.read_text())
    supersession = json.loads(supersession_manifest.read_text())
    if not bool(metric_parity.get("pass")):
        raise ValueError("Official metric parity audit did not pass")
    if not bool(package_identity.get("pass")):
        raise ValueError("Kinetics package identity audit did not pass")
    if not bool(official_release.get("pass")):
        raise ValueError("Official release package audit did not pass")
    if not bool(supersession.get("superseded")):
        raise ValueError("Supersession manifest is not marked superseded")
    official_commit = str(metric_parity.get("official_commit", ""))
    if not official_commit or official_commit != str(
        package_identity.get("official_commit", "")
    ):
        raise ValueError("Official source commit mismatch between protocol audits")
    if sha256(official_metric_source) != metric_parity["official_source_sha256"]:
        raise ValueError("Official metric source hash mismatch")
    if sha256(official_generator_source) != package_identity[
        "official_generator_sha256"
    ]:
        raise ValueError("Official generator source hash mismatch")

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
        "kind": "routeD_full_tapvid_kinetics_officialscale_protocol",
        "created_before_any_full_dataset_inference": False,
        "created_before_any_corrected_metric_inference": True,
        "protocol_correction": (
            "The previous full-package run used normalized coordinates multiplied "
            "by width-1/height-1. The pinned official reader multiplies by full "
            "width/height. This protocol freezes the independently verified metric "
            "correction before any corrected-metric inference; tracker, controller, "
            "policy, data order, query mode, and primary decision rule remain unchanged."
        ),
        "created_date": "2026-07-16",
        "dataset_scope": (
            "All 1,144 locally materialized video segments matched exactly and in "
            "order to the official 1,189-segment TAP-Vid-Kinetics annotation CSV"
        ),
        "claim_boundary": package_identity["claim_boundary"],
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
            "metric_implementation": {
                "path": str(metric_implementation),
                "sha256": sha256(metric_implementation),
            },
        },
        "official_protocol_evidence": {
            "official_repository": "google-deepmind/tapnet",
            "official_commit": official_commit,
            "metric_source": {
                "path": str(official_metric_source),
                "sha256": sha256(official_metric_source),
            },
            "generator_source": {
                "path": str(official_generator_source),
                "sha256": sha256(official_generator_source),
            },
            "metric_parity_audit": {
                "path": str(metric_parity_path),
                "sha256": sha256(metric_parity_path),
                "pass": True,
                "coordinate_contract": metric_parity["coordinate_contract"],
            },
            "package_identity_audit": {
                "path": str(package_identity_path),
                "sha256": sha256(package_identity_path),
                "pass": True,
                "official_csv_sha256": package_identity["official_csv_sha256"],
                "matched_samples": package_identity["matched_samples"],
                "missing_video_segments": package_identity["missing_video_segments"],
            },
            "official_release_audit": {
                "path": str(official_release_path),
                "sha256": sha256(official_release_path),
                "pass": True,
                "official_zip_sha256": official_release["official_zip_sha256"],
            },
            "supersession_manifest": {
                "path": str(supersession_manifest),
                "sha256": sha256(supersession_manifest),
            },
        },
        "evaluation": {
            "dataset": "tapvid_kinetics",
            "dataset_split": "train",
            "dataset_split_semantics": (
                "The value 'train' is the local sharded-manifest loader label. "
                "It is not an official TAP-Vid annotation split restriction."
            ),
            "official_annotation_scope": "full byte-verified release CSV",
            "official_annotation_groups": package_identity[
                "csv_annotation_groups"
            ],
            "official_annotation_materialized_groups": package_identity[
                "matched_samples"
            ],
            "official_annotation_missing_groups": package_identity[
                "missing_video_segments"
            ],
            "auxiliary_split_files_are_identity_authority": False,
            "auxiliary_split_file_counts": package_identity["split_file_counts"],
            "auxiliary_split_metadata_discrepancy": {
                "csv_ids_not_in_split_union": len(
                    package_identity["csv_ids_not_in_split_union"]
                ),
                "split_ids_not_in_csv": len(
                    package_identity["split_ids_not_in_csv"]
                ),
            },
            "query_mode": "first",
            "input_resolution": [256, 256],
            "metric_resolution": [256, 256],
            "normalized_to_raster_contract": "x * width, y * height",
            "closed_loop": True,
            "independent_baseline": True,
            "seed": int(args.seed),
            "controller_or_policy_tuning_after_start": False,
            "controller_or_policy_tuning_after_superseded_run": False,
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
