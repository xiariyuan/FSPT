#!/usr/bin/env python3
"""Reclassify a completed raw identity scan after metadata-assumption review."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_tapvid_kinetics_package_identity import (
    key_string,
    load_csv_groups,
    read_split_map,
    sha256,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scan", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--official-generator-source", required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    raw_path = Path(args.raw_scan).resolve()
    root = Path(args.dataset_root).resolve()
    generator_path = Path(args.official_generator_source).resolve()
    csv_path = root / "tapvid_kinetics.csv"
    index_path = root / "train.index.json"
    raw = json.loads(raw_path.read_text())
    index = json.loads(index_path.read_text())

    require(raw.get("status") == "complete", "Raw scan is not complete")
    require(raw.get("official_commit") == args.official_commit, "Commit mismatch")
    require(
        raw.get("official_generator_sha256") == sha256(generator_path),
        "Generator hash mismatch",
    )
    require(raw.get("official_csv_sha256") == sha256(csv_path), "CSV hash mismatch")
    require(raw.get("index_sha256") == sha256(index_path), "Index hash mismatch")
    require(not raw.get("unmatched_local_samples"), "Raw scan has unmatched samples")
    require(not raw.get("duplicate_csv_keys"), "Raw scan has duplicate CSV keys")
    require(
        float(raw.get("maximum_point_abs_difference", float("inf")))
        <= args.tolerance,
        "Raw scan point difference exceeds tolerance",
    )

    raw_shards = {int(row["shard_index"]): row for row in raw.get("shards", [])}
    require(len(raw_shards) == len(index["shards"]), "Raw shard count mismatch")
    for shard_index, shard_info in enumerate(index["shards"]):
        shard_path = root / shard_info["path"]
        raw_entry = raw_shards[shard_index]
        require(raw_entry["path"] == str(shard_path), f"Shard {shard_index} path mismatch")
        require(
            raw_entry["sha256"] == sha256(shard_path),
            f"Shard {shard_index} hash mismatch",
        )
        require(
            int(raw_entry["samples"]) == int(shard_info["num_samples"]),
            f"Shard {shard_index} sample mismatch",
        )

    groups = load_csv_groups(csv_path)
    csv_keys = [key_string(group["key"]) for group in groups]
    matched = raw["matched"]
    missing = raw["missing"]
    require(len(matched) == int(index["num_samples"]), "Matched count mismatch")
    require(len(groups) == len(matched) + len(missing), "CSV partition mismatch")
    missing_by_index = {int(row["csv_index"]): row for row in missing}
    require(len(missing_by_index) == len(missing), "Duplicate missing CSV indices")
    matched_cursor = 0
    for csv_index, csv_key in enumerate(csv_keys):
        if csv_index in missing_by_index:
            require(
                missing_by_index[csv_index]["video_key"] == csv_key,
                f"Missing key mismatch at CSV index {csv_index}",
            )
        else:
            require(matched_cursor < len(matched), "Matched sequence ended early")
            require(
                matched[matched_cursor]["video_key"] == csv_key,
                f"Matched order mismatch at CSV index {csv_index}",
            )
            matched_cursor += 1
    require(matched_cursor == len(matched), "Unused matched rows remain")

    split_map, split_counts = read_split_map(root)
    csv_video_ids = {group["key"][0] for group in groups}
    split_video_ids = set(split_map)
    split_label = lambda row: (
        row.get("split") or "unlisted_in_auxiliary_split_files"
    )
    matched_split_counts = dict(Counter(split_label(row) for row in matched))
    missing_split_counts = dict(Counter(split_label(row) for row in missing))
    matched_key_hash = hashlib.sha256(
        "\n".join(row["video_key"] for row in matched).encode()
    ).hexdigest()
    require(
        matched_key_hash == raw.get("matched_video_key_sha256"),
        "Matched key-sequence hash mismatch",
    )

    result = dict(raw)
    result.update(
        {
            "pass": True,
            "raw_scan": str(raw_path),
            "raw_scan_sha256": sha256(raw_path),
            "classification_revision": (
                "The byte-verified release CSV is the annotation identity authority. "
                "The bundled train/val/test text files are auxiliary metadata and "
                "are not an exact membership index for the CSV."
            ),
            "csv_annotation_groups": len(groups),
            "local_materialized_samples": len(matched),
            "matched_samples": len(matched),
            "missing_video_segments": len(missing),
            "split_file_counts": split_counts,
            "matched_split_counts": matched_split_counts,
            "missing_split_counts": missing_split_counts,
            "csv_ids_equal_split_union": csv_video_ids == split_video_ids,
            "csv_ids_not_in_split_union": sorted(csv_video_ids - split_video_ids),
            "split_ids_not_in_csv": sorted(split_video_ids - csv_video_ids),
            "auxiliary_split_files_are_identity_authority": False,
            "claim_boundary": (
                f"The package is an exact order-preserving materialization of "
                f"{len(matched):,} of the {len(groups):,} uniquely annotated video "
                f"segments in the byte-verified official TAP-Vid-Kinetics release "
                f"CSV; {len(missing):,} CSV segments were skipped by materialization. "
                "Report this exact release/materialization scope rather than a "
                "universally fixed 1,000-video set or an official train-only split."
            ),
        }
    )
    output = Path(args.output).resolve()
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "pass",
                    "csv_annotation_groups",
                    "local_materialized_samples",
                    "matched_samples",
                    "missing_video_segments",
                    "maximum_point_abs_difference",
                    "split_file_counts",
                    "matched_split_counts",
                    "missing_split_counts",
                    "csv_ids_equal_split_union",
                    "claim_boundary",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
