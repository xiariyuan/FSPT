#!/usr/bin/env python3
"""Materialize Kubric records while excluding raw identities from frozen manifests."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256
from scripts.preprocess_kubric_tfrecord_no_tf import convert_record, iter_tfrecord, sha256_file

SCHEMA = "routeD_kubric_excluding_manifest_materializer_v1"


def _manifest_samples(path: Path) -> Iterable[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    for entry in manifest["shards"]:
        shard = Path(entry["path"])
        if not shard.is_absolute():
            shard = path.parent / shard
        if file_sha256(shard) != entry["sha256"]:
            raise ValueError(f"excluded manifest shard hash drift: {shard}")
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        values = list(payload.values()) if isinstance(payload, dict) else payload
        if len(values) != int(entry["num_samples"]):
            raise ValueError("excluded manifest shard count drift")
        yield from values


def load_excluded_identities(paths: list[Path]) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    identities: set[tuple[str, int]] = set()
    authority = []
    for path in paths:
        resolved = path.resolve()
        before = len(identities)
        for sample in _manifest_samples(resolved):
            identity = (str(sample["source_tfrecord"]), int(sample["source_record_index"]))
            if identity in identities:
                raise ValueError(f"duplicate excluded raw identity: {identity}")
            identities.add(identity)
        authority.append(
            {
                "path": str(resolved),
                "sha256": file_sha256(resolved),
                "identities_added": len(identities) - before,
            }
        )
    return identities, authority


def split_record_lengths(dataset_info: Path, split: str) -> list[int]:
    payload = json.loads(dataset_info.read_text())
    for row in payload.get("splits", []):
        if row.get("name") == split:
            return [int(value) for value in row.get("shardLengths", [])]
    raise ValueError(f"split {split!r} absent from dataset_info")


def plan_selected_identities(
    source_files: list[Path],
    record_lengths: list[int],
    excluded: set[tuple[str, int]],
    max_samples: int,
) -> list[tuple[str, int]]:
    if len(source_files) != len(record_lengths):
        raise ValueError("source files and record lengths disagree")
    selected: list[tuple[str, int]] = []
    for source_path, record_count in zip(source_files, record_lengths):
        for record_index in range(int(record_count)):
            identity = (source_path.name, int(record_index))
            if identity in excluded:
                continue
            selected.append(identity)
            if len(selected) == int(max_samples):
                return selected
    raise ValueError("insufficient non-excluded raw records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfds-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--sampling-strategy", default="uniform")
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hash-source-files", action="store_true")
    parser.add_argument("--exclude-manifest", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tfds_root = Path(args.tfds_root).resolve()
    builder_dir = tfds_root / "256x256" / "1.0.0"
    if not builder_dir.exists():
        builder_dir = tfds_root
    dataset_info = builder_dir / "dataset_info.json"
    if not dataset_info.is_file():
        raise FileNotFoundError(f"dataset_info absent: {dataset_info}")
    source_files = sorted(builder_dir.glob(f"movi_e-{args.split}.tfrecord-*"))
    lengths = split_record_lengths(dataset_info, args.split)
    if not source_files or len(source_files) != len(lengths):
        raise ValueError("TFRecord shard count disagrees with dataset_info")

    excluded, exclusion_authority = load_excluded_identities(
        [Path(value) for value in args.exclude_manifest]
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory must be empty before materialization")

    max_samples = int(args.max_samples)
    planned_identities = plan_selected_identities(
        source_files, lengths, excluded, max_samples
    )
    shard_size = int(args.shard_size)
    shard_items: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    selected_identities: list[tuple[str, int]] = []
    processed = 0
    output_shard_index = 0
    fully_excluded_source_files = 0

    def flush_shard() -> None:
        nonlocal shard_items, output_shard_index
        if not shard_items:
            return
        name = f"{args.split}_{output_shard_index:05d}.pkl"
        path = output_dir / name
        with path.open("wb") as handle:
            pickle.dump(shard_items, handle, protocol=4)
        shard_manifest.append(
            {
                "path": name,
                "num_samples": len(shard_items),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        shard_items = []
        output_shard_index += 1

    for source_path, record_count in zip(source_files, lengths):
        if processed >= max_samples:
            break
        excluded_indices = {
            index for name, index in excluded if name == source_path.name
        }
        if excluded_indices == set(range(record_count)):
            fully_excluded_source_files += 1
            continue
        selected_here = 0
        skipped_here = 0
        for record_index, serialized in enumerate(iter_tfrecord(source_path)):
            identity = (source_path.name, int(record_index))
            if identity in excluded:
                skipped_here += 1
                continue
            if processed >= max_samples:
                break
            sample_seed = (int(args.seed) + processed * 1013) % (2**32)
            converted = convert_record(
                serialized,
                num_points=int(args.num_points),
                sampling_strategy=str(args.sampling_strategy),
                hard_fraction=float(args.hard_fraction),
                seed=sample_seed,
            )
            converted["source_tfrecord"] = source_path.name
            converted["source_record_index"] = int(record_index)
            shard_items.append(converted)
            selected_identities.append(identity)
            selected_here += 1
            processed += 1
            print(
                json.dumps(
                    {
                        "stage": "sample_complete",
                        "selected_index": processed - 1,
                        "source_tfrecord": source_path.name,
                        "source_record_index": record_index,
                        "video_name": converted["video_name"],
                    }
                ),
                flush=True,
            )
            if len(shard_items) >= shard_size:
                flush_shard()
        if selected_here or skipped_here:
            row = {
                "path": str(source_path),
                "size_bytes": source_path.stat().st_size,
                "dataset_info_records": int(record_count),
                "records_selected": int(selected_here),
                "records_skipped_excluded": int(skipped_here),
            }
            if args.hash_source_files and selected_here:
                row["sha256"] = sha256_file(source_path)
            source_stats[source_path.name] = row

    flush_shard()
    if processed != max_samples:
        raise RuntimeError(f"requested {max_samples} samples, materialized {processed}")
    if selected_identities != planned_identities:
        raise RuntimeError("materialized raw identity order differs from frozen plan")
    if len(selected_identities) != len(set(selected_identities)):
        raise RuntimeError("selected raw identities are not unique")
    overlap = set(selected_identities) & excluded
    if overlap:
        raise RuntimeError(f"selected identities overlap exclusion set: {sorted(overlap)[:3]}")

    manifest = {
        "schema_version": SCHEMA,
        "version": 2,
        "dataset": "tapvid_kubric",
        "source": "movi_e_tfrecord_without_tensorflow_excluding_manifest",
        "split": str(args.split),
        "num_samples": int(processed),
        "num_shards": len(shard_manifest),
        "num_points": int(args.num_points),
        "num_frames": 24,
        "resolution": [256, 256],
        "sampling_strategy": str(args.sampling_strategy),
        "hard_fraction": float(args.hard_fraction),
        "seed": int(args.seed),
        "dataset_info": str(dataset_info),
        "dataset_info_sha256": file_sha256(dataset_info),
        "available_raw_records": int(sum(lengths)),
        "exclude_manifests": exclusion_authority,
        "excluded_identity_count": len(excluded),
        "excluded_identity_digest": canonical_json_sha256(sorted(excluded)),
        "selected_identity_digest": canonical_json_sha256(selected_identities),
        "planned_identity_digest": canonical_json_sha256(planned_identities),
        "fully_excluded_source_files_skipped_without_read": fully_excluded_source_files,
        "source_files": list(source_stats.values()),
        "shards": shard_manifest,
    }
    manifest_path = output_dir / f"{args.split}.index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "samples": processed,
                "shards": len(shard_manifest),
                "selected_identity_digest": manifest["selected_identity_digest"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
