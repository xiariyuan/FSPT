#!/usr/bin/env python3
"""Materialize Kubric records while excluding raw identities from frozen manifests."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
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


def _decode_source_task(task: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(task["source_path"])
    selected_rows = {
        int(row["record_index"]): (int(row["selected_index"]), int(row["sample_seed"]))
        for row in task["selected_rows"]
    }
    maximum_record = max(selected_rows)
    output: list[tuple[int, dict[str, Any]]] = []
    for record_index, serialized in enumerate(iter_tfrecord(source_path)):
        if record_index in selected_rows:
            selected_index, sample_seed = selected_rows[record_index]
            converted = convert_record(
                serialized,
                num_points=int(task["num_points"]),
                sampling_strategy=str(task["sampling_strategy"]),
                hard_fraction=float(task["hard_fraction"]),
                seed=sample_seed,
            )
            converted["source_tfrecord"] = source_path.name
            converted["source_record_index"] = int(record_index)
            output.append((selected_index, converted))
        if record_index >= maximum_record and len(output) == len(selected_rows):
            break
    if len(output) != len(selected_rows):
        raise RuntimeError(f"failed to decode all selected records from {source_path.name}")
    output.sort(key=lambda row: row[0])
    stage_path = Path(task["stage_path"])
    with stage_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=4)
    return {
        "source_name": source_path.name,
        "stage_path": str(stage_path),
        "selected_indices": [row[0] for row in output],
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "dataset_info_records": int(task["record_count"]),
            "records_selected": len(output),
            "records_skipped_excluded": int(task["excluded_count"]),
            "sha256": sha256_file(source_path) if bool(task["hash_source_file"]) else None,
        },
    }


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
    parser.add_argument("--workers", type=int, default=1)
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
    planned_identities = plan_selected_identities(source_files, lengths, excluded, max_samples)
    selected_index = {identity: index for index, identity in enumerate(planned_identities)}
    source_lookup = {path.name: (path, lengths[index]) for index, path in enumerate(source_files)}
    source_order: list[str] = []
    rows_by_source: dict[str, list[dict[str, int]]] = {}
    for index, (source_name, record_index) in enumerate(planned_identities):
        if source_name not in rows_by_source:
            rows_by_source[source_name] = []
            source_order.append(source_name)
        rows_by_source[source_name].append(
            {
                "selected_index": index,
                "record_index": record_index,
                "sample_seed": (int(args.seed) + index * 1013) % (2**32),
            }
        )

    stage_dir = output_dir / ".stage"
    stage_dir.mkdir()
    tasks = []
    for task_index, source_name in enumerate(source_order):
        source_path, record_count = source_lookup[source_name]
        tasks.append(
            {
                "source_path": str(source_path),
                "record_count": int(record_count),
                "selected_rows": rows_by_source[source_name],
                "excluded_count": sum(1 for name, _ in excluded if name == source_name),
                "num_points": int(args.num_points),
                "sampling_strategy": str(args.sampling_strategy),
                "hard_fraction": float(args.hard_fraction),
                "hash_source_file": bool(args.hash_source_files),
                "stage_path": str(stage_dir / f"source_{task_index:05d}.pkl"),
            }
        )

    shard_size = int(args.shard_size)
    shard_items: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    selected_identities: list[tuple[str, int]] = []
    source_stats: list[dict[str, Any]] = []
    output_shard_index = 0

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

    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_decode_source_task, task) for task in tasks]
        for task, future in zip(tasks, futures):
            result = future.result()
            stage_path = Path(result["stage_path"])
            with stage_path.open("rb") as handle:
                decoded = pickle.load(handle)
            stage_path.unlink()
            expected_indices = [row["selected_index"] for row in task["selected_rows"]]
            if [row[0] for row in decoded] != expected_indices:
                raise RuntimeError("parallel decoder changed selected sample order")
            source_stats.append(result["source"])
            for global_index, sample in decoded:
                identity = (str(sample["source_tfrecord"]), int(sample["source_record_index"]))
                if selected_index.get(identity) != global_index:
                    raise RuntimeError("parallel decoder changed raw identity membership")
                selected_identities.append(identity)
                shard_items.append(sample)
                print(
                    json.dumps(
                        {
                            "stage": "sample_complete",
                            "selected_index": global_index,
                            "source_tfrecord": identity[0],
                            "source_record_index": identity[1],
                            "video_name": sample["video_name"],
                        }
                    ),
                    flush=True,
                )
                if len(shard_items) >= shard_size:
                    flush_shard()
    flush_shard()
    stage_dir.rmdir()

    if selected_identities != planned_identities:
        raise RuntimeError("materialized raw identity order differs from frozen plan")
    if len(selected_identities) != len(set(selected_identities)):
        raise RuntimeError("selected raw identities are not unique")
    if set(selected_identities) & excluded:
        raise RuntimeError("selected identities overlap exclusion set")

    fully_excluded_source_files = 0
    for source_path, record_count in zip(source_files, lengths):
        if source_path.name == source_order[0]:
            break
        if all((source_path.name, index) in excluded for index in range(record_count)):
            fully_excluded_source_files += 1

    manifest = {
        "schema_version": SCHEMA,
        "version": 2,
        "dataset": "tapvid_kubric",
        "source": "movi_e_tfrecord_without_tensorflow_excluding_manifest",
        "split": str(args.split),
        "num_samples": len(selected_identities),
        "num_shards": len(shard_manifest),
        "num_points": int(args.num_points),
        "num_frames": 24,
        "resolution": [256, 256],
        "sampling_strategy": str(args.sampling_strategy),
        "hard_fraction": float(args.hard_fraction),
        "seed": int(args.seed),
        "workers": workers,
        "dataset_info": str(dataset_info),
        "dataset_info_sha256": file_sha256(dataset_info),
        "available_raw_records": int(sum(lengths)),
        "exclude_manifests": exclusion_authority,
        "excluded_identity_count": len(excluded),
        "excluded_identity_digest": canonical_json_sha256(sorted(excluded)),
        "selected_identity_digest": canonical_json_sha256(selected_identities),
        "planned_identity_digest": canonical_json_sha256(planned_identities),
        "fully_excluded_source_files_skipped_without_read": fully_excluded_source_files,
        "source_files": source_stats,
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
                "samples": len(selected_identities),
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
