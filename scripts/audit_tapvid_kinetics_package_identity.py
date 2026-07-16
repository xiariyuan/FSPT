#!/usr/bin/env python3
"""Verify generated Kinetics shards against the official TAP-Vid CSV."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_string(key: Tuple[str, int, int]) -> str:
    return f"{key[0]}_{key[1]:06d}_{key[2]:06d}"


def load_csv_groups(csv_path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    current_key = None
    current_points: List[np.ndarray] = []
    current_occ: List[np.ndarray] = []

    def flush() -> None:
        nonlocal current_key, current_points, current_occ
        if current_key is None:
            return
        groups.append(
            {
                "key": current_key,
                "points": np.stack(current_points, axis=0),
                "occluded": np.stack(current_occ, axis=0),
            }
        )
        current_points = []
        current_occ = []

    with csv_path.open(newline="") as handle:
        for row_index, row in enumerate(csv.reader(handle)):
            if len(row) != 753:
                raise ValueError(f"CSV row {row_index} has {len(row)} fields; expected 753")
            key = (row[0], int(row[1]), int(row[2]))
            if current_key is None:
                current_key = key
            elif key != current_key:
                flush()
                current_key = key
            values = np.asarray(row[3:], dtype=np.float64).reshape(250, 3)
            current_points.append(values[:, :2])
            current_occ.append(values[:, 2].astype(bool))
    flush()
    return groups


def first_frame_size(sample: Dict[str, Any]) -> Tuple[int, int]:
    video = sample["video"]
    with Image.open(io.BytesIO(bytes(video[0]))) as image:
        width, height = image.size
    return height, width


def sample_matches_group(
    sample: Dict[str, Any], group: Dict[str, Any], tolerance: float = 1.0e-12
) -> Tuple[bool, float, Tuple[int, int]]:
    points = np.asarray(sample["points"], dtype=np.float64)
    occluded = np.asarray(sample["occluded"], dtype=bool)
    csv_points = np.asarray(group["points"], dtype=np.float64)
    csv_occ = np.asarray(group["occluded"], dtype=bool)
    height, width = first_frame_size(sample)

    if points.ndim != 3 or points.shape[-1] != 2:
        return False, float("inf"), (height, width)
    tracks, frames = points.shape[:2]
    if csv_points.shape[0] != tracks or frames > csv_points.shape[1]:
        return False, float("inf"), (height, width)
    if occluded.shape != (tracks, frames):
        return False, float("inf"), (height, width)
    if not np.array_equal(occluded, csv_occ[:, :frames]):
        return False, float("inf"), (height, width)

    expected = csv_points[:, :frames].copy()
    expected[..., 0] = (expected[..., 0] * width - 0.5) / width
    expected[..., 1] = (expected[..., 1] * height - 0.5) / height
    max_diff = float(np.max(np.abs(points - expected))) if points.size else 0.0
    return max_diff <= tolerance, max_diff, (height, width)


def read_split_map(root: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    mapping: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        counts[split] = len(ids)
        for video_id in ids:
            if video_id in mapping:
                raise ValueError(f"Video ID {video_id} occurs in multiple split files")
            mapping[video_id] = split
    return mapping, counts


def write_partial(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--official-generator-source", required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    csv_path = root / "tapvid_kinetics.csv"
    index_path = root / "train.index.json"
    generator_path = Path(args.official_generator_source).resolve()
    generator_text = generator_path.read_text()
    generator_formula_pass = all(
        token in generator_text
        for token in (
            "(p.x * width - 0.5) / width",
            "(p.y * height - 0.5) / height",
            "math.ceil(len(videos) / FLAGS.num_shards)",
        )
    )
    if not generator_formula_pass:
        raise ValueError("Pinned official generator does not contain the expected formulas")

    print("Loading official CSV groups...", flush=True)
    groups = load_csv_groups(csv_path)
    split_map, split_counts = read_split_map(root)
    index = json.loads(index_path.read_text())
    csv_keys = [group["key"] for group in groups]
    duplicate_keys = [
        key_string(key) for key, count in Counter(csv_keys).items() if count != 1
    ]

    cursor = 0
    matched: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    unmatched_local: List[Dict[str, Any]] = []
    shard_audit: List[Dict[str, Any]] = []
    max_point_diff = 0.0
    global_index = 0

    for shard_index, shard_info in enumerate(index["shards"]):
        shard_path = root / shard_info["path"]
        print(f"Loading shard {shard_index}: {shard_path.name}", flush=True)
        with shard_path.open("rb") as handle:
            payload = pickle.load(handle)
        expected_rows = int(shard_info["num_samples"])
        if len(payload) != expected_rows:
            raise ValueError(
                f"{shard_path.name} has {len(payload)} samples; expected {expected_rows}"
            )

        for local_index, sample in enumerate(payload):
            found = False
            while cursor < len(groups):
                group = groups[cursor]
                ok, diff, frame_size = sample_matches_group(
                    sample, group, tolerance=args.tolerance
                )
                if ok:
                    max_point_diff = max(max_point_diff, diff)
                    key = group["key"]
                    matched.append(
                        {
                            "global_index": global_index,
                            "shard_index": shard_index,
                            "shard_local_index": local_index,
                            "video_key": key_string(key),
                            "youtube_id": key[0],
                            "split": split_map.get(key[0]),
                            "frame_size": list(frame_size),
                            "frames": int(np.asarray(sample["points"]).shape[1]),
                            "tracks": int(np.asarray(sample["points"]).shape[0]),
                            "max_point_abs_diff": diff,
                        }
                    )
                    cursor += 1
                    found = True
                    break
                skipped_key = group["key"]
                missing.append(
                    {
                        "csv_index": cursor,
                        "video_key": key_string(skipped_key),
                        "youtube_id": skipped_key[0],
                        "split": split_map.get(skipped_key[0]),
                    }
                )
                cursor += 1

            if not found:
                unmatched_local.append(
                    {
                        "global_index": global_index,
                        "shard_index": shard_index,
                        "shard_local_index": local_index,
                    }
                )
            global_index += 1

        shard_audit.append(
            {
                "shard_index": shard_index,
                "path": str(shard_path),
                "sha256": sha256(shard_path),
                "samples": len(payload),
            }
        )
        del payload
        gc.collect()
        write_partial(
            output,
            {
                "status": "running",
                "completed_shards": shard_index + 1,
                "matched_samples": len(matched),
                "missing_csv_groups_so_far": len(missing),
                "unmatched_local_samples": unmatched_local,
            },
        )
        print(
            f"Shard {shard_index} complete: matched={len(matched)} missing={len(missing)}",
            flush=True,
        )

    while cursor < len(groups):
        key = groups[cursor]["key"]
        missing.append(
            {
                "csv_index": cursor,
                "video_key": key_string(key),
                "youtube_id": key[0],
                "split": split_map.get(key[0]),
            }
        )
        cursor += 1

    split_label = lambda row: row["split"] or "unlisted_in_auxiliary_split_files"
    matched_split_counts = dict(Counter(split_label(row) for row in matched))
    missing_split_counts = dict(Counter(split_label(row) for row in missing))
    csv_video_ids = {key[0] for key in csv_keys}
    split_video_ids = set(split_map)
    csv_ids_not_in_split_union = sorted(csv_video_ids - split_video_ids)
    split_ids_not_in_csv = sorted(split_video_ids - csv_video_ids)
    identity_pass = bool(
        generator_formula_pass
        and not duplicate_keys
        and int(index["num_samples"]) == len(matched)
        and len(groups) == len(matched) + len(missing)
        and not unmatched_local
        and max_point_diff <= args.tolerance
    )
    matched_key_hash = hashlib.sha256(
        "\n".join(row["video_key"] for row in matched).encode()
    ).hexdigest()
    result = {
        "status": "complete",
        "pass": identity_pass,
        "official_repository": "google-deepmind/tapnet",
        "official_commit": args.official_commit,
        "official_generator_source": str(generator_path),
        "official_generator_sha256": sha256(generator_path),
        "official_generator_formula_pass": generator_formula_pass,
        "dataset_root": str(root),
        "official_csv": str(csv_path),
        "official_csv_sha256": sha256(csv_path),
        "readme_sha256": sha256(root / "README.md"),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "csv_annotation_groups": len(groups),
        "local_materialized_samples": int(index["num_samples"]),
        "matched_samples": len(matched),
        "missing_video_segments": len(missing),
        "unmatched_local_samples": unmatched_local,
        "duplicate_csv_keys": duplicate_keys,
        "maximum_point_abs_difference": max_point_diff,
        "tolerance": args.tolerance,
        "split_file_counts": split_counts,
        "matched_split_counts": matched_split_counts,
        "missing_split_counts": missing_split_counts,
        "csv_ids_equal_split_union": csv_video_ids == split_video_ids,
        "csv_ids_not_in_split_union": csv_ids_not_in_split_union,
        "split_ids_not_in_csv": split_ids_not_in_csv,
        "auxiliary_split_files_are_identity_authority": False,
        "split_metadata_interpretation": (
            "The train/val/test text files are retained as auxiliary release "
            "metadata but are not an exact membership index for this release CSV."
        ),
        "matched_video_key_sha256": matched_key_hash,
        "matched": matched,
        "missing": missing,
        "shards": shard_audit,
        "claim_boundary": (
            f"The package is an exact order-preserving materialization of "
            f"{len(matched):,} of the {len(groups):,} uniquely annotated video "
            f"segments in the byte-verified official TAP-Vid-Kinetics release CSV; "
            f"{len(missing):,} CSV segments were skipped by materialization. Report "
            "this exact release/materialization scope rather than a universally "
            "fixed 1,000-video set or an official train-only split."
        ),
    }
    write_partial(output, result)
    print(json.dumps({key: result[key] for key in (
        "pass",
        "csv_annotation_groups",
        "local_materialized_samples",
        "matched_samples",
        "missing_video_segments",
        "maximum_point_abs_difference",
        "split_file_counts",
        "matched_split_counts",
        "missing_split_counts",
        "claim_boundary",
    )}, indent=2))
    if not identity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
