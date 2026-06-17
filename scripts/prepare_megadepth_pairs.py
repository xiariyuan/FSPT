#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def find_scene_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))


def first_key(data: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in data:
            return key
    return None


def to_list_of_paths(values, base_root: Path, relative_paths: bool) -> List[str]:
    items: List[str] = []
    for value in values.tolist() if hasattr(values, "tolist") else values:
        text = str(value)
        path = Path(text)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        normalized = path.relative_to(base_root) if relative_paths else path
        items.append(normalized.as_posix())
    return items


def maybe_matrix(data: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Optional[np.ndarray]:
    key = first_key(data, keys)
    if key is None:
        return None
    return np.asarray(data[key])


def maybe_list(data: np.lib.npyio.NpzFile, keys: Sequence[str]):
    key = first_key(data, keys)
    if key is None:
        return None
    value = data[key]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def maybe_path(values, index: int, root: Path, relative_paths: bool) -> Optional[str]:
    if values is None or index >= len(values):
        return None
    raw = values[index]
    if raw is None:
        return None
    text = str(raw)
    path = Path(text)
    if not path.is_absolute():
        path = (root / path).resolve()
    normalized = path.relative_to(root) if relative_paths else path
    return normalized.as_posix()


def maybe_value_list(values, index: int):
    if values is None or index >= len(values):
        return None
    raw = values[index]
    if raw is None:
        return None
    return np.asarray(raw).tolist()


def sample_scene_pairs(
    scene_file: Path,
    root: Path,
    min_overlap: float,
    max_overlap: float,
    min_scale_ratio: float,
    max_scale_ratio: float,
    max_pairs_per_scene: int,
    require_depth: bool,
    relative_paths: bool,
    rng: random.Random,
) -> List[dict]:
    data = np.load(scene_file, allow_pickle=True)
    overlap = maybe_matrix(data, ("overlap_matrix", "overlaps", "overlap"))
    image_key = first_key(data, ("image_paths", "images", "image_names", "image_paths_undistorted"))
    pair_infos = maybe_list(data, ("pair_infos", "pairs", "pair_info"))
    if overlap is None and pair_infos is None:
        return []
    if image_key is None:
        return []

    image_paths = to_list_of_paths(np.asarray(data[image_key]), root, relative_paths)
    depth_key = first_key(data, ("depth_paths", "depths", "depth_names"))
    depth_paths = to_list_of_paths(np.asarray(data[depth_key]), root, relative_paths) if depth_key else []
    intrinsics = maybe_matrix(data, ("intrinsics", "K", "Ks", "intrinsics_matrices"))
    poses = maybe_matrix(data, ("poses", "pose", "T_wc", "world_to_cam"))
    scale_ratio = maybe_matrix(data, ("scale_ratio_matrix", "scale_ratios"))

    candidates: List[dict] = []
    if pair_infos is not None:
        for pair_info in pair_infos:
            if pair_info is None or len(pair_info) < 2:
                continue
            pair_index = np.asarray(pair_info[0]).tolist()
            if len(pair_index) != 2:
                continue
            i, j = int(pair_index[0]), int(pair_index[1])
            ov = float(pair_info[1])
            if ov < min_overlap or ov > max_overlap:
                continue
            image0 = maybe_path(np.asarray(data[image_key]).tolist(), i, root, relative_paths)
            image1 = maybe_path(np.asarray(data[image_key]).tolist(), j, root, relative_paths)
            if image0 is None or image1 is None:
                continue
            depth0 = maybe_path(np.asarray(data[depth_key]).tolist(), i, root, relative_paths) if depth_key else None
            depth1 = maybe_path(np.asarray(data[depth_key]).tolist(), j, root, relative_paths) if depth_key else None
            if require_depth and (depth0 is None or depth1 is None):
                continue

            record = {
                "dataset": "megadepth",
                "scene_id": scene_file.stem,
                "image0": image0,
                "image1": image1,
                "overlap": ov,
            }
            if depth0 is not None:
                record["depth0"] = depth0
            if depth1 is not None:
                record["depth1"] = depth1
            intr0 = maybe_value_list(intrinsics, i)
            intr1 = maybe_value_list(intrinsics, j)
            pose0 = maybe_value_list(poses, i)
            pose1 = maybe_value_list(poses, j)
            if intr0 is not None:
                record["intrinsics0"] = intr0
            if intr1 is not None:
                record["intrinsics1"] = intr1
            if pose0 is not None:
                record["pose0"] = pose0
            if pose1 is not None:
                record["pose1"] = pose1
            if len(pair_info) >= 3 and pair_info[2] is not None:
                record["pair_meta"] = np.asarray(pair_info[2]).tolist()
            candidates.append(record)
    else:
        count = len(image_paths)
        for i in range(count):
            for j in range(i + 1, count):
                ov = float(overlap[i, j])
                if ov < min_overlap or ov > max_overlap:
                    continue
                ratio = None
                if scale_ratio is not None:
                    ratio = float(scale_ratio[i, j])
                    if ratio < min_scale_ratio or ratio > max_scale_ratio:
                        continue
                if require_depth and (not depth_paths or i >= len(depth_paths) or j >= len(depth_paths)):
                    continue
                record = {
                    "dataset": "megadepth",
                    "scene_id": scene_file.stem,
                    "image0": image_paths[i],
                    "image1": image_paths[j],
                    "overlap": ov,
                }
                if depth_paths:
                    if i < len(depth_paths):
                        record["depth0"] = depth_paths[i]
                    if j < len(depth_paths):
                        record["depth1"] = depth_paths[j]
                if intrinsics is not None:
                    if i < len(intrinsics):
                        record["intrinsics0"] = np.asarray(intrinsics[i]).tolist()
                    if j < len(intrinsics):
                        record["intrinsics1"] = np.asarray(intrinsics[j]).tolist()
                if poses is not None:
                    if i < len(poses):
                        record["pose0"] = np.asarray(poses[i]).tolist()
                    if j < len(poses):
                        record["pose1"] = np.asarray(poses[j]).tolist()
                if ratio is not None:
                    record["scale_ratio"] = ratio
                candidates.append(record)

    if len(candidates) > max_pairs_per_scene:
        candidates = rng.sample(candidates, k=max_pairs_per_scene)
    return candidates


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MegaDepth two-view training pairs.")
    parser.add_argument("--scene-info-root", type=Path, required=True, help="Directory containing MegaDepth scene_info npz files.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--min-overlap", type=float, default=0.2)
    parser.add_argument("--max-overlap", type=float, default=0.95)
    parser.add_argument("--min-scale-ratio", type=float, default=0.0)
    parser.add_argument("--max-scale-ratio", type=float, default=10.0)
    parser.add_argument("--max-pairs-per-scene", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--relative-paths", action="store_true", help="Store paths relative to --scene-info-root parent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    scene_root = args.scene_info_root.resolve()
    base_root = scene_root.parent.resolve()
    scene_files = find_scene_files(scene_root)
    if not scene_files:
        raise FileNotFoundError(f"No npz scene files found under {scene_root}")

    all_records: List[dict] = []
    for scene_file in scene_files:
        records = sample_scene_pairs(
            scene_file=scene_file,
            root=base_root,
            min_overlap=args.min_overlap,
            max_overlap=args.max_overlap,
            min_scale_ratio=args.min_scale_ratio,
            max_scale_ratio=args.max_scale_ratio,
            max_pairs_per_scene=args.max_pairs_per_scene,
            require_depth=args.require_depth,
            relative_paths=args.relative_paths,
            rng=rng,
        )
        all_records.extend(records)

    write_jsonl(args.output, all_records)
    print(f"[done] wrote {len(all_records)} pairs to {args.output}")


if __name__ == "__main__":
    main()
