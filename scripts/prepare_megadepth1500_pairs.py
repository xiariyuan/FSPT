#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def parse_intrinsics(values: List[float]) -> List[List[float]]:
    if len(values) != 9:
        raise ValueError(f"Expected 9 intrinsic values, got {len(values)}")
    return [
        [float(values[0]), float(values[1]), float(values[2])],
        [float(values[3]), float(values[4]), float(values[5])],
        [float(values[6]), float(values[7]), float(values[8])],
    ]


def parse_relative_pose(values: List[float]) -> List[List[float]]:
    if len(values) != 12:
        raise ValueError(f"Expected 12 pose values, got {len(values)}")
    return [
        [float(values[0]), float(values[1]), float(values[2]), float(values[3])],
        [float(values[4]), float(values[5]), float(values[6]), float(values[7])],
        [float(values[8]), float(values[9]), float(values[10]), float(values[11])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def parse_pairs_file(root: Path, relative_paths: bool) -> List[dict]:
    pairs_path = root / "pairs_calibrated.txt"
    image_root = root / "images"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Missing pairs_calibrated.txt under {root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing images directory under {root}")

    records: List[dict] = []
    for line_idx, raw_line in enumerate(pairs_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 32:
            raise ValueError(f"Line {line_idx} in {pairs_path} has {len(parts)} fields, expected 32")
        image0_rel = Path(parts[0])
        image1_rel = Path(parts[1])
        intrinsics0 = parse_intrinsics([float(v) for v in parts[2:11]])
        intrinsics1 = parse_intrinsics([float(v) for v in parts[11:20]])
        relative_pose = parse_relative_pose([float(v) for v in parts[20:32]])

        image0 = (image_root / image0_rel).resolve()
        image1 = (image_root / image1_rel).resolve()
        if not image0.is_file() or not image1.is_file():
            continue

        if relative_paths:
            image0_out = image0.relative_to(root).as_posix()
            image1_out = image1.relative_to(root).as_posix()
        else:
            image0_out = image0.as_posix()
            image1_out = image1.as_posix()

        records.append(
            {
                "dataset": "megadepth1500",
                "scene_id": image0_rel.parts[0] if len(image0_rel.parts) > 1 else image0_rel.stem,
                "image0": image0_out,
                "image1": image1_out,
                "intrinsics0": intrinsics0,
                "intrinsics1": intrinsics1,
                "relative_pose": relative_pose,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MegaDepth-1500 calibrated two-view pairs.")
    parser.add_argument("--root", type=Path, required=True, help="Extracted megadepth1500 directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--relative-paths", action="store_true", help="Store image paths relative to --root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = parse_pairs_file(root=root, relative_paths=bool(args.relative_paths))
    write_jsonl(args.output, records)
    print(f"[done] wrote {len(records)} MegaDepth-1500 pairs to {args.output}")


if __name__ == "__main__":
    main()
