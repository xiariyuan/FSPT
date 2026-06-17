#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def rotation_deg(pose0: np.ndarray, pose1: np.ndarray) -> float:
    r0 = pose0[:3, :3]
    r1 = pose1[:3, :3]
    rel = r1 @ r0.T
    trace = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def camera_center(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose[:3, 3], dtype=np.float64)


def discover_transforms(scene_dir: Path, image_layout: str) -> Optional[Path]:
    candidates = []
    if image_layout == "undistorted":
        candidates.extend(
            [
                scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json",
                scene_dir / "dslr" / "transforms_undistorted.json",
                scene_dir / "transforms_undistorted.json",
            ]
        )
    candidates.extend(
        [
            scene_dir / "dslr" / "nerfstudio" / "transforms.json",
            scene_dir / "dslr" / "transforms.json",
            scene_dir / "transforms.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_transforms_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_intrinsics(meta: Dict, frame: Dict) -> List[List[float]]:
    fx = frame.get("fl_x", meta.get("fl_x"))
    fy = frame.get("fl_y", meta.get("fl_y", fx))
    cx = frame.get("cx", meta.get("cx", 0.0))
    cy = frame.get("cy", meta.get("cy", 0.0))
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]


def normalize_file_path(scene_dir: Path, transforms_path: Path, raw_path: str, image_layout: str) -> Path:
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    candidates = [
        (transforms_path.parent / raw).resolve(),
        (scene_dir / raw).resolve(),
    ]
    if image_layout == "undistorted":
        candidates.extend(
            [
                (scene_dir / "dslr" / "undistorted_images" / raw.name).resolve(),
                (scene_dir / "undistorted_images" / raw.name).resolve(),
            ]
        )
    candidates.extend(
        [
            (scene_dir / "dslr" / "images" / raw.name).resolve(),
            (scene_dir / "images" / raw.name).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def fetch_manifest(url: str, token: str, output_path: Optional[Path]) -> Optional[Dict]:
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
    data = json.loads(payload)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def iter_scene_dirs(data_root: Path, scene_list: Optional[Path]) -> Iterable[Path]:
    if scene_list is None:
        for path in sorted(p for p in data_root.iterdir() if p.is_dir()):
            yield path
        return
    with scene_list.open("r", encoding="utf-8") as f:
        for line in f:
            scene = line.strip()
            if not scene:
                continue
            yield data_root / scene


def build_pairs_for_scene(
    scene_dir: Path,
    image_layout: str,
    min_baseline: float,
    max_baseline: float,
    max_rotation_deg: float,
    max_pairs_per_scene: int,
    relative_paths: bool,
    base_root: Path,
    rng: random.Random,
) -> List[dict]:
    transforms_path = discover_transforms(scene_dir, image_layout=image_layout)
    if transforms_path is None:
        return []
    meta = load_transforms_json(transforms_path)
    frames = meta.get("frames", [])
    if not frames:
        return []

    prepared = []
    for idx, frame in enumerate(frames):
        if "transform_matrix" not in frame or "file_path" not in frame:
            continue
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        image_path = normalize_file_path(scene_dir, transforms_path, str(frame["file_path"]), image_layout=image_layout)
        intrinsics = build_intrinsics(meta, frame)
        prepared.append(
            {
                "frame_index": idx,
                "image_path": image_path,
                "pose": pose,
                "intrinsics": intrinsics,
            }
        )

    records: List[dict] = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            pose0 = prepared[i]["pose"]
            pose1 = prepared[j]["pose"]
            baseline = float(np.linalg.norm(camera_center(pose0) - camera_center(pose1)))
            if baseline < min_baseline or baseline > max_baseline:
                continue
            rot = rotation_deg(pose0, pose1)
            if rot > max_rotation_deg:
                continue
            image0 = prepared[i]["image_path"]
            image1 = prepared[j]["image_path"]
            record = {
                "dataset": "scannetpp",
                "scene_id": scene_dir.name,
                "image_layout": image_layout,
                "image0": (image0.relative_to(base_root) if relative_paths else image0).as_posix(),
                "image1": (image1.relative_to(base_root) if relative_paths else image1).as_posix(),
                "intrinsics0": prepared[i]["intrinsics"],
                "intrinsics1": prepared[j]["intrinsics"],
                "pose0": pose0.tolist(),
                "pose1": pose1.tolist(),
                "baseline": baseline,
                "rotation_deg": rot,
            }
            records.append(record)

    if len(records) > max_pairs_per_scene:
        records = rng.sample(records, k=max_pairs_per_scene)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ScanNet++ two-view pairs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--image-layout", choices=("undistorted", "pinhole"), default="undistorted")
    parser.add_argument("--manifest-url", type=str, default="")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--token-env", type=str, default="SCANNETPP_TOKEN")
    parser.add_argument("--min-baseline", type=float, default=0.05)
    parser.add_argument("--max-baseline", type=float, default=1.50)
    parser.add_argument("--max-rotation-deg", type=float, default=45.0)
    parser.add_argument("--max-pairs-per-scene", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = args.data_root.resolve()
    if args.manifest_url:
        token_value = os.environ.get(args.token_env, "")
        if not token_value:
            raise RuntimeError(f"Missing token in environment variable: {args.token_env}")
        fetch_manifest(args.manifest_url, token_value, args.manifest_out)

    all_records: List[dict] = []
    for scene_dir in iter_scene_dirs(data_root, args.scene_list):
        if not scene_dir.exists():
            continue
        all_records.extend(
            build_pairs_for_scene(
                scene_dir=scene_dir,
                image_layout=args.image_layout,
                min_baseline=args.min_baseline,
                max_baseline=args.max_baseline,
                max_rotation_deg=args.max_rotation_deg,
                max_pairs_per_scene=args.max_pairs_per_scene,
                relative_paths=args.relative_paths,
                base_root=data_root,
                rng=rng,
            )
        )

    write_jsonl(args.output, all_records)
    print(f"[done] wrote {len(all_records)} ScanNet++ pairs to {args.output}")


if __name__ == "__main__":
    main()
