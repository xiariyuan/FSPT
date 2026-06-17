#!/usr/bin/env python3
"""
Validate the prepared MegaDepth two-view pair dataset.

This script checks three things:
1. The annotation JSONL is readable and contains the expected pose/image/depth fields.
2. The referenced image and depth files exist under the configured MegaDepth root.
3. Optionally, a small number of samples can be decoded to catch corruption early.

The script is intentionally lightweight so it can be run before launching a long
pretraining job.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass
class RecordStatus:
    index: int
    scene_id: str
    pair_name: str
    image0_exists: bool
    image1_exists: bool
    depth0_exists: bool
    depth1_exists: bool
    has_pose0: bool
    has_pose1: bool

    @property
    def ready(self) -> bool:
        return (
            self.image0_exists
            and self.image1_exists
            and self.depth0_exists
            and self.depth1_exists
            and self.has_pose0
            and self.has_pose1
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_image_from_depth(root: Path, image_value: Any, depth_value: Any) -> Optional[Path]:
    candidates: List[Path] = []
    if image_value is not None:
        candidates.append(_resolve_path(root, image_value))
    depth_path: Optional[Path] = None
    if depth_value is not None:
        depth_path = _resolve_path(root, depth_value)
        if depth_path.exists():
            parent = depth_path.parent.parent
            stem = depth_path.stem
            candidates.append(parent / "imgs" / f"{stem}.jpg")
            candidates.append(parent / "imgs" / f"{stem}.png")
    if image_value is not None and depth_path is not None:
        candidates.append(depth_path.parent.parent / "imgs" / Path(str(image_value)).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _optional_path(root: Path, record: Dict[str, Any], field: str) -> Optional[Path]:
    value = record.get(field, None)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _resolve_path(root, text)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Failed to parse {path} line {line_idx}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path} line {line_idx}, got {type(record)!r}")
            records.append(record)
    return records


def _check_image(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # pragma: no cover - depends on local dataset state
        return False, str(exc)


def _check_depth(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, f"h5py unavailable: {exc}"

    try:
        with h5py.File(path, "r") as f:  # type: ignore[operator]
            if "depth" not in f:
                keys = ", ".join(sorted(f.keys()))
                return False, f"missing key 'depth' (available keys: {keys})"
            depth = f["depth"]
            if len(depth.shape) < 2:
                return False, f"unexpected depth shape {depth.shape}"
        return True, None
    except Exception as exc:  # pragma: no cover - depends on local dataset state
        return False, str(exc)


def _summarize(records: Sequence[Dict[str, Any]], root: Path) -> Tuple[List[RecordStatus], Counter]:
    statuses: List[RecordStatus] = []
    counters: Counter = Counter()

    for idx, record in enumerate(records):
        scene_id = str(record.get("scene_id", ""))
        pair_name = str(
            record.get(
                "pair_name",
                f"{scene_id}_{Path(str(record.get('image0', f'record_{idx:06d}'))).stem}",
            )
        )
        has_pose0 = "pose0" in record
        has_pose1 = "pose1" in record
        depth0 = _optional_path(root, record, "depth0")
        depth1 = _optional_path(root, record, "depth1")
        image0 = _resolve_image_from_depth(root, record.get("image0"), record.get("depth0"))
        image1 = _resolve_image_from_depth(root, record.get("image1"), record.get("depth1"))

        status = RecordStatus(
            index=idx,
            scene_id=scene_id,
            pair_name=pair_name,
            image0_exists=image0 is not None and image0.exists(),
            image1_exists=image1 is not None and image1.exists(),
            depth0_exists=depth0 is not None and depth0.exists(),
            depth1_exists=depth1 is not None and depth1.exists(),
            has_pose0=has_pose0,
            has_pose1=has_pose1,
        )
        statuses.append(status)

        counters["total"] += 1
        counters["has_pose0"] += int(has_pose0)
        counters["has_pose1"] += int(has_pose1)
        counters["image0_exists"] += int(image0 is not None and image0.exists())
        counters["image1_exists"] += int(image1 is not None and image1.exists())
        counters["depth0_exists"] += int(depth0 is not None and depth0.exists())
        counters["depth1_exists"] += int(depth1 is not None and depth1.exists())
        counters["ready"] += int(status.ready)
        counters["scene_id_nonempty"] += int(bool(scene_id))

    return statuses, counters


def _print_examples(label: str, examples: Sequence[str], max_examples: int) -> None:
    if not examples:
        return
    print(f"{label}:")
    for item in list(examples)[:max_examples]:
        print(f"  - {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MegaDepth pair pretraining data")
    parser.add_argument(
        "--root",
        type=str,
        default="/gemini/code/datasets/megadepth",
        help="MegaDepth dataset root directory",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="data/two_view/megadepth_scene_info_pairs_sampled.jsonl",
        help="JSONL file with the prepared image/depth pairs",
    )
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=3,
        help="Decode this many ready samples with PIL/h5py to catch corruption early",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="How many failing examples to print per category",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any record is missing required files",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    annotation_path = Path(args.annotation_file)
    if not annotation_path.is_absolute():
        annotation_path = (repo_root / annotation_path).resolve()
    root = Path(args.root).resolve()

    if not annotation_path.exists():
        print(f"Annotation file not found: {annotation_path}")
        return 2
    if not root.exists():
        print(f"MegaDepth root not found: {root}")
        return 2

    records = _load_jsonl(annotation_path)
    statuses, counters = _summarize(records, root)

    ready_statuses = [s for s in statuses if s.ready]
    image_ready = sum(int(s.image0_exists and s.image1_exists) for s in statuses)
    depth_ready = sum(int(s.depth0_exists and s.depth1_exists) for s in statuses)

    print(f"Annotation file: {annotation_path}")
    print(f"MegaDepth root:   {root}")
    print(f"Total records:    {counters['total']}")
    print(f"Ready records:    {counters['ready']}")
    print(f"Image pairs:      {image_ready}")
    print(f"Depth pairs:      {depth_ready}")
    pose_pairs = sum(int(s.has_pose0 and s.has_pose1) for s in statuses)
    print(f"Pose pairs:       {pose_pairs}")
    print(f"Non-empty scenes:  {counters['scene_id_nonempty']}")

    missing_pose_examples: List[str] = []
    missing_image_examples: List[str] = []
    missing_depth_examples: List[str] = []
    for status in statuses:
        if not (status.has_pose0 and status.has_pose1) and len(missing_pose_examples) < args.max_examples:
            missing_pose_examples.append(f"[{status.index}] {status.pair_name} missing pose0/pose1")
        if not (status.image0_exists and status.image1_exists) and len(missing_image_examples) < args.max_examples:
            missing_image_examples.append(
                f"[{status.index}] {status.pair_name} missing image0/image1"
            )
        if not (status.depth0_exists and status.depth1_exists) and len(missing_depth_examples) < args.max_examples:
            missing_depth_examples.append(
                f"[{status.index}] {status.pair_name} missing depth0/depth1"
            )

    _print_examples("Missing poses", missing_pose_examples, args.max_examples)
    _print_examples("Missing images", missing_image_examples, args.max_examples)
    _print_examples("Missing depths", missing_depth_examples, args.max_examples)

    if args.decode_samples > 0 and ready_statuses:
        decode_count = min(args.decode_samples, len(ready_statuses))
        print(f"Decoding {decode_count} ready samples...")
        decode_ok = 0
        decode_failures: List[str] = []
        h5py_available = True
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            h5py_available = False
            print(f"h5py unavailable, skipping depth decode checks: {exc}")

        for status in ready_statuses[:decode_count]:
            record = records[status.index]
            image0 = _resolve_image_from_depth(root, record.get("image0"), record.get("depth0"))
            image1 = _resolve_image_from_depth(root, record.get("image1"), record.get("depth1"))
            depth0 = _resolve_path(root, record["depth0"])
            depth1 = _resolve_path(root, record["depth1"])
            try:
                if image0 is None or image1 is None:
                    raise FileNotFoundError(
                        f"Could not resolve MegaDepth images for {status.pair_name}"
                    )
                ok_img0, err_img0 = _check_image(image0)
                ok_img1, err_img1 = _check_image(image1)
                if not ok_img0 or not ok_img1:
                    raise RuntimeError(f"image decode failed: {err_img0 or ''} {err_img1 or ''}".strip())
                if h5py_available:
                    ok0, err0 = _check_depth(depth0)
                    ok1, err1 = _check_depth(depth1)
                    if not ok0 or not ok1:
                        raise RuntimeError(f"depth decode failed: {err0 or ''} {err1 or ''}".strip())
                decode_ok += 1
            except Exception as exc:  # pragma: no cover - depends on local dataset state
                decode_failures.append(f"[{status.index}] {status.pair_name}: {exc}")

        print(f"Decode checks passed: {decode_ok}/{decode_count}")
        _print_examples("Decode failures", decode_failures, args.max_examples)

    if args.strict and counters["ready"] != counters["total"]:
        print("Strict mode requested and the dataset is incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
