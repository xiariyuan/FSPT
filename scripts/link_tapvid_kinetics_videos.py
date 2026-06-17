#!/usr/bin/env python3
"""
Link official Kinetics clip files into TAP-Vid's expected `videos/<id>.mp4` layout.

The official Kinetics archives extract into directories such as:
  datasets/tapvid_kinetics/videos_val_unpacked/<label>/<youtubeid>_<start>_<end>.mp4
  datasets/tapvid_kinetics/videos_test_unpacked/Kinetics700-2020-test/<youtubeid>_<start>_<end>.mp4

TAP-Vid Kinetics annotations refer to the `youtubeid` and the clip window given
by the first two CSV columns. This script reads the official CSV package and
creates symlinks:

  datasets/tapvid_kinetics/videos/<youtubeid>.mp4 -> <exact clip file>

It can be run once or in watch mode so new extractions are linked automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SEGMENT_RE = re.compile(r"^(.*)_([0-9]{6})_([0-9]{6})\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link TAP-Vid Kinetics videos")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/gemini/code/datasets/tapvid_kinetics"),
        help="TAP-Vid Kinetics root directory",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Official TAP-Vid Kinetics CSV (default: <root>/tapvid_kinetics.csv)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Target directory for symlinks (default: <root>/videos)",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="*",
        default=None,
        help="Optional source directories to scan. Defaults to all <root>/videos_*_unpacked dirs.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching for newly extracted clips and relink periodically.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=30,
        help="Seconds to sleep between watch iterations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing non-symlink files at the target path.",
    )
    return parser.parse_args()


def resolve_path(base: Path, path: Path | None, default: Path) -> Path:
    candidate = default if path is None else path
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def load_targets(csv_path: Path) -> Dict[str, Tuple[int, int]]:
    targets: Dict[str, Tuple[int, int]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"Malformed CSV row #{row_idx}: expected at least 3 columns.")
            video_id = row[0].strip()
            if not video_id:
                continue
            meta = (int(float(row[1])), int(float(row[2])))
            prev = targets.get(video_id)
            if prev is None:
                targets[video_id] = meta
            elif prev != meta:
                raise ValueError(
                    f"Conflicting meta columns for video_id={video_id!r}: {prev} vs {meta}"
                )
    return targets


def discover_sources(root: Path, sources: List[Path] | None) -> List[Path]:
    if sources:
        resolved: List[Path] = []
        for src in sources:
            candidate = src if src.is_absolute() else root / src
            if candidate.is_dir():
                resolved.append(candidate.resolve())
        return sorted(resolved)

    candidates = [p.resolve() for p in sorted(root.glob("videos_*_unpacked")) if p.is_dir()]
    return candidates


def build_segment_index(source_dirs: Iterable[Path]) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for source_dir in source_dirs:
        for mp4_path in sorted(source_dir.rglob("*.mp4")):
            if SEGMENT_RE.match(mp4_path.name) is None:
                continue
            existing = index.get(mp4_path.name)
            if existing is None:
                index[mp4_path.name] = mp4_path.resolve()
            elif existing.resolve() != mp4_path.resolve():
                duplicates[mp4_path.name].append(mp4_path.resolve())
    return index, duplicates


def link_target(target: Path, source: Path, force: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    rel_source = os.path.relpath(source, start=target.parent)

    if target.is_symlink():
        current = os.readlink(target)
        if current == rel_source:
            return "kept"
        target.unlink()
    elif target.exists():
        if not force:
            return "exists"
        if target.is_dir():
            raise IsADirectoryError(f"Target path is a directory: {target}")
        target.unlink()

    target.symlink_to(rel_source)
    return "linked"


def run_once(root: Path, csv_path: Path, videos_dir: Path, sources: List[Path] | None, force: bool) -> Tuple[int, int, int]:
    targets = load_targets(csv_path)
    source_dirs = discover_sources(root, sources)
    segment_index, duplicates = build_segment_index(source_dirs)

    linked = 0
    missing = 0
    unchanged = 0

    for video_id, (start, end) in sorted(targets.items()):
        segment_name = f"{video_id}_{start:06d}_{end:06d}.mp4"
        source = segment_index.get(segment_name)
        target = videos_dir / f"{video_id}.mp4"
        if source is None:
            missing += 1
            continue
        status = link_target(target, source, force=force)
        if status == "linked":
            linked += 1
        elif status == "kept":
            unchanged += 1

    print(
        f"[summary] targets={len(targets)} linked={linked} kept={unchanged} "
        f"missing={missing} sources={len(source_dirs)} segments={len(segment_index)}"
    )
    if duplicates:
        dup_count = sum(len(v) for v in duplicates.values())
        print(f"[warn] duplicate clip basenames detected: {len(duplicates)} keys / {dup_count} extras")
    return linked, unchanged, missing


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else Path("/gemini/code") / args.root
    root = root.resolve()
    csv_path = resolve_path(root, args.csv, Path("tapvid_kinetics.csv"))
    videos_dir = resolve_path(root, args.videos_dir, Path("videos"))
    videos_dir.mkdir(parents=True, exist_ok=True)

    source_paths = args.sources
    if source_paths is not None:
        source_paths = [
            src if src.is_absolute() else (root / src)
            for src in source_paths
        ]

    print(f"[info] root={root}")
    print(f"[info] csv={csv_path}")
    print(f"[info] videos_dir={videos_dir}")
    if source_paths:
        print(f"[info] sources={', '.join(str(p) for p in source_paths)}")
    else:
        print("[info] sources=auto-detected videos_*_unpacked directories")
    sys.stdout.flush()

    if not args.watch:
        run_once(root, csv_path, videos_dir, source_paths, force=args.force)
        return 0

    while True:
        run_once(root, csv_path, videos_dir, source_paths, force=args.force)
        time.sleep(max(1, int(args.sleep)))


if __name__ == "__main__":
    raise SystemExit(main())
