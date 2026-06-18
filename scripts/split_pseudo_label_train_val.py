#!/usr/bin/env python3
"""Split pseudo-label JSONL into train / val by video_id.

Splitting by video_id avoids leakage across train and val.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split pseudo-label JSONL into train/val.")
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--val-video-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.labels_jsonl))
    if not rows:
        summary = {
            "input_n": 0,
            "train_n": 0,
            "val_n": 0,
            "train_videos": [],
            "val_videos": [],
            "leakage": {"shared_videos": [], "shared_tracks": 0},
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        _write_jsonl(Path(args.train_jsonl), [])
        _write_jsonl(Path(args.val_jsonl), [])
        print(json.dumps(summary, indent=2))
        return

    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(str(row.get("video_id", "unknown")), []).append(row)

    videos = sorted(by_video)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(videos)

    n_val = max(1, int(round(len(videos) * args.val_video_frac))) if len(videos) > 1 else 0
    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    train_rows = [row for row in rows if str(row.get("video_id", "unknown")) in train_videos]
    val_rows = [row for row in rows if str(row.get("video_id", "unknown")) in val_videos]

    shared_videos = sorted(train_videos.intersection(val_videos))
    train_tracks = {(str(r.get("video_id", "unknown")), int(r.get("track_id", -1))) for r in train_rows}
    val_tracks = {(str(r.get("video_id", "unknown")), int(r.get("track_id", -1))) for r in val_rows}
    shared_tracks = len(train_tracks.intersection(val_tracks))

    _write_jsonl(Path(args.train_jsonl), train_rows)
    _write_jsonl(Path(args.val_jsonl), val_rows)

    summary = {
        "input_n": len(rows),
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
        "val_video_frac": args.val_video_frac,
        "seed": args.seed,
        "leakage": {
            "shared_videos": shared_videos,
            "shared_tracks": shared_tracks,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.train_jsonl}")
    print(f"Wrote {args.val_jsonl}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
