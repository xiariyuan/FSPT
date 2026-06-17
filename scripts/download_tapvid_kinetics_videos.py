#!/usr/bin/env python3
"""
Download TAP-Vid Kinetics videos from YouTube by split.

This downloader only pulls the 1189 videos referenced by TAP-Vid Kinetics,
instead of downloading the full Kinetics-700 archive shards.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TAP-Vid Kinetics YouTube videos")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/gemini/code/datasets/tapvid_kinetics"),
        help="Dataset root containing train.txt / val.txt / test.txt",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which TAP-Vid Kinetics split to download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of videos to download per run",
    )
    return parser.parse_args()


def collect_ids(root: Path, split: str) -> list[str]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    ids: list[str] = []
    for split_name in splits:
        split_file = root / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        ids.extend([line.strip() for line in split_file.read_text().splitlines() if line.strip()])
    return ids


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else Path("/gemini/code") / args.root
    root = root.resolve()
    ids = collect_ids(root, args.split)
    if args.limit is not None and args.limit > 0:
        ids = ids[: args.limit]

    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    urls_file = root / f".tapvid_kinetics_{args.split}_urls.txt"
    archive_file = root / f".tapvid_kinetics_{args.split}_archive.txt"

    urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in ids]
    urls_file.write_text("\n".join(urls) + "\n", encoding="utf-8")

    yt_dlp_bin = Path.home() / ".local" / "bin" / "yt-dlp"
    if not yt_dlp_bin.exists():
        raise FileNotFoundError(f"yt-dlp not found at {yt_dlp_bin}")

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        str(yt_dlp_bin),
        "--ignore-errors",
        "--no-overwrites",
        "--continue",
        "--retries",
        "infinite",
        "--fragment-retries",
        "infinite",
        "--download-archive",
        str(archive_file),
        "--batch-file",
        str(urls_file),
        "--paths",
        str(videos_dir),
        "--output",
        "%(id)s.%(ext)s",
        "--ffmpeg-location",
        ffmpeg_bin,
        "--merge-output-format",
        "mp4",
        "--format",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--concurrent-fragments",
        "8",
    ]

    print(f"[info] split={args.split} total_urls={len(urls)}")
    print(f"[info] videos_dir={videos_dir}")
    print(f"[info] archive_file={archive_file}")
    print(f"[info] ffmpeg={ffmpeg_bin}")
    print("[cmd] " + " ".join(cmd))
    sys.stdout.flush()

    env = dict(os.environ)
    env["PATH"] = f"{yt_dlp_bin.parent}:{env.get('PATH', '')}"
    completed = subprocess.run(cmd, env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
