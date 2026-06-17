#!/usr/bin/env python3
"""
Precompute CoTracker base tracks for TAP-Vid evaluation datasets.

This is mainly useful for Route A (cotracker_refiner) to avoid running CoTracker
inside every evaluation loop. It can also be used for training only when your
dataset sampling is deterministic (no random temporal window / no random points),
and your point sampling strategy matches (e.g., occlusion_balanced vs uniform),
otherwise cached query_points will not match.

Example (DAVIS):
  python scripts/precompute_base_tracks.py \\
    --dataset davis \\
    --root /gemini/code/datasets/tapvid_davis \\
    --checkpoint baselines/cotracker/checkpoints/scaled_offline.pth \\
    --output-dir outputs/base_tracks
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CoTracker base tracks")
    parser.add_argument("--dataset", type=str, required=True, choices=["davis", "kinetics", "kubric"])
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--split", type=str, default=None, help="Split (kubric only)")
    parser.add_argument("--checkpoint", type=str, required=True, help="CoTracker checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs/base_tracks", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional input resolution (H W) to match your evaluation/training setting.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Kubric only: number of frames per sample (must match your dataloader setting).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Kubric only: number of query points per sample (must match your dataloader setting).",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "occlusion_balanced"],
        help="Kubric only: point sampling strategy (must match training config).",
    )
    parser.add_argument(
        "--sampling-hard-fraction",
        type=float,
        default=0.5,
        help="Kubric only: fraction of hard/occlusion points (used with --sampling-strategy occlusion_balanced).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic sampling for Kubric (useful only when caching must match query_points).",
    )
    parser.add_argument(
        "--deterministic-seed",
        type=int,
        default=42,
        help="Seed for deterministic Kubric sampling (only used with --deterministic).",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_path() -> None:
    """Avoid name collision with HuggingFace `datasets` package."""
    root = _project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main():
    _ensure_project_on_path()
    args = parse_args()
    project_root = _project_root()
    root = Path(args.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"CoTracker checkpoint not found: {checkpoint}")

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    out_dir = out_root / str(args.dataset).lower().strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Import CoTracker wrapper from the project (requires cotracker installed on server).
    from datasets import get_dataloader
    from models.cotracker_refiner import CoTrackerBase

    extra = {}
    # Always disable augmentation for caching: precomputed query_points/base_tracks must be stable.
    extra["augmentation"] = {"enabled": False}
    if args.resolution is not None:
        extra["resolution"] = tuple(int(v) for v in args.resolution)
    if args.dataset == "kubric":
        if args.split is not None:
            extra["split"] = args.split
        # Kubric sampling is often stochastic (random window + random points).
        # Use with caution: the cache is only valid if query_points are deterministic.
        if args.sampling_strategy is not None:
            extra["sampling"] = {
                "strategy": str(args.sampling_strategy).lower().strip(),
                "hard_fraction": float(args.sampling_hard_fraction),
            }
        if args.deterministic:
            extra["deterministic_sampling"] = True
            extra["deterministic_seed"] = int(args.deterministic_seed)
        if args.num_frames is not None:
            extra["num_frames"] = int(args.num_frames)
        if args.num_points is not None:
            extra["num_points"] = int(args.num_points)

    dataloader = get_dataloader(
        name=args.dataset,
        root=str(root),
        batch_size=1,
        num_workers=int(args.num_workers),
        seed=42,
        distributed=False,
        **extra,
    )

    tracker = CoTrackerBase(
        checkpoint=str(checkpoint),
        offline=True,
        window_len=60,
        v2=False,
        require=True,
    ).to(device)
    tracker.eval()

    saved = 0
    skipped = 0
    iterator = tqdm(dataloader, desc=f"Precompute {args.dataset}", total=None)
    for batch in iterator:
        if batch is None or not isinstance(batch, dict):
            continue
        video = batch.get("video")
        query_points = batch.get("query_points")
        if not isinstance(video, torch.Tensor) or not isinstance(query_points, torch.Tensor):
            continue
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)

        video_name = batch.get("video_name", None)
        if isinstance(video_name, (list, tuple)):
            video_name = video_name[0] if video_name else None
        if isinstance(video_name, torch.Tensor):
            try:
                video_name = str(video_name.item())
            except Exception:
                video_name = None
        if video_name is None:
            video_name = f"sample_{saved+skipped:06d}"
        video_name = str(video_name)

        out_path = out_dir / f"{video_name}.pt"
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            if args.limit is not None and (saved + skipped) >= int(args.limit):
                break
            continue

        with torch.no_grad():
            tracks, vis = tracker(video.to(device), query_points.to(device))

        payload = {
            "video_name": video_name,
            "query_points": query_points[0].detach().cpu(),
            "base_tracks": tracks[0].detach().cpu(),
            "base_visibility": vis[0].detach().cpu(),
            "checkpoint": str(checkpoint),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            original_size = batch.get("original_size", None)
            if isinstance(original_size, (list, tuple)) and len(original_size) >= 2:
                payload["original_size"] = (int(original_size[0]), int(original_size[1]))
        except Exception:
            pass

        tmp_path = out_path.with_suffix(".tmp")
        torch.save(payload, str(tmp_path))
        tmp_path.replace(out_path)

        saved += 1
        iterator.set_postfix({"saved": saved, "skipped": skipped})
        if args.limit is not None and saved >= int(args.limit):
            break

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "root": str(root),
                "checkpoint": str(checkpoint),
                "resolution": list(args.resolution) if args.resolution is not None else None,
                "num_frames": int(args.num_frames) if args.num_frames is not None else None,
                "num_points": int(args.num_points) if args.num_points is not None else None,
                "sampling_strategy": str(args.sampling_strategy)
                if args.dataset == "kubric" and args.sampling_strategy is not None
                else None,
                "sampling_hard_fraction": float(args.sampling_hard_fraction)
                if args.dataset == "kubric" and args.sampling_strategy is not None
                else None,
                "deterministic": bool(args.deterministic),
                "deterministic_seed": int(args.deterministic_seed) if args.deterministic else None,
                "saved": saved,
                "skipped": skipped,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Saved {saved} cache files to {out_dir}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
