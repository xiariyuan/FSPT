#!/usr/bin/env python3
"""Check exported per-video base_tracks against the source unified cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _video_name(record: Dict[str, Any], idx: int) -> str:
    for key in ("video_name", "video_id", "sequence_name"):
        value = record.get(key, None)
        if value is not None and str(value).strip():
            return str(value)
    return f"sequence_{idx:04d}"


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _to_bool_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().bool()
    return torch.as_tensor(np.asarray(value), dtype=torch.bool)


def _scale_cache_normalized_to_davis_runtime(
    record: Dict[str, Any],
    tracks: torch.Tensor,
    query_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_size = record.get("original_size", None)
    if original_size is None:
        raise ValueError("davis_runtime convention requires original_size in source cache record")
    size = torch.as_tensor(np.asarray(original_size), dtype=torch.float32).flatten()
    if size.numel() < 2:
        raise ValueError(f"invalid original_size={original_size!r}")
    height = float(size[0].item())
    width = float(size[1].item())
    if height <= 1.0 or width <= 1.0:
        raise ValueError(f"invalid original_size height/width: {original_size!r}")
    tracks = tracks.clone()
    query_points = query_points.clone()
    tracks[..., 0] *= (height - 1.0) / height
    tracks[..., 1] *= (width - 1.0) / width
    query_points[..., 1] *= (height - 1.0) / height
    query_points[..., 2] *= (width - 1.0) / width
    return tracks.contiguous(), query_points.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported base_tracks files against source cache")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--base-tracks-dir", required=True)
    parser.add_argument("--dataset-subdir", default="")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--coord-convention",
        choices=("auto", "cache", "davis_runtime"),
        default="auto",
        help="Expected exported coordinate convention. auto reads manifest base_tracks_coord_convention.",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    base_dir = Path(args.base_tracks_dir)
    root = base_dir / args.dataset_subdir if args.dataset_subdir else base_dir

    cache = _load_torch(cache_path)
    if not isinstance(cache, dict) or "records" not in cache:
        raise ValueError(f"{cache_path} is not a unified cache with records")
    records = cache["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("cache records are empty")

    manifest_path = base_dir / "manifest.json"
    manifest_ok = manifest_path.exists()
    manifest = json.load(open(manifest_path)) if manifest_ok else None
    coord_convention = str(args.coord_convention)
    if coord_convention == "auto":
        if isinstance(manifest, dict):
            coord_convention = str(manifest.get("base_tracks_coord_convention", "cache") or "cache")
        else:
            coord_convention = "cache"

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    max_track_diff = 0.0
    max_query_diff = 0.0
    visibility_mismatches = 0

    for idx, record in enumerate(records):
        name = _video_name(record, idx)
        path = root / f"{name}.pt"
        if not path.exists():
            failures.append(f"missing export for {name}: {path}")
            continue
        payload = _load_torch(path)
        if not isinstance(payload, dict):
            failures.append(f"{name}: export payload is not dict")
            continue

        src_tracks = _to_float_tensor(record["pred_tracks"])
        src_vis = _to_bool_tensor(record["pred_visibility"])
        src_query = _to_float_tensor(record["query_points"])
        if coord_convention == "davis_runtime":
            src_tracks, src_query = _scale_cache_normalized_to_davis_runtime(record, src_tracks, src_query)
        elif coord_convention != "cache":
            raise ValueError(f"unsupported coord_convention={coord_convention!r}")
        out_tracks = _to_float_tensor(payload.get("base_tracks"))
        out_vis = _to_bool_tensor(payload.get("base_visibility"))
        out_query = _to_float_tensor(payload.get("query_points"))

        if out_tracks.shape != src_tracks.shape:
            failures.append(f"{name}: base_tracks shape {tuple(out_tracks.shape)} != source {tuple(src_tracks.shape)}")
            continue
        if out_vis.shape != src_vis.shape:
            failures.append(f"{name}: base_visibility shape {tuple(out_vis.shape)} != source {tuple(src_vis.shape)}")
            continue
        if out_query.shape != src_query.shape:
            failures.append(f"{name}: query_points shape {tuple(out_query.shape)} != source {tuple(src_query.shape)}")
            continue

        track_diff = float(torch.max(torch.abs(out_tracks - src_tracks)).item()) if src_tracks.numel() else 0.0
        query_diff = float(torch.max(torch.abs(out_query - src_query)).item()) if src_query.numel() else 0.0
        vis_mismatch = int((out_vis != src_vis).sum().item())
        max_track_diff = max(max_track_diff, track_diff)
        max_query_diff = max(max_query_diff, query_diff)
        visibility_mismatches += vis_mismatch

        if track_diff > args.atol:
            failures.append(f"{name}: max track diff {track_diff:.3e} > {args.atol:.3e}")
        if query_diff > args.atol:
            failures.append(f"{name}: max query diff {query_diff:.3e} > {args.atol:.3e}")
        if vis_mismatch != 0:
            failures.append(f"{name}: visibility mismatches {vis_mismatch}")

        rows.append({
            "video_name": name,
            "num_points": int(src_tracks.shape[0]),
            "num_frames": int(src_tracks.shape[1]),
            "max_track_diff": track_diff,
            "max_query_diff": query_diff,
            "visibility_mismatches": vis_mismatch,
        })

    existing_exports = sorted(root.glob("*.pt"))
    if len(existing_exports) != len(records):
        failures.append(f"export file count {len(existing_exports)} != cache records {len(records)}")

    summary = {
        "ok": not failures,
        "cache_path": str(cache_path),
        "base_tracks_dir": str(root),
        "n_records": len(records),
        "n_export_files": len(existing_exports),
        "manifest_ok": manifest_ok,
        "manifest_n_records": manifest.get("n_records") if isinstance(manifest, dict) else None,
        "coord_convention": coord_convention,
        "max_track_diff": max_track_diff,
        "max_query_diff": max_query_diff,
        "visibility_mismatches": visibility_mismatches,
        "failures": failures,
        "checked": rows,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "checked"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
