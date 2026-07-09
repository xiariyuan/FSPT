#!/usr/bin/env python3
"""Export a unified teacher/ensemble cache into per-video base_tracks files.

The exported directory is compatible with datasets_code.base_tracks.BaseTracksCacheWrapper
and datasets.base_tracks.BaseTracksCacheWrapper.  Each output file contains:

  {
    "video_name": str,
    "query_points": Tensor (N,3),
    "base_tracks": Tensor (N,T,2),
    "base_visibility": Tensor (N,T),
    ...metadata...
  }

This lets an evaluated teacher ensemble be injected into train.py through
`data.train.base_tracks_dir` without reopening deprecated refiner routes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _load_cache(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{path} is not a unified cache dict with records")
    return payload


def _to_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
    else:
        out = torch.as_tensor(np.asarray(value))
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out.contiguous()




def _scale_cache_normalized_to_davis_runtime(record: Dict[str, Any], tracks: torch.Tensor, query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert cache-normalized [y,x] coords to TAPVidDAVIS runtime normalized coords.

    The unified cache/eval path stores normalized spatial coordinates using the
    pixel-corner convention (divide by H-1/W-1).  TAPVidDAVIS runtime samples in
    this repo expose normalized coordinates from the annotation/generator path
    using H/W.  For base_tracks strict injection, the cached teacher coordinates
    must match the runtime dataset convention.
    """
    original_size = record.get("original_size", None)
    if original_size is None:
        raise ValueError("coord-convention=davis_runtime requires original_size in every record")
    size = torch.as_tensor(np.asarray(original_size), dtype=torch.float32)
    if size.numel() < 2:
        raise ValueError(f"invalid original_size={original_size!r}")
    height = float(size.flatten()[0].item())
    width = float(size.flatten()[1].item())
    if height <= 1.0 or width <= 1.0:
        raise ValueError(f"invalid original_size height/width: {original_size!r}")
    y_scale = (height - 1.0) / height
    x_scale = (width - 1.0) / width
    tracks = tracks.clone()
    query_points = query_points.clone()
    tracks[..., 0] *= y_scale
    tracks[..., 1] *= x_scale
    query_points[..., 1] *= y_scale
    query_points[..., 2] *= x_scale
    return tracks.contiguous(), query_points.contiguous()


def _video_name(record: Dict[str, Any], fallback_idx: int) -> str:
    for key in ("video_name", "video_id", "sequence_name"):
        value = record.get(key, None)
        if value is not None and str(value).strip():
            return str(value)
    return f"sequence_{fallback_idx:04d}"


def export_cache(
    cache_path: Path,
    output_dir: Path,
    dataset_subdir: str | None = None,
    overwrite: bool = False,
    coord_convention: str = "cache",
) -> Dict[str, Any]:
    payload = _load_cache(cache_path)
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("cache has no records")

    root = output_dir / dataset_subdir if dataset_subdir else output_dir
    root.mkdir(parents=True, exist_ok=True)

    exported = []
    for idx, record in enumerate(records):
        name = _video_name(record, idx)
        out_path = root / f"{name}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace")

        base_tracks = _to_tensor(record["pred_tracks"], torch.float32)
        base_visibility = _to_tensor(record["pred_visibility"], torch.bool)
        query_points = _to_tensor(record["query_points"], torch.float32)
        if coord_convention == "davis_runtime":
            base_tracks, query_points = _scale_cache_normalized_to_davis_runtime(record, base_tracks, query_points)
        elif coord_convention != "cache":
            raise ValueError(f"unsupported coord_convention={coord_convention!r}")

        if base_tracks.ndim != 3 or base_tracks.shape[-1] != 2:
            raise ValueError(f"{name}: expected pred_tracks shape (N,T,2), got {tuple(base_tracks.shape)}")
        if base_visibility.shape != base_tracks.shape[:2]:
            raise ValueError(
                f"{name}: pred_visibility shape {tuple(base_visibility.shape)} does not match tracks {tuple(base_tracks.shape[:2])}"
            )
        if query_points.ndim != 2 or query_points.shape[0] != base_tracks.shape[0]:
            raise ValueError(f"{name}: query_points shape {tuple(query_points.shape)} incompatible with tracks")

        out_payload: Dict[str, Any] = {
            "video_name": name,
            "video_id": str(record.get("video_id", name)),
            "sequence_index": int(record.get("sequence_index", idx)),
            "query_points": query_points,
            "base_tracks": base_tracks,
            "base_visibility": base_visibility,
            "source_cache": str(cache_path),
            "source_model_name": str(record.get("model_name", payload.get("model_name", "unknown"))),
            "ensemble_members": list(record.get("ensemble_members", payload.get("ensemble_members", []))),
            "ensemble_track_aggregation": record.get("ensemble_track_aggregation", payload.get("ensemble_track_aggregation", None)),
            "ensemble_visibility_strategy": record.get("ensemble_visibility_strategy", payload.get("ensemble_visibility_strategy", None)),
            "base_tracks_coord_convention": coord_convention,
        }
        if "original_size" in record:
            out_payload["original_size"] = _to_tensor(record["original_size"], torch.int64)
        if "model_input_size" in record:
            out_payload["model_input_size"] = _to_tensor(record["model_input_size"], torch.int64)

        torch.save(out_payload, out_path)
        exported.append({
            "video_name": name,
            "path": str(out_path),
            "num_points": int(base_tracks.shape[0]),
            "num_frames": int(base_tracks.shape[1]),
            "visible_frac": round(float(base_visibility.float().mean().item()), 6),
        })

    manifest = {
        "cache_path": str(cache_path),
        "output_dir": str(root),
        "dataset_subdir": dataset_subdir,
        "n_records": len(records),
        "model_name": payload.get("model_name", "unknown"),
        "ensemble_track_aggregation": payload.get("ensemble_track_aggregation", None),
        "ensemble_visibility_strategy": payload.get("ensemble_visibility_strategy", None),
        "base_tracks_coord_convention": coord_convention,
        "exported": exported,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unified ensemble cache to per-video base_tracks files")
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-subdir", default="", help="Optional subdir such as davis; wrapper will prefer it when dataset_name is set")
    parser.add_argument(
        "--coord-convention",
        choices=("cache", "davis_runtime"),
        default="cache",
        help=(
            "Coordinate convention for exported base_tracks/query_points. "
            "cache preserves source cache values; davis_runtime rescales normalized "
            "[y,x] coords from H-1/W-1 to the TAPVidDAVIS runtime H/W convention."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = export_cache(
        cache_path=Path(args.cache_path),
        output_dir=Path(args.output_dir),
        dataset_subdir=str(args.dataset_subdir).strip() or None,
        overwrite=bool(args.overwrite),
        coord_convention=str(args.coord_convention),
    )
    print(f"Wrote {manifest['n_records']} base-track files to {manifest['output_dir']}")
    print(f"manifest={Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
