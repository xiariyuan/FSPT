#!/usr/bin/env python3
"""Export repo-native .npz prediction caches to Attempt 0 unified schema.

Bridges the gap between repo-native TAP-Vid caches (pixel x,y, B,T,N,2)
and Attempt 0 unified caches (normalized y,x, N,T,2).

Currently supports the `first + input` bridge:
  - query protocol: first (queried_first=True)
  - resolution space: input (256×256)

Example:
    python scripts/attempt0_export_tapvid_repo_cache.py \
        --baseline-name trackon2_dinov3 \
        --cache-dir outputs/trackon2_dinov3_davis_cache/davis/trackon2/ \
        --tapvid-pkl datasets_data/tapvid_davis/tapvid_davis.pkl \
        --dataset-type davis \
        --query-protocol first \
        --space input \
        --out-cache outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt \
        --out-report outputs/attempt0_2026-06-15_recovery/reports/trackon2_davis_first_input_export_report.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache, save_attempt0_cache, write_json_report


# ---------------------------------------------------------------------------
# TAP-Vid dataset helpers (mirrored from baselines/track_on/dataset/tapvid.py)
# ---------------------------------------------------------------------------

def _sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Sample first-frame queries (queried_first=True).

    Args:
        target_occluded: (N, T) bool, True=occluded
        target_points: (N, T, 2) float32, [x, y] pixel space

    Returns:
        dict with query_points (N, 3) [t, y, x], target_points (N, T, 2) [x, y],
        occluded (N, T) bool.
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "query_points": query_points,
        "target_points": target_points,
        "occluded": target_occluded,
    }


def _sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    query_stride: int = 5,
) -> Dict[str, np.ndarray]:
    """Sample strided queries (queried_first=False).

    Args:
        target_occluded: (N, T) bool, True=occluded
        target_points: (N, T, 2) float32, [x, y] pixel space

    Returns:
        dict with query_points, target_points, occluded.
    """
    tracks = []
    occs = []
    queries = []
    trackgroup = np.arange(target_occluded.shape[0])

    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],  # y
                target_points[:, i, 0],  # x
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])

    return {
        "query_points": np.concatenate(queries, axis=0),
        "target_points": np.concatenate(tracks, axis=0),
        "occluded": np.concatenate(occs, axis=0),
    }


# ---------------------------------------------------------------------------
# Core exporter
# ---------------------------------------------------------------------------

def _normalize_pixel_to_input_256(arr: np.ndarray) -> np.ndarray:
    """Normalize pixel-space [0, 255] to [0, 1] for 256×256 input."""
    return arr.astype(np.float32) / 255.0


def export_one_video(
    *,
    npz_path: Path,
    video_name: str,
    sequence_index: int,
    original_size_hw: Tuple[int, int],
    input_size_hw: Tuple[int, int],
    query_protocol: str,
    pkl_entry: Optional[Dict[str, Any]] = None,
    baseline_name: str = "",
    checkpoint_path: str = "",
    repo_commit: str = "",
    adapter_version: str = "",
) -> Dict[str, Any]:
    """Export a single video's repo-native .npz cache to unified schema.

    Args:
        npz_path: Path to .npz with 'tracks' (1, T, N, 2) and 'visibility' (1, T, N).
        video_name: Name of the video.
        sequence_index: Index in the dataset ordering.
        original_size_hw: Original (H, W) of the video.
        input_size_hw: Model input (H, W), typically (256, 256).
        query_protocol: 'first' or 'strided'.
        pkl_entry: Dict from the tapvid pkl with 'points', 'occluded', 'video'.
        baseline_name: Model name for the cache.
        checkpoint_path: Checkpoint path for metadata.
        repo_commit: Repo commit for metadata.
        adapter_version: Adapter version string.

    Returns:
        Unified cache record dict.
    """
    cached = np.load(npz_path)
    pred_tracks_raw = cached["tracks"]  # (1, T, N, 2) float32, [x, y] pixel 0-255
    pred_vis_raw = cached["visibility"]  # (1, T, N) bool, True=visible

    B, T, N, _ = pred_tracks_raw.shape
    assert B == 1, f"Expected batch=1, got {B}"

    # --- Get GT and queries from pkl entry ---
    if pkl_entry is None:
        raise ValueError(f"No pkl_entry provided for {video_name}")

    points = pkl_entry["points"].copy()  # (N_orig, T, 2) [x, y] normalized 0-1
    occluded = pkl_entry["occluded"].copy()  # (N_orig, T) bool, True=occluded

    # Scale points to pixel space (input_size)
    input_h, input_w = input_size_hw
    points_pixel = points * np.array([float(input_w), float(input_h)])  # [x, y] pixel space

    # Resize video (just for query generation, not stored)
    # We skip actual video loading since queries can be generated from GT alone

    if query_protocol == "first":
        converted = _sample_queries_first(occluded, points_pixel)
    elif query_protocol == "strided":
        converted = _sample_queries_strided(occluded, points_pixel)
    else:
        raise ValueError(f"Unknown query_protocol: {query_protocol}")

    query_points_px = converted["query_points"]  # (N_out, 3) [t, y, x] pixel
    gt_tracks_px = converted["target_points"]  # (N_out, T, 2) [x, y] pixel
    gt_occluded = converted["occluded"]  # (N_out, T) bool, True=occluded

    # Verify N matches
    assert query_points_px.shape[0] == N, (
        f"Query count mismatch after sampling: pkl gives {query_points_px.shape[0]}, "
        f"npz gives {N}"
    )

    # --- Convert predictions ---
    # pred_tracks_raw: (1, T, N, 2) [x, y] pixel 0-255
    # -> remove batch: (T, N, 2)
    # -> transpose: (N, T, 2)
    # -> swap xy->yx: (N, T, 2)
    # -> normalize by input: (N, T, 2) in [0, 1]
    pred_tracks_bhw = pred_tracks_raw[0]  # (T, N, 2)
    pred_tracks_nt = pred_tracks_bhw.transpose(1, 0, 2)  # (N, T, 2)
    pred_tracks_yx = pred_tracks_nt[..., [1, 0]]  # swap x,y -> y,x
    pred_tracks_norm = _normalize_pixel_to_input_256(pred_tracks_yx)  # / 255

    # --- Convert visibility ---
    # pred_vis_raw: (1, T, N) bool
    # -> remove batch: (T, N)
    # -> transpose: (N, T)
    pred_vis_bhw = pred_vis_raw[0]  # (T, N)
    pred_vis_nt = pred_vis_bhw.transpose(1, 0)  # (N, T)
    pred_visibility = pred_vis_nt.astype(np.bool_)

    # --- Convert GT ---
    # gt_tracks_px: (N, T, 2) [x, y] pixel
    # -> swap xy->yx: [y, x]
    # -> normalize by 255: [0, 1]
    gt_tracks_yx = gt_tracks_px[..., [1, 0]]
    gt_tracks_norm = _normalize_pixel_to_input_256(gt_tracks_yx)

    # gt_occluded: (N, T) bool, True=occluded
    # -> invert to visibility: True=visible
    gt_visibility = (~gt_occluded).astype(np.bool_)

    # --- Convert query points ---
    # query_points_px: (N, 3) [t, y, x] pixel
    # -> normalize by (input_size - 1) = 255 to match unified rescaler convention
    query_points_norm = np.zeros_like(query_points_px, dtype=np.float32)
    query_points_norm[:, 0] = query_points_px[:, 0]  # t stays as-is
    query_points_norm[:, 1] = query_points_px[:, 1] / 255.0  # y / (256-1)
    query_points_norm[:, 2] = query_points_px[:, 2] / 255.0  # x / (256-1)

    # --- Build record ---
    record = {
        "video_id": video_name,
        "sequence_index": sequence_index,
        "frame_count": T,
        "query_points": query_points_norm.astype(np.float32),
        "pred_tracks": pred_tracks_norm.astype(np.float32),
        "pred_visibility": pred_visibility,
        "gt_tracks": gt_tracks_norm.astype(np.float32),
        "gt_visibility": gt_visibility,
        "original_size": np.array(original_size_hw, dtype=np.int32),
        "model_input_size": np.array(input_size_hw, dtype=np.int32),
        "adapter_version": adapter_version,
        "raw_coordinate_note": (
            f"source_query_protocol={query_protocol}, "
            f"source_space=input{input_h}, "
            f"source_track_format=xy, "
            f"export_track_format=yx_normalized"
        ),
    }

    return record


def run_export(
    *,
    baseline_name: str,
    cache_dir: Path,
    pkl_path: Path,
    dataset_type: str,
    query_protocol: str,
    space: str,
    out_cache: Path,
    out_report: Path,
    checkpoint_path: str = "",
    repo_commit: str = "",
    adapter_version: str = "",
) -> Dict[str, Any]:
    """Run the full export pipeline.

    Args:
        baseline_name: Name for the model (e.g., 'trackon2_dinov3').
        cache_dir: Directory containing .npz prediction files.
        pkl_path: Path to the tapvid .pkl dataset.
        dataset_type: 'davis', 'kinetics', etc.
        query_protocol: 'first' or 'strided'.
        space: 'input' or 'original'.
        out_cache: Output path for the unified .pt cache.
        out_report: Output path for the export report.
        checkpoint_path: Checkpoint path for metadata.
        repo_commit: Repo commit hash.
        adapter_version: Adapter version string.

    Returns:
        Export report dict.
    """
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    video_names = sorted(list(pkl_data.keys()))
    num_videos = len(video_names)

    # Determine input size
    if space == "input":
        input_size_hw = (256, 256)
    else:
        raise ValueError(f"space={space!r} not yet supported in this bridge; use 'input'")

    records: List[Dict[str, Any]] = []
    per_video: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, video_name in enumerate(video_names):
        npz_path = cache_dir / f"{idx:06d}.npz"
        if not npz_path.exists():
            msg = f"Missing npz cache: {npz_path}"
            errors.append(msg)
            per_video.append({"video_id": video_name, "index": idx, "status": "error", "error": msg})
            continue

        entry = pkl_data[video_name]
        original_video = entry["video"]
        if hasattr(original_video, "shape"):
            original_size_hw = (int(original_video.shape[1]), int(original_video.shape[2]))
        else:
            original_size_hw = (480, 854)  # fallback

        try:
            record = export_one_video(
                npz_path=npz_path,
                video_name=video_name,
                sequence_index=idx,
                original_size_hw=original_size_hw,
                input_size_hw=input_size_hw,
                query_protocol=query_protocol,
                pkl_entry=entry,
                baseline_name=baseline_name,
                checkpoint_path=checkpoint_path,
                repo_commit=repo_commit,
                adapter_version=adapter_version,
            )
            records.append(record)
            per_video.append({
                "video_id": video_name,
                "index": idx,
                "status": "ok",
                "num_queries": int(record["query_points"].shape[0]),
                "frame_count": record["frame_count"],
                "original_size_hw": list(original_size_hw),
            })
        except Exception as exc:
            msg = f"[{video_name}] {exc}"
            errors.append(msg)
            per_video.append({"video_id": video_name, "index": idx, "status": "error", "error": str(exc)})

    # Build cache payload
    payload = {
        "schema_version": 1,
        "model_name": baseline_name,
        "repo_commit": repo_commit,
        "checkpoint_path": checkpoint_path,
        "dataset_name": f"tapvid_{dataset_type}",
        "split": "test",
        "protocol": f"{query_protocol}+{space}",
        "records": records,
    }

    save_attempt0_cache(out_cache, payload)

    report = {
        "baseline_name": baseline_name,
        "dataset_type": dataset_type,
        "query_protocol": query_protocol,
        "space": space,
        "input_size_hw": list(input_size_hw),
        "num_videos_total": num_videos,
        "num_videos_exported": len(records),
        "total_queries": sum(r["query_points"].shape[0] for r in records),
        "errors": errors,
        "per_video": per_video,
    }

    write_json_report(out_report, report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export repo-native .npz caches to Attempt 0 unified schema."
    )
    parser.add_argument("--baseline-name", type=str, required=True,
                        help="Model name (e.g., trackon2_dinov3).")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Directory with .npz prediction files.")
    parser.add_argument("--tapvid-pkl", type=str, required=True,
                        help="Path to the tapvid .pkl dataset.")
    parser.add_argument("--dataset-type", type=str, required=True,
                        help="Dataset type: davis, kinetics, etc.")
    parser.add_argument("--query-protocol", type=str, default="first",
                        choices=("first", "strided"),
                        help="Query protocol to generate GT alignment.")
    parser.add_argument("--space", type=str, default="input",
                        choices=("input", "original"),
                        help="Resolution space for normalization.")
    parser.add_argument("--out-cache", type=str, required=True,
                        help="Output path for unified .pt cache.")
    parser.add_argument("--out-report", type=str, default="",
                        help="Output path for export report JSON.")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Checkpoint path for metadata.")
    parser.add_argument("--repo-commit", type=str, default="",
                        help="Repo commit for metadata.")
    parser.add_argument("--adapter-version", type=str, default="",
                        help="Adapter version for metadata.")
    args = parser.parse_args()

    out_report = Path(args.out_report) if args.out_report else Path(args.out_cache).with_suffix(".export_report.json")

    report = run_export(
        baseline_name=args.baseline_name,
        cache_dir=Path(args.cache_dir),
        pkl_path=Path(args.tapvid_pkl),
        dataset_type=args.dataset_type,
        query_protocol=args.query_protocol,
        space=args.space,
        out_cache=Path(args.out_cache),
        out_report=out_report,
        checkpoint_path=args.checkpoint_path,
        repo_commit=args.repo_commit,
        adapter_version=args.adapter_version,
    )

    print(json.dumps({
        "num_exported": report["num_videos_exported"],
        "num_errors": len(report["errors"]),
        "total_queries": report["total_queries"],
    }, indent=2))

    if report["errors"]:
        print(f"[warn] {len(report['errors'])} export errors; see {out_report}")
    print(f"[ok] exported {report['num_videos_exported']}/{report['num_videos_total']} videos to {args.out_cache}")


if __name__ == "__main__":
    main()
