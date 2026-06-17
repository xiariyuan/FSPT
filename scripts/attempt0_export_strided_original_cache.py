#!/usr/bin/env python3
"""Export strided+original repo-native .npz caches to Attempt 0 unified schema.

Bridges the gap between repo-native strided+original evaluation caches and the
Attempt 0 unified schema.

Key differences from first+input exporter:
- query protocol: strided (every query_stride frames)
- resolution space: original (actual video dimensions, not 256×256)
- tracks normalized by (original_H - 1, original_W - 1), not 255
- query points normalized by (original_H - 1, original_W - 1), not 255
- model_input_size == original_size (no resize step)
- original_size stored per-video (varies by sample)
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

from utils.attempt0_schema import save_attempt0_cache, write_json_report


def _sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Sample first-frame queries (queried_first=True)."""
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))
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
    """Sample strided queries (queried_first=False)."""
    tracks = []
    occs = []
    queries = []

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


def _normalize_by_size(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Normalize pixel-space [0, H-1]×[0, W-1] to [0, 1] for original resolution."""
    scale_y = max(h - 1, 1)
    scale_x = max(w - 1, 1)
    result = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 2:
        result[:, 0] = arr[:, 0] / scale_y
        result[:, 1] = arr[:, 1] / scale_x
    elif arr.ndim == 3:
        result[..., 0] = arr[..., 0] / scale_y
        result[..., 1] = arr[..., 1] / scale_x
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim}")
    return result


def export_strided_original(
    *,
    npz_path: Path,
    video_name: str,
    sequence_index: int,
    original_size_hw: Tuple[int, int],
    query_protocol: str = "strided",
    pkl_entry: Optional[Dict[str, Any]] = None,
    baseline_name: str = "",
    adapter_version: str = "",
) -> Dict[str, Any]:
    """Export a single video's strided+original cache to unified schema.

    Args:
        npz_path: Path to .npz with 'tracks' and 'visibility'.
        video_name: Name of the video.
        sequence_index: Index in the dataset ordering.
        original_size_hw: Original (H, W) of the video.
        query_protocol: 'strided' for strided+original protocol.
        pkl_entry: Dict from tapvid pkl with 'points', 'occluded'.
        baseline_name: Model name for the cache.
        adapter_version: Adapter version string.

    Returns:
        Unified cache record dict.
    """
    cached = np.load(npz_path)
    pred_tracks_raw = cached["tracks"]  # (1, T, N, 2) [x, y] pixel
    pred_vis_raw = cached["visibility"]  # (1, T, N) bool

    B, T, N, _ = pred_tracks_raw.shape
    assert B == 1, f"Expected batch=1, got {B}"

    if pkl_entry is None:
        raise ValueError(f"No pkl_entry provided for {video_name}")

    points = pkl_entry["points"].copy()  # (N_orig, T, 2) [x, y] normalized 0-1
    occluded = pkl_entry["occluded"].copy()  # (N_orig, T) bool

    # Scale points to original pixel space
    orig_h, orig_w = original_size_hw
    points_pixel = points * np.array([float(orig_w), float(orig_h)])

    if query_protocol == "first":
        converted = _sample_queries_first(occluded, points_pixel)
    elif query_protocol == "strided":
        converted = _sample_queries_strided(occluded, points_pixel)
    else:
        raise ValueError(f"Unknown query_protocol: {query_protocol}")

    query_points_px = converted["query_points"]  # (N_out, 3) [t, y, x] pixel
    gt_tracks_px = converted["target_points"]  # (N_out, T, 2) [x, y] pixel
    gt_occluded = converted["occluded"]  # (N_out, T) bool

    assert query_points_px.shape[0] == N, (
        f"Query count mismatch: pkl gives {query_points_px.shape[0]}, npz gives {N}"
    )

    # --- Convert predictions ---
    # pred_tracks_raw: (1, T, N, 2) [x, y] pixel
    # -> (T, N, 2) -> (N, T, 2) -> [y, x] -> normalize by (orig_h-1, orig_w-1)
    pred_tracks_bhw = pred_tracks_raw[0]
    pred_tracks_nt = pred_tracks_bhw.transpose(1, 0, 2)
    pred_tracks_yx = pred_tracks_nt[..., [1, 0]]
    pred_tracks_norm = _normalize_by_size(pred_tracks_yx, orig_h, orig_w)

    # --- Convert visibility ---
    pred_vis_bhw = pred_vis_raw[0]
    pred_vis_nt = pred_vis_bhw.transpose(1, 0)
    pred_visibility = pred_vis_nt.astype(np.bool_)

    # --- Convert GT ---
    gt_tracks_yx = gt_tracks_px[..., [1, 0]]
    gt_tracks_norm = _normalize_by_size(gt_tracks_yx, orig_h, orig_w)
    gt_visibility = (~gt_occluded).astype(np.bool_)

    # --- Convert query points ---
    # query_points_px: (N, 3) [t, y, x] pixel in original space
    # -> keep t as frame index, normalize y by (orig_h-1), x by (orig_w-1)
    query_points_norm = np.zeros_like(query_points_px, dtype=np.float32)
    query_points_norm[:, 0] = query_points_px[:, 0]  # frame index stays in [0, T-1]
    query_points_norm[:, 1] = query_points_px[:, 1] / max(orig_h - 1, 1)  # y
    query_points_norm[:, 2] = query_points_px[:, 2] / max(orig_w - 1, 1)  # x

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
        "model_input_size": np.array(original_size_hw, dtype=np.int32),  # same as original
        "adapter_version": adapter_version,
        "raw_coordinate_note": (
            f"source_query_protocol={query_protocol}, "
            f"source_space=original{orig_h}x{orig_w}, "
            f"source_track_format=xy, "
            f"normalization_denominator=[y:{orig_h-1}, x:{orig_w-1}]"
        ),
    }
    return record


def get_video_original_sizes(pkl_path: str) -> Dict[str, Tuple[int, int]]:
    """Extract original video sizes from TAP-Vid pkl."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    sizes = {}
    if isinstance(data, dict):
        for video_name, entry in data.items():
            frames = entry.get("video")
            if frames is None:
                frames = entry.get("frames")
            if frames is not None:
                if isinstance(frames[0], bytes):
                    import io
                    from PIL import Image
                    byteio = io.BytesIO(frames[0])
                    img = Image.open(byteio)
                    sizes[video_name] = (img.size[1], img.size[0])  # (H, W)
                else:
                    sizes[video_name] = (frames[0].shape[0], frames[0].shape[1])
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Export strided+original caches to unified schema")
    parser.add_argument("--baseline-name", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Directory containing .npz files (from eval script)")
    parser.add_argument("--tapvid-pkl", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, default="davis")
    parser.add_argument("--query-protocol", type=str, default="strided",
                        choices=["first", "strided"])
    parser.add_argument("--out-cache", type=str, required=True)
    parser.add_argument("--out-report", type=str, required=True)
    parser.add_argument("--adapter-version", type=str, default="strided_original_bridge_v1")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    npz_files = sorted(cache_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    # Load pkl
    with open(args.tapvid_pkl, "rb") as f:
        pkl_data = pickle.load(f)

    # Get video names from pkl (dataset ordering)
    if isinstance(pkl_data, dict):
        video_names = sorted(list(pkl_data.keys()))
    else:
        raise ValueError("Expected pkl to be a dict")

    # Get original sizes
    original_sizes = get_video_original_sizes(args.tapvid_pkl)

    records = []
    for seq_idx, npz_file in enumerate(npz_files):
        video_name = video_names[seq_idx]
        pkl_entry = pkl_data.get(video_name)
        if pkl_entry is None:
            print(f"Warning: no pkl entry for {video_name}, skipping")
            continue

        original_hw = original_sizes.get(video_name, (256, 256))

        record = export_strided_original(
            npz_path=npz_file,
            video_name=video_name,
            sequence_index=seq_idx,
            original_size_hw=original_hw,
            query_protocol=args.query_protocol,
            pkl_entry=pkl_entry,
            baseline_name=args.baseline_name,
            adapter_version=args.adapter_version,
        )
        records.append(record)

    payload = {
        "schema_version": 1,
        "model_name": args.baseline_name,
        "repo_commit": "",
        "checkpoint_path": "",
        "dataset_name": f"tapvid_{args.dataset_type}",
        "split": "test",
        "protocol": f"{args.query_protocol}+original",
        "records": records,
    }

    save_attempt0_cache(args.out_cache, payload)
    print(f"Exported {len(records)} records to {args.out_cache}")

    # Write report
    report = {
        "baseline_name": args.baseline_name,
        "query_protocol": args.query_protocol,
        "metric_resolution_mode": "original",
        "num_records": len(records),
        "npz_files_found": len(npz_files),
        "adapter_version": args.adapter_version,
        "original_sizes": {k: list(v) for k, v in original_sizes.items()},
    }
    write_json_report(args.out_report, report)
    print(f"Report written to {args.out_report}")


if __name__ == "__main__":
    main()
