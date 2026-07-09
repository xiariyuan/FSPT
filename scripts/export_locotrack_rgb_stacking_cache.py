#!/usr/bin/env python3
"""Export LocoTrack predictions to the unified RGB-Stacking cache schema.

This is a feasibility/export script for the ReEntry-VisGuard improvement-paper
cross-source experiment. It mirrors the CoTracker RGB-Stacking exporter but uses
`baselines/track_on/ensemble/locotrack/LocoTrackPredictor`.

Required checkpoint (per baselines/track_on/ensemble/README.md):

    https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt

or Anthro-LocoTrack if intentionally testing that variant.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRACKON_ROOT = PROJECT_ROOT / "baselines" / "track_on"
if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))

from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset
from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return ""


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_record(sample: Dict[str, Any], pred_tracks_xy_nt: torch.Tensor, pred_vis_nt: torch.Tensor, idx: int, model_name: str) -> Dict[str, Any]:
    """Convert LocoTrack output to unified normalized yx schema.

    LocoTrackPredictor returns:
      tracks: (B,N,T,2) in original-resolution xy pixels
      visibility: (B,N,T) boolean
    """
    tracks_xy_nt = pred_tracks_xy_nt[0].detach().cpu().float().numpy()
    vis_nt = pred_vis_nt[0].detach().cpu().bool().numpy()
    tracks_yx_nt = tracks_xy_nt[..., [1, 0]].astype(np.float32)

    q = _to_numpy(sample["query_points"]).astype(np.float32)
    gt = _to_numpy(sample["target_points"]).astype(np.float32)
    occ = _to_numpy(sample["occluded"]).astype(np.bool_)
    osz = _to_numpy(sample["original_size"]).astype(np.int32).reshape(-1)
    h, w = int(osz[0]), int(osz[1])

    pred_norm = np.empty_like(tracks_yx_nt, dtype=np.float32)
    pred_norm[..., 0] = tracks_yx_nt[..., 0] / max(h - 1, 1)
    pred_norm[..., 1] = tracks_yx_nt[..., 1] / max(w - 1, 1)

    return {
        "video_id": str(sample.get("video_name", f"rgb_stacking_{idx:06d}")),
        "sequence_index": int(idx),
        "frame_count": int(gt.shape[1]),
        "query_points": q.astype(np.float32),
        "pred_tracks": pred_norm.astype(np.float32),
        "pred_visibility": vis_nt.astype(np.bool_),
        "gt_tracks": gt.astype(np.float32),
        "gt_visibility": (~occ).astype(np.bool_),
        "original_size": np.asarray([h, w], dtype=np.int32),
        "model_input_size": np.asarray([256, 256], dtype=np.int32),
        "adapter_version": "locotrack_rgb_stacking_strided_original_v1",
        "raw_coordinate_note": "LocoTrack output is xy pixel at original resolution after predictor rescale; converted to unified normalized yx by original_size denominators [H-1,W-1]. Queries are [t,y,x] normalized in stored cache and [t,x,y] pixels for LocoTrack predictor.",
        "model_name": model_name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LocoTrack RGB-Stacking predictions to unified cache schema.")
    ap.add_argument("--data-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--checkpoint", required=True, help="Path to locotrack_base.ckpt or compatible LocoTrack checkpoint.")
    ap.add_argument("--model-size", default="base", choices=["base", "small"])
    ap.add_argument("--out-cache", default="outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_1video.pt")
    ap.add_argument("--out-report", default="outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_1video_report.json")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-videos", type=int, default=1)
    ap.add_argument("--query-stride", type=int, default=5)
    ap.add_argument("--query-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"LocoTrack checkpoint not found: {ckpt}\n"
            "Download per baselines/track_on/ensemble/README.md, e.g.\n"
            "wget -P path/to/ckpt https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt"
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    out_cache = Path(args.out_cache)
    out_report = Path(args.out_report)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    print("loading RGB-Stacking dataset...", flush=True)
    ds = TAPVidRGBStackingDataset(
        root=args.data_root,
        query_mode="strided",
        query_stride=args.query_stride,
        max_videos=args.max_videos,
        start_index=args.start_index,
        num_videos=args.num_videos,
    )

    print("loading LocoTrack", {"checkpoint": str(ckpt), "model_size": args.model_size, "device": str(device)}, flush=True)
    t0 = time.time()
    predictor = LocoTrackPredictor(str(ckpt), model_size=args.model_size).to(device).eval()
    print({"model_load_sec": round(time.time() - t0, 2)}, flush=True)

    model_name = f"locotrack_{args.model_size}_rgb_stacking"
    records: List[Dict[str, Any]] = []
    per_video: List[Dict[str, Any]] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        name = str(sample.get("video_name", f"rgb_stacking_{idx:06d}"))
        h, w = int(sample["original_size"][0]), int(sample["original_size"][1])
        video = sample["video"].unsqueeze(0).to(device, non_blocking=True) * 255.0
        q_norm = sample["query_points"].clone().unsqueeze(0).to(device, non_blocking=True)

        # LocoTrackPredictor expects queries as [t, x, y] in original pixels.
        queries_xy = torch.zeros_like(q_norm)
        queries_xy[:, :, 0] = q_norm[:, :, 0]
        queries_xy[:, :, 1] = q_norm[:, :, 2] * max(w - 1, 1)
        queries_xy[:, :, 2] = q_norm[:, :, 1] * max(h - 1, 1)

        qbs = max(1, int(args.query_batch_size))
        n_queries = int(queries_xy.shape[1])
        track_chunks = []
        vis_chunks = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        for qs in range(0, n_queries, qbs):
            qe = min(n_queries, qs + qbs)
            q_chunk = queries_xy[:, qs:qe]
            with torch.no_grad():
                tracks_chunk, vis_chunk = predictor(video, q_chunk)
            track_chunks.append(tracks_chunk.detach().cpu())
            vis_chunks.append(vis_chunk.detach().cpu())
            print({"idx": idx, "query_chunk": [qs, qe], "total_queries": n_queries}, flush=True)
            del q_chunk, tracks_chunk, vis_chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()

        tracks = torch.cat(track_chunks, dim=1)
        visibility = torch.cat(vis_chunks, dim=1)
        sec = time.time() - t1
        record = _make_record(sample, tracks, visibility, idx, model_name=model_name)
        records.append(record)
        info = {
            "idx": int(idx),
            "video_id": name,
            "frames": int(sample["video"].shape[0]),
            "queries": int(sample["query_points"].shape[0]),
            "query_batch_size": int(args.query_batch_size),
            "size_hw": [h, w],
            "sec": round(sec, 3),
            "vis_rate": round(float(record["pred_visibility"].mean()), 4),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if device.type == "cuda" else None,
        }
        per_video.append(info)
        print(info, flush=True)
        del video, q_norm, queries_xy, tracks, visibility, track_chunks, vis_chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "repo_commit": _git_commit(),
        "checkpoint_path": str(ckpt),
        "dataset_name": "tapvid_rgb_stacking",
        "start_index": int(args.start_index),
        "num_videos_arg": int(args.num_videos),
        "split": "test",
        "protocol": f"strided_original_qs{args.query_stride}",
        "records": records,
    }
    torch.save(payload, out_cache)
    report = {
        "out_cache": str(out_cache),
        "checkpoint_path": str(ckpt),
        "start_index": int(args.start_index),
        "num_videos_arg": int(args.num_videos),
        "n_records": len(records),
        "n_queries": int(sum(r["query_points"].shape[0] for r in records)),
        "total_sec": round(sum(v["sec"] for v in per_video), 3),
        "mean_vis_rate": round(float(np.mean([v["vis_rate"] for v in per_video])), 4) if per_video else None,
        "per_video": per_video,
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CACHE_OK", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
