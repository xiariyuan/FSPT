#!/usr/bin/env python3
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
COTRACKER_ROOT = PROJECT_ROOT / "baselines" / "cotracker"
if str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return ""


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_record(sample: Dict[str, Any], pred_tracks_xy_tn: torch.Tensor, pred_vis_tn: torch.Tensor, idx: int, model_name: str) -> Dict[str, Any]:
    tracks_xy_tn = pred_tracks_xy_tn[0].detach().cpu().float().numpy()
    vis_tn = pred_vis_tn[0].detach().cpu().bool().numpy()
    tracks_yx_nt = tracks_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32)
    pred_vis_nt = vis_tn.transpose(1, 0).astype(np.bool_)

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
        "pred_visibility": pred_vis_nt,
        "gt_tracks": gt.astype(np.float32),
        "gt_visibility": (~occ).astype(np.bool_),
        "original_size": np.asarray([h, w], dtype=np.int32),
        "model_input_size": np.asarray([h, w], dtype=np.int32),
        "adapter_version": "cotracker3_rgb_stacking_strided_original_v1",
        "raw_coordinate_note": "RGB-Stacking CoTracker output xy pixel at original 256x256; converted to unified normalized yx by original_size denominators [H-1,W-1]. Queries are [t,y,x] normalized.",
        "model_name": model_name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--model", choices=["offline", "online"], default="offline")
    ap.add_argument("--checkpoint-offline", default="baselines/cotracker/checkpoints/scaled_offline.pth")
    ap.add_argument("--checkpoint-online", default="baselines/cotracker/checkpoints/scaled_online.pth")
    ap.add_argument("--out-cache", default="")
    ap.add_argument("--out-report", default="")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-videos", type=int, default=0)
    ap.add_argument("--query-stride", type=int, default=5)
    ap.add_argument("--query-batch-size", type=int, default=128)
    args = ap.parse_args()

    model_name = f"cotracker3_{args.model}_rgb_stacking"
    out_dir = Path("outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke")
    out_cache = Path(args.out_cache) if args.out_cache else out_dir / f"{model_name}_1video.pt"
    out_report = Path(args.out_report) if args.out_report else out_dir / f"{model_name}_1video_report.json"
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CoTracker export in this script.")

    print("loading RGB-Stacking dataset...", flush=True)
    ds = TAPVidRGBStackingDataset(
        root=args.data_root,
        query_mode="strided",
        query_stride=args.query_stride,
        max_videos=args.max_videos,
        start_index=args.start_index,
        num_videos=args.num_videos,
    )
    n = len(ds)

    print("loading CoTracker3", args.model, flush=True)
    t0 = time.time()
    if args.model == "online":
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint_online).cuda().eval()
    else:
        model = CoTrackerPredictor(checkpoint=args.checkpoint_offline).cuda().eval()
    print({"model_load_sec": round(time.time() - t0, 2)}, flush=True)

    records: List[Dict[str, Any]] = []
    per_video = []
    for idx in range(n):
        s = ds[idx]
        name = str(s.get("video_name", f"rgb_stacking_{idx:06d}"))
        h, w = int(s["original_size"][0]), int(s["original_size"][1])
        video = s["video"].unsqueeze(0).cuda(non_blocking=True) * 255.0
        q = s["query_points"].clone().unsqueeze(0).cuda(non_blocking=True)
        queries = torch.zeros_like(q)
        queries[:, :, 0] = q[:, :, 0]
        queries[:, :, 1] = q[:, :, 2] * max(w - 1, 1)
        queries[:, :, 2] = q[:, :, 1] * max(h - 1, 1)

        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        qbs = max(1, int(args.query_batch_size))
        n_queries = int(queries.shape[1])
        track_chunks = []
        vis_chunks = []
        for qs in range(0, n_queries, qbs):
            qe = min(n_queries, qs + qbs)
            q_chunk = queries[:, qs:qe]
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                if args.model == "online":
                    model(video_chunk=video, is_first_step=True, queries=q_chunk, add_support_grid=False, grid_size=0)
                    t_chunk = None
                    v_chunk = None
                    for ind in range(0, video.shape[1] - model.step, model.step):
                        chunk = video[:, ind : ind + model.step * 2]
                        t_chunk, v_chunk = model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
                    if t_chunk is None or v_chunk is None:
                        raise RuntimeError("CoTracker3 online predictor produced no chunks")
                else:
                    t_chunk, v_chunk = model(video, queries=q_chunk, grid_size=0)
            torch.cuda.synchronize()
            track_chunks.append(t_chunk.detach().cpu())
            vis_chunks.append(v_chunk.detach().cpu())
            print({"idx": idx, "query_chunk": [qs, qe], "total_queries": n_queries}, flush=True)
            del q_chunk, t_chunk, v_chunk
            torch.cuda.empty_cache()
        tracks = torch.cat(track_chunks, dim=2)
        vis = torch.cat(vis_chunks, dim=2)
        sec = time.time() - t1
        rec = _make_record(s, tracks, vis, idx, model_name=model_name)
        records.append(rec)
        info = {
            "idx": idx,
            "video_id": name,
            "frames": int(s["video"].shape[0]),
            "queries": int(s["query_points"].shape[0]),
            "query_batch_size": int(args.query_batch_size),
            "size_hw": [h, w],
            "sec": round(sec, 3),
            "vis_rate": round(float(rec["pred_visibility"].mean()), 4),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        }
        per_video.append(info)
        print(info, flush=True)
        del video, q, queries, tracks, vis, track_chunks, vis_chunks
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "repo_commit": _git_commit(),
        "checkpoint_path": str(args.checkpoint_online if args.model == "online" else args.checkpoint_offline),
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
        "start_index": int(args.start_index),
        "num_videos_arg": int(args.num_videos),
        "n_records": len(records),
        "n_queries": int(sum(r["query_points"].shape[0] for r in records)),
        "total_sec": round(sum(v["sec"] for v in per_video), 3),
        "mean_vis_rate": round(float(np.mean([v["vis_rate"] for v in per_video])), 4) if per_video else None,
        "per_video": per_video,
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("CACHE_OK", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
