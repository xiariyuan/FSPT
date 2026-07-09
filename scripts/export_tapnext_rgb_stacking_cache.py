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
TRACKON_ROOT = PROJECT_ROOT / "baselines" / "track_on"
if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))

from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset
from ensemble.tapnext.tapnext_predictor import TAPNextPredictor


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return ""


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_record(sample: Dict[str, Any], pred_tracks_xy_tn: torch.Tensor, pred_vis_tn: torch.Tensor, idx: int, model_input_size) -> Dict[str, Any]:
    tracks_xy_tn = pred_tracks_xy_tn[0].detach().cpu().float().numpy()  # T,N,2 xy pixel
    vis_tn = pred_vis_tn[0].detach().cpu().bool().numpy()  # T,N
    tracks_yx_nt = tracks_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32)  # N,T,2 yx pixel
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
        "model_input_size": np.asarray(model_input_size, dtype=np.int32),
        "adapter_version": "tapnext_rgb_stacking_strided_original_v1",
        "raw_coordinate_note": "RGB-Stacking TAPNext output xy pixel at original 256x256; converted to unified normalized yx by original_size denominators [H-1,W-1]. Queries are [t,y,x] normalized.",
        "model_name": "tapnext_bootstapnext_rgb_stacking",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--checkpoint", default="checkpoints/tapnext/bootstapnext_ckpt.npz")
    ap.add_argument("--out-cache", default="outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video.pt")
    ap.add_argument("--out-report", default="outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video_report.json")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-videos", type=int, default=0)
    ap.add_argument("--query-stride", type=int, default=5)
    args = ap.parse_args()

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
    n = len(ds)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TAPNext export in this script.")

    print("loading TAPNext...", flush=True)
    t0 = time.time()
    model = TAPNextPredictor(args.checkpoint).cuda().eval()
    print({"model_load_sec": round(time.time() - t0, 2)}, flush=True)

    records: List[Dict[str, Any]] = []
    per_video = []
    for idx in range(n):
        s = ds[idx]
        name = str(s.get("video_name", f"rgb_stacking_{idx:06d}"))
        h, w = int(s["original_size"][0]), int(s["original_size"][1])
        video = s["video"].unsqueeze(0).cuda(non_blocking=True) * 255.0  # B,T,C,H,W
        q = s["query_points"].clone().unsqueeze(0).cuda(non_blocking=True)  # B,N,[t,y_norm,x_norm]
        queries = torch.zeros_like(q)
        queries[:, :, 0] = q[:, :, 0]
        queries[:, :, 1] = q[:, :, 2] * max(w - 1, 1)  # x pixel
        queries[:, :, 2] = q[:, :, 1] * max(h - 1, 1)  # y pixel

        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            tracks, vis = model(video, queries)
        torch.cuda.synchronize()
        sec = time.time() - t1
        rec = _make_record(s, tracks, vis, idx, model_input_size=(h, w))
        records.append(rec)
        info = {
            "idx": idx,
            "video_id": name,
            "frames": int(s["video"].shape[0]),
            "queries": int(s["query_points"].shape[0]),
            "size_hw": [h, w],
            "sec": round(sec, 3),
            "vis_rate": round(float(rec["pred_visibility"].mean()), 4),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        }
        per_video.append(info)
        print(info, flush=True)
        del video, q, queries, tracks, vis
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "model_name": "tapnext_bootstapnext_rgb_stacking",
        "repo_commit": _git_commit(),
        "checkpoint_path": str(Path(args.checkpoint)),
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
