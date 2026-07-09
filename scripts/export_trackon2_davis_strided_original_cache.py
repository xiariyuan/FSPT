#!/usr/bin/env python3
"""Export TrackOn2 predictions on TAPVid-DAVIS strided/original into FSPT cache schema.

This is a protocol-parity exporter, not a ReEntry experiment.

Goal:
  - rebuild TrackOn2 DAVIS strided/original cache from raw TAPVid-DAVIS data;
  - preserve FSPT cache schema used by current evaluation/audit scripts;
  - run small smoke exports before any 30-video run.

Coordinate conventions:
  FSPT cache:
    query_points: [t, y_norm, x_norm]
    pred_tracks:  [N, T, 2] normalized [y, x], denominator [H-1, W-1]

  TrackOn2 predictor input:
    queries: [t, x_px, y_px]

  TrackOn2 predictor output:
    tracks: [B, T, N, 2] pixel [x, y]
    visibility: [B, T, N]
"""
from __future__ import annotations

import argparse
import json
import os
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

from datasets.tapvid_davis import TAPVidDAVISDataset  # noqa: E402

TRACKON_ROOT = PROJECT_ROOT / "baselines" / "track_on"
if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))

from model.trackon_predictor import Predictor as TrackOnPredictor  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402


DEFAULT_DINOV3_LOCAL_DIR = PROJECT_ROOT / "third_party_weights" / "dinov3" / "facebook_dinov3_vits16plus_pretrain_lvd1689m"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return ""


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def _summ(vals: List[float]) -> Dict[str, Any]:
    arr = np.asarray([float(v) for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _query_anchor_errors_px(record: Dict[str, Any]) -> List[float]:
    q = npy(record["query_points"], np.float32)
    pred = npy(record["pred_tracks"], np.float32)
    osz = npy(record["original_size"], np.float32).reshape(-1)
    h, w = float(osz[0]), float(osz[1])
    n, t_len = pred.shape[:2]
    out: List[float] = []
    for i in range(n):
        t = int(round(float(q[i, 0])))
        if 0 <= t < t_len:
            dy = (float(pred[i, t, 0]) - float(q[i, 1])) * max(h - 1.0, 1.0)
            dx = (float(pred[i, t, 1]) - float(q[i, 2])) * max(w - 1.0, 1.0)
            out.append(float((dy * dy + dx * dx) ** 0.5))
    return out


def _make_record(
    sample: Dict[str, Any],
    pred_tracks_xy_tn: torch.Tensor,
    pred_vis_tn: torch.Tensor,
    *,
    idx: int,
    config_summary: Dict[str, Any],
) -> Dict[str, Any]:
    tracks_xy_tn = pred_tracks_xy_tn[0].detach().cpu().float().numpy()  # T,N,2 xy pixel
    vis_tn = pred_vis_tn[0].detach().cpu().bool().numpy()  # T,N

    tracks_yx_nt = tracks_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32)
    pred_vis_nt = vis_tn.transpose(1, 0).astype(np.bool_)

    q = npy(sample["query_points"], np.float32)
    gt = npy(sample["target_points"], np.float32)
    occ = npy(sample["occluded"], bool)
    gv = (~occ).astype(np.bool_)
    osz = npy(sample["original_size"], np.int32).reshape(-1)
    h, w = int(osz[0]), int(osz[1])

    pred_norm = np.empty_like(tracks_yx_nt, dtype=np.float32)
    pred_norm[..., 0] = tracks_yx_nt[..., 0] / max(h - 1, 1)
    pred_norm[..., 1] = tracks_yx_nt[..., 1] / max(w - 1, 1)

    rec = {
        "video_id": str(sample.get("video_name", f"davis_{idx:06d}")),
        "sequence_index": int(idx),
        "frame_count": int(gt.shape[1]),
        "query_points": q.astype(np.float32),
        "pred_tracks": pred_norm.astype(np.float32),
        "pred_visibility": pred_vis_nt,
        "gt_tracks": gt.astype(np.float32),
        "target_points": gt.astype(np.float32),
        "gt_visibility": gv,
        "occluded": occ.astype(np.bool_),
        "original_size": np.asarray([h, w], dtype=np.int32),
        "model_input_size": np.asarray(config_summary.get("model_input_size", [h, w]), dtype=np.int32),
        "model_name": "trackon2_dinov3_davis_strided_original",
        "adapter_version": "trackon2_davis_strided_original_v1",
        "raw_coordinate_note": (
            "TrackOn2 output xy pixel; converted to FSPT normalized yx by original_size [H-1,W-1]. "
            "Queries are FSPT [t,y_norm,x_norm]."
        ),
    }
    return rec


def _prepare_video(sample: Dict[str, Any], *, input_scale: str, device: torch.device) -> torch.Tensor:
    # sample['video']: (T,3,H,W), usually float 0..1 from TAPVidDAVISDataset
    video = torch.as_tensor(sample["video"]).float()
    if video.ndim != 4:
        raise ValueError(f"expected sample video shape (T,3,H,W), got {tuple(video.shape)}")
    if input_scale == "uint8":
        if float(video.max().item()) <= 1.5:
            video = video * 255.0
    elif input_scale == "unit":
        if float(video.max().item()) > 1.5:
            video = video / 255.0
    else:
        raise ValueError(f"unsupported input_scale={input_scale!r}")
    return video.unsqueeze(0).to(device, non_blocking=True)


def _prepare_queries(sample: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    q = torch.as_tensor(sample["query_points"]).float()
    osz = torch.as_tensor(sample["original_size"]).float().reshape(-1)
    h, w = float(osz[0].item()), float(osz[1].item())
    queries = torch.zeros_like(q)
    queries[:, 0] = q[:, 0]
    queries[:, 1] = q[:, 2] * max(w - 1.0, 1.0)  # x px
    queries[:, 2] = q[:, 1] * max(h - 1.0, 1.0)  # y px
    return queries.unsqueeze(0).to(device, non_blocking=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export TrackOn2 DAVIS strided/original cache")
    ap.add_argument("--dataset-root", default="/gemini/code/datasets/tapvid_davis")
    ap.add_argument("--checkpoint", default="baselines/track_on/checkpoints_trackon2_dinov3.pt")
    ap.add_argument("--config", default="baselines/track_on/config/test.yaml")
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-parity", required=True)
    ap.add_argument("--query-mode", default="strided", choices=["strided", "first"])
    ap.add_argument("--query-stride", type=int, default=5)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=1)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--support-grid-size", type=int, default=20)
    ap.add_argument("--memory-policy", default="unconditional", choices=["unconditional", "visibility_selective"])
    ap.add_argument("--delta-v", type=float, default=None)
    ap.add_argument("--input-scale", default="uint8", choices=["uint8", "unit"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dinov3-local-dir", default=str(DEFAULT_DINOV3_LOCAL_DIR))
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dinov3_dir = Path(args.dinov3_local_dir)
    if dinov3_dir.is_dir() and not os.environ.get("DINOV3_LOCAL_DIR"):
        os.environ["DINOV3_LOCAL_DIR"] = str(dinov3_dir)

    out_cache = Path(args.out_cache)
    out_report = Path(args.out_report)
    out_parity = Path(args.out_parity)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_parity.parent.mkdir(parents=True, exist_ok=True)

    print("loading TAPVid-DAVIS dataset...", flush=True)
    dataset = TAPVidDAVISDataset(
        root=args.dataset_root,
        resolution=None,
        augmentation=False,
        query_mode=args.query_mode,
        query_stride=int(args.query_stride),
        points_order="xy",
    )

    print("loading TrackOn2...", flush=True)
    t0 = time.time()
    model_args = load_args_from_yaml(args.config)
    model_args.grad_checkpoint = False
    model_args.memory_update_policy = args.memory_policy
    if args.delta_v is not None:
        model_args.delta_v = float(args.delta_v)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model = TrackOnPredictor(
        model_args,
        checkpoint_path=str(checkpoint_path),
        support_grid_size=int(args.support_grid_size),
    ).to(device).eval()
    model_input_size = list(getattr(model.model, "input_size", []))
    config_summary = {
        "model_input_size": model_input_size,
        "memory_policy": str(model_args.memory_update_policy),
        "delta_v": float(model_args.delta_v),
        "input_scale": args.input_scale,
        "support_grid_size": int(args.support_grid_size),
        "dinov3_local_dir": os.environ.get("DINOV3_LOCAL_DIR", ""),
    }
    print({"model_load_sec": round(time.time() - t0, 2), **config_summary}, flush=True)

    start = max(0, int(args.start_index))
    end = len(dataset) if int(args.max_videos) <= 0 else min(len(dataset), start + int(args.max_videos))
    records: List[Dict[str, Any]] = []
    per_video: List[Dict[str, Any]] = []
    all_anchor_errors: List[float] = []

    for idx in range(start, end):
        sample = dataset[idx]
        if int(args.max_queries) > 0:
            mq = int(args.max_queries)
            for key in ("query_points", "target_points", "occluded"):
                sample[key] = sample[key][:mq]
        video_id = str(sample.get("video_name", f"davis_{idx:06d}"))
        q_np = npy(sample["query_points"], np.float32)
        osz = npy(sample["original_size"], np.int32).reshape(-1)
        video = _prepare_video(sample, input_scale=args.input_scale, device=device)
        queries = _prepare_queries(sample, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    tracks, vis = model(video, queries)
                torch.cuda.synchronize()
            else:
                tracks, vis = model(video, queries)
        sec = time.time() - t1

        rec = _make_record(sample, tracks, vis, idx=idx, config_summary=config_summary)
        anchor = _query_anchor_errors_px(rec)
        all_anchor_errors.extend(anchor)
        records.append(rec)

        info = {
            "idx": int(idx),
            "video_id": video_id,
            "frames": int(rec["frame_count"]),
            "queries": int(q_np.shape[0]),
            "size_hw": [int(osz[0]), int(osz[1])],
            "sec": round(sec, 3),
            "pred_vis_rate": round(float(rec["pred_visibility"].mean()), 6),
            "gt_vis_rate": round(float(rec["gt_visibility"].mean()), 6),
            "anchor_err_px_mean": round(float(np.mean(anchor)), 6) if anchor else None,
            "anchor_err_px_max": round(float(np.max(anchor)), 6) if anchor else None,
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if device.type == "cuda" else None,
        }
        per_video.append(info)
        print(info, flush=True)

        del video, queries, tracks, vis
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "model_name": "trackon2_dinov3_davis_strided_original",
        "repo_commit": _git_commit(),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str((PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()),
        "dataset_name": "tapvid_davis",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "protocol": f"{args.query_mode}/original",
        "query_mode": str(args.query_mode),
        "query_stride": int(args.query_stride),
        "split": "davis",
        "adapter_version": "trackon2_davis_strided_original_v1",
        "config_summary": config_summary,
        "records": records,
    }
    torch.save(payload, out_cache)

    report = {
        "out_cache": str(out_cache),
        "n_records": int(len(records)),
        "n_queries": int(sum(r["query_points"].shape[0] for r in records)),
        "total_model_sec": round(float(sum(v["sec"] for v in per_video)), 3),
        "mean_pred_vis_rate": round(float(np.mean([v["pred_vis_rate"] for v in per_video])), 6) if per_video else None,
        "mean_gt_vis_rate": round(float(np.mean([v["gt_vis_rate"] for v in per_video])), 6) if per_video else None,
        "anchor_error_px": _summ(all_anchor_errors),
        "per_video": per_video,
        "config_summary": config_summary,
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    parity = {
        "cache": str(out_cache),
        "pass_query_anchor_mean_le_1px": bool((report["anchor_error_px"]["mean"] or 999.0) <= 1.0),
        "pass_query_anchor_max_le_3px": bool((report["anchor_error_px"]["max"] or 999.0) <= 3.0),
        "anchor_error_px": report["anchor_error_px"],
        "visibility_rate": {
            "pred_mean": report["mean_pred_vis_rate"],
            "gt_mean": report["mean_gt_vis_rate"],
        },
        "decision": "geometry_smoke_only__standard_metrics_not_yet_evaluated",
        "next_required": "run audit_trackon2_cache_geometry.py and standard/AJ_RD evaluators before any paper use",
    }
    out_parity.write_text(json.dumps(parity, indent=2, ensure_ascii=False))
    print("CACHE_OK", json.dumps(report, ensure_ascii=False), flush=True)
    print("PARITY_SMOKE", json.dumps(parity, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
