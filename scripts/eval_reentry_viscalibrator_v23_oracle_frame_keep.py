#!/usr/bin/env python3
"""Oracle evaluation for V2.3 frame-keep / risk decoding.

This script answers a prerequisite question before training V2.3:

    If we had a perfect keep/drop classifier for V1-proposed recovery frames,
    how much improvement is theoretically available?

It uses GT only for oracle decisions, never as an inference method.  It should be
used for upper-bound analysis, not as a deployable method.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics  # noqa: E402
from models.reentry_viscalibrator import build_model_from_checkpoint_payload  # noqa: E402
from utils.reentry_frame_keep_features import frame_keep_utility  # noqa: E402
from utils.reentry_viscalibrator_features import (  # noqa: E402
    CandidateConfig,
    build_candidate_example,
    check_alignment,
    find_candidate_triggers,
    npy,
)


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def run_ajrd(cache_path: Path, output_json: Path) -> Dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_aj_rd_from_cache.py",
            "--cache-path",
            str(cache_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(output_json.read_text(encoding="utf-8"))


def eval_standard(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    q_total = 0
    for r in cache["records"]:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        q = torch.from_numpy(npy(r["query_points"], np.float32))
        q_total += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        da.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256": round(float(np.mean(aj)) * 100.0, 4) if aj else None,
        "OA_256": round(float(np.mean(oa)) * 100.0, 4) if oa else None,
        "delta_avg_256": round(float(np.mean(da)) * 100.0, 4) if da else None,
        "n_records": len(cache["records"]),
        "n_queries": int(q_total),
    }


def oracle_keep_mask(ex: Dict[str, Any], raw_changed: np.ndarray, mode: str) -> np.ndarray:
    """Return local keep mask for raw changed recovery frames."""
    raw_changed = np.asarray(raw_changed, dtype=bool)
    gt_vis = np.asarray(ex["gt_visibility"], dtype=bool)
    err = np.asarray(ex["base_error_256"], dtype=np.float32)
    keep = np.zeros_like(raw_changed, dtype=bool)

    if mode == "v1_keep_all":
        return raw_changed.copy()
    if mode == "drop_all":
        return keep
    if mode == "gt_visible":
        keep = raw_changed & gt_vis
    elif mode.startswith("safe"):
        thr = float(mode.replace("safe", ""))
        keep = raw_changed & gt_vis & (err < thr)
    elif mode == "utility":
        idxs = np.where(raw_changed)[0]
        for i in idxs:
            keep[i] = frame_keep_utility(gt_visible=bool(gt_vis[i]), base_error_256=float(err[i])) > 0.0
    elif mode == "utility_nonnegative":
        idxs = np.where(raw_changed)[0]
        for i in idxs:
            keep[i] = frame_keep_utility(gt_visible=bool(gt_vis[i]), base_error_256=float(err[i])) >= 0.0
    else:
        raise ValueError(f"Unknown oracle mode: {mode}")
    return keep


def apply_oracle(
    base: Dict[str, Any],
    override: Dict[str, Any],
    v1_ckpt: Dict[str, Any],
    *,
    mode: str,
    v1_threshold: float,
    candidate_cfg: CandidateConfig,
    require_override_visible: bool,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = build_model_from_checkpoint_payload(v1_ckpt).to(device).eval()
    mean = np.asarray(v1_ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(v1_ckpt["feature_std"], dtype=np.float32)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "oracle_mode": mode,
        "v1_threshold": float(v1_threshold),
        "candidate_windows": 0,
        "candidate_frames": 0,
        "raw_recovery_frames": 0,
        "raw_changed_recovery_frames": 0,
        "kept_changed_recovery_frames": 0,
        "dropped_changed_recovery_frames": 0,
        "tracks_with_candidate": 0,
        "tracks_with_kept_recovery": 0,
        "require_override_visible": bool(require_override_visible),
        "uses_gt_at_inference": True,
        "per_video": [],
    }

    for br, orr in zip(base["records"], override["records"]):
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        pred_vis = base_vis.copy()
        n_tracks, t_len = pred_vis.shape
        video_stats = {
            "video_id": str(br["video_id"]),
            "candidate_windows": 0,
            "raw_recovery_frames": 0,
            "raw_changed_recovery_frames": 0,
            "kept_changed_recovery_frames": 0,
            "dropped_changed_recovery_frames": 0,
            "tracks_with_candidate": 0,
            "tracks_with_kept_recovery": 0,
        }
        for qi in range(n_tracks):
            query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, candidate_cfg)
            if triggers:
                stats["tracks_with_candidate"] += 1
                video_stats["tracks_with_candidate"] += 1
            kept_this_track = False
            for trigger_t in triggers:
                ex = build_candidate_example(br, orr, qi, int(trigger_t), candidate_cfg)
                x = (ex["x"].astype(np.float32) - mean[None, :]) / std[None, :]
                x[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(x[None]).to(device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(model(xb, valid_mask=valid))[0].detach().cpu().numpy()

                candidate_mask = ex["loss_mask"].astype(bool)
                if require_override_visible:
                    candidate_mask &= ex["override_visibility"] > 0.5
                raw = candidate_mask & (prob >= float(v1_threshold))
                base_local_vis = ex["base_visibility"] > 0.5
                raw_changed = raw & (~base_local_vis)
                keep = oracle_keep_mask(ex, raw_changed, mode=mode)

                frame_indices = ex["frame_indices"]
                valid_keep = keep & (frame_indices >= 0) & (frame_indices < t_len)
                frames = frame_indices[valid_keep]
                if frames.size:
                    pred_vis[qi, frames] = True
                    kept_this_track = True

                raw_count = int(raw.sum())
                raw_changed_count = int(raw_changed.sum())
                kept_count = int(valid_keep.sum())
                dropped_count = max(0, raw_changed_count - kept_count)
                stats["candidate_windows"] += 1
                stats["candidate_frames"] += int(candidate_mask.sum())
                stats["raw_recovery_frames"] += raw_count
                stats["raw_changed_recovery_frames"] += raw_changed_count
                stats["kept_changed_recovery_frames"] += kept_count
                stats["dropped_changed_recovery_frames"] += dropped_count
                video_stats["candidate_windows"] += 1
                video_stats["raw_recovery_frames"] += raw_count
                video_stats["raw_changed_recovery_frames"] += raw_changed_count
                video_stats["kept_changed_recovery_frames"] += kept_count
                video_stats["dropped_changed_recovery_frames"] += dropped_count
            if kept_this_track:
                stats["tracks_with_kept_recovery"] += 1
                video_stats["tracks_with_kept_recovery"] += 1

        nr = dict(br)
        nr["pred_tracks"] = npy(br["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_v23_oracle_frame_keep"] = {
            "method": "ReEntry-VisCalibrator-V2.3-OracleFrameKeep",
            "oracle_mode": mode,
            "coordinates": "base/offline coordinates",
            "visibility": "base visibility plus oracle-kept V1 proposed recovery frames",
            "uses_gt_at_inference": True,
            "candidate_config": candidate_cfg.__dict__,
        }
        records.append(nr)
        stats["per_video"].append(video_stats)

    stats["kept_over_raw_changed_rate"] = round(
        float(stats["kept_changed_recovery_frames"]) / max(float(stats["raw_changed_recovery_frames"]), 1.0), 6
    )
    stats["dropped_over_raw_changed_rate"] = round(
        float(stats["dropped_changed_recovery_frames"]) / max(float(stats["raw_changed_recovery_frames"]), 1.0), 6
    )
    stats["kept_over_raw_recovery_rate"] = round(
        float(stats["kept_changed_recovery_frames"]) / max(float(stats["raw_recovery_frames"]), 1.0), 6
    )
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate V2.3 oracle frame-keep upper bound")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="reentry_v23_oracle_frame_keep")
    ap.add_argument("--oracle-mode", required=True, choices=[
        "v1_keep_all",
        "drop_all",
        "gt_visible",
        "safe16",
        "safe8",
        "safe4",
        "safe2",
        "safe1",
        "utility",
        "utility_nonnegative",
    ])
    ap.add_argument("--v1-threshold", type=float, default=0.10)
    ap.add_argument("--no-require-override-visible", action="store_true")
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")
    v1_ckpt = torch.load(args.v1_model, map_location="cpu", weights_only=False)
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    records, stats = apply_oracle(
        base,
        override,
        v1_ckpt,
        mode=args.oracle_mode,
        v1_threshold=float(args.v1_threshold),
        candidate_cfg=cfg,
        require_override_visible=not args.no_require_override_visible,
        device=torch.device(args.device),
    )
    payload = dict(base)
    payload["records"] = records
    payload["model_name"] = args.name
    payload["reentry_v23_oracle_frame_keep"] = {
        "method": "ReEntry-VisCalibrator-V2.3-OracleFrameKeep",
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
        "uses_gt_at_inference": True,
        **{k: v for k, v in stats.items() if k != "per_video"},
    }
    cache_path = out_dir / f"{args.name}.pt"
    ajrd_json = out_dir / f"{args.name}_ajrd.json"
    standard_json = out_dir / f"{args.name}_standard.json"
    manifest_json = out_dir / "manifest.json"
    torch.save(payload, cache_path)
    ajrd = run_ajrd(cache_path, ajrd_json)
    std = eval_standard(payload)
    standard_json.write_text(json.dumps({**std, **stats}, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "method": "ReEntry-VisCalibrator-V2.3-OracleFrameKeep",
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
        "uses_gt_at_inference": True,
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        **std,
        **stats,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
