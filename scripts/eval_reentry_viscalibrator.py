#!/usr/bin/env python3
"""Apply a learned ReEntry-VisCalibrator to unified caches and evaluate.

The learned module only recovers visibility. Coordinates remain from the base
cache.  By default, recovery is additionally gated by override_visible=True for
safety.
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
from utils.reentry_viscalibrator_features import (  # noqa: E402
    CandidateConfig,
    build_candidate_example,
    check_alignment,
    find_candidate_triggers,
    npy,
)

DEFAULT_BASE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt"
DEFAULT_MODEL = "outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_viscalibrator/eval/rgb_fresh20_49_natural"


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


def apply_calibrator(
    base: Dict[str, Any],
    override: Dict[str, Any],
    ckpt: Dict[str, Any],
    *,
    threshold: float | None,
    candidate_cfg: CandidateConfig,
    require_override_visible: bool,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = build_model_from_checkpoint_payload(ckpt).to(device).eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    thr = float(ckpt.get("best_threshold", 0.5) if threshold is None else threshold)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "threshold": thr,
        "candidate_windows": 0,
        "candidate_frames": 0,
        "recovered_frames": 0,
        "tracks_with_candidate": 0,
        "tracks_with_recovery": 0,
        "require_override_visible": bool(require_override_visible),
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
            "tracks": int(n_tracks),
            "candidate_windows": 0,
            "candidate_frames": 0,
            "recovered_frames": 0,
            "tracks_with_candidate": 0,
            "tracks_with_recovery": 0,
        }

        for qi in range(n_tracks):
            query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, candidate_cfg)
            if triggers:
                stats["tracks_with_candidate"] += 1
                video_stats["tracks_with_candidate"] += 1
            recovered_this_track = False
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
                    candidate_mask = candidate_mask & (ex["override_visibility"] > 0.5)
                recover_local = candidate_mask & (prob >= thr)
                frame_indices = ex["frame_indices"]
                valid_frames = recover_local & (frame_indices >= 0) & (frame_indices < t_len)
                frames = frame_indices[valid_frames]
                if frames.size:
                    pred_vis[qi, frames] = True
                    recovered_this_track = True
                    stats["recovered_frames"] += int(frames.size)
                    video_stats["recovered_frames"] += int(frames.size)
                stats["candidate_windows"] += 1
                stats["candidate_frames"] += int(candidate_mask.sum())
                video_stats["candidate_windows"] += 1
                video_stats["candidate_frames"] += int(candidate_mask.sum())
            if recovered_this_track:
                stats["tracks_with_recovery"] += 1
                video_stats["tracks_with_recovery"] += 1

        nr = dict(br)
        nr["pred_tracks"] = npy(br["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_viscalibrator"] = {
            "method": "ReEntry-VisCalibrator",
            "coordinates": "base/offline coordinates",
            "visibility": "base visibility plus learned candidate-window recovery",
            "threshold": thr,
            "require_override_visible": bool(require_override_visible),
            "uses_gt_at_inference": False,
            "candidate_config": candidate_cfg.__dict__,
        }
        records.append(nr)
        stats["per_video"].append(video_stats)

    stats["recovered_frame_rate_over_candidate_frames"] = round(
        float(stats["recovered_frames"]) / max(int(stats["candidate_frames"]), 1), 6
    )
    return records, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate learned ReEntry-VisCalibrator.")
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--override-cache", default=DEFAULT_OVERRIDE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="reentry_viscalibrator_v1")
    ap.add_argument("--threshold", type=float, default=-1.0, help="Override model threshold; <0 uses checkpoint best threshold.")
    ap.add_argument("--no-require-override-visible", action="store_true")
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    threshold = None if args.threshold < 0 else float(args.threshold)
    records, stats = apply_calibrator(
        base,
        override,
        ckpt,
        threshold=threshold,
        candidate_cfg=cfg,
        require_override_visible=not args.no_require_override_visible,
        device=device,
    )
    payload = dict(base)
    payload["records"] = records
    payload["model_name"] = args.name
    payload["reentry_viscalibrator"] = {
        "model": str(args.model),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "coordinates": "base/offline coordinates",
        "visibility": "learned recovery inside predicted candidate windows",
        "uses_gt_at_inference": False,
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
        "method": "ReEntry-VisCalibrator",
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "model": str(args.model),
        "ajrd_json": str(ajrd_json),
        "standard_json": str(standard_json),
        "uses_gt_at_inference": False,
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        **std,
        **stats,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
