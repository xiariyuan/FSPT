#!/usr/bin/env python3
"""Evaluate V2.3 learned frame-keep decoder.

The decoder only filters V1-proposed recovery frames.  It does not propose new
visible frames and keeps base/offline coordinates unchanged.
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
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics  # noqa: E402
from models.reentry_viscalibrator import build_model_from_checkpoint_payload  # noqa: E402
from utils.reentry_event_gate_features import build_event_features  # noqa: E402
from utils.reentry_frame_keep_features import build_frame_keep_samples  # noqa: E402
from utils.reentry_viscalibrator_features import (  # noqa: E402
    CandidateConfig,
    build_candidate_example,
    check_alignment,
    find_candidate_triggers,
    npy,
)


class EventGateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FrameKeepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def run_ajrd(cache_path: Path, output_json: Path) -> Dict[str, Any]:
    subprocess.run(
        [sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(cache_path), "--output-json", str(output_json)],
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


def build_gate_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> EventGateMLP:
    cfg = ckpt.get("model_config", {})
    model = EventGateMLP(int(cfg.get("in_dim", len(ckpt["feature_mean"]))), int(cfg.get("hidden", 128)), float(cfg.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def build_keep_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> FrameKeepMLP:
    cfg = ckpt.get("model_config", {})
    model = FrameKeepMLP(int(cfg.get("in_dim", len(ckpt["feature_mean"]))), int(cfg.get("hidden", 128)), float(cfg.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def apply_frame_keep(
    base: Dict[str, Any],
    override: Dict[str, Any],
    v1_ckpt: Dict[str, Any],
    keep_ckpt: Dict[str, Any],
    gate_ckpt: Dict[str, Any] | None,
    *,
    v1_threshold: float,
    keep_threshold: float,
    candidate_cfg: CandidateConfig,
    require_override_visible: bool,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v1_model = build_model_from_checkpoint_payload(v1_ckpt).to(device).eval()
    v1_mean = np.asarray(v1_ckpt["feature_mean"], dtype=np.float32)
    v1_std = np.asarray(v1_ckpt["feature_std"], dtype=np.float32)

    keep_model = build_keep_from_ckpt(keep_ckpt, device)
    keep_mean = np.asarray(keep_ckpt["feature_mean"], dtype=np.float32)
    keep_std = np.asarray(keep_ckpt["feature_std"], dtype=np.float32)

    gate_model = None
    gate_mean = gate_std = None
    if gate_ckpt is not None:
        gate_model = build_gate_from_ckpt(gate_ckpt, device)
        gate_mean = np.asarray(gate_ckpt["feature_mean"], dtype=np.float32)
        gate_std = np.asarray(gate_ckpt["feature_std"], dtype=np.float32)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "v1_threshold": float(v1_threshold),
        "keep_threshold": float(keep_threshold),
        "keep_model_label": str(keep_ckpt.get("label", "unknown")),
        "candidate_windows": 0,
        "candidate_frames": 0,
        "raw_recovery_frames": 0,
        "raw_changed_recovery_frames": 0,
        "kept_changed_recovery_frames": 0,
        "dropped_changed_recovery_frames": 0,
        "tracks_with_candidate": 0,
        "tracks_with_kept_recovery": 0,
        "require_override_visible": bool(require_override_visible),
        "uses_gt_at_inference": False,
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
                xf = (ex["x"].astype(np.float32) - v1_mean[None, :]) / v1_std[None, :]
                xf[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(xf[None]).to(device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(v1_model(xb, valid_mask=valid))[0].detach().cpu().numpy()

                gate_prob = None
                if gate_model is not None:
                    evx = build_event_features(ex, prob, threshold=v1_threshold, require_override_visible=require_override_visible)
                    evxn = (evx.astype(np.float32) - gate_mean) / gate_std
                    with torch.no_grad():
                        gate_prob = float(torch.sigmoid(gate_model(torch.from_numpy(evxn[None]).to(device).float()))[0].detach().cpu())

                pack = build_frame_keep_samples(
                    ex,
                    prob,
                    threshold=float(v1_threshold),
                    require_override_visible=require_override_visible,
                    gate_prob=gate_prob,
                    include_base_visible=False,
                )
                stats["candidate_windows"] += 1
                stats["candidate_frames"] += int(pack["candidate_frames"])
                stats["raw_recovery_frames"] += int(pack["raw_recovery_frames"])
                stats["raw_changed_recovery_frames"] += int(pack["changed_recovery_frames"])
                video_stats["candidate_windows"] += 1
                video_stats["raw_recovery_frames"] += int(pack["raw_recovery_frames"])
                video_stats["raw_changed_recovery_frames"] += int(pack["changed_recovery_frames"])
                if pack["X"].shape[0] == 0:
                    continue
                X = np.asarray(pack["X"], dtype=np.float32)
                Xn = (X - keep_mean[None, :]) / keep_std[None, :]
                with torch.no_grad():
                    keep_prob = torch.sigmoid(keep_model(torch.from_numpy(Xn).to(device).float())).detach().cpu().numpy()
                keep = keep_prob >= float(keep_threshold)
                metas = pack["meta"]
                kept_count = 0
                for j, m in enumerate(metas):
                    if not bool(keep[j]):
                        continue
                    frame_t = int(m["frame_t"])
                    if 0 <= frame_t < t_len:
                        pred_vis[qi, frame_t] = True
                        kept_count += 1
                dropped_count = max(0, int(pack["changed_recovery_frames"]) - kept_count)
                stats["kept_changed_recovery_frames"] += kept_count
                stats["dropped_changed_recovery_frames"] += dropped_count
                video_stats["kept_changed_recovery_frames"] += kept_count
                video_stats["dropped_changed_recovery_frames"] += dropped_count
                if kept_count > 0:
                    kept_this_track = True
            if kept_this_track:
                stats["tracks_with_kept_recovery"] += 1
                video_stats["tracks_with_kept_recovery"] += 1

        nr = dict(br)
        nr["pred_tracks"] = npy(br["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_v23_frame_keep"] = {
            "method": "ReEntry-VisCalibrator-V2.3-FrameKeep",
            "coordinates": "base/offline coordinates",
            "visibility": "base visibility plus learned-kept V1 proposed recovery frames",
            "uses_gt_at_inference": False,
            "candidate_config": candidate_cfg.__dict__,
        }
        records.append(nr)
        stats["per_video"].append(video_stats)

    stats["kept_over_raw_changed_rate"] = round(float(stats["kept_changed_recovery_frames"]) / max(float(stats["raw_changed_recovery_frames"]), 1.0), 6)
    stats["dropped_over_raw_changed_rate"] = round(float(stats["dropped_changed_recovery_frames"]) / max(float(stats["raw_changed_recovery_frames"]), 1.0), 6)
    stats["kept_over_raw_recovery_rate"] = round(float(stats["kept_changed_recovery_frames"]) / max(float(stats["raw_recovery_frames"]), 1.0), 6)
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate V2.3 learned frame-keep decoder")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--keep-model", required=True)
    ap.add_argument("--gate-model", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="reentry_v23_frame_keep")
    ap.add_argument("--v1-threshold", type=float, default=0.10)
    ap.add_argument("--keep-threshold", type=float, default=-1.0, help="<0 uses keep model sample threshold")
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
    keep_ckpt = torch.load(args.keep_model, map_location="cpu", weights_only=False)
    gate_ckpt = torch.load(args.gate_model, map_location="cpu", weights_only=False) if args.gate_model else None
    keep_threshold = float(keep_ckpt.get("best_threshold_sample", 0.5) if args.keep_threshold < 0 else args.keep_threshold)
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    records, stats = apply_frame_keep(
        base,
        override,
        v1_ckpt,
        keep_ckpt,
        gate_ckpt,
        v1_threshold=float(args.v1_threshold),
        keep_threshold=keep_threshold,
        candidate_cfg=cfg,
        require_override_visible=not args.no_require_override_visible,
        device=torch.device(args.device),
    )
    payload = dict(base)
    payload["records"] = records
    payload["model_name"] = args.name
    payload["reentry_v23_frame_keep"] = {
        "method": "ReEntry-VisCalibrator-V2.3-FrameKeep",
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
        "keep_model": str(args.keep_model),
        "gate_model": str(args.gate_model),
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
        "method": "ReEntry-VisCalibrator-V2.3-FrameKeep",
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
        "keep_model": str(args.keep_model),
        "gate_model": str(args.gate_model),
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
