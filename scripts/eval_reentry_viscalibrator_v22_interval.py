#!/usr/bin/env python3
"""Evaluate ReEntry-VisCalibrator V2.2 interval post-processing.

V2.2 does not train a new model.  It reuses V1 frame probabilities and optionally
V2.1 event-gate probabilities, but it avoids the coarse whole-window block of
V2.1.  Instead, it cleans the V1 recovery proposal inside each candidate window:

  1. build V1 recovery mask: candidate & override-visible & prob >= v1_threshold
  2. split into consecutive temporal segments
  3. remove short segments
  4. trim low-confidence segment edges
  5. optionally adapt min length / edge threshold using event-gate probability

Coordinates remain from the base/offline cache.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics  # noqa: E402
from models.reentry_viscalibrator import build_model_from_checkpoint_payload  # noqa: E402
from utils.reentry_event_gate_features import build_event_features  # noqa: E402
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


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive local-index segments for a 1D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    segs: List[Tuple[int, int]] = []
    i = 0
    n = int(mask.shape[0])
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def choose_params(
    *,
    gate_prob: float | None,
    adaptive_gate: bool,
    min_segment_len: int,
    edge_threshold: float,
    core_threshold: float,
    strict_gate_threshold: float,
    medium_gate_threshold: float,
    medium_min_segment_len: int,
    strict_min_segment_len: int,
    medium_edge_threshold: float,
    strict_edge_threshold: float,
    medium_core_threshold: float,
    strict_core_threshold: float,
) -> Tuple[int, float, float, str]:
    if (not adaptive_gate) or gate_prob is None:
        return int(min_segment_len), float(edge_threshold), float(core_threshold), "fixed"
    if float(gate_prob) < float(strict_gate_threshold):
        return int(strict_min_segment_len), float(strict_edge_threshold), float(strict_core_threshold), "strict"
    if float(gate_prob) < float(medium_gate_threshold):
        return int(medium_min_segment_len), float(medium_edge_threshold), float(medium_core_threshold), "medium"
    return int(min_segment_len), float(edge_threshold), float(core_threshold), "light"


def clean_recovery_mask(
    raw: np.ndarray,
    prob: np.ndarray,
    *,
    min_segment_len: int,
    edge_threshold: float,
    core_threshold: float,
    keep_peak_if_trim_empty: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    raw = np.asarray(raw, dtype=bool)
    prob = np.asarray(prob, dtype=np.float32)
    clean = np.zeros_like(raw, dtype=bool)
    stats = {
        "raw_segments": 0,
        "kept_segments": 0,
        "dropped_short_segments": 0,
        "dropped_low_core_segments": 0,
        "trimmed_edge_frames": 0,
        "peak_rescue_segments": 0,
    }
    for lo, hi in contiguous_segments(raw):
        stats["raw_segments"] += 1
        seg_len = hi - lo + 1
        seg_prob = prob[lo : hi + 1]
        if float(seg_prob.max()) < float(core_threshold):
            stats["dropped_low_core_segments"] += 1
            continue
        a, b = lo, hi
        while a <= b and float(prob[a]) < float(edge_threshold):
            a += 1
            stats["trimmed_edge_frames"] += 1
        while b >= a and float(prob[b]) < float(edge_threshold):
            b -= 1
            stats["trimmed_edge_frames"] += 1
        if a > b:
            if keep_peak_if_trim_empty and seg_len >= int(min_segment_len):
                peak = int(lo + int(np.argmax(seg_prob)))
                clean[peak] = True
                stats["kept_segments"] += 1
                stats["peak_rescue_segments"] += 1
            else:
                stats["dropped_low_core_segments"] += 1
            continue
        if (b - a + 1) < int(min_segment_len):
            stats["dropped_short_segments"] += 1
            continue
        clean[a : b + 1] = True
        stats["kept_segments"] += 1
    return clean, stats


def apply_interval_calibrator(
    base: Dict[str, Any],
    override: Dict[str, Any],
    v1_ckpt: Dict[str, Any],
    gate_ckpt: Dict[str, Any] | None,
    *,
    v1_threshold: float,
    candidate_cfg: CandidateConfig,
    require_override_visible: bool,
    device: torch.device,
    min_segment_len: int,
    edge_threshold: float,
    core_threshold: float,
    keep_peak_if_trim_empty: bool,
    adaptive_gate: bool,
    gate_block_threshold: float,
    strict_gate_threshold: float,
    medium_gate_threshold: float,
    medium_min_segment_len: int,
    strict_min_segment_len: int,
    medium_edge_threshold: float,
    strict_edge_threshold: float,
    medium_core_threshold: float,
    strict_core_threshold: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v1_model = build_model_from_checkpoint_payload(v1_ckpt).to(device).eval()
    v1_mean = np.asarray(v1_ckpt["feature_mean"], dtype=np.float32)
    v1_std = np.asarray(v1_ckpt["feature_std"], dtype=np.float32)

    gate_model = None
    gate_mean = gate_std = None
    if gate_ckpt is not None:
        gate_model = build_gate_from_ckpt(gate_ckpt, device)
        gate_mean = np.asarray(gate_ckpt["feature_mean"], dtype=np.float32)
        gate_std = np.asarray(gate_ckpt["feature_std"], dtype=np.float32)

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "v1_threshold": float(v1_threshold),
        "min_segment_len": int(min_segment_len),
        "edge_threshold": float(edge_threshold),
        "core_threshold": float(core_threshold),
        "keep_peak_if_trim_empty": bool(keep_peak_if_trim_empty),
        "adaptive_gate": bool(adaptive_gate),
        "gate_block_threshold": float(gate_block_threshold),
        "gate_blocked_windows": 0,
        "candidate_windows": 0,
        "candidate_frames": 0,
        "raw_recovery_frames": 0,
        "final_recovery_frames": 0,
        "removed_recovery_frames": 0,
        "tracks_with_candidate": 0,
        "tracks_with_recovery": 0,
        "mode_counts": {"fixed": 0, "light": 0, "medium": 0, "strict": 0, "blocked": 0},
        "segment_stats": {
            "raw_segments": 0,
            "kept_segments": 0,
            "dropped_short_segments": 0,
            "dropped_low_core_segments": 0,
            "trimmed_edge_frames": 0,
            "peak_rescue_segments": 0,
        },
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
            "candidate_windows": 0,
            "gate_blocked_windows": 0,
            "raw_recovery_frames": 0,
            "final_recovery_frames": 0,
            "removed_recovery_frames": 0,
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
                xf = (ex["x"].astype(np.float32) - v1_mean[None, :]) / v1_std[None, :]
                xf[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(xf[None]).to(device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(v1_model(xb, valid_mask=valid))[0].detach().cpu().numpy()

                candidate_mask = ex["loss_mask"].astype(bool)
                if require_override_visible:
                    candidate_mask &= ex["override_visibility"] > 0.5
                raw = candidate_mask & (prob >= float(v1_threshold))

                gate_prob = None
                if gate_model is not None:
                    evx = build_event_features(ex, prob, threshold=v1_threshold, require_override_visible=require_override_visible)
                    evxn = (evx.astype(np.float32) - gate_mean) / gate_std
                    with torch.no_grad():
                        gate_prob = float(torch.sigmoid(gate_model(torch.from_numpy(evxn[None]).to(device).float()))[0].detach().cpu())

                # Optional V2.1-style coarse blocking for the lowest-confidence events.
                if gate_prob is not None and float(gate_block_threshold) >= 0.0 and float(gate_prob) < float(gate_block_threshold):
                    clean = np.zeros_like(raw, dtype=bool)
                    seg_stats = {
                        "raw_segments": len(contiguous_segments(raw)),
                        "kept_segments": 0,
                        "dropped_short_segments": 0,
                        "dropped_low_core_segments": 0,
                        "trimmed_edge_frames": 0,
                        "peak_rescue_segments": 0,
                    }
                    mode = "blocked"
                    stats["gate_blocked_windows"] += 1
                    video_stats["gate_blocked_windows"] += 1
                else:
                    local_min_len, local_edge, local_core, mode = choose_params(
                        gate_prob=gate_prob,
                        adaptive_gate=adaptive_gate,
                        min_segment_len=min_segment_len,
                        edge_threshold=edge_threshold,
                        core_threshold=core_threshold,
                        strict_gate_threshold=strict_gate_threshold,
                        medium_gate_threshold=medium_gate_threshold,
                        medium_min_segment_len=medium_min_segment_len,
                        strict_min_segment_len=strict_min_segment_len,
                        medium_edge_threshold=medium_edge_threshold,
                        strict_edge_threshold=strict_edge_threshold,
                        medium_core_threshold=medium_core_threshold,
                        strict_core_threshold=strict_core_threshold,
                    )
                    clean, seg_stats = clean_recovery_mask(
                        raw,
                        prob,
                        min_segment_len=local_min_len,
                        edge_threshold=local_edge,
                        core_threshold=local_core,
                        keep_peak_if_trim_empty=keep_peak_if_trim_empty,
                    )

                frame_indices = ex["frame_indices"]
                valid_frames = clean & (frame_indices >= 0) & (frame_indices < t_len)
                frames = frame_indices[valid_frames]
                if frames.size:
                    pred_vis[qi, frames] = True
                    recovered_this_track = True
                raw_count = int(raw.sum())
                final_count = int(valid_frames.sum())
                stats["candidate_windows"] += 1
                stats["candidate_frames"] += int(candidate_mask.sum())
                stats["raw_recovery_frames"] += raw_count
                stats["final_recovery_frames"] += final_count
                stats["removed_recovery_frames"] += max(0, raw_count - final_count)
                stats["mode_counts"][mode] = int(stats["mode_counts"].get(mode, 0)) + 1
                for k, v in seg_stats.items():
                    stats["segment_stats"][k] = int(stats["segment_stats"].get(k, 0)) + int(v)
                video_stats["candidate_windows"] += 1
                video_stats["raw_recovery_frames"] += raw_count
                video_stats["final_recovery_frames"] += final_count
                video_stats["removed_recovery_frames"] += max(0, raw_count - final_count)
            if recovered_this_track:
                stats["tracks_with_recovery"] += 1
                video_stats["tracks_with_recovery"] += 1

        nr = dict(br)
        nr["pred_tracks"] = npy(br["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_interval_v22"] = {
            "method": "ReEntry-VisCalibrator-V2.2-IntervalPostProcessing",
            "coordinates": "base/offline coordinates",
            "visibility": "base visibility plus interval-cleaned learned recovery",
            "uses_gt_at_inference": False,
            "candidate_config": candidate_cfg.__dict__,
        }
        records.append(nr)
        stats["per_video"].append(video_stats)

    stats["final_over_raw_recovery_rate"] = round(float(stats["final_recovery_frames"]) / max(float(stats["raw_recovery_frames"]), 1.0), 6)
    stats["removed_over_raw_recovery_rate"] = round(float(stats["removed_recovery_frames"]) / max(float(stats["raw_recovery_frames"]), 1.0), 6)
    stats["final_recovered_frame_rate_over_candidate_frames"] = round(float(stats["final_recovery_frames"]) / max(float(stats["candidate_frames"]), 1.0), 6)
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate V2.2 interval post-processing")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--gate-model", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="reentry_viscalibrator_v22_interval")
    ap.add_argument("--v1-threshold", type=float, default=0.10)
    ap.add_argument("--min-segment-len", type=int, default=2)
    ap.add_argument("--edge-threshold", type=float, default=0.12)
    ap.add_argument("--core-threshold", type=float, default=0.10)
    ap.add_argument("--keep-peak-if-trim-empty", action="store_true")
    ap.add_argument("--adaptive-gate", action="store_true")
    ap.add_argument("--gate-block-threshold", type=float, default=-1.0, help="If gate model is provided, block whole event when gate prob is below this threshold. <0 disables.")
    ap.add_argument("--strict-gate-threshold", type=float, default=0.005)
    ap.add_argument("--medium-gate-threshold", type=float, default=0.02)
    ap.add_argument("--medium-min-segment-len", type=int, default=2)
    ap.add_argument("--strict-min-segment-len", type=int, default=3)
    ap.add_argument("--medium-edge-threshold", type=float, default=0.12)
    ap.add_argument("--strict-edge-threshold", type=float, default=0.15)
    ap.add_argument("--medium-core-threshold", type=float, default=0.10)
    ap.add_argument("--strict-core-threshold", type=float, default=0.10)
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
    gate_ckpt = torch.load(args.gate_model, map_location="cpu", weights_only=False) if args.gate_model else None
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    records, stats = apply_interval_calibrator(
        base,
        override,
        v1_ckpt,
        gate_ckpt,
        v1_threshold=float(args.v1_threshold),
        candidate_cfg=cfg,
        require_override_visible=not args.no_require_override_visible,
        device=torch.device(args.device),
        min_segment_len=int(args.min_segment_len),
        edge_threshold=float(args.edge_threshold),
        core_threshold=float(args.core_threshold),
        keep_peak_if_trim_empty=bool(args.keep_peak_if_trim_empty),
        adaptive_gate=bool(args.adaptive_gate),
        gate_block_threshold=float(args.gate_block_threshold),
        strict_gate_threshold=float(args.strict_gate_threshold),
        medium_gate_threshold=float(args.medium_gate_threshold),
        medium_min_segment_len=int(args.medium_min_segment_len),
        strict_min_segment_len=int(args.strict_min_segment_len),
        medium_edge_threshold=float(args.medium_edge_threshold),
        strict_edge_threshold=float(args.strict_edge_threshold),
        medium_core_threshold=float(args.medium_core_threshold),
        strict_core_threshold=float(args.strict_core_threshold),
    )
    payload = dict(base)
    payload["records"] = records
    payload["model_name"] = args.name
    payload["reentry_interval_v22"] = {
        "method": "ReEntry-VisCalibrator-V2.2-IntervalPostProcessing",
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
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
        "method": "ReEntry-VisCalibrator-V2.2-IntervalPostProcessing",
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "v1_model": str(args.v1_model),
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
