#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

DEFAULT_BASE = "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline_repro"


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def check_alignment(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    br = base["records"]
    orr = override["records"]
    if len(br) != len(orr):
        raise ValueError(f"record count mismatch: {len(br)} vs {len(orr)}")
    for i, (b, o) in enumerate(zip(br, orr)):
        if str(b["video_id"]) != str(o["video_id"]):
            raise ValueError(f"video_id mismatch at {i}: {b['video_id']} vs {o['video_id']}")
        for key in ["query_points", "gt_tracks", "gt_visibility", "original_size"]:
            if not np.allclose(npy(b[key]), npy(o[key]), atol=1e-6, rtol=1e-6):
                raise ValueError(f"alignment mismatch at {i}, key={key}")


def invisible_run_before(v: np.ndarray, t: int) -> int:
    count = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        count += 1
        j -= 1
    return count


def first_trigger_mask(base_v: np.ndarray, override_v: np.ndarray, query_t: int, k: int, pre: int, post: int) -> Tuple[np.ndarray, List[int]]:
    t_len = int(base_v.shape[0])
    mask = np.zeros(t_len, dtype=bool)
    triggers: List[int] = []
    t = max(1, int(query_t) + 1)
    while t < t_len:
        if invisible_run_before(base_v, t) >= int(k) and bool(override_v[t]):
            lo = max(0, t - int(pre))
            hi = t_len if int(post) >= 9999 else min(t_len, t + int(post) + 1)
            mask[lo:hi] = True
            triggers.append(int(t))
            t = hi
        else:
            t += 1
    return mask, triggers


def build_b2_records(base: Dict[str, Any], override: Dict[str, Any], k: int, pre: int, post: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    total_triggers = 0
    tracks_with_trigger = 0
    triggered_reentry_tracks = 0
    triggered_nonreentry_tracks = 0
    missed_reentry_tracks = 0
    per_video_trigger_stats = []

    for b, o in zip(base["records"], override["records"]):
        pred_tracks = npy(b["pred_tracks"], np.float32).copy()
        pred_vis = npy(b["pred_visibility"], bool).copy()
        base_vis = pred_vis.copy()
        over_tracks = npy(o["pred_tracks"], np.float32)
        over_vis = npy(o["pred_visibility"], bool)
        gt_vis = npy(b["gt_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        n, t_len = pred_vis.shape
        video_stats = {
            "video_id": str(b["video_id"]),
            "tracks": int(n),
            "triggers": 0,
            "tracks_with_trigger": 0,
            "triggered_reentry_tracks": 0,
            "triggered_nonreentry_tracks": 0,
            "missed_reentry_tracks": 0,
        }
        for qi in range(n):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            mask, triggers = first_trigger_mask(base_vis[qi], over_vis[qi], qt, k=k, pre=pre, post=post)
            has_reentry = bool(find_reentry_events(gt_vis[qi], qt))
            if triggers:
                total_triggers += len(triggers)
                tracks_with_trigger += 1
                video_stats["triggers"] += len(triggers)
                video_stats["tracks_with_trigger"] += 1
                if has_reentry:
                    triggered_reentry_tracks += 1
                    video_stats["triggered_reentry_tracks"] += 1
                else:
                    triggered_nonreentry_tracks += 1
                    video_stats["triggered_nonreentry_tracks"] += 1
                pred_tracks[qi, mask] = over_tracks[qi, mask]
                pred_vis[qi, mask] = over_vis[qi, mask]
            elif has_reentry:
                missed_reentry_tracks += 1
                video_stats["missed_reentry_tracks"] += 1
        r = dict(b)
        r["pred_tracks"] = pred_tracks.astype(np.float32)
        r["pred_visibility"] = pred_vis.astype(bool)
        r["b2_pred_k"] = int(k)
        r["b2_pred_pre"] = int(pre)
        r["b2_pred_post"] = int(post)
        r["b2_pred_mode"] = "base_inv_over_vis"
        records.append(r)
        per_video_trigger_stats.append(video_stats)

    stats = {
        "total_triggers": int(total_triggers),
        "tracks_with_trigger": int(tracks_with_trigger),
        "triggered_reentry_tracks": int(triggered_reentry_tracks),
        "triggered_nonreentry_tracks": int(triggered_nonreentry_tracks),
        "missed_reentry_tracks": int(missed_reentry_tracks),
        "trigger_precision_track": round(triggered_reentry_tracks / max(tracks_with_trigger, 1), 6),
        "trigger_recall_track": round(triggered_reentry_tracks / max(triggered_reentry_tracks + missed_reentry_tracks, 1), 6),
        "per_video_trigger_stats": per_video_trigger_stats,
    }
    return records, stats


def mean(vals: List[float]) -> float | None:
    return round(float(np.mean(vals)), 6) if vals else None


def pct(x: float | None) -> float | None:
    return None if x is None else round(float(x) * 100.0, 4)


def eval_standard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aj, oa, delta = [], [], []
    for r in records:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        qp = torch.from_numpy(npy(r["query_points"], np.float32))
        m = compute_tapvid_metrics(pred, gt, pv, gv, qp, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        delta.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256_pct": pct(mean(aj)),
        "OA_256_pct": pct(mean(oa)),
        "delta_avg_256_pct": pct(mean(delta)),
    }


def visibility_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    precisions, recalls, fprs, misses, re_pred_rate, re_miss = [], [], [], [], [], []
    for r in records:
        pv = npy(r["pred_visibility"], bool)
        gv = npy(r["gt_visibility"], bool)
        qpts = npy(r["query_points"], np.float32)
        n, t_len = gv.shape
        mask = np.ones((n, t_len), dtype=bool)
        for qi in range(n):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            mask[qi, qt] = False
        pred = pv[mask]
        gt = gv[mask]
        tp = int(np.sum(pred & gt))
        fp = int(np.sum(pred & ~gt))
        fn = int(np.sum((~pred) & gt))
        total = max(int(pred.size), 1)
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
        fprs.append(fp / total)
        misses.append(fn / total)
        rp, rg = [], []
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            evs = find_reentry_events(gv[qi], qt)
            if not evs:
                continue
            rt = int(evs[0]["reentry_frame"])
            rp.append(bool(pv[qi, rt]))
            rg.append(bool(gv[qi, rt]))
        if rg:
            rp_arr = np.asarray(rp, dtype=bool)
            rg_arr = np.asarray(rg, dtype=bool)
            re_pred_rate.append(float(np.mean(rp_arr)))
            re_miss.append(float(np.sum((~rp_arr) & rg_arr)) / max(int(np.sum(rg_arr)), 1))
    return {
        "visible_precision": mean(precisions),
        "visible_recall": mean(recalls),
        "false_visible_rate_all": mean(fprs),
        "missed_visible_rate_all": mean(misses),
        "reentry_pred_visible_rate": mean(re_pred_rate),
        "reentry_missed_visible_rate": mean(re_miss),
    }


def run_ajrd(cache_path: Path, output_json: Path) -> Dict[str, Any]:
    subprocess.run([
        sys.executable,
        "scripts/eval_aj_rd_from_cache.py",
        "--cache-path",
        str(cache_path),
        "--output-json",
        str(output_json),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(output_json))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the clean B2 predicted post-reentry localized override mainline.")
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--override-cache", default=DEFAULT_OVERRIDE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--pre", type=int, default=1)
    ap.add_argument("--post", type=int, default=9999, help="9999 means full post segment")
    ap.add_argument("--name", default="b2_predicted_mainline")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_payload(args.base_cache)
    override = load_payload(args.override_cache)
    check_alignment(base, override)
    records, trigger_stats = build_b2_records(base, override, k=args.k, pre=args.pre, post=args.post)

    payload = dict(base)
    payload["model_name"] = args.name
    payload["b2_method"] = "predicted_post_reentry_localized_override"
    payload["b2_base_cache"] = str(args.base_cache)
    payload["b2_override_cache"] = str(args.override_cache)
    payload["b2_trigger_rule"] = "base invisible run >= k and override visible"
    payload["b2_pred_k"] = int(args.k)
    payload["b2_pred_pre"] = int(args.pre)
    payload["b2_pred_post"] = int(args.post)
    payload["b2_pred_mode"] = "base_inv_over_vis"
    payload["records"] = records

    cache_path = out_dir / f"{args.name}.pt"
    ajrd_json = out_dir / f"{args.name}_ajrd.json"
    standard_json = out_dir / f"{args.name}_standard_visibility.json"
    manifest_path = out_dir / "manifest.json"
    torch.save(payload, cache_path)

    ajrd = run_ajrd(cache_path, ajrd_json)
    standard = eval_standard(records)
    visibility = visibility_stats(records)
    standard_visibility = {**standard, **visibility, **trigger_stats}
    standard_json.write_text(json.dumps(standard_visibility, indent=2, ensure_ascii=False))

    bd = ajrd.get("aj_rd_by_dmin_256") or {}
    manifest = {
        "method": "B2 predicted post-reentry localized override",
        "name": args.name,
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "trigger_rule": "base invisible run >= k and override visible",
        "k": int(args.k),
        "pre": int(args.pre),
        "post": "full_post_segment" if int(args.post) >= 9999 else int(args.post),
        "cache": str(cache_path),
        "ajrd_json": str(ajrd_json),
        "standard_visibility_json": str(standard_json),
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        "first_reentry_frame_proxy": ajrd.get("first_reentry_frame_proxy"),
        "dmin1_256": bd.get("1"),
        "dmin4_256": bd.get("4"),
        "dmin16_256": bd.get("16"),
        **standard_visibility,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
