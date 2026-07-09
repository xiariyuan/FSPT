#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, yx_norm_to_xy_256

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_false_trigger_taxonomy")
BASE_CACHE = Path("outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
OVERRIDE_CACHE = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt")
B2_CACHE = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt")
THRESHOLDS = (1, 2, 4, 8, 16)


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = t - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def first_b2_trigger(base_v: np.ndarray, over_v: np.ndarray, query_t: int, k: int = 1) -> Optional[int]:
    for t in range(max(1, int(query_t) + 1), len(base_v)):
        if invisible_run_before(base_v, t) >= int(k) and bool(over_v[t]):
            return int(t)
    return None


def point_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, start_t: int = 0) -> float:
    t_len = gt_vis.shape[0]
    mask = np.ones(t_len, dtype=bool)
    qt = max(0, min(t_len - 1, int(query_t)))
    mask[qt] = False
    if start_t > 0:
        mask[: int(start_t)] = False
    if not np.any(mask):
        return 0.0
    pred_px = yx_norm_to_xy_256(np.asarray(pred_tracks, dtype=np.float32))
    gt_px = yx_norm_to_xy_256(np.asarray(gt_tracks, dtype=np.float32))
    sq = np.sum((pred_px - gt_px) ** 2, axis=-1)
    vals = []
    for thr in THRESHOLDS:
        within = sq < float(thr) ** 2
        gt_pos = gt_vis.astype(bool)
        pv = pred_vis.astype(bool)
        tp = float(np.sum(mask & within & gt_pos & pv))
        gp = float(np.sum(mask & gt_pos))
        fp = float(np.sum(mask & pv & ((~gt_pos) | (~within))))
        vals.append(tp / (gp + fp) if (gp + fp) > 0 else 0.0)
    return float(np.mean(vals))


def px_dist(a_yx: np.ndarray, b_yx: np.ndarray) -> float:
    a = yx_norm_to_xy_256(np.asarray(a_yx, dtype=np.float32)[None, :])[0]
    b = yx_norm_to_xy_256(np.asarray(b_yx, dtype=np.float32)[None, :])[0]
    return float(np.linalg.norm(a - b))


def summarize_vals(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    arr = np.asarray(vals, dtype=np.float32)
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load(BASE_CACHE)
    over = load(OVERRIDE_CACHE)
    b2 = load(B2_CACHE)
    base_records = base["records"]
    over_records = over["records"]
    b2_records = b2["records"]

    rows: List[Dict[str, Any]] = []
    video: Dict[str, Dict[str, Any]] = {}
    counts = {
        "total_tracks": 0,
        "gt_reentry_tracks": 0,
        "triggered_tracks": 0,
        "true_trigger_tracks": 0,
        "false_trigger_tracks": 0,
        "no_trigger_reentry_tracks": 0,
        "nontrigger_nonreentry_tracks": 0,
    }
    false_rows: List[Dict[str, Any]] = []
    true_rows: List[Dict[str, Any]] = []

    for br, orr, b2r in zip(base_records, over_records, b2_records):
        vid = str(br["video_id"])
        v = video.setdefault(vid, {
            "video_id": vid,
            "total_tracks": 0,
            "gt_reentry_tracks": 0,
            "triggered_tracks": 0,
            "true_trigger_tracks": 0,
            "false_trigger_tracks": 0,
            "no_trigger_reentry_tracks": 0,
            "false_gt_visible_at_trigger": 0,
            "false_gt_invisible_at_trigger": 0,
            "false_harmful_full_gt_0p05": 0,
            "false_severe_full_gt_0p10": 0,
            "false_low_cost_full_ge_minus_0p01": 0,
            "false_delta_full": [],
            "false_delta_post": [],
            "true_delta_full": [],
            "true_delta_post": [],
            "false_base_override_dist": [],
        })
        bvis = npy(br["pred_visibility"], bool)
        ovis = npy(orr["pred_visibility"], bool)
        b2vis = npy(b2r["pred_visibility"], bool)
        gv = npy(br["gt_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        btracks = npy(br["pred_tracks"], np.float32)
        otracks = npy(orr["pred_tracks"], np.float32)
        b2tracks = npy(b2r["pred_tracks"], np.float32)
        gttracks = npy(br["gt_tracks"], np.float32)
        n, t_len = gv.shape
        for qi in range(n):
            counts["total_tracks"] += 1
            v["total_tracks"] += 1
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            events = find_reentry_events(gv[qi], qt)
            has_re = bool(events)
            if has_re:
                counts["gt_reentry_tracks"] += 1
                v["gt_reentry_tracks"] += 1
            trig = first_b2_trigger(bvis[qi], ovis[qi], qt, k=args.k)
            triggered = trig is not None
            if triggered:
                counts["triggered_tracks"] += 1
                v["triggered_tracks"] += 1
            if triggered and has_re:
                counts["true_trigger_tracks"] += 1
                v["true_trigger_tracks"] += 1
            elif triggered and not has_re:
                counts["false_trigger_tracks"] += 1
                v["false_trigger_tracks"] += 1
            elif (not triggered) and has_re:
                counts["no_trigger_reentry_tracks"] += 1
                v["no_trigger_reentry_tracks"] += 1
            else:
                counts["nontrigger_nonreentry_tracks"] += 1

            base_aj = point_aj(btracks[qi], gttracks[qi], bvis[qi], gv[qi], qt, start_t=0)
            b2_aj = point_aj(b2tracks[qi], gttracks[qi], b2vis[qi], gv[qi], qt, start_t=0)
            delta_full = float(b2_aj - base_aj)
            if trig is not None:
                base_post = point_aj(btracks[qi], gttracks[qi], bvis[qi], gv[qi], qt, start_t=max(0, trig - 1))
                b2_post = point_aj(b2tracks[qi], gttracks[qi], b2vis[qi], gv[qi], qt, start_t=max(0, trig - 1))
                delta_post = float(b2_post - base_post)
                dist_bo = px_dist(btracks[qi, trig], otracks[qi, trig])
                base_gt_err = px_dist(btracks[qi, trig], gttracks[qi, trig]) if bool(gv[qi, trig]) else None
                over_gt_err = px_dist(otracks[qi, trig], gttracks[qi, trig]) if bool(gv[qi, trig]) else None
            else:
                base_post = b2_post = delta_post = None
                dist_bo = base_gt_err = over_gt_err = None

            row = {
                "video_id": vid,
                "query_idx": int(qi),
                "query_t": int(qt),
                "has_gt_reentry": has_re,
                "n_gt_reentry_events": int(len(events)),
                "first_gt_reentry_t": int(events[0]["reentry_frame"]) if events else None,
                "triggered": triggered,
                "first_trigger_t": trig,
                "class": "true_trigger" if triggered and has_re else ("false_trigger" if triggered else ("no_trigger_reentry" if has_re else "none")),
                "gt_visible_at_trigger": bool(gv[qi, trig]) if trig is not None else None,
                "base_visible_at_trigger": bool(bvis[qi, trig]) if trig is not None else None,
                "override_visible_at_trigger": bool(ovis[qi, trig]) if trig is not None else None,
                "base_invisible_run_before_trigger": invisible_run_before(bvis[qi], trig) if trig is not None else None,
                "base_override_dist_256_at_trigger": round(dist_bo, 4) if dist_bo is not None else None,
                "base_gt_err_256_at_trigger": round(base_gt_err, 4) if base_gt_err is not None else None,
                "override_gt_err_256_at_trigger": round(over_gt_err, 4) if over_gt_err is not None else None,
                "fixed_query_AJ_256": round(base_aj, 6),
                "b2_query_AJ_256": round(b2_aj, 6),
                "delta_b2_minus_fixed_full_AJ": round(delta_full, 6),
                "fixed_post_AJ_256": round(base_post, 6) if base_post is not None else None,
                "b2_post_AJ_256": round(b2_post, 6) if b2_post is not None else None,
                "delta_b2_minus_fixed_post_AJ": round(delta_post, 6) if delta_post is not None else None,
            }
            rows.append(row)
            if row["class"] == "false_trigger":
                false_rows.append(row)
                if row["gt_visible_at_trigger"]:
                    v["false_gt_visible_at_trigger"] += 1
                else:
                    v["false_gt_invisible_at_trigger"] += 1
                v["false_delta_full"].append(delta_full)
                if delta_post is not None:
                    v["false_delta_post"].append(delta_post)
                if dist_bo is not None:
                    v["false_base_override_dist"].append(dist_bo)
                if delta_full >= -0.01:
                    v["false_low_cost_full_ge_minus_0p01"] += 1
                if delta_full < -0.05:
                    v["false_harmful_full_gt_0p05"] += 1
                if delta_full < -0.10:
                    v["false_severe_full_gt_0p10"] += 1
            elif row["class"] == "true_trigger":
                true_rows.append(row)
                v["true_delta_full"].append(delta_full)
                if delta_post is not None:
                    v["true_delta_post"].append(delta_post)

    false_delta_full = [float(r["delta_b2_minus_fixed_full_AJ"]) for r in false_rows]
    false_delta_post = [float(r["delta_b2_minus_fixed_post_AJ"]) for r in false_rows if r["delta_b2_minus_fixed_post_AJ"] is not None]
    true_delta_full = [float(r["delta_b2_minus_fixed_full_AJ"]) for r in true_rows]
    true_delta_post = [float(r["delta_b2_minus_fixed_post_AJ"]) for r in true_rows if r["delta_b2_minus_fixed_post_AJ"] is not None]

    false_gt_visible = sum(1 for r in false_rows if r["gt_visible_at_trigger"])
    false_gt_invisible = sum(1 for r in false_rows if r["gt_visible_at_trigger"] is False)
    false_low_cost = sum(1 for r in false_rows if float(r["delta_b2_minus_fixed_full_AJ"]) >= -0.01)
    false_harmful = sum(1 for r in false_rows if float(r["delta_b2_minus_fixed_full_AJ"]) < -0.05)
    false_severe = sum(1 for r in false_rows if float(r["delta_b2_minus_fixed_full_AJ"]) < -0.10)

    video_rows = []
    for st in video.values():
        fd = st.pop("false_delta_full")
        fdp = st.pop("false_delta_post")
        td = st.pop("true_delta_full")
        tdp = st.pop("true_delta_post")
        bod = st.pop("false_base_override_dist")
        st["false_delta_full_stats"] = summarize_vals(fd)
        st["false_delta_post_stats"] = summarize_vals(fdp)
        st["true_delta_full_stats"] = summarize_vals(td)
        st["true_delta_post_stats"] = summarize_vals(tdp)
        st["false_base_override_dist_stats"] = summarize_vals(bod)
        st["false_trigger_rate"] = round(st["false_trigger_tracks"] / max(st["total_tracks"], 1), 6)
        st["false_harmful_rate_among_false"] = round(st["false_harmful_full_gt_0p05"] / max(st["false_trigger_tracks"], 1), 6)
        video_rows.append(st)
    video_rows = sorted(video_rows, key=lambda x: (x["false_harmful_full_gt_0p05"], x["false_trigger_tracks"]), reverse=True)

    worst_false_rows = sorted(false_rows, key=lambda r: float(r["delta_b2_minus_fixed_full_AJ"]))[:50]
    largest_true_gain_rows = sorted(true_rows, key=lambda r: float(r["delta_b2_minus_fixed_full_AJ"]), reverse=True)[:50]

    summary = {
        "rule": "base invisible run >= 1 and override visible; pre=1; post=full",
        "counts": counts,
        "false_trigger_summary": {
            "false_triggers": len(false_rows),
            "gt_visible_at_trigger": int(false_gt_visible),
            "gt_invisible_at_trigger": int(false_gt_invisible),
            "gt_visible_rate": round(false_gt_visible / max(len(false_rows), 1), 6),
            "low_cost_full_delta_ge_minus_0p01": int(false_low_cost),
            "low_cost_rate": round(false_low_cost / max(len(false_rows), 1), 6),
            "harmful_full_delta_lt_minus_0p05": int(false_harmful),
            "harmful_rate": round(false_harmful / max(len(false_rows), 1), 6),
            "severe_full_delta_lt_minus_0p10": int(false_severe),
            "severe_rate": round(false_severe / max(len(false_rows), 1), 6),
            "delta_full_stats": summarize_vals(false_delta_full),
            "delta_post_stats": summarize_vals(false_delta_post),
        },
        "true_trigger_summary": {
            "true_triggers": len(true_rows),
            "delta_full_stats": summarize_vals(true_delta_full),
            "delta_post_stats": summarize_vals(true_delta_post),
        },
        "top_videos_by_harmful_false_triggers": video_rows[:15],
        "worst_false_trigger_rows": worst_false_rows,
        "largest_true_trigger_gain_rows": largest_true_gain_rows,
        "all_video_rows": video_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "all_track_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "false_trigger_rows.jsonl").open("w") as f:
        for r in false_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["counts", "false_trigger_summary", "true_trigger_summary", "top_videos_by_harmful_false_triggers"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
