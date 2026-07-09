#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events

OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_trigger_quality")
BASE_CACHE = Path("outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
OVERRIDE_CACHE = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt")


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = t - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_times(base_v: np.ndarray, over_v: np.ndarray, query_t: int, k: int = 1) -> List[int]:
    times = []
    for t in range(max(1, query_t + 1), len(base_v)):
        if invisible_run_before(base_v, t) >= k and bool(over_v[t]):
            times.append(t)
    return times


def closest_delay(trigger_t: int, events: List[Dict[str, Any]]) -> int | None:
    if not events:
        return None
    frames = [int(e["reentry_frame"]) for e in events]
    return int(trigger_t - min(frames, key=lambda x: abs(trigger_t - x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--base-cache", default=str(BASE_CACHE))
    ap.add_argument("--override-cache", default=str(OVERRIDE_CACHE))
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--match-window", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
    over = torch.load(args.override_cache, map_location="cpu", weights_only=False)
    rows = []
    total_tracks = 0
    gt_reentry_tracks = 0
    triggered_tracks = 0
    triggered_with_reentry = 0
    triggered_without_reentry = 0
    no_trigger_with_reentry = 0
    all_delays = []
    first_delays = []
    matched_trigger_count = 0
    total_trigger_count = 0
    trigger_before = trigger_at = trigger_after = 0
    by_video: Dict[str, Dict[str, Any]] = {}

    for b, o in zip(base["records"], over["records"]):
        vid = str(b["video_id"])
        bvis = npy(b["pred_visibility"], bool)
        ovis = npy(o["pred_visibility"], bool)
        gtvis = npy(b["gt_visibility"], bool)
        qpts = npy(b["query_points"], np.float32)
        n, t_len = gtvis.shape
        vstat = by_video.setdefault(vid, {
            "video_id": vid,
            "tracks": 0,
            "gt_reentry_tracks": 0,
            "triggered_tracks": 0,
            "triggered_with_reentry": 0,
            "triggered_without_reentry": 0,
            "no_trigger_with_reentry": 0,
            "trigger_count": 0,
            "matched_trigger_count": 0,
            "delays": [],
        })
        for qi in range(n):
            total_tracks += 1
            vstat["tracks"] += 1
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            events = find_reentry_events(gtvis[qi], qt)
            has_reentry = bool(events)
            if has_reentry:
                gt_reentry_tracks += 1
                vstat["gt_reentry_tracks"] += 1
            ts = trigger_times(bvis[qi], ovis[qi], qt, k=int(args.k))
            if ts:
                triggered_tracks += 1
                vstat["triggered_tracks"] += 1
                if has_reentry:
                    triggered_with_reentry += 1
                    vstat["triggered_with_reentry"] += 1
                else:
                    triggered_without_reentry += 1
                    vstat["triggered_without_reentry"] += 1
            elif has_reentry:
                no_trigger_with_reentry += 1
                vstat["no_trigger_with_reentry"] += 1
            total_trigger_count += len(ts)
            vstat["trigger_count"] += len(ts)
            delays = []
            for tt in ts:
                d = closest_delay(tt, events)
                if d is None:
                    continue
                delays.append(d)
                all_delays.append(d)
                vstat["delays"].append(d)
                if abs(d) <= int(args.match_window):
                    matched_trigger_count += 1
                    vstat["matched_trigger_count"] += 1
                if d < 0:
                    trigger_before += 1
                elif d == 0:
                    trigger_at += 1
                else:
                    trigger_after += 1
            if delays:
                first_delays.append(delays[0])
            rows.append({
                "video_id": vid,
                "query_idx": int(qi),
                "query_t": int(qt),
                "has_gt_reentry": has_reentry,
                "n_gt_reentry_events": int(len(events)),
                "first_gt_reentry_t": int(events[0]["reentry_frame"]) if events else None,
                "n_triggers": int(len(ts)),
                "first_trigger_t": int(ts[0]) if ts else None,
                "first_delay_to_gt": int(delays[0]) if delays else None,
                "min_abs_delay_to_gt": int(min(delays, key=lambda x: abs(x))) if delays else None,
            })

    def mean(xs): return round(float(np.mean(xs)), 6) if xs else None
    def median(xs): return round(float(np.median(xs)), 6) if xs else None
    def p90_abs(xs): return round(float(np.percentile(np.abs(xs), 90)), 6) if xs else None
    per_video = []
    for vid, st in by_video.items():
        delays = st.pop("delays")
        st["trigger_precision_track"] = round(st["triggered_with_reentry"] / max(st["triggered_tracks"], 1), 6)
        st["trigger_recall_track"] = round(st["triggered_with_reentry"] / max(st["gt_reentry_tracks"], 1), 6)
        st["matched_trigger_rate"] = round(st["matched_trigger_count"] / max(st["trigger_count"], 1), 6)
        st["delay_mean"] = mean(delays)
        st["delay_median"] = median(delays)
        st["delay_abs_p90"] = p90_abs(delays)
        per_video.append(st)
    per_video = sorted(per_video, key=lambda x: x["video_id"])

    summary = {
        "rule": "base invisible run >= k and override visible",
        "k": int(args.k),
        "match_window": int(args.match_window),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "total_tracks": int(total_tracks),
        "gt_reentry_tracks": int(gt_reentry_tracks),
        "triggered_tracks": int(triggered_tracks),
        "triggered_with_reentry": int(triggered_with_reentry),
        "triggered_without_reentry": int(triggered_without_reentry),
        "no_trigger_with_reentry": int(no_trigger_with_reentry),
        "trigger_precision_track": round(triggered_with_reentry / max(triggered_tracks, 1), 6),
        "trigger_recall_track": round(triggered_with_reentry / max(gt_reentry_tracks, 1), 6),
        "triggered_track_rate": round(triggered_tracks / max(total_tracks, 1), 6),
        "total_trigger_count": int(total_trigger_count),
        "matched_trigger_count": int(matched_trigger_count),
        "matched_trigger_rate": round(matched_trigger_count / max(total_trigger_count, 1), 6),
        "delay_mean": mean(all_delays),
        "delay_median": median(all_delays),
        "delay_abs_p90": p90_abs(all_delays),
        "first_delay_mean": mean(first_delays),
        "first_delay_median": median(first_delays),
        "trigger_before_count": int(trigger_before),
        "trigger_at_count": int(trigger_at),
        "trigger_after_count": int(trigger_after),
        "trigger_before_rate": round(trigger_before / max(len(all_delays), 1), 6),
        "trigger_at_rate": round(trigger_at / max(len(all_delays), 1), 6),
        "trigger_after_rate": round(trigger_after / max(len(all_delays), 1), 6),
        "per_video": per_video,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "trigger_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
