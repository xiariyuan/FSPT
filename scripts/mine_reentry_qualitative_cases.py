#!/usr/bin/env python3
"""Mine qualitative cases for ReEntry-VisCalibrator.

The goal is to create paper-useful case lists before rendering figures:

1. base_fail_learned_success:
   Base has poor AJ_RD on a re-entry query, learned substantially improves.
2. det_over_recovery_learned_stable:
   Deterministic positive-only recovery turns on more frames than learned, and
   learned has better query AJ_RD / fewer false-visible frames.
3. learned_failure:
   Learned is still poor or worse than base/deterministic.

The script works directly from unified caches and writes JSON + Markdown reports.
It does not require raw video frames; it identifies video_id/query_idx/frame ranges
that can later be rendered into visual panels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_aj_rd_from_cache import compute_reentry_metrics  # noqa: E402
from utils.coords import yx_norm_to_xy_256  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def cache_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(r["video_id"]): r for r in payload["records"]}


def reentry_queries(record: Dict[str, Any], pred_tracks: np.ndarray, pred_vis: np.ndarray) -> Dict[int, Dict[str, Any]]:
    h, w = int(npy(record["original_size"])[0]), int(npy(record["original_size"])[1])
    gt_tracks = npy(record["gt_tracks"], np.float32)
    gt_vis = npy(record["gt_visibility"], bool)
    query_points = npy(record["query_points"], np.float32)
    m = compute_reentry_metrics(
        pred_tracks=pred_tracks,
        gt_tracks=gt_tracks,
        pred_vis=pred_vis,
        gt_vis=gt_vis,
        query_points=query_points,
        height=h,
        width=w,
    )
    return {int(q["query_idx"]): q for q in m.get("per_query", [])}


def query_ajrd(q: Dict[str, Any]) -> float | None:
    s = q.get("ajrd_summary_256", {})
    v = s.get("aj_rd")
    return None if v is None else float(v)


def first_reentry_t_from_gt(gt_vis: np.ndarray, query_t: int) -> int | None:
    # Find first invisible run after query followed by visible.
    t = int(query_t) + 1
    n = int(gt_vis.shape[0])
    while t < n:
        if bool(gt_vis[t]):
            t += 1
            continue
        # invisible run starts
        j = t
        while j < n and not bool(gt_vis[j]):
            j += 1
        if j < n and bool(gt_vis[j]):
            return int(j)
        t = j + 1
    return None


def segment_stats(record: Dict[str, Any], query_idx: int, center_t: int, radius: int, methods: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    qidx = int(query_idx)
    gt_vis = npy(record["gt_visibility"], bool)[qidx]
    gt_xy = yx_norm_to_xy_256(npy(record["gt_tracks"], np.float32)[qidx])
    n = int(gt_vis.shape[0])
    lo = max(0, int(center_t) - int(radius))
    hi = min(n - 1, int(center_t) + int(radius))
    out = {"frame_lo": lo, "frame_hi": hi, "center_t": int(center_t)}
    for name, rec in methods.items():
        pvis = npy(rec["pred_visibility"], bool)[qidx]
        pxy = yx_norm_to_xy_256(npy(rec["pred_tracks"], np.float32)[qidx])
        err = np.linalg.norm(pxy[lo : hi + 1] - gt_xy[lo : hi + 1], axis=-1)
        out[name] = {
            "visible_rate_segment": round(float(pvis[lo : hi + 1].mean()), 6),
            "gt_visible_rate_segment": round(float(gt_vis[lo : hi + 1].mean()), 6),
            "false_visible_frames_segment": int(np.sum(pvis[lo : hi + 1] & ~gt_vis[lo : hi + 1])),
            "missed_visible_frames_segment": int(np.sum((~pvis[lo : hi + 1]) & gt_vis[lo : hi + 1])),
            "mean_coord_error_256_segment": round(float(np.mean(err)), 4),
            "median_coord_error_256_segment": round(float(np.median(err)), 4),
        }
    return out


def case_base_fail_learned_success(item: Dict[str, Any]) -> float:
    b = item["ajrd"].get("base")
    l = item["ajrd"].get("learned")
    if b is None or l is None:
        return -1e9
    # Prefer poor base and strong learned improvement.
    return (l - b) + 0.25 * max(0.0, 0.35 - b) + 0.05 * item.get("occ_length", 0) / 50.0


def case_det_over_recovery(item: Dict[str, Any]) -> float:
    d = item["ajrd"].get("det")
    l = item["ajrd"].get("learned")
    if d is None or l is None:
        return -1e9
    seg = item.get("segment", {})
    det_false = seg.get("det", {}).get("false_visible_frames_segment", 0)
    learn_false = seg.get("learned", {}).get("false_visible_frames_segment", 0)
    det_vis = seg.get("det", {}).get("visible_rate_segment", 0)
    learn_vis = seg.get("learned", {}).get("visible_rate_segment", 0)
    return (l - d) + 0.02 * max(0, det_false - learn_false) + 0.01 * max(0.0, det_vis - learn_vis)


def case_learned_failure(item: Dict[str, Any]) -> float:
    l = item["ajrd"].get("learned")
    b = item["ajrd"].get("base")
    d = item["ajrd"].get("det")
    if l is None:
        return -1e9
    worst_gap = 0.0
    if b is not None:
        worst_gap = max(worst_gap, b - l)
    if d is not None:
        worst_gap = max(worst_gap, d - l)
    return worst_gap + max(0.0, 0.35 - l)


def mine_cases(
    base_path: Path,
    rule_path: Path,
    det_path: Path,
    learned_path: Path,
    max_cases: int,
    segment_radius: int,
) -> Dict[str, Any]:
    payloads = {
        "base": load_cache(base_path),
        "rule": load_cache(rule_path),
        "det": load_cache(det_path),
        "learned": load_cache(learned_path),
    }
    maps = {k: cache_map(v) for k, v in payloads.items()}
    common_videos = sorted(set.intersection(*(set(m.keys()) for m in maps.values())))
    all_items: List[Dict[str, Any]] = []

    for vid in common_videos:
        records = {name: maps[name][vid] for name in maps}
        base_rec = records["base"]
        query_points = npy(base_rec["query_points"], np.float32)
        gt_vis_all = npy(base_rec["gt_visibility"], bool)
        perq = {}
        for name, rec in records.items():
            perq[name] = reentry_queries(
                base_rec,
                pred_tracks=npy(rec["pred_tracks"], np.float32),
                pred_vis=npy(rec["pred_visibility"], bool),
            )
        common_q = sorted(set.intersection(*(set(d.keys()) for d in perq.values())))
        for qi in common_q:
            q0 = perq["base"][qi]
            query_t = int(round(float(query_points[qi, 0])))
            reentry_t = int(q0.get("reentry_t", first_reentry_t_from_gt(gt_vis_all[qi], query_t) or query_t))
            occ_length = int(q0.get("occ_length", max(0, reentry_t - query_t)))
            ajrd = {name: query_ajrd(perq[name][qi]) for name in perq}
            proxy = {
                name: {
                    "proxy_error_px": perq[name][qi].get("proxy_error_px"),
                    "proxy_pred_visible": perq[name][qi].get("proxy_pred_visible"),
                    "first_reentry_proxy": perq[name][qi].get("aj_proxy"),
                }
                for name in perq
            }
            seg = segment_stats(base_rec, qi, reentry_t, segment_radius, records)
            item = {
                "video_id": vid,
                "query_idx": int(qi),
                "query_t": int(query_t),
                "reentry_t": int(reentry_t),
                "occ_length": occ_length,
                "ajrd": ajrd,
                "proxy": proxy,
                "segment": seg,
            }
            item["deltas"] = {
                "learned_minus_base": None if ajrd["learned"] is None or ajrd["base"] is None else round(float(ajrd["learned"] - ajrd["base"]), 6),
                "learned_minus_rule": None if ajrd["learned"] is None or ajrd["rule"] is None else round(float(ajrd["learned"] - ajrd["rule"]), 6),
                "learned_minus_det": None if ajrd["learned"] is None or ajrd["det"] is None else round(float(ajrd["learned"] - ajrd["det"]), 6),
            }
            all_items.append(item)

    success = sorted(all_items, key=case_base_fail_learned_success, reverse=True)[:max_cases]
    det_over = sorted(all_items, key=case_det_over_recovery, reverse=True)[:max_cases]
    failures = sorted(all_items, key=case_learned_failure, reverse=True)[:max_cases]
    return {
        "n_videos": len(common_videos),
        "n_reentry_query_items": len(all_items),
        "common_videos": common_videos,
        "categories": {
            "base_fail_learned_success": success,
            "det_over_recovery_learned_stable": det_over,
            "learned_failure": failures,
        },
    }


def format_case(item: Dict[str, Any]) -> str:
    a = item["ajrd"]
    d = item["deltas"]
    return (
        f"`{item['video_id']}` q={item['query_idx']} "
        f"query_t={item['query_t']} reentry_t={item['reentry_t']} occ={item['occ_length']} | "
        f"AJRD base={a.get('base')} rule={a.get('rule')} det={a.get('det')} learned={a.get('learned')} | "
        f"ΔL-B={d.get('learned_minus_base')} ΔL-Det={d.get('learned_minus_det')}"
    )


def write_markdown(result: Dict[str, Any], setting: str, out_md: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Qualitative Case Mining — {setting}")
    lines.append("")
    lines.append(f"Videos: {result['n_videos']}")
    lines.append(f"Re-entry query items: {result['n_reentry_query_items']}")
    lines.append("")
    lines.append("These are candidate cases for later rendering into paper figures. They identify video_id, query_idx, query_t, and reentry_t.")
    lines.append("")
    for cat, items in result["categories"].items():
        lines.append(f"## {cat}")
        lines.append("")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {format_case(item)}")
            seg = item.get("segment", {})
            if seg:
                lines.append(
                    f"   segment frames [{seg.get('frame_lo')}, {seg.get('frame_hi')}], "
                    f"det false-visible={seg.get('det', {}).get('false_visible_frames_segment')}, "
                    f"learned false-visible={seg.get('learned', {}).get('false_visible_frames_segment')}, "
                    f"det missed-visible={seg.get('det', {}).get('missed_visible_frames_segment')}, "
                    f"learned missed-visible={seg.get('learned', {}).get('missed_visible_frames_segment')}"
                )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mine qualitative cases for ReEntry methods.")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--rule-cache", required=True)
    ap.add_argument("--det-cache", required=True)
    ap.add_argument("--learned-cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--segment-radius", type=int, default=12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = mine_cases(
        Path(args.base_cache),
        Path(args.rule_cache),
        Path(args.det_cache),
        Path(args.learned_cache),
        max_cases=args.max_cases,
        segment_radius=args.segment_radius,
    )
    result["setting"] = args.setting
    result["inputs"] = {
        "base_cache": args.base_cache,
        "rule_cache": args.rule_cache,
        "det_cache": args.det_cache,
        "learned_cache": args.learned_cache,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, args.setting, Path(args.out_md))
    print(json.dumps({
        "setting": args.setting,
        "out_json": str(out_json),
        "out_md": args.out_md,
        "n_videos": result["n_videos"],
        "n_reentry_query_items": result["n_reentry_query_items"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
