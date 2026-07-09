#!/usr/bin/env python3
"""Paired per-video statistics for ReEntry method comparisons.

This script is designed for paper-level sanity checks.  It aligns methods by
video_id and reports per-video AJ_RD_256, AJ_256, OA_256 plus paired deltas,
bootstrap confidence intervals, and exact sign-test p-values.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics  # noqa: E402
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_method(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"--method must be name=path, got: {s}")
    name, path = s.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Bad --method argument: {s}")
    return name, path


def per_video_metrics(cache_path: str | Path) -> List[Dict[str, Any]]:
    payload = load_cache(cache_path)
    rows: List[Dict[str, Any]] = []
    for r in payload["records"]:
        vid = str(r["video_id"])
        pred = npy(r["pred_tracks"], np.float32)
        gt = npy(r["gt_tracks"], np.float32)
        pvis = npy(r["pred_visibility"], bool)
        gvis = npy(r["gt_visibility"], bool)
        q = npy(r["query_points"], np.float32)
        h, w = int(npy(r["original_size"])[0]), int(npy(r["original_size"])[1])

        ajrd = compute_reentry_metrics(
            pred_tracks=pred,
            gt_tracks=gt,
            pred_vis=pvis,
            gt_vis=gvis,
            query_points=q,
            height=h,
            width=w,
        )
        std = compute_tapvid_metrics(
            torch.from_numpy(pred),
            torch.from_numpy(gt),
            torch.from_numpy(pvis),
            torch.from_numpy(gvis),
            torch.from_numpy(q),
            resolution=256,
            query_mode="strided",
        )
        rows.append({
            "video_id": vid,
            "n_queries": int(q.shape[0]),
            "n_reentry_queries": int(ajrd.get("n_reentry_queries", 0)),
            "AJ_RD_256": None if ajrd.get("true_AJ_RD_256") is None else float(ajrd["true_AJ_RD_256"]),
            "AJ_256": float(std.get("AJ", 0.0)) * 100.0,
            "OA_256": float(std.get("OA", 0.0)) * 100.0,
            "delta_avg_256": float(std.get("average_pts_within_thresh", 0.0)) * 100.0,
        })
    return rows


def exact_sign_test_p(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    # two-sided binomial under p=0.5
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * prob)


def paired_summary(diffs: np.ndarray, *, seed: int = 20260703, n_boot: int = 10000) -> Dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    n = int(diffs.size)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = float(np.mean(diffs[rng.integers(0, n, size=n)]))
    wins = int(np.sum(diffs > 1e-12))
    losses = int(np.sum(diffs < -1e-12))
    ties = int(n - wins - losses)
    return {
        "n": n,
        "mean": round(float(np.mean(diffs)), 6),
        "median": round(float(np.median(diffs)), 6),
        "std": round(float(np.std(diffs, ddof=1)), 6) if n > 1 else 0.0,
        "min": round(float(np.min(diffs)), 6),
        "max": round(float(np.max(diffs)), 6),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_non_tie": round(float(wins / max(wins + losses, 1)), 6),
        "sign_test_p_two_sided": round(float(exact_sign_test_p(wins, losses)), 8),
        "bootstrap_mean_ci95": [
            round(float(np.percentile(boot, 2.5)), 6),
            round(float(np.percentile(boot, 97.5)), 6),
        ],
    }


def align_by_video(method_rows: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
    maps = {m: {r["video_id"]: r for r in rows} for m, rows in method_rows.items()}
    common = sorted(set.intersection(*(set(v.keys()) for v in maps.values())))
    aligned = {m: {vid: maps[m][vid] for vid in common} for m in maps}
    return common, aligned


def aggregate_method(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean_key(k: str) -> float | None:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        return round(float(np.mean(vals)), 6) if vals else None
    return {
        "n_videos": len(rows),
        "n_queries": int(sum(int(r.get("n_queries", 0)) for r in rows)),
        "n_reentry_queries": int(sum(int(r.get("n_reentry_queries", 0)) for r in rows)),
        "video_mean_AJ_RD_256": mean_key("AJ_RD_256"),
        "video_mean_AJ_256": mean_key("AJ_256"),
        "video_mean_OA_256": mean_key("OA_256"),
    }


def make_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Paired Video Statistics — {result['setting']}")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("| Method | Cache |")
    lines.append("|---|---|")
    for m, p in result["methods"].items():
        lines.append(f"| {m} | `{p}` |")
    lines.append("")
    lines.append("## Video-weighted method means")
    lines.append("")
    lines.append("| Method | Videos | Re-entry queries | AJ_RD_256 | AJ_256 | OA_256 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m, a in result["method_aggregates"].items():
        lines.append(
            f"| {m} | {a['n_videos']} | {a['n_reentry_queries']} | "
            f"{a['video_mean_AJ_RD_256']:.6f} | {a['video_mean_AJ_256']:.4f} | {a['video_mean_OA_256']:.4f} |"
        )
    lines.append("")
    lines.append(f"## Paired comparisons against `{result['primary']}`")
    lines.append("")
    for comp_name, comp in result["comparisons"].items():
        lines.append(f"### {comp_name}")
        lines.append("")
        lines.append("| Metric | Mean diff | 95% bootstrap CI | Wins / Losses / Ties | Sign-test p |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric, s in comp["metrics"].items():
            ci = s.get("bootstrap_mean_ci95", [None, None])
            ci_str = f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci[0] is not None else "NA"
            lines.append(
                f"| {metric} | {s.get('mean', 0):.6f} | {ci_str} | "
                f"{s.get('wins', 0)} / {s.get('losses', 0)} / {s.get('ties', 0)} | {s.get('sign_test_p_two_sided', 1):.6f} |"
            )
        lines.append("")
    lines.append("## Per-video table")
    lines.append("")
    primary = result["primary"]
    header = ["video_id"]
    for m in result["method_order"]:
        header.extend([f"{m}_AJ_RD", f"{m}_AJ", f"{m}_OA"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    for row in result["per_video_aligned"]:
        vals = [row["video_id"]]
        for m in result["method_order"]:
            vals.extend([
                f"{row[m]['AJ_RD_256']:.6f}" if row[m]["AJ_RD_256"] is not None else "NA",
                f"{row[m]['AJ_256']:.4f}",
                f"{row[m]['OA_256']:.4f}",
            ])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired per-video statistics for ReEntry methods.")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--method", action="append", required=True, help="name=cache_path; repeat for each method")
    ap.add_argument("--primary", required=True, help="Primary method name to compare against baselines")
    ap.add_argument("--baseline", action="append", default=None, help="Baseline method name; repeat. Defaults to all non-primary")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260703)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    methods = dict(parse_method(s) for s in args.method)
    if args.primary not in methods:
        raise ValueError(f"primary {args.primary} not in methods: {list(methods)}")
    baselines = args.baseline or [m for m in methods if m != args.primary]
    for b in baselines:
        if b not in methods:
            raise ValueError(f"baseline {b} not in methods")

    method_rows = {}
    for name, path in methods.items():
        print(json.dumps({"computing": name, "path": path}, ensure_ascii=False), flush=True)
        method_rows[name] = per_video_metrics(path)

    common_videos, aligned = align_by_video(method_rows)
    per_video_aligned: List[Dict[str, Any]] = []
    for vid in common_videos:
        row = {"video_id": vid}
        for m in methods:
            row[m] = aligned[m][vid]
        per_video_aligned.append(row)

    comparisons: Dict[str, Any] = {}
    metrics = ["AJ_RD_256", "AJ_256", "OA_256"]
    for b in baselines:
        comp_key = f"{args.primary}_minus_{b}"
        comp = {"baseline": b, "primary": args.primary, "metrics": {}}
        for metric in metrics:
            diffs = []
            for vid in common_videos:
                a = aligned[args.primary][vid].get(metric)
                c = aligned[b][vid].get(metric)
                if a is None or c is None:
                    continue
                diffs.append(float(a) - float(c))
            comp["metrics"][metric] = paired_summary(np.asarray(diffs), seed=args.seed, n_boot=args.bootstrap)
        comparisons[comp_key] = comp

    result = {
        "setting": args.setting,
        "primary": args.primary,
        "methods": methods,
        "method_order": list(methods.keys()),
        "common_videos": common_videos,
        "method_aggregates": {
            m: aggregate_method([aligned[m][vid] for vid in common_videos]) for m in methods
        },
        "comparisons": comparisons,
        "per_video_aligned": per_video_aligned,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "common_videos": len(common_videos)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
