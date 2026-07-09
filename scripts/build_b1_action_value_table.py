#!/usr/bin/env python3
"""Build B1 candidate action-value table and candidate-action oracle.

This script evaluates a fixed set of deployable B1 fusion actions at the
per-query AJ_RD_256 level, then computes an action oracle over those actions.
It is a diagnostic gate before training any learned router.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

DEFAULT_OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_action_oracle")

DEFAULT_ACTIONS: Dict[str, str] = {
    # Current baseline / near-best variants.
    "vis4_gated288": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt",
    "vis4_gated320": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated320.pt",
    "vis4_gated384": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated384.pt",
    "vis4_gated256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated256.pt",
    "all4_gated192": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/all4_gated192.pt",
    "all4_gated256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/all4_gated256.pt",
    "all4_gated288": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/all4_gated288.pt",
    "all4_gated320": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/all4_gated320.pt",
    # Union / half alternatives.
    "b1_all_median4_union": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_all_median4_union.pt",
    "b1_visible_median4_union": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_visible_median4_union.pt",
    "b1_all_median4_half": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/b1_all_median4_half.pt",
    # Old-3 fallbacks.
    "old3_all_median_gated144": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/old3_all_median_gated144.pt",
    "old3_all_median_union": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick/old3_all_median_union.pt",
    # Hybrid / trim variants from refine.
    "trimTrack24_visG256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/trimTrack24_visG256.pt",
    "trimTrack48_visG256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/trimTrack48_visG256.pt",
    "hybridTrack64_visG256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/hybridTrack64_visG256.pt",
    "hybridTrack128_visG256": "outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/hybridTrack128_visG256.pt",
}


def _query_scores_for_cache(cache_path: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[str, Any]]:
    payload = load_attempt0_cache(cache_path)
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    per_video_counts: Counter[str] = Counter()
    per_video_scores: Dict[str, List[float]] = defaultdict(list)
    for r in payload["records"]:
        vid = str(r["video_id"])
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        metrics = compute_reentry_metrics(
            pred_tracks=np.asarray(r["pred_tracks"], dtype=np.float32),
            gt_tracks=np.asarray(r["gt_tracks"], dtype=np.float32),
            pred_vis=np.asarray(r["pred_visibility"], dtype=bool),
            gt_vis=np.asarray(r["gt_visibility"], dtype=bool),
            query_points=np.asarray(r["query_points"], dtype=np.float32),
            height=h,
            width=w,
        )
        for q in metrics.get("per_query", []):
            qi = int(q["query_idx"])
            score = (q.get("ajrd_summary_256") or {}).get("aj_rd")
            if score is None:
                continue
            score = float(score)
            key = (vid, qi)
            rows[key] = {
                "video_id": vid,
                "query_idx": qi,
                "query_t": int(q.get("query_t", -1)),
                "reentry_t": int(q.get("reentry_t", -1)),
                "occ_length": int(q.get("occ_length", -1)),
                "score": score,
                "proxy": float(q.get("aj_proxy", 0.0)),
                "proxy_error_px": float(q.get("proxy_error_px", 0.0)),
            }
            per_video_counts[vid] += 1
            per_video_scores[vid].append(score)
    flat_scores = [v["score"] for v in rows.values()]
    summary = {
        "cache_path": str(cache_path),
        "n_queries": len(rows),
        "mean_score": round(float(np.mean(flat_scores)), 4) if flat_scores else None,
        "per_video_mean": {
            vid: round(float(np.mean(vals)), 4) for vid, vals in sorted(per_video_scores.items())
        },
        "per_video_counts": dict(sorted(per_video_counts.items())),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--baseline-action", default="vis4_gated288")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = {name: Path(path) for name, path in DEFAULT_ACTIONS.items()}
    missing = {name: str(path) for name, path in actions.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing action caches: {missing}")
    if args.baseline_action not in actions:
        raise KeyError(f"baseline action {args.baseline_action!r} not in actions")

    all_rows: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    action_summaries: Dict[str, Any] = {}
    for name, path in actions.items():
        print(f"ACTION {name} {path}", flush=True)
        rows, summary = _query_scores_for_cache(path)
        all_rows[name] = rows
        action_summaries[name] = summary
        print(f"  n={summary['n_queries']} mean={summary['mean_score']}", flush=True)

    baseline_rows = all_rows[args.baseline_action]
    keys = sorted(baseline_rows.keys())
    if not keys:
        raise RuntimeError("baseline has no query scores")

    table_path = out_dir / "action_value_table.jsonl"
    action_usage = Counter()
    rows_out: List[Dict[str, Any]] = []
    baseline_scores: List[float] = []
    oracle_scores: List[float] = []
    best_delta_vals: List[float] = []
    by_video: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"baseline": [], "oracle": [], "delta": []})

    with table_path.open("w") as f:
        for key in keys:
            bmeta = baseline_rows[key]
            scores: Dict[str, float] = {}
            for name in actions:
                row = all_rows[name].get(key)
                if row is not None and row.get("score") is not None:
                    scores[name] = float(row["score"])
            if args.baseline_action not in scores:
                continue
            best_action, best_score = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
            baseline_score = scores[args.baseline_action]
            delta = best_score - baseline_score
            action_usage[best_action] += 1
            baseline_scores.append(baseline_score)
            oracle_scores.append(best_score)
            best_delta_vals.append(delta)
            vid = bmeta["video_id"]
            by_video[vid]["baseline"].append(baseline_score)
            by_video[vid]["oracle"].append(best_score)
            by_video[vid]["delta"].append(delta)
            out = {
                "video_id": vid,
                "query_idx": int(bmeta["query_idx"]),
                "query_t": int(bmeta.get("query_t", -1)),
                "reentry_t": int(bmeta.get("reentry_t", -1)),
                "occ_length": int(bmeta.get("occ_length", -1)),
                "scores": {k: round(v, 6) for k, v in sorted(scores.items())},
                "baseline_action": args.baseline_action,
                "baseline_score": round(baseline_score, 6),
                "best_action": best_action,
                "best_score": round(best_score, 6),
                "best_delta_vs_baseline": round(delta, 6),
            }
            rows_out.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    baseline_mean = float(np.mean(baseline_scores))
    oracle_mean = float(np.mean(oracle_scores))
    delta_mean = oracle_mean - baseline_mean

    single_action_ranking = []
    for name, rows in all_rows.items():
        vals = [rows[k]["score"] for k in keys if k in rows]
        single_action_ranking.append({
            "action": name,
            "mean_score_on_baseline_queries": round(float(np.mean(vals)), 6) if vals else None,
            "n": len(vals),
            "delta_vs_baseline": round(float(np.mean(vals)) - baseline_mean, 6) if vals else None,
        })
    single_action_ranking.sort(key=lambda r: r["mean_score_on_baseline_queries"] if r["mean_score_on_baseline_queries"] is not None else -1, reverse=True)

    per_video = []
    for vid, vals in sorted(by_video.items()):
        b = float(np.mean(vals["baseline"]))
        o = float(np.mean(vals["oracle"]))
        per_video.append({
            "video_id": vid,
            "n": len(vals["baseline"]),
            "baseline": round(b, 6),
            "action_oracle": round(o, 6),
            "delta": round(o - b, 6),
        })
    per_video.sort(key=lambda r: r["delta"], reverse=True)

    thresholds = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    delta_threshold_counts = {
        str(t): int(sum(1 for x in best_delta_vals if x > t)) for t in thresholds
    }

    summary = {
        "baseline_action": args.baseline_action,
        "n_queries": len(baseline_scores),
        "n_actions": len(actions),
        "baseline_score": round(baseline_mean, 6),
        "candidate_action_oracle_score": round(oracle_mean, 6),
        "candidate_action_oracle_gain": round(delta_mean, 6),
        "single_action_ranking": single_action_ranking,
        "action_usage": dict(action_usage.most_common()),
        "delta_threshold_counts": delta_threshold_counts,
        "per_video": per_video,
        "top_video_gains": per_video[:10],
        "worst_video_gains": sorted(per_video, key=lambda r: r["delta"])[:10],
        "action_summaries": action_summaries,
        "table_path": str(table_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=== ACTION ORACLE SUMMARY ===")
    print(json.dumps({k: summary[k] for k in ["baseline_action", "n_queries", "n_actions", "baseline_score", "candidate_action_oracle_score", "candidate_action_oracle_gain", "action_usage", "delta_threshold_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
