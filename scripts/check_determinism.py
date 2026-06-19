#!/usr/bin/env python3
"""Determinism check: run eval_aj_rd_from_cache.py twice on the same cache,
verifying that core metric differences are ≤ 1e-6.

Plan §4.6: "同一 cache 重跑 2 次，核心指标差异 <= 1e-6"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_eval(cache_path: str, max_videos: int = 0) -> dict:
    """Run eval_aj_rd_from_cache.py and return the summary dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve().parent / "eval_aj_rd_from_cache.py"),
        "--cache-path", cache_path,
        "--output-json", out_path,
    ]
    if max_videos > 0:
        cmd.extend(["--max-videos", str(max_videos)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"eval failed:\n{result.stdout}\n{result.stderr}")

    with open(out_path) as f:
        data = json.load(f)

    Path(out_path).unlink(missing_ok=True)
    return data


def check_determinism(cache_path: str, max_videos: int = 0, tol: float = 1e-6) -> dict:
    """Run twice and compare."""
    print(f"Running determinism check on {cache_path}...")
    print(f"  Max videos: {max_videos if max_videos > 0 else 'all'}")

    run1 = run_eval(cache_path, max_videos)
    run2 = run_eval(cache_path, max_videos)

    # Keys to compare
    compare_keys = [
        "first_reentry_frame_proxy",
        "true_AJ_RD",
        "true_AJ_RD_256",
        "n_reentry_queries_total",
    ]

    results = {}
    all_pass = True

    for key in compare_keys:
        v1 = run1.get(key)
        v2 = run2.get(key)
        if v1 is None and v2 is None:
            diff = 0.0
        elif v1 is None or v2 is None:
            diff = float("inf")
        else:
            diff = abs(float(v1) - float(v2))

        pass_ = diff <= tol
        if not pass_:
            all_pass = False

        results[key] = {
            "run1": v1,
            "run2": v2,
            "diff": diff,
            "pass": pass_,
        }
        status = "PASS" if pass_ else "FAIL"
        print(f"  {key}: run1={v1}, run2={v2}, diff={diff:.2e} -> {status}")

    # Also compare per-video AJ_RD
    pv1 = {v["video_id"]: v for v in run1.get("per_video", [])}
    pv2 = {v["video_id"]: v for v in run2.get("per_video", [])}
    per_video_diffs = []
    for vid in pv1:
        if vid not in pv2:
            continue
        for key in ["first_reentry_frame_proxy", "true_AJ_RD"]:
            v1 = pv1[vid].get(key)
            v2 = pv2[vid].get(key)
            if v1 is not None and v2 is not None:
                diff = abs(float(v1) - float(v2))
                per_video_diffs.append(diff)
                if diff > tol:
                    print(f"  Per-video {vid}.{key}: diff={diff:.2e} -> FAIL")
                    all_pass = False

    if per_video_diffs:
        max_pv_diff = max(per_video_diffs)
        print(f"  Per-video max diff: {max_pv_diff:.2e}")
    else:
        max_pv_diff = 0.0

    results["_summary"] = {
        "cache_path": cache_path,
        "n_comparisons": len(compare_keys) + len(per_video_diffs),
        "all_pass": all_pass,
        "max_diff": max(
            [v["diff"] for v in results.values() if isinstance(v, dict) and "diff" in v] + [0.0]
        ),
        "tol": tol,
    }

    status = "PASS" if all_pass else "FAIL"
    print(f"\n  Overall: {status} (max diff <= {tol})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    results = check_determinism(args.cache_path, args.max_videos, args.tol)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
