#!/usr/bin/env python3
"""
Summarize long-occlusion TTA results across multiple seeds.

The input is one or more JSON files produced by `scripts/eval_long_occlusion_tta.py`.
For each threshold, the script aggregates base/refined/adapted metrics and the
delta between adapted and base. The goal is to produce a paper-ready table and a
quick diagnostic of whether the gain is coming from visibility calibration or
actual position improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRICS = ("AJ", "OA", "<avg", "avg_error_px", "median_error_px", "<16px")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt(mean_val: Optional[float], std_val: Optional[float], precision: int = 4, signed: bool = False) -> str:
    if mean_val is None:
        return "-"
    sign = "+" if signed and mean_val >= 0 else ""
    if std_val is None:
        return f"{sign}{mean_val:.{precision}f}"
    return f"{sign}{mean_val:.{precision}f} +/- {std_val:.{precision}f}"


def _load_run(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seed = raw.get("seed", None)
    if seed is None:
        seed = path.stem
    results = raw.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Unexpected results format in {path}")
    by_thr: Dict[int, Dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        thr = row.get("threshold", None)
        try:
            thr_i = int(thr)
        except Exception:
            continue
        by_thr[thr_i] = row
    return {"path": path, "seed": seed, "results": by_thr}


def _collect_metric(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _extract_metric(row: Dict[str, Any], group: str, metric: str) -> Optional[float]:
    block = row.get(group, None)
    if not isinstance(block, dict):
        return None
    return _safe_float(block.get(metric))


def _extract_block_metric(row: Dict[str, Any], block_name: str, metric: str) -> Optional[float]:
    block = row.get(block_name, None)
    if not isinstance(block, dict):
        return None
    return _safe_float(block.get(metric))


def _extract_delta_vs_refined(row: Dict[str, Any], metric: str) -> Optional[float]:
    return _extract_block_metric(row, "delta_vs_refined", metric)


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]:
    thresholds = sorted({thr for run in runs for thr in run["results"].keys()})
    out: Dict[int, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]] = {}
    for thr in thresholds:
        thr_rows = [run["results"][thr] for run in runs if thr in run["results"]]
        if not thr_rows:
            continue

        metric_stats: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
        for group in ("base", "refined", "adapted"):
            group_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
            for metric in METRICS:
                vals = [_extract_metric(row, group, metric) for row in thr_rows]
                group_stats[metric] = _collect_metric(vals)
            metric_stats[group] = group_stats

        has_oracle = any(
            isinstance(row.get("oracle_refined"), dict) or isinstance(row.get("oracle_adapted"), dict)
            for row in thr_rows
        )
        if has_oracle:
            for group in ("oracle_refined", "oracle_adapted"):
                group_stats = {}
                for metric in METRICS:
                    vals = [_extract_metric(row, group, metric) for row in thr_rows]
                    group_stats[metric] = _collect_metric(vals)
                metric_stats[group] = group_stats

        delta_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        delta_vs_refined_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for metric in METRICS:
            delta_vals = [_extract_block_metric(row, "delta_adapted", metric) for row in thr_rows]
            delta_ref_vals = [_extract_delta_vs_refined(row, metric) for row in thr_rows]
            delta_stats[metric] = _collect_metric(delta_vals)
            delta_vs_refined_stats[metric] = _collect_metric(delta_ref_vals)

        metric_stats["delta_adapted"] = delta_stats
        metric_stats["delta_vs_refined"] = delta_vs_refined_stats
        if has_oracle:
            delta_oracle_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
            for metric in METRICS:
                delta_oracle_vals = [_extract_block_metric(row, "delta_oracle_vs_refined", metric) for row in thr_rows]
                delta_oracle_stats[metric] = _collect_metric(delta_oracle_vals)
            metric_stats["delta_oracle_vs_refined"] = delta_oracle_stats
        out[thr] = metric_stats
    return out


def _print_table(agg: Dict[int, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]) -> None:
    headers = [
        "thr",
        "AJ_b",
        "AJ_a",
        "dAJ",
        "OA_b",
        "OA_a",
        "dOA",
        "<avg_b",
        "<avg_a",
        "d<avg",
        "err_b",
        "err_a",
        "derr",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for thr in sorted(agg.keys()):
        stats = agg[thr]
        row = [
            str(thr),
            _fmt(*stats["base"]["AJ"]),
            _fmt(*stats["adapted"]["AJ"]),
            _fmt(*stats["delta_adapted"]["AJ"], signed=True),
            _fmt(*stats["base"]["OA"]),
            _fmt(*stats["adapted"]["OA"]),
            _fmt(*stats["delta_adapted"]["OA"], signed=True),
            _fmt(*stats["base"]["<avg"]),
            _fmt(*stats["adapted"]["<avg"]),
            _fmt(*stats["delta_adapted"]["<avg"], signed=True),
            _fmt(*stats["base"]["avg_error_px"]),
            _fmt(*stats["adapted"]["avg_error_px"]),
            _fmt(*stats["delta_adapted"]["avg_error_px"], signed=True),
        ]
        print("| " + " | ".join(row) + " |")


def _print_diagnostic(agg: Dict[int, Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]) -> None:
    print()
    print("Diagnostic: adaptation effect decomposition")
    for thr in sorted(agg.keys()):
        stats = agg[thr]
        d_aj = stats["delta_adapted"]["AJ"][0]
        d_oa = stats["delta_adapted"]["OA"][0]
        d_avg = stats["delta_adapted"]["avg_error_px"][0]
        d_pos = stats["delta_adapted"]["<avg"][0]
        print(
            f"- thr={thr}: AJ={d_aj:+.4f} | OA={d_oa:+.4f} | <avg={d_pos:+.4f} | avg_error_px={d_avg:+.4f}"
        )

    if any("delta_oracle_vs_refined" in stats for stats in agg.values()):
        print()
        print("Diagnostic: oracle-visibility decomposition")
        for thr in sorted(agg.keys()):
            stats = agg[thr]
            oracle = stats.get("delta_oracle_vs_refined", {})
            d_aj = oracle.get("AJ", (None, None))[0]
            d_avg = oracle.get("avg_error_px", (None, None))[0]
            if d_aj is None:
                continue
            print(f"- thr={thr}: oracle_dAJ={d_aj:+.4f} | avg_error_px={d_avg:+.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize long-occlusion TTA results across seeds.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="List of TTA result JSON files produced by eval_long_occlusion_tta.py.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional path to write the markdown table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.runs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing result JSON: {p}")

    runs = [_load_run(p) for p in paths]
    agg = _aggregate(runs)

    print("Long-occlusion TTA summary")
    print("Runs:")
    for run in runs:
        print(f"- seed={run['seed']} path={run['path']}")
    print()
    _print_table(agg)
    _print_diagnostic(agg)

    if any("oracle_refined" in stats for stats in agg.values()):
        print()
        print("Oracle-visibility decomposition (GT visibility, same tracks)")
        headers = ["thr", "AJ_or_r", "AJ_or_a", "dAJ_or", "err_r", "err_a"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for thr in sorted(agg.keys()):
            stats = agg[thr]
            if "oracle_refined" not in stats or "oracle_adapted" not in stats:
                continue
            row = [
                str(thr),
                _fmt(*stats["oracle_refined"]["AJ"]),
                _fmt(*stats["oracle_adapted"]["AJ"]),
                _fmt(*stats["delta_oracle_vs_refined"]["AJ"], signed=True),
                _fmt(*stats["oracle_refined"]["avg_error_px"]),
                _fmt(*stats["oracle_adapted"]["avg_error_px"]),
            ]
            print("| " + " | ".join(row) + " |")

    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            lines: List[str] = []
            lines.append("Long-occlusion TTA summary")
            lines.append("")
            lines.append("Runs:")
            for run in runs:
                lines.append(f"- seed={run['seed']} path={run['path']}")
            lines.append("")
            headers = [
                "thr",
                "AJ_b",
                "AJ_a",
                "dAJ",
                "OA_b",
                "OA_a",
                "dOA",
                "<avg_b",
                "<avg_a",
                "d<avg",
                "err_b",
                "err_a",
                "derr",
            ]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for thr in sorted(agg.keys()):
                stats = agg[thr]
                row = [
                    str(thr),
                    _fmt(*stats["base"]["AJ"]),
                    _fmt(*stats["adapted"]["AJ"]),
                    _fmt(*stats["delta_adapted"]["AJ"], signed=True),
                    _fmt(*stats["base"]["OA"]),
                    _fmt(*stats["adapted"]["OA"]),
                    _fmt(*stats["delta_adapted"]["OA"], signed=True),
                    _fmt(*stats["base"]["<avg"]),
                    _fmt(*stats["adapted"]["<avg"]),
                    _fmt(*stats["delta_adapted"]["<avg"], signed=True),
                    _fmt(*stats["base"]["avg_error_px"]),
                    _fmt(*stats["adapted"]["avg_error_px"]),
                    _fmt(*stats["delta_adapted"]["avg_error_px"], signed=True),
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            lines.append("Diagnostic: adaptation effect decomposition")
            for thr in sorted(agg.keys()):
                stats = agg[thr]
                d_aj = stats["delta_adapted"]["AJ"][0]
                d_oa = stats["delta_adapted"]["OA"][0]
                d_avg = stats["delta_adapted"]["avg_error_px"][0]
                d_pos = stats["delta_adapted"]["<avg"][0]
                lines.append(
                    f"- thr={thr}: AJ={d_aj:+.4f} | OA={d_oa:+.4f} | <avg={d_pos:+.4f} | avg_error_px={d_avg:+.4f}"
                )
            if any("delta_oracle_vs_refined" in stats for stats in agg.values()):
                lines.append("")
                lines.append("Diagnostic: oracle-visibility decomposition")
                for thr in sorted(agg.keys()):
                    stats = agg[thr]
                    oracle = stats.get("delta_oracle_vs_refined", {})
                    d_aj = oracle.get("AJ", (None, None))[0]
                    d_avg = oracle.get("avg_error_px", (None, None))[0]
                    if d_aj is None:
                        continue
                    lines.append(f"- thr={thr}: oracle_dAJ={d_aj:+.4f} | avg_error_px={d_avg:+.4f}")
                lines.append("")
                lines.append("Oracle-visibility decomposition (GT visibility, same tracks)")
                lines.append("| thr | AJ_or_r | AJ_or_a | dAJ_or | err_r | err_a |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for thr in sorted(agg.keys()):
                    stats = agg[thr]
                    if "oracle_refined" not in stats or "oracle_adapted" not in stats:
                        continue
                    row = [
                        str(thr),
                        _fmt(*stats["oracle_refined"]["AJ"]),
                        _fmt(*stats["oracle_adapted"]["AJ"]),
                        _fmt(*stats["delta_oracle_vs_refined"]["AJ"], signed=True),
                        _fmt(*stats["oracle_refined"]["avg_error_px"]),
                        _fmt(*stats["oracle_adapted"]["avg_error_px"]),
                    ]
                    lines.append("| " + " | ".join(row) + " |")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
