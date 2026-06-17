#!/usr/bin/env python3
"""
Summarize full TAP-Vid TTA results across multiple seeds.

The input is one or more JSON files produced by `scripts/eval_long_occlusion_tta.py`
after the full-metric patch. Each file should contain a top-level `full` block with
base/refined/adapted metrics and an oracle-visibility diagnostic.
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
    full = raw.get("full", None)
    if not isinstance(full, dict):
        raise ValueError(f"Missing top-level 'full' block in {path}")
    return {"path": path, "seed": seed, "full": full}


def _collect_metric(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _extract_metric(block: Dict[str, Any], metric: str) -> Optional[float]:
    return _safe_float(block.get(metric))


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    groups = ("base", "refined", "adapted", "oracle_refined", "oracle_adapted")
    out: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for group in groups:
        block_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for metric in METRICS:
            vals = [_extract_metric(run["full"].get(group, {}), metric) for run in runs]
            block_stats[metric] = _collect_metric(vals)
        out[group] = block_stats

    delta_blocks = ("delta_refined", "delta_adapted", "delta_vs_refined", "delta_oracle_vs_refined")
    for block_name in delta_blocks:
        block_stats = {}
        for metric in METRICS:
            vals = [_extract_metric(run["full"].get(block_name, {}), metric) for run in runs]
            block_stats[metric] = _collect_metric(vals)
        out[block_name] = block_stats

    num_videos = [_safe_float(run["full"].get("num_videos")) for run in runs]
    num_queries = [_safe_float(run["full"].get("num_queries")) for run in runs]
    out["num_videos"] = {"value": _collect_metric(num_videos)}  # type: ignore[assignment]
    out["num_queries"] = {"value": _collect_metric(num_queries)}  # type: ignore[assignment]
    return out


def _print_table(agg: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]) -> None:
    headers = [
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
    row = [
        _fmt(*agg["base"]["AJ"]),
        _fmt(*agg["adapted"]["AJ"]),
        _fmt(*agg["delta_adapted"]["AJ"], signed=True),
        _fmt(*agg["base"]["OA"]),
        _fmt(*agg["adapted"]["OA"]),
        _fmt(*agg["delta_adapted"]["OA"], signed=True),
        _fmt(*agg["base"]["<avg"]),
        _fmt(*agg["adapted"]["<avg"]),
        _fmt(*agg["delta_adapted"]["<avg"], signed=True),
        _fmt(*agg["base"]["avg_error_px"]),
        _fmt(*agg["adapted"]["avg_error_px"]),
        _fmt(*agg["delta_adapted"]["avg_error_px"], signed=True),
    ]
    print("| " + " | ".join(row) + " |")


def _print_diagnostic(agg: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]) -> None:
    print()
    print("Diagnostic: adaptation effect decomposition")
    print(
        f"- AJ={agg['delta_adapted']['AJ'][0]:+.4f} | "
        f"OA={agg['delta_adapted']['OA'][0]:+.4f} | "
        f"<avg={agg['delta_adapted']['<avg'][0]:+.4f} | "
        f"avg_error_px={agg['delta_adapted']['avg_error_px'][0]:+.4f}"
    )
    print()
    print("Oracle-visibility decomposition")
    print(
        f"- AJ_or={agg['delta_oracle_vs_refined']['AJ'][0]:+.4f} | "
        f"avg_error_px={agg['delta_oracle_vs_refined']['avg_error_px'][0]:+.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full TAP-Vid TTA results across seeds.")
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

    print("Full TAP-Vid TTA summary")
    print("Runs:")
    for run in runs:
        print(f"- seed={run['seed']} path={run['path']}")
    num_v, _ = agg["num_videos"]["value"]
    num_q, _ = agg["num_queries"]["value"]
    print(f"videos={num_v} queries={num_q}")
    print()
    _print_table(agg)
    _print_diagnostic(agg)

    if args.out_md:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("Full TAP-Vid TTA summary")
        lines.append("")
        lines.append("Runs:")
        for run in runs:
            lines.append(f"- seed={run['seed']} path={run['path']}")
        lines.append(f"- videos={num_v} queries={num_q}")
        lines.append("")
        headers = [
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
        row = [
            _fmt(*agg["base"]["AJ"]),
            _fmt(*agg["adapted"]["AJ"]),
            _fmt(*agg["delta_adapted"]["AJ"], signed=True),
            _fmt(*agg["base"]["OA"]),
            _fmt(*agg["adapted"]["OA"]),
            _fmt(*agg["delta_adapted"]["OA"], signed=True),
            _fmt(*agg["base"]["<avg"]),
            _fmt(*agg["adapted"]["<avg"]),
            _fmt(*agg["delta_adapted"]["<avg"], signed=True),
            _fmt(*agg["base"]["avg_error_px"]),
            _fmt(*agg["adapted"]["avg_error_px"]),
            _fmt(*agg["delta_adapted"]["avg_error_px"], signed=True),
        ]
        lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("Diagnostic: adaptation effect decomposition")
        lines.append(
            f"- AJ={agg['delta_adapted']['AJ'][0]:+.4f} | "
            f"OA={agg['delta_adapted']['OA'][0]:+.4f} | "
            f"<avg={agg['delta_adapted']['<avg'][0]:+.4f} | "
            f"avg_error_px={agg['delta_adapted']['avg_error_px'][0]:+.4f}"
        )
        lines.append("")
        lines.append("Oracle-visibility decomposition")
        lines.append(
            f"- AJ_or={agg['delta_oracle_vs_refined']['AJ'][0]:+.4f} | "
            f"avg_error_px={agg['delta_oracle_vs_refined']['avg_error_px'][0]:+.4f}"
        )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
