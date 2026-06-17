#!/usr/bin/env python3
"""
Create a compact (paper-ready) comparison table for the long-occlusion story.

We usually have multiple evaluation folders produced by `evaluate.py`, each with
`summary.json` containing refined/base/delta metrics. This script reads several
`summary.json` files and writes:
  - a Markdown table
  - a LaTeX (booktabs) table

Example (server):
  cd /gemini/code/FSPT
  /root/miniconda3/bin/python scripts/make_longocc_comparison_table.py \\
    --runs \\
      outputs/eval_stage2_residual_longocc_m20_best \\
      outputs/eval_stage3_relocal_m20_v1_best \\
      outputs/eval_stage3_relocal_m20_v2_best \\
    --out-md outputs/longocc_comparison.md \\
    --out-tex outputs/longocc_comparison.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_path(value: str) -> Path:
    p = Path(value)
    if p.is_dir():
        p = p / "summary.json"
    return p


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(stats: Any) -> Optional[float]:
    if stats is None:
        return None
    if isinstance(stats, dict) and "mean" in stats:
        return _safe_float(stats.get("mean"))
    return _safe_float(stats)


def _load_run(summary_path: Path) -> Dict[str, Any]:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    results = raw.get("results", {})

    ref = results.get("davis", {}) or {}
    base = results.get("davis_base", {}) or {}
    delta = results.get("davis_delta", {}) or {}

    def g(d: Dict[str, Any], key: str) -> Optional[float]:
        return _mean(d.get(key))

    return {
        "name": summary_path.parent.name,
        "path": str(summary_path),
        "AJ_base": g(base, "AJ"),
        "AJ_ref": g(ref, "AJ"),
        "AJ_delta": g(delta, "AJ"),
        "AJ20_base": g(base, "AJ_longocc20"),
        "AJ20_ref": g(ref, "AJ_longocc20"),
        "AJ20_delta": g(delta, "AJ_longocc20"),
        "AJ30_base": g(base, "AJ_longocc30"),
        "AJ30_ref": g(ref, "AJ_longocc30"),
        "AJ30_delta": g(delta, "AJ_longocc30"),
        "nq20": g(ref, "num_queries_longocc20"),
        "nv20": g(ref, "num_videos_longocc20"),
        "nq30": g(ref, "num_queries_longocc30"),
        "nv30": g(ref, "num_videos_longocc30"),
    }


def _fmt(value: Optional[float], precision: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed:
        return f"{value:+.{precision}f}"
    return f"{value:.{precision}f}"


def _write_md(runs: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "run",
        "AJ_base",
        "AJ_ref",
        "AJ_Δ",
        "AJ20_base",
        "AJ20_ref",
        "AJ20_Δ",
        "AJ30_base",
        "AJ30_ref",
        "AJ30_Δ",
        "Nq20",
        "Nv20",
        "Nq30",
        "Nv30",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["name"]),
                    _fmt(r["AJ_base"]),
                    _fmt(r["AJ_ref"]),
                    _fmt(r["AJ_delta"], signed=True),
                    _fmt(r["AJ20_base"]),
                    _fmt(r["AJ20_ref"]),
                    _fmt(r["AJ20_delta"], signed=True),
                    _fmt(r["AJ30_base"]),
                    _fmt(r["AJ30_ref"]),
                    _fmt(r["AJ30_delta"], signed=True),
                    _fmt(r["nq20"], precision=0),
                    _fmt(r["nv20"], precision=0),
                    _fmt(r["nq30"], precision=0),
                    _fmt(r["nv30"], precision=0),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex(runs: List[Dict[str, Any]], path: Path) -> None:
    lines = []
    lines.append("% Requires: \\usepackage{booktabs}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Run & $AJ_{20}^{base}$ & $AJ_{20}$ & $\\Delta AJ_{20}$ & $AJ_{30}^{base}$ & $AJ_{30}$ & $\\Delta AJ_{30}$ \\\\")
    lines.append("\\midrule")
    for r in runs:
        lines.append(
            f"{r['name']} & "
            f"{_fmt(r['AJ20_base'])} & {_fmt(r['AJ20_ref'])} & {_fmt(r['AJ20_delta'], signed=True)} & "
            f"{_fmt(r['AJ30_base'])} & {_fmt(r['AJ30_ref'])} & {_fmt(r['AJ30_delta'], signed=True)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make long-occlusion comparison table from evaluate.py summaries.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="List of evaluation output dirs OR summary.json paths.",
    )
    parser.add_argument("--out-md", type=str, default=None, help="Optional output Markdown path.")
    parser.add_argument("--out-tex", type=str, default=None, help="Optional output LaTeX path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_paths = [_as_path(p) for p in args.runs]
    for p in summary_paths:
        if not p.exists():
            raise FileNotFoundError(f"summary.json not found: {p}")

    runs = [_load_run(p) for p in summary_paths]

    # Print a compact summary to stdout.
    print("Long-occlusion comparison (DAVIS):")
    for r in runs:
        print(
            f"- {r['name']}: AJ20_delta={_fmt(r['AJ20_delta'], signed=True)}, "
            f"AJ30_delta={_fmt(r['AJ30_delta'], signed=True)}, AJ_delta={_fmt(r['AJ_delta'], signed=True)}"
        )

    if args.out_md:
        _write_md(runs, Path(args.out_md))
    if args.out_tex:
        _write_tex(runs, Path(args.out_tex))


if __name__ == "__main__":
    main()

