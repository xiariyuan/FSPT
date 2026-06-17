#!/usr/bin/env python3
"""
Run `evaluate.py` for a list of configs and produce a comparison table.

This is the missing "glue" for fast iteration:
  - generate explicit YAML variants (see `scripts/make_config_variants.py`)
  - run eval for each variant
  - aggregate `summary.json` into a paper-ready Markdown/LaTeX table

Example (server)
---------------
  cd /gemini/code/FSPT
  /root/miniconda3/bin/python scripts/run_eval_sweep.py \\
    --checkpoint outputs/<exp>/checkpoints/best_aj.pth \\
    --configs-dir outputs/sweeps/occ_corr_thr/configs \\
    --dataset davis \\
    --query-mode strided \\
    --metric-resolution-mode original \\
    --out-root outputs/eval_sweep_occCorr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_path(p: str) -> Path:
    return Path(p).expanduser()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval sweep runner.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    p.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Explicit config paths. If omitted, use --configs-dir.",
    )
    p.add_argument("--configs-dir", default=None, help="Directory containing YAML configs.")
    p.add_argument("--dataset", default="davis", choices=["davis", "kinetics", "all"], help="Dataset.")
    p.add_argument("--query-mode", default=None, choices=["first", "strided"], help="Query protocol override.")
    p.add_argument(
        "--metric-resolution-mode",
        default=None,
        choices=["original", "input"],
        help="Metric resolution mode override.",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="Output root dir. Default: outputs/eval_sweep_<timestamp>.",
    )
    p.add_argument("--compare-base", action="store_true", help="Also report base tracker metrics.")
    p.add_argument("--skip-existing", action="store_true", help="Skip configs with existing summary.json.")
    return p.parse_args()


def _collect_configs(args: argparse.Namespace) -> List[Path]:
    if args.configs:
        return [_as_path(p) for p in args.configs]
    if not args.configs_dir:
        raise SystemExit("Provide --configs or --configs-dir")
    cfg_dir = _as_path(args.configs_dir)
    if not cfg_dir.is_dir():
        raise SystemExit(f"--configs-dir is not a dir: {cfg_dir}")
    return sorted([p for p in cfg_dir.glob("*.yaml")])


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    configs = _collect_configs(args)
    if not configs:
        raise SystemExit("No configs found.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = _as_path(args.out_root) if args.out_root else PROJECT_ROOT / "outputs" / f"eval_sweep_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs: List[Path] = []
    for cfg_path in configs:
        run_name = cfg_path.stem
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[skip] {cfg_path} (summary.json exists)")
            run_dirs.append(out_dir)
            continue

        cmd = [
            str(PROJECT_ROOT / "evaluate.py"),
            "--checkpoint",
            str(_as_path(args.checkpoint)),
            "--config",
            str(cfg_path),
            "--dataset",
            args.dataset,
            "--output-dir",
            str(out_dir),
        ]
        if args.compare_base:
            cmd.append("--compare-base")
        if args.query_mode:
            cmd.extend(["--query-mode", args.query_mode])
        if args.metric_resolution_mode:
            cmd.extend(["--metric-resolution-mode", args.metric_resolution_mode])

        # Use the same python executable running this script.
        cmd = [sys.executable, *cmd]
        _run(cmd)
        run_dirs.append(out_dir)

    # Build comparison tables
    out_md = out_root / "longocc_comparison.md"
    out_tex = out_root / "longocc_comparison.tex"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "make_longocc_comparison_table.py"),
        "--runs",
        *[str(d) for d in run_dirs],
        "--out-md",
        str(out_md),
        "--out-tex",
        str(out_tex),
    ]
    _run(cmd)
    print(f"Done. Tables written to:\n  {out_md}\n  {out_tex}")


if __name__ == "__main__":
    main()
