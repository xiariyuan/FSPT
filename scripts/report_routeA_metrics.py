#!/usr/bin/env python3
"""
Route A training report helper.

This script is intentionally lightweight and dependency-free so it can be run on
both local (Windows) and server (Linux) environments:

  python scripts/report_routeA_metrics.py --exp-dir outputs/<experiment>

It reads `epoch_metrics.jsonl` and prints compact tables for:
  - primary metrics (as logged by evaluate/train.py) with base + delta
  - auxiliary `*_{input,original}` metrics (if present)

The goal is to answer the common questions:
  - Did we beat the CoTracker3 base? (AJ_delta > 0)
  - Which epoch is best so far (by AJ_delta / AJ_original_delta)?
  - How far are we from "SOTA" (base tracker) for this run?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_cell(value: Optional[float], width: int = 9, precision: int = 6) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    formatted = f"{value:.{precision}f}"
    if len(formatted) >= width:
        return formatted[:width]
    return formatted.rjust(width)


def _extract_epoch_metrics(log_path: Path) -> Dict[int, Dict[str, Any]]:
    epochs: Dict[int, Dict[str, Any]] = {}
    for rec in _read_jsonl(log_path):
        try:
            epoch = int(rec.get("epoch", -1))
        except Exception:
            continue
        event = rec.get("event")
        if event not in ("train_epoch_end", "val_epoch_end"):
            continue
        row = epochs.setdefault(epoch, {"epoch": epoch})
        if event == "train_epoch_end":
            row["train_loss"] = rec.get("train_loss")
            row["lr"] = rec.get("lr")
            row["stage_idx"] = rec.get("stage_idx", row.get("stage_idx"))
        else:
            row["val_metrics"] = rec.get("metrics")
            row["stage_idx"] = rec.get("stage_idx", row.get("stage_idx"))
    return dict(sorted(epochs.items()))


def _best_epoch(epoch_rows: Dict[int, Dict[str, Any]], key: str, mode: str = "max") -> Optional[int]:
    best_epoch: Optional[int] = None
    best_value: Optional[float] = None
    for epoch, row in epoch_rows.items():
        metrics = row.get("val_metrics")
        if not isinstance(metrics, dict):
            continue
        value = _safe_float(metrics.get(key))
        if value is None:
            continue
        if best_value is None:
            best_epoch = epoch
            best_value = value
            continue
        if mode == "min":
            if value < best_value:
                best_epoch = epoch
                best_value = value
        else:
            if value > best_value:
                best_epoch = epoch
                best_value = value
    return best_epoch


def _print_table(
    epoch_rows: Dict[int, Dict[str, Any]],
    *,
    label: str,
    suffix: str,
    best_epoch: Optional[int],
    show_epochs: Optional[List[int]] = None,
) -> None:
    header = (
        f"{label:>9} |"
        f"{'epoch':>5} |"
        f"{'AJ_base':>9} {'AJ':>9} {'AJ_d':>9} |"
        f"{'<4px_b':>9} {'<4px':>9} {'<4px_d':>9} |"
        f"{'err_b':>9} {'err':>9} {'err_d':>9}"
    )
    print(header)
    print("-" * len(header))

    epochs_to_print = show_epochs if show_epochs is not None else list(epoch_rows.keys())
    for epoch in epochs_to_print:
        row = epoch_rows.get(epoch)
        if not row:
            continue
        metrics = row.get("val_metrics")
        if not isinstance(metrics, dict):
            continue

        def g(name: str) -> Optional[float]:
            return _safe_float(metrics.get(f"{name}{suffix}"))

        def gb(name: str) -> Optional[float]:
            return _safe_float(metrics.get(f"{name}{suffix}_base"))

        def gd(name: str) -> Optional[float]:
            return _safe_float(metrics.get(f"{name}{suffix}_delta"))

        mark = "*" if best_epoch is not None and epoch == best_epoch else " "
        print(
            f"{mark}{label:>8} |"
            f"{epoch:5d} |"
            f"{_format_cell(gb('AJ'))} {_format_cell(g('AJ'))} {_format_cell(gd('AJ'))} |"
            f"{_format_cell(gb('<4px'))} {_format_cell(g('<4px'))} {_format_cell(gd('<4px'))} |"
            f"{_format_cell(gb('avg_error_px'))} {_format_cell(g('avg_error_px'))} {_format_cell(gd('avg_error_px'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Route A base/refined/delta metrics from epoch_metrics.jsonl.")
    parser.add_argument("--exp-dir", type=str, required=True, help="outputs/<experiment_name> directory")
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only print the last N epochs (0 prints all).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    log_path = exp_dir / "epoch_metrics.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"epoch_metrics.jsonl not found: {log_path}")

    epoch_rows = _extract_epoch_metrics(log_path)
    if not epoch_rows:
        print("No epoch records found.")
        return

    epochs_list = list(epoch_rows.keys())
    if args.last and args.last > 0:
        epochs_list = epochs_list[-int(args.last) :]

    best_input = _best_epoch(epoch_rows, key="AJ_delta", mode="max")
    best_aux: Dict[str, Optional[int]] = {}
    for mode in ("input", "original"):
        best_aux[mode] = _best_epoch(epoch_rows, key=f"AJ_{mode}_delta", mode="max")

    print(f"Exp: {exp_dir}")
    if best_input is not None:
        print(f"Best (main):     epoch={best_input} by AJ_delta")
    for mode in ("input", "original"):
        if best_aux.get(mode) is not None:
            print(f"Best ({mode}):    epoch={best_aux[mode]} by AJ_{mode}_delta")
    print()

    _print_table(
        epoch_rows,
        label="main",
        suffix="",
        best_epoch=best_input,
        show_epochs=epochs_list,
    )
    for mode in ("input", "original"):
        suffix = f"_{mode}"
        if any(
            isinstance(r.get("val_metrics"), dict) and f"AJ{suffix}" in r.get("val_metrics", {})
            for r in epoch_rows.values()
        ):
            print()
            _print_table(
                epoch_rows,
                label=mode,
                suffix=suffix,
                best_epoch=best_aux.get(mode),
                show_epochs=epochs_list,
            )


if __name__ == "__main__":
    main()
