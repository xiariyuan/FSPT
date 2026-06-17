#!/usr/bin/env python3
"""
Summarize training logs from outputs/<exp>/epoch_metrics.jsonl.

This is intended for paper-quality ablations where you want:
  - Best epoch overall for a metric (e.g., AJ)
  - Best epoch per refiner stage (Route A freeze/unfreeze schedule)
  - A CSV/Markdown table for quick copy/paste

Example:
  python scripts/summarize_epoch_metrics.py --exp-dir outputs/fspt_cotracker_refine \\
    --metric AJ --mode max --save-csv stage_best.csv --save-md stage_best.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize epoch_metrics.jsonl")
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment output directory (contains epoch_metrics.jsonl)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional explicit path to epoch_metrics.jsonl (overrides --exp-dir).",
    )
    parser.add_argument("--metric", type=str, default="AJ", help="Metric key to optimize (default: AJ)")
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"], help="Optimize direction")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional output CSV path")
    parser.add_argument("--save-md", type=str, default=None, help="Optional output Markdown path")
    return parser.parse_args()


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_epoch_records(log_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Merge train_epoch_end and val_epoch_end records into per-epoch dicts.
    """
    epochs: Dict[int, Dict[str, Any]] = {}
    for rec in _read_jsonl(log_path):
        if not isinstance(rec, dict):
            continue
        epoch = rec.get("epoch", None)
        try:
            epoch_i = int(epoch)
        except Exception:
            continue
        e = epochs.setdefault(epoch_i, {"epoch": epoch_i})
        event = rec.get("event", None)
        if event == "train_epoch_end":
            e["train_loss"] = rec.get("train_loss", None)
            e["lr"] = rec.get("lr", None)
            e["lrs"] = rec.get("lrs", None)
            e["stage_idx"] = rec.get("stage_idx", e.get("stage_idx", None))
            e["data"] = rec.get("data", None)
        elif event == "val_epoch_end":
            e["val_metrics"] = rec.get("metrics", None)
            e["stage_idx"] = rec.get("stage_idx", e.get("stage_idx", None))
        else:
            # ignore other events
            continue
    return epochs


def _best_by_metric(
    epoch_rows: Dict[int, Dict[str, Any]],
    metric: str,
    mode: str,
    stage_idx: Optional[int] = None,
) -> Optional[Tuple[int, float, Dict[str, Any]]]:
    """
    Return (best_epoch, best_value, row) or None.
    """
    best_epoch = None
    best_value = None
    best_row = None
    for epoch, row in sorted(epoch_rows.items()):
        if stage_idx is not None:
            try:
                if int(row.get("stage_idx", -1)) != int(stage_idx):
                    continue
            except Exception:
                continue
        metrics = row.get("val_metrics", None)
        if not isinstance(metrics, dict):
            continue
        v = _safe_float(metrics.get(metric, None))
        if v is None:
            continue
        if best_value is None:
            best_epoch, best_value, best_row = epoch, v, row
            continue
        if mode == "max":
            if v > best_value:
                best_epoch, best_value, best_row = epoch, v, row
        else:
            if v < best_value:
                best_epoch, best_value, best_row = epoch, v, row
    if best_epoch is None or best_value is None or best_row is None:
        return None
    return best_epoch, float(best_value), best_row


def _collect_metric_keys(epoch_rows: Dict[int, Dict[str, Any]]) -> list:
    keys = set()
    for row in epoch_rows.values():
        metrics = row.get("val_metrics", None)
        if isinstance(metrics, dict):
            keys |= set(metrics.keys())
    # Keep main ones first if present.
    preferred = ["AJ", "<4px", "OA"]
    ordered = [k for k in preferred if k in keys]
    ordered += sorted([k for k in keys if k not in set(ordered)])
    return ordered


def _write_csv(path: Path, rows: List[Dict[str, Any]], metric_keys: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage_idx", "epoch", "train_loss", "lr", *metric_keys])
        for r in rows:
            metrics = r.get("val_metrics", {}) if isinstance(r.get("val_metrics", None), dict) else {}
            writer.writerow(
                [
                    r.get("stage_idx", -1),
                    r.get("epoch", ""),
                    r.get("train_loss", ""),
                    r.get("lr", ""),
                    *[metrics.get(k, "") for k in metric_keys],
                ]
            )


def _write_md(path: Path, rows: List[Dict[str, Any]], metric_keys: List[str]) -> None:
    headers = ["stage_idx", "epoch", "train_loss", "lr", *metric_keys]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        metrics = r.get("val_metrics", {}) if isinstance(r.get("val_metrics", None), dict) else {}
        values = [
            str(r.get("stage_idx", -1)),
            str(r.get("epoch", "")),
            str(r.get("train_loss", "")),
            str(r.get("lr", "")),
        ] + [str(metrics.get(k, "")) for k in metric_keys]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    log_path = Path(args.log) if args.log is not None else (exp_dir / "epoch_metrics.jsonl")
    if not log_path.is_absolute():
        log_path = (Path.cwd() / log_path).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"epoch_metrics.jsonl not found: {log_path}")

    epoch_rows = _extract_epoch_records(log_path)
    if not epoch_rows:
        print(f"No valid epoch records found in {log_path}")
        return

    metric = str(args.metric)
    mode = str(args.mode).lower()

    best_overall = _best_by_metric(epoch_rows, metric=metric, mode=mode, stage_idx=None)
    if best_overall is not None:
        best_epoch, best_value, row = best_overall
        stage_idx = row.get("stage_idx", -1)
        print(f"Best overall ({metric}, {mode}): epoch={best_epoch}, value={best_value:.6f}, stage={stage_idx}")
    else:
        print(f"No val metrics found for metric={metric!r}.")

    # Best per stage
    stages = sorted({int(r.get("stage_idx", -1)) for r in epoch_rows.values() if r.get("stage_idx", None) is not None})
    stage_best_rows = []
    for s in stages:
        best = _best_by_metric(epoch_rows, metric=metric, mode=mode, stage_idx=s)
        if best is None:
            continue
        best_epoch, best_value, row = best
        stage_best_rows.append(row)
        print(f"  stage={s:>3}: best_epoch={best_epoch}, {metric}={best_value:.6f}")

    metric_keys = _collect_metric_keys(epoch_rows)

    if args.save_csv:
        out = Path(args.save_csv)
        if not out.is_absolute():
            out = (exp_dir / out).resolve()
        _write_csv(out, stage_best_rows, metric_keys=metric_keys)
        print(f"Saved CSV: {out}")
    if args.save_md:
        out = Path(args.save_md)
        if not out.is_absolute():
            out = (exp_dir / out).resolve()
        _write_md(out, stage_best_rows, metric_keys=metric_keys)
        print(f"Saved Markdown: {out}")


if __name__ == "__main__":
    main()
