#!/usr/bin/env python3
"""Filter pseudo-label JSONL by simple quality rules.

This script is intentionally conservative:
  - keeps labels that satisfy optional quality flag requirements
  - filters on FB / flow consistency only when those fields are present
  - can optionally enforce a minimum teacher confidence
  - can optionally drop GT-only debug fields before writing output
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _passes(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    flags = set(row.get("quality_flags", []))
    required_flags = set(args.require_flag)
    if required_flags and not required_flags.issubset(flags):
        return False

    if args.min_teacher_conf is not None:
        conf = row.get("teacher_conf", None)
        if conf is None or float(conf) < args.min_teacher_conf:
            return False

    if args.max_fb_error is not None:
        fb = float(row.get("fb_error", -1.0))
        if fb < 0 or fb > args.max_fb_error:
            return False

    if args.max_flow_error is not None:
        flow = float(row.get("flow_consistency_error", -1.0))
        if flow < 0 or flow > args.max_flow_error:
            return False

    if args.min_occ_run_len is not None:
        if int(row.get("occ_run_len", 0)) < args.min_occ_run_len:
            return False

    if args.max_ct_offline_error is not None:
        ct = row.get("ct_offline_error_px", None)
        if ct is None or float(ct) > args.max_ct_offline_error:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pseudo-label JSONL by quality rules.")
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--require-flag", type=str, action="append", default=[])
    parser.add_argument("--min-teacher-conf", type=float, default=None)
    parser.add_argument("--max-fb-error", type=float, default=None)
    parser.add_argument("--max-flow-error", type=float, default=None)
    parser.add_argument("--min-occ-run-len", type=int, default=None)
    parser.add_argument("--max-ct-offline-error", type=float, default=None)
    parser.add_argument("--drop-debug-fields", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.labels_jsonl))
    kept = [row for row in rows if _passes(row, args)]

    if args.drop_debug_fields:
        drop_keys = {"gt_xy_px", "pseudo_label_error_px", "ct_offline_error_px"}
        kept = [{k: v for k, v in row.items() if k not in drop_keys} for row in kept]

    kept_counts = Counter(str(row.get("source_bucket", "unknown")) for row in kept)
    dropped = len(rows) - len(kept)
    summary = {
        "input_n": len(rows),
        "output_n": len(kept),
        "dropped_n": dropped,
        "keep_rate": round(len(kept) / max(len(rows), 1), 4),
        "kept_by_source_bucket": dict(kept_counts),
        "filters": {
            "require_flag": args.require_flag,
            "min_teacher_conf": args.min_teacher_conf,
            "max_fb_error": args.max_fb_error,
            "max_flow_error": args.max_flow_error,
            "min_occ_run_len": args.min_occ_run_len,
            "max_ct_offline_error": args.max_ct_offline_error,
            "drop_debug_fields": args.drop_debug_fields,
        },
    }

    out_jsonl = Path(args.output_jsonl)
    out_json = Path(args.output_json)
    _write_jsonl(out_jsonl, kept)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
