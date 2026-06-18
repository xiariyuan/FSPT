#!/usr/bin/env python3
"""Audit FB / flow consistency fields in pseudo-label JSONL.

This is a lightweight audit for the pseudo-label pipeline. It reports whether
FB / flow metrics are present, how many samples pass the configured thresholds,
and whether the available coverage is sufficient to trust the gate.

If the input labels only contain placeholder values (for example -1.0), the
script will surface that as unavailable rather than silently treating the
placeholders as valid measurements.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _metric_value(row: Dict[str, Any], key: str) -> float | None:
    value = row.get(key, None)
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if value_f < 0:
        return None
    return value_f


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {
            "n": 0,
            "mean_px": None,
            "median_px": None,
            "p95_px": None,
            "lt1px": None,
            "lt2px": None,
            "lt4px": None,
        }
    return {
        "n": int(arr.size),
        "mean_px": round(float(arr.mean()), 2),
        "median_px": round(float(np.median(arr)), 2),
        "p95_px": round(float(np.percentile(arr, 95)), 2),
        "lt1px": round(float(np.mean(arr <= 1.0)), 4),
        "lt2px": round(float(np.mean(arr <= 2.0)), 4),
        "lt4px": round(float(np.mean(arr <= 4.0)), 4),
    }


def audit_fb_consistency(
    labels: List[Dict[str, Any]],
    *,
    fb_threshold: float = 1.0,
    flow_threshold: float = 1.0,
) -> Dict[str, Any]:
    n_rows = len(labels)
    fb_vals = [float(_metric_value(row, "fb_error")) for row in labels if _metric_value(row, "fb_error") is not None]
    flow_vals = [float(_metric_value(row, "flow_consistency_error")) for row in labels if _metric_value(row, "flow_consistency_error") is not None]

    fb_available = len(fb_vals) > 0
    flow_available = len(flow_vals) > 0
    fb_pass_rate = float(np.mean(np.asarray(fb_vals) <= fb_threshold)) if fb_vals else None
    flow_pass_rate = float(np.mean(np.asarray(flow_vals) <= flow_threshold)) if flow_vals else None

    source_bucket = Counter(str(row.get("source_bucket", "unknown")) for row in labels)
    flags = Counter(flag for row in labels for flag in row.get("quality_flags", []))

    if n_rows == 0:
        verdict = "STOP"
        reason = "no labels found"
    elif not fb_available and not flow_available:
        verdict = "NO_DATA"
        reason = "fb / flow metrics are unavailable in the current labels"
    elif (fb_pass_rate is not None and fb_pass_rate >= 0.80) and ((flow_pass_rate is not None and flow_pass_rate >= 0.80) or not flow_available):
        verdict = "GO"
        reason = "fb / flow consistency pass rates meet the current threshold"
    else:
        verdict = "STOP"
        reason = "fb / flow consistency is below threshold or incomplete"

    return {
        "n_labels": n_rows,
        "fb_consistency": {
            "available": fb_available,
            "n": len(fb_vals),
            "pass_rate": round(fb_pass_rate, 4) if fb_pass_rate is not None else None,
            "threshold_px": fb_threshold,
            "stats": _stats(fb_vals),
        },
        "flow_consistency": {
            "available": flow_available,
            "n": len(flow_vals),
            "pass_rate": round(flow_pass_rate, 4) if flow_pass_rate is not None else None,
            "threshold_px": flow_threshold,
            "stats": _stats(flow_vals),
        },
        "source_bucket": dict(source_bucket),
        "quality_flags": dict(flags),
        "verdict": verdict,
        "reason": reason,
    }


def _render_md(result: Dict[str, Any]) -> str:
    lines = []
    lines.append("# FB / Flow Consistency Audit")
    lines.append("")
    lines.append(f"Verdict: `{result['verdict']}`")
    lines.append(f"Reason: {result['reason']}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Labels: `{result['n_labels']}`")
    lines.append(f"- FB available: `{result['fb_consistency']['available']}`")
    lines.append(f"- FB available n: `{result['fb_consistency']['n']}`")
    lines.append(f"- FB pass rate: `{result['fb_consistency']['pass_rate']}`")
    lines.append(f"- Flow available: `{result['flow_consistency']['available']}`")
    lines.append(f"- Flow available n: `{result['flow_consistency']['n']}`")
    lines.append(f"- Flow pass rate: `{result['flow_consistency']['pass_rate']}`")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    fb = result["fb_consistency"]["stats"]
    flow = result["flow_consistency"]["stats"]
    lines.append("| Metric | FB | Flow |")
    lines.append("|---|---:|---:|")
    lines.append(f"| n | {fb['n']} | {flow['n']} |")
    lines.append(f"| median px | {fb['median_px']} | {flow['median_px']} |")
    lines.append(f"| mean px | {fb['mean_px']} | {flow['mean_px']} |")
    lines.append(f"| p95 px | {fb['p95_px']} | {flow['p95_px']} |")
    lines.append(f"| <=1px | {fb['lt1px']} | {flow['lt1px']} |")
    lines.append(f"| <=2px | {fb['lt2px']} | {flow['lt2px']} |")
    lines.append(f"| <=4px | {fb['lt4px']} | {flow['lt4px']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FB / flow consistency fields.")
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--fb-threshold", type=float, default=1.0)
    parser.add_argument("--flow-threshold", type=float, default=1.0)
    args = parser.parse_args()

    labels = _load_jsonl(Path(args.labels_jsonl))
    result = audit_fb_consistency(labels, fb_threshold=args.fb_threshold, flow_threshold=args.flow_threshold)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(result))

    print(json.dumps({k: result[k] for k in ["verdict", "reason", "n_labels"]}, indent=2, ensure_ascii=True))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
