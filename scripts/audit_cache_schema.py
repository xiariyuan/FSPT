#!/usr/bin/env python3
"""Audit unified cache schema for protocol consistency.

Checks:
  - All canonical record fields exist
  - Normalized GT / query coordinates are sane
  - Predicted tracks stay finite and are only warned on when they extrapolate
  - Visibility is bool
  - Shapes are consistent within each record
  - Query points t in [0, T-1]
  - No NaN or Inf

Notes:
  - `model_input_size` is optional metadata. `strided+original` means query
    mode is strided and metrics are computed in original resolution; it does
    not require model input size to equal original size.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache, validate_attempt0_cache, write_json_report


def _is_pred_range_warning(msg: str) -> bool:
    return "pred_tracks appear non-normalized" in msg


def _rewrite_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade tolerated pred_track extrapolation from error to warning."""
    errors = []
    warnings = list(report.get("warnings", []))
    per_record = []

    for rec in report.get("per_record", []):
        rec_errors = []
        rec_warnings = list(rec.get("warnings", []))
        for err in rec.get("errors", []):
            if _is_pred_range_warning(err):
                rec_warnings.append(err)
            else:
                rec_errors.append(err)
        rec = dict(rec)
        rec["errors"] = rec_errors
        rec["warnings"] = rec_warnings
        rec["valid"] = len(rec_errors) == 0
        per_record.append(rec)
        errors.extend(rec_errors)
        warnings.extend(rec_warnings)

    report = dict(report)
    report["errors"] = errors
    report["warnings"] = warnings
    report["per_record"] = per_record
    report["valid"] = len(errors) == 0
    report["summary"] = dict(report.get("summary", {}))
    report["summary"]["num_pred_track_extrapolation_warnings"] = sum(
        1 for rec in per_record for msg in rec.get("warnings", []) if _is_pred_range_warning(msg)
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    payload = load_attempt0_cache(Path(args.cache_path))
    report = validate_attempt0_cache(
        payload,
        require_gt=True,
        normalized_coord_threshold=1.5,
        query_anchor_tol_px=2.0,
    )
    report = _rewrite_report(report)

    records = payload["records"]
    n_records = len(records)
    n_clean = sum(1 for rec in report.get("per_record", []) if rec.get("valid", False))

    result = {
        "cache_path": args.cache_path,
        "model_name": payload.get("model_name", "unknown"),
        "protocol": payload.get("protocol", "unknown"),
        "n_records": n_records,
        "n_clean": n_clean,
        "n_issues": len(report.get("errors", [])),
        "n_warnings": len(report.get("warnings", [])),
        "pass": len(report.get("errors", [])) == 0,
        "issues": report.get("errors", [])[:50],  # cap for readability
        "warnings": report.get("warnings", [])[:50],
        "report": report,
    }

    write_json_report(args.output_json, result)

    print(f"Schema audit: {n_clean}/{n_records} clean, {result['n_issues']} issues, {result['n_warnings']} warnings")
    if result["issues"]:
        for issue in result["issues"][:10]:
            print(f"  - {issue}")
    if result["warnings"]:
        for warning in result["warnings"][:10]:
            print(f"  ! {warning}")
    print(f"PASS: {result['pass']}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
