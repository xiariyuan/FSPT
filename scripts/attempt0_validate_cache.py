#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache, validate_attempt0_cache, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an Attempt 0 unified prediction cache.")
    parser.add_argument("--cache", type=str, required=True, help="Path to a .pt/.pth/.json cache payload.")
    parser.add_argument("--require-gt", action="store_true", help="Require gt_tracks / gt_visibility in each record.")
    parser.add_argument(
        "--norm-tol",
        type=float,
        default=1.5,
        help="Max absolute value for normalized coordinates (default 1.5). "
             "Raise for predictors that can extrapolate beyond frame bounds (e.g., CoTracker3).",
    )
    parser.add_argument(
        "--query-anchor-tol-px",
        type=float,
        default=2.0,
        help="Max query-anchor error in pixels for warnings (default 2.0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON report path. Defaults to <cache>.validation.json",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    payload = load_attempt0_cache(cache_path)
    report = validate_attempt0_cache(
        payload,
        require_gt=args.require_gt,
        normalized_coord_threshold=args.norm_tol,
        query_anchor_tol_px=args.query_anchor_tol_px,
    )

    out_path = Path(args.out) if args.out else cache_path.with_suffix(cache_path.suffix + ".validation.json")
    write_json_report(out_path, report)

    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    if report["warnings"]:
        print(f"[warn] {len(report['warnings'])} warnings written to {out_path}")
    if not report["valid"]:
        print(f"[error] cache validation failed; see {out_path}")
        raise SystemExit(1)
    print(f"[ok] cache validation passed: {out_path}")


if __name__ == "__main__":
    main()

