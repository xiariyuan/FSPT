#!/usr/bin/env python3
"""Audit unified cache schema for protocol consistency.

Checks:
  - All records have required fields
  - Coordinate ranges are in [0,1] for normalized fields
  - Visibility is bool
  - original_size == model_input_size (strided+original invariant)
  - Shapes are consistent within each record
  - Query points t in [0, T-1]
  - No NaN or Inf
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import assert_coord_range
from utils.attempt0_schema import load_attempt0_cache


REQUIRED_FIELDS = [
    "video_id", "sequence_index", "frame_count",
    "query_points", "pred_tracks", "pred_visibility",
    "gt_tracks", "gt_visibility",
    "original_size", "model_input_size",
    "adapter_version", "raw_coordinate_note",
]


def audit_record(r: dict, idx: int) -> List[str]:
    """Audit a single cache record. Returns list of issue strings."""
    issues = []
    vid = r.get("video_id", f"record_{idx}")
    tag = f"[{vid}]"

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in r:
            issues.append(f"{tag} missing field: {field}")

    if issues:
        return issues  # can't check further without basic fields

    T = int(r["frame_count"])
    orig_h, orig_w = int(r["original_size"][0]), int(r["original_size"][1])
    model_h, model_w = int(r["model_input_size"][0]), int(r["model_input_size"][1])

    # Protocol check: model_input_size should equal original_size for strided+original
    if orig_h != model_h or orig_w != model_w:
        issues.append(
            f"{tag} model_input_size [{model_h},{model_w}] != original_size [{orig_h},{orig_w}]"
        )

    # Shape checks
    qp = np.asarray(r["query_points"], dtype=np.float32)
    pt = np.asarray(r["pred_tracks"], dtype=np.float32)
    pv = np.asarray(r["pred_visibility"])
    gt_t = np.asarray(r["gt_tracks"], dtype=np.float32)
    gt_v = np.asarray(r["gt_visibility"])

    if qp.ndim != 2 or qp.shape[1] < 3:
        issues.append(f"{tag} query_points shape {qp.shape}, expected (N, 3+)")
    N_q = qp.shape[0]

    if pt.ndim != 3 or pt.shape[0] != N_q or pt.shape[1] != T or pt.shape[2] != 2:
        issues.append(f"{tag} pred_tracks shape {pt.shape}, expected ({N_q}, {T}, 2)")

    if pv.ndim != 2 or pv.shape[0] != N_q or pv.shape[1] != T:
        issues.append(f"{tag} pred_visibility shape {pv.shape}, expected ({N_q}, {T})")

    if gt_t.ndim != 3 or gt_t.shape != (N_q, T, 2):
        issues.append(f"{tag} gt_tracks shape {gt_t.shape}, expected ({N_q}, {T}, 2)")

    if gt_v.ndim != 2 or gt_v.shape != (N_q, T):
        issues.append(f"{tag} gt_visibility shape {gt_v.shape}, expected ({N_q}, {T})")

    # Coordinate range checks (normalized [0,1])
    try:
        assert_coord_range(pt, 0.0, 1.0, f"{tag} pred_tracks")
        assert_coord_range(gt_t, 0.0, 1.0, f"{tag} gt_tracks")
        assert_coord_range(qp[:, 1:], 0.0, 1.0, f"{tag} query_points xy")
    except ValueError as e:
        issues.append(str(e))

    # Query t range
    if qp[:, 0].min() < 0 or qp[:, 0].max() > T - 1:
        issues.append(
            f"{tag} query_t range [{qp[:, 0].min():.0f}, {qp[:, 0].max():.0f}] vs T={T}"
        )

    # Visibility dtype
    if pv.dtype != bool:
        issues.append(f"{tag} pred_visibility dtype {pv.dtype} != bool")
    if gt_v.dtype != bool:
        issues.append(f"{tag} gt_visibility dtype {gt_v.dtype} != bool")

    # NaN / Inf checks
    for name, arr in [("pred_tracks", pt), ("gt_tracks", gt_t), ("query_points", qp)]:
        if np.isnan(arr).any():
            issues.append(f"{tag} {name} contains NaN")
        if np.isinf(arr).any():
            issues.append(f"{tag} {name} contains Inf")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    payload = load_attempt0_cache(Path(args.cache_path))
    records = payload["records"]

    all_issues = []
    for idx, r in enumerate(records):
        record_issues = audit_record(r, idx)
        all_issues.extend(record_issues)

    n_records = len(records)
    n_clean = sum(1 for i in range(n_records) if not any(
        f"[{records[i].get('video_id', f'record_{i}')}]" in issue
        for issue in all_issues
    ))

    result = {
        "cache_path": args.cache_path,
        "model_name": payload.get("model_name", "unknown"),
        "protocol": payload.get("protocol", "unknown"),
        "n_records": n_records,
        "n_clean": n_clean,
        "n_issues": len(all_issues),
        "pass": len(all_issues) == 0,
        "issues": all_issues[:50],  # cap for readability
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Schema audit: {n_clean}/{n_records} clean, {len(all_issues)} issues")
    if all_issues:
        for issue in all_issues[:10]:
            print(f"  - {issue}")
    print(f"PASS: {result['pass']}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
