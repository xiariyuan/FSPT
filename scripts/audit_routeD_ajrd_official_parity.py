#!/usr/bin/env python3
"""Compare Route-D AJ_RD against the hash-pinned official TAPNext++ code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_redetection_metrics import compute_official_aj_rd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_tapnextpp_aj_rd", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official AJ_RD source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same(a: float, b: float, tolerance: float = 0.0) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tolerance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-source", required=True)
    ap.add_argument("--official-commit", required=True)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17018)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.official_source).resolve()
    official = load_official(source)
    generator = torch.Generator().manual_seed(args.seed)
    maximum = 0.0
    mismatches = []
    scalar_keys = [
        *(f"AJ_RD_D{d}_dmin{m}" for m in (1, 4, 16, 64, 256) for d in (1, 2, 4, 8, 16)),
        *(f"AJ_RD_dmin{m}" for m in (1, 4, 16, 64, 256)),
        "AJ_RD",
    ]
    edge_cases = []
    for case in range(args.cases):
        batch = 1 + case % 3
        frames = 8 + case % 19
        points = 1 + case % 7
        gt_tracks = torch.rand(batch, frames, points, 2, generator=generator) * 256.0
        pred_tracks = gt_tracks + torch.randn(batch, frames, points, 2, generator=generator) * (case % 9)
        gt_visible = torch.rand(batch, frames, points, generator=generator) > (0.15 + 0.05 * (case % 5))
        gt_visible[:, 0] = True
        # Force deterministic events, including increasingly long record-breaking occlusions.
        if frames >= 8:
            gt_visible[:, 2:4, 0] = False
            gt_visible[:, 4, 0] = True
            if frames >= 14:
                gt_visible[:, 7:12, 0] = False
                gt_visible[:, 12, 0] = True
        pred_visible = torch.rand(batch, frames, points, generator=generator) > 0.25
        expected = official.compute_redetection_metrics(
            pred_tracks, pred_visible, gt_tracks, gt_visible
        )
        current = compute_official_aj_rd(
            pred_tracks, pred_visible, gt_tracks, gt_visible
        )
        for key in scalar_keys:
            a = float(expected[key])
            b = float(current[key])
            if not (math.isnan(a) or math.isnan(b)):
                maximum = max(maximum, abs(a - b))
            if not same(a, b):
                mismatches.append({"case": case, "key": key, "official": a, "routeD": b})
                if len(mismatches) >= 20:
                    break
        if mismatches:
            break
        edge_cases.append({
            "case": case,
            "shape": [batch, frames, points],
            "events": current["eligible_event_count"],
            "AJ_RD": current["AJ_RD"],
        })
    result = {
        "schema_version": "routeD_official_tapnextpp_ajrd_parity_v0",
        "official_repository": "https://github.com/google-deepmind/tapnet",
        "official_commit": args.official_commit,
        "official_source": str(source),
        "official_source_sha256": sha256(source),
        "cases": args.cases,
        "seed": args.seed,
        "maximum_absolute_scalar_difference": maximum,
        "mismatches": mismatches,
        "pass": not mismatches,
        "sample_cases": edge_cases[:10],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
