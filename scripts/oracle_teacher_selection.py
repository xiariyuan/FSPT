#!/usr/bin/env python3
"""Oracle teacher selection: compute per-event oracle from multiple teachers.

Given multiple teacher caches, produces:
  - oracle_teacher event list (which teacher is best per event)
  - oracle gain vs fixed best teacher
  - teacher usage distribution
  - hard-tail examples where all teachers fail

This script answers: "is there headroom for multi-teacher distillation?"
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_teacher_reentry_audit import audit_teachers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-caches", type=str, nargs="+", required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    cache_map = {}
    for entry in args.teacher_caches:
        name, path = entry.split("=", 1)
        cache_map[name] = path

    results = audit_teachers(cache_map, args.max_videos)

    # Decision logic
    aj_rd_delta = results["aj_rd_comparison"]["delta"]
    if aj_rd_delta is None:
        decision = "STOP"
        reason = "AJ_RD delta unavailable; audit did not produce a valid comparison"
    elif aj_rd_delta >= 0.05:
        decision = "GO"
        reason = f"oracle teacher selection AJ_RD gain +{aj_rd_delta*100:.1f}pp >= 5pp"
    elif aj_rd_delta >= 0.02:
        decision = "WEAK_GO"
        reason = f"oracle gain +{aj_rd_delta*100:.1f}pp in [2pp, 5pp), proceed but don't over-promise"
    else:
        decision = "STOP"
        reason = f"oracle gain +{aj_rd_delta*100:.1f}pp < 2pp, insufficient headroom"

    results["decision"] = {
        "verdict": decision,
        "reason": reason,
        "aj_rd_delta": round(aj_rd_delta, 4) if aj_rd_delta is not None else None,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Decision: {decision} — {reason}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
