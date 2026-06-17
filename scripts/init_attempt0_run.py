#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import build_attempt0_status_skeleton, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize an Attempt 0 output workspace.")
    parser.add_argument("--name", type=str, default="", help="Run name. Default: attempt0_YYYYmmdd_HHMMSS")
    parser.add_argument(
        "--out-root",
        type=str,
        default="outputs",
        help="Parent output directory. Default: outputs",
    )
    args = parser.parse_args()

    run_name = args.name.strip() or f"attempt0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.out_root) / run_name
    subdirs = [
        "manifests",
        "prediction_caches",
        "repo_native",
        "unified_rescoring",
        "reports",
        "status",
        "logs",
    ]
    for subdir in subdirs:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "main_protocol": {
            "datasets": ["tapvid_davis", "tapvid_kinetics"],
            "query_mode": "strided",
            "metric_resolution_mode": "original",
            "metrics": ["AJ", "OA", "<avg", "<4px"],
        },
        "aux_tables": ["long-occ>20", "re-entry first-frame", "AJ_RD-if-available"],
        "models": [
            "cotracker3_baseline",
            "cotracker3_offline",
            "trackon2",
            "trackonr",
            "tapnextpp",
            "alltracker",
        ],
    }
    write_json_report(root / "manifests" / "attempt0_manifest.json", manifest)

    for model_name in manifest["models"]:
        status = build_attempt0_status_skeleton(model_name)
        write_json_report(root / "status" / f"{model_name}.json", status)

    readme = [
        f"Attempt 0 workspace: {run_name}",
        "",
        "Subdirectories:",
        "- manifests",
        "- prediction_caches",
        "- repo_native",
        "- unified_rescoring",
        "- reports",
        "- status",
        "- logs",
        "",
        "Status files were initialized for:",
        *[f"- {m}" for m in manifest["models"]],
    ]
    with open(root / "README.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(readme) + "\n")

    print(json.dumps({"run_root": str(root), "models": manifest["models"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
