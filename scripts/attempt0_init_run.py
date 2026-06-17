#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


MODELS = [
    "cotracker3_online",
    "cotracker3_offline",
    "trackon2",
    "trackonr",
    "tapnextpp",
    "alltracker",
]

DATASETS = ["davis", "kinetics"]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Attempt 0 output tree.")
    parser.add_argument("--out-root", type=str, default="outputs/attempt0")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"attempt0_{ts}"
    root = Path(args.out_root).expanduser().resolve() / run_name

    dirs = [
        "manifests",
        "reports",
        "decisions",
        "rescoring",
        "status",
        "logs",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for dataset in DATASETS:
            (root / "caches" / model / dataset).mkdir(parents=True, exist_ok=True)

    run_info = {
        "run_name": run_name,
        "created_at": ts,
        "models": MODELS,
        "datasets": DATASETS,
        "notes": "Attempt 0 unified reproduction workspace.",
    }
    _write_json(root / "RUN_INFO.json", run_info)

    template = {
        "model_name": None,
        "repo_url": None,
        "repo_commit": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "dataset_name": None,
        "split": None,
        "run_date": None,
        "repo_native_protocol": None,
        "repo_native_metric_names": None,
        "repo_native_numbers": None,
        "official_reference_numbers": None,
        "delta_vs_official": None,
        "reproduction_status": "pending",
        "raw_coordinate_format": None,
        "raw_visibility_format": None,
        "query_format": None,
        "adapter_version": None,
        "adapter_sanity_status": "pending",
        "adapter_notes": None,
        "query_mode": None,
        "metric_resolution_mode": None,
        "AJ": None,
        "OA": None,
        "<avg": None,
        "<4px": None,
        "delta_vs_cotracker3_baseline": None,
        "rescoring_status": "pending",
        "long_occ_definition": None,
        "long_occ_AJ": None,
        "long_occ_<avg": None,
        "long_occ_<4px": None,
        "reentry_first_frame_error": None,
        "AJ_RD": None,
        "gpu_type": None,
        "batch_size": None,
        "peak_memory_gb": None,
        "eval_wall_time": None,
        "notes": None,
        "keep_for_main_ranking": None,
        "keep_as_teacher_candidate": None,
        "known_blockers": None,
        "next_action": None,
    }
    for model in MODELS:
        for dataset in DATASETS:
            _write_json(root / "status" / f"{model}__{dataset}.json", {**template, "model_name": model, "dataset_name": dataset})

    print(root)


if __name__ == "__main__":
    main()
