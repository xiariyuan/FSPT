#!/usr/bin/env python3
"""Compatibility launcher for a hash-pinned official CoTracker evaluation.

This file replaces only Hydra CLI composition, which is incompatible with the
current Python 3.11 environment. It imports and calls the unmodified official
``run_eval`` function and official ``DefaultConfig`` dataclass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu-idx", type=int, default=0)
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if not source_root.is_dir() or not dataset_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("source, dataset, or checkpoint is missing")
    sys.path.insert(0, str(source_root))

    from cotracker.evaluation.evaluate import DefaultConfig, run_eval

    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = DefaultConfig(
        exp_dir=str(exp_dir),
        dataset_name="tapvid_davis_first",
        dataset_root=str(dataset_root),
        checkpoint=str(checkpoint),
        grid_size=5,
        local_grid_size=8,
        num_uniformly_sampled_pts=0,
        sift_size=0,
        single_point=True,
        offline_model=False,
        window_len=16,
        n_iters=6,
        seed=0,
        gpu_idx=int(args.gpu_idx),
        local_extent=50,
        v2=False,
    )
    manifest = {
        "schema_version": "official_cotracker3_python311_compat_launcher_v0",
        "scope": "Hydra CLI replacement only; official run_eval, model, dataset and evaluator are unmodified",
        "source_root": str(source_root),
        "official_evaluate_sha256": sha256(source_root / "cotracker/evaluation/evaluate.py"),
        "official_evaluator_sha256": sha256(source_root / "cotracker/evaluation/core/evaluator.py"),
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "frozen_config": dict(cfg.__dict__),
        "cuda_visible_devices_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    manifest_path = exp_dir / "compatibility_launcher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run_eval(cfg)


if __name__ == "__main__":
    main()
