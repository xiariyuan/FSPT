#!/usr/bin/env python3
"""Create a hash-pinned protocol for the official CoTracker3 baseline gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

KEY_FILES = (
    "cotracker/evaluation/evaluate.py",
    "cotracker/evaluation/core/evaluator.py",
    "cotracker/evaluation/core/eval_utils.py",
    "cotracker/models/evaluation_predictor.py",
    "cotracker/models/core/cotracker/cotracker3_online.py",
    "cotracker/evaluation/configs/eval_tapvid_davis_first.yaml",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--source-archive", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--site-packages", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-checkpoint-sha256", default="205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218")
    args = ap.parse_args()

    src = Path(args.source_root).resolve()
    archive = Path(args.source_archive).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.dataset).resolve()
    site = Path(args.site_packages).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for path in (src, archive, checkpoint, dataset, site):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(f"official checkpoint SHA drift: {checkpoint_sha}")

    source_hashes = {}
    for rel in KEY_FILES:
        path = src / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[rel] = sha256(path)

    wheel_hash_file = site.parent / "wheel_sha256.txt"
    command = [
        sys.executable,
        str(src / "cotracker/evaluation/evaluate.py"),
        "--config-name", "eval_tapvid_davis_first",
        f"exp_dir={output}",
        f"dataset_root={dataset.parent.parent}",
        f"checkpoint={checkpoint}",
        "single_point=True",
        "offline_model=False",
        "window_len=16",
        "n_iters=6",
        "seed=0",
        "gpu_idx=0",
    ]
    payload = {
        "schema_version": "official_cotracker3_alignment_protocol_v0",
        "benchmark": "TAP-Vid-DAVIS first-query",
        "aggregation": "official evaluator video mean",
        "single_point": True,
        "model": "CoTracker3 scaled online",
        "official_source": {
            "repository": "https://github.com/facebookresearch/co-tracker",
            "commit": args.source_commit,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "key_file_sha256": source_hashes,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "isolated_site_packages": str(site),
            "wheel_sha256_file": str(wheel_hash_file) if wheel_hash_file.exists() else None,
            "wheel_sha256_file_sha256": sha256(wheel_hash_file) if wheel_hash_file.exists() else None,
        },
        "frozen_config": {
            "query_mode": "first",
            "single_point": True,
            "offline_model": False,
            "window_len": 16,
            "n_iters": 6,
            "seed": 0,
            "input_contract": "official TapVidDataset",
            "metric_contract": "official CoTracker evaluator using TAP-Vid formulas",
        },
        "command": command,
        "gate": {
            "required_before_redetection_training": True,
            "metric": "average_jaccard",
            "published_target_note": "Official repo exact-paper reproduction requires single_point=True.",
            "acceptance": "result must be in the published CoTracker3-online DAVIS-first range and all provenance hashes must match",
        },
    }
    protocol = output / "protocol.json"
    protocol.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"protocol": str(protocol), "protocol_sha256": sha256(protocol), "command": command}, indent=2))


if __name__ == "__main__":
    main()
