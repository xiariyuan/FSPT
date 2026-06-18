#!/usr/bin/env python3
"""Write run manifest JSON for canonical_redetection phase outputs.

Plan §1.2: each phase must output a manifest.json with:
  - git_commit, command_line, dataset_path, protocol
  - cache_schema_version, coord_format
  - teacher_name, feature_source, checkpoint_sha256
  - environment (python, device, cuda version)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def get_git_diff_hash() -> str:
    try:
        return subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()[:80]
    except Exception:
        return "unknown"


def get_environment() -> Dict[str, str]:
    env = {
        "python_version": sys.version.split()[0],
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }
    try:
        import torch
        env["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "none"
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def write_manifest(
    output_path: str,
    *,
    protocol: str = "strided+original",
    dataset_path: str = "",
    cache_schema_version: int = 1,
    coord_format: str = "yx_normalized_by_size_minus_1",
    teacher_name: str = "",
    feature_source: str = "",
    checkpoint_path: str = "",
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    manifest = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "git_commit": get_git_commit(),
        "git_diff": get_git_diff_hash(),
        "protocol": protocol,
        "dataset_path": dataset_path,
        "cache_schema_version": cache_schema_version,
        "coord_format": coord_format,
        "teacher_name": teacher_name,
        "feature_source": feature_source,
        "checkpoint_path": checkpoint_path,
        "environment": get_environment(),
    }
    if extra:
        # Put extra fields at top level for readability
        manifest.update(extra)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="strided+original")
    parser.add_argument("--dataset-path", type=str, default="/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
    parser.add_argument("--cache-schema-version", type=int, default=1)
    parser.add_argument("--coord-format", type=str, default="yx_normalized_by_size_minus_1")
    parser.add_argument("--teacher-name", type=str, default="")
    parser.add_argument("--feature-source", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--phase", type=str, default="")
    parser.add_argument("--extra", type=str, default="", help="JSON string of extra fields")
    args = parser.parse_args()

    extra = json.loads(args.extra) if args.extra else {}

    if args.phase:
        extra["phase"] = args.phase

    write_manifest(
        output_path=args.output_json,
        protocol=args.protocol,
        dataset_path=args.dataset_path,
        cache_schema_version=args.cache_schema_version,
        coord_format=args.coord_format,
        teacher_name=args.teacher_name,
        feature_source=args.feature_source,
        checkpoint_path=args.checkpoint_path,
        extra=extra,
    )


if __name__ == "__main__":
    main()
