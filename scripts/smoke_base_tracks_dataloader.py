#!/usr/bin/env python3
"""Smoke-test base_tracks injection from a train.py config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import load_config
from datasets import get_dataloader


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that base_tracks_dir is injected into dataloader batches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Reuse train.py's dataloader construction so this smoke matches the real training entrypoint.
    # The previous hand-rolled path passed DAVIS-only unsupported keys such as num_frames.
    from train import create_dataloaders

    train_loader, val_loader = create_dataloaders(cfg, debug=False, distributed=False, rank=0, world_size=1)
    loader = train_loader if args.split == "train" else val_loader
    if loader is None:
        raise RuntimeError(f"{args.split} loader is None")

    seen = 0
    summaries = []
    for batch in loader:
        seen += 1
        if "base_tracks" not in batch or "base_visibility" not in batch:
            raise RuntimeError("batch is missing base_tracks/base_visibility")
        bt = batch["base_tracks"]
        bv = batch["base_visibility"]
        qp = batch.get("query_points")
        if not isinstance(bt, torch.Tensor) or bt.ndim != 4 or bt.shape[-1] != 2:
            raise RuntimeError(f"bad base_tracks shape: {type(bt)} {getattr(bt, 'shape', None)}")
        if not isinstance(bv, torch.Tensor) or bv.shape != bt.shape[:3]:
            raise RuntimeError(f"bad base_visibility shape: {type(bv)} {getattr(bv, 'shape', None)} vs {bt.shape[:3]}")
        if not torch.isfinite(bt).all():
            raise RuntimeError("base_tracks contains non-finite values")
        if isinstance(qp, torch.Tensor) and qp.shape[0] != bt.shape[0]:
            raise RuntimeError("query_points batch dimension mismatch")
        summaries.append({
            "batch": seen,
            "video_name": batch.get("video_name"),
            "base_tracks_shape": list(bt.shape),
            "base_visibility_shape": list(bv.shape),
            "base_visible_frac": round(float(bv.float().mean().item()), 6),
            "base_tracks_min": round(float(bt.min().item()), 6),
            "base_tracks_max": round(float(bt.max().item()), 6),
        })
        if seen >= args.max_batches:
            break

    result = {"ok": True, "config": args.config, "split": args.split, "batches_checked": seen, "batches": summaries}
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
