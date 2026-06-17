#!/usr/bin/env python3
"""
Precompute / warm up CLIP semantic cache for evaluation datasets.

This script populates the on-disk DiskTensorCache used by SemanticEncoder.

Notes:
  - The cache stores RAW CLIP features (pre-projection). This is safe even if
    semantic_encoder.projection is trained/changed across checkpoints, as long
    as the CLIP backbone weights are the same (usually frozen).
  - Keys include (video_name, T, input HxW, mode), so changing your dataset
    resize setting will automatically create a separate cache.

Example:
  python scripts/precompute_semantic_cache.py \\
    --config configs/fspt_cotracker_refine.yaml \\
    --dataset all
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Semantic (CLIP) cache")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, default=None, help="Path to config yaml")
    src.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path; uses the embedded config from the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["davis", "kinetics", "all"],
        help="Dataset to precompute",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache root directory (default: model.semantic.cache.root_dir from config).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["global", "spatial"],
        help="Override cache mode (default: model.semantic.cache.mode from config).",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per dataset")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional input resolution (H W) to match evaluation setting.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_video_name(raw, fallback: str) -> str:
    if raw is None:
        return fallback
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else fallback
    if isinstance(raw, torch.Tensor):
        try:
            if raw.ndim == 0:
                return str(raw.item())
            if raw.ndim == 1 and raw.numel() > 0:
                return str(raw[0].item())
        except Exception:
            return fallback
    return str(raw)


def _resolve_dataset_root(config, dataset_norm: str, project_root: Path) -> Path:
    root = None
    resolution = None
    extra_args = {}
    if hasattr(config, "data"):
        val_cfg = getattr(config.data, "val", None)
        if val_cfg is not None:
            try:
                from datasets import normalize_dataset_name

                if normalize_dataset_name(val_cfg.dataset) == dataset_norm:
                    root = getattr(val_cfg, "root", None)
                    resolution = getattr(val_cfg, "resolution", None)
                    extra_args["augmentation"] = getattr(val_cfg, "augmentation", {"enabled": False})
            except Exception:
                pass
    if root is None:
        root = str(project_root / "datasets" / f"tapvid_{dataset_norm}")
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = (project_root / root_path).resolve()
    return root_path


def main():
    args = parse_args()
    project_root = _project_root()

    cfg_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    if args.config is not None:
        cfg_path = Path(str(args.config))
        if not cfg_path.is_absolute():
            cfg_path = (project_root / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        from train import load_config

        config = load_config(str(cfg_path))
    else:
        checkpoint_path = Path(str(args.checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = (project_root / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "config" not in checkpoint:
            raise ValueError("Checkpoint missing 'config'. Please pass --config instead.")
        from omegaconf import OmegaConf

        config = OmegaConf.create(checkpoint["config"])
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        raise ValueError("Config missing 'model' section.")

    # Seed for reproducible dataloader sampling (evaluation datasets are deterministic anyway).
    seed = 42
    if hasattr(config, "experiment") and hasattr(config.experiment, "seed"):
        try:
            seed = int(config.experiment.seed)
        except Exception:
            seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cache_cfg = getattr(getattr(model_cfg, "semantic", None), "cache", None)
    default_cache_dir = getattr(cache_cfg, "root_dir", "outputs/semantic_cache") if cache_cfg is not None else "outputs/semantic_cache"
    default_mode = str(getattr(cache_cfg, "mode", "global")).lower().strip() if cache_cfg is not None else "global"

    cache_dir = args.cache_dir if args.cache_dir is not None else str(default_cache_dir)
    mode = str(args.mode).lower().strip() if args.mode is not None else default_mode

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    from models.semantic_encoder import SemanticEncoder, CLIP_AVAILABLE
    from utils.tensor_cache import DiskTensorCache

    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP backend is not available. Install openai/CLIP or open-clip-torch.")
    semantic_enabled = bool(getattr(getattr(model_cfg, "semantic", None), "enabled", False))
    if not semantic_enabled:
        raise RuntimeError("Config has model.semantic.enabled=false; semantic cache precompute is not applicable.")

    clip_cfg = getattr(model_cfg, "clip", None)
    temporal_cfg = getattr(model_cfg, "temporal", None)
    if clip_cfg is None or temporal_cfg is None:
        raise ValueError("Config missing model.clip or model.temporal.")

    encoder = SemanticEncoder(
        clip_model=str(getattr(clip_cfg, "model", "ViT-B/16")),
        output_dim=int(getattr(temporal_cfg, "dim", 256) or 256),
        freeze=bool(getattr(clip_cfg, "freeze", True)),
        device=str(device),
    ).to(device)
    encoder.eval()

    keep_in_memory = mode != "spatial"
    cache = DiskTensorCache(root_dir=str(cache_dir), enabled=True, write=True, keep_in_memory=keep_in_memory)

    if args.dataset == "all":
        datasets = []
        if hasattr(config, "evaluation") and hasattr(config.evaluation, "datasets"):
            from datasets import normalize_dataset_name

            datasets = [normalize_dataset_name(d) for d in config.evaluation.datasets]
        if not datasets:
            datasets = ["davis", "kinetics"]
    else:
        datasets = [args.dataset]

    from datasets import get_dataloader, normalize_dataset_name

    cache_meta = {
        "config": str(cfg_path) if cfg_path is not None else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "cache_dir": str(Path(cache_dir).resolve() if not Path(cache_dir).is_absolute() else cache_dir),
        "mode": mode,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
    }

    for ds in datasets:
        ds_norm = normalize_dataset_name(ds)
        root_path = _resolve_dataset_root(config, ds_norm, project_root)
        if not root_path.exists():
            print(f"[WARN] Dataset root does not exist: {root_path} (skipping {ds_norm})")
            continue

        extra = {"augmentation": {"enabled": False}, "seed": seed}
        if args.resolution is not None:
            extra["resolution"] = tuple(int(v) for v in args.resolution)

        loader = get_dataloader(
            name=ds_norm,
            root=str(root_path),
            batch_size=1,
            split="val",
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            distributed=False,
            **extra,
        )

        seen = 0
        iterator = tqdm(loader, desc=f"Semantic cache {ds_norm}:{mode}")
        for batch in iterator:
            if batch is None or not isinstance(batch, dict):
                continue
            video = batch.get("video", None)
            if not isinstance(video, torch.Tensor):
                continue
            if video.dim() == 4:
                video = video.unsqueeze(0)
            video = video.to(device)

            video_name = _as_video_name(batch.get("video_name", None), fallback=f"{ds_norm}_{seen:06d}")

            with torch.no_grad():
                _ = encoder(video, mode=mode, cache=cache, cache_key=video_name)

            seen += 1
            if args.limit is not None and seen >= int(args.limit):
                break

        cache_meta["datasets"][ds_norm] = {
            "root": str(root_path),
            "seen": int(seen),
        }

    meta_path = Path(cache_dir)
    if not meta_path.is_absolute():
        meta_path = (project_root / meta_path).resolve()
    meta_path.mkdir(parents=True, exist_ok=True)
    meta_json = meta_path / "precompute_meta.json"
    meta_json.write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")
    print(f"Saved meta: {meta_json}")


if __name__ == "__main__":
    main()
