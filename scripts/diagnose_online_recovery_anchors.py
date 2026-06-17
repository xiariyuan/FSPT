#!/usr/bin/env python3
"""
Diagnose online recovery failure modes on real data.

This script is complementary to debug_online_recovery_real_eval.py:
it does not try to summarize trigger counts only. Instead, it measures
where the recovery pipeline fails for retracked queries:

1. base track error at retrack start t0
2. relocal anchor error at t0
3. post-retracking error at t0
4. post-retracking tail error on frames >= t0

The goal is to separate:
- bad anchor selection / bad relocalization
- good anchor but poor retracking tail
- good retracking tail but harmful splice policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str):
    from omegaconf import OmegaConf
    import yaml

    def _load_and_resolve(path, _seen=None):
        if _seen is None:
            _seen = set()
        path = str(Path(path).resolve())
        if path in _seen:
            return OmegaConf.create({})
        _seen.add(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            return OmegaConf.create({})
        defaults = raw.pop("defaults", []) or []
        base = OmegaConf.create({})
        for entry in defaults:
            if isinstance(entry, str) and entry != "_self_":
                bp = Path(path).parent / f"{entry}.yaml"
                if not bp.exists():
                    bp = Path(path).parent / entry
                if bp.exists():
                    base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
            elif isinstance(entry, dict):
                for _, v in entry.items():
                    bp = Path(path).parent / f"{v}.yaml"
                    if not bp.exists():
                        bp = Path(path).parent / v
                    if bp.exists():
                        base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
        current = OmegaConf.create(raw)
        return OmegaConf.merge(base, current)

    return _load_and_resolve(config_path)


def build_model(cfg, checkpoint_path, device):
    from models.cotracker_refiner import CoTrackerFSPTRefiner

    model_cfg = cfg.get("model", {})
    model = CoTrackerFSPTRefiner(model_cfg)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model_state = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    return model


def build_val_loader(config):
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config

    return _build_val_loader_from_config(config)


def _as_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p5": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _l2_px(pred_yx: torch.Tensor, gt_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    dy = (pred_yx[..., 0] - gt_yx[..., 0]) * float(height)
    dx = (pred_yx[..., 1] - gt_yx[..., 1]) * float(width)
    return torch.sqrt(dy * dy + dx * dx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)
    dataloader = build_val_loader(cfg)

    base_t0_err: List[float] = []
    anchor_t0_err: List[float] = []
    pre_retrack_t0_err: List[float] = []
    post_retrack_t0_err: List[float] = []
    base_tail_err: List[float] = []
    post_retrack_tail_err: List[float] = []
    conf_t0_vals: List[float] = []
    retracked_queries = 0
    relocal_queries = 0
    per_batch: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue
        video = batch.get("video", batch.get("frames"))
        query_points = batch.get("query_points", batch.get("queries"))
        target_points = batch.get("target_points")
        if video is None or query_points is None or target_points is None:
            continue

        meta = {k: v for k, v in batch.items() if k not in ("video", "frames", "query_points", "queries")}
        video = video.to(device)
        query_points = query_points.to(device)
        target_points = target_points.to(device)

        with torch.no_grad():
            outputs = model(video, query_points, meta=meta, return_info=True)

        tracks, _vis, info = outputs
        relocal_mask = info.get("relocal_mask")
        retrack_mask = info.get("retracking_retrack_mask_bn")
        retrack_t0 = info.get("retracking_retrack_t0")
        base_tracks = info.get("base_tracks")
        anchor_tracks = info.get("relocal_anchor_tracks")
        pre_retracking_tracks = info.get("pre_retracking_tracks")
        relocal_conf = info.get("relocal_conf")

        if not isinstance(relocal_mask, torch.Tensor):
            continue
        relocal_queries += int(relocal_mask.any(dim=-1).sum().item())

        batch_row: Dict[str, Any] = {"batch": batch_idx}
        batch_retracked = 0

        # Determine which queries to analyze:
        # Use retrack_mask if available, else fall back to relocal_mask
        diag_mask = None
        diag_t0 = None
        if isinstance(retrack_mask, torch.Tensor) and isinstance(retrack_t0, torch.Tensor):
            diag_mask = retrack_mask
            diag_t0 = retrack_t0
        elif isinstance(relocal_mask, torch.Tensor):
            # Fall back: use relocal_mask and find first triggered frame per query
            diag_mask = relocal_mask.any(dim=-1)  # (B, N)
            # Build t0 from first triggered frame
            diag_t0 = torch.zeros(diag_mask.shape[0], diag_mask.shape[1], dtype=torch.long)
            for bb in range(relocal_mask.shape[0]):
                for nn in range(relocal_mask.shape[1]):
                    t_idx = torch.nonzero(relocal_mask[bb, nn], as_tuple=False)
                    if t_idx.numel() > 0:
                        diag_t0[bb, nn] = t_idx[0]

        if (
            isinstance(diag_mask, torch.Tensor)
            and isinstance(diag_t0, torch.Tensor)
            and isinstance(base_tracks, torch.Tensor)
            and isinstance(anchor_tracks, torch.Tensor)
        ):
            B, N, T, _ = tracks.shape
            H, W = int(video.shape[-2]), int(video.shape[-1])
            for b in range(B):
                idx = torch.nonzero(diag_mask[b], as_tuple=False).squeeze(-1)
                for n_idx in idx.tolist():
                    t0 = int(diag_t0[b, n_idx].item())
                    if not (0 <= t0 < T):
                        continue
                    gt_t0 = target_points[b, n_idx, t0]
                    base_t0 = base_tracks[b, n_idx, t0]
                    anchor_t0 = anchor_tracks[b, n_idx, t0]
                    pre_t0 = pre_retracking_tracks[b, n_idx, t0]
                    post_t0 = tracks[b, n_idx, t0]

                    base_t0_err.append(float(_l2_px(base_t0, gt_t0, H, W).item()))
                    anchor_t0_err.append(float(_l2_px(anchor_t0, gt_t0, H, W).item()))
                    pre_retrack_t0_err.append(float(_l2_px(pre_t0, gt_t0, H, W).item()))
                    post_retrack_t0_err.append(float(_l2_px(post_t0, gt_t0, H, W).item()))

                    tail_gt = target_points[b, n_idx, t0:]
                    tail_base = base_tracks[b, n_idx, t0:]
                    tail_post = tracks[b, n_idx, t0:]
                    if tail_gt.numel() > 0:
                        base_tail_err.extend(_l2_px(tail_base, tail_gt, H, W).detach().cpu().tolist())
                        post_retrack_tail_err.extend(_l2_px(tail_post, tail_gt, H, W).detach().cpu().tolist())

                    if isinstance(relocal_conf, torch.Tensor) and relocal_conf.shape[:3] == tracks.shape[:3]:
                        conf_t0_vals.append(float(relocal_conf[b, n_idx, t0].item()))

                    retracked_queries += 1
                    batch_retracked += 1

        batch_row["retracked_queries"] = batch_retracked
        per_batch.append(batch_row)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_batches": args.max_batches,
        "relocal_queries": relocal_queries,
        "retracked_queries": retracked_queries,
        "base_t0_error_px": _as_stats(base_t0_err),
        "anchor_t0_error_px": _as_stats(anchor_t0_err),
        "pre_retracking_t0_error_px": _as_stats(pre_retrack_t0_err),
        "post_retracking_t0_error_px": _as_stats(post_retrack_t0_err),
        "base_tail_error_px": _as_stats(base_tail_err),
        "post_retracking_tail_error_px": _as_stats(post_retrack_tail_err),
        "relocal_conf_t0": _as_stats(conf_t0_vals),
        "per_batch": per_batch,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
