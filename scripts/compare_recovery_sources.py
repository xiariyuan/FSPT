#!/usr/bin/env python3
"""
Compare recovery quality: baseline vs cotracker-recovery vs dino-recovery.

Runs the full eval pipeline on TAP-Vid DAVIS with return_info=True,
computes per-query position errors, and compares across feature sources.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str):
    from omegaconf import OmegaConf
    import yaml
    def _load_and_resolve(path, _seen=None):
        if _seen is None: _seen = set()
        path = str(Path(path).resolve())
        if path in _seen: return OmegaConf.create({})
        _seen.add(path)
        with open(path) as f: raw = yaml.safe_load(f)
        if raw is None: return OmegaConf.create({})
        defaults = raw.pop("defaults", []) or []
        base = OmegaConf.create({})
        for entry in defaults:
            if isinstance(entry, str) and entry != "_self_":
                bp = Path(path).parent / f"{entry}.yaml"
                if not bp.exists(): bp = Path(path).parent / entry
                if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    if k != "_self_":
                        bp = Path(path).parent / f"{v}.yaml"
                        if not bp.exists(): bp = Path(path).parent / v
                        if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
        current = OmegaConf.create(raw)
        return OmegaConf.merge(base, current)
    return _load_and_resolve(config_path)


def build_model(cfg, checkpoint_path, device):
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
    return model.to(device).eval()


def build_val_loader(config):
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    return _build_val_loader_from_config(config)


def run_eval(model, dataloader, device, max_batches=50):
    """Run eval and collect per-query position errors + recovery info."""
    results = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches: break
        if isinstance(batch, dict):
            video = batch.get("video", batch.get("frames"))
            query_points = batch.get("query_points", batch.get("queries"))
            gt_tracks = batch.get("tracks", batch.get("gt_tracks"))
            meta = {k: v for k, v in batch.items() if k not in ("video", "frames", "query_points", "queries", "tracks", "gt_tracks")}
        else:
            continue
        if video is None or query_points is None: continue
        video = video.to(device)
        query_points = query_points.to(device)

        with torch.no_grad():
            try:
                outputs = model(video, query_points, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        pred_tracks = outputs[0]  # (B, N, T, 2)
        pred_vis = outputs[1] if len(outputs) > 1 else None
        info = outputs[2] if len(outputs) > 3 and isinstance(outputs[2], dict) else (outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {})

        # Get base tracks if available
        base_tracks = info.get("base_tracks", None)

        B, N, T, _ = pred_tracks.shape

        # Check recovery fields
        relocal_mask = info.get("relocal_mask", None)
        relocal_conf = info.get("relocal_conf", None)
        retracking_mask = info.get("retracking_retrack_mask_bn", None)

        for b in range(B):
            for n in range(N):
                entry = {"batch": batch_idx, "b": b, "n": n}
                if isinstance(relocal_mask, torch.Tensor):
                    rm = relocal_mask[b, n]  # (T,)
                    entry["relocal_mask_nframes"] = int((rm != 0).sum().item())
                    entry["relocal_mask_any"] = bool(rm.any().item())
                if isinstance(relocal_conf, torch.Tensor):
                    rc = relocal_conf[b, n]
                    entry["relocal_conf_max"] = float(rc.max().item())
                    entry["relocal_conf_mean"] = float(rc[rc > 0].mean().item()) if (rc > 0).any() else 0.0
                if isinstance(retracking_mask, torch.Tensor):
                    entry["retracked"] = bool(retracking_mask[b, n].item())
                if isinstance(base_tracks, torch.Tensor):
                    bt = base_tracks[b, n]  # (T, 2)
                    pt = pred_tracks[b, n]  # (T, 2)
                    entry["base_vs_pred_diff_mean"] = float((bt - pt).norm(dim=-1).mean().item())
                results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/recovery_comparison_audit.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "baseline": "configs/fspt_online_recovery_real_eval256.yaml",
        "cotracker_recovery": "configs/fspt_online_recovery_cotracker_conf0005.yaml",
        "dino_recovery": "configs/fspt_online_recovery_dino_conf0005.yaml",
    }

    all_results = {}
    for name, config_path in configs.items():
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        cfg = load_config(config_path)
        model = build_model(cfg, args.checkpoint, device)
        dataloader = build_val_loader(cfg)
        results = run_eval(model, dataloader, device, args.max_batches)
        all_results[name] = results

        # Summary stats
        n_queries = len(results)
        n_relocal = sum(1 for r in results if r.get("relocal_mask_any", False))
        n_retracked = sum(1 for r in results if r.get("retracked", False))
        conf_vals = [r.get("relocal_conf_max", 0) for r in results if r.get("relocal_conf_max", 0) > 0]

        print(f"  Queries: {n_queries}")
        print(f"  Relocal triggered: {n_relocal}")
        print(f"  Retracked: {n_retracked}")
        if conf_vals:
            print(f"  Relocal conf: mean={np.mean(conf_vals):.4f}, max={np.max(conf_vals):.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Convert to serializable
    save_data = {}
    for name, results in all_results.items():
        save_data[name] = {
            "n_queries": len(results),
            "n_relocal": sum(1 for r in results if r.get("relocal_mask_any", False)),
            "n_retracked": sum(1 for r in results if r.get("retracked", False)),
            "samples": results[:100],
        }
    Path(args.output).write_text(json.dumps(save_data, indent=2, default=str) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
