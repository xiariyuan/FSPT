#!/usr/bin/env python3
"""
Real-data eval for online recovery: runs CoTracker3 refiner on TAP-Vid DAVIS
with return_info=True, collects recovery statistics.

This is a lightweight debug runner, not a full eval pipeline. It:
1. Loads the val dataloader from config
2. Runs model(video, query_points, return_info=True) on each batch
3. Aggregates recovery field statistics
4. Saves audit JSON

Usage:
  python scripts/debug_online_recovery_real_eval.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --max-batches 50
"""

from __future__ import annotations

import argparse
import json
import sys
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
                for k, v in entry.items():
                    if k != "_self_":
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/online_recovery_real_eval_audit.json")
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)

    summary = getattr(model, "online_recovery_summary", {})
    print(f"online_recovery_summary: {json.dumps(summary, default=str)}")

    print(f"Building val dataloader...")
    try:
        dataloader = build_val_loader(cfg)
    except Exception as e:
        print(f"Failed to build dataloader: {e}")
        print("Trying with minimal data config...")
        # Fallback: try with base config's data
        import traceback
        traceback.print_exc()
        audit = {"status": "DATALOADER_ERROR", "error": str(e),
                 "online_recovery_summary": summary}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(audit, indent=2, default=str) + "\n")
        return

    # Aggregate statistics
    total_batches = 0
    total_queries = 0
    total_query_frames = 0
    relocal_nonzero_elements = 0
    relocal_nonzero_queries = 0
    verifier_nonzero_elements = 0
    verifier_decisions_nonzero_elements = 0
    retracking_triggered_batches = 0
    retracking_mask_nonzero = 0
    all_verifier_scores = []
    all_relocal_conf_masked = []  # relocal_conf at relocal_mask != 0 positions
    all_relocal_conf_all = []
    per_batch_stats = []

    print(f"\nRunning eval (max {args.max_batches} batches)...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break

        if isinstance(batch, dict):
            video = batch.get("video", batch.get("frames"))
            query_points = batch.get("query_points", batch.get("queries"))
            meta = {k: v for k, v in batch.items() if k not in ("video", "frames", "query_points", "queries")}
        else:
            continue

        if video is None or query_points is None:
            continue

        video = video.to(device)
        query_points = query_points.to(device)

        with torch.no_grad():
            try:
                outputs = model(video, query_points, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        if isinstance(outputs, (list, tuple)):
            info = outputs[2] if len(outputs) >= 3 and isinstance(outputs[2], dict) else {}
        else:
            info = {}

        total_batches += 1
        B, N, T, _ = outputs[0].shape if isinstance(outputs[0], torch.Tensor) else (1, 1, 1, 2)
        n_queries = B * N
        n_query_frames = B * N * T
        total_queries += n_queries
        total_query_frames += n_query_frames

        # Check recovery fields
        reloc_mask = info.get("relocal_mask", None)
        reloc_conf = info.get("relocal_conf", None)
        verifier_scores = info.get("verifier_scores", None)
        verifier_decisions = info.get("verifier_decisions", None)
        retracking = info.get("retracking", None)
        retracking_mask = info.get("retracking_retrack_mask_bn", None)

        batch_stats = {"batch": batch_idx, "B": B, "N": N, "T": T}

        if isinstance(reloc_mask, torch.Tensor):
            nz_elements = int((reloc_mask != 0).sum().item())
            nz_queries = int(reloc_mask.any(dim=-1).sum().item())  # (B,N) any over T
            relocal_nonzero_elements += nz_elements
            relocal_nonzero_queries += nz_queries
            batch_stats["relocal_mask_nonzero_elements"] = nz_elements
            batch_stats["relocal_mask_nonzero_queries"] = nz_queries

        if isinstance(reloc_conf, torch.Tensor):
            all_relocal_conf_all.extend(reloc_conf.cpu().flatten().tolist()[:200])
            if isinstance(reloc_mask, torch.Tensor):
                masked_conf = reloc_conf[reloc_mask != 0].cpu().flatten().tolist()
                all_relocal_conf_masked.extend(masked_conf[:200])

        if isinstance(verifier_scores, torch.Tensor):
            nz = int((verifier_scores != 0).sum().item())
            verifier_nonzero_elements += nz
            batch_stats["verifier_scores_nonzero_elements"] = nz
            all_verifier_scores.extend(verifier_scores.cpu().flatten().tolist()[:100])

        if isinstance(verifier_decisions, torch.Tensor):
            nz = int((verifier_decisions != 0).sum().item())
            verifier_decisions_nonzero_elements += nz
            batch_stats["verifier_decisions_nonzero_elements"] = nz

        if isinstance(retracking_mask, torch.Tensor):
            nz = int((retracking_mask != 0).sum().item())
            retracking_mask_nonzero += nz
            if nz > 0:
                retracking_triggered_batches += 1
            batch_stats["retracking_mask_nonzero"] = nz

        if retracking is not None and isinstance(retracking, dict):
            batch_stats["retracking_keys"] = list(retracking.keys())

        per_batch_stats.append(batch_stats)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}: relocal_queries={relocal_nonzero_queries}, "
                  f"verifier_nz={verifier_nonzero_elements}, "
                  f"retrack_batches={retracking_triggered_batches}")

    # Build audit
    def _stats(arr):
        if not arr:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        a = np.array(arr)
        return {
            "count": len(arr),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }

    audit = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "online_recovery_summary": summary,
        "total_batches": total_batches,
        "total_queries": total_queries,
        "total_query_frames": total_query_frames,
        "relocal_mask": {
            "nonzero_elements": relocal_nonzero_elements,
            "nonzero_queries": relocal_nonzero_queries,
            "element_trigger_rate": relocal_nonzero_elements / max(total_query_frames, 1),
            "query_trigger_rate": relocal_nonzero_queries / max(total_queries, 1),
        },
        "verifier_scores_nonzero_elements": verifier_nonzero_elements,
        "verifier_decisions_nonzero_elements": verifier_decisions_nonzero_elements,
        "retracking_triggered_batches": retracking_triggered_batches,
        "retracking_mask_nonzero_total": retracking_mask_nonzero,
        "verifier_scores_stats": _stats(all_verifier_scores),
        "relocal_conf_masked_stats": _stats(all_relocal_conf_masked),
        "relocal_conf_all_stats": _stats(all_relocal_conf_all),
        "per_batch": per_batch_stats[:20],
    }

    print(f"\n=== Audit Summary ===")
    print(f"  Batches: {total_batches}")
    print(f"  Queries: {total_queries}")
    print(f"  Query-frames: {total_query_frames}")
    print(f"  Relocal mask nonzero elements: {relocal_nonzero_elements}")
    print(f"  Relocal mask nonzero queries: {relocal_nonzero_queries}")
    print(f"  Verifier scores nonzero elements: {verifier_nonzero_elements}")
    print(f"  Verifier decisions nonzero elements: {verifier_decisions_nonzero_elements}")
    print(f"  Retracking triggered batches: {retracking_triggered_batches}")
    print(f"  Retracking mask nonzero: {retracking_mask_nonzero}")

    reloc_conf_masked = audit["relocal_conf_masked_stats"]
    if reloc_conf_masked["count"] > 0:
        print(f"\n  Relocal conf (at relocal_mask!=0):")
        print(f"    count={reloc_conf_masked['count']}, mean={reloc_conf_masked['mean']:.4f}, "
              f"std={reloc_conf_masked['std']:.4f}")
        print(f"    min={reloc_conf_masked['min']:.4f}, max={reloc_conf_masked['max']:.4f}")

    if relocal_nonzero_queries == 0:
        audit["status"] = "RELOCAL_NEVER_TRIGGERED"
        print("\n  WARNING: relocal_mask is zero on all real data batches.")
    elif retracking_mask_nonzero == 0:
        audit["status"] = "RELOCAL_TRIGGERED_NO_RETRACKING"
        print("\n  Relocal triggered but retracking never fired.")
        if reloc_conf_masked["count"] > 0 and reloc_conf_masked["max"] is not None:
            if reloc_conf_masked["max"] < 0.1:
                print(f"  Likely cause: relocal_conf max={reloc_conf_masked['max']:.4f} < min_conf=0.1")
            else:
                print(f"  relocal_conf max={reloc_conf_masked['max']:.4f} >= min_conf=0.1; check other retracking conditions")
    else:
        audit["status"] = "FULLY_TRIGGERED"
        print("\n  OK: recovery pipeline fully triggered on real data.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
