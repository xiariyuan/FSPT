#!/usr/bin/env python3
"""
Smoke test for the online recovery adapter in CoTracker3 refiner.

This script:
1. Builds the model from a config with online_recovery enabled
2. Runs one forward pass with return_info=True
3. Checks that recovery-related fields are present and non-trivial
4. Saves an audit JSON to outputs/

Usage:
  python scripts/debug_online_recovery_smoke.py \
    --config configs/fspt_online_recovery_smoke_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str):
    """Load config with Hydra-style defaults chain resolution."""
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


def build_model(cfg, checkpoint_path: str | None, device: torch.device):
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model_cfg = cfg.get("model", {})
    model = CoTrackerFSPTRefiner(model_cfg)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Filter out shape mismatches for new fields
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if skipped:
            print(f"  Skipped {len(skipped)} mismatched keys (expected for new config)")
        model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    return model


def make_dummy_input(batch_size: int = 2, n_points: int = 4, n_frames: int = 8,
                     height: int = 256, width: int = 256, device: torch.device = torch.device("cpu")):
    """Create minimal synthetic input for a smoke forward pass."""
    video = torch.randn(batch_size, n_frames, 3, height, width, device=device) * 255.0
    # query_points: (B, N, 3) — [t, x, y]
    query_points = torch.zeros(batch_size, n_points, 3, device=device)
    query_points[:, :, 0] = 0  # frame 0
    # Place points at different locations
    positions = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for i in range(min(n_points, len(positions))):
        query_points[:, i, 1] = width * positions[i][0]
        query_points[:, i, 2] = height * positions[i][1]
    return video, query_points


def main():
    parser = argparse.ArgumentParser(description="Online recovery smoke test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/online_recovery_smoke_audit.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-points", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)

    print(f"Building model...")
    model = build_model(cfg, args.checkpoint, device)

    # Print recovery summary
    if hasattr(model, "online_recovery_summary"):
        print(f"\n=== online_recovery_summary ===")
        for k, v in model.online_recovery_summary.items():
            print(f"  {k}: {v}")
    else:
        print("WARNING: model has no online_recovery_summary attribute")

    print(f"\nRunning smoke forward pass (batch={args.batch_size}, points={args.n_points}, "
          f"frames={args.n_frames}, res={args.height}x{args.width})...")
    video, query_points = make_dummy_input(
        args.batch_size, args.n_points, args.n_frames,
        args.height, args.width, device,
    )

    with torch.no_grad():
        try:
            outputs = model(video, query_points, return_info=True)
        except Exception as e:
            print(f"ERROR during forward: {e}")
            import traceback
            traceback.print_exc()
            # Save error
            audit = {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "status": "ERROR",
                "error": str(e),
                "online_recovery_summary": getattr(model, "online_recovery_summary", None),
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(audit, indent=2) + "\n")
            print(f"Saved error audit to {args.output}")
            return

    # Unpack outputs
    if isinstance(outputs, tuple):
        tracks = outputs[0]
        visibility = outputs[1] if len(outputs) > 1 else None
        info = outputs[2] if len(outputs) > 2 else {}
    else:
        tracks = outputs
        visibility = None
        info = {}

    print(f"\n=== Forward pass succeeded ===")
    print(f"  tracks shape: {tracks.shape}")
    if visibility is not None:
        print(f"  visibility shape: {visibility.shape}")

    # Check recovery fields
    recovery_fields = [
        "online_recovery_requested",
        "online_recovery_mode",
        "online_recovery_summary",
        "relocal_mask",
        "relocal_conf",
        "verifier_scores",
        "verifier_decisions",
    ]
    retracking_fields = [
        "retracking",
        "retracking_retrack_mask_bn",
        "retracking_retrack_t0",
    ]

    audit = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "input_shape": {
            "batch_size": args.batch_size,
            "n_points": args.n_points,
            "n_frames": args.n_frames,
            "height": args.height,
            "width": args.width,
        },
        "tracks_shape": list(tracks.shape),
        "visibility_shape": list(visibility.shape) if visibility is not None else None,
        "online_recovery_summary": getattr(model, "online_recovery_summary", None),
        "field_check": {},
    }

    print(f"\n=== Recovery field check ===")
    all_ok = True
    for field in recovery_fields:
        if field in info:
            val = info[field]
            if isinstance(val, torch.Tensor):
                nonzero = int((val != 0).sum().item())
                total = val.numel()
                print(f"  {field}: shape={list(val.shape)}, nonzero={nonzero}/{total}")
                audit["field_check"][field] = {
                    "present": True,
                    "shape": list(val.shape),
                    "nonzero_count": nonzero,
                    "total": total,
                }
            elif isinstance(val, dict):
                print(f"  {field}: dict with keys {list(val.keys())}")
                audit["field_check"][field] = {"present": True, "type": "dict", "keys": list(val.keys())}
            else:
                print(f"  {field}: {val}")
                audit["field_check"][field] = {"present": True, "value": val}
        else:
            print(f"  {field}: MISSING")
            audit["field_check"][field] = {"present": False}
            all_ok = False

    print(f"\n--- Retracking fields ---")
    for field in retracking_fields:
        if field in info:
            val = info[field]
            if isinstance(val, torch.Tensor):
                nonzero = int((val != 0).sum().item())
                total = val.numel()
                print(f"  {field}: shape={list(val.shape)}, nonzero={nonzero}/{total}")
                audit["field_check"][field] = {
                    "present": True,
                    "shape": list(val.shape),
                    "nonzero_count": nonzero,
                    "total": total,
                }
            elif isinstance(val, dict):
                print(f"  {field}: dict with keys {list(val.keys())}")
                audit["field_check"][field] = {"present": True, "type": "dict", "keys": list(val.keys())}
            elif isinstance(val, (list, tuple)):
                print(f"  {field}: list/tuple len={len(val)}")
                audit["field_check"][field] = {"present": True, "type": "list", "length": len(val)}
            else:
                print(f"  {field}: {val}")
                audit["field_check"][field] = {"present": True, "value": val}
        else:
            print(f"  {field}: MISSING")
            audit["field_check"][field] = {"present": False}

    # Overall status
    summary = audit.get("online_recovery_summary", {})
    status = "PASS"
    checks = []

    if not summary.get("requested", False):
        status = "FAIL"
        checks.append("online_recovery not requested")

    if not summary.get("retracking_enabled", False):
        checks.append("retracking not enabled (warning)")

    relocal_present = audit["field_check"].get("relocal_mask", {}).get("present", False)
    if not relocal_present:
        status = "FAIL"
        checks.append("relocal_mask missing")

    relocal_nonzero = audit["field_check"].get("relocal_mask", {}).get("nonzero_count", 0)
    if relocal_nonzero == 0 and relocal_present:
        checks.append("relocal_mask is all-zero (expected on synthetic input)")

    audit["status"] = status
    audit["checks"] = checks

    print(f"\n=== Overall: {status} ===")
    for c in checks:
        print(f"  - {c}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(f"\nSaved audit to {args.output}")


if __name__ == "__main__":
    main()
