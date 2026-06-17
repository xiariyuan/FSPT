#!/usr/bin/env python3
"""
Build M1 gate training dataset from real online recovery pipeline.

For each triggered query on TAP-Vid DAVIS:
  1. Run CoTracker with online_recovery enabled
  2. At each relocal_mask trigger, extract gate features
  3. Compute GT labels from base/anchor errors
  4. Save per-sample with sequence-level metadata

Usage:
  python scripts/build_online_recovery_gate_dataset_v1.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --output-dir outputs/online_recovery_gate_dataset_v1 \
    --max-batches 30
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.recovery_features import build_gate_features_scalar, build_gate_labels, GATE_FEATURE_NAMES


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


def main():
    parser = argparse.ArgumentParser(description="Build M1 gate dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)
    dataloader = build_val_loader(cfg)

    summary = getattr(model, "online_recovery_summary", {})
    print(f"Model: {json.dumps(summary, default=str)}")

    all_samples = []

    print(f"\nRunning tracker (max {args.max_batches} batches)...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points", batch.get("tracks"))
        occluded = batch.get("occluded", batch.get("visibility"))
        video_name = batch.get("video_name", ["unknown"])

        if video is None or query_points is None or target_points is None:
            continue

        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)
        if target_points.dim() == 3: target_points = target_points.unsqueeze(0)
        if occluded.dim() == 2: occluded = occluded.unsqueeze(0)

        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]

        meta = {
            "video_name": video_name,
            "base_tracks": batch.get("base_tracks"),
            "base_visibility": batch.get("base_visibility"),
        }

        with torch.no_grad():
            try:
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        pred_tracks = outputs[0].cpu()
        pred_vis = outputs[1].cpu() if len(outputs) > 1 else None
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}

        relocal_mask = info.get("relocal_mask")
        relocal_conf = info.get("relocal_conf")
        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        anchor_tracks = info.get("relocal_anchor_tracks")

        if not isinstance(relocal_mask, torch.Tensor):
            continue
        if base_tracks is not None:
            base_tracks = base_tracks.cpu()
        if anchor_tracks is not None:
            anchor_tracks = anchor_tracks.cpu()

        gt_tracks = target_points.cpu()
        gt_occluded = occluded.cpu()

        # Get image size for pixel conversion
        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)
        batch_samples = 0

        for b in range(B):
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered_frames = np.where(rm)[0]
                if len(triggered_frames) == 0:
                    continue

                # Use first triggered frame as t0
                t0 = int(triggered_frames[0])

                # Base position
                if base_tracks is not None:
                    base_norm = base_tracks[b, n_idx, t0].numpy()
                else:
                    base_norm = pred_tracks[b, n_idx, t0].numpy()
                base_xy_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h], dtype=np.float32)

                # Anchor position
                if anchor_tracks is not None:
                    anchor_norm = anchor_tracks[b, n_idx, t0].numpy()
                else:
                    anchor_norm = base_norm.copy()
                anchor_xy_px = np.array([anchor_norm[0] * orig_w, anchor_norm[1] * orig_h], dtype=np.float32)

                # GT position
                gt_norm = gt_tracks[b, n_idx, t0].numpy()
                gt_xy_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h], dtype=np.float32)

                base_err = float(np.linalg.norm(base_xy_px - gt_xy_px))
                anchor_err = float(np.linalg.norm(anchor_xy_px - gt_xy_px))

                # Relocal conf
                if isinstance(relocal_conf, torch.Tensor):
                    rc = float(relocal_conf[b, n_idx, t0].item())
                else:
                    rc = 0.0

                # Base vis
                bv = float(pred_vis[b, n_idx, t0].item()) if pred_vis is not None else 0.0

                # Occlusion length
                occ_np = gt_occluded[b, n_idx].numpy().astype(bool)
                occ_start = t0 - 1
                for t_check in range(t0 - 1, -1, -1):
                    if occ_np[t_check]:
                        occ_start = t_check
                    else:
                        break
                occ_len = t0 - occ_start

                # Build features and labels
                gate_feat = build_gate_features_scalar(base_norm, anchor_norm, rc, bv, occ_len)
                gate_label = build_gate_labels(base_err, anchor_err)

                sample = {
                    "sample_id": len(all_samples),
                    "video_name": vid_name,
                    "point_idx": int(n_idx),
                    "t0": t0,
                    "occ_length": int(occ_len),
                    "base_xy_norm": base_norm.tolist(),
                    "anchor_xy_norm": anchor_norm.tolist(),
                    "gt_xy_norm": gt_norm.tolist(),
                    "base_xy_px": base_xy_px.tolist(),
                    "anchor_xy_px": anchor_xy_px.tolist(),
                    "gt_xy_px": gt_xy_px.tolist(),
                    "base_error_px": round(base_err, 2),
                    "anchor_error_px": round(anchor_err, 2),
                    "relocal_conf": round(rc, 6),
                    "base_vis": round(bv, 4),
                    "gate_features": gate_feat.tolist(),
                    "feature_names": GATE_FEATURE_NAMES,
                    **gate_label,
                }

                # Save features as npz
                seq_dir = output_dir / vid_name
                seq_dir.mkdir(parents=True, exist_ok=True)
                sample_file = f"sample_{sample['sample_id']:06d}.npz"
                sample["file"] = f"{vid_name}/{sample_file}"
                np.savez_compressed(seq_dir / sample_file, gate_features=gate_feat)

                all_samples.append(sample)
                batch_samples += 1

        if batch_samples > 0:
            print(f"  Batch {batch_idx}: +{batch_samples} samples (total={len(all_samples)})")

    # Split by video
    unique_videos = sorted(set(s["video_name"] for s in all_samples))
    rng.shuffle(unique_videos)
    n_train_vids = max(1, int(len(unique_videos) * 0.7))
    train_vids = set(unique_videos[:n_train_vids])
    val_vids = set(unique_videos[n_train_vids:])

    train_idx = [s for s in all_samples if s["video_name"] in train_vids]
    val_idx = [s for s in all_samples if s["video_name"] in val_vids]

    (output_dir / "index_all.json").write_text(json.dumps(all_samples, indent=2) + "\n")
    (output_dir / "index_train.json").write_text(json.dumps(train_idx, indent=2) + "\n")
    (output_dir / "index_val.json").write_text(json.dumps(val_idx, indent=2) + "\n")

    # Stats
    accept_labels = np.array([s["accept_label"] for s in all_samples])
    base_errors = np.array([s["base_error_px"] for s in all_samples])

    stats = {
        "total_samples": len(all_samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_videos": len(unique_videos),
        "train_videos": sorted(train_vids),
        "val_videos": sorted(val_vids),
        "accept_pos_frac": round(float((accept_labels == 1).mean()), 4),
        "accept_neg_frac": round(float((accept_labels == 0).mean()), 4),
        "accept_ambiguous_frac": round(float((accept_labels == -1).mean()), 4),
        "base_good_frac": round(float((base_errors <= 4).mean()), 4),
        "base_bad_frac": round(float((base_errors > 16).mean()), 4),
        "base_very_bad_frac": round(float((base_errors > 32).mean()), 4),
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\n{'='*50}")
    print(f"Dataset: {len(all_samples)} samples, {len(unique_videos)} videos")
    print(f"  Train: {len(train_idx)} ({sorted(train_vids)})")
    print(f"  Val:   {len(val_idx)} ({sorted(val_vids)})")
    print(f"  Accept pos: {stats['accept_pos_frac']:.3f}")
    print(f"  Accept neg: {stats['accept_neg_frac']:.3f}")
    print(f"  Ambiguous:  {stats['accept_ambiguous_frac']:.3f}")
    print(f"  Base good (<=4px):  {stats['base_good_frac']:.3f}")
    print(f"  Base bad (>16px):   {stats['base_bad_frac']:.3f}")
    print(f"  Base very bad (>32): {stats['base_very_bad_frac']:.3f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
