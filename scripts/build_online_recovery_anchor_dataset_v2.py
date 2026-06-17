#!/usr/bin/env python3
"""
Build tracker-conditioned recovery-anchor dataset.

Unlike the previous version, this script:
  1. Loads real TAP-Vid data via the existing dataloader
  2. Runs CoTracker3 base tracker (with online_recovery enabled) to get real base_tracks
  3. At each GT-defined re-entry event, extracts:
     - base_xy_t0 from real tracker output (NOT GT 3D reprojection)
     - gt_xy_t0 from GT annotations
     - DINO search feature map from the actual re-entry frame
     - support descriptors from pre-occlusion frames
     - tracker metadata (visibility, confidence)
  4. Saves per-sequence shards with global index

Usage:
  python scripts/build_online_recovery_anchor_dataset_v2.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --data-root /gemini/code/FSPT/datasets/pointodyssey \
    --splits val \
    --max-sequences 5 \
    --max-samples 100 \
    --output-dir outputs/recovery_anchor_dataset_v2
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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


def build_val_loader(config):
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    return _build_val_loader_from_config(config)


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


def find_reentry_events_from_occluded(occluded_np: np.ndarray, query_t: int, min_occ_len: int = 20) -> List[Dict]:
    """Find re-entry frames from GT occlusion mask for a single point.

    Args:
        occluded_np: (T,) bool, True = invisible
        query_t: query frame index
        min_occ_len: minimum occlusion length

    Returns:
        List of {reentry_frame, occ_length} dicts
    """
    T = len(occluded_np)
    events = []
    in_occ = False
    occ_start = -1

    for t in range(query_t + 1, T):
        if occluded_np[t]:
            if not in_occ:
                occ_start = t
                in_occ = True
        else:
            if in_occ:
                occ_len = t - occ_start
                if occ_len >= min_occ_len:
                    events.append({"reentry_frame": t, "occ_length": occ_len})
                in_occ = False
    return events


def main():
    parser = argparse.ArgumentParser(description="Build tracker-conditioned recovery-anchor dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    cfg = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)
    dataloader = build_val_loader(cfg)

    summary = getattr(model, "online_recovery_summary", {})
    print(f"Model: {summary}")
    print(f"Output: {output_dir}")

    # Initialize DINO extractor for support/query features
    from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
    dino_weights = Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    dino_extractor = DINOFeatureExtractor(dino_weights, device)

    def extract_patch_feature(patch_rgb_uint8: torch.Tensor) -> np.ndarray:
        """Extract DINOv2 global feature from a patch. (H,W,3) uint8 -> (D,) float32."""
        t = patch_rgb_uint8.float()
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0) / 255.0
        t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            feat = dino_extractor.model(t)[-1].mean(dim=[-2, -1]).float()
            feat = F.normalize(feat, dim=-1)
        return feat[0].cpu().numpy()

    def extract_crop(img: np.ndarray, xy: np.ndarray, crop_size: int) -> np.ndarray:
        """Extract a square crop centered on xy, zero-padded at borders."""
        h, w = img.shape[:2]
        half = crop_size // 2
        cx, cy = int(round(float(xy[0]))), int(round(float(xy[1])))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, x0 + crop_size), min(h, y0 + crop_size)
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        src = img[y0:y1, x0:x1]
        dy, dx = src.shape[:2]
        crop[:dy, :dx] = src
        return crop

    all_samples = []
    batch_count = 0

    print(f"\nRunning tracker and extracting samples (max {args.max_batches} batches)...")
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

        if video is None or query_points is None or target_points is None or occluded is None:
            continue

        video = video.to(device)
        query_points = query_points.to(device)
        if video.dim() == 4:
            video = video.unsqueeze(0)
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)
        if target_points.dim() == 3:
            target_points = target_points.unsqueeze(0)
        if occluded.dim() == 2:
            occluded = occluded.unsqueeze(0)
        B, T, C, H, W = video.shape
        N_pts = query_points.shape[1]

        # Run tracker with return_info
        meta = {
            "video_name": video_name,
            "base_tracks": batch.get("base_tracks"),
            "base_visibility": batch.get("base_visibility"),
        }
        with torch.no_grad():
            try:
                outputs = model(video, query_points, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        pred_tracks = outputs[0].cpu()       # (B, N, T, 2) normalized [0,1]
        pred_vis = outputs[1].cpu() if len(outputs) > 1 else None
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}

        base_tracks = info.get("base_tracks", None)
        if base_tracks is None:
            base_tracks = batch.get("base_tracks", None)
        if base_tracks is not None:
            base_tracks = base_tracks.cpu()

        # GT data
        gt_tracks = target_points.cpu()  # (B, N, T, 2) pixel coords
        gt_occluded = occluded.cpu()     # (B, N, T) bool
        query_pts = query_points.cpu()   # (B, N, 3)

        # Get video frames as numpy for crop extraction
        video_np = video.cpu().numpy()   # (B, T, 3, H, W) float [0, 255]

        # Normalize GT tracks
        orig_size = batch.get("original_size", None)
        if isinstance(orig_size, torch.Tensor):
            orig_h = int(orig_size[0, 0].item())
            orig_w = int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        batch_count += 1
        batch_samples = 0

        for b in range(B):
            for n in range(N_pts):
                qt = int(query_pts[b, n, 0].item())
                occ_np = gt_occluded[b, n].numpy().astype(bool)

                # Find re-entry events
                events = find_reentry_events_from_occluded(occ_np, qt, args.min_occ_length)
                if len(events) == 0:
                    continue

                for evt in events:
                    t_re = evt["reentry_frame"]
                    occ_len = evt["occ_length"]

                    # Base tracker position at t_re (normalized coords)
                    if base_tracks is not None:
                        base_xy_norm = base_tracks[b, n, t_re].numpy()  # (2,) normalized
                    else:
                        base_xy_norm = pred_tracks[b, n, t_re].numpy()

                    # GT position at t_re (normalized coords, from TAP-Vid dataloader)
                    gt_xy_norm = gt_tracks[b, n, t_re].numpy().astype(np.float32)

                    # Convert both to pixel coords
                    base_xy_px = np.array([base_xy_norm[0] * orig_w, base_xy_norm[1] * orig_h], dtype=np.float32)
                    gt_xy_px = np.array([gt_xy_norm[0] * orig_w, gt_xy_norm[1] * orig_h], dtype=np.float32)

                    base_err = float(np.linalg.norm(base_xy_px - gt_xy_px))

                    # Tracker visibility/confidence at t_re
                    if pred_vis is not None:
                        tracker_vis_t0 = float(pred_vis[b, n, t_re].item())
                    else:
                        tracker_vis_t0 = float(not occ_np[t_re])

                    # Extract RGB frames for crop extraction
                    # video_np is (B, T, 3, H, W) float
                    frame_re = video_np[b, t_re].transpose(1, 2, 0).astype(np.uint8)  # (H, W, 3)
                    frame_q = video_np[b, qt].transpose(1, 2, 0).astype(np.uint8)

                    # Extract support patches (last M visible frames before occlusion)
                    vis_np = ~occ_np  # True = visible
                    support_frames = []
                    for t in range(qt, -1, -1):
                        if vis_np[t]:
                            support_frames.append(t)
                        if len(support_frames) >= args.num_support:
                            break
                    support_frames.reverse()

                    support_patches = []
                    for t_sf in support_frames:
                        sf = video_np[b, t_sf].transpose(1, 2, 0).astype(np.uint8)
                        # Use GT position for support crop (visible frame, so GT is ground truth)
                        gt_xy_sf = gt_tracks[b, n, t_sf].numpy().astype(np.float32)
                        patch = extract_crop(sf, gt_xy_sf, 112)
                        support_patches.append(patch)

                    if len(support_patches) == 0:
                        continue

                    support_patches_np = np.stack(support_patches, axis=0)  # (K, 112, 112, 3)

                    # Extract search crop centered on base position
                    search_crop = extract_crop(frame_re, base_xy_px, 224)

                    # Extract query crop at query frame (GT position, visible)
                    gt_xy_q = gt_tracks[b, n, qt].numpy().astype(np.float32)
                    query_crop = extract_crop(frame_q, gt_xy_q, 112)

                    sample = {
                        "sample_id": len(all_samples),
                        "video_name": str(video_name[b] if isinstance(video_name, list) else video_name),
                        "point_idx": int(n),
                        "query_frame": qt,
                        "reentry_frame": t_re,
                        "occ_length": int(occ_len),
                        "base_xy": base_xy_px.tolist(),
                        "gt_xy": gt_xy_px.tolist(),
                        "base_error_px": round(base_err, 2),
                        "tracker_visibility_t0": round(tracker_vis_t0, 4),
                        "support_count": len(support_patches),
                        "support_frames": [int(x) for x in support_frames],
                        "search_crop_size": 224,
                        "query_crop_size": 112,
                        "image_H": int(H),
                        "image_W": int(W),
                    }

                    # Save tensors
                    seq_name = sample["video_name"]
                    seq_dir = output_dir / seq_name
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    sample_file = f"sample_{sample['sample_id']:06d}.npz"
                    sample["file"] = f"{seq_name}/{sample_file}"

                    np.savez_compressed(
                        seq_dir / sample_file,
                        search_crop=search_crop.astype(np.uint8),
                        query_crop=query_crop.astype(np.uint8),
                        support_patches=support_patches_np.astype(np.uint8),
                    )
                    all_samples.append(sample)
                    batch_samples += 1

        print(f"  Batch {batch_idx}: +{batch_samples} samples (total={len(all_samples)})")

    # Save global index
    (output_dir / "index_all.json").write_text(json.dumps(all_samples, indent=2) + "\n")

    # Create train/val split by video_name (leave-one-out for val)
    unique_videos = list(set(s["video_name"] for s in all_samples))
    unique_videos.sort()
    rng.shuffle(unique_videos)

    # Simple split: first 70% train, last 30% val
    n_train_vids = max(1, int(len(unique_videos) * 0.7))
    train_vids = set(unique_videos[:n_train_vids])
    val_vids = set(unique_videos[n_train_vids:])

    train_idx = [s for s in all_samples if s["video_name"] in train_vids]
    val_idx = [s for s in all_samples if s["video_name"] in val_vids]

    (output_dir / "index_train.json").write_text(json.dumps(train_idx, indent=2) + "\n")
    (output_dir / "index_val.json").write_text(json.dumps(val_idx, indent=2) + "\n")

    # Stats
    stats = {
        "total_samples": len(all_samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_videos": len(unique_videos),
        "train_videos": sorted(train_vids),
        "val_videos": sorted(val_vids),
        "config": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "min_occ_length": args.min_occ_length,
            "num_support": args.num_support,
            "max_batches": args.max_batches,
        },
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # Base error distribution
    base_errs = np.array([s["base_error_px"] for s in all_samples])
    print(f"\n{'='*50}")
    print(f"Dataset built: {len(all_samples)} samples from {len(unique_videos)} videos")
    print(f"  Train: {len(train_idx)} samples ({len(train_vids)} videos)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_vids)} videos)")
    print(f"  Base error: mean={np.mean(base_errs):.1f}, median={np.median(base_errs):.1f}, p95={np.percentile(base_errs,95):.1f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
