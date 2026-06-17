#!/usr/bin/env python3
"""
Build tracker-conditioned recovery-anchor dataset (v3).

Key fixes over v2:
  - Uses shared extract_crop from utils/crop_utils.py (cv2.getRectSubPix)
  - Saves training-ready fields: support_descriptors, search_feature_map
  - Supports explicit --train-videos / --val-videos split
  - Saves base_xy in both normalized and pixel coords
  - Fixed file indexing (sample_file in metadata, no samples.index() hack)

Usage:
  python scripts/build_online_recovery_anchor_dataset_v3.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --max-batches 20 \
    --val-videos bike-packing,bmx-bumps,soapbox \
    --output-dir outputs/recovery_anchor_dataset_v3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.crop_utils import extract_crop


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


def find_reentry_events(occluded_np: np.ndarray, query_t: int, min_occ_len: int = 20) -> List[Dict]:
    """Find re-entry frames from GT occlusion mask."""
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


def extract_dino_support_descriptor(
    extractor, support_patches_np: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Extract mean-pooled DINO descriptor from support patches.

    Args:
        support_patches_np: (K, H, W, 3) uint8
    Returns:
        (D,) float32 normalized descriptor
    """
    K = support_patches_np.shape[0]
    t = torch.from_numpy(support_patches_np).float().permute(0, 3, 1, 2) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        feat = extractor.model(t)[-1].mean(dim=[-2, -1]).float()  # (K, D)
        feat = F.normalize(feat, dim=-1)
    pooled = feat.mean(dim=0, keepdim=True)
    pooled = F.normalize(pooled, dim=-1)
    return pooled[0].cpu().numpy()


def extract_dino_feature_map(
    extractor, image_rgb_uint8: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Extract DINOv2 feature map from an image.

    Args:
        image_rgb_uint8: (H, W, 3) uint8
    Returns:
        (D, h, w) float32 normalized feature map
    """
    t = torch.from_numpy(image_rgb_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        feat = extractor.model(t)[-1]  # (1, D, h, w)
        feat = F.normalize(feat.float(), dim=1)
    return feat[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Build v3 recovery-anchor dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--val-videos", type=str, default="",
                        help="Comma-separated video names for val split. Rest goes to train.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-features", action="store_true",
                        help="Save DINO support descriptors and search feature maps (slower)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)
    dataloader = build_val_loader(cfg)

    # Initialize DINO extractor
    from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
    dino_weights = Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    dino_extractor = DINOFeatureExtractor(dino_weights, device)

    summary = getattr(model, "online_recovery_summary", {})
    print(f"Model: {summary}")
    print(f"Save features: {args.save_features}")

    # Parse val videos
    val_videos = set(v.strip() for v in args.val_videos.split(",") if v.strip()) if args.val_videos else None

    all_samples = []
    batch_count = 0

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

        if video is None or query_points is None or target_points is None or occluded is None:
            continue

        video = video.to(device)
        query_points = query_points.to(device)
        if video.dim() == 4: video = video.unsqueeze(0)
        if query_points.dim() == 2: query_points = query_points.unsqueeze(0)
        if target_points.dim() == 3: target_points = target_points.unsqueeze(0)
        if occluded.dim() == 2: occluded = occluded.unsqueeze(0)
        B, T, C, H, W = video.shape
        N_pts = query_points.shape[1]

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

        pred_tracks = outputs[0].cpu()
        pred_vis = outputs[1].cpu() if len(outputs) > 1 else None
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}

        base_tracks = info.get("base_tracks")
        if base_tracks is None:
            base_tracks = batch.get("base_tracks")
        if base_tracks is not None:
            base_tracks = base_tracks.cpu()

        gt_tracks = target_points.cpu()
        gt_occluded = occluded.cpu()
        query_pts = query_points.cpu()
        video_np = video.cpu().numpy()

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        batch_count += 1
        batch_samples = 0

        for b in range(B):
            for n in range(N_pts):
                qt = int(query_pts[b, n, 0].item())
                occ_np = gt_occluded[b, n].numpy().astype(bool)
                events = find_reentry_events(occ_np, qt, args.min_occ_length)
                if not events:
                    continue

                for evt in events:
                    t_re = evt["reentry_frame"]
                    occ_len = evt["occ_length"]

                    # Base position from tracker (normalized -> pixel)
                    if base_tracks is not None:
                        base_norm = base_tracks[b, n, t_re].numpy()
                    else:
                        base_norm = pred_tracks[b, n, t_re].numpy()
                    base_xy = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h], dtype=np.float32)

                    # GT position (normalized -> pixel)
                    gt_norm = gt_tracks[b, n, t_re].numpy()
                    gt_xy = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h], dtype=np.float32)

                    base_err = float(np.linalg.norm(base_xy - gt_xy))

                    # Tracker visibility at t_re
                    tracker_vis = float(pred_vis[b, n, t_re].item()) if pred_vis is not None else 0.0

                    # RGB frames
                    frame_re = video_np[b, t_re].transpose(1, 2, 0).astype(np.uint8)
                    frame_q = video_np[b, qt].transpose(1, 2, 0).astype(np.uint8)

                    # Support patches
                    vis_np = ~occ_np
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
                        gt_sf = gt_tracks[b, n, t_sf].numpy()
                        gt_sf_px = np.array([gt_sf[0] * orig_w, gt_sf[1] * orig_h], dtype=np.float32)
                        patch = extract_crop(sf, gt_sf_px, args.query_crop_size)
                        support_patches.append(patch)

                    if len(support_patches) == 0:
                        continue
                    support_patches_np = np.stack(support_patches, axis=0)

                    # Search crop around base position (uses shared extract_crop)
                    search_crop = extract_crop(frame_re, base_xy, args.search_crop_size)

                    # Query crop at GT position in query frame
                    gt_q = gt_tracks[b, n, qt].numpy()
                    gt_q_px = np.array([gt_q[0] * orig_w, gt_q[1] * orig_h], dtype=np.float32)
                    query_crop = extract_crop(frame_q, gt_q_px, args.query_crop_size)

                    # Build sample metadata
                    vid_name = str(video_name[b] if isinstance(video_name, list) else video_name)
                    sample_id = len(all_samples)
                    sample_file = f"sample_{sample_id:06d}.npz"

                    sample = {
                        "sample_id": sample_id,
                        "video_name": vid_name,
                        "point_idx": int(n),
                        "query_frame": qt,
                        "reentry_frame": t_re,
                        "occ_length": int(occ_len),
                        "base_xy": base_xy.tolist(),
                        "gt_xy": gt_xy.tolist(),
                        "base_xy_norm": base_norm.tolist(),
                        "gt_xy_norm": gt_norm.tolist(),
                        "base_error_px": round(base_err, 2),
                        "tracker_visibility_t0": round(tracker_vis, 4),
                        "support_count": len(support_patches),
                        "support_frames": [int(x) for x in support_frames],
                        "search_crop_size": args.search_crop_size,
                        "query_crop_size": args.query_crop_size,
                        "image_H": int(orig_h),
                        "image_W": int(orig_w),
                        "file": f"{vid_name}/{sample_file}",
                    }

                    # Save tensors
                    seq_dir = output_dir / vid_name
                    seq_dir.mkdir(parents=True, exist_ok=True)

                    save_dict = {
                        "search_crop": search_crop.astype(np.uint8),
                        "query_crop": query_crop.astype(np.uint8),
                        "support_patches": support_patches_np.astype(np.uint8),
                    }

                    # Optional: save precomputed features
                    if args.save_features:
                        supp_desc = extract_dino_support_descriptor(dino_extractor, support_patches_np, device)
                        search_fmap = extract_dino_feature_map(dino_extractor, search_crop, device)
                        save_dict["support_descriptor"] = supp_desc.astype(np.float32)
                        save_dict["search_feature_map"] = search_fmap.astype(np.float32)

                    np.savez_compressed(seq_dir / sample_file, **save_dict)
                    all_samples.append(sample)
                    batch_samples += 1

        print(f"  Batch {batch_idx}: +{batch_samples} samples (total={len(all_samples)})")

    # Split by video
    unique_videos = sorted(set(s["video_name"] for s in all_samples))
    if val_videos:
        train_vids = set(v for v in unique_videos if v not in val_videos)
        actual_val_vids = set(v for v in unique_videos if v in val_videos)
    else:
        # Default: first video = val, rest = train
        actual_val_vids = {unique_videos[-1]} if unique_videos else set()
        train_vids = set(unique_videos[:-1])

    train_idx = [s for s in all_samples if s["video_name"] in train_vids]
    val_idx = [s for s in all_samples if s["video_name"] in actual_val_vids]

    (output_dir / "index_all.json").write_text(json.dumps(all_samples, indent=2) + "\n")
    (output_dir / "index_train.json").write_text(json.dumps(train_idx, indent=2) + "\n")
    (output_dir / "index_val.json").write_text(json.dumps(val_idx, indent=2) + "\n")

    stats = {
        "total_samples": len(all_samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_videos": len(unique_videos),
        "train_videos": sorted(train_vids),
        "val_videos": sorted(actual_val_vids),
        "save_features": args.save_features,
        "config": vars(args),
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    base_errs = np.array([s["base_error_px"] for s in all_samples])
    print(f"\n{'='*50}")
    print(f"Dataset: {len(all_samples)} samples, {len(unique_videos)} videos")
    print(f"  Train: {len(train_idx)} ({sorted(train_vids)})")
    print(f"  Val:   {len(val_idx)} ({sorted(actual_val_vids)})")
    print(f"  Base error: mean={np.mean(base_errs):.1f}, median={np.median(base_errs):.1f}, p95={np.percentile(base_errs,95):.1f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
