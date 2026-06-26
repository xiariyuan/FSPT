#!/usr/bin/env python3
"""Build CT-offline-centered local refiner dataset (Phase 3).

Key differences from v3 builder:
  - Center is CoTracker3 offline prediction at re-entry (not base tracker)
  - Uses unified .pt caches (no model forward pass)
  - Adds ct_offline_xy, raw_ct_error_px, gt_inside fields
  - Search crop centered on CT-offline prediction, radius=16px default
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

from fspt.paths import repo_root, resolve_repo_path

PROJECT_ROOT = repo_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.crop_utils import extract_crop
from utils.attempt0_schema import load_attempt0_cache


def find_first_reentry(gt_visibility: np.ndarray, query_t: int, min_occ_len: int = 0):
    """Returns (reentry_t, occ_length, last_visible_t) or None."""
    in_occ = False
    occ_len = 0
    last_visible_t = query_t
    for t in range(int(query_t) + 1, int(gt_visibility.shape[0])):
        visible = bool(gt_visibility[t])
        if not visible:
            if not in_occ:
                last_visible_t = t - 1
            in_occ = True
            occ_len += 1
            continue
        if in_occ:
            if occ_len >= min_occ_len:
                return t, occ_len, last_visible_t
            else:
                in_occ = False
                occ_len = 0
        # Continue searching even after a short occlusion
    return None


def extract_dino_support_descriptor(
    dino_model: torch.nn.Module, support_patches_np: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Extract mean-pooled DINO descriptor from support patches."""
    K = support_patches_np.shape[0]
    t = torch.from_numpy(support_patches_np).float().permute(0, 3, 1, 2) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        feat = dino_model(t)[-1].mean(dim=[-2, -1]).float()
        feat = F.normalize(feat, dim=-1)
    pooled = feat.mean(dim=0, keepdim=True)
    pooled = F.normalize(pooled, dim=-1)
    return pooled[0].cpu().numpy()


def extract_dino_feature_map(
    dino_model: torch.nn.Module, image_rgb_uint8: np.ndarray, device: torch.device,
) -> np.ndarray:
    """Extract DINOv2 feature map from an image crop."""
    t = torch.from_numpy(image_rgb_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        feat = dino_model(t)[-1]
        feat = F.normalize(feat.float(), dim=1)
    return feat[0].cpu().numpy()


def yx_norm_to_xy_px(yx: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert [y,x] normalized [0,1] to [x,y] pixel."""
    return np.array([
        float(yx[1]) * max(float(w) - 1, 1),
        float(yx[0]) * max(float(h) - 1, 1),
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-offline-cache", type=str,
                        default="outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
    parser.add_argument("--gt-cache", type=str,
                        default="caches/trackon2_strided_original.pt")
    parser.add_argument("--pkl-path", type=str,
                        default=str(resolve_repo_path("..", "datasets", "tapvid_davis", "tapvid_davis.pkl")))
    parser.add_argument("--max-videos", type=int, default=0, help="0 = all")
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all")
    parser.add_argument("--min-occ-length", type=int, default=0)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--local-radius-px", type=int, default=16)
    parser.add_argument("--val-videos", type=str, default="",
                        help="Comma-separated video names for val split")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load caches
    ct_off_cache = load_attempt0_cache(Path(args.ct_offline_cache))
    gt_cache = load_attempt0_cache(Path(args.gt_cache))
    ct_off_records = ct_off_cache["records"]
    gt_records = gt_cache["records"]

    if args.max_videos > 0:
        ct_off_records = ct_off_records[:args.max_videos]
        gt_records = gt_records[:args.max_videos]

    # Load pkl for RGB frames
    import pickle
    with open(args.pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    # DINOv2 feature extractor
    from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
    dino_weights = resolve_repo_path("weights", "dinov2", "dinov2_vits14_pretrain.pth")
    dino_extractor = DINOFeatureExtractor(dino_weights, device)
    dino_model = dino_extractor.model
    dino_model.eval()

    # Parse val videos
    val_videos = set(v.strip() for v in args.val_videos.split(",") if v.strip()) if args.val_videos else None

    all_samples = []
    total_queries = 0
    stop = False

    for rec_idx in range(len(gt_records)):
        if stop:
            break

        ct_r = ct_off_records[rec_idx]
        gt_r = gt_records[rec_idx]
        video_id = str(gt_r["video_id"])

        gt_visibility = np.asarray(gt_r["gt_visibility"], dtype=bool)
        gt_tracks = np.asarray(gt_r["gt_tracks"], dtype=np.float32)
        query_points = np.asarray(gt_r["query_points"], dtype=np.float32)
        ct_pred = np.asarray(ct_r["pred_tracks"], dtype=np.float32)

        orig_h = int(gt_r["original_size"][0])
        orig_w = int(gt_r["original_size"][1])

        # Get video frames from pkl
        video_entry = pkl_data[video_id]
        video_rgb = np.asarray(video_entry["video"], dtype=np.uint8)

        frame_cache: Dict[int, np.ndarray] = {}

        def get_frame(t):
            if t not in frame_cache:
                frame_cache[t] = video_rgb[t]
            return frame_cache[t]

        for qi in range(query_points.shape[0]):
            if args.max_queries > 0 and total_queries >= args.max_queries:
                stop = True
                break

            query_t = int(np.clip(round(float(query_points[qi, 0])), 0, gt_visibility.shape[1] - 1))
            reentry_info = find_first_reentry(gt_visibility[qi], query_t, args.min_occ_length)
            if reentry_info is None:
                continue
            reentry_t, occ_length, last_visible_t = reentry_info

            # GT position at re-entry (pixel xy)
            gt_yx = gt_tracks[qi, reentry_t]
            gt_xy = yx_norm_to_xy_px(gt_yx, orig_h, orig_w)

            # CT-offline prediction at re-entry (pixel xy)
            ct_off_yx = ct_pred[qi, reentry_t]
            ct_off_xy = yx_norm_to_xy_px(ct_off_yx, orig_h, orig_w)

            # Raw CT-offline error
            raw_ct_error_px = float(np.linalg.norm(ct_off_xy - gt_xy))

            # Search crop centered on CT-offline prediction
            frame_re = get_frame(reentry_t)
            search_crop = extract_crop(frame_re, ct_off_xy, args.search_crop_size)

            # Check if GT is inside search crop
            half = args.search_crop_size / 2.0
            offset = gt_xy - ct_off_xy
            gt_inside = bool(abs(offset[0]) <= half and abs(offset[1]) <= half)

            # GT inside local radius (for verification)
            gt_inside_radius = bool(np.linalg.norm(offset) <= args.local_radius_px)

            # Support patches: last visible frames before occlusion
            vis_np = gt_visibility[qi]
            support_frames = []
            for t in range(last_visible_t, query_t, -1):
                if vis_np[t]:
                    support_frames.append(t)
                if len(support_frames) >= args.num_support:
                    break
            support_frames.reverse()

            support_patches = []
            for t_sf in support_frames:
                sf = get_frame(t_sf)
                gt_sf_yx = gt_tracks[qi, t_sf]
                gt_sf_px = yx_norm_to_xy_px(gt_sf_yx, orig_h, orig_w)
                patch = extract_crop(sf, gt_sf_px, args.query_crop_size)
                support_patches.append(patch)

            if len(support_patches) == 0:
                continue
            support_patches_np = np.stack(support_patches, axis=0)

            # Extract DINO features
            supp_desc = extract_dino_support_descriptor(dino_model, support_patches_np, device)
            search_fmap = extract_dino_feature_map(dino_model, search_crop, device)

            # Save sample
            vid_name = video_id
            sample_id = len(all_samples)
            sample_file = f"sample_{sample_id:06d}.npz"

            sample = {
                "sample_id": sample_id,
                "video_name": vid_name,
                "point_idx": int(qi),
                "query_frame": int(query_t),
                "reentry_frame": int(reentry_t),
                "occ_length": int(occ_length),
                "ct_offline_xy": ct_off_xy.tolist(),
                "gt_xy": gt_xy.tolist(),
                "raw_ct_error_px": round(raw_ct_error_px, 2),
                "gt_inside": int(gt_inside),
                "gt_inside_radius": int(gt_inside_radius),
                "local_radius_px": args.local_radius_px,
                "base_xy": ct_off_xy.tolist(),  # RecoveryAnchorDataset expects base_xy
                "base_xy_norm": ct_off_yx.tolist(),
                "base_error_px": round(raw_ct_error_px, 2),
                "tracker_visibility_t0": 0.0,  # No tracker visibility in CT-offline
                "support_count": len(support_patches),
                "support_frames": support_frames,
                "search_crop_size": args.search_crop_size,
                "query_crop_size": args.query_crop_size,
                "image_H": int(orig_h),
                "image_W": int(orig_w),
                "file": f"{vid_name}/{sample_file}",
            }

            seq_dir = output_dir / vid_name
            seq_dir.mkdir(parents=True, exist_ok=True)

            save_dict = {
                "support_descriptor": supp_desc.astype(np.float32),
                "search_feature_map": search_fmap.astype(np.float32),
                "search_crop": search_crop.astype(np.uint8),
                "support_patches": support_patches_np.astype(np.uint8),
            }

            np.savez_compressed(seq_dir / sample_file, **save_dict)
            all_samples.append(sample)
            total_queries += 1

        print(f"  Video {video_id}: +{sum(1 for s in all_samples if s['video_name']==video_id)} samples (total={len(all_samples)})", flush=True)

    # Split by video
    unique_videos = sorted(set(s["video_name"] for s in all_samples))
    if val_videos:
        train_vids = {v for v in unique_videos if v not in val_videos}
        actual_val_vids = {v for v in unique_videos if v in val_videos}
    else:
        actual_val_vids = {unique_videos[-1]} if unique_videos else set()
        train_vids = set(unique_videos[:-1])

    train_idx = [s for s in all_samples if s["video_name"] in train_vids]
    val_idx = [s for s in all_samples if s["video_name"] in actual_val_vids]

    (output_dir / "index_all.json").write_text(json.dumps(all_samples, indent=2) + "\n")
    (output_dir / "index_train.json").write_text(json.dumps(train_idx, indent=2) + "\n")
    (output_dir / "index_val.json").write_text(json.dumps(val_idx, indent=2) + "\n")

    # Stats
    raw_errors = np.array([s["raw_ct_error_px"] for s in all_samples])
    gt_inside_flags = np.array([s["gt_inside"] for s in all_samples])
    long_occ_mask = np.array([s["occ_length"] >= 20 for s in all_samples])

    stats = {
        "total_samples": len(all_samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_videos": len(unique_videos),
        "train_videos": sorted(train_vids),
        "val_videos": sorted(actual_val_vids),
        "local_radius_px": args.local_radius_px,
        "search_crop_size": args.search_crop_size,
        "config": vars(args),
        "baseline_metrics": {
            "n": int(len(raw_errors)),
            "raw_ct_median_px": round(float(np.median(raw_errors)), 2),
            "raw_ct_lt4px": round(float(np.mean(raw_errors < 4)), 4),
            "raw_ct_lt8px": round(float(np.mean(raw_errors < 8)), 4),
            "gt_inside_pct": round(float(gt_inside_flags.mean()), 4),
            "long_occ_n": int(long_occ_mask.sum()),
            "long_occ_raw_ct_lt4px": round(float(np.mean(raw_errors[long_occ_mask] < 4)) if long_occ_mask.any() else 0, 4),
            "long_occ_raw_ct_median_px": round(float(np.median(raw_errors[long_occ_mask])) if long_occ_mask.any() else 0, 2),
        },
    }

    # Oracle metrics (within search crop)
    # Oracle = GT is inside search crop, so oracle can always find it within the crop
    gt_inside_ratio = gt_inside_flags.mean()
    long_occ_inside = np.mean([s["gt_inside"] for s_idx, s in enumerate(all_samples) if long_occ_mask[s_idx]]) if long_occ_mask.any() else 0

    stats["oracle_metrics"] = {
        f"radius_{args.local_radius_px}px_gt_inside_frac": round(float(gt_inside_ratio), 4),
        f"radius_{args.local_radius_px}px_long_occ_gt_inside": round(float(long_occ_inside), 4),
        "note": "Oracle best-of-K within search crop: gt_inside_frac = theoretical max <4px",
    }

    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\n{'='*50}")
    print(f"Dataset: {len(all_samples)} samples, {len(unique_videos)} videos")
    print(f"  Train: {len(train_idx)} ({sorted(train_vids)})")
    print(f"  Val:   {len(val_idx)} ({sorted(actual_val_vids)})")
    print(f"  Raw CT-offline median: {np.median(raw_errors):.1f}px")
    print(f"  Raw CT-offline <4px: {np.mean(raw_errors < 4)*100:.1f}%")
    print(f"  GT inside search crop ({args.local_radius_px}px): {gt_inside_flags.mean()*100:.1f}%")
    print(f"  Long-occ raw CT <4px: {np.mean(raw_errors[long_occ_mask] < 4)*100:.1f}%")
    print(f"  Long-occ GT inside ({args.local_radius_px}px): {long_occ_inside*100:.1f}%")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
