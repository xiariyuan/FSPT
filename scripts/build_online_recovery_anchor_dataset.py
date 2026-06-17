#!/usr/bin/env python3
"""
Build offline recovery-anchor dataset for training the geometric recovery head.

For each re-entry event in PointOdyssey:
  1. Run CoTracker base tracker through the sequence
  2. At each re-entry frame, extract:
     - base tracker position (anchor)
     - GT position
     - DINO feature map of search region
     - support memory descriptors from pre-occlusion frames
     - tracker metadata (visibility, confidence, occ_length)
  3. Save per-sequence shards + global index

Usage:
  python scripts/build_online_recovery_anchor_dataset.py \
    --data-root /gemini/code/FSPT/datasets/pointodyssey \
    --splits val \
    --max-sequences 2 \
    --max-samples 50 \
    --output-dir outputs/recovery_anchor_dataset_smoke
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage0 import discover_sequences, load_sequence, project_3d_to_2d
from scripts.build_prt_splits import load_image_size, classify_reentry_type


def find_reentry_events(
    seq_path: Path,
    min_occ_length: int,
    min_camera_motion: float,
    max_events: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Find re-entry events in a sequence using GT trajectories."""
    seq = load_sequence(seq_path)
    trajs_2d = seq["trajs_2d"]
    trajs_3d = seq["trajs_3d"]
    visibs = seq["visibs"]
    intrinsics = seq["intrinsics"]
    extrinsics = seq["extrinsics"]
    height, width = load_image_size(seq_path)

    t_total, n_points = visibs.shape
    extr_inv = np.linalg.inv(extrinsics)
    events = []

    for i in range(n_points):
        vis = visibs[:, i].astype(bool)
        if not vis.any():
            continue
        run_start = -1
        last_vis_before = -1
        for t in range(t_total):
            if vis[t]:
                if run_start >= 0 and (t - run_start) >= min_occ_length and last_vis_before >= 0:
                    e_rel = extrinsics[t] @ extr_inv[last_vis_before]
                    cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                    if cam_motion < min_camera_motion:
                        last_vis_before = t
                        run_start = -1
                        continue

                    occ_length = t - run_start
                    events.append({
                        "point_idx": i,
                        "query_frame": last_vis_before,
                        "reentry_frame": t,
                        "occ_length": occ_length,
                        "camera_motion": cam_motion,
                        "gt_xy": trajs_2d[t, i].astype(np.float32).tolist(),
                        "query_xy": trajs_2d[last_vis_before, i].astype(np.float32).tolist(),
                        "hold_3d": trajs_3d[last_vis_before, i].astype(np.float32).tolist(),
                    })
                last_vis_before = t
                run_start = -1
            else:
                if run_start < 0:
                    run_start = t

    rng.shuffle(events)
    return events[:max_events]


def load_rgb_frame(seq_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Load RGB frame from PointOdyssey sequence directory."""
    rgb_dir = seq_path / "rgbs"
    if not rgb_dir.is_dir():
        rgb_dir = seq_path / "rgb"
    if not rgb_dir.is_dir():
        return None
    # Try rgbs_NNNNN.jpg pattern first, then fallback to sorted glob
    frame_file = rgb_dir / f"rgb_{frame_idx:05d}.jpg"
    if not frame_file.exists():
        frame_file = rgb_dir / f"rgb_{frame_idx:05d}.png"
    if not frame_file.exists():
        # Fallback: sorted glob
        frames = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        if frame_idx < 0 or frame_idx >= len(frames):
            return None
        frame_file = frames[frame_idx]
    img = cv2.imread(str(frame_file))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_crop(img: np.ndarray, center_xy: np.ndarray, crop_size: int) -> np.ndarray:
    """Extract a square crop around center_xy."""
    h, w = img.shape[:2]
    half = crop_size // 2
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    src = img[y0:y1, x0:x1]
    dy, dx = src.shape[:2]
    crop[:dy, :dx] = src
    return crop


def main():
    parser = argparse.ArgumentParser(description="Build recovery-anchor dataset")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = discover_sequences(Path(args.data_root), args.splits.split(","))
    if args.max_sequences > 0:
        seqs = seqs[:args.max_sequences]

    print(f"Building recovery-anchor dataset: {len(seqs)} sequences, max {args.max_samples} samples each")

    global_index = {"train": [], "val": []}
    total_samples = 0

    for seq_idx, seq_path in enumerate(seqs):
        seq_name = seq_path.name
        print(f"\n[{seq_idx+1}/{len(seqs)}] Processing {seq_name}...")

        events = find_reentry_events(
            seq_path,
            min_occ_length=args.min_occ_length,
            min_camera_motion=args.min_camera_motion,
            max_events=args.max_samples,
            rng=rng,
        )
        print(f"  Found {len(events)} re-entry events")

        if len(events) == 0:
            continue

        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        visibs = seq["visibs"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]
        seq_output = output_dir / seq_name
        seq_output.mkdir(parents=True, exist_ok=True)

        samples = []
        for evt_idx, evt in enumerate(events):
            i = evt["point_idx"]
            t_q = evt["query_frame"]
            t_re = evt["reentry_frame"]

            # Load frames
            query_img = load_rgb_frame(seq_path, t_q)
            reentry_img = load_rgb_frame(seq_path, t_re)
            if query_img is None or reentry_img is None:
                continue

            # Extract crops
            query_xy = np.array(evt["query_xy"], dtype=np.float32)
            gt_xy = np.array(evt["gt_xy"], dtype=np.float32)

            # Noisy baseline (same as PRT pipeline)
            hold_3d = np.array(evt["hold_3d"], dtype=np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if not np.isfinite(z_depth) or z_depth <= 1e-6:
                continue

            eps = float(np.clip(rng.normal(0.0, 0.10), -0.35, 0.35))
            noisy_depth = z_depth * np.exp(eps)
            pixels_h = np.array([query_xy[0], query_xy[1], 1.0], dtype=np.float32)
            k_inv = np.linalg.inv(intrinsics[t_q])
            pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
            e_inv = np.linalg.inv(extrinsics[t_q])
            noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
            base_xy = project_3d_to_2d(noisy_world, intrinsics[t_re], extrinsics[t_re]).astype(np.float32)

            base_err = float(np.linalg.norm(base_xy - gt_xy))

            # Extract search crop around baseline position
            search_crop = extract_crop(reentry_img, base_xy, args.search_crop_size)

            # Extract query crop
            query_crop = extract_crop(query_img, query_xy, args.query_crop_size)

            # Extract support patches (last M visible frames before occlusion)
            vis = visibs[:, i].astype(bool)
            support_frames = []
            for t in range(t_q, -1, -1):
                if vis[t]:
                    support_frames.append(t)
                if len(support_frames) >= args.num_support:
                    break
            support_frames.reverse()

            support_patches = []
            for t in support_frames:
                img = load_rgb_frame(seq_path, t)
                if img is None:
                    continue
                xy = trajs_2d[t, i].astype(np.float32)
                if not np.all(np.isfinite(xy)):
                    continue
                patch = extract_crop(img, xy, args.query_crop_size)
                support_patches.append(patch)

            if len(support_patches) == 0:
                continue

            # Save sample
            sample = {
                "sample_id": total_samples,
                "seq_name": seq_name,
                "point_idx": i,
                "query_frame": t_q,
                "reentry_frame": t_re,
                "occ_length": evt["occ_length"],
                "camera_motion": round(evt["camera_motion"], 4),
                "base_xy": base_xy.tolist(),
                "gt_xy": gt_xy.tolist(),
                "base_error_px": round(base_err, 2),
                "support_count": len(support_patches),
                "support_frames": support_frames,
                "search_crop_size": args.search_crop_size,
                "query_crop_size": args.query_crop_size,
            }

            # Save tensors as npz
            tensors = {
                "search_crop": search_crop.astype(np.uint8),
                "query_crop": query_crop.astype(np.uint8),
                "support_patches": np.stack(support_patches, axis=0).astype(np.uint8),
            }
            np.savez_compressed(seq_output / f"sample_{evt_idx:04d}.npz", **tensors)
            samples.append(sample)
            total_samples += 1

            if (evt_idx + 1) % 10 == 0:
                print(f"  {evt_idx+1}/{len(events)} samples processed")

        # Save index for this sequence
        (seq_output / "index.json").write_text(json.dumps(samples, indent=2) + "\n")

        # Assign to train/val (first seq = val, rest = train)
        split = "val" if seq_idx == 0 else "train"
        for s in samples:
            global_index[split].append({
                "seq_name": s["seq_name"],
                "sample_id": s["sample_id"],
                "file": f"{s['seq_name']}/sample_{samples.index(s):04d}.npz",
                "base_error_px": s["base_error_px"],
                "occ_length": s["occ_length"],
            })
        print(f"  Saved {len(samples)} samples to {seq_output} (split={split})")

    # Save global index
    for split in ["train", "val"]:
        (output_dir / f"index_{split}.json").write_text(
            json.dumps(global_index[split], indent=2) + "\n"
        )

    # Save build stats
    stats = {
        "total_samples": total_samples,
        "n_train": len(global_index["train"]),
        "n_val": len(global_index["val"]),
        "n_sequences": len(seqs),
        "config": vars(args),
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nDone! {total_samples} samples, train={len(global_index['train'])}, val={len(global_index['val'])}")


if __name__ == "__main__":
    main()
