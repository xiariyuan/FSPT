#!/usr/bin/env python3
"""
Build patch verifier dataset by extracting real RGB patches for each candidate.

Reads the existing selector dataset (with candidate positions), loads video
frames, extracts patches at each candidate location.

Usage:
  python scripts/build_local_patch_verifier_dataset.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --selector-jsonl outputs/local_selector_dataset_v3_from_anchor_smoke/val.jsonl \
    --output-dir outputs/local_patch_verifier_dataset \
    --crop-radius 64 --max-batches 30
"""

from __future__ import annotations
import argparse, json, sys, signal
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_base_centered_local_recovery import (
    load_config, _TimeoutError, _timeout_handler,
    find_t_last_visible, precompute_frame_dino_features, local_search,
)
from utils.crop_utils import extract_crop
from models.recovery_features import DINORecoveryExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--selector-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--crop-radius", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load selector dataset
    selector_samples = [json.loads(l) for l in open(args.selector_jsonl) if l.strip()]
    # Group by video_name
    sel_by_video = {}
    for s in selector_samples:
        vid = s["video_name"]
        if vid not in sel_by_video:
            sel_by_video[vid] = []
        sel_by_video[vid].append(s)

    print(f"Selector dataset: {len(selector_samples)} samples across {len(sel_by_video)} videos")

    # Val loader
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    r = args.crop_radius
    patch_size = r * 2  # 128 for r=64
    query_patch_size = 64  # smaller patch for query/support

    import time as _time
    results = []
    matched = 0

    # Pre-load all video frames by video name
    print("Loading video frames...")
    video_frames = {}  # video_name -> (T, H, W, C) uint8
    video_meta = {}    # video_name -> (orig_h, orig_w)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue
        video_name = batch.get("video_name", ["unknown"])
        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)
        if vid_name not in sel_by_video:
            continue
        video = batch.get("video")
        if video is None:
            continue
        if video.dim() == 4: video = video.unsqueeze(0)
        video_np = (video[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        video_frames[vid_name] = video_np
        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            video_meta[vid_name] = (int(orig_size[0, 0].item()), int(orig_size[0, 1].item()))
        else:
            video_meta[vid_name] = (video_np.shape[1], video_np.shape[2])
        print(f"  Loaded {vid_name}: T={video_np.shape[0]}, H={video_np.shape[1]}, W={video_np.shape[2]}")

    print(f"Loaded {len(video_frames)} videos")

    # Process each selector entry
    for s in selector_samples:
        vid = s["video_name"]
        if vid not in video_frames:
            continue

        vn_np = video_frames[vid]
        orig_h, orig_w = video_meta[vid]
        T = vn_np.shape[0]

        t_reentry = s["t_reentry"]
        t_last_vis = s.get("t_last_visible", t_reentry)
        if t_reentry >= T or t_last_vis >= T:
            continue

        base_norm = np.array(s["base_xy_norm"], dtype=np.float32)
        gt_norm = np.array(s["gt_xy_norm"], dtype=np.float32)
        base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h], dtype=np.float32)
        gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h], dtype=np.float32)

        # Extract patches
        frame_re = vn_np[t_reentry]
        frame_support = vn_np[t_last_vis]

        # Search crop around base in reentry frame
        base_patch = extract_crop(frame_re, base_px, patch_size)

        # Query patch: use base position at last visible (what tracker thinks)
        base_last = np.array(s.get("base_xy_norm", base_norm), dtype=np.float32)
        # For query patch, use the point's appearance at last visible frame
        # Use base position at last visible as approximation
        query_px = np.array([base_last[0] * orig_w, base_last[1] * orig_h], dtype=np.float32)
        query_patch = extract_crop(frame_support, query_px, query_patch_size)

        # Candidate patches from selector dataset's candidates
        cands = s.get("candidates", [])
        cand_patches = []
        for c in cands:
            cn = np.array(c["cand_xy_norm"], dtype=np.float32)
            cp = np.array([cn[0] * orig_w, cn[1] * orig_h], dtype=np.float32)
            cand_patch = extract_crop(frame_re, cp, patch_size)
            cand_patches.append(cand_patch)

        if not cand_patches:
            continue

        # Save
        entry = {
            "sample_id": len(results),
            "video_name": vid,
            "point_idx": s["point_idx"],
            "t_reentry": t_reentry,
            "t_last_visible": t_last_vis,
            "base_xy_norm": base_norm.tolist(),
            "gt_xy_norm": gt_norm.tolist(),
            "base_error_px": s["base_error_px"],
            "crop_radius": r,
            "patch_size": patch_size,
            "query_patch_size": query_patch_size,
            "has_positive_candidate": s.get("has_positive_candidate", False),
            "oracle_index": s.get("oracle_index", -1),
            "oracle_error_px": s.get("oracle_error_px", 999),
            "candidates": cands,
            "label": s.get("label", -1),
        }

        npz_name = f"{vid}_n{s['point_idx']}_t{t_reentry}.npz"
        npz_path = out_dir / "patches" / npz_name
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz_path),
            base_patch=base_patch.astype(np.uint8),
            query_patch=query_patch.astype(np.uint8),
            cand_patches=np.stack(cand_patches, axis=0).astype(np.uint8),
        )

        entry["npz_file"] = f"patches/{npz_name}"
        entry["base_patch_nnz"] = int(np.count_nonzero(base_patch))
        entry["query_patch_nnz"] = int(np.count_nonzero(query_patch))
        entry["cand_patches_nnz"] = [int(np.count_nonzero(cp)) for cp in cand_patches]

        results.append(entry)
        matched += 1

    # Save index
    with open(out_dir / "index.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Patch Verifier Dataset")
    print(f"{'='*60}")
    print(f"  n={len(results)}, r={r}, patch_size={patch_size}")

    # Check patch quality
    nnz_base = [e["base_patch_nnz"] for e in results]
    nnz_query = [e["query_patch_nnz"] for e in results]
    print(f"  base_patch nnz: min={min(nnz_base)}, max={max(nnz_base)}, median={sorted(nnz_base)[len(nnz_base)//2]}")
    print(f"  query_patch nnz: min={min(nnz_query)}, max={max(nnz_query)}, median={sorted(nnz_query)[len(nnz_query)//2]}")

    n_zero = sum(1 for n in nnz_base if n == 0)
    print(f"  zero base patches: {n_zero}/{len(results)}")

    pos = [e for e in results if e.get("has_positive_candidate")]
    print(f"  positive events: {len(pos)}")

    # Save contact sheet for 20 random samples
    try:
        import cv2
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(results), min(20, len(results)), replace=False)
        for idx, si in enumerate(sample_indices):
            e = results[si]
            npz = np.load(str(out_dir / e["npz_file"]))
            bp = npz["base_patch"]
            qp = npz["query_patch"]
            # Concatenate base + query patches
            h = max(bp.shape[0], qp.shape[0])
            canvas = np.zeros((h, bp.shape[1] + qp.shape[1] + 10, 3), dtype=np.uint8)
            canvas[:bp.shape[0], :bp.shape[1]] = bp
            canvas[:qp.shape[0], bp.shape[1]+10:] = qp
            cv2.imwrite(str(out_dir / f"debug_patch_{idx:02d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  Saved {len(sample_indices)} debug patch images")
    except Exception as ex:
        print(f"  Debug images failed: {ex}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
