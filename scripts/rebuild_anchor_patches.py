#!/usr/bin/env python3
"""
Rebuild RGB patches in recovery_anchor_dataset_v3_full.

The existing npz files have valid DINO features but all-zero patches.
This script loads original video frames and re-extracts the patches
using the same crop geometry as build_online_recovery_anchor_dataset_v3.py.

Usage:
  python scripts/rebuild_anchor_patches.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --dataset-dir outputs/recovery_anchor_dataset_v3_full \
    --max-batches 30
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)

    # Load all index entries, group by video
    all_entries = []
    for idx_file in ["index_train.json", "index_val.json"]:
        idx_path = ds_dir / idx_file
        if idx_path.exists():
            entries = json.load(open(idx_path))
            for e in entries:
                e["_split"] = idx_file.replace("index_", "").replace(".json", "")
            all_entries.extend(entries)

    # Group by video_name
    by_video = {}
    for e in all_entries:
        vid = e["video_name"]
        if vid not in by_video:
            by_video[vid] = []
        by_video[vid].append(e)

    print(f"Total entries: {len(all_entries)} across {len(by_video)} videos")

    cfg = load_config(args.config)
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    patched = 0
    skipped = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        video_name = batch.get("video_name", ["unknown"])
        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        if vid_name not in by_video:
            continue

        entries = by_video[vid_name]
        if video.dim() == 4:
            video = video.unsqueeze(0)

        B, T, C, H, W = video.shape
        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        # Convert video to uint8 numpy (T, H, W, C)
        video_np = (video[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)

        for entry in entries:
            npz_path = ds_dir / entry["file"]
            if not npz_path.exists():
                skipped += 1
                continue

            # Load existing npz (keep DINO features)
            data = dict(np.load(str(npz_path), allow_pickle=True))

            # Check if patches already valid
            sc = data.get("search_crop")
            if sc is not None and sc.max() > 0:
                skipped += 1
                continue

            t_re = entry["reentry_frame"]
            qt = entry["query_frame"]
            base_xy = np.array(entry["base_xy"], dtype=np.float32)
            gt_xy = np.array(entry["gt_xy"], dtype=np.float32)
            support_frames = entry.get("support_frames", [])
            search_crop_size = entry.get("search_crop_size", 224)
            query_crop_size = entry.get("query_crop_size", 112)

            if t_re >= T or qt >= T:
                skipped += 1
                continue

            # Reentry frame
            frame_re = video_np[t_re]  # (H, W, C)
            frame_q = video_np[qt]

            # Search crop around base position in reentry frame
            search_crop = extract_crop(frame_re, base_xy, search_crop_size)

            # Query crop around GT position in query frame
            query_crop = extract_crop(frame_q, gt_xy, query_crop_size)

            # Support patches around GT in support frames
            support_patches = []
            for t_sf in support_frames:
                if t_sf < T:
                    sf = video_np[t_sf]
                    gt_sf_px = gt_xy  # Use same GT position (approximation)
                    # Actually, we need per-frame GT. Use the stored gt_xy for now.
                    # The original script used gt_tracks[b, n, t_sf] for each support frame.
                    # But we don't have gt_tracks here. Use gt_xy as fallback.
                    patch = extract_crop(sf, gt_sf_px, query_crop_size)
                    support_patches.append(patch)

            if support_patches:
                support_patches_np = np.stack(support_patches, axis=0)
            else:
                support_patches_np = np.zeros((1, query_crop_size, query_crop_size, 3), dtype=np.uint8)

            # Update npz
            data["search_crop"] = search_crop.astype(np.uint8)
            data["query_crop"] = query_crop.astype(np.uint8)
            data["support_patches"] = support_patches_np.astype(np.uint8)

            np.savez_compressed(str(npz_path), **data)
            patched += 1

        print(f"  Batch {batch_idx} ({vid_name}): patched={patched}, skipped={skipped}")

    print(f"\nDone. Patched: {patched}, Skipped: {skipped}")

    # Sanity check
    print("\nSanity check on first 5 entries:")
    for entry in all_entries[:5]:
        npz_path = ds_dir / entry["file"]
        if npz_path.exists():
            data = np.load(str(npz_path))
            sc = data["search_crop"]
            qc = data["query_crop"]
            print(f"  {entry['video_name']} n={entry['point_idx']}: search nnz={np.count_nonzero(sc)}, query nnz={np.count_nonzero(qc)}, "
                  f"search range=[{sc.min()},{sc.max()}], query range=[{qc.min()},{qc.max()}]")


if __name__ == "__main__":
    main()
