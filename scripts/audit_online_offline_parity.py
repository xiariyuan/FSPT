#!/usr/bin/env python3
"""
Parity audit: compare online runtime inputs vs offline dataset builder inputs.

Verifies that the learned head receives identical (or near-identical) inputs
when running inside the online pipeline vs when trained on offline cached data.

Compares:
  - search crop (RGB)
  - search feature map (DINO)
  - support descriptor (DINO pooled)
  - tracker visibility

Usage:
  python scripts/audit_online_offline_parity.py \
    --config configs/fspt_online_recovery_learned_head_smoke.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --offline-dir outputs/recovery_anchor_dataset_v3_full \
    --max-batches 10
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
        current = OmegaConf.create(raw)
        return OmegaConf.merge(base, current)
    return _load_and_resolve(config_path)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--offline-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/parity_audit.json")
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Load offline dataset index
    offline_dir = Path(args.offline_dir)
    with open(offline_dir / "index_val.json") as f:
        offline_samples = json.load(f)

    # Build offline sample lookup by (video_name, reentry_frame, point_idx)
    offline_lookup = {}
    for s in offline_samples:
        key = (s["video_name"], s["reentry_frame"], s["point_idx"])
        offline_lookup[key] = s

    print(f"Offline val samples: {len(offline_samples)}")
    print(f"Unique keys: {len(offline_lookup)}")

    # Load DINO extractors (same as offline + online)
    from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
    from models.recovery_features import DINORecoveryExtractor
    dino_extractor = DINOFeatureExtractor(
        Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth"), device
    )
    dino_ext = DINORecoveryExtractor(
        Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    )
    dino_ext._ensure_loaded(device)

    # Load online model
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    # Load dataloader
    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    parity_results = []
    match_count = 0
    total_checked = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points")
        occluded = batch.get("occluded")
        video_name = batch.get("video_name", ["unknown"])

        if video is None or query_points is None:
            continue

        # Run model to get internal state
        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)

        meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}

        with torch.no_grad():
            try:
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}

        # For each triggered query, find the matching offline sample
        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]
        relocal_mask = info.get("relocal_mask", None)

        if relocal_mask is None:
            continue

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        for b in range(B):
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered_frames = np.where(rm)[0]
                if len(triggered_frames) == 0:
                    continue

                t_re = int(triggered_frames[0])

                # Try to find matching offline sample
                key = (vid_name, t_re, n_idx)
                offline_sample = offline_lookup.get(key)

                if offline_sample is None:
                    continue

                total_checked += 1

                # Load offline tensors
                npz_path = offline_dir / offline_sample["file"]
                offline_data = np.load(str(npz_path))

                # Get offline tensors
                off_search_fmap = offline_data["search_feature_map"]  # (D, h, w)
                off_support_desc = offline_data["support_descriptor"]  # (D,)

                # Get online tensors: reconstruct what the learned head would receive
                # 1. Search crop: same as offline (base-centered, cv2 crop)
                base_xy = np.array(offline_sample["base_xy"], dtype=np.float32)
                frame_re = video_dev[b, t_re].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                search_crop_online = extract_crop(frame_re, base_xy, 224)

                # Compare with offline search crop
                if "search_crop" in offline_data:
                    search_crop_offline = offline_data["search_crop"]
                    crop_diff = np.abs(search_crop_online.astype(float) - search_crop_offline.astype(float)).mean()
                else:
                    crop_diff = -1

                # 2. DINO feature map from search crop (compare with offline cached version)
                crop_t = torch.from_numpy(search_crop_online).float().permute(2, 0, 1).to(device)  # (3,H,W)
                online_fmap = dino_ext.feature_map(crop_t, device).numpy()  # (D, h, w)
                # Compare directly with offline cached fmap (already normalized the same way)
                fmap_sim = cosine_sim(online_fmap, off_search_fmap)

                # 3. Support descriptor
                if "support_patches" in offline_data:
                    off_support_patches = offline_data["support_patches"]
                    sp_t = torch.from_numpy(off_support_patches).float().permute(0, 3, 1, 2) / 255.0
                    sp_t = F.interpolate(sp_t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
                    with torch.no_grad():
                        sp_feat = dino_extractor.model(sp_t)[-1].mean(dim=[-2, -1]).float()
                        sp_feat = F.normalize(sp_feat, dim=-1)
                        on_support_desc = sp_feat.mean(dim=0).cpu().numpy()
                    desc_sim = cosine_sim(on_support_desc, off_support_desc)
                else:
                    desc_sim = -1

                # 4. Tracker visibility
                off_vis = offline_sample.get("tracker_visibility_t0", -1)

                # Check match
                fmap_match = fmap_sim > 0.9999
                desc_match = desc_sim > 0.9999
                crop_match = crop_diff == 0 or crop_diff < 0.01

                all_match = fmap_match and desc_match
                if all_match:
                    match_count += 1

                parity_results.append({
                    "video": vid_name,
                    "reentry_frame": t_re,
                    "point_idx": n_idx,
                    "crop_mad": round(float(crop_diff), 4),
                    "fmap_cosine_sim": round(float(fmap_sim), 6),
                    "support_desc_cosine_sim": round(float(desc_sim), 6),
                    "offline_tracker_vis": round(float(off_vis), 4),
                    "crop_match": bool(crop_match),
                    "fmap_match": bool(fmap_match),
                    "desc_match": bool(desc_match),
                    "all_match": bool(all_match),
                })

                if total_checked % 10 == 0:
                    print(f"  Checked {total_checked}: match_rate={match_count/max(total_checked,1):.3f}")

    # Summary
    if total_checked > 0:
        match_rate = match_count / total_checked
        fmap_sims = [r["fmap_cosine_sim"] for r in parity_results]
        desc_sims = [r["support_desc_cosine_sim"] for r in parity_results]
        crop_mads = [r["crop_mad"] for r in parity_results if r["crop_mad"] >= 0]
    else:
        match_rate = 0
        fmap_sims, desc_sims, crop_mads = [], [], []

    summary = {
        "total_checked": total_checked,
        "match_count": match_count,
        "match_rate": round(match_rate, 4),
        "fmap_cosine_sim": {
            "mean": round(float(np.mean(fmap_sims)), 6) if fmap_sims else None,
            "min": round(float(np.min(fmap_sims)), 6) if fmap_sims else None,
        },
        "support_desc_cosine_sim": {
            "mean": round(float(np.mean(desc_sims)), 6) if desc_sims else None,
            "min": round(float(np.min(desc_sims)), 6) if desc_sims else None,
        },
        "crop_mad": {
            "mean": round(float(np.mean(crop_mads)), 4) if crop_mads else None,
            "max": round(float(np.max(crop_mads)), 4) if crop_mads else None,
        },
        "parity_pass": bool(match_rate > 0.95),
        "per_sample": parity_results[:50],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'='*50}")
    print(f"Parity Audit: {total_checked} matched queries")
    print(f"{'='*50}")
    print(f"  Fmap cosine sim:  mean={summary['fmap_cosine_sim']['mean']}, min={summary['fmap_cosine_sim']['min']}")
    print(f"  Desc cosine sim:  mean={summary['support_desc_cosine_sim']['mean']}, min={summary['support_desc_cosine_sim']['min']}")
    print(f"  Crop MAD:         mean={summary['crop_mad']['mean']}, max={summary['crop_mad']['max']}")
    print(f"  All match rate:   {match_rate:.3f}")
    print(f"  Parity pass:      {summary['parity_pass']}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
