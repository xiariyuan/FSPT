#!/usr/bin/env python3
"""Build CT-offline-centered local grid verifier dataset.

For each re-entry query:
  - CT-offline prediction at re-entry = center
  - Local grid: 8px radius / 2px stride (~37 candidates)
  - Extract 64×64 DINO patch embedding for each candidate
  - Extract 64×64 DINO patch embedding for last-visible support
  - Label: oracle candidate (closest to GT) = positive

Used by a lightweight ranker (MLP on pairwise DINO features).
"""
from __future__ import annotations

import argparse, json, sys, pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from fspt.paths import repo_root, resolve_repo_path

PROJECT_ROOT = repo_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.crop_utils import extract_crop
from utils.attempt0_schema import load_attempt0_cache


def find_first_reentry(gt_visibility: np.ndarray, query_t: int):
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
            return t, occ_len, last_visible_t
    return None


def yx_norm_to_xy_px(yx: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array([
        float(yx[1]) * max(float(w) - 1, 1),
        float(yx[0]) * max(float(h) - 1, 1),
    ], dtype=np.float32)


def encode_patch_dino(dino_model, patch_uint8: np.ndarray, device) -> np.ndarray:
    """Encode single (H,W,3) uint8 patch to (384,) DINO embedding."""
    t = torch.from_numpy(patch_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        feat = dino_model(t)[-1].mean(dim=[-2, -1]).float()
        feat = F.normalize(feat, dim=-1)
    return feat[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-offline-cache", type=str,
                        default="outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
    parser.add_argument("--gt-cache", type=str, default="caches/trackon2_strided_original.pt")
    parser.add_argument("--pkl-path", type=str, default=str(resolve_repo_path("..", "datasets", "tapvid_davis", "tapvid_davis.pkl")))
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=128)
    parser.add_argument("--radius-px", type=int, default=8)
    parser.add_argument("--stride-px", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DINO
    from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
    dino = DINOFeatureExtractor(
        resolve_repo_path("weights", "dinov2", "dinov2_vits14_pretrain.pth"), device
    )
    dino_model = dino.model.eval()

    # Load caches
    ct_off_cache = load_attempt0_cache(Path(args.ct_offline_cache))
    gt_cache = load_attempt0_cache(Path(args.gt_cache))
    ct_off_records = ct_off_cache["records"][:args.max_videos]
    gt_records = gt_cache["records"][:args.max_videos]

    with open(args.pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    all_samples: List[Dict[str, Any]] = []
    total_queries = 0
    stop = False

    for rec_idx in range(len(gt_records)):
        if stop:
            break
        ct_r = ct_off_records[rec_idx]
        gt_r = gt_records[rec_idx]
        video_id = str(gt_r["video_id"])

        gt_vis = np.asarray(gt_r["gt_visibility"], dtype=bool)
        gt_tracks = np.asarray(gt_r["gt_tracks"], dtype=np.float32)
        qpts = np.asarray(gt_r["query_points"], dtype=np.float32)
        ct_pred = np.asarray(ct_r["pred_tracks"], dtype=np.float32)
        orig_h = int(gt_r["original_size"][0])
        orig_w = int(gt_r["original_size"][1])

        video_rgb = np.asarray(pkl_data[video_id]["video"], dtype=np.uint8)
        frame_cache: Dict[int, np.ndarray] = {}

        def get_frame(t):
            if t not in frame_cache:
                frame_cache[t] = video_rgb[t]
            return frame_cache[t]

        for qi in range(qpts.shape[0]):
            if args.max_queries > 0 and total_queries >= args.max_queries:
                stop = True
                break

            qt = int(round(qpts[qi, 0]))
            re_info = find_first_reentry(gt_vis[qi], qt)
            if re_info is None:
                continue
            reentry_t, occ_len, last_visible_t = re_info

            gt_yx = gt_tracks[qi, reentry_t]
            gt_xy = yx_norm_to_xy_px(gt_yx, orig_h, orig_w)
            ct_yx = ct_pred[qi, reentry_t]
            ct_xy = yx_norm_to_xy_px(ct_yx, orig_h, orig_w)
            raw_ct_error = float(np.linalg.norm(ct_xy - gt_xy))

            # Support patch (last visible frame before occlusion)
            last_vis_yx = gt_tracks[qi, last_visible_t]
            last_vis_xy = yx_norm_to_xy_px(last_vis_yx, orig_h, orig_w)
            support_frame = get_frame(last_visible_t)
            support_patch = extract_crop(support_frame, last_vis_xy, args.patch_size)

            # Re-entry frame
            re_frame = get_frame(reentry_t)

            # Generate local grid candidates
            r = args.radius_px
            s = args.stride_px
            candidates = []
            for dy in range(-r, r + 1, s):
                for dx in range(-r, r + 1, s):
                    cand_px = ct_xy + np.array([dx, dy], dtype=np.float32)
                    cand_px[0] = max(0, min(cand_px[0], orig_w - 1))
                    cand_px[1] = max(0, min(cand_px[1], orig_h - 1))
                    cand_patch = extract_crop(re_frame, cand_px, args.patch_size)
                    err = float(np.linalg.norm(cand_px - gt_xy))
                    candidates.append({
                        "xy": cand_px.tolist(),
                        "error_px": err,
                        "patch": cand_patch,
                    })

            K = len(candidates)
            if K == 0:
                continue

            # Encode patches with DINO
            support_emb = encode_patch_dino(dino_model, support_patch, device)
            cand_embs = []
            for c in candidates:
                emb = encode_patch_dino(dino_model, c["patch"], device)
                cand_embs.append(emb)

            # Oracle candidate = closest to GT
            best_idx = int(np.argmin([c["error_px"] for c in candidates]))
            oracle_error = candidates[best_idx]["error_px"]

            # CT-offline patch embedding (patch at CT-offline prediction on re-entry frame)
            ct_patch = extract_crop(re_frame, ct_xy, args.patch_size)
            ct_emb = encode_patch_dino(dino_model, ct_patch, device)

            # Build features for each candidate
            feat_dim = 4  # cos_support, cos_ct, l2_support, l2_ct
            cand_features = np.zeros((K, feat_dim), dtype=np.float32)
            labels = np.zeros(K, dtype=np.float32)
            labels[best_idx] = 1.0

            for j, (ce, c) in enumerate(zip(cand_embs, candidates)):
                cos_supp = float(np.dot(support_emb, ce))
                cos_ct = float(np.dot(ct_emb, ce))
                l2_supp = float(np.linalg.norm(support_emb - ce))
                l2_ct = float(np.linalg.norm(ct_emb - ce))
                cand_features[j] = [cos_supp, cos_ct, l2_supp, l2_ct]

            sample = {
                "video_name": video_id,
                "point_idx": int(qi),
                "t_reentry": int(reentry_t),
                "occ_length": int(occ_len),
                "raw_ct_error_px": round(raw_ct_error, 2),
                "oracle_best_error_px": round(oracle_error, 2),
                "oracle_lt4px": int(oracle_error < 4),
                "radius_px": r,
                "grid_stride_px": s,
                "n_candidates": K,
                "gt_xy": gt_xy.tolist(),
                "ct_xy": ct_xy.tolist(),
                "support_emb": support_emb.tolist(),
                "ct_emb": ct_emb.tolist(),
                "cand_features": cand_features.tolist(),
                "cand_errors_px": [c["error_px"] for c in candidates],
                "labels": labels.tolist(),
                "cand_xy": [c["xy"] for c in candidates],
            }
            all_samples.append(sample)
            total_queries += 1

            if total_queries % 20 == 0:
                print(f"  {total_queries} samples...", flush=True)

    # Stats
    raw_errors = np.array([s["raw_ct_error_px"] for s in all_samples])
    oracle_errors = np.array([s["oracle_best_error_px"] for s in all_samples])
    long_mask = np.array([s["occ_length"] >= 20 for s in all_samples])

    stats = {
        "n": len(all_samples),
        "radius_px": args.radius_px,
        "n_candidates_per_query": K,
        "raw_ct_median_px": round(float(np.median(raw_errors)), 2),
        "raw_ct_lt4px": round(float(np.mean(raw_errors < 4)), 4),
        "raw_ct_lt8px": round(float(np.mean(raw_errors < 8)), 4),
        "oracle_median_px": round(float(np.median(oracle_errors)), 2),
        "oracle_lt4px": round(float(np.mean(oracle_errors < 4)), 4),
        "lt4px_improvement_pp": round(float(np.mean(oracle_errors < 4) - np.mean(raw_errors < 4)), 4),
        "long_occ_n": int(long_mask.sum()),
        "long_occ_raw_lt4px": round(float(np.mean(raw_errors[long_mask] < 4)) if long_mask.any() else 0, 4),
        "long_occ_oracle_lt4px": round(float(np.mean(oracle_errors[long_mask] < 4)) if long_mask.any() else 0, 4),
    }

    out_path = out_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump({"samples": all_samples, "stats": stats}, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset: {len(all_samples)} samples, {K} candidates/query")
    print(f"Raw CT median: {np.median(raw_errors):.1f}px, <4px: {np.mean(raw_errors<4)*100:.1f}%")
    print(f"Oracle best median: {np.median(oracle_errors):.1f}px, <4px: {np.mean(oracle_errors<4)*100:.1f}%")
    print(f"Oracle <4px improvement: {np.mean(oracle_errors<4)*100 - np.mean(raw_errors<4)*100:+.1f}pp")
    print(f"Long-occ oracle <4px: {np.mean(oracle_errors[long_mask]<4)*100:.1f}%")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
