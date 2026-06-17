#!/usr/bin/env python3
"""
Stage 2b: Dense support-memory readout for PRT re-entry.

Instead of discrete top-k candidates, this script:
  1. Creates a dense score map around the baseline prior using support memory
  2. Evaluates oracle (best pixel) vs top-1 vs top-1+dustbin
  3. No training required - pure inference-time feature matching

Hypothesis: dense readout should have a better score distribution than
discrete top-k because it doesn't lose spatial structure.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage0 import (
    discover_sequences,
    find_reentry_queries,
    load_sequence,
    project_3d_to_2d,
)
from scripts.eval_world_state_stage2_causal_dino import (
    DINOFeatureExtractor,
    extract_crop,
    load_rgb_frame,
)
from scripts.build_prt_splits import load_image_size
from scripts.eval_multi_support_dino import extract_support_features, score_candidate_with_support


def dense_support_score_map(
    extractor: DINOFeatureExtractor,
    support_features: torch.Tensor,
    reentry_img: np.ndarray,
    center_xy: np.ndarray,
    window_size: int = 64,
    stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dense support-memory similarity map in a local window around center.

    Returns:
        score_map: (H, W) cosine similarity scores
        grid_yx: (H*W, 2) grid positions in original image coords
        scores_flat: (H*W,) flattened scores
    """
    half = window_size // 2
    H_img, W_img = reentry_img.shape[:2]
    cy, cx = float(center_xy[0]), float(center_xy[1])

    # Build grid of candidate positions
    y_coords = np.arange(cy - half, cy + half, stride)
    x_coords = np.arange(cx - half, cx + half, stride)
    y_coords = np.clip(y_coords, 0, H_img - 1)
    x_coords = np.clip(x_coords, 0, W_img - 1)

    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    grid_yx = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)

    # Score each grid position against support memory
    scores = []
    for pos in grid_yx:
        patch = extract_crop(reentry_img, pos, 64)
        feat = extractor.feature_map(patch)
        score = score_candidate_with_support(support_features, feat, method="mean")
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)
    score_map = scores.reshape(len(y_coords), len(x_coords))
    return score_map, grid_yx, scores


def main():
    parser = argparse.ArgumentParser(description="Dense support-memory readout")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dustbin-thresholds", type=str, default="-0.3,-0.2,-0.1,0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-json", type=str,
                        default="/gemini/code/FSPT/outputs/dense_readout_smoke.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)
    rng = np.random.default_rng(42)

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sequences = discover_sequences(data_root, splits)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]

    dustbin_thresholds = [float(x) for x in args.dustbin_thresholds.split(",") if x.strip()]

    results = []
    for seq_path in sequences:
        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        trajs_3d = seq["trajs_3d"]
        visibs = seq["visibs"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]
        extr_inv = np.linalg.inv(extrinsics)

        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        filtered = []
        for q in queries:
            e_rel = extrinsics[q.reentry_frame] @ extr_inv[q.query_frame]
            cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
            if cam_motion >= args.min_camera_motion:
                filtered.append((q, cam_motion))

        collected = 0
        for q, cam_motion in filtered:
            if collected >= args.max_samples:
                break

            t_q, t_re, i = q.query_frame, q.reentry_frame, q.point_idx

            # Extract support memory
            support_feat, support_frames = extract_support_features(
                extractor, seq_path, i, t_q, trajs_2d, visibs,
                num_support=args.num_support, patch_size=64,
            )
            if support_feat is None or len(support_frames) < 2:
                continue

            # Get baseline prediction
            hold_3d = trajs_3d[t_q, i].astype(np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if not np.isfinite(z_depth) or z_depth <= 1e-6:
                continue

            eps = float(np.clip(rng.normal(0.0, 0.10), -0.35, 0.35))
            noisy_depth = z_depth * math.exp(eps)
            pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
            k_inv = np.linalg.inv(intrinsics[t_q])
            pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
            e_inv = np.linalg.inv(extrinsics[t_q])
            noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
            baseline_xy = project_3d_to_2d(noisy_world, intrinsics[t_re], extrinsics[t_re]).astype(np.float32)

            gt_xy = trajs_2d[t_re, i].astype(np.float32)
            baseline_err = float(np.linalg.norm(baseline_xy - gt_xy))

            reentry_img = load_rgb_frame(seq_path, t_re)
            if reentry_img is None:
                continue

            # Dense readout
            score_map, grid_yx, scores_flat = dense_support_score_map(
                extractor, support_feat, reentry_img, baseline_xy,
                window_size=args.window_size, stride=args.stride,
            )

            # Oracle: best position in grid
            best_idx = int(np.argmax(scores_flat))
            oracle_xy = grid_yx[best_idx]
            oracle_err = float(np.linalg.norm(oracle_xy - gt_xy))

            # Top-1: best score position
            top1_xy = oracle_xy.copy()
            top1_err = oracle_err

            # Also compute distance from each grid point to GT
            gt_errors = np.linalg.norm(grid_yx - gt_xy[None, :], axis=1)

            # Score-quality correlation
            from sklearn.metrics import roc_auc_score
            # Is high score correlated with low error?
            median_err = np.median(gt_errors)
            low_err_labels = (gt_errors < median_err).astype(int)
            try:
                score_auc = float(roc_auc_score(low_err_labels, scores_flat))
            except:
                score_auc = 0.5

            # Dustbin analysis: reject top-1 if its score is below threshold
            top1_score = float(scores_flat[best_idx])
            mean_score = float(scores_flat.mean())
            std_score = float(scores_flat.std())
            normalized_top1 = (top1_score - mean_score) / (std_score + 1e-6)

            dustbin_results = {}
            for thr in dustbin_thresholds:
                if normalized_top1 >= thr:
                    dustbin_err = top1_err
                    accepted = True
                else:
                    dustbin_err = baseline_err
                    accepted = False
                dustbin_results[f"thr_{thr:.3f}"] = {
                    "err": float(dustbin_err),
                    "accepted": accepted,
                }

            results.append({
                "seq": seq_path.name,
                "baseline_err": baseline_err,
                "oracle_err": oracle_err,
                "top1_err": top1_err,
                "top1_score": top1_score,
                "normalized_top1": normalized_top1,
                "score_auc": score_auc,
                "occ_length": int(q.occ_length),
                "camera_motion": cam_motion,
                "grid_size": len(scores_flat),
                "dustbin": dustbin_results,
            })
            collected += 1

    # Analysis
    print(f"\nTotal samples: {len(results)}", flush=True)
    if not results:
        print("No results.", flush=True)
        return

    baseline_errs = np.array([r["baseline_err"] for r in results])
    oracle_errs = np.array([r["oracle_err"] for r in results])
    top1_errs = np.array([r["top1_err"] for r in results])
    score_aucs = np.array([r["score_auc"] for r in results])

    print(f"\n=== Dense Readout Summary ===")
    print(f"  Baseline median:     {np.median(baseline_errs):.2f}")
    print(f"  Oracle (dense):      {np.median(oracle_errs):.2f}")
    print(f"  Top-1 (dense):       {np.median(top1_errs):.2f}")
    print(f"  Oracle better frac:  {np.mean(oracle_errs < baseline_errs):.2%}")
    print(f"  Top-1 better frac:   {np.mean(top1_errs < baseline_errs):.2%}")
    print(f"  Score AUC (median):  {np.median(score_aucs):.4f}")

    # Dustbin analysis
    print(f"\n=== Dustbin Analysis ===")
    print(f"  {'threshold':>12} {'coverage':>10} {'median_err':>12} {'better_frac':>12} {'<4px':>8}")
    for thr_key in sorted(results[0]["dustbin"].keys()):
        errs = []
        accepted_count = 0
        for r in results:
            d = r["dustbin"][thr_key]
            errs.append(d["err"])
            if d["accepted"]:
                accepted_count += 1
        errs = np.array(errs)
        coverage = accepted_count / len(results)
        if coverage > 0:
            accepted_errs = np.array([r["dustbin"][thr_key]["err"] for r in results if r["dustbin"][thr_key]["accepted"]])
            accept_median = float(np.median(accepted_errs))
        else:
            accept_median = float('nan')
        print(f"  {thr_key:>12} {coverage:>10.1%} {np.median(errs):>12.2f} {np.mean(errs < baseline_errs):>12.2%} {np.mean(errs < 4):>7.1%}")

    # Save
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args),
        "n": len(results),
        "summary": {
            "baseline_median": float(np.median(baseline_errs)),
            "oracle_dense_median": float(np.median(oracle_errs)),
            "top1_dense_median": float(np.median(top1_errs)),
            "score_auc_median": float(np.median(score_aucs)),
        },
        "results": results,
    }, indent=2, default=str) + "\n")
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
