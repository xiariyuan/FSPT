#!/usr/bin/env python3
"""
Zero-training multi-support DINO scorer for PRT re-entry.

Hypothesis: Using M pre-occlusion support patches (not just the query frame)
to build a persistent point descriptor, then scoring re-entry candidates
against this descriptor, should outperform single-frame DINO score.

No training. Pure inference-time feature aggregation.
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


def extract_support_features(
    extractor: DINOFeatureExtractor,
    seq_path: Path,
    point_idx: int,
    query_frame: int,
    trajs_2d: np.ndarray,
    visibs: np.ndarray,
    num_support: int = 4,
    patch_size: int = 112,
) -> Tuple[torch.Tensor, List[int]]:
    """Extract frozen DINOv2 features from M pre-occlusion visible frames.

    Returns:
        support_features: (M, D) tensor of aggregated support descriptors
        support_frames: list of frame indices used
    """
    # Find last M visible frames before query_frame
    vis = visibs[:, point_idx].astype(bool)
    visible_frames = []
    for t in range(query_frame, -1, -1):
        if vis[t]:
            visible_frames.append(t)
        if len(visible_frames) >= num_support:
            break
    visible_frames.reverse()  # chronological order

    if not visible_frames:
        return None, []

    features = []
    for t in visible_frames:
        img = load_rgb_frame(seq_path, t)
        if img is None:
            continue
        # Get point position at this frame
        xy = trajs_2d[t, point_idx]
        if not np.all(np.isfinite(xy)):
            continue
        crop = extract_crop(img, xy, patch_size)
        feat = extractor.feature_map(crop)  # (D, h, w)
        # Global average pool to get point-specific descriptor
        feat_vec = feat.mean(dim=[1, 2])  # (D,)
        features.append(feat_vec)

    if not features:
        return None, []
    return torch.stack(features, dim=0), visible_frames


def score_candidate_with_support(
    support_features: torch.Tensor,
    cand_patch_feat: torch.Tensor,
    method: str = "mean",
) -> float:
    """Score a candidate patch against the support memory.

    Args:
        support_features: (M, D) support descriptors
        cand_patch_feat: (D, h, w) candidate patch features
        method: aggregation method
    Returns:
        similarity score (higher = more consistent with support memory)
    """
    # Global avg pool candidate features
    cand_vec = cand_patch_feat.mean(dim=[1, 2])  # (D,)
    cand_vec = F.normalize(cand_vec, dim=0)

    # Cosine similarity with each support view
    support_normed = F.normalize(support_features, dim=1)  # (M, D)
    sims = torch.mv(support_normed, cand_vec)  # (M,)

    if method == "max":
        return float(sims.max())
    elif method == "mean":
        return float(sims.mean())
    elif method == "top2_mean":
        top2 = torch.topk(sims, k=min(2, len(sims))).values
        return float(top2.mean())
    elif method == "view_weighted":
        # Earlier views get slightly more weight (they're closer to the point's identity)
        weights = torch.softmax(torch.arange(len(sims), dtype=torch.float32), dim=0)
        return float((sims * weights).sum())
    else:
        return float(sims.mean())


def get_score_list(result: Dict, score_key: str) -> List[float]:
    if score_key == "single_frame_scores":
        return result["single_frame_scores"]
    return result["multi_support_scores"][score_key]


def direct_selection_metrics(results: List[Dict], score_key: str) -> Dict[str, float]:
    baseline_errs = np.asarray([r["baseline_err"] for r in results], dtype=np.float32)
    pred_errs = []
    chosen_ranks = []
    for r in results:
        scores = get_score_list(r, score_key)
        best_idx = int(np.argmax(scores))
        chosen_ranks.append(best_idx)
        pred_errs.append(float(r["cand_errors"][best_idx]))
    pred_errs = np.asarray(pred_errs, dtype=np.float32)
    return {
        "median_px": float(np.median(pred_errs)),
        "lt4px": float(np.mean(pred_errs < 4.0)),
        "better_frac": float(np.mean(pred_errs < baseline_errs)),
        "chosen_rank_mean": float(np.mean(chosen_ranks)),
    }


def oracle_fallback_metrics(results: List[Dict], score_key: str) -> Dict[str, float]:
    baseline_errs = np.asarray([r["baseline_err"] for r in results], dtype=np.float32)
    pred_errs = []
    for r in results:
        scores = get_score_list(r, score_key)
        best_idx = int(np.argmax(scores))
        pred_errs.append(min(float(r["cand_errors"][best_idx]), float(r["baseline_err"])))
    pred_errs = np.asarray(pred_errs, dtype=np.float32)
    return {
        "median_px": float(np.median(pred_errs)),
        "lt4px": float(np.mean(pred_errs < 4.0)),
        "better_frac": float(np.mean(pred_errs < baseline_errs)),
    }


def margin_abstention_metrics(results: List[Dict], method: str, margin_thr: float) -> Dict[str, float]:
    baseline_errs = np.asarray([r["baseline_err"] for r in results], dtype=np.float32)
    final_errs = []
    accepted_candidate_errs = []
    accept_flags = []
    score_margins = []
    for r in results:
        cand_scores = np.asarray(r["multi_support_scores"][method], dtype=np.float32)
        baseline_score = float(r["baseline_ms_scores"][method])
        best_idx = int(np.argmax(cand_scores))
        best_score = float(cand_scores[best_idx])
        best_err = float(r["cand_errors"][best_idx])
        margin = best_score - baseline_score
        score_margins.append(margin)
        if margin >= margin_thr:
            final_errs.append(best_err)
            accepted_candidate_errs.append(best_err)
            accept_flags.append(True)
        else:
            final_errs.append(float(r["baseline_err"]))
            accept_flags.append(False)
    final_errs = np.asarray(final_errs, dtype=np.float32)
    accept_flags = np.asarray(accept_flags, dtype=bool)
    accepted_candidate_errs = np.asarray(accepted_candidate_errs, dtype=np.float32)
    return {
        "margin_thr": float(margin_thr),
        "coverage": float(np.mean(accept_flags)),
        "median_px": float(np.median(final_errs)),
        "lt4px": float(np.mean(final_errs < 4.0)),
        "better_frac": float(np.mean(final_errs < baseline_errs)),
        "accept_only_median_px": (
            float(np.median(accepted_candidate_errs)) if accepted_candidate_errs.size > 0 else None
        ),
        "mean_score_margin": float(np.mean(score_margins)),
    }


def risk_coverage_curve(results: List[Dict], method: str, num_points: int = 11) -> List[Dict[str, float]]:
    margins = []
    for r in results:
        cand_scores = np.asarray(r["multi_support_scores"][method], dtype=np.float32)
        baseline_score = float(r["baseline_ms_scores"][method])
        best_idx = int(np.argmax(cand_scores))
        margins.append(float(cand_scores[best_idx]) - baseline_score)
    margins = np.asarray(margins, dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, num_points)
    thresholds = np.quantile(margins, quantiles)
    curve = []
    seen = set()
    for thr in thresholds.tolist():
        thr = round(float(thr), 6)
        if thr in seen:
            continue
        seen.add(thr)
        curve.append(margin_abstention_metrics(results, method=method, margin_thr=thr))
    return curve


def main():
    parser = argparse.ArgumentParser(description="Zero-training multi-support DINO scorer")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-json", type=str,
                        default="/gemini/code/FSPT/outputs/multi_support_dino_smoke.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)
    rng = np.random.default_rng(42)

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sequences = discover_sequences(data_root, splits)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]

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

        # Filter by camera motion
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

            # Extract support memory from pre-occlusion frames
            support_feat, support_frames = extract_support_features(
                extractor, seq_path, i, t_q, trajs_2d, visibs,
                num_support=args.num_support, patch_size=args.query_crop_size,
            )
            if support_feat is None or len(support_frames) < 2:
                continue

            # Get baseline prediction (noisy world-state hold)
            hold_3d = trajs_3d[t_q, i].astype(np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if not np.isfinite(z_depth) or z_depth <= 1e-6:
                continue

            eps = float(np.clip(rng.normal(0.0, args.depth_noise_sigma), -0.35, 0.35))
            noisy_depth = z_depth * math.exp(eps)
            pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
            k_inv = np.linalg.inv(intrinsics[t_q])
            pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
            e_inv = np.linalg.inv(extrinsics[t_q])
            noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
            baseline_xy = project_3d_to_2d(noisy_world, intrinsics[t_re], extrinsics[t_re]).astype(np.float32)

            gt_xy = trajs_2d[t_re, i].astype(np.float32)
            baseline_err = float(np.linalg.norm(baseline_xy - gt_xy))

            # Get reentry image
            reentry_img = load_rgb_frame(seq_path, t_re)
            if reentry_img is None:
                continue

            # Generate candidates via DINO search (reuse existing logic)
            from scripts.eval_world_state_stage2_causal_dino import (
                extract_template, template_score_map, rgb_patch_ncc,
            )
            query_img = load_rgb_frame(seq_path, t_q)
            q_crop = extract_crop(query_img, trajs_2d[t_q, i].astype(np.float32), args.query_crop_size)
            r_crop = extract_crop(reentry_img, baseline_xy, args.search_crop_size)
            q_feat = extractor.feature_map(q_crop)
            r_feat = extractor.feature_map(r_crop)
            _, hr, wr = r_feat.shape

            q_center_xy = np.array([args.query_crop_size / 2.0, args.query_crop_size / 2.0], dtype=np.float32)
            templates = [extract_template(q_feat, q_center_xy, args.query_crop_size, r) for r in (1, 2)]
            sims = torch.stack([template_score_map(t, r_feat) for t in templates], dim=0).mean(dim=0)

            k_actual = min(args.topk, sims.numel())
            top_vals, top_idx = torch.topk(sims.reshape(-1), k=k_actual)
            patch_scale = args.search_crop_size / float(wr)
            center_offset = np.array([args.search_crop_size / 2.0, args.search_crop_size / 2.0], dtype=np.float32)

            # Score each candidate with both single-frame and multi-support
            cand_errors = []
            single_frame_scores = []
            multi_support_scores = {"max": [], "mean": [], "top2_mean": [], "view_weighted": []}

            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                by = idx // wr
                bx = idx % wr
                offset_xy = np.array([(bx + 0.5) * patch_scale, (by + 0.5) * patch_scale], dtype=np.float32)
                pred_xy = baseline_xy - center_offset + offset_xy
                cand_err = float(np.linalg.norm(pred_xy - gt_xy))
                cand_errors.append(cand_err)
                single_frame_scores.append(float(val))

                # Get candidate patch DINO features
                cand_patch = extract_crop(reentry_img, pred_xy, args.query_crop_size)
                cand_patch_feat = extractor.feature_map(cand_patch)

                # Score with multi-support
                for method in multi_support_scores:
                    score = score_candidate_with_support(support_feat, cand_patch_feat, method=method)
                    multi_support_scores[method].append(score)

            # Also score the baseline position
            baseline_patch = extract_crop(reentry_img, baseline_xy, args.query_crop_size)
            baseline_patch_feat = extractor.feature_map(baseline_patch)
            baseline_ms_scores = {}
            for method in multi_support_scores:
                baseline_ms_scores[method] = score_candidate_with_support(support_feat, baseline_patch_feat, method=method)

            results.append({
                "seq": seq_path.name,
                "baseline_err": baseline_err,
                "cand_errors": cand_errors,
                "single_frame_scores": single_frame_scores,
                "multi_support_scores": {k: v for k, v in multi_support_scores.items()},
                "baseline_ms_scores": baseline_ms_scores,
                "occ_length": int(q.occ_length),
                "camera_motion": cam_motion,
                "num_support": len(support_frames),
            })
            collected += 1

    # Analysis
    print(f"\nTotal samples: {len(results)}", flush=True)
    if not results:
        print("No results.", flush=True)
        return

    from sklearn.metrics import roc_auc_score

    baseline_errs = np.array([r["baseline_err"] for r in results])

    # --- 1. AUC: candidate-vs-baseline binary classification ---
    all_labels = []
    all_single = []
    all_ms = {m: [] for m in ["max", "mean", "top2_mean", "view_weighted"]}

    for r in results:
        be = r["baseline_err"]
        for j, ce in enumerate(r["cand_errors"]):
            all_labels.append(1 if ce < be else 0)
            all_single.append(r["single_frame_scores"][j])
            for m in all_ms:
                all_ms[m].append(r["multi_support_scores"][m][j])

    all_labels = np.array(all_labels)
    all_single = np.array(all_single)

    auc_summary = {
        "single_frame": float(roc_auc_score(all_labels, all_single)),
        "multi_support": {},
    }
    print(f"\n=== AUC (candidate better than baseline) ===")
    print(f"  Single-frame DINO:  {auc_summary['single_frame']:.4f}")
    for m in all_ms:
        arr = np.array(all_ms[m])
        auc_summary["multi_support"][m] = float(roc_auc_score(all_labels, arr))
        print(f"  Multi-support ({m}): {auc_summary['multi_support'][m]:.4f}")

    # --- 2. Direct selection median (NO oracle fallback) ---
    print(f"\n=== Direct selection median (NO fallback) ===")
    print(f"  Baseline only:       median={np.median(baseline_errs):.2f}")

    oracle_errs = np.array([min(r["baseline_err"], min(r["cand_errors"])) for r in results])
    print(f"  Oracle:              median={np.median(oracle_errs):.2f}")

    direct_summary = {
        "baseline_only": {
            "median_px": float(np.median(baseline_errs)),
            "lt4px": float(np.mean(baseline_errs < 4.0)),
        },
        "oracle": {
            "median_px": float(np.median(oracle_errs)),
            "lt4px": float(np.mean(oracle_errs < 4.0)),
        },
        "single_frame": direct_selection_metrics(results, "single_frame_scores"),
        "multi_support": {},
    }

    for label, score_key in [("Single-frame", "single_frame_scores")] + [(f"Multi-supp({m})", m) for m in all_ms]:
        metrics = direct_selection_metrics(results, score_key)
        if score_key != "single_frame_scores":
            direct_summary["multi_support"][score_key] = metrics
        print(f"  {label:20s}: median={metrics['median_px']:.2f}, better_frac={metrics['better_frac']:.2%}")

    # --- 3. Margin-based abstention + coverage-risk ---
    print(f"\n=== Margin-based abstention (best_method=max) ===")
    method = "max"
    print(f"  {'margin_thr':>12} {'coverage':>10} {'median_err':>10} {'better_frac':>12} {'accept_median':>14}")
    margin_thresholds = [-0.5, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    margin_summary = []
    for thr in [-0.5, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        metrics = margin_abstention_metrics(results, method=method, margin_thr=thr)
        margin_summary.append(metrics)
        if metrics["coverage"] == 0.0:
            continue
        accept_only = metrics["accept_only_median_px"]
        accept_only_str = "None" if accept_only is None else f"{accept_only:.2f}"
        print(
            f"  {thr:>12.2f} {metrics['coverage']:>10.1%} {metrics['median_px']:>10.2f}"
            f" {metrics['better_frac']:>12.2%} {accept_only_str:>14s}"
        )

    fallback_summary = {"single_frame": oracle_fallback_metrics(results, "single_frame_scores"), "multi_support": {}}
    for m in all_ms:
        fallback_summary["multi_support"][m] = oracle_fallback_metrics(results, m)

    risk_coverage_summary = {
        m: risk_coverage_curve(results, method=m, num_points=11)
        for m in all_ms
    }

    # Save
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "n": len(results),
        "summary": {
            "auc": auc_summary,
            "direct_selection": direct_summary,
            "oracle_fallback": fallback_summary,
            "margin_abstention": {
                "method": method,
                "thresholds": margin_thresholds,
                "rows": margin_summary,
            },
            "risk_coverage": risk_coverage_summary,
        },
        "results": results,  # save all results
    }, indent=2, default=str) + "\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
