#!/usr/bin/env python3
"""
Build a cached candidate dataset for PRT temporal verification.

This script does not train a model. It materializes the intermediate data
needed by the next-stage temporal verifier:

  - PRT query metadata
  - baseline re-entry prediction from noisy query-depth world-state hold
  - top-k causal DINO local candidates around the baseline
  - query / baseline / candidate patches at re-entry
  - post-reentry short window frame indices and GT annotations
  - GT-derived training labels:
      baseline / best_candidate / abstain

Outputs:
  - dataset_cache.npz
  - samples_meta.jsonl
  - stats.json
  - preview.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_prt_splits import PRTQuery, extract_prt_queries_for_sequence, load_image_size, classify_reentry_type
from scripts.eval_world_state_stage0 import load_sequence, project_3d_to_2d
from scripts.eval_world_state_stage2_causal_dino import (
    DINOFeatureExtractor,
    extract_crop,
    extract_template,
    load_rgb_frame,
    rgb_patch_ncc,
    template_score_map,
)


def extract_support_patch_stack(
    seq_path: Path,
    point_idx: int,
    query_frame: int,
    trajs_2d: np.ndarray,
    visibs: np.ndarray,
    num_support: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vis = visibs[:, point_idx].astype(bool)
    support_frames: List[int] = []
    for t in range(query_frame, -1, -1):
        if vis[t]:
            support_frames.append(t)
        if len(support_frames) >= num_support:
            break
    support_frames.reverse()

    patches: List[np.ndarray] = []
    for t in support_frames:
        img = load_rgb_frame(seq_path, int(t))
        if img is None:
            continue
        xy = trajs_2d[t, point_idx].astype(np.float32)
        if not np.all(np.isfinite(xy)):
            continue
        patch = extract_crop(img, xy, patch_size)
        patches.append((patch.astype(np.float32) / 255.0).transpose(2, 0, 1))

    if not patches:
        return np.zeros((0, 3, patch_size, patch_size), dtype=np.float16), np.zeros((0,), dtype=np.int32)
    return np.stack(patches, axis=0).astype(np.float16), np.asarray(support_frames[: len(patches)], dtype=np.int32)


def project_noisy_query_world(
    query_xy: np.ndarray,
    hold_3d: np.ndarray,
    query_intrinsics: np.ndarray,
    query_extrinsics: np.ndarray,
    reentry_intrinsics: np.ndarray,
    reentry_extrinsics: np.ndarray,
    depth_noise_sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    pt_cam = query_extrinsics[:3, :3] @ hold_3d + query_extrinsics[:3, 3]
    z_depth = float(pt_cam[2])
    if not np.isfinite(z_depth) or z_depth <= 1e-6:
        raise ValueError("Invalid query depth.")

    eps = float(np.clip(rng.normal(0.0, depth_noise_sigma), -0.35, 0.35))
    noisy_depth = z_depth * math.exp(eps)
    pixels_h = np.array([query_xy[0], query_xy[1], 1.0], dtype=np.float32)
    k_inv = np.linalg.inv(query_intrinsics)
    pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
    e_inv = np.linalg.inv(query_extrinsics)
    noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
    baseline_xy = project_3d_to_2d(noisy_world, reentry_intrinsics, reentry_extrinsics).astype(np.float32)
    return baseline_xy, noisy_depth


def collect_topk_candidates(
    extractor: DINOFeatureExtractor,
    query_img: np.ndarray,
    reentry_img: np.ndarray,
    query_xy: np.ndarray,
    center_xy: np.ndarray,
    topk: int,
    query_crop_size: int,
    search_crop_size: int,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top-k candidate xy / DINO score / RGB-NCC / distance / patches."""
    q_crop = extract_crop(query_img, query_xy, query_crop_size)
    r_crop = extract_crop(reentry_img, center_xy, search_crop_size)

    q_feat = extractor.feature_map(q_crop)
    r_feat = extractor.feature_map(r_crop)
    _, hr, wr = r_feat.shape

    q_center_xy = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)
    templates = [extract_template(q_feat, q_center_xy, query_crop_size, radius) for radius in (1, 2)]
    sims = torch.stack([template_score_map(t, r_feat) for t in templates], dim=0).mean(dim=0)

    topk_actual = min(topk, sims.numel())
    top_vals, top_idx = torch.topk(sims.reshape(-1), k=topk_actual)

    patch_scale = search_crop_size / float(wr)
    center_offset = np.array([search_crop_size / 2.0, search_crop_size / 2.0], dtype=np.float32)
    q_rgb_center = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)

    cand_xy: List[np.ndarray] = []
    cand_scores: List[float] = []
    cand_ncc: List[float] = []
    cand_dist_norm: List[float] = []
    cand_patches: List[np.ndarray] = []

    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        by = idx // wr
        bx = idx % wr
        offset_xy = np.array([(bx + 0.5) * patch_scale, (by + 0.5) * patch_scale], dtype=np.float32)
        pred_xy = center_xy - center_offset + offset_xy
        rgb_score = rgb_patch_ncc(
            query_crop=q_crop,
            reentry_crop=r_crop,
            query_center_xy=q_rgb_center,
            reentry_center_xy=offset_xy,
            patch_size=32,
        )
        dist_norm = float(
            np.linalg.norm(offset_xy - center_offset) / (search_crop_size / 2.0 + 1e-6)
        )
        patch = extract_crop(reentry_img, pred_xy, patch_size)
        cand_xy.append(pred_xy.astype(np.float32))
        cand_scores.append(float(val))
        cand_ncc.append(float(rgb_score))
        cand_dist_norm.append(dist_norm)
        cand_patches.append((patch.astype(np.float32) / 255.0).transpose(2, 0, 1))

    return (
        np.stack(cand_xy, axis=0),
        np.asarray(cand_scores, dtype=np.float32),
        np.asarray(cand_ncc, dtype=np.float32),
        np.asarray(cand_dist_norm, dtype=np.float32),
        np.stack(cand_patches, axis=0),
    )


def local_match_top1(
    extractor: DINOFeatureExtractor,
    query_img: np.ndarray,
    next_img: np.ndarray,
    query_xy: np.ndarray,
    center_xy: np.ndarray,
    query_crop_size: int,
    search_crop_size: int,
) -> Tuple[np.ndarray, float]:
    cand_xy, cand_scores, _, _, _ = collect_topk_candidates(
        extractor=extractor,
        query_img=query_img,
        reentry_img=next_img,
        query_xy=query_xy,
        center_xy=center_xy,
        topk=1,
        query_crop_size=query_crop_size,
        search_crop_size=search_crop_size,
        patch_size=32,
    )
    return cand_xy[0], float(cand_scores[0])


def build_tracklet_window(
    extractor: DINOFeatureExtractor,
    frame_images: List[np.ndarray],
    init_xy: np.ndarray,
    query_crop_size: int,
    search_crop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track a point causally through the short post-reentry window."""
    coords = [init_xy.astype(np.float32)]
    scores = [1.0]
    prev_xy = init_xy.astype(np.float32)
    for t in range(1, len(frame_images)):
        pred_xy, score = local_match_top1(
            extractor=extractor,
            query_img=frame_images[t - 1],
            next_img=frame_images[t],
            query_xy=prev_xy,
            center_xy=prev_xy,
            query_crop_size=query_crop_size,
            search_crop_size=search_crop_size,
        )
        coords.append(pred_xy.astype(np.float32))
        scores.append(float(score))
        prev_xy = pred_xy.astype(np.float32)
    return np.stack(coords, axis=0), np.asarray(scores, dtype=np.float32)


def decide_label(
    baseline_err: float,
    cand_errors: np.ndarray,
    positive_margin_px: float,
    abstain_threshold_px: float,
) -> Tuple[int, str, int]:
    """Return (label_index, label_name, oracle_index).

    label_index:
      -1: abstain
       0: baseline
       1..K: selected candidate index + 1

    oracle_index:
       0: baseline best
       1..K: candidate best
    """
    if len(cand_errors) == 0:
        if baseline_err <= abstain_threshold_px:
            return 0, "baseline", 0
        return -1, "abstain", 0

    best_idx = int(np.argmin(cand_errors))
    best_err = float(cand_errors[best_idx])
    oracle_index = 0 if baseline_err <= best_err else (best_idx + 1)

    if best_err < baseline_err - positive_margin_px:
        return best_idx + 1, "best_candidate", oracle_index
    if baseline_err <= abstain_threshold_px:
        return 0, "baseline", oracle_index
    return -1, "abstain", oracle_index


def reentry_type_to_id(name: str) -> int:
    mapping = {
        "in_frame_occlusion": 0,
        "offscreen_return": 1,
        "mixed": 2,
    }
    return mapping.get(name, 2)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def occ_bucket_name(
    occ_length: int,
    short_max: int,
    medium_max: int,
    long_max: int,
) -> str:
    if occ_length < short_max:
        return "short"
    if occ_length < medium_max:
        return "medium"
    if occ_length < long_max:
        return "long"
    return "very_long"


def sampling_bucket_name(
    reentry_type: str,
    occ_length: int,
    short_max: int,
    medium_max: int,
    long_max: int,
) -> str:
    if reentry_type == "offscreen_return":
        return "offscreen"
    prefix = "in_frame" if reentry_type == "in_frame_occlusion" else "mixed"
    return f"{prefix}_{occ_bucket_name(occ_length, short_max, medium_max, long_max)}"


def bucket_caps_from_args(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "in_frame_short": int(args.bucket_cap_inframe_short),
        "in_frame_medium": int(args.bucket_cap_inframe_medium),
        "in_frame_long": int(args.bucket_cap_inframe_long),
        "in_frame_very_long": int(args.bucket_cap_inframe_very_long),
        "offscreen": int(args.bucket_cap_offscreen),
        "mixed_short": int(args.bucket_cap_mixed_short),
        "mixed_medium": int(args.bucket_cap_mixed_medium),
        "mixed_long": int(args.bucket_cap_mixed_long),
        "mixed_very_long": int(args.bucket_cap_mixed_very_long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PRT candidate dataset cache")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--reentry-types", type=str, default="in_frame_occlusion,offscreen_return")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences-per-split", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--post-window", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--positive-margin-px", type=float, default=2.0)
    parser.add_argument("--abstain-threshold-px", type=float, default=32.0)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--stratified-sampling", action="store_true")
    parser.add_argument("--occ-short-max", type=int, default=100)
    parser.add_argument("--occ-medium-max", type=int, default=200)
    parser.add_argument("--occ-long-max", type=int, default=500)
    parser.add_argument("--bucket-cap-inframe-short", type=int, default=100)
    parser.add_argument("--bucket-cap-inframe-medium", type=int, default=150)
    parser.add_argument("--bucket-cap-inframe-long", type=int, default=250)
    parser.add_argument("--bucket-cap-inframe-very-long", type=int, default=50)
    parser.add_argument("--bucket-cap-offscreen", type=int, default=100)
    parser.add_argument("--bucket-cap-mixed-short", type=int, default=50)
    parser.add_argument("--bucket-cap-mixed-medium", type=int, default=75)
    parser.add_argument("--bucket-cap-mixed-long", type=int, default=100)
    parser.add_argument("--bucket-cap-mixed-very-long", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reentry_types = {x.strip() for x in args.reentry_types.split(",") if x.strip()}
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    query_patches: List[np.ndarray] = []
    baseline_patches: List[np.ndarray] = []
    cand_patches: List[np.ndarray] = []
    support_patches_all: List[np.ndarray] = []
    support_frames_all: List[np.ndarray] = []
    support_count_all: List[int] = []
    query_xy_all: List[np.ndarray] = []
    gt_reentry_xy_all: List[np.ndarray] = []
    baseline_xy_all: List[np.ndarray] = []
    cand_xy_all: List[np.ndarray] = []
    baseline_err_all: List[float] = []
    cand_err_all: List[np.ndarray] = []
    cand_score_all: List[np.ndarray] = []
    cand_ncc_all: List[np.ndarray] = []
    cand_dist_all: List[np.ndarray] = []
    label_index_all: List[int] = []
    oracle_index_all: List[int] = []
    reentry_type_id_all: List[int] = []
    occ_length_all: List[int] = []
    camera_motion_all: List[float] = []
    gt_window_xy_all: List[np.ndarray] = []
    gt_window_vis_all: List[np.ndarray] = []
    window_frames_all: List[np.ndarray] = []
    baseline_track_xy_all: List[np.ndarray] = []
    baseline_track_score_all: List[np.ndarray] = []
    baseline_track_err_all: List[np.ndarray] = []
    cand_track_xy_all: List[np.ndarray] = []
    cand_track_score_all: List[np.ndarray] = []
    cand_track_err_all: List[np.ndarray] = []
    meta_records: List[Dict[str, Any]] = []
    planned_bucket_counts_total: Dict[str, int] = {}
    available_bucket_counts_total: Dict[str, int] = {}

    sample_count = 0
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    extr_inv_cache: Dict[str, Any] = {}  # lazy pre-compute per sequence

    for split in splits:
        split_root = data_root / split
        if not split_root.is_dir():
            continue
        seq_paths = sorted([d for d in split_root.iterdir() if d.is_dir() and (d / "anno.npz").exists()])
        if args.max_sequences_per_split > 0:
            seq_paths = seq_paths[: args.max_sequences_per_split]

        # Compute per-sequence sample budget for uniform coverage
        n_seqs = len(seq_paths)
        per_seq_budget = max(1, args.max_samples // n_seqs)
        remainder = args.max_samples - per_seq_budget * n_seqs

        for seq_idx, seq_path in enumerate(seq_paths):
            if sample_count >= args.max_samples:
                break
            # First `remainder` sequences get one extra sample
            seq_budget = per_seq_budget + (1 if seq_idx < remainder else 0)

            seq = load_sequence(seq_path)
            # Use fast find_reentry_queries (1s for 114K queries) instead of
            # slow extract_prt_queries_for_sequence (30+ min for same data).
            # Filter camera_motion lazily per-sample below.
            from scripts.eval_world_state_stage0 import find_reentry_queries as _fast_find
            fast_queries = _fast_find(seq, min_occ_length=args.min_occ_length)
            seq_collected = 0
            extr_inv = np.linalg.inv(seq["extrinsics"])  # pre-compute once
            trajs_2d = seq["trajs_2d"]
            trajs_3d = seq["trajs_3d"]
            visibs = seq["visibs"]
            intrinsics = seq["intrinsics"]
            extrinsics = seq["extrinsics"]
            height, width = load_image_size(seq_path)
            bucket_caps = bucket_caps_from_args(args)
            planned_items: List[Dict[str, Any]] = []
            if args.stratified_sampling:
                bucket_to_items: Dict[str, List[Dict[str, Any]]] = {}
                for q in fast_queries:
                    t_q = int(q.query_frame)
                    t_re = int(q.reentry_frame)
                    i = int(q.point_idx)

                    e_rel = extrinsics[t_re] @ extr_inv[t_q]
                    cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                    if cam_motion < args.min_camera_motion:
                        continue

                    vis_i = visibs[:, i].astype(bool)
                    valid_i = (
                        np.ones_like(vis_i)
                        if not hasattr(seq.get("valids", None), "shape")
                        else seq["valids"][:, i].astype(bool)
                    )
                    reentry_type, _, _ = classify_reentry_type(
                        trajs_2d_i=trajs_2d[:, i],
                        visibs_i=vis_i,
                        valids_i=valid_i,
                        width=width,
                        height=height,
                        occ_start=t_q + 1,
                        occ_end=t_re - 1,
                    )
                    if reentry_type not in reentry_types:
                        continue

                    sampling_bucket = sampling_bucket_name(
                        reentry_type=reentry_type,
                        occ_length=int(q.occ_length),
                        short_max=args.occ_short_max,
                        medium_max=args.occ_medium_max,
                        long_max=args.occ_long_max,
                    )
                    camera_translation = float(np.linalg.norm(e_rel[:3, 3]))
                    rec = {
                        "q": q,
                        "reentry_type": reentry_type,
                        "camera_motion": cam_motion,
                        "camera_translation": camera_translation,
                        "sampling_bucket": sampling_bucket,
                    }
                    bucket_to_items.setdefault(sampling_bucket, []).append(rec)

                for bucket_name, items in bucket_to_items.items():
                    available_bucket_counts_total[bucket_name] = (
                        available_bucket_counts_total.get(bucket_name, 0) + len(items)
                    )
                    rng.shuffle(items)
                    cap = bucket_caps.get(bucket_name, len(items))
                    take = items[: min(len(items), max(0, cap))]
                    planned_items.extend(take)
                    planned_bucket_counts_total[bucket_name] = (
                        planned_bucket_counts_total.get(bucket_name, 0) + len(take)
                    )

                rng.shuffle(planned_items)
                if len(planned_items) > seq_budget:
                    planned_items = planned_items[:seq_budget]
            else:
                # Shuffle queries so we sample uniformly across the sequence.
                rng.shuffle(fast_queries)

            iter_items = planned_items if args.stratified_sampling else fast_queries

            for item in iter_items:
                if sample_count >= args.max_samples:
                    break
                if seq_collected >= seq_budget:
                    break

                if args.stratified_sampling:
                    q = item["q"]
                    reentry_type = str(item["reentry_type"])
                    cam_motion = float(item["camera_motion"])
                    camera_translation = float(item["camera_translation"])
                    sampling_bucket = str(item["sampling_bucket"])
                else:
                    q = item
                    t_q = int(q.query_frame)
                    t_re = int(q.reentry_frame)
                    i = int(q.point_idx)

                    e_rel = extrinsics[t_re] @ extr_inv[t_q]
                    cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                    if cam_motion < args.min_camera_motion:
                        continue

                    vis_i = visibs[:, i].astype(bool)
                    valid_i = (
                        np.ones_like(vis_i)
                        if not hasattr(seq.get("valids", None), "shape")
                        else seq["valids"][:, i].astype(bool)
                    )
                    reentry_type, _, _ = classify_reentry_type(
                        trajs_2d_i=trajs_2d[:, i],
                        visibs_i=vis_i,
                        valids_i=valid_i,
                        width=width,
                        height=height,
                        occ_start=t_q + 1,
                        occ_end=t_re - 1,
                    )
                    if reentry_type not in reentry_types:
                        continue
                    camera_translation = float(np.linalg.norm(e_rel[:3, 3]))
                    sampling_bucket = sampling_bucket_name(
                        reentry_type=reentry_type,
                        occ_length=int(q.occ_length),
                        short_max=args.occ_short_max,
                        medium_max=args.occ_medium_max,
                        long_max=args.occ_long_max,
                    )

                t_q = int(q.query_frame)
                t_re = int(q.reentry_frame)
                i = int(q.point_idx)

                query_img = load_rgb_frame(seq_path, t_q)
                reentry_img = load_rgb_frame(seq_path, t_re)
                if query_img is None or reentry_img is None:
                    continue

                query_xy = trajs_2d[t_q, i].astype(np.float32)
                gt_reentry_xy = trajs_2d[t_re, i].astype(np.float32)
                hold_3d = trajs_3d[t_q, i].astype(np.float32)

                try:
                    baseline_xy, noisy_depth = project_noisy_query_world(
                        query_xy=query_xy,
                        hold_3d=hold_3d,
                        query_intrinsics=intrinsics[t_q],
                        query_extrinsics=extrinsics[t_q],
                        reentry_intrinsics=intrinsics[t_re],
                        reentry_extrinsics=extrinsics[t_re],
                        depth_noise_sigma=args.depth_noise_sigma,
                        rng=rng,
                    )
                except ValueError:
                    continue

                baseline_err = float(np.linalg.norm(baseline_xy - gt_reentry_xy))
                try:
                    (
                        cand_xy,
                        cand_scores,
                        cand_ncc,
                        cand_dist,
                        cand_patch_stack,
                    ) = collect_topk_candidates(
                        extractor=extractor,
                        query_img=query_img,
                        reentry_img=reentry_img,
                        query_xy=query_xy,
                        center_xy=baseline_xy,
                        topk=args.topk,
                        query_crop_size=args.query_crop_size,
                        search_crop_size=args.search_crop_size,
                        patch_size=args.patch_size,
                    )
                except Exception:
                    continue

                cand_errors = np.linalg.norm(cand_xy - gt_reentry_xy[None, :], axis=1).astype(np.float32)
                label_index, label_name, oracle_index = decide_label(
                    baseline_err=baseline_err,
                    cand_errors=cand_errors,
                    positive_margin_px=args.positive_margin_px,
                    abstain_threshold_px=args.abstain_threshold_px,
                )

                # Query / baseline patches at re-entry.
                query_patch = extract_crop(query_img, query_xy, args.patch_size)
                baseline_patch = extract_crop(reentry_img, baseline_xy, args.patch_size)
                query_patch = (query_patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
                baseline_patch = (baseline_patch.astype(np.float32) / 255.0).transpose(2, 0, 1)

                support_patches, support_frames = extract_support_patch_stack(
                    seq_path=seq_path,
                    point_idx=i,
                    query_frame=t_q,
                    trajs_2d=trajs_2d,
                    visibs=visibs,
                    num_support=args.num_support,
                    patch_size=args.patch_size,
                )
                if support_patches.shape[0] == 0:
                    continue
                if support_patches.shape[0] < args.num_support:
                    pad_n = args.num_support - support_patches.shape[0]
                    support_patches = np.pad(
                        support_patches,
                        ((0, pad_n), (0, 0), (0, 0), (0, 0)),
                        mode="edge",
                    )
                    support_frames = np.pad(support_frames, (0, pad_n), mode="edge")

                # Short post-reentry window GT annotations.
                window_end = min(t_re + args.post_window, trajs_2d.shape[0])
                frame_ids = np.arange(t_re, window_end, dtype=np.int32)
                gt_window_xy = trajs_2d[frame_ids, i].astype(np.float32)
                gt_window_vis = visibs[frame_ids, i].astype(np.float32)

                if len(frame_ids) < args.post_window:
                    pad_n = args.post_window - len(frame_ids)
                    frame_ids = np.pad(frame_ids, (0, pad_n), mode="edge")
                    gt_window_xy = np.pad(gt_window_xy, ((0, pad_n), (0, 0)), mode="edge")
                    gt_window_vis = np.pad(gt_window_vis, (0, pad_n), mode="constant")

                window_images: List[np.ndarray] = []
                valid_window = True
                for frame_id in frame_ids.tolist():
                    img = load_rgb_frame(seq_path, int(frame_id))
                    if img is None:
                        valid_window = False
                        break
                    window_images.append(img)
                if not valid_window:
                    continue

                baseline_track_xy, baseline_track_score = build_tracklet_window(
                    extractor=extractor,
                    frame_images=window_images,
                    init_xy=baseline_xy,
                    query_crop_size=args.query_crop_size,
                    search_crop_size=args.search_crop_size,
                )
                baseline_track_err = np.linalg.norm(
                    baseline_track_xy - gt_window_xy,
                    axis=1,
                ).astype(np.float32)

                cand_track_xy_list: List[np.ndarray] = []
                cand_track_score_list: List[np.ndarray] = []
                cand_track_err_list: List[np.ndarray] = []
                for cand_init_xy in cand_xy:
                    track_xy, track_score = build_tracklet_window(
                        extractor=extractor,
                        frame_images=window_images,
                        init_xy=cand_init_xy,
                        query_crop_size=args.query_crop_size,
                        search_crop_size=args.search_crop_size,
                    )
                    track_err = np.linalg.norm(track_xy - gt_window_xy, axis=1).astype(np.float32)
                    cand_track_xy_list.append(track_xy)
                    cand_track_score_list.append(track_score)
                    cand_track_err_list.append(track_err)

                query_patches.append(query_patch.astype(np.float16))
                baseline_patches.append(baseline_patch.astype(np.float16))
                cand_patches.append(cand_patch_stack.astype(np.float16))
                support_patches_all.append(support_patches.astype(np.float16))
                support_frames_all.append(support_frames.astype(np.int32))
                support_count_all.append(int(min(len(support_frames), args.num_support)))
                query_xy_all.append(query_xy)
                gt_reentry_xy_all.append(gt_reentry_xy)
                baseline_xy_all.append(baseline_xy)
                cand_xy_all.append(cand_xy)
                baseline_err_all.append(baseline_err)
                cand_err_all.append(cand_errors)
                cand_score_all.append(cand_scores)
                cand_ncc_all.append(cand_ncc)
                cand_dist_all.append(cand_dist)
                label_index_all.append(label_index)
                oracle_index_all.append(oracle_index)
                reentry_type_id_all.append(reentry_type_to_id(reentry_type))
                occ_length_all.append(q.occ_length)
                camera_motion_all.append(cam_motion)
                gt_window_xy_all.append(gt_window_xy)
                gt_window_vis_all.append(gt_window_vis)
                window_frames_all.append(frame_ids)
                baseline_track_xy_all.append(baseline_track_xy)
                baseline_track_score_all.append(baseline_track_score)
                baseline_track_err_all.append(baseline_track_err)
                cand_track_xy_all.append(np.stack(cand_track_xy_list, axis=0))
                cand_track_score_all.append(np.stack(cand_track_score_list, axis=0))
                cand_track_err_all.append(np.stack(cand_track_err_list, axis=0))

                meta_records.append(
                    {
                        "sample_id": sample_count,
                        "seq_name": seq_path.name,
                        "split": split,
                        "point_idx": i,
                        "query_frame": t_q,
                        "reentry_frame": t_re,
                        "occ_length": q.occ_length,
                        "camera_motion": cam_motion,
                        "camera_translation": camera_translation,
                        "reentry_type": reentry_type,
                        "sampling_bucket": sampling_bucket,
                        "num_support": int(min(len(support_frames), args.num_support)),
                        "support_frames": support_frames.tolist(),
                        "query_xy": query_xy.tolist(),
                        "gt_reentry_xy": gt_reentry_xy.tolist(),
                        "baseline_xy": baseline_xy.tolist(),
                        "baseline_err": baseline_err,
                        "cand_xy": cand_xy.tolist(),
                        "cand_errors": cand_errors.tolist(),
                        "cand_scores": cand_scores.tolist(),
                        "cand_ncc": cand_ncc.tolist(),
                        "cand_dist_norm": cand_dist.tolist(),
                        "label_index": label_index,
                        "label_name": label_name,
                        "oracle_index": oracle_index,
                        "noisy_query_depth": noisy_depth,
                        "window_frames": frame_ids.tolist(),
                        "baseline_track_err": baseline_track_err.tolist(),
                        "best_candidate_track_err": (
                            cand_track_err_list[int(np.argmin(cand_errors))].tolist() if len(cand_track_err_list) > 0 else []
                        ),
                    }
                )

                sample_count += 1
                seq_collected += 1

            if sample_count >= args.max_samples:
                break
        if sample_count >= args.max_samples:
            break

    if sample_count == 0:
        raise RuntimeError("No valid PRT candidate samples collected.")

    dataset_cache = {
        "query_patch": np.stack(query_patches, axis=0),
        "baseline_patch": np.stack(baseline_patches, axis=0),
        "cand_patches": np.stack(cand_patches, axis=0),
        "support_patches": np.stack(support_patches_all, axis=0),
        "support_frames": np.stack(support_frames_all, axis=0).astype(np.int32),
        "support_count": np.asarray(support_count_all, dtype=np.int32),
        "query_xy": np.stack(query_xy_all, axis=0).astype(np.float32),
        "gt_reentry_xy": np.stack(gt_reentry_xy_all, axis=0).astype(np.float32),
        "baseline_xy": np.stack(baseline_xy_all, axis=0).astype(np.float32),
        "cand_xy": np.stack(cand_xy_all, axis=0).astype(np.float32),
        "baseline_err": np.asarray(baseline_err_all, dtype=np.float32),
        "cand_err": np.stack(cand_err_all, axis=0).astype(np.float32),
        "cand_score": np.stack(cand_score_all, axis=0).astype(np.float32),
        "cand_ncc": np.stack(cand_ncc_all, axis=0).astype(np.float32),
        "cand_dist_norm": np.stack(cand_dist_all, axis=0).astype(np.float32),
        "label_index": np.asarray(label_index_all, dtype=np.int32),
        "oracle_index": np.asarray(oracle_index_all, dtype=np.int32),
        "reentry_type_id": np.asarray(reentry_type_id_all, dtype=np.int32),
        "occ_length": np.asarray(occ_length_all, dtype=np.int32),
        "camera_motion": np.asarray(camera_motion_all, dtype=np.float32),
        "gt_window_xy": np.stack(gt_window_xy_all, axis=0).astype(np.float32),
        "gt_window_vis": np.stack(gt_window_vis_all, axis=0).astype(np.float32),
        "window_frames": np.stack(window_frames_all, axis=0).astype(np.int32),
        "baseline_track_xy": np.stack(baseline_track_xy_all, axis=0).astype(np.float32),
        "baseline_track_score": np.stack(baseline_track_score_all, axis=0).astype(np.float32),
        "baseline_track_err": np.stack(baseline_track_err_all, axis=0).astype(np.float32),
        "cand_track_xy": np.stack(cand_track_xy_all, axis=0).astype(np.float32),
        "cand_track_score": np.stack(cand_track_score_all, axis=0).astype(np.float32),
        "cand_track_err": np.stack(cand_track_err_all, axis=0).astype(np.float32),
    }
    np.savez_compressed(output_dir / "dataset_cache.npz", **dataset_cache)

    with (output_dir / "samples_meta.jsonl").open("w") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")

    label_name_counts: Dict[str, int] = {}
    for rec in meta_records:
        label_name_counts[rec["label_name"]] = label_name_counts.get(rec["label_name"], 0) + 1
    reentry_type_counts: Dict[str, int] = {}
    for rec in meta_records:
        reentry_type_counts[rec["reentry_type"]] = reentry_type_counts.get(rec["reentry_type"], 0) + 1
    sampling_bucket_counts: Dict[str, int] = {}
    for rec in meta_records:
        bucket_name = str(rec["sampling_bucket"])
        sampling_bucket_counts[bucket_name] = sampling_bucket_counts.get(bucket_name, 0) + 1

    baseline_err_np = dataset_cache["baseline_err"]
    cand_err_np = dataset_cache["cand_err"]
    best_cand_err_np = cand_err_np.min(axis=1)
    stats = {
        "config": vars(args),
        "sampling_mode": "stratified" if args.stratified_sampling else "random",
        "count": sample_count,
        "reentry_type_counts": reentry_type_counts,
        "sampling_bucket_counts": sampling_bucket_counts,
        "sampling_bucket_available_counts": available_bucket_counts_total,
        "sampling_bucket_planned_counts": planned_bucket_counts_total,
        "label_name_counts": label_name_counts,
        "baseline_median_px": float(np.median(baseline_err_np)),
        "best_candidate_median_px": float(np.median(best_cand_err_np)),
        "oracle_gap_median_px": float(np.median(baseline_err_np - best_cand_err_np)),
        "candidate_beats_baseline_frac": float(np.mean(best_cand_err_np < baseline_err_np)),
        "baseline_lt4px": float(np.mean(baseline_err_np < 4.0)),
        "best_candidate_lt4px": float(np.mean(best_cand_err_np < 4.0)),
        "baseline_track_window_median_px": float(np.median(dataset_cache["baseline_track_err"])),
        "best_candidate_track_window_median_px": float(np.median(dataset_cache["cand_track_err"].min(axis=1))),
        "post_window": int(args.post_window),
        "topk": int(args.topk),
        "fields": {
            "label_index": {"-1": "abstain", "0": "baseline", "1..K": "best candidate index + 1"},
            "oracle_index": {"0": "baseline oracle", "1..K": "candidate oracle"},
            "reentry_type_id": {"0": "in_frame_occlusion", "1": "offscreen_return", "2": "mixed"},
        },
    }
    save_json(output_dir / "stats.json", stats)

    preview = {
        "config": vars(args),
        "count": sample_count,
        "preview": meta_records[: args.preview_count],
    }
    save_json(output_dir / "preview.json", preview)

    print(json.dumps(stats, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
