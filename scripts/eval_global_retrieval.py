#!/usr/bin/env python3
"""
Whole-frame dense patch retrieval for PRT re-entry.

Goal:
  Test whether the GT re-entry location is recoverable by global dense
  retrieval over the whole frame, without relying on a local search window
  around the baseline projection.

This is a zero-training evaluation script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage0 import discover_sequences, find_reentry_queries, load_sequence, project_3d_to_2d
from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor, extract_crop, extract_template, load_rgb_frame, template_score_map


def extract_support_descriptor(
    extractor: DINOFeatureExtractor,
    seq_path: Path,
    point_idx: int,
    query_frame: int,
    trajs_2d: np.ndarray,
    visibs: np.ndarray,
    num_support: int,
    patch_size: int,
) -> Tuple[torch.Tensor, List[int]]:
    vis = visibs[:, point_idx].astype(bool)
    frames: List[int] = []
    for t in range(query_frame, -1, -1):
        if vis[t]:
            frames.append(t)
        if len(frames) >= num_support:
            break
    frames.reverse()
    feats = []
    for t in frames:
        img = load_rgb_frame(seq_path, t)
        if img is None:
            continue
        xy = trajs_2d[t, point_idx]
        if not np.all(np.isfinite(xy)):
            continue
        crop = extract_crop(img, xy, patch_size)
        feat = extractor.feature_map(crop).mean(dim=[1, 2])
        feats.append(feat)
    if not feats:
        return None, []
    return torch.stack(feats, dim=0), frames


def dense_global_match_map(
    extractor: DINOFeatureExtractor,
    query_img: np.ndarray,
    support_feat: torch.Tensor,
    query_xy: np.ndarray,
    patch_size: int,
) -> Tuple[torch.Tensor, int]:
    """
    Build dense patch score map over the whole frame using query patch templates
    matched to the full-frame DINO feature map.
    """
    q_crop = extract_crop(query_img, query_xy, patch_size)
    q_feat = extractor.feature_map(q_crop)
    q_center_xy = np.array([patch_size / 2.0, patch_size / 2.0], dtype=np.float32)
    # Keep each radius separate because the template spatial size differs.
    q_templates = [extract_template(q_feat, q_center_xy, patch_size, radius) for radius in (1, 2)]
    return q_templates, q_feat.shape[-1]


def score_full_frame(
    extractor: DINOFeatureExtractor,
    query_img: np.ndarray,
    reentry_img: np.ndarray,
    query_xy: np.ndarray,
    patch_size: int,
    support_feat: torch.Tensor,
    support_mode: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_templates, _ = dense_global_match_map(extractor, query_img, support_feat, query_xy, patch_size)
    r_feat = extractor.feature_map(reentry_img)
    score_maps = [template_score_map(t, r_feat) for t in q_templates]
    dino_map = torch.stack(score_maps, dim=0).mean(dim=0)

    # Lightweight support readout: dot each spatial feature with support prototypes.
    _, h, w = r_feat.shape
    support_norm = F.normalize(support_feat, dim=1)  # (S, D)
    local_scores = torch.einsum("chw,sd->shw", r_feat, support_norm)
    if support_mode == "max":
        support_map = local_scores.max(dim=0).values
    else:
        support_map = local_scores.mean(dim=0)
    return dino_map, support_map


def topk_coords(score_map: torch.Tensor, topk: int, image_width: int, image_height: int) -> np.ndarray:
    h, w = score_map.shape
    vals, idx = torch.topk(score_map.reshape(-1), k=min(topk, score_map.numel()))
    coords = []
    for i in idx.tolist():
        y = i // w
        x = i % w
        coords.append([(x + 0.5) * image_width / float(w), (y + 0.5) * image_height / float(h)])
    return np.asarray(coords, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-frame dense retrieval for PRT")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-json", type=str, default="/gemini/code/FSPT/outputs/global_retrieval_smoke.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    sequences = discover_sequences(Path(args.data_root), [s.strip() for s in args.splits.split(",") if s.strip()])
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    results: List[Dict] = []
    for seq_path in sequences:
        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        trajs_3d = seq["trajs_3d"]
        visibs = seq["visibs"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        collected = 0
        for q in queries:
            if collected >= args.max_samples:
                break
            t_q, t_re, i = q.query_frame, q.reentry_frame, q.point_idx
            e_rel = extrinsics[t_re] @ np.linalg.inv(extrinsics[t_q])
            cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
            if cam_motion < args.min_camera_motion:
                continue

            query_img = load_rgb_frame(seq_path, t_q)
            reentry_img = load_rgb_frame(seq_path, t_re)
            if query_img is None or reentry_img is None:
                continue

            support_feat, support_frames = extract_support_descriptor(
                extractor,
                seq_path,
                i,
                t_q,
                trajs_2d,
                visibs,
                num_support=args.num_support,
                patch_size=args.patch_size,
            )
            if support_feat is None:
                continue

            gt_xy = trajs_2d[t_re, i].astype(np.float32)
            query_xy = trajs_2d[t_q, i].astype(np.float32)
            baseline_xy = gt_xy.copy()
            # baseline is only used as a reference; retrieval itself is whole-frame.

            dino_map, support_map = score_full_frame(
                extractor=extractor,
                query_img=query_img,
                reentry_img=reentry_img,
                query_xy=query_xy,
                patch_size=args.query_crop_size,
                support_feat=support_feat,
                support_mode="mean",
            )
            joint_map = 0.6 * dino_map + 0.4 * support_map

            # top-k recall at GT location
            gt_errs = []
            image_height, image_width = reentry_img.shape[:2]
            for sm in [dino_map, support_map, joint_map]:
                coords = topk_coords(sm, topk=args.topk, image_width=image_width, image_height=image_height)
                dists = np.linalg.norm(coords - gt_xy[None, :], axis=1)
                gt_errs.append(float(dists.min()))

            results.append(
                {
                    "seq": seq_path.name,
                    "point_idx": int(i),
                    "query_frame": int(t_q),
                    "reentry_frame": int(t_re),
                    "occ_length": int(q.occ_length),
                    "camera_motion": cam_motion,
                    "support_frames": support_frames,
                    "gt_xy": gt_xy.tolist(),
                    "topk_gt_dist_dino": gt_errs[0],
                    "topk_gt_dist_support": gt_errs[1],
                    "topk_gt_dist_joint": gt_errs[2],
                }
            )
            collected += 1

    if not results:
        raise RuntimeError("No valid samples collected.")

    dino_recall = float(np.mean([r["topk_gt_dist_dino"] <= 8.0 for r in results]))
    support_recall = float(np.mean([r["topk_gt_dist_support"] <= 8.0 for r in results]))
    joint_recall = float(np.mean([r["topk_gt_dist_joint"] <= 8.0 for r in results]))
    summary = {
        "n": len(results),
        "top8_recall": {
            "dino": dino_recall,
            "support": support_recall,
            "joint": joint_recall,
        },
        "median_topk_gt_dist": {
            "dino": float(np.median([r["topk_gt_dist_dino"] for r in results])),
            "support": float(np.median([r["topk_gt_dist_support"] for r in results])),
            "joint": float(np.median([r["topk_gt_dist_joint"] for r in results])),
        },
    }

    out = {
        "config": vars(args),
        "summary": summary,
        "results": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
