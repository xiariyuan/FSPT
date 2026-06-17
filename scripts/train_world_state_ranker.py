#!/usr/bin/env python3
"""
Stage 2: Patch-Pair Candidate Ranker for causal world-state re-entry.

Instead of low-dimensional hand-crafted features, this ranker directly takes
image patches (query, candidate location, baseline location) and learns to
rank candidates by quality.

Design:
  - For each candidate, extract 3 patches: query, candidate-location, baseline-location
  - Shared CNN encodes each patch
  - Score = MLP(query_enc || candidate_enc || baseline_enc || geometry_scalars)
  - Training: pairwise margin ranking loss (if A better than B, score_A > score_B + margin)
  - Inference: pick highest-scoring candidate, fallback to baseline if score < threshold
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
import torch.nn as nn
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
    extract_template,
    load_rgb_frame,
    template_score_map,
    rgb_patch_ncc,
    xy_to_feat_idx,
)


# ---------------------------------------------------------------------------
# Data: collect candidates with image patches
# ---------------------------------------------------------------------------

def collect_candidates_with_patches(
    extractor: DINOFeatureExtractor,
    seq_path: Path,
    queries,
    seq_data: dict,
    depth_noise_sigma: float,
    rng: np.random.Generator,
    topk: int = 5,
    query_crop_size: int = 112,
    search_crop_size: int = 224,
    patch_size: int = 64,
    min_cam_motion: float = 0.20,
    max_samples: int = 0,
) -> List[dict]:
    """Collect candidates with pre-extracted image patches for ranker training."""
    trajs_2d = seq_data["trajs_2d"]
    trajs_3d = seq_data["trajs_3d"]
    intrinsics = seq_data["intrinsics"]
    extrinsics = seq_data["extrinsics"]

    print(f"  Pre-filtering {len(queries)} queries...", flush=True)
    filtered = []
    for q in queries:
        t_q, t_re = q.query_frame, q.reentry_frame
        E_rel = extrinsics[t_re] @ np.linalg.inv(extrinsics[t_q])
        cam_motion = float(np.linalg.norm(E_rel[:3, :3] - np.eye(3)))
        if cam_motion >= min_cam_motion:
            filtered.append((q, cam_motion))
    print(f"  {len(filtered)}/{len(queries)} passed filter", flush=True)

    samples = []
    for qi, (q, cam_motion) in enumerate(filtered):
        if qi % 20 == 0:
            print(f"    {qi}/{len(filtered)}, collected {len(samples)}...", flush=True)
        t_q, t_re, i = q.query_frame, q.reentry_frame, q.point_idx

        query_img = load_rgb_frame(seq_path, t_q)
        reentry_img = load_rgb_frame(seq_path, t_re)
        if query_img is None or reentry_img is None:
            continue

        hold_3d = trajs_3d[t_q, i].astype(np.float32)
        pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
        z_depth = float(pt_cam[2])
        if not np.isfinite(z_depth) or z_depth <= 1e-6:
            continue

        eps = float(np.clip(rng.normal(0.0, depth_noise_sigma), -0.35, 0.35))
        noisy_depth = z_depth * math.exp(eps)
        pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
        k_inv = np.linalg.inv(intrinsics[t_q])
        pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
        e_inv = np.linalg.inv(extrinsics[t_q])
        noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
        baseline_xy = project_3d_to_2d(noisy_world, intrinsics[t_re], extrinsics[t_re]).astype(np.float32)
        gt_xy = trajs_2d[t_re, i].astype(np.float32)
        baseline_err = float(np.linalg.norm(baseline_xy - gt_xy))

        # DINO search
        q_crop = extract_crop(query_img, trajs_2d[t_q, i].astype(np.float32), query_crop_size)
        r_crop = extract_crop(reentry_img, baseline_xy, search_crop_size)
        q_feat = extractor.feature_map(q_crop)
        r_feat = extractor.feature_map(r_crop)
        _, hr, wr = r_feat.shape

        q_center_xy = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)
        query_templates = []
        for radius in [1, 2]:
            query_templates.append(extract_template(q_feat, q_center_xy, query_crop_size, radius))
        score_maps = [template_score_map(tmpl, r_feat) for tmpl in query_templates]
        sims = torch.stack(score_maps, dim=0).mean(dim=0)

        k_actual = min(topk, sims.numel())
        top_vals, top_idx = torch.topk(sims.reshape(-1), k=k_actual)
        patch_scale = search_crop_size / float(wr)
        center_offset = np.array([search_crop_size / 2.0, search_crop_size / 2.0], dtype=np.float32)

        # Extract patches for each candidate and baseline
        # query patch (always the same for this sample)
        query_patch = extract_crop(query_img, trajs_2d[t_q, i].astype(np.float32), patch_size)
        # baseline patch in reentry frame
        baseline_patch = extract_crop(reentry_img, baseline_xy, patch_size)

        candidates = []
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist())):
            by = idx // wr
            bx = idx % wr
            offset_xy = np.array([(bx + 0.5) * patch_scale, (by + 0.5) * patch_scale], dtype=np.float32)
            pred_xy = baseline_xy - center_offset + offset_xy
            cand_err = float(np.linalg.norm(pred_xy - gt_xy))

            cand_patch = extract_crop(reentry_img, pred_xy, patch_size)

            candidates.append({
                "dino_score": float(val),
                "pred_xy": pred_xy.tolist(),
                "error_px": cand_err,
                "is_better": float(cand_err < baseline_err),
                "rank": rank,
                "cand_patch": cand_patch,  # (H, W, 3) uint8
            })

        # Pre-encode patches as float arrays (normalized)
        query_patch_f = (query_patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
        baseline_patch_f = (baseline_patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
        cand_patches_f = []
        for c in candidates:
            p = c["cand_patch"]
            if p is None or p.size == 0:
                p = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            cand_patches_f.append((p.astype(np.float32) / 255.0).transpose(2, 0, 1))

        samples.append({
            "query_patch": query_patch_f,
            "baseline_patch": baseline_patch_f,
            "baseline_err": baseline_err,
            "baseline_xy": baseline_xy.tolist(),
            "gt_xy": gt_xy.tolist(),
            "occ_length": int(q.occ_length),
            "camera_motion": cam_motion,
            "cand_patches": np.stack(cand_patches_f, axis=0),  # (K, 3, H, W)
            "cand_dino_scores": np.array([c["dino_score"] for c in candidates], dtype=np.float32),
            "cand_errors": np.array([c["error_px"] for c in candidates], dtype=np.float32),
            "cand_is_better": np.array([c["is_better"] for c in candidates], dtype=np.float32),
        })

        if max_samples > 0 and len(samples) >= max_samples:
            print(f"  Reached {max_samples} samples.", flush=True)
            break

    return samples


# ---------------------------------------------------------------------------
# Patch-Pair Candidate Ranker
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """Shared CNN to encode a 3x64x64 patch into a vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, out_dim)"""
        return self.proj(self.conv(x).flatten(1))


class PatchPairRanker(nn.Module):
    """
    Ranks candidates by scoring (query_patch, candidate_patch, baseline_patch) triples.
    """

    def __init__(self, patch_feat_dim: int = 64, hidden_dim: int = 128, geom_dim: int = 2):
        super().__init__()
        self.patch_enc = PatchEncoder(out_dim=patch_feat_dim)
        # Score = MLP(query_feat || cand_feat || baseline_feat || geom)
        score_in_dim = patch_feat_dim * 3 + geom_dim
        self.score_head = nn.Sequential(
            nn.Linear(score_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def score_candidate(
        self,
        query_feat: torch.Tensor,
        cand_feat: torch.Tensor,
        baseline_feat: torch.Tensor,
        geom: torch.Tensor,
    ) -> torch.Tensor:
        """Score a single candidate. All inputs (B, dim). Returns (B, 1)."""
        x = torch.cat([query_feat, cand_feat, baseline_feat, geom], dim=-1)
        return self.score_head(x)

    def forward(
        self,
        query_patches: torch.Tensor,
        cand_patches: torch.Tensor,
        baseline_patches: torch.Tensor,
        cand_dino_scores: torch.Tensor,
        cand_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_patches: (B, 3, H, W) - same for all candidates
            cand_patches: (B, K, 3, H, W) - candidate patches
            baseline_patches: (B, 3, H, W)
            cand_dino_scores: (B, K)
            cand_count: (B,) int - actual number of candidates per sample
        Returns:
            scores: (B, K+1) - baseline at index 0, candidates at 1..K
        """
        B, K = cand_patches.shape[:2]
        device = query_patches.device

        # Encode all patches
        q_feat = self.patch_enc(query_patches)  # (B, D)
        b_feat = self.patch_enc(baseline_patches)  # (B, D)

        # Flatten candidates for batch encoding
        cand_flat = cand_patches.reshape(B * K, *cand_patches.shape[2:])
        c_feat = self.patch_enc(cand_flat).reshape(B, K, -1)  # (B, K, D)

        # Baseline score (geom: dino_score=0, dist=0 for baseline)
        baseline_geom = torch.zeros(B, 2, device=device)
        baseline_score = self.score_candidate(q_feat, b_feat, b_feat, baseline_geom)  # (B, 1)

        # Candidate scores
        cand_scores = []
        for k in range(K):
            geom = torch.stack([
                cand_dino_scores[:, k],
                torch.zeros(B, device=device),  # dist placeholder
            ], dim=-1)
            s = self.score_candidate(q_feat, c_feat[:, k], b_feat, geom)  # (B, 1)
            cand_scores.append(s)
        cand_scores = torch.cat(cand_scores, dim=1)  # (B, K)

        # Mask out invalid candidates
        valid_mask = torch.arange(K, device=device).unsqueeze(0) < cand_count.unsqueeze(1)
        cand_scores = cand_scores.masked_fill(~valid_mask, -1e9)

        all_scores = torch.cat([baseline_score, cand_scores], dim=1)  # (B, K+1)
        return all_scores


class FrozenDINOSelector(nn.Module):
    """
    Select baseline vs candidates from frozen DINO patch embeddings + geometry.
    """

    def __init__(self, dino_feat_dim: int = 384, geom_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dino_feat_dim * 3 + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        baseline_feat: torch.Tensor,
        cand_feats: torch.Tensor,
        geom_feats: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = cand_feats.shape
        q = query_feat.unsqueeze(1).expand(B, N, D)
        b = baseline_feat.unsqueeze(1).expand(B, N, D)
        x = torch.cat([q, cand_feats, b, geom_feats], dim=-1)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training with pairwise margin ranking loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(scores: torch.Tensor, errors: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    For each pair (i, j) where error_i < error_j, enforce score_i > score_j + margin.

    scores: (B, K+1) - baseline + candidates
    errors: (B, K+1) - errors (lower is better)
    """
    B, N = scores.shape
    loss_sum = torch.tensor(0.0, device=scores.device)
    count = 0
    for b in range(B):
        s = scores[b]  # (N,)
        e = errors[b]  # (N,)
        valid = torch.isfinite(e)
        if valid.sum() < 2:
            continue
        s_v = s[valid]
        e_v = e[valid]
        n_v = len(s_v)
        # All pairs
        for i in range(n_v):
            for j in range(i + 1, n_v):
                if e_v[i] < e_v[j]:
                    # i should score higher than j
                    loss_sum = loss_sum + F.relu(margin - (s_v[i] - s_v[j]))
                elif e_v[j] < e_v[i]:
                    loss_sum = loss_sum + F.relu(margin - (s_v[j] - s_v[i]))
                count += 1
    if count > 0:
        return loss_sum / count
    return loss_sum


def batched_pairwise_ranking_loss(scores: torch.Tensor, errors: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Vectorized pairwise ranking loss.

    scores: (B, N), errors: (B, N) where lower error = better.
    For each valid pair (i,j) where i is better, enforce score_i > score_j + margin.
    """
    B, N = scores.shape

    # Mask invalid entries (inf error = padding)
    e_masked = errors.clone()
    valid_mask = torch.isfinite(errors)
    e_masked[~valid_mask] = 1e9

    # Pairwise differences
    e_diff = e_masked.unsqueeze(2) - e_masked.unsqueeze(1)  # (B, N, N): negative if i is better
    s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (B, N, N)

    # target_sign: +1 if i is better, -1 if j is better
    target_sign = torch.sign(-e_diff)
    loss_per_pair = F.relu(margin - target_sign * s_diff)  # (B, N, N)

    # Valid pair mask: both must be valid, and errors must differ
    both_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # (B, N, N)
    diff_pair = e_diff.abs() > 1e-6  # (B, N, N)
    # Upper triangle only to avoid double counting
    triu = torch.triu(torch.ones(N, N, device=scores.device, dtype=torch.bool), diagonal=1)
    mask = both_valid & diff_pair & triu.unsqueeze(0)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return (loss_per_pair * mask.float()).sum() / mask.float().sum()


def evaluate_ranker(model: PatchPairRanker, samples: List[dict], device: torch.device,
                    patch_size: int = 64) -> Dict[str, float]:
    model.eval()
    pred_errs, base_errs, oracle_errs = [], [], []
    chosen_indices = []

    with torch.no_grad():
        for s in samples:
            K = len(s["cand_errors"])
            if K == 0:
                pred_errs.append(s["baseline_err"])
                base_errs.append(s["baseline_err"])
                oracle_errs.append(s["baseline_err"])
                chosen_indices.append(0)
                continue

            q_patch = torch.from_numpy(s["query_patch"]).float().unsqueeze(0).to(device)
            b_patch = torch.from_numpy(s["baseline_patch"]).float().unsqueeze(0).to(device)
            c_patches = torch.from_numpy(s["cand_patches"]).float().unsqueeze(0).to(device)
            dino_scores = torch.from_numpy(s["cand_dino_scores"]).float().unsqueeze(0).to(device)
            cand_count = torch.tensor([K], device=device)

            all_scores = model(q_patch, c_patches, b_patch, dino_scores, cand_count)
            idx = int(all_scores[0].argmax().item())

            # errors: [baseline_error, cand0_error, cand1_error, ...]
            all_errors = [s["baseline_err"]] + [float(e) for e in s["cand_errors"]]
            pred_errs.append(all_errors[idx])
            base_errs.append(s["baseline_err"])
            oracle_errs.append(min(all_errors))
            chosen_indices.append(idx)

    pred = np.array(pred_errs)
    base = np.array(base_errs)
    oracle = np.array(oracle_errs)
    chosen = np.array(chosen_indices)

    return {
        "ranker_pred_median": float(np.median(pred)),
        "ranker_baseline_median": float(np.median(base)),
        "ranker_oracle_median": float(np.median(oracle)),
        "ranker_better_frac": float(np.mean(pred < base)),
        "ranker_pred_lt4px": float(np.mean(pred < 4.0)),
        "ranker_baseline_lt4px": float(np.mean(base < 4.0)),
        "ranker_oracle_lt4px": float(np.mean(oracle < 4.0)),
        "ranker_choose_baseline_frac": float(np.mean(chosen == 0)),
    }


def encode_patch_batch_dino(extractor: DINOFeatureExtractor, patches: torch.Tensor) -> torch.Tensor:
    """patches: (N, 3, H, W) in [0,1] -> (N, 384) normalized"""
    with torch.no_grad():
        x = F.interpolate(patches, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
        feats = extractor.model(x)[-1].mean(dim=[-2, -1]).float()
        feats = F.normalize(feats, dim=-1)
    return feats


def prepare_dino_selector_sample(
    extractor: DINOFeatureExtractor,
    sample: dict,
    device: torch.device,
) -> dict:
    q = torch.from_numpy(sample["query_patch"]).float().unsqueeze(0).to(device)
    b = torch.from_numpy(sample["baseline_patch"]).float().unsqueeze(0).to(device)
    c = torch.from_numpy(sample["cand_patches"]).float().to(device)

    q_feat = encode_patch_batch_dino(extractor, q)[0].cpu()
    b_feat = encode_patch_batch_dino(extractor, b)[0].cpu()
    c_feat = encode_patch_batch_dino(extractor, c).cpu()

    K = len(sample["cand_errors"])
    geom_rows = [[
        0.0,
        sample["baseline_err"] / 256.0,
        sample["occ_length"] / 300.0,
        sample["camera_motion"],
    ]]
    for i in range(K):
        geom_rows.append([
            float(sample["cand_dino_scores"][i]),
            sample["baseline_err"] / 256.0,
            sample["occ_length"] / 300.0,
            sample["camera_motion"],
        ])

    errors = torch.tensor(
        [sample["baseline_err"]] + [float(e) for e in sample["cand_errors"]],
        dtype=torch.float32,
    )
    target = int(torch.argmin(errors).item())
    cand_feats = torch.cat([b_feat.unsqueeze(0), c_feat], dim=0)
    geom_feats = torch.tensor(geom_rows, dtype=torch.float32)
    return {
        "query_feat": q_feat,
        "baseline_feat": b_feat,
        "cand_feats": cand_feats,
        "geom_feats": geom_feats,
        "errors": errors,
        "target": target,
    }


def evaluate_dino_selector(model: FrozenDINOSelector, dataset: List[dict], device: torch.device) -> Dict[str, float]:
    model.eval()
    pred_errs, base_errs, oracle_errs, chosen = [], [], [], []
    with torch.no_grad():
        for s in dataset:
            q = s["query_feat"].unsqueeze(0).to(device)
            b = s["baseline_feat"].unsqueeze(0).to(device)
            c = s["cand_feats"].unsqueeze(0).to(device)
            g = s["geom_feats"].unsqueeze(0).to(device)
            logits = model(q, b, c, g)[0]
            idx = int(torch.argmax(logits).item())
            errs = s["errors"].numpy()
            pred_errs.append(float(errs[idx]))
            base_errs.append(float(errs[0]))
            oracle_errs.append(float(errs.min()))
            chosen.append(idx)

    pred = np.asarray(pred_errs)
    base = np.asarray(base_errs)
    oracle = np.asarray(oracle_errs)
    chosen = np.asarray(chosen)
    return {
        "dino_selector_pred_median": float(np.median(pred)),
        "dino_selector_baseline_median": float(np.median(base)),
        "dino_selector_oracle_median": float(np.median(oracle)),
        "dino_selector_better_frac": float(np.mean(pred < base)),
        "dino_selector_pred_lt4px": float(np.mean(pred < 4.0)),
        "dino_selector_baseline_lt4px": float(np.mean(base < 4.0)),
        "dino_selector_oracle_lt4px": float(np.mean(oracle < 4.0)),
        "dino_selector_choose_baseline_frac": float(np.mean(chosen == 0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-Pair Candidate Ranker")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--train-splits", type=str, default="train")
    parser.add_argument("--val-splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--train-max-sequences", type=int, default=1)
    parser.add_argument("--val-max-sequences", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=200)
    parser.add_argument("--val-max-samples", type=int, default=100)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--patch-feat-dim", type=int, default=64)
    parser.add_argument("--mode", type=str, default="ranker", choices=["ranker", "dino_selector"])
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str, default="/gemini/code/FSPT/outputs/world_state_ranker_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DINOFeatureExtractor(Path(args.weights), device)

    # Collect training data
    print("Collecting training data...", flush=True)
    train_seqs = discover_sequences(Path(args.data_root), args.train_splits.split(","))
    if args.train_max_sequences > 0:
        train_seqs = train_seqs[:args.train_max_sequences]
    train_samples = []
    for sp in train_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates_with_patches(
            extractor, sp, queries, seq, args.depth_noise_sigma,
            np.random.default_rng(42), topk=args.topk,
            query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            patch_size=args.patch_size, min_cam_motion=args.min_camera_motion,
            max_samples=args.train_max_samples - len(train_samples),
        )
        train_samples.extend(cands)
        if len(train_samples) >= args.train_max_samples:
            break
    print(f"  Train: {len(train_samples)} samples", flush=True)

    # Collect validation data
    print("Collecting validation data...", flush=True)
    val_seqs = discover_sequences(Path(args.data_root), args.val_splits.split(","))
    if args.val_max_sequences > 0:
        val_seqs = val_seqs[:args.val_max_sequences]
    val_samples = []
    for sp in val_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates_with_patches(
            extractor, sp, queries, seq, args.depth_noise_sigma,
            np.random.default_rng(999), topk=args.topk,
            query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            patch_size=args.patch_size, min_cam_motion=args.min_camera_motion,
            max_samples=args.val_max_samples - len(val_samples),
        )
        val_samples.extend(cands)
        if len(val_samples) >= args.val_max_samples:
            break
    print(f"  Val: {len(val_samples)} samples", flush=True)

    if not train_samples or not val_samples:
        raise RuntimeError("No samples collected.")

    # Print oracle stats
    all_base = np.array([s["baseline_err"] for s in train_samples + val_samples])
    all_oracle = []
    for s in train_samples + val_samples:
        errs = [s["baseline_err"]] + [float(e) for e in s["cand_errors"]]
        all_oracle.append(min(errs))
    all_oracle = np.array(all_oracle)
    print(f"\nOracle stats: baseline_median={np.median(all_base):.2f}, oracle_median={np.median(all_oracle):.2f}", flush=True)
    print(f"  baseline <4px: {np.mean(all_base < 4):.1%}, oracle <4px: {np.mean(all_oracle < 4):.1%}", flush=True)

    if args.mode == "dino_selector":
        print("Precomputing frozen DINO selector features...", flush=True)
        train_prepared = [prepare_dino_selector_sample(extractor, s, device) for s in train_samples]
        val_prepared = [prepare_dino_selector_sample(extractor, s, device) for s in val_samples]
        model = FrozenDINOSelector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        model = PatchPairRanker(patch_feat_dim=args.patch_feat_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    history = []
    best_metric = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = np.random.permutation(len(train_samples))

        if args.mode == "dino_selector":
            for sid in perm:
                s = train_prepared[int(sid)]
                q = s["query_feat"].unsqueeze(0).to(device)
                b = s["baseline_feat"].unsqueeze(0).to(device)
                c = s["cand_feats"].unsqueeze(0).to(device)
                g = s["geom_feats"].unsqueeze(0).to(device)
                y = torch.tensor([s["target"]], dtype=torch.long, device=device)
                errs = s["errors"].to(device)
                logits = model(q, b, c, g)
                loss_ce = F.cross_entropy(logits, y)
                probs = torch.softmax(logits[0], dim=0)
                norm_err = errs / errs.max().clamp(min=1.0)
                loss_cost = (probs * norm_err).sum()
                loss = loss_ce + 0.5 * loss_cost
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss)
                n_batches += 1
        else:
            for start in range(0, len(perm), args.batch_size):
                batch_indices = perm[start:start + args.batch_size]
                batch_samples = [train_samples[idx] for idx in batch_indices]

                # Build batch tensors
                K_max = max(len(s["cand_errors"]) for s in batch_samples)
                if K_max == 0:
                    continue

                q_patches = []
                b_patches = []
                c_patches_list = []
                dino_scores_list = []
                cand_counts = []
                all_errors_list = []

                for s in batch_samples:
                    K = len(s["cand_errors"])
                    q_patches.append(torch.from_numpy(s["query_patch"]).float())
                    b_patches.append(torch.from_numpy(s["baseline_patch"]).float())

                    c_p = torch.from_numpy(s["cand_patches"]).float()  # (K, 3, H, W)
                    if K < K_max:
                        pad = torch.zeros(K_max - K, *c_p.shape[1:])
                        c_p = torch.cat([c_p, pad], dim=0)
                    c_patches_list.append(c_p)

                    ds = torch.from_numpy(s["cand_dino_scores"]).float()
                    if K < K_max:
                        ds = torch.cat([ds, torch.zeros(K_max - K)])
                    dino_scores_list.append(ds)

                    cand_counts.append(K)

                    # errors: [baseline, cand0, cand1, ...]
                    errs = [s["baseline_err"]] + [float(e) for e in s["cand_errors"]]
                    if len(errs) < K_max + 1:
                        errs.extend([float('inf')] * (K_max + 1 - len(errs)))
                    all_errors_list.append(errs)

                q_batch = torch.stack(q_patches).to(device)
                b_batch = torch.stack(b_patches).to(device)
                c_batch = torch.stack(c_patches_list).to(device)
                d_batch = torch.stack(dino_scores_list).to(device)
                counts = torch.tensor(cand_counts, device=device)
                err_batch = torch.tensor(all_errors_list, device=device)

                scores = model(q_batch, c_batch, b_batch, d_batch, counts)
                loss = batched_pairwise_ranking_loss(scores, err_batch, margin=args.margin)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += float(loss)
                n_batches += 1

        scheduler.step()

        # Evaluate
        if args.mode == "dino_selector":
            metrics = evaluate_dino_selector(model, val_prepared, device)
            metric = metrics.get("dino_selector_better_frac", 0.0)
        else:
            metrics = evaluate_ranker(model, val_samples, device, patch_size=args.patch_size)
            metric = metrics.get("ranker_better_frac", 0.0)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = epoch_loss / max(n_batches, 1)
        metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), output_dir / ("dino_selector_best.pt" if args.mode == "dino_selector" else "ranker_best.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break

    # Save
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "val_samples.json").write_text(json.dumps(val_samples[:50], indent=2, default=str) + "\n")

    if args.mode == "dino_selector":
        best = max(history, key=lambda h: h.get("dino_selector_better_frac", 0))
    else:
        best = max(history, key=lambda h: h.get("ranker_better_frac", 0))
    print(f"\nBest epoch: {json.dumps(best)}", flush=True)
    print(f"Saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
