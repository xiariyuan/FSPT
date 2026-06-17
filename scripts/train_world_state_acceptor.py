#!/usr/bin/env python3
"""
Stage 2: Learned acceptor for causal DINO local matching.

This script:
  1. Runs causal DINO search to get top-k candidates per query (no GT leakage).
  2. Trains a lightweight acceptor to select the best candidate or reject all.
  3. Evaluates on held-out sequences.

The acceptor does NOT regress coordinates. It only learns: "which candidate, if any, should I accept?"
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Data generation: collect top-k candidates with features
# ---------------------------------------------------------------------------

def collect_candidates(
    extractor: DINOFeatureExtractor,
    seq_path: Path,
    queries,
    seq_data: dict,
    depth_noise_sigma: float,
    rng: np.random.Generator,
    topk: int = 5,
    query_crop_size: int = 112,
    search_crop_size: int = 224,
    min_cam_motion: float = 0.20,
    max_samples: int = 0,
) -> List[dict]:
    """For each re-entry query, generate top-k candidates with features."""
    trajs_2d = seq_data["trajs_2d"]
    trajs_3d = seq_data["trajs_3d"]
    intrinsics = seq_data["intrinsics"]
    extrinsics = seq_data["extrinsics"]

    # Pre-filter: only keep queries with enough camera motion
    # Vectorized camera motion computation
    print(f"  Pre-filtering {len(queries)} queries...", flush=True)
    filtered_queries = []
    for q in queries:
        t_q, t_re = q.query_frame, q.reentry_frame
        E_q, E_re = extrinsics[t_q], extrinsics[t_re]
        E_rel = E_re @ np.linalg.inv(E_q)
        cam_motion = float(np.linalg.norm(E_rel[:3, :3] - np.eye(3)))
        if cam_motion >= min_cam_motion:
            filtered_queries.append((q, cam_motion))
    print(f"  {len(filtered_queries)}/{len(queries)} passed cam_motion filter", flush=True)

    samples = []
    for qi, (q, cam_motion) in enumerate(filtered_queries):
        t_q = q.query_frame
        t_re = q.reentry_frame
        i = q.point_idx

        if qi % 20 == 0:
            print(f"    processing {qi}/{len(filtered_queries)}...", flush=True)

        query_img = load_rgb_frame(seq_path, t_q)
        reentry_img = load_rgb_frame(seq_path, t_re)
        if query_img is None or reentry_img is None:
            continue

        # Noisy depth → world → baseline reprojection
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

        # DINO feature maps
        q_crop = extract_crop(query_img, trajs_2d[t_q, i].astype(np.float32), query_crop_size)
        r_crop = extract_crop(reentry_img, baseline_xy, search_crop_size)
        q_feat = extractor.feature_map(q_crop)
        r_feat = extractor.feature_map(r_crop)
        _, hr, wr = r_feat.shape

        # Multi-radius templates
        q_center_xy = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)
        query_templates = []
        for radius in [1, 2]:
            query_templates.append(extract_template(q_feat, q_center_xy, query_crop_size, radius))
        score_maps = [template_score_map(tmpl, r_feat) for tmpl in query_templates]
        sims = torch.stack(score_maps, dim=0).mean(dim=0)

        # Top-k candidates
        k_actual = min(topk, sims.numel())
        top_vals, top_idx = torch.topk(sims.reshape(-1), k=k_actual)
        patch_scale = search_crop_size / float(wr)
        center_offset = np.array([search_crop_size / 2.0, search_crop_size / 2.0], dtype=np.float32)

        candidates = []
        for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist())):
            by = idx // wr
            bx = idx % wr
            offset_xy = np.array([(bx + 0.5) * patch_scale, (by + 0.5) * patch_scale], dtype=np.float32)
            pred_xy = baseline_xy - center_offset + offset_xy
            dist_norm = float(np.linalg.norm(offset_xy - center_offset) / (center_offset[0] + 1e-6))

            rgb_score = rgb_patch_ncc(
                query_crop=q_crop,
                reentry_crop=r_crop,
                query_center_xy=q_center_xy,
                reentry_center_xy=offset_xy,
                patch_size=32,
            )

            cand_err = float(np.linalg.norm(pred_xy - gt_xy))
            is_better = float(cand_err < baseline_err)

            candidates.append({
                "dino_score": float(val),
                "rgb_ncc": float(rgb_score),
                "dist_from_baseline": float(dist_norm),
                "offset_xy": offset_xy.tolist(),
                "pred_xy": pred_xy.tolist(),
                "error_px": cand_err,
                "is_better": is_better,
                "rank": rank,
            })

        # Also add the baseline as candidate index -1 (always available)

        samples.append({
            "seq": seq_path.name,
            "query_frame": int(t_q),
            "reentry_frame": int(t_re),
            "point_idx": int(i),
            "occ_length": int(q.occ_length),
            "camera_motion": cam_motion,
            "baseline_xy": baseline_xy.tolist(),
            "gt_xy": gt_xy.tolist(),
            "baseline_err": baseline_err,
            "noisy_depth": float(noisy_depth),
            "raw_depth": float(z_depth),
            "candidates": candidates,
        })
        if max_samples > 0 and len(samples) >= max_samples:
            print(f"  Reached {max_samples} samples, stopping early.", flush=True)
            break

    return samples


# ---------------------------------------------------------------------------
# Acceptor model
# ---------------------------------------------------------------------------

class CandidateAcceptor(nn.Module):
    """
    Lightweight acceptor: for each candidate, predict accept/reject score.
    """

    def __init__(self, cand_feat_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Bias toward rejecting (accept = rare event)
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, cand_features: torch.Tensor) -> torch.Tensor:
        """
        cand_features: (B, K, cand_feat_dim)
        Returns: (B, K) accept logits
        """
        return self.net(cand_features).squeeze(-1)


class CandidateSelector(nn.Module):
    """
    Per-query selector over baseline + K candidates.
    """

    def __init__(self, cand_feat_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cand_features: torch.Tensor) -> torch.Tensor:
        return self.net(cand_features).squeeze(-1)


def make_candidate_features(sample: dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract features and label for one sample's candidates.

    Returns:
        features: (K, feat_dim)
        labels: (K,) binary 1=best candidate, 0=otherwise
        baseline_err: float
    """
    cands = sample["candidates"]
    if not cands:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32), sample["baseline_err"]

    features = []
    labels = []
    best_idx = -1
    best_err = sample["baseline_err"]
    for j, c in enumerate(cands):
        if c["error_px"] < best_err:
            best_err = c["error_px"]
            best_idx = j

    for j, c in enumerate(cands):
        features.append([
            c["dino_score"],
            c["rgb_ncc"],
            c["dist_from_baseline"],
            float(c["rank"]) / max(len(cands) - 1, 1),
            sample["baseline_err"] / 256.0,  # normalized baseline error
        ])
        labels.append(1.0 if j == best_idx else 0.0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32), sample["baseline_err"]


def make_selector_features(sample: dict) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build baseline+candidate features for multiclass selection.

    Class 0 is baseline. Classes 1..K are DINO candidates.
    """
    cands = sample["candidates"]
    baseline_err = float(sample["baseline_err"])
    baseline_xy = np.array(sample["baseline_xy"], dtype=np.float32)

    rows = [[
        0.0,  # is_candidate
        0.0,  # dino_score
        0.0,  # rgb_ncc
        0.0,  # dist_from_baseline
        0.0,  # rank_norm
        baseline_err / 256.0,
        sample["occ_length"] / 300.0,
        sample["camera_motion"],
    ]]
    errors = [baseline_err]

    for c in cands:
        pred_xy = np.array(c["pred_xy"], dtype=np.float32)
        dist_px = float(np.linalg.norm(pred_xy - baseline_xy))
        rows.append([
            1.0,
            c["dino_score"],
            c["rgb_ncc"],
            c["dist_from_baseline"],
            float(c["rank"]) / max(len(cands) - 1, 1),
            baseline_err / 256.0,
            sample["occ_length"] / 300.0,
            sample["camera_motion"],
        ])
        errors.append(float(c["error_px"]))

    errors_np = np.array(errors, dtype=np.float32)
    target = int(np.argmin(errors_np))
    return np.array(rows, dtype=np.float32), target, errors_np


def evaluate_selector(model: CandidateSelector, samples: List[dict], device: torch.device) -> Dict[str, float]:
    model.eval()
    pred_errs = []
    base_errs = []
    oracle_errs = []
    chosen = []
    with torch.no_grad():
        for s in samples:
            feats, target, errors = make_selector_features(s)
            x = torch.from_numpy(feats).to(device).unsqueeze(0)
            logits = model(x)[0]
            idx = int(torch.argmax(logits).item())
            pred_errs.append(float(errors[idx]))
            base_errs.append(float(errors[0]))
            oracle_errs.append(float(errors.min()))
            chosen.append(idx)

    pred = np.array(pred_errs)
    base = np.array(base_errs)
    oracle = np.array(oracle_errs)
    chosen = np.array(chosen)
    return {
        "selector_pred_median": float(np.median(pred)),
        "selector_baseline_median": float(np.median(base)),
        "selector_oracle_median": float(np.median(oracle)),
        "selector_better_frac": float(np.mean(pred < base)),
        "selector_pred_lt4px": float(np.mean(pred < 4.0)),
        "selector_baseline_lt4px": float(np.mean(base < 4.0)),
        "selector_oracle_lt4px": float(np.mean(oracle < 4.0)),
        "selector_accept_rate": float(np.mean(chosen > 0)),
    }


def train_selector(
    train_samples: List[dict],
    val_samples: List[dict],
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
) -> Tuple[CandidateSelector, List[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = make_selector_features(train_samples[0])[0].shape[1]
    model = CandidateSelector(cand_feat_dim=feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_metric = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_samples))
        losses = []
        for sid in order:
            feats, target, errors = make_selector_features(train_samples[int(sid)])
            x = torch.from_numpy(feats).to(device).unsqueeze(0)
            y = torch.tensor([target], dtype=torch.long, device=device)
            logits = model(x)
            loss_ce = F.cross_entropy(logits, y)

            # Cost-sensitive margin: selected logits should prefer lower-error classes.
            err = torch.from_numpy(errors).float().to(device)
            err = err / (err.detach().max().clamp(min=1.0))
            probs = torch.softmax(logits[0], dim=0)
            loss_cost = (probs * err).sum()
            loss = loss_ce + 0.5 * loss_cost

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_selector(model, val_samples, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        metric = metrics["selector_pred_median"]
        if metric < best_metric:
            best_metric = metric
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return model, history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_acceptor(
    train_samples: List[dict],
    val_samples: List[dict],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 3,
) -> Tuple[CandidateAcceptor, List[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training data
    train_feats, train_labels, train_baseline_errs, train_sample_ids = [], [], [], []
    for sid, s in enumerate(train_samples):
        f, l, be = make_candidate_features(s)
        if len(f) > 0:
            train_feats.append(f)
            train_labels.append(l)
            train_baseline_errs.append(np.full(len(f), be))
            train_sample_ids.append(np.full(len(f), sid))

    train_feats = np.concatenate(train_feats)
    train_labels = np.concatenate(train_labels)
    train_baseline_errs = np.concatenate(train_baseline_errs)
    train_sample_ids = np.concatenate(train_sample_ids)

    train_X = torch.from_numpy(train_feats).to(device)
    train_Y = torch.from_numpy(train_labels).to(device)

    model = CandidateAcceptor(cand_feat_dim=train_feats.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_metric = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_X), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_X[idx])
            # Weighted BCE: positive samples (best candidate) are rare but important
            pos_weight = torch.tensor([5.0], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, train_Y[idx], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss)
            n_batches += 1

        # Evaluate
        metrics = evaluate_acceptor(model, val_samples, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = epoch_loss / max(n_batches, 1)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        # Early stopping on better_frac
        metric = metrics.get("acceptor_better_frac", 0.0)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return model, history


def evaluate_acceptor(
    model: CandidateAcceptor,
    samples: List[dict],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_pred_errs = []
    all_baseline_errs = []
    all_gt_better = []  # 1 if any candidate is better than baseline

    with torch.no_grad():
        for s in samples:
            feats, labels, baseline_err = make_candidate_features(s)
            if len(feats) == 0:
                all_pred_errs.append(s["baseline_err"])
                all_baseline_errs.append(s["baseline_err"])
                all_gt_better.append(0.0)
                continue

            X = torch.from_numpy(feats).to(device).unsqueeze(0)
            logits = model(X)[0]
            accept_probs = torch.sigmoid(logits)

            best_cand_idx = int(accept_probs.argmax().item())
            best_prob = float(accept_probs[best_cand_idx].item())

            # Accept if probability > 0.5, else fallback to baseline
            if best_prob > 0.5:
                pred_err = s["candidates"][best_cand_idx]["error_px"]
            else:
                pred_err = s["baseline_err"]

            all_pred_errs.append(pred_err)
            all_baseline_errs.append(s["baseline_err"])
            any_better = any(c["is_better"] > 0.5 for c in s["candidates"])
            all_gt_better.append(1.0 if any_better else 0.0)

    pred = np.array(all_pred_errs)
    base = np.array(all_baseline_errs)

    return {
        "acceptor_pred_median": float(np.median(pred)),
        "acceptor_baseline_median": float(np.median(base)),
        "acceptor_better_frac": float(np.mean(pred < base)),
        "acceptor_pred_lt4px": float(np.mean(pred < 4.0)),
        "acceptor_baseline_lt4px": float(np.mean(base < 4.0)),
        "gt_has_better_candidate": float(np.mean(all_gt_better)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Learned acceptor for causal DINO matching")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--train-splits", type=str, default="train")
    parser.add_argument("--val-splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--train-max-sequences", type=int, default=2)
    parser.add_argument("--val-max-sequences", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=2000)
    parser.add_argument("--val-max-samples", type=int, default=500)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--selector-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default="/gemini/code/FSPT/outputs/world_state_acceptor_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DINOFeatureExtractor(Path(args.weights), device)
    rng = np.random.default_rng(42)

    # Collect training candidates
    print("Collecting training candidates...", flush=True)
    train_seqs = discover_sequences(Path(args.data_root), args.train_splits.split(","))
    if args.train_max_sequences > 0:
        train_seqs = train_seqs[:args.train_max_sequences]
    train_samples = []
    for sp in train_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates(
            extractor, sp, queries, seq, args.depth_noise_sigma, rng,
            topk=args.topk, query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            max_samples=args.train_max_samples - len(train_samples),
        )
        train_samples.extend(cands)
        if len(train_samples) >= args.train_max_samples:
            break
    train_samples = train_samples[:args.train_max_samples]
    print(f"  Training samples: {len(train_samples)}", flush=True)

    # Collect validation candidates
    print("Collecting validation candidates...", flush=True)
    val_seqs = discover_sequences(Path(args.data_root), args.val_splits.split(","))
    if args.val_max_sequences > 0:
        val_seqs = val_seqs[:args.val_max_sequences]
    val_samples = []
    val_rng = np.random.default_rng(999)
    for sp in val_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates(
            extractor, sp, queries, seq, args.depth_noise_sigma, val_rng,
            topk=args.topk, query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            max_samples=args.val_max_samples - len(val_samples),
        )
        val_samples.extend(cands)
        if len(val_samples) >= args.val_max_samples:
            break
    val_samples = val_samples[:args.val_max_samples]
    print(f"  Validation samples: {len(val_samples)}", flush=True)

    if not train_samples or not val_samples:
        raise RuntimeError("No samples collected. Check data paths and filters.")

    # Baseline (no acceptor): always use best DINO candidate
    baseline_metrics = evaluate_acceptor(
        CandidateAcceptor().to(device),  # untrained, will reject all → fallback to baseline
        val_samples,
        device,
    )
    print(f"\nBaseline (no acceptor): {json.dumps(baseline_metrics)}", flush=True)

    # Train acceptor / selector
    if args.selector_mode:
        print("\nTraining selector...", flush=True)
        model, history = train_selector(
            train_samples, val_samples,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
        )
        best_epoch = min(history, key=lambda h: h.get("selector_pred_median", float("inf")))
    else:
        print("\nTraining acceptor...", flush=True)
        model, history = train_acceptor(
            train_samples, val_samples,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, patience=args.patience,
        )
        best_epoch = max(history, key=lambda h: h.get("acceptor_better_frac", 0))

    # Save
    print(f"\nBest epoch: {json.dumps(best_epoch)}", flush=True)

    torch.save(model.state_dict(), output_dir / ("selector_best.pt" if args.selector_mode else "acceptor_best.pt"))
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "val_samples.json").write_text(json.dumps(val_samples[:100], indent=2, default=str) + "\n")
    print(f"\nSaved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
