#!/usr/bin/env python3
"""
Trajectory Manifold Verifier v1: verify if trajectory local motion consistency
can serve as a reliability signal for point tracking.

Three experiments:
  1. Statistics baseline: hand-crafted trajectory statistics as reliability score
  2. Reconstruction proxy: neighbor-based trajectory reconstruction error
  3. Lightweight learned head: small MLP on trajectory features + reconstruction

Usage:
  python scripts/train_trajectory_manifold_verifier.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/trajectory_manifold_verifier_v1 \
    --max-batches 30
"""

from __future__ import annotations
import argparse, json, sys, signal
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn
import torch.nn.functional as F

from scripts.audit_local_motion_prior_recovery import (
    load_config, _TimeoutError, _timeout_handler,
    find_t_last_visible, select_escort_points,
)


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


# ---------------------------------------------------------------------------
# Trajectory feature extraction
# ---------------------------------------------------------------------------
def compute_trajectory_stats(tracks_n, occ_n, n_idx, T, window=5):
    """Compute hand-crafted trajectory statistics for point n_idx.

    Returns dict of features.
    """
    # Track positions (normalized)
    traj = tracks_n[n_idx]  # (T, 2)
    occ = occ_n[n_idx]  # (T,)

    # Visible frames
    vis_mask = ~occ
    vis_frames = np.where(vis_mask)[0]

    if len(vis_frames) < 3:
        return {"valid": False}

    # Velocity (frame-to-frame displacement)
    velocities = []
    for i in range(1, len(vis_frames)):
        t_prev, t_curr = vis_frames[i-1], vis_frames[i]
        if t_curr == t_prev + 1:  # consecutive
            v = traj[t_curr] - traj[t_prev]
            velocities.append(v)

    if len(velocities) < 2:
        return {"valid": False}

    velocities = np.array(velocities)  # (K-1, 2)

    # Acceleration
    accelerations = np.diff(velocities, axis=0)  # (K-2, 2) if K >= 3

    # Stats
    vel_mean = np.mean(velocities, axis=0)
    vel_std = np.std(velocities, axis=0)
    vel_mag = np.linalg.norm(velocities, axis=1)

    acc_mag = np.linalg.norm(accelerations, axis=1) if len(accelerations) > 0 else np.array([0.0])

    # Discontinuity: max velocity change
    vel_changes = np.linalg.norm(np.diff(velocities, axis=0), axis=1) if len(velocities) > 2 else np.array([0.0])

    # Max occlusion length
    max_occ = 0
    curr_occ = 0
    for t in range(T):
        if occ[t]:
            curr_occ += 1
            max_occ = max(max_occ, curr_occ)
        else:
            curr_occ = 0

    # Track length (visible fraction)
    vis_frac = float(vis_mask.sum()) / T

    return {
        "valid": True,
        "vel_mean_x": float(vel_mean[0]),
        "vel_mean_y": float(vel_mean[1]),
        "vel_std_x": float(vel_std[0]),
        "vel_std_y": float(vel_std[1]),
        "vel_mag_mean": float(np.mean(vel_mag)),
        "vel_mag_std": float(np.std(vel_mag)),
        "acc_mag_mean": float(np.mean(acc_mag)),
        "acc_mag_std": float(np.std(acc_mag)),
        "acc_mag_max": float(np.max(acc_mag)),
        "jerk_mag_mean": float(np.mean(vel_changes)),
        "discontinuity_max": float(np.max(vel_changes)),
        "vis_frac": vis_frac,
        "max_occ_len": int(max_occ),
        "track_len": int(len(vis_frames)),
    }


def compute_reconstruction_error(tracks_n, occ_n, query_idx, K=16, T=None):
    """Reconstruct query trajectory from K nearest neighbors.

    Uses weighted average of neighbor displacements.
    Returns per-frame reconstruction error.
    """
    if T is None:
        T = tracks_n.shape[1]

    query_traj = tracks_n[query_idx]  # (T, 2)
    query_occ = occ_n[query_idx]  # (T,)

    # Find neighbors visible when query is visible
    vis_query = ~query_occ  # (T,)
    vis_frames = np.where(vis_query)[0]

    if len(vis_query) < 2:
        return None, None

    # Find all other points visible at the same frames
    neighbor_candidates = []
    for i in range(tracks_n.shape[0]):
        if i == query_idx:
            continue
        n_vis = ~occ_n[i]
        # Count co-visible frames
        covis = (vis_query & n_vis).sum()
        if covis >= 3:
            neighbor_candidates.append(i)

    if len(neighbor_candidates) < 4:
        return None, None

    # Select K nearest by average position distance
    query_positions = query_traj[vis_frames]  # (V, 2)
    dists = []
    for ni in neighbor_candidates:
        n_positions = tracks_n[ni][vis_frames]
        d = np.mean(np.linalg.norm(query_positions - n_positions, axis=1))
        dists.append((d, ni))
    dists.sort()
    neighbors = [ni for _, ni in dists[:K]]

    # Reconstruct: for each frame, use weighted average of neighbor positions
    # Weights: inverse distance at the frame
    recon = np.zeros((T, 2), dtype=np.float32)
    recon_valid = np.zeros(T, dtype=bool)

    for t in vis_frames:
        q_pos = query_traj[t]
        n_positions = np.array([tracks_n[ni][t] for ni in neighbors])  # (K, 2)
        n_vis = np.array([~occ_n[ni][t] for ni in neighbors])  # (K,)

        if not n_vis.any():
            continue

        # Visible neighbors only
        n_pos_vis = n_positions[n_vis]
        # Weights: inverse distance
        dists_to_q = np.linalg.norm(n_pos_vis - q_pos, axis=1) + 1e-6
        weights = 1.0 / dists_to_q
        weights = weights / weights.sum()

        recon[t] = np.average(n_pos_vis, weights=weights, axis=0)
        recon_valid[t] = True

    # Per-frame error (in normalized coords)
    errors = np.full(T, np.nan, dtype=np.float32)
    for t in vis_frames:
        if recon_valid[t]:
            errors[t] = np.linalg.norm(recon[t] - query_traj[t])

    return errors, recon_valid


def compute_neighbor_consistency(tracks_n, occ_n, query_idx, K=16, T=None):
    """Compute how consistent the query trajectory is with its local neighborhood.

    For each visible frame, compute the deviation of query from its K nearest
    neighbor centroid.
    """
    if T is None:
        T = tracks_n.shape[1]

    query_traj = tracks_n[query_idx]  # (T, 2)
    query_occ = occ_n[query_idx]

    vis_frames = np.where(~query_occ)[0]
    if len(vis_frames) < 3:
        return None

    # Find co-visible neighbors
    neighbor_candidates = []
    for i in range(tracks_n.shape[0]):
        if i == query_idx:
            continue
        covis = ((~query_occ) & (~occ_n[i])).sum()
        if covis >= 3:
            neighbor_candidates.append(i)

    if len(neighbor_candidates) < 4:
        return None

    # Select K nearest
    query_positions = query_traj[vis_frames]
    dists = []
    for ni in neighbor_candidates:
        n_positions = tracks_n[ni][vis_frames]
        d = np.mean(np.linalg.norm(query_positions - n_positions, axis=1))
        dists.append((d, ni))
    dists.sort()
    neighbors = [ni for _, ni in dists[:K]]

    # Per-frame deviation from neighbor centroid
    deviations = np.full(T, np.nan, dtype=np.float32)
    for t in vis_frames:
        q_pos = query_traj[t]
        n_positions = np.array([tracks_n[ni][t] for ni in neighbors])
        n_vis = np.array([~occ_n[ni][t] for ni in neighbors])
        if not n_vis.any():
            continue
        centroid = np.mean(n_positions[n_vis], axis=0)
        deviations[t] = np.linalg.norm(q_pos - centroid)

    return deviations


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TrajectoryVerifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_auc(labels, scores):
    """Simple ROC-AUC computation."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except:
        return 0.5


def compute_pr_auc(labels, scores):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(labels, scores))
    except:
        return 0.0


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / max(1, len(labels)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--error-threshold", type=float, default=16.0,
                        help="Threshold for 'high error' binary label (px)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model for predicted tracks (if checkpoint provided)
    model = None
    if args.checkpoint:
        from models.cotracker_refiner import CoTrackerFSPTRefiner
        model = CoTrackerFSPTRefiner(cfg.get("model", {}))
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        model = model.to(device).eval()

    # Val loader (num_workers=0 to avoid OOM kills)
    from datasets import get_dataloader
    ds_cfg = cfg.data.val
    dataloader = get_dataloader(
        name=ds_cfg.dataset, root=ds_cfg.root, batch_size=1,
        split="val", num_workers=0, pin_memory=False,
        seed=0,
    )

    samples = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points", batch.get("tracks"))
        occluded = batch.get("occluded")
        video_name = batch.get("video_name", ["unknown"])
        if video is None or query_points is None or target_points is None:
            continue

        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)
        if target_points.dim() == 3: target_points = target_points.unsqueeze(0)
        occ_np = None
        if occluded is not None:
            if occluded.dim() == 2: occluded = occluded.unsqueeze(0)
            occ_np = occluded.cpu().numpy()

        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]

        # Get predicted tracks
        if model is not None:
            meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}
            with torch.no_grad():
                try:
                    old_h = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(90)
                    outputs = model(video_dev, query_dev, meta=meta, return_info=True)
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_h)
                except _TimeoutError:
                    signal.alarm(0)
                    print(f"  Batch {batch_idx}: TIMEOUT")
                    continue
                except Exception as e:
                    signal.alarm(0)
                    print(f"  Batch {batch_idx}: ERROR {e}")
                    continue
            pred_tracks = outputs[0].cpu().numpy()[0]  # (N, T, 2)
        else:
            pred_tracks = None

        gt_tracks = target_points.cpu().numpy()[0]  # (N, T, 2)

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Use GT tracks for trajectory manifold analysis
        tracks_for_analysis = gt_tracks  # Use GT as "the tracks to verify"

        for n_idx in range(N_pts):
            occ_n = occ_np[0] if occ_np is not None else np.zeros((N_pts, T), dtype=bool)

            # Only evaluate at frames where point is visible
            vis_mask = ~occ_n[n_idx]
            vis_frames = np.where(vis_mask)[0]
            if len(vis_frames) < 3:
                continue

            # Compute GT error: if we have pred tracks, use them; otherwise use distance from neighbor centroid
            if pred_tracks is not None:
                gt_px = gt_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
                pred_px = pred_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
                errors_px = np.linalg.norm(gt_px - pred_px, axis=1)
                mean_error = float(np.mean(errors_px))
            else:
                mean_error = 0.0  # No pred tracks available

            # Compute trajectory stats
            stats = compute_trajectory_stats(tracks_for_analysis, occ_n, n_idx, T)
            if not stats.get("valid", False):
                continue

            # Reconstruction error
            recon_errors, recon_valid = compute_reconstruction_error(
                tracks_for_analysis, occ_n, n_idx, K=args.neighbor_k, T=T
            )

            # Neighbor consistency
            deviations = compute_neighbor_consistency(
                tracks_for_analysis, occ_n, n_idx, K=args.neighbor_k, T=T
            )

            # Compute aggregate reliability scores
            recon_score = float(np.nanmean(recon_errors[vis_frames])) if recon_errors is not None and recon_valid is not None and recon_valid[vis_frames].any() else np.nan
            deviation_score = float(np.nanmean(deviations[vis_frames])) if deviations is not None and not np.all(np.isnan(deviations[vis_frames])) else np.nan

            # Visibility-based score (how many frames visible)
            vis_score = float(vis_mask.sum()) / T

            # Combined statistics score
            stat_score = (
                stats.get("vel_mag_std", 0) * 2.0 +
                stats.get("acc_mag_max", 0) * 1.0 +
                stats.get("discontinuity_max", 0) * 1.0 +
                stats.get("max_occ_len", 0) / 300.0
            )

            sample = {
                "sample_id": len(samples),
                "video_name": vid_name,
                "point_idx": int(n_idx),
                "occ_length": int(stats.get("max_occ_len", 0)),
                "vis_frac": round(stats["vis_frac"], 3),
                "mean_error_px": round(mean_error, 2),
                "is_high_error": bool(mean_error > args.error_threshold),
                # Baselines
                "visibility_score": round(vis_score, 4),
                "statistics_score": round(stat_score, 4),
                # Trajectory features
                "vel_mag_mean": round(stats["vel_mag_mean"], 6),
                "vel_mag_std": round(stats["vel_mag_std"], 6),
                "acc_mag_mean": round(stats["acc_mag_mean"], 6),
                "acc_mag_max": round(stats["acc_mag_max"], 6),
                "jerk_mag_mean": round(stats["jerk_mag_mean"], 6),
                "discontinuity_max": round(stats["discontinuity_max"], 6),
                # Reconstruction proxy
                "reconstruction_error": round(recon_score, 6) if np.isfinite(recon_score) else None,
                "neighbor_deviation": round(deviation_score, 6) if np.isfinite(deviation_score) else None,
            }

            samples.append(sample)

        n_so_far = len(samples)
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: {n_so_far} samples")
            sys.stdout.flush()

    print(f"\nTotal: {len(samples)} samples")

    if len(samples) < 10:
        print("Too few samples!")
        return

    # Filter valid samples
    valid_samples = [s for s in samples if s.get("reconstruction_error") is not None and s.get("neighbor_deviation") is not None]
    print(f"Valid for analysis: {len(valid_samples)}")

    if len(valid_samples) < 10:
        print("Too few valid samples for analysis!")
        return

    # ====================================================================
    # Experiment 1: Statistics baseline
    # ====================================================================
    print("\n--- Experiment 1: Statistics baseline ---")
    errors = np.array([s["mean_error_px"] for s in valid_samples])
    is_high = np.array([s["is_high_error"] for s in valid_samples]).astype(float)

    stat_scores = np.array([-s["statistics_score"] for s in valid_samples])  # negative: lower = more reliable
    vis_scores = np.array([s["visibility_score"] for s in valid_samples])

    # ROC-AUC for high error detection
    stat_auc = compute_auc(is_high, stat_scores)
    vis_auc = compute_auc(is_high, vis_scores)
    stat_pr = compute_pr_auc(is_high, stat_scores)
    vis_pr = compute_pr_auc(is_high, vis_scores)

    print(f"  Statistics score AUC: {stat_auc:.3f}, PR-AUC: {stat_pr:.3f}")
    print(f"  Visibility score AUC: {vis_auc:.3f}, PR-AUC: {vis_pr:.3f}")

    # ====================================================================
    # Experiment 2: Reconstruction proxy
    # ====================================================================
    print("\n--- Experiment 2: Reconstruction proxy ---")
    recon_scores = np.array([s["reconstruction_error"] for s in valid_samples])
    dev_scores = np.array([s["neighbor_deviation"] for s in valid_samples])

    recon_auc = compute_auc(is_high, recon_scores)
    recon_pr = compute_pr_auc(is_high, recon_scores)
    dev_auc = compute_auc(is_high, dev_scores)
    dev_pr = compute_pr_auc(is_high, dev_scores)

    # Correlation
    from scipy.stats import pearsonr, spearmanr
    pearson_recon, _ = pearsonr(recon_scores, errors)
    spearman_recon, _ = spearmanr(recon_scores, errors)
    pearson_dev, _ = pearsonr(dev_scores, errors)
    spearman_dev, _ = spearmanr(dev_scores, errors)

    print(f"  Reconstruction error AUC: {recon_auc:.3f}, PR-AUC: {recon_pr:.3f}")
    print(f"  Neighbor deviation AUC: {dev_auc:.3f}, PR-AUC: {dev_pr:.3f}")
    print(f"  Reconstruction-error correlation: Pearson={pearson_recon:.3f}, Spearman={spearman_recon:.3f}")
    print(f"  Neighbor-deviation correlation: Pearson={pearson_dev:.3f}, Spearman={spearman_dev:.3f}")

    # ====================================================================
    # Experiment 3: Learned head (if signal exists)
    # ====================================================================
    learned_auc = 0.5
    learned_pr = 0.0

    if recon_auc > 0.55 or dev_auc > 0.55:
        print("\n--- Experiment 3: Learned reliability head ---")
        # Build feature matrix
        feat_keys = ["vel_mag_mean", "vel_mag_std", "acc_mag_mean", "acc_mag_max",
                      "jerk_mag_mean", "discontinuity_max", "vis_frac",
                      "reconstruction_error", "neighbor_deviation"]
        X = np.array([[s.get(k, 0) or 0 for k in feat_keys] for s in valid_samples], dtype=np.float32)
        y = is_high.copy()

        # Replace NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize
        mu, sigma = X.mean(0), X.std(0) + 1e-6
        X_n = (X - mu) / sigma

        # Simple train/eval (no sequence split — small data, just signal check)
        model_head = TrajectoryVerifierMLP(X.shape[1], hidden=32)
        optimizer = torch.optim.Adam(model_head.parameters(), lr=1e-3)
        Xt = torch.from_numpy(X_n)
        yt = torch.from_numpy(y)

        # Binary cross-entropy
        pos_frac = float(y.mean())
        pw = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        best_state = None
        best_auc = 0.5

        for epoch in range(100):
            model_head.train()
            optimizer.zero_grad()
            logits = model_head(Xt)
            loss = criterion(logits, yt)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                model_head.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(model_head(Xt)).numpy()
                auc = compute_auc(y, probs)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.clone() for k, v in model_head.state_dict().items()}

        if best_state:
            model_head.load_state_dict(best_state)
            model_head.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model_head(Xt)).numpy()
            learned_auc = compute_auc(y, probs)
            learned_pr = compute_pr_auc(y, probs)
            learned_ece = compute_ece(probs, y)
            print(f"  Learned head AUC: {learned_auc:.3f}, PR-AUC: {learned_pr:.3f}, ECE: {learned_ece:.3f}")
        else:
            print(f"  Learned head: no improvement over random")
    else:
        print("\n--- Experiment 3: Skipped (no signal in reconstruction proxy) ---")

    # ====================================================================
    # Subset analysis
    # ====================================================================
    def subset_metrics(mask, label):
        if mask.sum() == 0:
            return {"label": label, "n": 0}
        sub = [valid_samples[i] for i in range(len(valid_samples)) if mask[i]]
        sub_errors = np.array([s["mean_error_px"] for s in sub])
        sub_recon = np.array([s["reconstruction_error"] for s in sub])
        sub_high = np.array([s["is_high_error"] for s in sub]).astype(float)
        return {
            "label": label, "n": len(sub),
            "error_median": round(float(np.median(sub_errors)), 2),
            "high_error_frac": round(float(sub_high.mean()), 3),
            "recon_error_median": round(float(np.median(sub_recon)), 4),
            "recon_auc": round(compute_auc(sub_high, sub_recon), 3) if float(sub_high.sum()) > 0 and float((1-sub_high).sum()) > 0 else None,
        }

    occ_lengths = np.array([s["occ_length"] for s in valid_samples])
    errors_arr = np.array([s["mean_error_px"] for s in valid_samples])

    subset_all = subset_metrics(np.ones(len(valid_samples), dtype=bool), "all")
    subset_long_occ = subset_metrics(occ_lengths > 10, "long_occ>10")
    subset_hard = subset_metrics(errors_arr > args.error_threshold, "hard")
    subset_easy = subset_metrics(errors_arr <= 4.0, "easy")

    # ====================================================================
    # Verdict
    # ====================================================================
    c1 = recon_auc > vis_auc + 0.05  # reconstruction better than visibility
    c2 = spearman_recon > 0.3  # meaningful correlation
    c3 = subset_long_occ.get("recon_auc") is not None and subset_long_occ["recon_auc"] > 0.6  # long-occ signal
    c4 = learned_auc > 0.65  # learned head has signal

    criteria_passed = sum([c1, c2, c3, c4])
    if criteria_passed >= 2:
        verdict = "PASS"
    elif recon_auc < 0.55 and dev_auc < 0.55:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    # Save
    results = {
        "n_samples": len(samples),
        "n_valid": len(valid_samples),
        "error_threshold": args.error_threshold,
        "neighbor_k": args.neighbor_k,
        "experiments": {
            "statistics_baseline": {
                "stat_auc": round(stat_auc, 3),
                "stat_pr_auc": round(stat_pr, 3),
                "visibility_auc": round(vis_auc, 3),
                "visibility_pr_auc": round(vis_pr, 3),
            },
            "reconstruction_proxy": {
                "reconstruction_auc": round(recon_auc, 3),
                "reconstruction_pr_auc": round(recon_pr, 3),
                "neighbor_deviation_auc": round(dev_auc, 3),
                "neighbor_deviation_pr_auc": round(dev_pr, 3),
                "reconstruction_pearson": round(pearson_recon, 3),
                "reconstruction_spearman": round(spearman_recon, 3),
                "deviation_pearson": round(pearson_dev, 3),
                "deviation_spearman": round(spearman_dev, 3),
            },
            "learned_head": {
                "auc": round(learned_auc, 3),
                "pr_auc": round(learned_pr, 3),
            },
        },
        "subset_metrics": {
            "all": subset_all,
            "long_occ": subset_long_occ,
            "hard": subset_hard,
            "easy": subset_easy,
        },
        "criteria": {
            "recon_better_than_visibility": bool(c1),
            "meaningful_correlation": bool(c2),
            "long_occ_signal": bool(c3),
            "learned_head_signal": bool(c4),
            "criteria_passed": int(criteria_passed),
        },
        "verdict": verdict,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Risk-coverage (simplified)
    recon_order = np.argsort(recon_scores)[::-1]  # highest recon error first
    risk_coverage = []
    for k_pct in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        k = max(1, int(k_pct * len(valid_samples)))
        selected = recon_order[:k]
        sel_errors = errors[selected]
        risk_coverage.append({
            "coverage": round(k_pct, 2),
            "n_selected": k,
            "mean_error": round(float(np.mean(sel_errors)), 2),
            "median_error": round(float(np.median(sel_errors)), 2),
        })
    with open(out_dir / "risk_coverage.json", "w") as f:
        json.dump(risk_coverage, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"Trajectory Manifold Verifier v1")
    print(f"{'='*60}")
    print(f"  Samples: {len(samples)} total, {len(valid_samples)} valid")
    print(f"  High-error rate: {is_high.mean():.3f}")
    print(f"\n  AUC (high error detection):")
    print(f"    Visibility baseline: {vis_auc:.3f}")
    print(f"    Statistics score:    {stat_auc:.3f}")
    print(f"    Reconstruction:      {recon_auc:.3f}")
    print(f"    Neighbor deviation:  {dev_auc:.3f}")
    print(f"    Learned head:        {learned_auc:.3f}")
    print(f"\n  Correlation (reconstruction vs error):")
    print(f"    Pearson:  {pearson_recon:.3f}")
    print(f"    Spearman: {spearman_recon:.3f}")
    print(f"\n  Subsets:")
    for s in [subset_all, subset_long_occ, subset_hard, subset_easy]:
        if s.get("n", 0) > 0:
            print(f"    {s['label']:20s}: n={s['n']:>4d}, error_med={s['error_median']:>7.2f}, "
                  f"recon_med={s['recon_error_median']:>8.4f}, recon_auc={s.get('recon_auc', '?')}")
    print(f"\n  Criteria: recon>vis={c1}, corr={c2}, long_occ={c3}, learned={c4}")
    print(f"  Verdict: {verdict} ({criteria_passed}/4)")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
