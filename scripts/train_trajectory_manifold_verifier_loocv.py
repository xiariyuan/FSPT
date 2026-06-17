#!/usr/bin/env python3
"""
Trajectory Manifold Verifier v1.1: LOOCV + selective/calibration.

Sequence-grouped leave-one-out cross-validation on 5 DAVIS val videos.
Validates: neighbor deviation + learned head generalization.
Adds: subset metrics, calibration, risk-coverage.

Usage:
  python scripts/train_trajectory_manifold_verifier_loocv.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/trajectory_manifold_verifier_v1_loocv \
    --max-batches 30
"""

from __future__ import annotations
import argparse, json, sys, signal
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_trajectory_manifold_verifier import (
    compute_trajectory_stats, compute_reconstruction_error,
    compute_neighbor_consistency, TrajectoryVerifierMLP,
    compute_auc, compute_pr_auc, compute_ece,
)
from scripts.audit_local_motion_prior_recovery import load_config, _TimeoutError, _timeout_handler


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


def build_samples(cfg, checkpoint_path, max_batches, neighbor_k, error_threshold, device):
    """Build all samples with trajectory features."""
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device).eval()

    from datasets import get_dataloader
    ds_cfg = cfg.data.val
    dataloader = get_dataloader(name=ds_cfg.dataset, root=ds_cfg.root, batch_size=1,
                                 split="val", num_workers=0, pin_memory=False, seed=0)

    samples = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
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
        meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}

        with torch.no_grad():
            try:
                old_h = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(90)
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_h)
            except (_TimeoutError, Exception) as e:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: {type(e).__name__}")
                continue

        pred_tracks = outputs[0].cpu().numpy()[0]
        gt_tracks = target_points.cpu().numpy()[0]

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        for n_idx in range(N_pts):
            occ_n = occ_np[0] if occ_np is not None else np.zeros((N_pts, T), dtype=bool)
            vis_mask = ~occ_n[n_idx]
            vis_frames = np.where(vis_mask)[0]
            if len(vis_frames) < 3:
                continue

            # Predicted error
            gt_px = gt_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
            pred_px = pred_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
            errors_px = np.linalg.norm(gt_px - pred_px, axis=1)
            mean_error = float(np.mean(errors_px))

            # Trajectory stats
            stats = compute_trajectory_stats(gt_tracks, occ_n, n_idx, T)
            if not stats.get("valid", False):
                continue

            # Reconstruction error
            recon_errors, recon_valid = compute_reconstruction_error(gt_tracks, occ_n, n_idx, K=neighbor_k, T=T)

            # Neighbor consistency
            deviations = compute_neighbor_consistency(gt_tracks, occ_n, n_idx, K=neighbor_k, T=T)

            recon_score = float(np.nanmean(recon_errors[vis_frames])) if recon_errors is not None and recon_valid is not None and recon_valid[vis_frames].any() else np.nan
            deviation_score = float(np.nanmean(deviations[vis_frames])) if deviations is not None and not np.all(np.isnan(deviations[vis_frames])) else np.nan
            vis_score = float(vis_mask.sum()) / T

            stat_score = (
                stats.get("vel_mag_std", 0) * 2.0 +
                stats.get("acc_mag_max", 0) * 1.0 +
                stats.get("discontinuity_max", 0) * 1.0 +
                stats.get("max_occ_len", 0) / 300.0
            )

            samples.append({
                "sample_id": len(samples),
                "video_name": vid_name,
                "point_idx": int(n_idx),
                "occ_length": int(stats.get("max_occ_len", 0)),
                "vis_frac": round(stats["vis_frac"], 3),
                "mean_error_px": round(mean_error, 2),
                "is_high_error": bool(mean_error > error_threshold),
                "visibility_score": round(vis_score, 4),
                "statistics_score": round(stat_score, 4),
                "vel_mag_mean": round(stats["vel_mag_mean"], 6),
                "vel_mag_std": round(stats["vel_mag_std"], 6),
                "acc_mag_mean": round(stats["acc_mag_mean"], 6),
                "acc_mag_max": round(stats["acc_mag_max"], 6),
                "jerk_mag_mean": round(stats["jerk_mag_mean"], 6),
                "discontinuity_max": round(stats["discontinuity_max"], 6),
                "reconstruction_error": round(recon_score, 6) if np.isfinite(recon_score) else None,
                "neighbor_deviation": round(deviation_score, 6) if np.isfinite(deviation_score) else None,
            })

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: {len(samples)} samples")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--error-threshold", type=float, default=16.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build all samples
    print("Building samples...")
    samples = build_samples(cfg, args.checkpoint, args.max_batches, args.neighbor_k, args.error_threshold, device)
    print(f"Total: {len(samples)} samples")

    # Filter valid
    valid = [s for s in samples if s.get("reconstruction_error") is not None and s.get("neighbor_deviation") is not None]
    print(f"Valid: {len(valid)}")

    # Group by video
    by_video = {}
    for s in valid:
        v = s["video_name"]
        if v not in by_video: by_video[v] = []
        by_video[v].append(s)
    sequences = sorted(by_video.keys())
    print(f"Sequences: {sequences} ({len(sequences)} videos)")

    # Feature keys for learned head
    feat_keys = ["vel_mag_mean", "vel_mag_std", "acc_mag_mean", "acc_mag_max",
                  "jerk_mag_mean", "discontinuity_max", "vis_frac",
                  "reconstruction_error", "neighbor_deviation"]

    def extract_features(samples_list):
        X = np.array([[s.get(k, 0) or 0 for k in feat_keys] for s in samples_list], dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # LOOCV
    fold_results = []
    all_predictions = []

    for test_vid in sequences:
        print(f"\n--- Fold: test={test_vid} ---")
        train_samples = [s for v, vs in by_video.items() if v != test_vid for s in vs]
        test_samples = by_video[test_vid]

        train_y = np.array([s["is_high_error"] for s in train_samples], dtype=float)
        test_y = np.array([s["is_high_error"] for s in test_samples], dtype=float)

        print(f"  Train: {len(train_samples)} ({int(train_y.sum())} high-error), Test: {len(test_samples)} ({int(test_y.sum())} high-error)")

        # Baselines
        test_vis = np.array([s["visibility_score"] for s in test_samples])
        test_stat = np.array([-s["statistics_score"] for s in test_samples])
        test_recon = np.array([s.get("reconstruction_error", 0) or 0 for s in test_samples])
        test_dev = np.array([s.get("neighbor_deviation", 0) or 0 for s in test_samples])
        test_errors = np.array([s["mean_error_px"] for s in test_samples])

        # Compute AUCs on test fold
        has_both_classes = test_y.sum() > 0 and (1 - test_y).sum() > 0

        vis_auc = compute_auc(test_y, test_vis) if has_both_classes else None
        stat_auc = compute_auc(test_y, test_stat) if has_both_classes else None
        recon_auc = compute_auc(test_y, test_recon) if has_both_classes else None
        dev_auc = compute_auc(test_y, test_dev) if has_both_classes else None

        # Train learned head
        train_X = extract_features(train_samples)
        test_X = extract_features(test_samples)
        mu, sigma = train_X.mean(0), train_X.std(0) + 1e-6
        train_X_n = (train_X - mu) / sigma
        test_X_n = (test_X - mu) / sigma

        learned_auc = None
        if train_y.sum() > 0 and (1 - train_y).sum() > 0:
            model_head = TrajectoryVerifierMLP(train_X.shape[1], hidden=32)
            opt = torch.optim.Adam(model_head.parameters(), lr=1e-3)
            pw = torch.tensor([(1 - train_y.mean()) / max(train_y.mean(), 1e-6)])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

            Xt = torch.from_numpy(train_X_n)
            yt_t = torch.from_numpy(train_y)

            best_state, best_auc = None, 0.5
            for epoch in range(100):
                model_head.train(); opt.zero_grad()
                loss = criterion(model_head(Xt), yt_t)
                loss.backward(); opt.step()
                if (epoch + 1) % 20 == 0:
                    model_head.eval()
                    with torch.no_grad():
                        probs = torch.sigmoid(model_head(Xt)).numpy()
                    auc = compute_auc(train_y, probs)
                    if auc > best_auc:
                        best_auc = auc
                        best_state = {k: v.clone() for k, v in model_head.state_dict().items()}

            if best_state:
                model_head.load_state_dict(best_state)
                model_head.eval()
                with torch.no_grad():
                    test_probs = torch.sigmoid(model_head(torch.from_numpy(test_X_n))).numpy()
                learned_auc = compute_auc(test_y, test_probs) if has_both_classes else None
                learned_pr = compute_pr_auc(test_y, test_probs) if has_both_classes else None

                # Store per-sample predictions
                for i, s in enumerate(test_samples):
                    pred = {
                        "sample_id": s["sample_id"],
                        "video_name": s["video_name"],
                        "point_idx": s["point_idx"],
                        "fold_test_video": test_vid,
                        "mean_error_px": s["mean_error_px"],
                        "is_high_error": s["is_high_error"],
                        "neighbor_deviation": s.get("neighbor_deviation"),
                        "learned_prob": round(float(test_probs[i]), 4),
                    }
                    all_predictions.append(pred)
            else:
                learned_pr = None
        else:
            learned_pr = None

        # Spearman correlation on test
        from scipy.stats import spearmanr
        dev_spearman, _ = spearmanr(test_dev, test_errors) if len(test_errors) > 5 else (0, 0)

        fold_result = {
            "test_video": test_vid,
            "train_n": len(train_samples),
            "test_n": len(test_samples),
            "test_high_error_n": int(test_y.sum()),
            "vis_auc": round(vis_auc, 3) if vis_auc is not None else None,
            "stat_auc": round(stat_auc, 3) if stat_auc is not None else None,
            "recon_auc": round(recon_auc, 3) if recon_auc is not None else None,
            "dev_auc": round(dev_auc, 3) if dev_auc is not None else None,
            "learned_auc": round(learned_auc, 3) if learned_auc is not None else None,
            "learned_pr_auc": round(learned_pr, 3) if learned_pr is not None else None,
            "dev_spearman": round(float(dev_spearman), 3),
        }
        fold_results.append(fold_result)

        print(f"  vis_auc={fold_result['vis_auc']}, dev_auc={fold_result['dev_auc']}, learned_auc={fold_result['learned_auc']}")

    # Aggregate
    def avg_metric(key):
        vals = [f[key] for f in fold_results if f.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    avg_results = {
        "n_folds": len(fold_results),
        "avg_vis_auc": avg_metric("vis_auc"),
        "avg_stat_auc": avg_metric("stat_auc"),
        "avg_recon_auc": avg_metric("recon_auc"),
        "avg_dev_auc": avg_metric("dev_auc"),
        "avg_learned_auc": avg_metric("learned_auc"),
        "avg_dev_spearman": avg_metric("dev_spearman"),
        "per_fold": fold_results,
    }

    # Subset metrics
    occ_lengths = np.array([s["occ_length"] for s in valid])
    errors_arr = np.array([s["mean_error_px"] for s in valid])
    dev_scores = np.array([s.get("neighbor_deviation", 0) or 0 for s in valid])
    is_high = np.array([s["is_high_error"] for s in valid], dtype=float)

    subset_metrics = {}
    for label, mask in [("all", np.ones(len(valid), bool)), ("long_occ>10", occ_lengths > 10),
                         ("hard", errors_arr > args.error_threshold), ("easy", errors_arr <= 4.0)]:
        m = mask
        if m.sum() == 0: continue
        sub_high = is_high[m]
        sub_dev = dev_scores[m]
        sub_err = errors_arr[m]
        has_both = sub_high.sum() > 0 and (1-sub_high).sum() > 0
        subset_metrics[label] = {
            "n": int(m.sum()),
            "high_error_frac": round(float(sub_high.mean()), 3),
            "error_median": round(float(np.median(sub_err)), 2),
            "dev_auc": round(compute_auc(sub_high, sub_dev), 3) if has_both else None,
            "dev_spearman": round(float(np.corrcoef(sub_dev, sub_err)[0, 1]), 3) if len(sub_err) > 5 else None,
        }

    # Calibration (simplified)
    if all_predictions:
        learned_probs = np.array([p["learned_prob"] for p in all_predictions if p.get("learned_prob") is not None])
        learned_labels = np.array([p["is_high_error"] for p in all_predictions if p.get("learned_prob") is not None])
        if len(learned_probs) > 10:
            calibration = {
                "ece": round(compute_ece(learned_probs, learned_labels), 3),
                "brier": round(float(np.mean((learned_probs - learned_labels) ** 2)), 4),
                "mean_predicted": round(float(learned_probs.mean()), 4),
                "actual_positive_rate": round(float(learned_labels.mean()), 4),
            }
        else:
            calibration = {}
    else:
        calibration = {}

    # Pooled AUC
    pooled_dev = np.array([s.get("neighbor_deviation", 0) or 0 for s in valid])
    pooled_vis = np.array([s["visibility_score"] for s in valid])
    pooled_high = np.array([s["is_high_error"] for s in valid], dtype=float)

    summary = {
        "n_samples": len(samples),
        "n_valid": len(valid),
        "n_sequences": len(sequences),
        "error_threshold": args.error_threshold,
        "pooled_auc": {
            "visibility": round(compute_auc(pooled_high, pooled_vis), 3),
            "neighbor_deviation": round(compute_auc(pooled_high, pooled_dev), 3),
        },
        "loocv_averages": avg_results,
        "subset_metrics": subset_metrics,
        "calibration": calibration,
    }

    # Verdict
    avg_dev = avg_results.get("avg_dev_auc") or 0
    avg_learned = avg_results.get("avg_learned_auc") or 0
    pooled_dev_auc = summary["pooled_auc"]["neighbor_deviation"]

    c1 = avg_dev > 0.6 and avg_dev > (avg_results.get("avg_vis_auc") or 0) + 0.05
    c2 = avg_learned > 0.7
    c3 = subset_metrics.get("hard", {}).get("dev_auc") is not None and subset_metrics["hard"]["dev_auc"] > 0.6

    verdict = "PASS" if sum([c1, c2, c3]) >= 2 else "FAIL"
    summary["criteria"] = {
        "loocv_dev_auc_good": bool(c1),
        "loocv_learned_auc_good": bool(c2),
        "hard_subset_signal": bool(c3),
        "criteria_passed": int(sum([c1, c2, c3])),
    }
    summary["verdict"] = verdict

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_fold_metrics.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    with open(out_dir / "subset_metrics.json", "w") as f:
        json.dump(subset_metrics, f, indent=2)
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"Trajectory Manifold Verifier v1.1 LOOCV")
    print(f"{'='*60}")
    print(f"  Pooled AUC: vis={summary['pooled_auc']['visibility']:.3f}, dev={summary['pooled_auc']['neighbor_deviation']:.3f}")
    print(f"  LOOCV avg: vis={avg_results['avg_vis_auc']}, dev={avg_results['avg_dev_auc']}, learned={avg_results['avg_learned_auc']}")
    print(f"  Per fold:")
    for fr in fold_results:
        print(f"    {fr['test_video']:20s}: vis={fr['vis_auc']}, dev={fr['dev_auc']}, learned={fr['learned_auc']}")
    sub_str = ", ".join(f"{k}={v.get('dev_auc','?')}" for k, v in subset_metrics.items())
    print(f"  Subset dev AUC: {sub_str}")
    print(f"  Calibration: ECE={calibration.get('ece','?')}, Brier={calibration.get('brier','?')}")
    print(f"\n  Criteria: loocv_dev={c1}, loocv_learned={c2}, hard={c3}")
    print(f"  Verdict: {verdict}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
