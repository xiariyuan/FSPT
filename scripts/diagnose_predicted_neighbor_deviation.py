#!/usr/bin/env python3
"""
Stage 2: Predicted-track neighbor deviation signal quality diagnosis.

Runs teacher model on DAVIS val, computes neighbor deviation from pred_tracks,
and evaluates against GT error. No training.

Usage:
  python scripts/diagnose_predicted_neighbor_deviation.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/predicted_neighbor_deviation_signal_v1 \
    --max-batches 30 --error-threshold 16.0
"""

from __future__ import annotations
import argparse, json, sys, signal
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

from utils.neighbor_deviation import compute_neighbor_deviation_risk, normalize_risk_to_01
from scripts.audit_local_motion_prior_recovery import load_config, _TimeoutError, _timeout_handler


def _auc(labels, scores):
    try: return float(roc_auc_score(labels, scores))
    except: return 0.5


def _pr_auc(labels, scores):
    try: return float(average_precision_score(labels, scores))
    except: return 0.0


def selective_eval_by_coverage(scores, errors, is_high, coverages):
    """For each coverage, accept top-scoring fraction, report error metrics."""
    n = len(scores)
    results = []
    for cov in coverages:
        k = max(1, int(cov * n))
        # Higher score = more confident = accept
        order = np.argsort(scores)[::-1]
        selected = order[:k]
        results.append({
            "coverage": round(cov, 2),
            "n": k,
            "mean_error": round(float(np.mean(errors[selected])), 2),
            "median_error": round(float(np.median(errors[selected])), 2),
            "high_error_frac": round(float(is_high[selected].mean()), 3),
            "lt4px_frac": round(float((errors[selected] < 4).mean()), 3),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--error-threshold", type=float, default=16.0)
    parser.add_argument("--neighbor-k", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher model
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model._export_candidates_debug = True
    model = model.to(device).eval()

    from datasets import get_dataloader
    ds_cfg = cfg.data.val
    dataloader = get_dataloader(name=ds_cfg.dataset, root=ds_cfg.root, batch_size=1, split="val", num_workers=0, pin_memory=False, seed=0)

    all_samples = []

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
        pred_visibility = (outputs[1].cpu().numpy()[0] > 0).astype(float)  # (N, T) binary

        # Get verifier scores from outputs[2] if available
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        verifier_scores = None
        vs = info.get("verifier_scores")
        if vs is not None and isinstance(vs, torch.Tensor):
            verifier_scores = vs.cpu().numpy()[0]  # (N, T)

        gt_tracks = target_points.cpu().numpy()[0]  # (N, T, 2)
        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Compute predicted neighbor deviation risk
        pred_tracks_t = torch.from_numpy(pred_tracks).unsqueeze(0).float().to(device)  # (1, N, T, 2)
        pred_vis_t = torch.from_numpy(pred_visibility).unsqueeze(0).float().to(device)  # (1, N, T)
        risk_raw = compute_neighbor_deviation_risk(pred_tracks_t, pred_vis_t, K=args.neighbor_k)
        risk_norm = normalize_risk_to_01(risk_raw, pred_vis_t)
        confidence = 1.0 - risk_norm  # Higher = more reliable

        risk_np = risk_norm[0].cpu().numpy()  # (N, T)
        conf_np = confidence[0].cpu().numpy()  # (N, T)

        for n_idx in range(N_pts):
            vis = ~occ_np[0, n_idx] if occ_np is not None else np.ones(T, dtype=bool)
            vis_frames = np.where(vis)[0]
            if len(vis_frames) < 3:
                continue

            # Compute GT error
            gt_px = gt_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
            pred_px = pred_tracks[n_idx, vis_frames] * np.array([orig_w, orig_h])
            errors_px = np.linalg.norm(gt_px - pred_px, axis=1)
            mean_err = float(np.mean(errors_px))

            # Compute per-frame metrics
            nd_risk = float(np.mean(risk_np[n_idx, vis_frames]))
            nd_conf = float(np.mean(conf_np[n_idx, vis_frames]))
            pred_vis_frac = float(pred_visibility[n_idx, vis_frames].mean())

            # Verifier score (if available)
            vs_mean = None
            if verifier_scores is not None:
                vs_mean = float(verifier_scores[n_idx, vis_frames].mean())

            occ_len = int(((~vis).sum()))  # consecutive occlusion length

            sample = {
                "video_name": vid_name,
                "point_idx": int(n_idx),
                "mean_error_px": round(mean_err, 2),
                "is_high_error": bool(mean_err > args.error_threshold),
                "occ_length": occ_len,
                "pred_visibility_score": round(pred_vis_frac, 4),
                "neighbor_deviation_risk": round(nd_risk, 6),
                "neighbor_deviation_confidence": round(nd_conf, 6),
                "verifier_score": round(vs_mean, 6) if vs_mean is not None else None,
            }
            all_samples.append(sample)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: {len(all_samples)} samples")
            sys.stdout.flush()

    print(f"\nTotal: {len(all_samples)} samples")

    # ====================================================================
    # Compute metrics
    # ====================================================================
    errors = np.array([s["mean_error_px"] for s in all_samples])
    is_high = np.array([s["is_high_error"] for s in all_samples], dtype=float)
    nd_risk = np.array([s["neighbor_deviation_risk"] for s in all_samples])
    nd_conf = np.array([s["neighbor_deviation_confidence"] for s in all_samples])
    pred_vis = np.array([s["pred_visibility_score"] for s in all_samples])
    occ_len = np.array([s["occ_length"] for s in all_samples])

    vs_arr = np.array([s.get("verifier_score", 0) or 0 for s in all_samples])
    has_verifier = any(s.get("verifier_score") is not None for s in all_samples)

    # ====================================================================
    # Overall signal quality
    # ====================================================================
    print("\n--- Overall Signal Quality ---")
    has_both = is_high.sum() > 0 and (1 - is_high).sum() > 0

    scores_dict = {
        "pred_visibility": {"scores": pred_vis, "direction": "reliability"},
        "neighbor_dev_risk": {"scores": nd_risk, "direction": "risk"},
        "neighbor_dev_confidence": {"scores": nd_conf, "direction": "reliability"},
    }
    if has_verifier:
        scores_dict["verifier_scores"] = {"scores": vs_arr, "direction": "reliability"}

    overall_metrics = {}
    for name, info in scores_dict.items():
        scores = info["scores"]
        direction = info["direction"]
        # For risk: higher = worse, so detection AUC = roc_auc(-scores, high_error)
        # For reliability: higher = better, so detection AUC = roc_auc(scores, high_error)
        if direction == "risk":
            det_scores = -scores  # negated: higher neg risk = lower risk = more reliable
        else:
            det_scores = scores

        auc = _auc(is_high, det_scores) if has_both else None
        pr = _pr_auc(is_high, det_scores) if has_both else None
        sp, _ = spearmanr(scores, errors) if len(errors) > 5 else (0, 0)

        overall_metrics[name] = {
            "roc_auc": round(auc, 3) if auc is not None else None,
            "pr_auc": round(pr, 3) if pr is not None else None,
            "spearman": round(float(sp), 3),
        }
        auc_str = f"{auc:.3f}" if auc is not None else "?"
        print(f"  {name:25s}: AUC={auc_str:>5s}, Spearman={sp:.3f}")

    # ====================================================================
    # Long-occ subset
    # ====================================================================
    print("\n--- Long-Occ Subset (occ_length > 10) ---")
    long_occ_mask = occ_len > 10
    has_both_long = is_high[long_occ_mask].sum() > 0 and (1 - is_high[long_occ_mask]).sum() > 0

    long_occ_metrics = {}
    for name, info in scores_dict.items():
        scores = info["scores"]
        direction = info["direction"]
        if direction == "risk":
            det_scores = -scores
        else:
            det_scores = scores
        sub_det = det_scores[long_occ_mask]
        sub_high = is_high[long_occ_mask]
        auc = _auc(sub_high, sub_det) if has_both_long else None
        sp, _ = spearmanr(scores[long_occ_mask], errors[long_occ_mask]) if long_occ_mask.sum() > 5 else (0, 0)
        long_occ_metrics[name] = {"roc_auc": round(auc, 3) if auc else None, "spearman": round(float(sp), 3)}
        auc_str = f"{auc:.3f}" if auc is not None else "?"
        print(f"  {name:25s}: AUC={auc_str:>5s}, Spearman={sp:.3f}")

    # ====================================================================
    # Coverage-aligned accept table
    # ====================================================================
    print("\n--- Coverage-Aligned Accept Table ---")
    coverages = [0.50, 0.70, 0.90]

    coverage_tables = {}
    for name, info in scores_dict.items():
        scores = info["scores"]
        direction = info["direction"]
        if direction == "risk":
            ranking_scores = -scores
        else:
            ranking_scores = scores
        table = selective_eval_by_coverage(ranking_scores, errors, is_high, coverages)
        coverage_tables[name] = table
        print(f"\n  {name}:")
        for row in table:
            print(f"    cov={row['coverage']:.2f} mean_err={row['mean_error']:.2f} "
                  f"med_err={row['median_error']:.2f} high%={row['high_error_frac']:.3f} lt4px={row['lt4px_frac']:.3f}")

    # ====================================================================
    # Per-video summary
    # ====================================================================
    videos = sorted(set(s["video_name"] for s in all_samples))
    per_video = []
    for vid in videos:
        idx = [i for i, s in enumerate(all_samples) if s["video_name"] == vid]
        if len(idx) == 0: continue
        v_err = errors[idx]
        v_high = is_high[idx]
        has_both_v = v_high.sum() > 0 and (1 - v_high).sum() > 0

        entry = {"video": vid, "n": len(idx), "high_error_n": int(v_high.sum())}
        for name, info in scores_dict.items():
            scores = info["scores"]
            direction = info["direction"]
            v_scores = scores[idx]
            if direction == "risk":
                det = -v_scores
            else:
                det = v_scores
            entry[f"{name}_auc"] = round(_auc(v_high, det), 3) if has_both_v else None
            sp, _ = spearmanr(v_scores, v_err)
            entry[f"{name}_spearman"] = round(float(sp), 3)
        per_video.append(entry)

    # ====================================================================
    # Pass/Fail decision
    # ====================================================================
    nd_overall = overall_metrics.get("neighbor_dev_confidence", {})
    vis_overall = overall_metrics.get("pred_visibility", {})
    nd_long = long_occ_metrics.get("neighbor_dev_confidence", {})
    vis_long = long_occ_metrics.get("pred_visibility", {})

    # Criterion 1: long_occ dev > pred_visibility
    c1 = (nd_long.get("roc_auc") or 0) > (vis_long.get("roc_auc") or 0)

    # Criterion 2: overall not worse
    c2 = (nd_overall.get("roc_auc") or 0) >= (vis_overall.get("roc_auc") or 0) - 0.05

    # Criterion 3: coverage-aligned — at 50% coverage, nd accept set error < vis
    nd_c50 = next((r for r in coverage_tables.get("neighbor_dev_confidence", []) if r["coverage"] == 0.50), None)
    vis_c50 = next((r for r in coverage_tables.get("pred_visibility", []) if r["coverage"] == 0.50), None)
    c3 = True  # Default pass if we can't compare
    if nd_c50 and vis_c50:
        c3 = nd_c50["mean_error"] <= vis_c50["mean_error"] + 0.5

    passed = sum([c1, c2, c3])
    verdict = "PASS" if passed >= 2 else "FAIL"

    # Save
    summary = {
        "n_samples": len(all_samples),
        "error_threshold": args.error_threshold,
        "neighbor_k": args.neighbor_k,
        "overall_metrics": overall_metrics,
        "long_occ_metrics": long_occ_metrics,
        "coverage_tables": coverage_tables,
        "per_video": per_video,
        "pass_fail": {
            "long_occ_better_than_visibility": bool(c1),
            "overall_not_worse": bool(c2),
            "coverage_aligned_better": bool(c3),
            "criteria_passed": passed,
        },
        "verdict": verdict,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Predicted-Track Neighbor Deviation Diagnosis")
    print(f"{'='*60}")
    print(f"  Samples: {len(all_samples)}")
    print(f"  Verdict: {verdict} ({passed}/3)")
    print(f"  l1 long_occ better: {c1}")
    print(f"  l2 overall ok: {c2}")
    print(f"  l3 coverage ok: {c3}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
