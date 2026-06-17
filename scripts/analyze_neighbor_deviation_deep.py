#!/usr/bin/env python3
"""
Deep analysis of neighbor-deviation reliability signal.

Covers:
  1. Long-occ deep dive (per-video, occ-length bins)
  2. Failure case taxonomy (smooth-but-wrong, false-safe, etc.)
  3. Coverage-driven + fixed-risk threshold tables
  4. Learned head ablation
  5. Pseudo-label filtering prep stats

Usage:
  python scripts/analyze_neighbor_deviation_deep.py \
    --input-jsonl outputs/trajectory_manifold_verifier_v1/per_sample_predictions.jsonl \
    --loocv-results outputs/trajectory_manifold_verifier_v1_loocv/results.json \
    --output-dir outputs/neighbor_deviation_deep_analysis_v1 \
    --error-threshold 16.0
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def _auc(labels, scores):
    try: return float(roc_auc_score(labels, scores))
    except: return 0.5


def _minmax_apply(train_scores, test_scores):
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    s_min = float(train_scores.min())
    s_max = float(train_scores.max())
    if s_max > s_min:
        return np.clip((test_scores - s_min) / (s_max - s_min), 0.0, 1.0)
    return np.ones_like(test_scores) * 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--loocv-results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--error-threshold", type=float, default=16.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(l) for l in open(args.input_jsonl) if l.strip()]
    valid = [s for s in samples if s.get("neighbor_deviation") is not None]
    print(f"Samples: {len(samples)}, Valid: {len(valid)}")

    errors = np.array([s["mean_error_px"] for s in valid])
    is_high = np.array([s["is_high_error"] for s in valid], dtype=float)
    dev_scores = np.array([s.get("neighbor_deviation", 0) or 0 for s in valid])
    vis_scores = np.array([s["visibility_score"] for s in valid])
    occ_lengths = np.array([s.get("occ_length", 0) for s in valid])

    # Build held-out raw risk probabilities per video.
    videos = sorted(set(s["video_name"] for s in valid))
    nd_raw_risk = np.zeros(len(valid), dtype=float)
    vis_raw_risk = np.zeros(len(valid), dtype=float)
    for vid in videos:
        test_idx = np.array([i for i, s in enumerate(valid) if s["video_name"] == vid], dtype=int)
        train_idx = np.array([i for i, s in enumerate(valid) if s["video_name"] != vid], dtype=int)
        nd_raw_risk[test_idx] = _minmax_apply(dev_scores[train_idx], dev_scores[test_idx])
        vis_raw_risk[test_idx] = _minmax_apply(-vis_scores[train_idx], -vis_scores[test_idx])

    nd_rel = -dev_scores
    vis_rel = vis_scores

    # ====================================================================
    # 1. Long-occ deep dive
    # ====================================================================
    print("\n--- 1. Long-Occ Deep Dive ---")

    # Per-video analysis
    per_video = []
    for vid in videos:
        idx = [i for i, s in enumerate(valid) if s["video_name"] == vid]
        v_err = errors[idx]
        v_high = is_high[idx]
        v_dev = dev_scores[idx]
        v_vis = vis_scores[idx]
        v_occ = occ_lengths[idx]

        has_both = v_high.sum() > 0 and (1 - v_high).sum() > 0
        sp_dev, _ = spearmanr(v_dev, v_err) if len(v_err) > 2 else (0, 0)
        sp_vis, _ = spearmanr(v_vis, v_err) if len(v_err) > 2 else (0, 0)

        entry = {
            "video": vid,
            "n": len(idx),
            "high_error_n": int(v_high.sum()),
            "high_error_frac": round(float(v_high.mean()), 3),
            "error_median": round(float(np.median(v_err)), 2),
            "occ_median": round(float(np.median(v_occ)), 1),
            "dev_auc": round(_auc(v_high, v_dev), 3) if has_both else None,
            "vis_risk_auc": round(_auc(v_high, -v_vis), 3) if has_both else None,
            "dev_spearman": round(float(sp_dev), 3),
            "vis_spearman": round(float(sp_vis), 3),
            "dev_median": round(float(np.median(v_dev)), 6),
        }
        per_video.append(entry)

    # Sort by dev_auc descending
    per_video.sort(key=lambda x: x.get("dev_auc") or 0, reverse=True)

    print(f"  {'Video':25s} {'n':>5s} {'high%':>6s} {'err_med':>8s} {'occ_med':>8s} {'dev_AUC':>8s} {'vis_risk':>8s}")
    for v in per_video:
        print(f"  {v['video']:25s} {v['n']:>5d} {v['high_error_frac']:>6.3f} {v['error_median']:>8.2f} {v['occ_median']:>8.1f} "
              f"{str(v.get('dev_auc','—')):>8s} {str(v.get('vis_risk_auc','—')):>8s}")

    # Occ-length bins
    occ_bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 300)]
    occ_bin_stats = []
    for lo, hi in occ_bins:
        mask = (occ_lengths >= lo) & (occ_lengths < hi)
        if mask.sum() == 0: continue
        m_high = is_high[mask]
        m_dev = dev_scores[mask]
        m_vis = vis_scores[mask]
        has_both = m_high.sum() > 0 and (1 - m_high).sum() > 0
        sp, _ = spearmanr(m_dev, errors[mask]) if mask.sum() > 2 else (0, 0)
        occ_bin_stats.append({
            "occ_range": f"{lo}-{hi}",
            "n": int(mask.sum()),
            "high_frac": round(float(m_high.mean()), 3),
            "error_median": round(float(np.median(errors[mask])), 2),
            "dev_auc": round(_auc(m_high, m_dev), 3) if has_both else None,
            "vis_risk_auc": round(_auc(m_high, -m_vis), 3) if has_both else None,
            "dev_spearman": round(float(sp), 3),
        })

    print(f"\n  Occ-length bins:")
    print(f"  {'Range':12s} {'n':>5s} {'high%':>6s} {'err_med':>8s} {'dev_AUC':>8s} {'vis_risk':>8s}")
    for b in occ_bin_stats:
        print(f"  {b['occ_range']:12s} {b['n']:>5d} {b['high_frac']:>6.3f} {b['error_median']:>8.2f} "
              f"{str(b.get('dev_auc','—')):>8s} {str(b.get('vis_risk_auc','—')):>8s}")

    with open(out_dir / "long_occ_per_video.json", "w") as f:
        json.dump(per_video, f, indent=2)
    with open(out_dir / "long_occ_bins.json", "w") as f:
        json.dump(occ_bin_stats, f, indent=2)

    # ====================================================================
    # 2. Failure case taxonomy
    # ====================================================================
    print("\n--- 2. Failure Case Taxonomy ---")

    # Category A: smooth-but-wrong (low deviation but high error)
    smooth_but_wrong = []
    for i, s in enumerate(valid):
        if s["is_high_error"] and dev_scores[i] < np.percentile(dev_scores, 25):
            smooth_but_wrong.append({
                "sample_id": s["sample_id"], "video": s["video_name"],
                "point_idx": s["point_idx"], "occ_length": s.get("occ_length", 0),
                "error_px": round(s["mean_error_px"], 2),
                "visibility": round(s["visibility_score"], 4),
                "deviation": round(dev_scores[i], 6),
            })
    smooth_but_wrong.sort(key=lambda x: -x["error_px"])

    # Category B: false-safe (low deviation, NOT high error, but still >4px)
    false_safe = []
    for i, s in enumerate(valid):
        if not s["is_high_error"] and s["mean_error_px"] > 4.0 and dev_scores[i] < np.percentile(dev_scores, 25):
            false_safe.append({
                "sample_id": s["sample_id"], "video": s["video_name"],
                "point_idx": s["point_idx"], "error_px": round(s["mean_error_px"], 2),
                "deviation": round(dev_scores[i], 6),
            })

    # Category C: high-deviation-but-accurate (high deviation but low error)
    high_dev_accurate = []
    for i, s in enumerate(valid):
        if s["mean_error_px"] < 4.0 and dev_scores[i] > np.percentile(dev_scores, 75):
            high_dev_accurate.append({
                "sample_id": s["sample_id"], "video": s["video_name"],
                "point_idx": s["point_idx"], "error_px": round(s["mean_error_px"], 2),
                "deviation": round(dev_scores[i], 6),
            })

    # Category D: neighbor-drift (high deviation, high error, neighbors also wrong)
    neighbor_drift = []
    for i, s in enumerate(valid):
        if s["is_high_error"] and dev_scores[i] > np.percentile(dev_scores, 50):
            neighbor_drift.append({
                "sample_id": s["sample_id"], "video": s["video_name"],
                "point_idx": s["point_idx"], "error_px": round(s["mean_error_px"], 2),
                "deviation": round(dev_scores[i], 6),
                "occ_length": s.get("occ_length", 0),
            })

    failure_taxonomy = {
        "smooth_but_wrong": {"n": len(smooth_but_wrong), "description": "Low deviation but high error — neighbors also wrong",
                              "top_cases": smooth_but_wrong[:10]},
        "false_safe": {"n": len(false_safe), "description": "Low deviation, moderate error (4-16px) — appears safe but not",
                       "top_cases": false_safe[:10]},
        "high_dev_accurate": {"n": len(high_dev_accurate), "description": "High deviation but low error — outlier with lucky outcome",
                              "top_cases": high_dev_accurate[:10]},
        "neighbor_drift": {"n": len(neighbor_drift), "description": "High deviation, high error, neighbors also off",
                           "top_cases": neighbor_drift[:10]},
    }

    for cat, data in failure_taxonomy.items():
        print(f"  {cat:25s}: n={data['n']:>4d}  ({data['description']})")

    with open(out_dir / "failure_cases.json", "w") as f:
        json.dump(failure_taxonomy, f, indent=2, default=str)

    # ====================================================================
    # 3. Threshold sweep tables
    # ====================================================================
    print("\n--- 3. Threshold Sweep Tables ---")

    # Coverage-driven
    coverages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    cov_table = []
    for cov in coverages:
        # Select top-coverage fraction by reliability.
        k = max(1, int(cov * len(valid)))
        sel_dev = np.argsort(nd_rel)[::-1][:k]
        sel_vis = np.argsort(vis_rel)[::-1][:k]

        cov_table.append({
            "coverage": round(cov, 2),
            "dev_mean_error": round(float(np.mean(errors[sel_dev])), 2),
            "dev_high_frac": round(float(is_high[sel_dev].mean()), 3),
            "vis_mean_error": round(float(np.mean(errors[sel_vis])), 2),
            "vis_high_frac": round(float(is_high[sel_vis].mean()), 3),
        })

    print(f"  {'Cov':>5s} {'dev_err':>8s} {'dev_high%':>9s} {'vis_err':>8s} {'vis_high%':>9s}")
    for row in cov_table:
        print(f"  {row['coverage']:>5.2f} {row['dev_mean_error']:>8.2f} {row['dev_high_frac']:>9.3f} "
              f"{row['vis_mean_error']:>8.2f} {row['vis_high_frac']:>9.3f}")

    with open(out_dir / "threshold_sweep_by_coverage.json", "w") as f:
        json.dump(cov_table, f, indent=2)

    # Risk-driven: fixed predicted risk thresholds
    # Use held-out raw-risk probability derived from neighbor deviation.
    risk_thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    risk_table = []
    for thr in risk_thresholds:
        accept = nd_raw_risk < thr
        n_accept = int(accept.sum())
        if n_accept == 0: continue
        risk_table.append({
            "risk_threshold": thr,
            "n_accept": n_accept,
            "coverage": round(n_accept / len(valid), 3),
            "mean_error": round(float(np.mean(errors[accept])), 2),
            "high_error_frac": round(float(is_high[accept].mean()), 3),
        })

    print(f"\n  Risk threshold sweep:")
    print(f"  {'Thr':>5s} {'N':>5s} {'Cov':>5s} {'Err':>7s} {'High%':>6s}")
    for row in risk_table:
        print(f"  {row['risk_threshold']:>5.2f} {row['n_accept']:>5d} {row['coverage']:>5.3f} {row['mean_error']:>7.2f} {row['high_error_frac']:>6.3f}")

    with open(out_dir / "threshold_sweep_by_risk.json", "w") as f:
        json.dump(risk_table, f, indent=2)

    # ====================================================================
    # 4. Learned head ablation
    # ====================================================================
    print("\n--- 4. Learned Head Ablation ---")
    loocv = json.load(open(args.loocv_results))
    per_fold = loocv.get("loocv_averages", {}).get("per_fold", [])

    # Per-fold comparison: dev_auc vs learned_auc
    fold_comparison = []
    for fold in per_fold:
        vid = fold.get("test_video", "?")
        dev_a = fold.get("dev_auc")
        learn_a = fold.get("learned_auc")
        if dev_a is not None and learn_a is not None:
            fold_comparison.append({
                "video": vid,
                "dev_auc": dev_a,
                "learned_auc": learn_a,
                "gain": round(learn_a - dev_a, 3),
            })

    fold_comparison.sort(key=lambda x: x["gain"], reverse=True)
    print(f"  {'Video':25s} {'dev_AUC':>8s} {'learn_AUC':>10s} {'gain':>6s}")
    for fc in fold_comparison:
        print(f"  {fc['video']:25s} {fc['dev_auc']:>8.3f} {fc['learned_auc']:>10.3f} {fc['gain']:>+6.3f}")

    gains = [fc["gain"] for fc in fold_comparison]
    print(f"\n  Mean gain: {np.mean(gains):.3f}, Median gain: {np.median(gains):.3f}")
    print(f"  Gains > 0: {sum(1 for g in gains if g > 0)}/{len(gains)}")
    print(f"  Gains < -0.1: {sum(1 for g in gains if g < -0.1)}/{len(gains)}")

    learned_head_verdict = "appendix_only" if np.median(gains) < 0.05 or sum(1 for g in gains if g < -0.1) > len(gains) // 3 else "potential_enhancement"

    with open(out_dir / "learned_head_ablation.json", "w") as f:
        json.dump({
            "per_fold": fold_comparison,
            "mean_gain": round(float(np.mean(gains)), 3),
            "median_gain": round(float(np.median(gains)), 3),
            "gains_positive": sum(1 for g in gains if g > 0),
            "gains_negative_10pct": sum(1 for g in gains if g < -0.1),
            "verdict": learned_head_verdict,
        }, f, indent=2)

    # ====================================================================
    # 5. Pseudo-label filtering prep stats
    # ====================================================================
    print("\n--- 5. Pseudo-Label Filtering Prep Stats ---")

    # What fraction of tracks would be filtered at different risk thresholds?
    risk_thresholds_pl = [0.05, 0.10, 0.20, 0.30]
    pl_stats = []
    for thr in risk_thresholds_pl:
        accept = nd_raw_risk < thr
        reject = ~accept
        n_acc = int(accept.sum())
        n_rej = int(reject.sum())
        # How many high-error tracks are correctly rejected?
        correctly_rejected = int((reject & is_high.astype(bool)).sum())
        incorrectly_rejected = int((reject & ~is_high.astype(bool)).sum())
        missed_high = int((accept & is_high.astype(bool)).sum())
        pl_stats.append({
            "risk_threshold": thr,
            "n_accept": n_acc,
            "n_reject": n_rej,
            "coverage": round(n_acc / len(valid), 3),
            "correctly_rejected_high_error": correctly_rejected,
            "incorrectly_rejected_low_error": incorrectly_rejected,
            "missed_high_error_accepted": missed_high,
            "reject_precision": round(correctly_rejected / max(1, n_rej), 3),
        })

    print(f"  {'Thr':>5s} {'Accept':>7s} {'Reject':>7s} {'Cov':>5s} {'CorrectRej':>11s} {'FalseRej':>9s} {'Missed':>7s}")
    for row in pl_stats:
        print(f"  {row['risk_threshold']:>5.2f} {row['n_accept']:>7d} {row['n_reject']:>7d} {row['coverage']:>5.3f} "
              f"{row['correctly_rejected_high_error']:>11d} {row['incorrectly_rejected_low_error']:>9d} {row['missed_high_error_accepted']:>7d}")

    with open(out_dir / "pseudo_label_filtering_stats.json", "w") as f:
        json.dump(pl_stats, f, indent=2)

    # ====================================================================
    # Summary
    # ====================================================================
    summary = {
        "n_samples": len(valid),
        "long_occ_per_video": {"n_videos": len(per_video), "best_5": per_video[:5], "worst_5": per_video[-5:]},
        "long_occ_bins": occ_bin_stats,
        "failure_taxonomy": {k: {"n": v["n"], "description": v["description"]} for k, v in failure_taxonomy.items()},
        "learned_head": {"verdict": learned_head_verdict, "mean_gain": round(float(np.mean(gains)), 3)},
        "pseudo_label": {"best_threshold": min(pl_stats, key=lambda x: x["missed_high_error_accepted"])["risk_threshold"] if pl_stats else None},
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
