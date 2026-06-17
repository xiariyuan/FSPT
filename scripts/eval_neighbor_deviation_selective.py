#!/usr/bin/env python3
"""
Neighbor-Deviation Reliability + Calibrated Selective Tracking Evaluation.

Reads existing trajectory manifold per-sample predictions, adds calibration
(Platt scaling / isotonic regression), runs selective tracking evaluation
with coverage sweep.

Convention:
  - reliability detection asks whether a score can detect high-error tracks
  - calibration fits risk probabilities p(high_error | score)
  - selective tracking ranks by reliability, so it must use 1 - risk

Usage:
  python scripts/eval_neighbor_deviation_selective.py \
    --input-jsonl outputs/trajectory_manifold_verifier_v1/per_sample_predictions.jsonl \
    --loocv-results outputs/trajectory_manifold_verifier_v1_loocv/results.json \
    --output-dir outputs/neighbor_deviation_selective_v1 \
    --error-threshold 16.0
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(labels, scores))
    except: return 0.5

def compute_pr_auc(labels, scores):
    from sklearn.metrics import average_precision_score
    try: return float(average_precision_score(labels, scores))
    except: return 0.0

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece / max(1, len(labels)))

def reliability_bins(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        result.append({
            "bin_lo": round(bins[i], 2), "bin_hi": round(bins[i+1], 2),
            "n": int(mask.sum()),
            "mean_predicted": round(float(probs[mask].mean()), 4),
            "actual_positive": round(float(labels[mask].mean()), 4),
        })
    return result


def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=float)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return np.ones_like(scores) * 0.5


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def platt_scale(scores, labels):
    """Platt scaling: fit logistic regression on scores."""
    from sklearn.linear_model import LogisticRegression
    X = scores.reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(X, labels)
    calibrated = lr.predict_proba(X)[:, 1]
    return calibrated, lr

def isotonic_calibrate(scores, labels):
    """Isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, labels)
    calibrated = ir.predict(scores)
    return calibrated, ir


def build_risk_views(raw_scores, labels, orientation):
    """Build raw/platt/isotonic risk probabilities.

    orientation:
      - "risk": higher raw score means higher error risk
      - "reliability": higher raw score means lower error risk
    """
    if orientation not in ("risk", "reliability"):
        raise ValueError(f"Unknown orientation: {orientation}")

    fit_scores = np.asarray(raw_scores, dtype=float)
    if orientation == "reliability":
        fit_scores = -fit_scores

    raw_risk = minmax_normalize(fit_scores)
    platt_risk, lr = platt_scale(fit_scores, labels)
    isotonic_risk, ir = isotonic_calibrate(fit_scores, labels)

    return {
        "fit_scores": fit_scores,
        "raw_risk": raw_risk,
        "platt_risk": platt_risk,
        "isotonic_risk": isotonic_risk,
        "platt_model": lr,
        "isotonic_model": ir,
    }


# ---------------------------------------------------------------------------
# Selective tracking evaluation
# ---------------------------------------------------------------------------
def selective_eval(scores, errors, is_high, coverage_levels):
    """Evaluate selective tracking at multiple coverage levels.

    Higher score = more reliable. Select top-coverage fraction by score.
    """
    n = len(scores)
    order = np.argsort(scores)[::-1]  # highest score first

    results = []
    for cov in coverage_levels:
        k = max(1, int(cov * n))
        selected = order[:k]
        sel_errors = errors[selected]
        sel_high = is_high[selected]

        results.append({
            "coverage": round(cov, 2),
            "n_selected": k,
            "mean_error": round(float(np.mean(sel_errors)), 2),
            "median_error": round(float(np.median(sel_errors)), 2),
            "high_error_frac": round(float(sel_high.mean()), 3),
            "precision": round(float(1 - sel_high.mean()), 3),
        })

    # AURC (trapezoidal)
    risks = [r["high_error_frac"] for r in results]
    covs = [r["coverage"] for r in results]
    aurc = float(np.trapz(risks, covs))

    # Selective AJ: max difference in accuracy across coverage
    accs = [1 - r["high_error_frac"] for r in results]
    max_aj = float(max(accs) - min(accs)) if accs else 0

    return {
        "coverage_levels": results,
        "aurc": round(aurc, 4),
        "max_selective_aj": round(max_aj, 4),
        "coverage_90pct_error": next((r["mean_error"] for r in results if abs(r["coverage"] - 0.9) < 0.05), None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--loocv-results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--error-threshold", type=float, default=16.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-sample predictions
    samples = [json.loads(l) for l in open(args.input_jsonl) if l.strip()]
    print(f"Loaded {len(samples)} samples")

    # Filter valid (has neighbor_deviation)
    valid = [s for s in samples if s.get("neighbor_deviation") is not None]
    print(f"Valid: {len(valid)}")

    errors = np.array([s["mean_error_px"] for s in valid])
    is_high = np.array([s["is_high_error"] for s in valid], dtype=float)
    vis_scores = np.array([s["visibility_score"] for s in valid])
    stat_scores = np.array([-s["statistics_score"] for s in valid])  # negative: lower stat = more reliable
    recon_scores = np.array([s.get("reconstruction_error", 0) or 0 for s in valid])
    dev_scores = np.array([s.get("neighbor_deviation", 0) or 0 for s in valid])

    # Subset masks
    occ_lengths = np.array([s.get("occ_length", 0) for s in valid])
    long_occ = occ_lengths > 10
    hard = errors > args.error_threshold
    easy = errors <= 4.0

    # ====================================================================
    # Phase 1: Baseline summary (reliability detection)
    # ====================================================================
    print("\n--- Phase 1: Reliability Detection ---")
    baseline = {}
    for name, scores in [("visibility", vis_scores), ("statistics", stat_scores),
                          ("reconstruction", recon_scores), ("neighbor_deviation", dev_scores)]:
        auc = compute_auc(is_high, scores)
        pr = compute_pr_auc(is_high, scores)
        from scipy.stats import pearsonr, spearmanr
        pearson, _ = pearsonr(scores, errors)
        spearman, _ = spearmanr(scores, errors)
        baseline[name] = {
            "roc_auc": round(auc, 3), "pr_auc": round(pr, 3),
            "pearson": round(float(pearson), 3), "spearman": round(float(spearman), 3),
        }
        print(f"  {name:25s}: AUC={auc:.3f}, PR-AUC={pr:.3f}, Spearman={spearman:.3f}")

    # LOOCV metrics from existing results
    loocv = json.load(open(args.loocv_results))
    per_fold = loocv.get("loocv_averages", {}).get("per_fold", [])

    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump({"reliability_detection": baseline, "loocv_avg": {
            "visibility": loocv.get("loocv_averages", {}).get("avg_vis_auc"),
            "neighbor_deviation": loocv.get("loocv_averages", {}).get("avg_dev_auc"),
            "learned": loocv.get("loocv_averages", {}).get("avg_learned_auc"),
        }}, f, indent=2)

    # ====================================================================
    # Phase 2: Calibration
    # ====================================================================
    print("\n--- Phase 2: Calibration (risk probability) ---")

    # Convenience reliability score for raw neighbor deviation selective ranking.
    dev_inv = -dev_scores

    calibration_results = {}
    calibration_views = {}
    calibration_specs = [
        ("visibility", vis_scores, "reliability"),
        ("neighbor_deviation", dev_scores, "risk"),
    ]

    for name, raw_scores, orientation in calibration_specs:
        views = build_risk_views(raw_scores, is_high, orientation)
        calibration_views[name] = views

        raw_risk = views["raw_risk"]
        ece_raw = compute_ece(raw_risk, is_high)
        brier_raw = float(np.mean((raw_risk - is_high) ** 2))
        calibration_results[f"{name}_raw"] = {
            "score_source": name,
            "method": "raw",
            "probability_type": "risk",
            "input_orientation": orientation,
            "ece": round(ece_raw, 4),
            "brier": round(brier_raw, 4),
            "mean_predicted": round(float(raw_risk.mean()), 4),
            "actual_positive": round(float(is_high.mean()), 4),
        }
        print(f"  {name} raw-risk: ECE={ece_raw:.4f}, Brier={brier_raw:.4f}")

        platt_risk = views["platt_risk"]
        lr = views["platt_model"]
        ece_p = compute_ece(platt_risk, is_high)
        brier_p = float(np.mean((platt_risk - is_high) ** 2))
        calibration_results[f"{name}_platt"] = {
            "score_source": name,
            "method": "platt",
            "probability_type": "risk",
            "input_orientation": orientation,
            "ece": round(ece_p, 4),
            "brier": round(brier_p, 4),
            "mean_predicted": round(float(platt_risk.mean()), 4),
            "actual_positive": round(float(is_high.mean()), 4),
            "platt_a": round(float(lr.coef_[0][0]), 4),
            "platt_b": round(float(lr.intercept_[0]), 4),
        }
        print(f"  {name} platt-risk: ECE={ece_p:.4f}, Brier={brier_p:.4f}")

        isotonic_risk = views["isotonic_risk"]
        ece_i = compute_ece(isotonic_risk, is_high)
        brier_i = float(np.mean((isotonic_risk - is_high) ** 2))
        calibration_results[f"{name}_isotonic"] = {
            "score_source": name,
            "method": "isotonic",
            "probability_type": "risk",
            "input_orientation": orientation,
            "ece": round(ece_i, 4),
            "brier": round(brier_i, 4),
            "mean_predicted": round(float(isotonic_risk.mean()), 4),
            "actual_positive": round(float(is_high.mean()), 4),
        }
        print(f"  {name} isotonic-risk: ECE={ece_i:.4f}, Brier={brier_i:.4f}")

    # Save calibration results
    for key, val in calibration_results.items():
        method = val["method"]
        source = val["score_source"]
        fname = f"calibration_{method}.json" if method != "raw" else "calibration_raw.json"
        # Append to per-method file
        fpath = out_dir / fname
        existing = json.load(open(fpath)) if fpath.exists() else {}
        existing[source] = val
        with open(fpath, "w") as f:
            json.dump(existing, f, indent=2)

    # Risk bins for best calibrated score.
    if "neighbor_deviation_platt" in calibration_results:
        rb = reliability_bins(calibration_views["neighbor_deviation"]["platt_risk"], is_high)
        with open(out_dir / "reliability_bins.json", "w") as f:
            json.dump(
                {
                    "score": "neighbor_deviation_platt_risk",
                    "label": "high_error",
                    "bins": rb,
                },
                f,
                indent=2,
            )

    # ====================================================================
    # Phase 3: Selective tracking evaluation
    # ====================================================================
    print("\n--- Phase 3: Selective Tracking ---")
    coverage_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    # Convert calibrated risk probabilities into reliability scores for ranking.
    dev_platt_risk = calibration_views["neighbor_deviation"]["platt_risk"]
    vis_platt_risk = calibration_views["visibility"]["platt_risk"]
    dev_platt_rel = 1.0 - dev_platt_risk
    vis_platt_rel = 1.0 - vis_platt_risk

    score_sources = {
        "visibility_raw": vis_scores,
        "visibility_platt": vis_platt_rel,
        "neighbor_dev_raw": dev_inv,
        "neighbor_dev_platt": dev_platt_rel,
    }

    selective_results = {}
    for name, scores in score_sources.items():
        sel = selective_eval(scores, errors, is_high, coverage_levels)
        selective_results[name] = sel
        print(f"  {name:25s}: AURC={sel['aurc']:.4f}, max_sel_aj={sel['max_selective_aj']:.4f}")

    # Overall selective metrics
    with open(out_dir / "selective_metrics_overall.json", "w") as f:
        json.dump(selective_results, f, indent=2)

    # Subset selective metrics
    subset_selective = {}
    for subset_name, mask in [("long_occ", long_occ), ("hard", hard), ("easy", easy), ("all", np.ones(len(valid), bool))]:
        if mask.sum() == 0: continue
        subset_sel = {}
        for name, scores in score_sources.items():
            sel = selective_eval(scores[mask], errors[mask], is_high[mask], coverage_levels)
            subset_sel[name] = sel
        subset_selective[subset_name] = {"n": int(mask.sum()), "scores": subset_sel}

    with open(out_dir / "selective_metrics_subsets.json", "w") as f:
        json.dump(subset_selective, f, indent=2)

    # Risk-coverage curves
    rc_curves = {}
    for name, scores in score_sources.items():
        order = np.argsort(scores)[::-1]
        curve = []
        for cov in coverage_levels:
            k = max(1, int(cov * len(scores)))
            sel = order[:k]
            curve.append({"coverage": round(cov, 2), "mean_error": round(float(np.mean(errors[sel])), 2),
                          "high_error_frac": round(float(is_high[sel].mean()), 3)})
        rc_curves[name] = curve

    with open(out_dir / "risk_coverage_curves.json", "w") as f:
        json.dump(rc_curves, f, indent=2)

    # Coverage sweep table
    sweep_table = []
    for cov in coverage_levels:
        row = {"coverage": round(cov, 2)}
        for name, scores in score_sources.items():
            order = np.argsort(scores)[::-1]
            k = max(1, int(cov * len(scores)))
            sel = order[:k]
            row[f"{name}_mean_error"] = round(float(np.mean(errors[sel])), 2)
            row[f"{name}_high_error_frac"] = round(float(is_high[sel].mean()), 3)
        sweep_table.append(row)

    with open(out_dir / "coverage_sweep_table.json", "w") as f:
        json.dump(sweep_table, f, indent=2)

    # Per-sample predictions (updated with all scores)
    for i, s in enumerate(valid):
        s["neighbor_dev_inv"] = round(float(dev_inv[i]), 6)
        s["neighbor_dev_risk_platt"] = round(float(dev_platt_risk[i]), 6)
        s["neighbor_dev_rel_platt"] = round(float(dev_platt_rel[i]), 6)
        s["visibility_risk_platt"] = round(float(vis_platt_risk[i]), 6)
        s["visibility_rel_platt"] = round(float(vis_platt_rel[i]), 6)

    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for s in valid:
            f.write(json.dumps(s) + "\n")

    # Subset metrics raw
    subset_raw = {}
    for name, scores in [("visibility", vis_scores), ("neighbor_deviation", dev_scores)]:
        subset_raw[name] = {}
        for sname, mask in [("all", np.ones(len(valid), bool)), ("long_occ", long_occ), ("hard", hard), ("easy", easy)]:
            if mask.sum() == 0: continue
            sub_s = scores[mask]
            sub_e = errors[mask]
            from scipy.stats import spearmanr
            sp, _ = spearmanr(sub_s, sub_e)
            sub_high = is_high[mask]
            has_both = sub_high.sum() > 0 and (1 - sub_high).sum() > 0
            subset_raw[name][sname] = {
                "n": int(mask.sum()),
                "spearman": round(float(sp), 3),
                "auc": round(compute_auc(sub_high, sub_s), 3) if has_both else None,
                "error_median": round(float(np.median(sub_e)), 2),
                "high_error_frac": round(float(sub_high.mean()), 3),
            }

    with open(out_dir / "subset_metrics_raw.json", "w") as f:
        json.dump(subset_raw, f, indent=2)

    # Per-fold from LOOCV
    with open(out_dir / "per_fold_metrics.json", "w") as f:
        json.dump(per_fold, f, indent=2)

    # ====================================================================
    # Print summary
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"Neighbor-Deviation Selective Tracking v1")
    print(f"{'='*70}")
    print(f"\n  Reliability Detection:")
    for name, m in baseline.items():
        print(f"    {name:25s}: AUC={m['roc_auc']:.3f}, Spearman={m['spearman']:.3f}")

    print(f"\n  Calibration:")
    for key, val in calibration_results.items():
        print(f"    {key:35s}: ECE={val['ece']:.4f}, Brier={val['brier']:.4f}")

    print(f"\n  Selective Tracking (overall):")
    for name, sel in selective_results.items():
        print(f"    {name:25s}: AURC={sel['aurc']:.4f}, max_sel_aj={sel['max_selective_aj']:.4f}")

    print(f"\n  Selective Tracking (hard, n={int(hard.sum())}):")
    if "hard" in subset_selective:
        for name, sel in subset_selective["hard"]["scores"].items():
            print(f"    {name:25s}: AURC={sel['aurc']:.4f}, max_sel_aj={sel['max_selective_aj']:.4f}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
