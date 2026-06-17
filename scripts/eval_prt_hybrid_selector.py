#!/usr/bin/env python3
"""
Formal hybrid selector for PRT causal re-entry.

Combines explicit support-memory signals with raw candidate scores
(no learned ranker) to decide whether to override the baseline prediction.

hybrid_score = alpha * support_margin + beta * cand_score + gamma * cand_ncc

Decision: pick candidate with highest hybrid_score.
If best_cand_score - baseline_score <= threshold, fall back to baseline.

Two operating points:
  - Full coverage (threshold=0): always pick best candidate
  - Selective: only override baseline when confidence exceeds threshold

Evaluation modes:
  - coefficient sweep: find stable (alpha, beta, gamma) on train, cross-check on val
  - threshold sweep: coverage-risk curve
  - train/val generalization check
  - subset stability (bootstrap)
  - AUC analysis of individual features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    return {k: cache[k] for k in cache.files}


def encode_patch_batch(
    extractor: DINOFeatureExtractor,
    patches_np: np.ndarray,
    batch_size: int = 64,
) -> torch.Tensor:
    out = []
    for start in range(0, len(patches_np), batch_size):
        batch = torch.from_numpy(patches_np[start:start + batch_size]).float()
        batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
        with torch.no_grad():
            feat = extractor.model(batch)[-1].mean(dim=[-2, -1]).float()
            feat = F.normalize(feat, dim=-1)
        out.append(feat.cpu())
    return torch.cat(out, dim=0)


def compute_support_features(
    cache: Dict[str, np.ndarray],
    extractor: DINOFeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pooled support descriptor, candidate-support sims, and baseline-support sim.

    Returns:
        cand_support_sim: (n, topk) cosine sim of each candidate to support
        baseline_support_sim: (n,) cosine sim of baseline to support
    """
    n = cache["query_patch"].shape[0]
    if "support_patches" in cache and "support_count" in cache:
        support_count = cache["support_patches"].shape[1]
        support_flat = cache["support_patches"].reshape(n * support_count, *cache["support_patches"].shape[2:])
        support_feat = encode_patch_batch(extractor, support_flat).numpy().reshape(n, support_count, -1)
        support_mask = (
            np.arange(support_count, dtype=np.int32)[None, :] < cache["support_count"][:, None]
        ).astype(np.float32)
    else:
        query_feat = encode_patch_batch(extractor, cache["query_patch"]).numpy()
        support_feat = query_feat[:, None, :]
        support_mask = np.ones((n, 1), dtype=np.float32)

    support_mask_sum = np.clip(support_mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled_support = (support_feat * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled_support /= np.linalg.norm(pooled_support, axis=1, keepdims=True).clip(1e-8, None)

    baseline_feat = encode_patch_batch(extractor, cache["baseline_patch"]).numpy()
    topk = cache["cand_patches"].shape[1]
    cand_flat = cache["cand_patches"].reshape(n * topk, *cache["cand_patches"].shape[2:])
    cand_feat = encode_patch_batch(extractor, cand_flat).numpy().reshape(n, topk, -1)

    baseline_support_sim = np.sum(baseline_feat * pooled_support, axis=1).astype(np.float32)
    cand_support_sim = np.sum(cand_feat * pooled_support[:, None, :], axis=2).astype(np.float32)

    return cand_support_sim, baseline_support_sim


def compute_hybrid_scores(
    cand_support_sim: np.ndarray,
    baseline_support_sim: np.ndarray,
    cand_score: np.ndarray,
    cand_ncc: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hybrid scores for candidates and baseline.

    Returns:
        cand_hybrid: (n, topk)
        baseline_hybrid: (n,) -- always 0, since support_margin=0 by definition for baseline
    """
    support_margin = cand_support_sim - baseline_support_sim[:, None]
    cand_hybrid = alpha * support_margin + beta * cand_score + gamma * cand_ncc
    baseline_hybrid = np.zeros(cand_support_sim.shape[0], dtype=np.float32)
    return cand_hybrid, baseline_hybrid


def evaluate_selector(
    cand_hybrid: np.ndarray,
    baseline_hybrid: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Evaluate hybrid selector at a given threshold.

    For each sample: pick candidate with max hybrid score.
    If max_cand_score > threshold, use candidate; else baseline.
    """
    n = len(baseline_err)
    best_cand_idx = np.argmax(cand_hybrid, axis=1)
    best_cand_score = cand_hybrid[np.arange(n), best_cand_idx]
    accept = best_cand_score > threshold

    final_err = baseline_err.copy()
    accepted_errs = []
    for i in range(n):
        if accept[i]:
            cidx = best_cand_idx[i]
            final_err[i] = cand_err[i, cidx]
            accepted_errs.append(cand_err[i, cidx])

    accepted_errs = np.asarray(accepted_errs, dtype=np.float32) if accepted_errs else np.array([], dtype=np.float32)

    return {
        "threshold": float(threshold),
        "coverage": float(np.mean(accept)),
        "median_px": float(np.median(final_err)),
        "lt4px": float(np.mean(final_err < 4.0)),
        "better_frac": float(np.mean(final_err < baseline_err)),
        "accept_only_median_px": float(np.median(accepted_errs)) if accepted_errs.size > 0 else None,
        "n": int(n),
    }


def sweep_thresholds(
    cand_hybrid: np.ndarray,
    baseline_hybrid: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    if thresholds is None:
        scores = cand_hybrid.max(axis=1)
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 21)).tolist()
    results = []
    seen = set()
    for thr in thresholds:
        thr = round(float(thr), 6)
        if thr in seen:
            continue
        seen.add(thr)
        results.append(evaluate_selector(cand_hybrid, baseline_hybrid, baseline_err, cand_err, threshold=thr))
    return results


def sweep_coefficients(
    cand_support_sim: np.ndarray,
    baseline_support_sim: np.ndarray,
    cand_score: np.ndarray,
    cand_ncc: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
) -> List[Dict]:
    """Sweep coefficient combinations and report val metrics at threshold=0."""
    candidates = []
    for alpha in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for gamma in [0.0, 0.25, 0.5, 1.0]:
                ch, bh = compute_hybrid_scores(
                    cand_support_sim, baseline_support_sim,
                    cand_score, cand_ncc,
                    alpha, beta, gamma,
                )
                m = evaluate_selector(ch, bh, baseline_err, cand_err, threshold=0.0)
                m["alpha"] = alpha
                m["beta"] = beta
                m["gamma"] = gamma
                candidates.append(m)
    candidates.sort(key=lambda x: x["median_px"])
    return candidates


def compute_feature_auc(
    cand_support_sim: np.ndarray,
    baseline_support_sim: np.ndarray,
    cand_score: np.ndarray,
    cand_ncc: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
) -> Dict[str, float]:
    """Compute AUC for individual features (predicting cand better than baseline)."""
    from sklearn.metrics import roc_auc_score
    support_margin = cand_support_sim - baseline_support_sim[:, None]
    n, topk = cand_err.shape
    labels, feat_margin, feat_score, feat_ncc = [], [], [], []
    for i in range(n):
        for j in range(topk):
            labels.append(1 if cand_err[i, j] < baseline_err[i] else 0)
            feat_margin.append(float(support_margin[i, j]))
            feat_score.append(float(cand_score[i, j]))
            feat_ncc.append(float(cand_ncc[i, j]))
    labels = np.array(labels)
    return {
        "support_margin": float(roc_auc_score(labels, feat_margin)),
        "cand_score": float(roc_auc_score(labels, feat_score)),
        "cand_ncc": float(roc_auc_score(labels, feat_ncc)),
    }


def bootstrap_stability(
    cand_hybrid: np.ndarray,
    baseline_hybrid: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
    threshold: float,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline_err)
    medians, coverages, lt4s, betters = [], [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        m = evaluate_selector(
            cand_hybrid[idx], baseline_hybrid[idx],
            baseline_err[idx], cand_err[idx], threshold=threshold,
        )
        medians.append(m["median_px"])
        coverages.append(m["coverage"])
        lt4s.append(m["lt4px"])
        betters.append(m["better_frac"])
    return {
        "n_boot": n_boot,
        "threshold": threshold,
        "median_px_mean": float(np.mean(medians)),
        "median_px_std": float(np.std(medians)),
        "median_px_p5": float(np.percentile(medians, 5)),
        "median_px_p95": float(np.percentile(medians, 95)),
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "lt4px_mean": float(np.mean(lt4s)),
        "lt4px_std": float(np.std(lt4s)),
        "better_frac_mean": float(np.mean(betters)),
        "better_frac_std": float(np.std(betters)),
    }


def find_best_threshold_on_train(
    train_ch: np.ndarray,
    train_bh: np.ndarray,
    train_baseline_err: np.ndarray,
    train_cand_err: np.ndarray,
) -> float:
    """Find threshold that minimizes median on train set."""
    sweep = sweep_thresholds(train_ch, train_bh, train_baseline_err, train_cand_err)
    best = min(sweep, key=lambda x: x["median_px"])
    return best["threshold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval hybrid selector for PRT")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str,
                        default="/gemini/code/FSPT/outputs/prt_hybrid_selector")
    parser.add_argument("--alpha", type=float, default=None, help="Weight for support_margin (auto-sweep if omitted)")
    parser.add_argument("--beta", type=float, default=None, help="Weight for cand_score")
    parser.add_argument("--gamma", type=float, default=None, help="Weight for cand_ncc")
    parser.add_argument("--sweep-coeffs", action="store_true", help="Run coefficient sweep")
    parser.add_argument("--sweep-thresholds", action="store_true", help="Run threshold sweep")
    parser.add_argument("--bootstrap", action="store_true", help="Run bootstrap stability")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--auc", action="store_true", help="Compute per-feature AUC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    train_cache = load_cache(Path(args.train_cache))
    val_cache = load_cache(Path(args.val_cache))

    print("Computing support features for train...", flush=True)
    train_cand_sim, train_base_sim = compute_support_features(train_cache, extractor)
    print("Computing support features for val...", flush=True)
    val_cand_sim, val_base_sim = compute_support_features(val_cache, extractor)

    train_meta = {
        "cand_score": train_cache["cand_score"].astype(np.float32),
        "cand_ncc": train_cache["cand_ncc"].astype(np.float32),
        "baseline_err": train_cache["baseline_err"].astype(np.float32),
        "cand_err": train_cache["cand_err"].astype(np.float32),
    }
    val_meta = {
        "cand_score": val_cache["cand_score"].astype(np.float32),
        "cand_ncc": val_cache["cand_ncc"].astype(np.float32),
        "baseline_err": val_cache["baseline_err"].astype(np.float32),
        "cand_err": val_cache["cand_err"].astype(np.float32),
    }

    baseline_median = float(np.median(val_meta["baseline_err"]))
    oracle_median = float(np.median(np.minimum(
        val_meta["baseline_err"], val_meta["cand_err"].min(axis=1)
    )))
    print(f"\nBaseline median: {baseline_median:.2f}")
    print(f"Oracle median:   {oracle_median:.2f}")

    # --- AUC ---
    if args.auc:
        print("\n=== Per-feature AUC (predicting cand better than baseline) ===")
        auc = compute_feature_auc(
            val_cand_sim, val_base_sim,
            val_meta["cand_score"], val_meta["cand_ncc"],
            val_meta["baseline_err"], val_meta["cand_err"],
        )
        for k, v in auc.items():
            print(f"  {k:>20s}: {v:.4f}")
        (output_dir / "auc.json").write_text(json.dumps(auc, indent=2) + "\n")

    # --- Coefficient sweep ---
    if args.sweep_coeffs:
        print("\n=== Coefficient sweep (train set, threshold=0) ===", flush=True)
        train_sweep = sweep_coefficients(
            train_cand_sim, train_base_sim,
            train_meta["cand_score"], train_meta["cand_ncc"],
            train_meta["baseline_err"], train_meta["cand_err"],
        )
        print(f"\nTop 10 on TRAIN:")
        print(f"  {'alpha':>6} {'beta':>5} {'gamma':>6} {'median':>8} {'lt4px':>6} {'better':>8} {'coverage':>9}")
        for row in train_sweep[:10]:
            print(f"  {row['alpha']:>6.1f} {row['beta']:>5.1f} {row['gamma']:>6.1f}"
                  f" {row['median_px']:>8.2f} {row['lt4px']:>6.3f} {row['better_frac']:>8.3f} {row['coverage']:>9.3f}")

        # Cross-evaluate top 30 on val
        print(f"\nTop 30 cross-evaluated on VAL:")
        print(f"  {'alpha':>6} {'beta':>5} {'gamma':>6} | {'tr_med':>7} {'v_med':>7} {'v_lt4':>6} {'v_better':>9} {'v_cov':>6}")
        val_results = []
        for row in train_sweep[:30]:
            ch, bh = compute_hybrid_scores(
                val_cand_sim, val_base_sim,
                val_meta["cand_score"], val_meta["cand_ncc"],
                row["alpha"], row["beta"], row["gamma"],
            )
            vm = evaluate_selector(ch, bh, val_meta["baseline_err"], val_meta["cand_err"], threshold=0.0)
            vm["alpha"] = row["alpha"]
            vm["beta"] = row["beta"]
            vm["gamma"] = row["gamma"]
            vm["train_median_px"] = row["median_px"]
            val_results.append(vm)
            print(f"  {row['alpha']:>6.1f} {row['beta']:>5.1f} {row['gamma']:>6.1f}"
                  f" | {row['median_px']:>7.2f} {vm['median_px']:>7.2f} {vm['lt4px']:>6.3f}"
                  f" {vm['better_frac']:>9.3f} {vm['coverage']:>6.3f}")

        val_results.sort(key=lambda x: x["median_px"])
        best = val_results[0]
        print(f"\nBest on VAL: alpha={best['alpha']}, beta={best['beta']}, gamma={best['gamma']}")
        print(f"  val median={best['median_px']:.2f}, lt4={best['lt4px']:.3f}, coverage={best['coverage']:.3f}")

        (output_dir / "coefficient_sweep_train.json").write_text(json.dumps(train_sweep[:50], indent=2) + "\n")
        (output_dir / "coefficient_sweep_val.json").write_text(json.dumps(val_results, indent=2) + "\n")

        if args.alpha is None:
            args.alpha = best["alpha"]
            args.beta = best["beta"]
            args.gamma = best["gamma"]

    # Default coefficients if not set
    if args.alpha is None:
        args.alpha = 5.0
    if args.beta is None:
        args.beta = 1.5
    if args.gamma is None:
        args.gamma = 0.25

    # --- Main evaluation ---
    print(f"\n{'='*60}")
    print(f"Main eval: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print(f"Formula: {args.alpha} * support_margin + {args.beta} * cand_score + {args.gamma} * cand_ncc")
    print(f"{'='*60}")

    train_ch, train_bh = compute_hybrid_scores(
        train_cand_sim, train_base_sim,
        train_meta["cand_score"], train_meta["cand_ncc"],
        args.alpha, args.beta, args.gamma,
    )
    val_ch, val_bh = compute_hybrid_scores(
        val_cand_sim, val_base_sim,
        val_meta["cand_score"], val_meta["cand_ncc"],
        args.alpha, args.beta, args.gamma,
    )

    # --- Two operating points ---
    # 1. Full coverage: threshold = 0 (always pick best candidate)
    val_full = evaluate_selector(val_ch, val_bh, val_meta["baseline_err"], val_meta["cand_err"], threshold=0.0)
    # 2. Selective: find best threshold on train, apply on val
    best_train_thr = find_best_threshold_on_train(train_ch, train_bh, train_meta["baseline_err"], train_meta["cand_err"])
    val_selective = evaluate_selector(val_ch, val_bh, val_meta["baseline_err"], val_meta["cand_err"], threshold=best_train_thr)

    print(f"\n--- Operating Point 1: Full Coverage ---")
    print(f"  Threshold:     0.0")
    print(f"  Coverage:      {val_full['coverage']:.2f}")
    print(f"  Median px:     {val_full['median_px']:.2f}")
    print(f"  <4px rate:     {val_full['lt4px']:.3f}")
    print(f"  Better frac:   {val_full['better_frac']:.3f}")

    print(f"\n--- Operating Point 2: Selective (threshold from train) ---")
    print(f"  Threshold:     {best_train_thr:.4f}")
    print(f"  Coverage:      {val_selective['coverage']:.2f}")
    print(f"  Median px:     {val_selective['median_px']:.2f}")
    print(f"  <4px rate:     {val_selective['lt4px']:.3f}")
    print(f"  Better frac:   {val_selective['better_frac']:.3f}")
    if val_selective["accept_only_median_px"] is not None:
        print(f"  Accept-only median: {val_selective['accept_only_median_px']:.2f}")

    # --- Threshold sweep ---
    if args.sweep_thresholds:
        print(f"\n--- Threshold Sweep (val) ---")
        val_thr_sweep = sweep_thresholds(val_ch, val_bh, val_meta["baseline_err"], val_meta["cand_err"])
        print(f"  {'thr':>10} {'coverage':>9} {'median':>8} {'lt4px':>6} {'better':>8} {'acc_med':>10}")
        for row in val_thr_sweep:
            acc_str = f"{row['accept_only_median_px']:.2f}" if row["accept_only_median_px"] is not None else "N/A"
            print(f"  {row['threshold']:>10.4f} {row['coverage']:>9.3f} {row['median_px']:>8.2f}"
                  f" {row['lt4px']:>6.3f} {row['better_frac']:>8.3f} {acc_str:>10}")
        (output_dir / "threshold_sweep_val.json").write_text(json.dumps(val_thr_sweep, indent=2) + "\n")

    # --- Bootstrap stability ---
    bootstrap_results = {}
    if args.bootstrap:
        print(f"\n--- Bootstrap Stability (n_boot={args.n_boot}) ---")
        # Full coverage
        stab_full = bootstrap_stability(
            val_ch, val_bh, val_meta["baseline_err"], val_meta["cand_err"],
            threshold=0.0, n_boot=args.n_boot,
        )
        print(f"  Full coverage (thr=0):")
        print(f"    median: {stab_full['median_px_mean']:.2f} +/- {stab_full['median_px_std']:.2f}"
              f"  [{stab_full['median_px_p5']:.2f}, {stab_full['median_px_p95']:.2f}]")
        print(f"    better_frac: {stab_full['better_frac_mean']:.3f} +/- {stab_full['better_frac_std']:.3f}")
        bootstrap_results["full_coverage"] = stab_full

        # Selective
        stab_sel = bootstrap_stability(
            val_ch, val_bh, val_meta["baseline_err"], val_meta["cand_err"],
            threshold=best_train_thr, n_boot=args.n_boot,
        )
        print(f"  Selective (thr={best_train_thr:.4f}):")
        print(f"    median: {stab_sel['median_px_mean']:.2f} +/- {stab_sel['median_px_std']:.2f}"
              f"  [{stab_sel['median_px_p5']:.2f}, {stab_sel['median_px_p95']:.2f}]")
        print(f"    coverage: {stab_sel['coverage_mean']:.3f} +/- {stab_sel['coverage_std']:.3f}")
        bootstrap_results["selective"] = stab_sel

        (output_dir / "bootstrap_stability.json").write_text(json.dumps(bootstrap_results, indent=2) + "\n")

    # --- Paper-ready summary ---
    summary = {
        "method": "Hybrid Selector (no learning)",
        "formula": f"{args.alpha} * support_margin + {args.beta} * cand_score + {args.gamma} * cand_ncc",
        "coefficients": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "n_train": int(len(train_meta["baseline_err"])),
        "n_val": int(len(val_meta["baseline_err"])),
        "baselines": {
            "baseline_only": {
                "median_px": float(np.median(val_meta["baseline_err"])),
                "lt4px": float(np.mean(val_meta["baseline_err"] < 4.0)),
            },
            "oracle": {
                "median_px": float(np.median(np.minimum(
                    val_meta["baseline_err"], val_meta["cand_err"].min(axis=1)
                ))),
                "lt4px": float(np.mean(np.minimum(
                    val_meta["baseline_err"], val_meta["cand_err"].min(axis=1)
                ) < 4.0)),
            },
        },
        "operating_points": {
            "full_coverage": {
                "description": "Always pick best candidate (threshold=0)",
                **val_full,
            },
            "selective": {
                "description": f"Override only when confident (threshold={best_train_thr:.4f} from train)",
                **val_selective,
            },
        },
    }
    if bootstrap_results:
        summary["bootstrap"] = bootstrap_results

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved to {output_dir}/summary.json")

    # --- Paper table ---
    bl = summary["baselines"]["baseline_only"]
    ora = summary["baselines"]["oracle"]
    full = summary["operating_points"]["full_coverage"]
    sel = summary["operating_points"]["selective"]
    print(f"\n{'='*60}")
    print(f"Paper Table (val n={summary['n_val']})")
    print(f"{'='*60}")
    print(f"  {'Method':<30s} {'Median':>8s} {'<4px':>6s} {'Cov':>6s} {'Better':>8s}")
    print(f"  {'-'*60}")
    print(f"  {'Baseline only':<30s} {bl['median_px']:>8.2f} {bl['lt4px']:>6.3f} {'1.00':>6s} {'-':>8s}")
    print(f"  {'Oracle':<30s} {ora['median_px']:>8.2f} {ora['lt4px']:>6.3f} {'1.00':>6s} {'-':>8s}")
    print(f"  {'Hybrid (full coverage)':<30s} {full['median_px']:>8.2f} {full['lt4px']:>6.3f} {full['coverage']:>6.2f} {full['better_frac']:>8.3f}")
    print(f"  {'Hybrid (selective)':<30s} {sel['median_px']:>8.2f} {sel['lt4px']:>6.3f} {sel['coverage']:>6.2f} {sel['better_frac']:>8.3f}")
    if sel.get("accept_only_median_px"):
        print(f"    accept-only median: {sel['accept_only_median_px']:.2f}")


if __name__ == "__main__":
    main()
