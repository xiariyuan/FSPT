#!/usr/bin/env python3
"""
Formal subgroup significance analysis for PRT hybrid selector.

Reads the 15-seq val cache and evaluates the hybrid selector on
predefined subgroups: by reentry type, by occlusion length, and
an effective zone (100-500 frames).

For each subgroup:
  - N, baseline median, hybrid median, delta
  - paired permutation p-value (mean diff)
  - bootstrap 90% CI for mean diff

Usage:
  python scripts/eval_prt_hybrid_selector_significance.py \
    --val-cache outputs/prt_val_15seq_1000each_stratified/dataset_cache.npz \
    --train-cache outputs/prt_candidate_train_v2/dataset_cache.npz \
    --output-dir outputs/prt_hybrid_selector_15seq_stratified_v3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.eval_prt_hybrid_selector import encode_patch_batch


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    c = np.load(path, allow_pickle=False)
    return {k: c[k] for k in c.files}


def compute_support_margin(cache: Dict, extractor, device) -> np.ndarray:
    """Compute per-candidate support_margin for a cache."""
    n = cache['query_patch'].shape[0]
    topk = cache['cand_patches'].shape[1]
    support_count = cache['support_patches'].shape[1]

    support_flat = cache['support_patches'].reshape(n * support_count, *cache['support_patches'].shape[2:])
    support_feat = encode_patch_batch(extractor, support_flat).numpy().reshape(n, support_count, -1)
    support_mask = (np.arange(support_count, dtype=np.int32)[None, :] < cache['support_count'][:, None]).astype(np.float32)
    support_mask_sum = np.clip(support_mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled = (support_feat * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-8, None)

    baseline_feat = encode_patch_batch(extractor, cache['baseline_patch']).numpy()
    cand_flat = cache['cand_patches'].reshape(n * topk, *cache['cand_patches'].shape[2:])
    cand_feat = encode_patch_batch(extractor, cand_flat).numpy().reshape(n, topk, -1)

    baseline_support_sim = np.sum(baseline_feat * pooled, axis=1).astype(np.float32)
    cand_support_sim = np.sum(cand_feat * pooled[:, None, :], axis=2).astype(np.float32)
    return cand_support_sim - baseline_support_sim[:, None]


def paired_permutation_test(diff: np.ndarray, n_perm: int = 10000, seed: int = 42) -> Dict:
    """Two-sided paired permutation test on mean diff."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(diff))
    perms = np.array([float(np.mean(diff * rng.choice([-1, 1], size=len(diff)))) for _ in range(n_perm)])
    p = float(np.mean(np.abs(perms) >= np.abs(obs)))
    return {"mean_diff": round(obs, 2), "p_value": round(p, 4), "test": "paired_permutation_mean", "n_perm": n_perm}


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Dict:
    """Bootstrap 90% CI for mean diff."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([float(np.mean(diff[rng.choice(n, n, replace=True)])) for _ in range(n_boot)])
    return {
        "mean": round(float(np.mean(means)), 2),
        "ci_90": [round(float(np.percentile(means, 5)), 2), round(float(np.percentile(means, 95)), 2)],
    }


def subgroup_analysis(
    mask: np.ndarray,
    label: str,
    baseline_err: np.ndarray,
    hybrid_err: np.ndarray,
    oracle_err: np.ndarray,
    seq_names: np.ndarray = None,
) -> Dict:
    """Analyze one subgroup."""
    m = mask
    n = int(m.sum())
    if n == 0:
        return {"group": label, "n": 0}

    be = baseline_err[m]
    he = hybrid_err[m]
    oe = oracle_err[m]
    diff = he - be  # negative = hybrid better

    result = {
        "group": label,
        "n": n,
        "baseline_median": round(float(np.median(be)), 2),
        "hybrid_median": round(float(np.median(he)), 2),
        "oracle_median": round(float(np.median(oe)), 2),
        "delta_median": round(float(np.median(be) - np.median(he)), 2),
        "mean_diff": round(float(np.mean(diff)), 2),
        "baseline_lt4px": round(float(np.mean(be < 4)), 4),
        "hybrid_lt4px": round(float(np.mean(he < 4)), 4),
        "better_frac": round(float(np.mean(he < be)), 3),
        "significance_sample_level": paired_permutation_test(diff),
        "bootstrap_mean_diff": bootstrap_ci(diff),
    }

    # Sequence-grouped bootstrap if seq_names provided
    if seq_names is not None:
        sn = seq_names[m]
        result["significance_seq_grouped"] = seq_grouped_bootstrap(diff, sn)
        result["significance_seq_permutation"] = seq_grouped_permutation(diff, sn)

    return result


def seq_grouped_bootstrap(diff: np.ndarray, seq_names: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Dict:
    """Equal-weight sequence-grouped bootstrap: resample sequences, take mean of per-sequence means."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seq_names)
    n_seq = len(unique)
    # Equal-weight: compute per-sequence mean diff first
    seq_means = np.array([float(np.mean(diff[seq_names == s])) for s in unique])
    boot_means = []
    for _ in range(n_boot):
        chosen = rng.choice(n_seq, size=n_seq, replace=True)
        boot_means.append(float(np.mean(seq_means[chosen])))
    ci = [round(float(np.percentile(boot_means, 5)), 2), round(float(np.percentile(boot_means, 95)), 2)]
    return {"mean": round(float(np.mean(boot_means)), 2), "ci_90": ci, "n_seq": int(n_seq)}


def seq_grouped_permutation(diff: np.ndarray, seq_names: np.ndarray, n_perm: int = 10000, seed: int = 42) -> Dict:
    """Equal-weight sequence-level permutation: flip sign per-sequence on per-sequence means."""
    rng = np.random.default_rng(seed)
    unique = np.unique(seq_names)
    n_seq = len(unique)
    seq_means = np.array([float(np.mean(diff[seq_names == s])) for s in unique])
    obs = float(np.mean(seq_means))  # equal-weight mean of per-sequence means
    perms = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n_seq)
        perms.append(float(np.mean(seq_means * signs)))
    p = float(np.mean(np.abs(np.array(perms)) >= np.abs(obs)))
    return {"seq_mean_diff": round(obs, 2), "p_value": round(p, 4), "n_seq": n_seq}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Hybrid score threshold for accepting candidate (0.0 = coverage~79%, -1e9 = forced 100%%)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load caches
    val = load_cache(Path(args.val_cache))
    train = load_cache(Path(args.train_cache))

    # Compute support_margin on train (for coefficient selection verification)
    print("Computing train support_margin...", flush=True)
    train_sm = compute_support_margin(train, extractor, device)
    train_cand_score = train['cand_score'].astype(np.float32)
    train_baseline_err = train['baseline_err'].astype(np.float32)
    train_cand_err = train['cand_err'].astype(np.float32)

    alpha = 10.0
    print(f"Using fixed alpha={alpha} (matching formal v2/v3 results)")

    # Compute on val
    print("Computing val support_margin...", flush=True)
    val_sm = compute_support_margin(val, extractor, device)
    val_cand_score = val['cand_score'].astype(np.float32)
    val_baseline_err = val['baseline_err'].astype(np.float32)
    val_cand_err = val['cand_err'].astype(np.float32)
    val_oracle_err = np.minimum(val_baseline_err, val_cand_err.min(axis=1))
    val_occ = val['occ_length'].astype(np.float32)
    val_type = val['reentry_type_id'].astype(int)

    n = len(val_baseline_err)
    hybrid = alpha * val_sm
    best_idx = np.argmax(hybrid, axis=1)
    best_hybrid_score = hybrid[np.arange(n), best_idx]
    cand_hybrid_err = np.array([val_cand_err[i, best_idx[i]] for i in range(n)])

    # Apply operating point: threshold on hybrid_score
    if args.threshold is None:
        threshold = 0.0
    else:
        threshold = args.threshold
    accept = best_hybrid_score > threshold
    hybrid_err = val_baseline_err.copy()
    for i in range(n):
        if accept[i]:
            hybrid_err[i] = cand_hybrid_err[i]
    coverage = float(accept.mean())
    print(f"Operating point: threshold={threshold}, coverage={coverage:.3f}")

    # Load sequence names for grouped tests
    meta_path = Path(args.val_cache).parent / "samples_meta.jsonl"
    if meta_path.exists():
        seq_names = np.array([json.loads(l).get("video_name", json.loads(l).get("seq_name", "unknown")) for l in open(meta_path) if l.strip()])
    else:
        seq_names = None
        print("WARNING: samples_meta.jsonl not found, skipping sequence-grouped tests")

    # Overall
    overall = subgroup_analysis(
        np.ones(n, dtype=bool), "overall",
        val_baseline_err, hybrid_err, val_oracle_err, seq_names,
    )

    # By reentry type
    by_type = [
        subgroup_analysis(val_type == 0, "in_frame_occlusion", val_baseline_err, hybrid_err, val_oracle_err, seq_names),
        subgroup_analysis(val_type == 1, "off_screen_return", val_baseline_err, hybrid_err, val_oracle_err, seq_names),
    ]

    # By occlusion length
    by_occ = []
    for lo, hi, label in [(20, 100, "20-100"), (100, 200, "100-200"), (200, 500, "200-500"), (500, 10000, "500+")]:
        by_occ.append(subgroup_analysis(
            (val_occ >= lo) & (val_occ < hi), label, val_baseline_err, hybrid_err, val_oracle_err, seq_names,
        ))

    # Effective zone: in-frame, 100-500
    eff_mask = (val_type == 0) & (val_occ >= 100) & (val_occ < 500)
    effective = subgroup_analysis(eff_mask, "in_frame_100_500", val_baseline_err, hybrid_err, val_oracle_err, seq_names)

    # Test config
    test_config = {
        "test_type": "paired_permutation_mean",
        "n_permutations": 10000,
        "bootstrap_n_boot": 2000,
        "bootstrap_ci_level": "90%",
        "metric": "pixel_error",
        "diff_direction": "negative_is_better",
        "multiple_comparison_correction": "none",
    }

    result = {
        "n_val": int(n),
        "formula": f"{alpha} * support_margin",
        "operating_point": {
            "threshold": threshold,
            "coverage": round(coverage, 3),
        },
        "test_config": test_config,
        "overall": overall,
        "by_reentry_type": by_type,
        "by_occ_length": by_occ,
        "effective_zone": effective,
    }

    (output_dir / "significance.json").write_text(json.dumps(result, indent=2) + "\n")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Subgroup Significance (n={n}, formula={alpha}*support_margin, thr={threshold}, cov={coverage:.3f})")
    print(f"{'='*80}")
    has_seq = seq_names is not None
    hdr = f"  {'Group':<25s} {'N':>5s} {'Base':>8s} {'Hybrid':>8s} {'Delta':>7s} {'p_samp':>8s}"
    if has_seq:
        hdr += f" {'p_seq':>7s} {'seq_CI90':>20s}"
    print(hdr)
    for r in [overall] + by_type + by_occ + [effective]:
        if r['n'] == 0:
            continue
        ci = r['bootstrap_mean_diff']['ci_90']
        p_s = r['significance_sample_level']['p_value']
        sig_s = '*' if p_s < 0.05 else ''
        line = f"  {r['group']:<23s} {r['n']:>5d} {r['baseline_median']:>8.2f} {r['hybrid_median']:>8.2f} {r['delta_median']:>+7.2f} {p_s:>7.4f}{sig_s}"
        if has_seq and 'significance_seq_permutation' in r:
            p_q = r['significance_seq_permutation']['p_value']
            sci = r['significance_seq_grouped']['ci_90']
            sig_q = '*' if p_q < 0.05 else ''
            line += f" {p_q:>6.4f}{sig_q} [{sci[0]:+.2f}, {sci[1]:+.2f}]"
        print(line)

    print(f"\nNote: delta_median = baseline - hybrid (positive = hybrid better)")
    print(f"      mean_diff = hybrid - baseline (negative = hybrid better)")
    print(f"      p_samp = sample-level paired permutation on mean diff")
    if has_seq:
        print(f"      p_seq = sequence-level permutation on per-sequence mean diff")

    print(f"\nSaved to {output_dir / 'significance.json'}")


if __name__ == "__main__":
    main()
