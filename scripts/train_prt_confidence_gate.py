#!/usr/bin/env python3
"""
Learned confidence gate for PRT hybrid selector.

Instead of a hard threshold on hybrid_score, train a small MLP to predict
whether replacing baseline with the best candidate will improve accuracy.

Input features (per sample):
  - hybrid_score of best candidate (4.0 * support_margin + 0.5 * cand_score)
  - support_margin of best candidate
  - cand_score of best candidate
  - cand_ncc of best candidate
  - occ_length (normalized)
  - camera_motion

Output: P(replacement improves over baseline)

Training: BCE loss on train set, where label = 1 if best candidate error < baseline error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.eval_prt_hybrid_selector import load_cache, encode_patch_batch


def compute_support_features_inline(cache, extractor):
    """Recompute support features inline (same logic as eval_prt_hybrid_selector)."""
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
    pooled = (support_feat * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-8, None)

    baseline_feat = encode_patch_batch(extractor, cache["baseline_patch"]).numpy()
    topk = cache["cand_patches"].shape[1]
    cand_flat = cache["cand_patches"].reshape(n * topk, *cache["cand_patches"].shape[2:])
    cand_feat = encode_patch_batch(extractor, cand_flat).numpy().reshape(n, topk, -1)

    baseline_support_sim = np.sum(baseline_feat * pooled, axis=1).astype(np.float32)
    cand_support_sim = np.sum(cand_feat * pooled[:, None, :], axis=2).astype(np.float32)
    return cand_support_sim, baseline_support_sim


def build_gate_features(
    cache: Dict[str, np.ndarray],
    extractor: DINOFeatureExtractor,
    alpha: float = 4.0,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build gate input features and labels from a cached dataset.

    Returns:
        features: (n, 6) array
            [hybrid_score, support_margin, cand_score, cand_ncc, occ_norm, cam_motion]
        labels: (n,) binary — 1 if best candidate < baseline
    """
    cand_support_sim, baseline_support_sim = compute_support_features_inline(cache, extractor)
    support_margin_all = cand_support_sim - baseline_support_sim[:, None]

    cand_score = cache["cand_score"].astype(np.float32)
    cand_ncc = cache["cand_ncc"].astype(np.float32)
    baseline_err = cache["baseline_err"].astype(np.float32)
    cand_err = cache["cand_err"].astype(np.float32)
    occ_length = cache["occ_length"].astype(np.float32)
    camera_motion = cache["camera_motion"].astype(np.float32)

    hybrid_all = alpha * support_margin_all + beta * cand_score
    best_idx = np.argmax(hybrid_all, axis=1)
    n = len(best_idx)

    # Per-sample features for the chosen candidate
    features = np.zeros((n, 6), dtype=np.float32)
    features[:, 0] = hybrid_all[np.arange(n), best_idx]          # hybrid score
    features[:, 1] = support_margin_all[np.arange(n), best_idx]  # support margin
    features[:, 2] = cand_score[np.arange(n), best_idx]          # cand score
    features[:, 3] = cand_ncc[np.arange(n), best_idx]            # cand ncc
    features[:, 4] = occ_length / 300.0                           # occ_length normalized
    features[:, 5] = camera_motion                                # camera_motion

    # Label: 1 if the chosen candidate is better than baseline
    chosen_err = cand_err[np.arange(n), best_idx]
    labels = (chosen_err < baseline_err).astype(np.float32)

    return features, labels, best_idx


class ConfidenceGate(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_gate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[ConfidenceGate, Dict]:
    model = ConfidenceGate(input_dim=train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_x = torch.from_numpy(train_features).float().to(device)
    train_y = torch.from_numpy(train_labels).float().to(device)
    val_x = torch.from_numpy(val_features).float().to(device)
    val_y = torch.from_numpy(val_labels).float().to(device)

    # Class weight: balance positive/negative
    pos_frac = train_labels.mean()
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 0.01)]).to(device)

    n_train = len(train_features)
    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(n_train)
        losses = []
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            logits = model(train_x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[idx], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_y, pos_weight=pos_weight))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(np.float32)
            val_acc = float((val_preds == val_labels).mean())

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_loss": best_val_loss}


def evaluate_gate(
    model: ConfidenceGate,
    cache: Dict[str, np.ndarray],
    gate_features: np.ndarray,
    best_idx: np.ndarray,
    alpha: float = 4.0,
    beta: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Evaluate gate: for each sample, if gate says 'accept', use best candidate; else baseline."""
    model.eval()
    baseline_err = cache["baseline_err"].astype(np.float32)
    cand_err = cache["cand_err"].astype(np.float32)
    n = len(best_idx)

    with torch.no_grad():
        gate_input = torch.from_numpy(gate_features).float().to(device)
        probs = torch.sigmoid(model(gate_input)).cpu().numpy()

    accept = probs > 0.5
    final_err = baseline_err.copy()
    accepted_errs = []
    for i in range(n):
        if accept[i]:
            final_err[i] = cand_err[i, best_idx[i]]
            accepted_errs.append(cand_err[i, best_idx[i]])

    accepted_errs = np.asarray(accepted_errs, dtype=np.float32) if accepted_errs else np.array([], dtype=np.float32)

    return {
        "accept_rate": float(accept.mean()),
        "median_px": float(np.median(final_err)),
        "lt4px": float(np.mean(final_err < 4.0)),
        "better_frac": float(np.mean(final_err < baseline_err)),
        "accept_only_median_px": float(np.median(accepted_errs)) if accepted_errs.size > 0 else None,
        "n_accept": int(accept.sum()),
        "n_total": int(n),
    }

    with torch.no_grad():
        gate_input = torch.from_numpy(gate_features).float().to(device)
        probs = torch.sigmoid(model(gate_input)).cpu().numpy()

    accept = probs > 0.5
    final_err = baseline_err.copy()
    accepted_errs = []
    for i in range(n):
        if accept[i]:
            final_err[i] = cand_err[i, best_idx[i]]
            accepted_errs.append(cand_err[i, best_idx[i]])

    accepted_errs = np.asarray(accepted_errs, dtype=np.float32) if accepted_errs else np.array([], dtype=np.float32)

    return {
        "accept_rate": float(accept.mean()),
        "median_px": float(np.median(final_err)),
        "lt4px": float(np.mean(final_err < 4.0)),
        "better_frac": float(np.mean(final_err < baseline_err)),
        "accept_only_median_px": float(np.median(accepted_errs)) if accepted_errs.size > 0 else None,
        "n_accept": int(accept.sum()),
        "n_total": int(n),
    }


def sweep_gate_threshold(
    model: ConfidenceGate,
    cache: Dict[str, np.ndarray],
    gate_features: np.ndarray,
    best_idx: np.ndarray,
    alpha: float = 4.0,
    beta: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> list:
    """Sweep gate probability threshold."""
    model.eval()
    baseline_err = cache["baseline_err"].astype(np.float32)
    cand_err = cache["cand_err"].astype(np.float32)
    n = len(best_idx)

    with torch.no_grad():
        gate_input = torch.from_numpy(gate_features).float().to(device)
        probs = torch.sigmoid(model(gate_input)).cpu().numpy()

    results = []
    for thr in np.linspace(0.0, 1.0, 21):
        accept = probs > thr
        final_err = baseline_err.copy()
        for i in range(n):
            if accept[i]:
                final_err[i] = cand_err[i, best_idx[i]]
        results.append({
            "gate_threshold": float(thr),
            "accept_rate": float(accept.mean()),
            "median_px": float(np.median(final_err)),
            "lt4px": float(np.mean(final_err < 4.0)),
            "better_frac": float(np.mean(final_err < baseline_err)),
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train confidence gate for PRT hybrid selector")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str,
                        default="/gemini/code/FSPT/outputs/prt_confidence_gate")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    train_cache = load_cache(Path(args.train_cache))
    val_cache = load_cache(Path(args.val_cache))

    print("Building train gate features...", flush=True)
    train_feat, train_label, train_best_idx = build_gate_features(train_cache, extractor, args.alpha, args.beta)
    print(f"  n={len(train_label)}, positive_rate={train_label.mean():.3f}", flush=True)

    print("Building val gate features...", flush=True)
    val_feat, val_label, val_best_idx = build_gate_features(val_cache, extractor, args.alpha, args.beta)
    print(f"  n={len(val_label)}, positive_rate={val_label.mean():.3f}", flush=True)

    # Save features for later analysis
    np.savez(output_dir / "gate_features.npz",
             train_feat=train_feat, train_label=train_label, train_best_idx=train_best_idx,
             val_feat=val_feat, val_label=val_label, val_best_idx=val_best_idx)

    # Train
    print("\nTraining confidence gate...", flush=True)
    model, train_info = train_gate(
        train_feat, train_label, val_feat, val_label,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    torch.save(model.state_dict(), output_dir / "gate_model.pt")
    (output_dir / "train_history.json").write_text(json.dumps(train_info["history"], indent=2) + "\n")

    # Evaluate at default threshold (0.5)
    print("\n=== Gate evaluation (threshold=0.5) ===")
    val_result = evaluate_gate(model, val_cache, val_feat, val_best_idx, args.alpha, args.beta, device)
    print(f"  Accept rate: {val_result['accept_rate']:.3f}")
    print(f"  Median px:   {val_result['median_px']:.2f}")
    print(f"  <4px:        {val_result['lt4px']:.3f}")
    print(f"  Better frac: {val_result['better_frac']:.3f}")

    # Baselines for comparison
    baseline_median = float(np.median(val_cache["baseline_err"].astype(np.float32)))
    oracle_err = np.minimum(
        val_cache["baseline_err"].astype(np.float32),
        val_cache["cand_err"].astype(np.float32).min(axis=1),
    )
    oracle_median = float(np.median(oracle_err))

    print(f"\n  Baseline median: {baseline_median:.2f}")
    print(f"  Oracle median:   {oracle_median:.2f}")
    print(f"  Hybrid hard-thr: 32.97 (from previous eval)")

    # Threshold sweep
    if args.sweep:
        print("\n=== Gate threshold sweep (val) ===")
        sweep = sweep_gate_threshold(model, val_cache, val_feat, val_best_idx, args.alpha, args.beta, device)
        print(f"  {'thr':>6} {'accept':>8} {'median':>8} {'lt4px':>6} {'better':>8}")
        for row in sweep:
            print(f"  {row['gate_threshold']:>6.2f} {row['accept_rate']:>8.3f} {row['median_px']:>8.2f}"
                  f" {row['lt4px']:>6.3f} {row['better_frac']:>8.3f}")
        (output_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")

        # Find best threshold
        best = min(sweep, key=lambda x: x["median_px"])
        print(f"\n  Best threshold: {best['gate_threshold']:.2f} → median={best['median_px']:.2f}")

    # Summary
    summary = {
        "method": "Learned Confidence Gate",
        "formula": f"hybrid = {args.alpha} * support_margin + {args.beta} * cand_score",
        "gate": "MLP(6→32→32→1), sigmoid output",
        "gate_features": ["hybrid_score", "support_margin", "cand_score", "cand_ncc", "occ_length_norm", "camera_motion"],
        "n_train": len(train_label),
        "n_val": len(val_label),
        "train_positive_rate": float(train_label.mean()),
        "val_positive_rate": float(val_label.mean()),
        "results": {
            "baseline": {"median_px": baseline_median, "lt4px": float(np.mean(val_cache["baseline_err"] < 4.0))},
            "oracle": {"median_px": oracle_median, "lt4px": float(np.mean(oracle_err < 4.0))},
            "gate_default": val_result,
        },
        "train_info": {"best_val_loss": train_info["best_val_loss"]},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved to {output_dir}/summary.json")


if __name__ == "__main__":
    main()
