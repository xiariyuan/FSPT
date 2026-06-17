#!/usr/bin/env python3
"""
Train and evaluate local selector for base-centered DINO candidates.

Loads dataset from outputs/local_selector_dataset/samples.jsonl,
trains an MLP selector (K+1 classification), evaluates with sequence-grouped split.

Usage:
  python scripts/train_local_selector.py \
    --dataset-dir outputs/local_selector_dataset \
    --output-dir outputs/local_selector_stage2 \
    --epochs 50 --lr 1e-3
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_event_features(sample, K):
    """Extract event-level features (not candidate-dependent)."""
    return [
        sample.get("occ_length", 0) / 300.0,  # normalized
        sample.get("top1_score", 0),
        sample.get("top1_margin", 0),
        sample.get("top1_shift_px", 0) / 256.0,  # normalized
        sample.get("score_std_topk", 0),
        sample.get("score_range_topk", 0),
        sample.get("shift_std_topk", 0) / 256.0,
        float(sample.get("num_candidates", K)) / K,
    ]


def extract_candidate_features(sample, cand_idx, K):
    """Extract per-candidate features."""
    cands = sample.get("candidates", [])
    if cand_idx >= len(cands):
        return [0.0] * 8
    c = cands[cand_idx]
    return [
        c.get("cand_score", 0),
        c.get("cand_margin_to_top2", 0),
        c.get("cand_shift_px_from_base", 0) / 256.0,
        c.get("cand_dist_to_base_px", 0) / 256.0,
        c.get("rank_by_dino", 0) / max(1, K - 1),
        c.get("cand_dist_to_top1_px", 0) / 256.0,
        c.get("score_rank_normalized", 0),
        c.get("shift_rank_normalized", 0),
    ]


def build_dataset_arrays(samples, K):
    """Build X (features) and y (labels) arrays."""
    event_feat_dim = 8
    cand_feat_dim = 8
    total_feat_dim = event_feat_dim + K * (cand_feat_dim + 1)  # +1 for per-candidate flag

    X_list = []
    y_list = []

    for s in samples:
        event_f = extract_event_features(s, K)
        cand_fs = []
        for j in range(K):
            cf = extract_candidate_features(s, j, K)
            cand_fs.extend(cf)
            cand_fs.append(1.0 if j < s.get("num_candidates", K) else 0.0)  # valid flag

        x = event_f + cand_fs
        X_list.append(x)
        y_list.append(s["label"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Sequence-grouped split
# ---------------------------------------------------------------------------
def sequence_grouped_split(samples, val_frac=0.4, seed=42):
    """Split by video_name, keeping sequences together."""
    rng = np.random.RandomState(seed)
    vids = sorted(set(s["video_name"] for s in samples))
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac))
    val_vids = set(vids[:n_val])
    train_vids = set(vids[n_val:])

    train_idx = [i for i, s in enumerate(samples) if s["video_name"] in train_vids]
    val_idx = [i for i, s in enumerate(samples) if s["video_name"] in val_vids]
    return train_idx, val_idx, train_vids, val_vids


# ---------------------------------------------------------------------------
# MLP Selector
# ---------------------------------------------------------------------------
class MLPSelector(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X, y, samples, K, device):
    """Evaluate selector on a dataset."""
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        logits = model(Xt)
        preds = logits.argmax(dim=1).cpu().numpy()

    n = len(y)
    base_errs = np.array([s["base_error_px"] for s in samples])
    top1_errs = np.array([s["top1_error_px"] for s in samples])
    oracle_errs = np.array([s["oracle_error_px"] for s in samples])

    # Selector error: for each sample, the error of the selected candidate
    selector_errs = []
    for i in range(n):
        pred = int(preds[i])
        if pred == K:  # abstain → use base
            selector_errs.append(base_errs[i])
        else:
            cands = samples[i].get("candidates", [])
            if pred < len(cands):
                selector_errs.append(cands[pred]["cand_error_px"])
            else:
                selector_errs.append(base_errs[i])
    selector_errs = np.array(selector_errs)

    # Better by 2px
    sel_better_2px = (selector_errs < base_errs - 2).sum()
    top1_better_2px = (top1_errs < base_errs - 2).sum()
    oracle_better_2px = (oracle_errs < base_errs - 2).sum()

    abstain_rate = (preds == K).mean()

    results = {
        "n": n,
        "base_median": round(float(np.median(base_errs)), 2),
        "top1_median": round(float(np.median(top1_errs)), 2),
        "selector_median": round(float(np.median(selector_errs)), 2),
        "oracle_median": round(float(np.median(oracle_errs)), 2),
        "top1_better_2px_frac": round(float(top1_better_2px / n), 3),
        "selector_better_2px_frac": round(float(sel_better_2px / n), 3),
        "oracle_better_2px_frac": round(float(oracle_better_2px / n), 3),
        "abstain_rate": round(float(abstain_rate), 3),
        "selector_top1_accuracy": round(float((preds == y).mean()), 3),
    }

    # Per-group
    for group_name, group_fn in [
        ("base16", lambda s: s["group_base16"]),
        ("base32", lambda s: s["group_base32"]),
    ]:
        g_idx = [i for i in range(n) if group_fn(samples[i])]
        if not g_idx:
            results[group_name] = {"n": 0}
            continue

        g_base = base_errs[g_idx]
        g_sel = selector_errs[g_idx]
        g_oracle = oracle_errs[g_idx]
        g_top1 = top1_errs[g_idx]
        g_preds = preds[g_idx]
        g_y = y[g_idx]

        results[group_name] = {
            "n": len(g_idx),
            "base_median": round(float(np.median(g_base)), 2),
            "top1_median": round(float(np.median(g_top1)), 2),
            "selector_median": round(float(np.median(g_sel)), 2),
            "oracle_median": round(float(np.median(g_oracle)), 2),
            "selector_better_2px_frac": round(float((g_sel < g_base - 2).mean()), 3),
            "oracle_better_2px_frac": round(float((g_oracle < g_base - 2).mean()), 3),
            "abstain_rate": round(float((g_preds == K).mean()), 3),
        }

    return results, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds_path = Path(args.dataset_dir) / "samples.jsonl"
    samples = [json.loads(l) for l in open(ds_path)]
    summary = json.load(open(Path(args.dataset_dir) / "summary.json"))
    K = summary["topk"]

    print(f"Loaded {len(samples)} samples, K={K}")

    # Build features
    X, y = build_dataset_arrays(samples, K)
    print(f"Feature dim: {X.shape[1]}, classes: {K+1}")

    # Replace NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # Sequence-grouped split
    train_idx, val_idx, train_vids, val_vids = sequence_grouped_split(samples, val_frac=args.val_frac, seed=args.seed)
    print(f"Train: {len(train_idx)} samples ({len(train_vids)} seqs), Val: {len(val_idx)} ({len(val_vids)} seqs)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    val_samples = [samples[i] for i in val_idx]

    # Class weights (inverse frequency)
    class_counts = np.bincount(y_train, minlength=K + 1).astype(np.float32)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * (K + 1)
    class_weights_t = torch.from_numpy(class_weights).to(device)

    print(f"Class distribution (train): {dict(enumerate(class_counts.astype(int).tolist()))}")

    # Normalize features
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma

    # Train MLP
    model = MLPSelector(X.shape[1], K + 1, hidden=args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    Xt = torch.from_numpy(X_train_n).to(device)
    yt = torch.from_numpy(y_train).to(device)

    best_val_metric = -1
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_results, _ = evaluate(model, X_val_n, y[val_idx], val_samples, K, device)
            metric = val_results.get("selector_better_2px_frac", 0)
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, val_sel_med={val_results['selector_median']:.1f}, "
                  f"val_sel_better2px={val_results['selector_better_2px_frac']:.3f}, abstain={val_results['abstain_rate']:.3f}")

            if metric > best_val_metric:
                best_val_metric = metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final evaluation
    val_results, val_preds = evaluate(model, X_val_n, y[val_idx], val_samples, K, device)

    # Oracle gap utilization
    oracle_gain = val_results["base_median"] - val_results["oracle_median"]
    selector_gain = val_results["base_median"] - val_results["selector_median"]
    utilization = selector_gain / oracle_gain if oracle_gain > 0 else 0

    output = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "model": "MLPSelector",
        "input_dim": X.shape[1],
        "hidden": args.hidden,
        "K": K,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "train_vids": sorted(list(train_vids)),
        "val_vids": sorted(list(val_vids)),
        "train_n": len(train_idx),
        "val_n": len(val_idx),
        "class_distribution_train": {str(k): int(v) for k, v in enumerate(class_counts)},
        "results": val_results,
        "oracle_gap_utilization": round(utilization, 3),
    }

    # Stage-2 criteria
    b16 = val_results.get("base16", {})
    criteria = []
    passed = 0

    # Criterion 1: base>16 selector better_2px >= 0.50
    c1 = b16.get("selector_better_2px_frac", 0) >= 0.50
    passed += int(c1)
    criteria.append(f"base>16 selector better_2px={b16.get('selector_better_2px_frac', 0):.3f} >= 0.50: {'PASS' if c1 else 'FAIL'}")

    # Criterion 2: base>16 selector median improvement >= 10%
    base_med_b16 = b16.get("base_median", 0)
    sel_med_b16 = b16.get("selector_median", 999)
    if base_med_b16 > 0:
        imp = (base_med_b16 - sel_med_b16) / base_med_b16
        c2 = imp >= 0.10
    else:
        imp = 0
        c2 = False
    passed += int(c2)
    criteria.append(f"base>16 selector median improvement={imp:.1%} >= 10%: {'PASS' if c2 else 'FAIL'}")

    # Criterion 3: overall selector median not worse by > 1px
    overall_diff = val_results["selector_median"] - val_results["base_median"]
    c3 = overall_diff <= 1.0
    passed += int(c3)
    criteria.append(f"overall selector median diff={overall_diff:.1f}px <= 1px: {'PASS' if c3 else 'FAIL'}")

    stage2_pass = passed >= 2
    output["stage2_verdict"] = "PASS" if stage2_pass else "FAIL"
    output["stage2_criteria_passed"] = passed
    output["stage2_criteria_details"] = criteria

    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"Local Selector Evaluation")
    print(f"{'='*60}")
    for key in ["base", "top1", "selector", "oracle"]:
        med = val_results.get(f"{key}_median", "?")
        b2 = val_results.get(f"{key}_better_2px_frac", "?")
        print(f"  {key:12s}: median={med}, better_2px={b2}")
    print(f"  abstain_rate: {val_results['abstain_rate']:.3f}")
    print(f"  oracle_gap_utilization: {utilization:.3f}")
    for g in ["base16", "base32"]:
        gdata = val_results.get(g, {})
        if gdata.get("n", 0) > 0:
            print(f"  {g} (n={gdata['n']}): base_med={gdata['base_median']}, sel_med={gdata['selector_median']}, oracle_med={gdata['oracle_median']}, sel_better2px={gdata['selector_better_2px_frac']:.3f}")
    print(f"\n  Stage-2: {output['stage2_verdict']} ({passed} criteria passed)")
    for d in criteria:
        print(f"    {d}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
