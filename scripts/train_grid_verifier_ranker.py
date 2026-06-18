#!/usr/bin/env python3
"""Train a lightweight ranker for CT-offline-centered local grid verifier.

Input: 128 samples, each with ~81 candidates.
Each candidate has 4 features: [cos_support, cos_ct, l2_support, l2_ct].
Label: 1 = oracle best candidate (closest to GT), 0 = others.

Model: MLP with 1 hidden layer (4→32→1).
Loss: listwise softmax cross-entropy (one correct per set).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("/gemini/code/FSPT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GridVerifierRanker(nn.Module):
    """Lightweight MLP ranker for local grid candidates."""
    def __init__(self, input_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
    """Listwise softmax cross-entropy: one positive per list."""
    loss = torch.tensor(0.0, device=scores.device)
    n = 0
    for sid in torch.unique(sample_ids):
        mask = sample_ids == sid
        s = scores[mask]
        l = labels[mask]
        pos_idx = (l == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue
        loss = loss - F.log_softmax(s, dim=0)[pos_idx[0]]
        n += 1
    return loss / max(n, 1)


def compute_metrics(scores_list: List[float], labels_list: List[float],
                     cand_errors_list: List[List[float]],
                     base_errors_list: List[float]) -> Dict[str, Any]:
    """Compute verifier metrics.

    For each sample, pick the candidate with highest score.
    Compare its error to the raw CT-offline error.
    """
    n = len(scores_list)
    if n == 0:
        return {"n": 0}

    verifier_errors = []
    raw_errors = []
    oracle_errors = []
    long_occ_verifier = []
    long_occ_raw = []
    long_occ_oracle = []
    better_count = 0

    for i in range(n):
        scores = np.array(scores_list[i])
        labels = np.array(labels_list[i])
        cand_errors = np.array(cand_errors_list[i])
        raw_err = base_errors_list[i]

        # Pick candidate with highest score
        best_idx = int(np.argmax(scores))
        verifier_err = float(cand_errors[best_idx])
        oracle_err = float(cand_errors[int(np.argmin(cand_errors))])

        verifier_errors.append(verifier_err)
        raw_errors.append(raw_err)
        oracle_errors.append(oracle_err)

        if verifier_err < raw_err:
            better_count += 1

    v = np.array(verifier_errors)
    r = np.array(raw_errors)
    o = np.array(oracle_errors)

    # No long-occ filtering in this simple version
    return {
        "n": n,
        "verifier_median_px": round(float(np.median(v)), 2),
        "verifier_lt4px": round(float(np.mean(v < 4)), 4),
        "verifier_lt8px": round(float(np.mean(v < 8)), 4),
        "raw_median_px": round(float(np.median(r)), 2),
        "raw_lt4px": round(float(np.mean(r < 4)), 4),
        "raw_lt8px": round(float(np.mean(r < 8)), 4),
        "oracle_median_px": round(float(np.median(o)), 2),
        "oracle_lt4px": round(float(np.mean(o < 4)), 4),
        "better_frac": round(float(better_count / n), 3),
        "lt4px_improvement_vs_raw_pp": round(float(np.mean(v < 4) - np.mean(r < 4)), 4),
        "lt8px_no_degradation": round(float(np.mean(v < 8) >= np.mean(r < 8)), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-video", type=str, default="bmx-trees")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(args.dataset_json) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Loaded {len(samples)} samples")

    # Split by video
    train_samples = [s for s in samples if s["video_name"] != args.val_video]
    val_samples = [s for s in samples if s["video_name"] == args.val_video]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Build tensors
    def make_tensors(sample_list):
        feats, labs, sids = [], [], []
        for si, s in enumerate(sample_list):
            cf = np.array(s["cand_features"])
            lb = np.array(s["labels"])
            feats.append(torch.from_numpy(cf).float())
            labs.append(torch.from_numpy(lb).float())
            sids.append(torch.full((len(cf),), si, dtype=torch.long))
        return (
            torch.cat(feats, dim=0),
            torch.cat(labs, dim=0),
            torch.cat(sids, dim=0),
        )

    X_train, y_train, sid_train = make_tensors(train_samples)
    X_val, y_val, sid_val = make_tensors(val_samples)
    print(f"Train: {X_train.shape[0]} candidates, Val: {X_val.shape[0]} candidates")

    # Model
    model = GridVerifierRanker(input_dim=X_train.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    history = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        opt = optimizer
        opt.zero_grad()
        out = model(X_train.to(device))
        loss = listwise_loss(out, y_train.to(device), sid_train.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = listwise_loss(val_out, y_val.to(device), sid_val.to(device))

        # Metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Compute full metrics on val set
            val_scores_list = []
            val_labels_list = []
            val_errors_list = []
            val_base_list = []
            for si, s in enumerate(val_samples):
                mask = sid_val == si
                sco = val_out[mask].detach().cpu().numpy()
                lab = y_val[mask].numpy()
                val_scores_list.append(sco.tolist())
                val_labels_list.append(lab.tolist())
                val_errors_list.append(s["cand_errors_px"])
                val_base_list.append(s["raw_ct_error_px"])

            val_metrics = compute_metrics(val_scores_list, val_labels_list,
                                           val_errors_list, val_base_list)

            print(f"Epoch {epoch+1:>3d}/{args.epochs}  "
                  f"train_loss={float(loss):.4f}  val_loss={float(val_loss):.4f}  "
                  f"v_med={val_metrics['verifier_median_px']:.2f}  "
                  f"v_lt4={val_metrics['verifier_lt4px']*100:.1f}%  "
                  f"better={val_metrics['better_frac']:.3f}")

        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            torch.save(model.state_dict(), out_dir / "best.pth")

    # Final evaluation
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_out = model(X_val.to(device))

    val_scores_list, val_labels_list, val_errors_list, val_base_list = [], [], [], []
    for si, s in enumerate(val_samples):
        mask = sid_val == si
        sco = val_out[mask].detach().cpu().numpy()
        lab = y_val[mask].numpy()
        val_scores_list.append(sco.tolist())
        val_labels_list.append(lab.tolist())
        val_errors_list.append(s["cand_errors_px"])
        val_base_list.append(s["raw_ct_error_px"])

    val_metrics = compute_metrics(val_scores_list, val_labels_list,
                                   val_errors_list, val_base_list)

    # Also compute train metrics
    train_out = model(X_train.to(device))
    train_scores_list, train_labels_list, train_errors_list, train_base_list = [], [], [], []
    for si, s in enumerate(train_samples):
        mask = sid_train == si
        sco = train_out[mask].detach().cpu().numpy()
        lab = y_train[mask].numpy()
        train_scores_list.append(sco.tolist())
        train_labels_list.append(lab.tolist())
        train_errors_list.append(s["cand_errors_px"])
        train_base_list.append(s["raw_ct_error_px"])
    train_metrics = compute_metrics(train_scores_list, train_labels_list,
                                     train_errors_list, train_base_list)

    results = {
        "config": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"  Train (n={train_metrics['n']}):")
    print(f"    Verifier median: {train_metrics['verifier_median_px']:.1f}px  (raw: {train_metrics['raw_median_px']:.1f}px)")
    print(f"    Verifier <4px: {train_metrics['verifier_lt4px']*100:.1f}%  (raw: {train_metrics['raw_lt4px']*100:.1f}%)")
    print(f"    Better frac: {train_metrics['better_frac']:.3f}")
    print(f"  Val (n={val_metrics['n']}):")
    print(f"    Verifier median: {val_metrics['verifier_median_px']:.1f}px  (raw: {val_metrics['raw_median_px']:.1f}px)")
    print(f"    Verifier <4px: {val_metrics['verifier_lt4px']*100:.1f}%  (raw: {val_metrics['raw_lt4px']*100:.1f}%)")
    print(f"    Better frac: {val_metrics['better_frac']:.3f}")
    print(f"    Oracle <4px: {val_metrics['oracle_lt4px']*100:.1f}%")
    print(f"    <4px improvement vs raw: {val_metrics['lt4px_improvement_vs_raw_pp']*100:+.1f}pp")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
