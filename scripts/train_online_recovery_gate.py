#!/usr/bin/env python3
"""
Train M1 online recovery gate.

Freezes CoTracker and DINO, trains only OnlineRecoveryGateHead.

Usage:
  python scripts/train_online_recovery_gate.py \
    --dataset-dir outputs/online_recovery_gate_dataset_v1 \
    --output-dir outputs/online_recovery_gate_m1_seed42 \
    --checkpoint-dir checkpoints/online_recovery_gate_m1_seed42 \
    --epochs 60 --seed 42
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.online_recovery_head import OnlineRecoveryGateHead


class GateDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str):
        self.dataset_dir = Path(dataset_dir)
        with open(self.dataset_dir / f"index_{split}.json") as f:
            self.samples = json.load(f)
        # Filter to samples with non-ambiguous labels
        self.valid_indices = [
            i for i, s in enumerate(self.samples)
            if s["accept_label"] >= 0  # 0 or 1, skip -1 (ambiguous)
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        s = self.samples[self.valid_indices[idx]]
        features = torch.tensor(s["gate_features"], dtype=torch.float32)
        label = torch.tensor(float(s["accept_label"]), dtype=torch.float32)
        return {
            "features": features,
            "label": label,
            "base_error_px": torch.tensor(s["base_error_px"], dtype=torch.float32),
            "anchor_error_px": torch.tensor(s["anchor_error_px"], dtype=torch.float32),
            "base_good": torch.tensor(float(s["base_good"]), dtype=torch.float32),
            "base_bad": torch.tensor(float(s["base_bad"]), dtype=torch.float32),
            "base_very_bad": torch.tensor(float(s["base_very_bad"]), dtype=torch.float32),
        }


def make_sample_weight(s: dict) -> float:
    """Upweight base_bad and base_very_bad samples."""
    if s["base_very_bad"]:
        return 4.0
    elif s["base_bad"]:
        return 2.0
    else:
        return 1.0


def compute_val_metrics(
    model: OnlineRecoveryGateHead,
    dataset: GateDataset,
    device: torch.device,
    threshold: float = 0.5,
    gate_features_np: np.ndarray = None,
) -> Dict:
    """Evaluate gate at a given threshold."""
    model.eval()
    all_probs = []
    all_labels = []
    all_base_err = []
    all_anchor_err = []
    all_base_good = []
    all_base_bad = []
    all_vbad = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            out = model(batch["features"].to(device))
            all_probs.append(out["prob"].cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_base_err.append(batch["base_error_px"].numpy())
            all_anchor_err.append(batch["anchor_error_px"].numpy())
            all_base_good.append(batch["base_good"].numpy())
            all_base_bad.append(batch["base_bad"].numpy())
            all_vbad.append(batch["base_very_bad"].numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    base_err = np.concatenate(all_base_err)
    anchor_err = np.concatenate(all_anchor_err)
    base_good = np.concatenate(all_base_good).astype(bool)
    base_bad = np.concatenate(all_base_bad).astype(bool)
    base_vbad = np.concatenate(all_vbad).astype(bool)

    accept = probs > threshold
    # When accepting, error = anchor_err; when rejecting, error = base_err
    final_err = np.where(accept, anchor_err, base_err)

    def _subset_metrics(mask, name):
        if mask.sum() == 0:
            return {"group": name, "n": 0}
        be = base_err[mask]
        ae = anchor_err[mask]
        fe = final_err[mask]
        a = accept[mask]
        # Accept precision: fraction of accepted that actually improve
        accepted_mask = a & mask
        if accepted_mask.sum() > 0:
            acc_precision = float((anchor_err[accepted_mask] < base_err[accepted_mask]).mean())
        else:
            acc_precision = 0.0
        # False override rate: accepted but anchor is worse
        if accepted_mask.sum() > 0:
            false_override = float((anchor_err[accepted_mask] >= base_err[accepted_mask]).mean())
        else:
            false_override = 0.0

        return {
            "group": name, "n": int(mask.sum()),
            "base_median": round(float(np.median(be)), 2),
            "pred_median": round(float(np.median(fe)), 2),
            "delta": round(float(np.median(be) - np.median(fe)), 2),
            "accept_rate": round(float(a.mean()), 3),
            "accept_precision": round(acc_precision, 3),
            "false_override_rate": round(false_override, 3),
            "better_frac": round(float(np.mean(fe < be)), 3),
        }

    return {
        "threshold": threshold,
        "overall": _subset_metrics(np.ones(n, dtype=bool), "overall"),
        "base_good": _subset_metrics(base_good, "base_good_<=4px"),
        "base_bad": _subset_metrics(base_bad, "base_bad_>16px"),
        "base_very_bad": _subset_metrics(base_vbad, "base_very_bad_>32px"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/online_recovery_gate_m1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/online_recovery_gate_m1")
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = GateDataset(args.dataset_dir, "train")
    val_ds = GateDataset(args.dataset_dir, "val")
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Weighted sampler
    weights = [make_sample_weight(train_ds.samples[train_ds.valid_indices[i]]) for i in range(len(train_ds))]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = OnlineRecoveryGateHead(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    print(f"Gate params: {model.param_count:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            out = model(batch["features"].to(device))
            loss = F.binary_cross_entropy_with_logits(out["logit"], batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss))
        scheduler.step()

        # Val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["features"].to(device))
                loss = F.binary_cross_entropy_with_logits(out["logit"], batch["label"].to(device))
                val_losses.append(float(loss))

        val_loss = float(np.mean(val_losses))
        rec = {"epoch": epoch + 1, "train_loss": round(float(np.mean(train_losses)), 4), "val_loss": round(val_loss, 4)}
        history.append(rec)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            rec["best"] = True

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}  train={rec['train_loss']:.4f}  val={rec['val_loss']:.4f}{'  *BEST*' if rec.get('best') else ''}")

    torch.save(model.state_dict(), ckpt_dir / "final.pth")
    (output_dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in history) + "\n")

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=True))

    # Threshold sweep
    threshold_sweep = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = compute_val_metrics(model, val_ds, device, threshold=thr)
        threshold_sweep.append(m)
    (output_dir / "threshold_sweep_val.json").write_text(json.dumps(threshold_sweep, indent=2) + "\n")

    # Best metrics at default threshold
    val_metrics = compute_val_metrics(model, val_ds, device, threshold=0.5)
    (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2) + "\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"M1 Gate Evaluation (seed={args.seed}, threshold=0.5)")
    print(f"{'='*70}")
    print(f"{'Group':<25s} {'N':>5s} {'Base':>8s} {'Pred':>8s} {'Delta':>7s} {'AccR':>6s} {'AccP':>6s} {'FOvr':>6s} {'BetF':>6s}")
    for key in ["overall", "base_good", "base_bad", "base_very_bad"]:
        r = val_metrics[key]
        if r["n"] == 0:
            continue
        print(f"  {r['group']:<23s} {r['n']:>5d} {r['base_median']:>8.2f} {r['pred_median']:>8.2f} {r['delta']:>+7.2f} {r['accept_rate']:>6.3f} {r['accept_precision']:>6.3f} {r['false_override_rate']:>6.3f} {r['better_frac']:>6.3f}")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
