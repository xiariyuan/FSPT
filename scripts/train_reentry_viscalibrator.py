#!/usr/bin/env python3
"""Train the lightweight ReEntry-VisCalibrator.

The model learns per-frame visible-recovery scores inside candidate re-entry
windows.  It does not predict coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.reentry_viscalibrator import ReEntryVisCalibrator, ReEntryVisCalibratorConfig  # noqa: E402

DEFAULT_DATASET = "outputs/paper_discovery_2026-06-27/reentry_viscalibrator/datasets/rgb_dev10_w16p2_candidates.npz"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1.pt"


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, hard: np.ndarray, valid: np.ndarray, loss_mask: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y = y
        self.hard = hard
        self.valid = valid
        self.loss_mask = loss_mask
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return {
            "x": torch.from_numpy(self.x[idx]),
            "y": torch.from_numpy(self.y[idx]),
            "hard": torch.from_numpy(self.hard[idx]),
            "valid": torch.from_numpy(self.valid[idx]),
            "loss_mask": torch.from_numpy(self.loss_mask[idx]),
        }


def compute_norm(x: np.ndarray, valid: np.ndarray, sample_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    frame_mask = valid & sample_mask[:, None]
    flat = x[frame_mask]
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_boost: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.ones_like(loss) + float(pos_boost) * target
    mask_f = mask.to(dtype=loss.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (loss * weights * mask_f).sum() / denom


def smoothness_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] <= 1:
        return logits.new_tensor(0.0)
    prob = torch.sigmoid(logits)
    diff = torch.abs(prob[:, 1:] - prob[:, :-1])
    pair_mask = (mask[:, 1:] & mask[:, :-1]).to(dtype=prob.dtype)
    denom = pair_mask.sum().clamp_min(1.0)
    return (diff * pair_mask).sum() / denom


@torch.no_grad()
def evaluate(model: ReEntryVisCalibrator, loader: DataLoader, device: torch.device, thresholds: np.ndarray) -> Dict[str, Any]:
    model.eval()
    all_prob = []
    all_hard = []
    all_mask = []
    losses = []
    for batch in loader:
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        valid = batch["valid"].to(device).bool()
        loss_mask = batch["loss_mask"].to(device).bool()
        hard = batch["hard"].to(device).float()
        logits = model(x, valid_mask=valid)
        loss = masked_bce_with_logits(logits, y, loss_mask, pos_boost=0.0)
        losses.append(float(loss.detach().cpu()))
        all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_hard.append(hard.detach().cpu().numpy())
        all_mask.append(loss_mask.detach().cpu().numpy())
    prob = np.concatenate(all_prob, axis=0)
    hard = np.concatenate(all_hard, axis=0).astype(bool)
    mask = np.concatenate(all_mask, axis=0).astype(bool)
    p = prob[mask]
    h = hard[mask]
    sweep = []
    best = None
    for thr in thresholds:
        pred = p >= float(thr)
        tp = int(np.sum(pred & h))
        fp = int(np.sum(pred & ~h))
        fn = int(np.sum(~pred & h))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        beta = 0.5
        fbeta = (1 + beta * beta) * precision * recall / max(beta * beta * precision + recall, 1e-9)
        utility = (tp - 0.75 * fp) / max(int(mask.sum()), 1)
        row = {
            "threshold": round(float(thr), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f0.5": round(float(fbeta), 6),
            "utility_tp_minus_0.75fp_per_frame": round(float(utility), 6),
            "pred_rate": round(float(np.mean(pred)), 6) if pred.size else 0.0,
        }
        sweep.append(row)
        key = (row["f0.5"], row["utility_tp_minus_0.75fp_per_frame"])
        if best is None or key > (best["f0.5"], best["utility_tp_minus_0.75fp_per_frame"]):
            best = row
    return {
        "bce": round(float(np.mean(losses)), 6) if losses else None,
        "hard_positive_rate": round(float(h.mean()), 6) if h.size else None,
        "prob_mean": round(float(p.mean()), 6) if p.size else None,
        "prob_p90": round(float(np.percentile(p, 90)), 6) if p.size else None,
        "best_threshold_by_f0.5": best,
        "threshold_sweep": sweep,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train ReEntry-VisCalibrator.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out-model", default=DEFAULT_OUT)
    ap.add_argument("--out-summary", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos-boost", type=float, default=2.0)
    ap.add_argument("--smooth-weight", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    x_raw = data["x"].astype(np.float32)
    y = data["y_soft"].astype(np.float32)
    hard = data["y_hard"].astype(np.float32)
    valid = data["valid_mask"].astype(bool)
    loss_mask = data["loss_mask"].astype(bool)
    split = data["split"].astype(np.int32)
    feature_names = [str(v) for v in data["feature_names"].tolist()]

    train_sample_mask = split == 0
    val_sample_mask = split == 1
    if not np.any(train_sample_mask):
        raise RuntimeError("No train samples in dataset")
    if not np.any(val_sample_mask):
        # Allow smoke training on tiny datasets.
        val_sample_mask = train_sample_mask.copy()

    mean, std = compute_norm(x_raw, valid, train_sample_mask)
    x = normalize(x_raw, mean, std)
    x[~valid] = 0.0

    train_idx = np.where(train_sample_mask)[0]
    val_idx = np.where(val_sample_mask)[0]
    train_ds = WindowDataset(x, y, hard, valid, loss_mask, train_idx)
    val_ds = WindowDataset(x, y, hard, valid, loss_mask, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device(args.device)
    cfg = ReEntryVisCalibratorConfig(
        input_dim=int(x.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )
    model = ReEntryVisCalibrator(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    thresholds = np.linspace(0.1, 0.9, 17)

    history = []
    best_payload = None
    best_score = -math.inf
    best_epoch = -1
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            xb = batch["x"].to(device).float()
            yb = batch["y"].to(device).float()
            valid_b = batch["valid"].to(device).bool()
            mask_b = batch["loss_mask"].to(device).bool()
            logits = model(xb, valid_mask=valid_b)
            loss = masked_bce_with_logits(logits, yb, mask_b, pos_boost=args.pos_boost)
            if args.smooth_weight > 0:
                loss = loss + float(args.smooth_weight) * smoothness_loss(logits, mask_b)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_losses.append(float(loss.detach().cpu()))

        val = evaluate(model, val_loader, device, thresholds)
        score = float(val["best_threshold_by_f0.5"]["f0.5"] if val["best_threshold_by_f0.5"] else 0.0)
        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 6),
            "val": val,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_payload = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": cfg.to_dict(),
                "feature_names": feature_names,
                "feature_mean": mean.astype(np.float32),
                "feature_std": std.astype(np.float32),
                "best_threshold": float(val["best_threshold_by_f0.5"]["threshold"]),
                "best_epoch": int(best_epoch),
                "dataset": str(args.dataset),
                "train_args": vars(args),
                "val_summary": val,
            }
            torch.save(best_payload, out_model)

    if best_payload is None:
        raise RuntimeError("Training produced no checkpoint")

    summary = {
        "out_model": str(out_model),
        "dataset": str(args.dataset),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_payload["best_threshold"]),
        "best_val": best_payload["val_summary"],
        "n_train_examples": int(train_idx.shape[0]),
        "n_val_examples": int(val_idx.shape[0]),
        "model_config": cfg.to_dict(),
        "history": history,
    }
    out_summary = Path(args.out_summary) if args.out_summary else out_model.with_suffix(".summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out_model": str(out_model),
        "out_summary": str(out_summary),
        "best_epoch": best_epoch,
        "best_threshold": best_payload["best_threshold"],
        "best_val_f0.5": best_payload["val_summary"]["best_threshold_by_f0.5"]["f0.5"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
