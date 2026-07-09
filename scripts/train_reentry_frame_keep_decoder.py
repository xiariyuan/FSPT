#!/usr/bin/env python3
"""Train V2.3 frame-keep / risk decoder.

The decoder is trained only on V1-proposed recovery frames.  It predicts whether
a proposed recovery frame should be kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FrameKeepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = [((x - mean[None, :]) / std[None, :]).astype(np.float32) for x in others]
    return mean, std, out


def eval_probs(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device).float()
            probs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros(0, dtype=np.float32)


def sweep_sample_threshold(probs: np.ndarray, y: np.ndarray, utility: np.ndarray) -> Dict[str, Any]:
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        keep = probs >= float(thr)
        acc = float((keep.astype(np.float32) == y).mean()) if y.size else 0.0
        # Utility here is diagnostic only. It is meaningful for utility/safe labels but
        # actual method selection should use downstream AJ_RD/AJ eval.
        total_utility = float(utility[keep].sum()) if utility.size else 0.0
        keep_rate = float(keep.mean()) if keep.size else 0.0
        pos_recall = float((keep & (y > 0.5)).sum()) / max(float((y > 0.5).sum()), 1.0)
        neg_drop = float(((~keep) & (y <= 0.5)).sum()) / max(float((y <= 0.5).sum()), 1.0)
        row = {
            "threshold": round(float(thr), 4),
            "accuracy": acc,
            "total_utility": total_utility,
            "keep_rate": keep_rate,
            "positive_recall": pos_recall,
            "negative_drop_rate": neg_drop,
        }
        if best is None or (row["accuracy"], row["total_utility"]) > (best["accuracy"], best["total_utility"]):
            best = row
    return best or {"threshold": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train V2.3 frame-keep decoder")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--label", required=True, choices=["y_gt_visible", "y_safe16", "y_safe8", "y_safe4", "y_utility"])
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-utility-weights", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train = load_npz(args.train_npz)
    val = load_npz(args.val_npz)
    Xtr = np.asarray(train["X"], dtype=np.float32)
    Xva = np.asarray(val["X"], dtype=np.float32)
    ytr = np.asarray(train[args.label], dtype=np.float32)
    yva = np.asarray(val[args.label], dtype=np.float32)
    utr = np.asarray(train["utility"], dtype=np.float32)
    uva = np.asarray(val["utility"], dtype=np.float32)
    if args.use_utility_weights:
        wtr = np.asarray(train["weight"], dtype=np.float32)
    else:
        wtr = np.ones_like(ytr, dtype=np.float32)
    mean, std, [Xtrn, Xvan] = standardize(Xtr, Xtr, Xva)

    device = torch.device(args.device)
    model = FrameKeepMLP(Xtr.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(Xtrn), torch.from_numpy(ytr), torch.from_numpy(wtr))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_payload = None
    best_score = -1e18
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, wb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            wb = wb.to(device).float()
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        pva = eval_probs(model, Xvan, device)
        sweep = sweep_sample_threshold(pva, yva, uva)
        hist = {"epoch": epoch, "train_loss": float(np.mean(losses)), **sweep}
        history.append(hist)
        score = float(sweep["accuracy"])
        if score > best_score:
            best_score = score
            best_payload = {
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "feature_mean": mean,
                "feature_std": std,
                "feature_names": [str(x) for x in train["feature_names"].tolist()],
                "label": args.label,
                "best_threshold_sample": float(sweep["threshold"]),
                "best_val_sample": sweep,
                "model_config": {"in_dim": int(Xtr.shape[1]), "hidden": int(args.hidden), "dropout": float(args.dropout)},
                "train_npz": str(args.train_npz),
                "val_npz": str(args.val_npz),
                "use_utility_weights": bool(args.use_utility_weights),
                "history": history,
            }
        print(json.dumps(hist, ensure_ascii=False), flush=True)

    out = Path(args.out_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, out)
    print(json.dumps({
        "out_model": str(out),
        "label": args.label,
        "best_epoch": best_payload["epoch"],
        "best_threshold_sample": best_payload["best_threshold_sample"],
        "best_val_sample": best_payload["best_val_sample"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
