#!/usr/bin/env python3
"""
Train the geometric recovery head (M0 offline).

Frozen: CoTracker, DINO backbone.
Trained: OnlineRecoveryHead only.

Usage:
  python scripts/train_online_recovery_head.py \
    --dataset-dir outputs/recovery_anchor_dataset_v3_full \
    --output-dir outputs/train_online_recovery_head_m0 \
    --epochs 40 --batch-size 16 --lr 1e-3
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.online_recovery_head import OnlineRecoveryHead
from datasets.recovery_anchor_dataset import RecoveryAnchorDataset, make_sample_weight


def compute_metrics(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    base_xy: np.ndarray,
    gt_inside: np.ndarray,
    base_errors: np.ndarray,
    search_crop_size: np.ndarray,
) -> dict:
    """Compute metrics in pixel coords, filtering to gt_inside samples."""
    # Convert pred from normalized [0,1] to pixel offset from crop center
    half = search_crop_size / 2.0
    pred_offset_x = (pred_xy[:, 0] - 0.5) * search_crop_size
    pred_offset_y = (pred_xy[:, 1] - 0.5) * search_crop_size
    pred_px = np.stack([
        base_xy[:, 0] + pred_offset_x,
        base_xy[:, 1] + pred_offset_y,
    ], axis=-1)

    errors = np.linalg.norm(pred_px - gt_xy, axis=-1)

    # Only evaluate on gt_inside samples
    valid = gt_inside.astype(bool)
    if valid.sum() == 0:
        return {"n": 0}

    err = errors[valid]
    be = base_errors[valid]

    def _stats(e, label=""):
        return {
            "mean": round(float(np.mean(e)), 2),
            "median": round(float(np.median(e)), 2),
            "p90": round(float(np.percentile(e, 90)), 2),
            "p95": round(float(np.percentile(e, 95)), 2),
            "lt4px": round(float(np.mean(e < 4.0)), 4),
        }

    result = {
        "n": int(valid.sum()),
        "head": _stats(err),
        "base": _stats(be),
        "better_frac": round(float(np.mean(err < be)), 3),
    }

    # Base-bad subsets
    for thr in [16, 32]:
        mask = valid & (base_errors > thr)
        if mask.sum() > 0:
            result[f"base_gt_{thr}"] = {
                "n": int(mask.sum()),
                "head": _stats(errors[mask]),
                "base": _stats(base_errors[mask]),
                "better_frac": round(float(np.mean(errors[mask] < base_errors[mask])), 3),
            }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/train_online_recovery_head_m0")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/online_recovery_head_m0")
    parser.add_argument("--feat-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=0.25)
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

    # Load datasets
    train_ds = RecoveryAnchorDataset(args.dataset_dir, "train")
    val_ds = RecoveryAnchorDataset(args.dataset_dir, "val")
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Weighted sampler for train
    weights = []
    for i in range(len(train_ds)):
        s = train_ds.samples[train_ds.valid_indices[i]]
        weights.append(make_sample_weight(s["base_error_px"]))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = OnlineRecoveryHead(
        feat_dim=args.feat_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)
    print(f"Model params: {model.param_count:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    history = []
    best_val_metric = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            search_fmap = batch["search_feature_map"].to(device)
            supp_desc = batch["support_descriptor"].to(device)
            tracker_vis = batch["tracker_vis"].to(device)
            heatmap_target = batch["heatmap_target"].to(device)
            gt_local = batch["gt_local_xy"].to(device)
            gt_inside = batch["gt_inside"].to(device).bool()  # (B,)

            out = model(search_fmap, supp_desc, tracker_vis)

            # L_heatmap: BCE, masked to gt_inside samples only
            loss_hm_raw = F.binary_cross_entropy_with_logits(
                out["heatmap"].squeeze(1), heatmap_target, reduction="none"
            ).mean(dim=[1, 2])  # (B,)

            # L_coord: L1, masked to gt_inside samples only
            loss_coord_raw = F.smooth_l1_loss(out["pred_xy"], gt_local, reduction="none").mean(dim=-1)  # (B,)

            if gt_inside.any():
                loss_hm = (loss_hm_raw * gt_inside.float()).sum() / gt_inside.float().sum()
                loss_coord = (loss_coord_raw * gt_inside.float()).sum() / gt_inside.float().sum()
                loss = args.heatmap_weight * loss_hm + args.coord_weight * loss_coord
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        scheduler.step()

        # Validation
        model.eval()
        all_pred_xy = []
        all_gt_xy = []
        all_base_xy = []
        all_gt_inside = []
        all_base_err = []
        all_search_size = []
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                search_fmap = batch["search_feature_map"].to(device)
                supp_desc = batch["support_descriptor"].to(device)
                tracker_vis = batch["tracker_vis"].to(device)
                heatmap_target = batch["heatmap_target"].to(device)
                gt_local = batch["gt_local_xy"].to(device)
                gt_inside = batch["gt_inside"].to(device).bool()

                out = model(search_fmap, supp_desc, tracker_vis)

                loss_hm_raw = F.binary_cross_entropy_with_logits(
                    out["heatmap"].squeeze(1), heatmap_target, reduction="none"
                ).mean(dim=[1, 2])
                loss_coord_raw = F.smooth_l1_loss(out["pred_xy"], gt_local, reduction="none").mean(dim=-1)

                if gt_inside.any():
                    loss_hm = (loss_hm_raw * gt_inside.float()).sum() / gt_inside.float().sum()
                    loss_coord = (loss_coord_raw * gt_inside.float()).sum() / gt_inside.float().sum()
                    val_losses.append(float(
                        (args.heatmap_weight * loss_hm + args.coord_weight * loss_coord).item()
                    ))
                else:
                    val_losses.append(0.0)

                all_pred_xy.append(out["pred_xy"].cpu().numpy())
                all_gt_xy.append(batch["gt_xy"].numpy())
                all_base_xy.append(batch["base_xy"].numpy())
                all_gt_inside.append(batch["gt_inside"].numpy())
                all_base_err.append(batch["base_error_px"].numpy())
                all_search_size.append(batch["search_crop_size"].numpy())

        pred_xy = np.concatenate(all_pred_xy)
        gt_xy = np.concatenate(all_gt_xy)
        base_xy = np.concatenate(all_base_xy)
        gt_inside = np.concatenate(all_gt_inside)
        base_err = np.concatenate(all_base_err)
        search_size = np.concatenate(all_search_size)

        metrics = compute_metrics(pred_xy, gt_xy, base_xy, gt_inside, base_err, search_size)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "val_loss": round(float(np.mean(val_losses)), 4),
            "val_metrics": metrics,
        }
        history.append(epoch_record)

        # Print
        n_val = metrics.get("n", 0)
        head_med = metrics.get("head", {}).get("median", float("nan"))
        base_med = metrics.get("base", {}).get("median", float("nan"))
        better = metrics.get("better_frac", 0)
        bad16 = metrics.get("base_gt_16", {})
        bad16_better = bad16.get("better_frac", 0) if bad16 else 0

        print(f"Epoch {epoch+1:>3d}/{args.epochs}  "
              f"train_loss={np.mean(train_losses):.4f}  "
              f"val_loss={np.mean(val_losses):.4f}  "
              f"head_med={head_med:.2f}  base_med={base_med:.2f}  "
              f"better={better:.3f}  "
              f"bad16_better={bad16_better:.3f}")

        # Save best
        val_key = metrics.get("base_gt_16", {}).get("head", {}).get("median", float("inf"))
        if val_key < best_val_metric:
            best_val_metric = val_key
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> Best (base_gt_16 head median: {val_key:.2f})")

    # Save final
    torch.save(model.state_dict(), ckpt_dir / "final.pth")
    (output_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")

    # Final best checkpoint eval
    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=True))
    model.eval()
    all_pred_xy = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                batch["search_feature_map"].to(device),
                batch["support_descriptor"].to(device),
                batch["tracker_vis"].to(device),
            )
            all_pred_xy.append(out["pred_xy"].cpu().numpy())

    pred_xy = np.concatenate(all_pred_xy)
    final_metrics = compute_metrics(pred_xy, gt_xy, base_xy, gt_inside, base_err, search_size)

    print(f"\n{'='*50}")
    print(f"Best checkpoint val metrics:")
    print(f"  All (n={final_metrics['n']}):")
    print(f"    Head: {final_metrics['head']}")
    print(f"    Base: {final_metrics['base']}")
    print(f"    better_frac: {final_metrics['better_frac']}")
    if "base_gt_16" in final_metrics:
        b = final_metrics["base_gt_16"]
        print(f"  Base>16px (n={b['n']}):")
        print(f"    Head: {b['head']}")
        print(f"    Base: {b['base']}")
        print(f"    better_frac: {b['better_frac']}")
    if "base_gt_32" in final_metrics:
        b = final_metrics["base_gt_32"]
        print(f"  Base>32px (n={b['n']}):")
        print(f"    Head: {b['head']}")
        print(f"    Base: {b['base']}")
        print(f"    better_frac: {b['better_frac']}")

    (output_dir / "val_metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
