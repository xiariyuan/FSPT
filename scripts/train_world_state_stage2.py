#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.world_state_dataset import PointOdysseyWorldStateDataset
from experiments.world_state_model import TemporalWorldStateRefiner
from scripts.eval_world_state_stage0 import project_3d_to_2d


def project_batch(world: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Differentiable 3D->2D projection. Returns PointOdyssey pixel coords in (x, y)."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    pts_cam = torch.bmm(R, world.unsqueeze(-1)).squeeze(-1) + t
    pts_2d_h = torch.bmm(intrinsics, pts_cam.unsqueeze(-1)).squeeze(-1)
    z = pts_2d_h[:, 2:3].clamp(min=1e-6)
    return pts_2d_h[:, :2] / z


def forward_model(model: TemporalWorldStateRefiner, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return model(
        batch["features"].to(device),
        batch["input_world"].to(device),
        batch["history_xy"].to(device),
        batch["history_xy_delta"].to(device),
        batch["history_world"].to(device),
        batch["history_world_delta"].to(device),
        batch["history_vis"].to(device),
        batch["history_camrot"].to(device),
        query_patch=batch["query_patch"].to(device) if "query_patch" in batch else None,
        reentry_patch=batch["reentry_patch"].to(device) if "reentry_patch" in batch else None,
        query_prev_patch=batch["query_prev_patch"].to(device) if "query_prev_patch" in batch else None,
    )


def _subset_metrics(err: torch.Tensor, base_err: torch.Tensor, mask: torch.Tensor, prefix: str) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return {
            f"{prefix}_reproj_median_px": float("nan"),
            f"{prefix}_baseline_median_px": float("nan"),
            f"{prefix}_reproj_lt4px": float("nan"),
            f"{prefix}_baseline_lt4px": float("nan"),
            f"{prefix}_better_frac": float("nan"),
        }
    e = err[mask]
    b = base_err[mask]
    return {
        f"{prefix}_reproj_median_px": float(torch.median(e)),
        f"{prefix}_baseline_median_px": float(torch.median(b)),
        f"{prefix}_reproj_lt4px": float((e < 4.0).float().mean()),
        f"{prefix}_baseline_lt4px": float((b < 4.0).float().mean()),
        f"{prefix}_better_frac": float((e < b).float().mean()),
    }


def evaluate(model: TemporalWorldStateRefiner, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    reproj_errors = []
    base_errors = []
    world_errors = []
    vis_acc = []
    gate_vals = []
    hard_masks = []
    with torch.no_grad():
        for batch in loader:
            input_world = batch["input_world"].to(device)
            target_world = batch["target_world"].to(device)
            reentry_xy = batch["reentry_xy"].to(device)
            reentry_visible = batch["reentry_visible"].to(device)
            intrinsics = batch["reentry_intrinsics"].to(device)
            extrinsics = batch["reentry_extrinsics"].to(device)
            hard_mask = batch["hard_mask"].to(device) > 0.5

            out = forward_model(model, batch, device)
            refined_world = out["refined_world"]
            reproj_xy = project_batch(refined_world, intrinsics, extrinsics)
            base_xy = project_batch(input_world, intrinsics, extrinsics)

            reproj_err = torch.norm(reproj_xy - reentry_xy, dim=-1)
            base_err = torch.norm(base_xy - reentry_xy, dim=-1)
            world_err = torch.norm(refined_world - target_world, dim=-1)
            vis_pred = (torch.sigmoid(out["visibility_logit"]) > 0.5).float()

            reproj_errors.append(reproj_err.cpu())
            base_errors.append(base_err.cpu())
            world_errors.append(world_err.cpu())
            vis_acc.append((vis_pred == reentry_visible).float().cpu())
            gate_vals.append(out["gate"].cpu())
            hard_masks.append(hard_mask.cpu())

    reproj_errors = torch.cat(reproj_errors)
    base_errors = torch.cat(base_errors)
    world_errors = torch.cat(world_errors)
    vis_acc = torch.cat(vis_acc)
    gate_vals = torch.cat(gate_vals)
    hard_masks = torch.cat(hard_masks)
    metrics = {
        "reproj_median_px": float(torch.median(reproj_errors)),
        "reproj_lt4px": float((reproj_errors < 4.0).float().mean()),
        "baseline_median_px": float(torch.median(base_errors)),
        "baseline_lt4px": float((base_errors < 4.0).float().mean()),
        "better_frac": float((reproj_errors < base_errors).float().mean()),
        "world_l2_mean": float(world_errors.mean()),
        "vis_acc": float(vis_acc.mean()),
        "gate_mean": float(gate_vals.mean()),
    }
    metrics.update(_subset_metrics(reproj_errors, base_errors, hard_masks, "hard"))
    metrics.update(_subset_metrics(reproj_errors, base_errors, ~hard_masks, "easy"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 world-state training")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--train-splits", type=str, default="train")
    parser.add_argument("--val-splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=10)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--train-max-sequences", type=int, default=2)
    parser.add_argument("--val-max-sequences", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=200000)
    parser.add_argument("--val-max-samples", type=int, default=50000)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--train-min-camera-motion", type=float, default=0.0)
    parser.add_argument("--val-min-camera-motion", type=float, default=0.0)
    parser.add_argument("--hard-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--appearance-mode", type=str, default="scratch", choices=["scratch", "dino"])
    parser.add_argument("--use-query-prev-patch", action="store_true")
    parser.add_argument("--patch-feat-dim", type=int, default=128)
    parser.add_argument("--preload-patches", action="store_true")
    parser.add_argument("--causal-mode", action="store_true")
    parser.add_argument("--gate-bias-init", type=float, default=-1.0)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--loss-delta-weight", type=float, default=0.2)
    parser.add_argument("--loss-world-weight", type=float, default=0.2)
    parser.add_argument("--loss-reproj-hard-weight", type=float, default=3.0)
    parser.add_argument("--loss-vis-weight", type=float, default=0.1)
    parser.add_argument("--loss-noharm-easy-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="/gemini/code/FSPT/outputs/world_state_stage2_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PointOdysseyWorldStateDataset(
        data_root=args.data_root,
        splits=args.train_splits,
        min_occ_length=args.min_occ_length,
        max_sequences=args.train_max_sequences,
        max_samples=args.train_max_samples,
        depth_noise_sigma=args.depth_noise_sigma,
        history_len=args.history_len,
        min_camera_motion=args.train_min_camera_motion,
        hard_only=args.hard_only,
        preload_patches=args.preload_patches,
        causal_mode=args.causal_mode,
        seed=42,
    )
    val_dataset = PointOdysseyWorldStateDataset(
        data_root=args.data_root,
        splits=args.val_splits,
        min_occ_length=args.min_occ_length,
        max_sequences=args.val_max_sequences,
        max_samples=args.val_max_samples,
        depth_noise_sigma=args.depth_noise_sigma,
        history_len=args.history_len,
        min_camera_motion=args.val_min_camera_motion,
        hard_only=args.hard_only,
        preload_patches=args.preload_patches,
        causal_mode=args.causal_mode,
        seed=123,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Increase --train-max-sequences or relax hard filters.")
    if len(val_dataset) == 0:
        raise RuntimeError("Val dataset is empty. Increase --val-max-sequences or relax hard filters.")

    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "appearance_mode": args.appearance_mode,
                "use_query_prev_patch": bool(args.use_query_prev_patch),
                "preload_patches": bool(args.preload_patches),
                "causal_mode": bool(args.causal_mode),
                "gate_bias_init": args.gate_bias_init,
                "delta_scale": args.delta_scale,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    sample = train_dataset[0]
    dinov2_weights = str(PROJECT_ROOT / "weights" / "dinov2" / "dinov2_vits14_pretrain.pth")
    model = TemporalWorldStateRefiner(
        feature_dim=int(sample["features"].numel()),
        history_dim=int(
            sample["history_xy"].shape[-1]
            + sample["history_xy_delta"].shape[-1]
            + sample["history_world"].shape[-1]
            + sample["history_world_delta"].shape[-1]
            + sample["history_vis"].shape[-1]
            + sample["history_camrot"].shape[-1]
        ),
        history_len=args.history_len,
        patch_feat_dim=args.patch_feat_dim,
        appearance_mode=args.appearance_mode,
        dinov2_weights=dinov2_weights,
        use_query_prev_patch=args.use_query_prev_patch,
        gate_bias_init=args.gate_bias_init,
        delta_scale=args.delta_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            input_world = batch["input_world"].to(device)
            target_world = batch["target_world"].to(device)
            target_delta = batch["target_delta"].to(device)
            reentry_xy = batch["reentry_xy"].to(device)
            reentry_visible = batch["reentry_visible"].to(device)
            hard_mask = batch["hard_mask"].to(device)
            intrinsics = batch["reentry_intrinsics"].to(device)
            extrinsics = batch["reentry_extrinsics"].to(device)

            out = forward_model(model, batch, device)
            refined_world = out["refined_world"]
            reproj_xy = project_batch(refined_world, intrinsics, extrinsics)
            base_xy = project_batch(input_world, intrinsics, extrinsics)

            base_err = torch.norm(base_xy - reentry_xy, dim=-1)
            reproj_err = torch.norm(reproj_xy - reentry_xy, dim=-1)
            hard = hard_mask
            easy = 1.0 - hard

            loss_delta = F.smooth_l1_loss(out["delta_world"], target_delta)
            loss_world = F.smooth_l1_loss(refined_world, target_world)
            loss_vis = F.binary_cross_entropy_with_logits(out["visibility_logit"], reentry_visible)
            loss_reproj_hard = (hard * (reproj_err / (base_err.detach() + 1.0))).sum() / (hard.sum() + 1e-6)
            loss_noharm_easy = (easy * F.relu(reproj_err - base_err - 0.5)).sum() / (easy.sum() + 1e-6)
            # Heavier weight on reproj_hard so gate gets gradient signal.
            # Lower delta/world so they don't dominate and kill the gate.
            loss = (
                args.loss_delta_weight * loss_delta
                + args.loss_world_weight * loss_world
                + args.loss_reproj_hard_weight * loss_reproj_hard
                + args.loss_vis_weight * loss_vis
                + args.loss_noharm_easy_weight * loss_noharm_easy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(epoch_losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        current_metric = metrics.get("hard_reproj_median_px", float("inf"))
        if np.isfinite(current_metric) and current_metric < best_metric:
            best_metric = float(current_metric)
            best_epoch = epoch + 1
            epochs_without_improve = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "history": history, "best_epoch": best_epoch, "best_metric": best_metric},
                output_dir / "best.pt",
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                break

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    torch.save(
        {"model": model.state_dict(), "args": vars(args), "history": history, "best_epoch": best_epoch, "best_metric": best_metric},
        output_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
