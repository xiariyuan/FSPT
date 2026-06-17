#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aim_track.adapters.minimal_vit_adapter import MinimalViTAdapter
from aim_track.backbones.dinov2_wrapper import DinoV2Backbone
from aim_track.data.megadepth_homography_dataset import MegaDepthHomographyDataset
from aim_track.global_matcher.pairwise_global_matcher import PairwiseGlobalMatcher


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    return {
        "source_image": torch.stack([item.source_image for item in batch], dim=0),
        "target_image": torch.stack([item.target_image for item in batch], dim=0),
        "source_points_yx": torch.stack([item.source_points_yx for item in batch], dim=0),
        "target_points_yx": torch.stack([item.target_points_yx for item in batch], dim=0),
        "image_path": [item.image_path for item in batch],
    }


def sample_point_features(feature_map: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    grid = points_yx[..., [1, 0]].mul(2.0).sub(1.0).unsqueeze(2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()
    return F.normalize(sampled, dim=-1)


class PairwiseHomographyWarmupModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        backbone_cfg = cfg["model"]["backbone"]
        adapter_cfg = cfg["model"]["adapter"]
        matcher_cfg = cfg["model"]["matcher"]
        self.backbone = DinoV2Backbone(
            model_name=backbone_cfg["model_name"],
            pretrained=bool(backbone_cfg.get("pretrained", False)),
            weights=backbone_cfg.get("weights", None),
            out_indices=tuple(backbone_cfg.get("out_indices", (8, 10, 11))),
            train_backbone=bool(backbone_cfg.get("train_backbone", False)),
        )
        self.adapter = MinimalViTAdapter(
            in_dims=[self.backbone.embedding_dim] * len(tuple(backbone_cfg.get("out_indices", (8, 10, 11)))),
            hidden_dim=int(adapter_cfg.get("hidden_dim", 256)),
            out_dim=int(adapter_cfg.get("out_dim", 256)),
            fuse_mode=str(adapter_cfg.get("fuse_mode", "mean")),
            base_stride=int(adapter_cfg.get("base_stride", 14)),
        )
        self.matcher = PairwiseGlobalMatcher(
            feature_dim=int(adapter_cfg.get("out_dim", 256)),
            hidden_dim=int(matcher_cfg.get("hidden_dim", 256)),
            topk=int(matcher_cfg.get("topk", 5)),
            temperature=float(matcher_cfg.get("temperature", 0.07)),
            position_bias_strength=float(matcher_cfg.get("position_bias_strength", 0.0)),
            position_bias_sigma=float(matcher_cfg.get("position_bias_sigma", 0.25)),
            use_context_mixer=bool(matcher_cfg.get("use_context_mixer", True)),
        )

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, source_points_yx: torch.Tensor):
        source_backbone = self.backbone(source_image)
        target_backbone = self.backbone(target_image)
        source_pyramid = self.adapter(source_backbone.feature_maps)
        target_pyramid = self.adapter(target_backbone.feature_maps)
        source_feature = source_pyramid.fused_feature
        target_feature = target_pyramid.fused_feature
        anchor_features = sample_point_features(source_feature, source_points_yx)
        matcher_out = self.matcher(anchor_features=anchor_features, dense_features=target_feature, anchor_points=source_points_yx)
        return matcher_out


def build_targets(points_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    y = (points_yx[..., 0] * float(max(height - 1, 1))).round().long().clamp(0, max(height - 1, 0))
    x = (points_yx[..., 1] * float(max(width - 1, 1))).round().long().clamp(0, max(width - 1, 0))
    return y * width + x


def compute_metrics(pred_points: torch.Tensor, target_points: torch.Tensor, height: int, width: int) -> Dict[str, float]:
    dy = (pred_points[..., 0] - target_points[..., 0]) * float(max(height - 1, 1))
    dx = (pred_points[..., 1] - target_points[..., 1]) * float(max(width - 1, 1))
    error_px = torch.sqrt(dy.square() + dx.square() + 1.0e-8)
    metrics = {
        "avg_error_px": float(error_px.mean().item()),
        "median_error_px": float(error_px.median().item()),
        "within_2px": float((error_px <= 2.0).float().mean().item()),
        "within_4px": float((error_px <= 4.0).float().mean().item()),
        "within_8px": float((error_px <= 8.0).float().mean().item()),
    }
    return metrics


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    metric_sum: Dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            source_image = batch["source_image"].to(device, non_blocking=True)
            target_image = batch["target_image"].to(device, non_blocking=True)
            source_points = batch["source_points_yx"].to(device, non_blocking=True)
            target_points = batch["target_points_yx"].to(device, non_blocking=True)
            output = model(source_image, target_image, source_points)
            logits = output.logits.reshape(output.logits.shape[0], output.logits.shape[1], -1)
            target_index = build_targets(target_points, output.logits.shape[-2], output.logits.shape[-1])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_index.reshape(-1))
            reg = F.smooth_l1_loss(output.expected_points, target_points)
            loss = ce + reg
            loss_sum += float(loss.item())
            batch_count += 1
            metrics = compute_metrics(output.expected_points, target_points, output.logits.shape[-2], output.logits.shape[-1])
            for key, value in metrics.items():
                metric_sum[key] = metric_sum.get(key, 0.0) + float(value)
    result = {"val_loss": loss_sum / max(batch_count, 1)}
    for key, value in metric_sum.items():
        result[key] = value / max(batch_count, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise homography warm-up for AnchorBank Stage-0.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseHomographyWarmupModel(cfg).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"].get("lr", 2.0e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-4)),
    )

    data_cfg = cfg["data"]
    train_dataset = MegaDepthHomographyDataset(
        image_root=data_cfg["image_root"],
        split="train",
        image_size=tuple(data_cfg.get("image_size", [224, 224])),
        num_points=int(data_cfg.get("num_points", 32)),
        jitter_ratio=float(data_cfg.get("jitter_ratio", 0.2)),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        photometric_jitter=float(data_cfg.get("photometric_jitter", 0.2)),
        seed=int(args.seed),
        repeat_factor=int(data_cfg.get("train_repeat_factor", 256)),
    )
    val_dataset = MegaDepthHomographyDataset(
        image_root=data_cfg["image_root"],
        split="val",
        image_size=tuple(data_cfg.get("image_size", [224, 224])),
        num_points=int(data_cfg.get("num_points", 32)),
        jitter_ratio=float(data_cfg.get("jitter_ratio", 0.2)),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        photometric_jitter=0.0,
        seed=int(args.seed) + 1337,
        repeat_factor=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"].get("val_batch_size", cfg["training"].get("batch_size", 4))),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    max_steps = int(cfg["training"].get("max_steps", 10000))
    eval_interval = int(cfg["training"].get("eval_interval", 500))
    save_interval = int(cfg["training"].get("save_interval", eval_interval))
    log_interval = int(cfg["training"].get("log_interval", 50))
    coordinate_weight = float(cfg["loss"].get("coordinate", 1.0))
    heatmap_weight = float(cfg["loss"].get("heatmap", 1.0))
    best_metric = math.inf
    step = 0
    epoch = 0
    epoch_metrics_path = output_dir / "epoch_metrics.jsonl"
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best.pth"

    while step < max_steps:
        model.train()
        for batch in train_loader:
            source_image = batch["source_image"].to(device, non_blocking=True)
            target_image = batch["target_image"].to(device, non_blocking=True)
            source_points = batch["source_points_yx"].to(device, non_blocking=True)
            target_points = batch["target_points_yx"].to(device, non_blocking=True)
            output = model(source_image, target_image, source_points)
            logits = output.logits.reshape(output.logits.shape[0], output.logits.shape[1], -1)
            target_index = build_targets(target_points, output.logits.shape[-2], output.logits.shape[-1])
            heatmap_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_index.reshape(-1))
            coordinate_loss = F.smooth_l1_loss(output.expected_points, target_points)
            total_loss = heatmap_weight * heatmap_loss + coordinate_weight * coordinate_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            step += 1
            if step % log_interval == 0 or step == 1:
                metrics = compute_metrics(output.expected_points.detach(), target_points.detach(), output.logits.shape[-2], output.logits.shape[-1])
                record = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "heatmap_loss": float(heatmap_loss.item()),
                    "coordinate_loss": float(coordinate_loss.item()),
                    "loss": float(total_loss.item()),
                    **metrics,
                }
                with epoch_metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if step % eval_interval == 0 or step == max_steps:
                eval_metrics = run_eval(model, val_loader, device=device)
                record = {"event": "eval", "epoch": epoch, "step": step, **eval_metrics}
                with epoch_metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": cfg,
                    "eval": eval_metrics,
                }
                torch.save(checkpoint, latest_path)
                if float(eval_metrics["val_loss"]) < best_metric:
                    best_metric = float(eval_metrics["val_loss"])
                    torch.save(checkpoint, best_path)
            if step >= max_steps:
                break
        epoch += 1


if __name__ == "__main__":
    main()

