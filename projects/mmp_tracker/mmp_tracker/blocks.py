import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SimpleFeatureEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_dim, stride=2),
            ConvBlock(base_dim, base_dim, stride=1),
            ConvBlock(base_dim, out_dim, stride=2),
            ConvBlock(out_dim, out_dim, stride=1),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = video.shape
        feat = self.stem(video.reshape(batch * time, channels, height, width))
        out_h, out_w = feat.shape[-2:]
        feat = feat.reshape(batch, time, feat.shape[1], out_h, out_w)
        return F.normalize(feat, dim=2)


class TimmFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 256,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        pretrained_path: str = "",
        train_backbone: bool = False,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        fuse_mode: str = "sum",
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("TimmFeatureEncoder currently expects RGB inputs.")
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for TimmFeatureEncoder.") from exc
        weight_path = Path(str(pretrained_path or "")).expanduser() if str(pretrained_path or "").strip() else None
        self.out_indices = tuple(int(idx) for idx in tuple(out_indices))
        self.backbone_name = str(backbone_name)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained) and weight_path is None,
            in_chans=in_channels,
            dynamic_img_size=True,
        )
        if weight_path is not None:
            if not weight_path.is_file():
                raise FileNotFoundError(f"Pretrained weight file not found: {weight_path}")
            state_dict = torch.load(str(weight_path), map_location="cpu")
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict and isinstance(state_dict["model"], dict):
                    state_dict = state_dict["model"]
            load_info = self.backbone.load_state_dict(state_dict, strict=False)
            unexpected = [key for key in load_info.unexpected_keys if key != "mask_token"]
            if load_info.missing_keys or unexpected:
                raise RuntimeError(
                    "Failed to cleanly load pretrained weights from {0}. Missing={1} Unexpected={2}".format(
                        weight_path,
                        load_info.missing_keys,
                        unexpected,
                    )
                )
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", (1, 1))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        channels = [int(getattr(self.backbone, "num_features", out_dim)) for _ in self.out_indices]
        self.projections = nn.ModuleList([nn.Conv2d(int(ch), out_dim, kernel_size=1) for ch in channels])
        self.fuse_mode = str(fuse_mode or "sum").strip().lower()
        if self.fuse_mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported fuse_mode: {self.fuse_mode}")
        if not bool(train_backbone):
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _to_bchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(feat.shape)}")
        if feat.shape[1] >= feat.shape[-1] and feat.shape[1] >= feat.shape[-2]:
            return feat
        return feat.permute(0, 3, 1, 2).contiguous()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = video.shape
        x = video.reshape(batch * time, channels, height, width)
        patch_h, patch_w = self.patch_size
        target_h = max(patch_h, int(round(float(height) / float(patch_h))) * patch_h)
        target_w = max(patch_w, int(round(float(width) / float(patch_w))) * patch_w)
        if target_h != height or target_w != width:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if hasattr(self.backbone, "forward_intermediates"):
            feature_maps = self.backbone.forward_intermediates(
                x,
                indices=self.out_indices,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        elif hasattr(self.backbone, "get_intermediate_layers"):
            feature_maps = self.backbone.get_intermediate_layers(
                x,
                n=self.out_indices,
                reshape=True,
            )
        else:
            raise RuntimeError(
                f"Backbone {self.backbone_name} does not expose intermediate features for dense tracking."
            )
        if not isinstance(feature_maps, (list, tuple)):
            feature_maps = [feature_maps]
        feature_maps = [self._to_bchw(feat) for feat in feature_maps]
        target_h = max(int(feat.shape[-2]) for feat in feature_maps)
        target_w = max(int(feat.shape[-1]) for feat in feature_maps)
        fused = None
        for feat, proj in zip(feature_maps, self.projections):
            feat = proj(feat)
            if feat.shape[-2] != target_h or feat.shape[-1] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            fused = feat if fused is None else fused + feat
        if fused is None:
            raise RuntimeError("Backbone returned no feature maps.")
        if self.fuse_mode == "mean":
            fused = fused / float(len(feature_maps))
        fused = fused.reshape(batch, time, fused.shape[1], target_h, target_w)
        return F.normalize(fused, dim=2)


def yx_to_grid_xy(points_yx: torch.Tensor) -> torch.Tensor:
    points_xy = points_yx[..., [1, 0]]
    return points_xy.mul(2.0).sub(1.0)


def sample_point_features(frame_feat: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    batch, channels, _, _ = frame_feat.shape
    num_points = points_yx.shape[1]
    grid = yx_to_grid_xy(points_yx).unsqueeze(2)
    sampled = F.grid_sample(frame_feat, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(batch, num_points, channels)
    return F.normalize(sampled, dim=-1)


def sample_point_features_per_point(frame_feat_bnchw: torch.Tensor, points_yx: torch.Tensor) -> torch.Tensor:
    batch, num_points, channels, _, _ = frame_feat_bnchw.shape
    feat = frame_feat_bnchw.reshape(batch * num_points, channels, frame_feat_bnchw.shape[-2], frame_feat_bnchw.shape[-1])
    grid = yx_to_grid_xy(points_yx.reshape(batch * num_points, 1, 2)).unsqueeze(2)
    sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).squeeze(-1).reshape(batch, num_points, channels)
    return F.normalize(sampled, dim=-1)


def sample_patch_features(frame_feat: torch.Tensor, points_yx: torch.Tensor, radius: int) -> torch.Tensor:
    batch, channels, height, width = frame_feat.shape
    num_points = points_yx.shape[1]
    feat = frame_feat.unsqueeze(1).expand(batch, num_points, channels, height, width)
    return sample_patch_features_per_point(feat, points_yx, radius)


def sample_patch_features_per_point(frame_feat_bnchw: torch.Tensor, points_yx: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    batch, num_points, channels, height, width = frame_feat_bnchw.shape
    kernel = 2 * radius + 1

    feat = frame_feat_bnchw.reshape(batch * num_points, channels, height, width)
    offsets = local_offset_grid(radius, height, width, feat.device, feat.dtype).view(1, kernel, kernel, 2)
    centers = points_yx.reshape(batch * num_points, 2)
    grid_yx = (centers[:, None, None, :] + offsets).clamp(0.0, 1.0)
    grid_xy = yx_to_grid_xy(grid_yx)

    patch = F.grid_sample(feat, grid_xy, mode="bilinear", padding_mode="border", align_corners=True)
    patch = F.normalize(patch, dim=1)
    patch = patch.flatten(2).transpose(1, 2)
    patch = F.normalize(patch, dim=-1)
    return patch.reshape(batch, num_points, kernel * kernel, channels)


def local_offset_grid(radius: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if height <= 1:
        y_scale = 0.0
    else:
        y_scale = 1.0 / float(height - 1)
    if width <= 1:
        x_scale = 0.0
    else:
        x_scale = 1.0 / float(width - 1)
    ys = torch.arange(-radius, radius + 1, device=device, dtype=dtype) * y_scale
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype) * x_scale
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx], dim=-1)


def normalized_entropy(prob: torch.Tensor, dim: int = -1, eps: float = 1.0e-8) -> torch.Tensor:
    support = max(prob.shape[dim], 1)
    entropy = -(prob * prob.clamp_min(eps).log()).sum(dim=dim)
    if support <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(float(support))
