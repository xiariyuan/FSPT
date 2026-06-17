import math

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
