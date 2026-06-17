from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_resolve_group_count(out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


@dataclass
class DensePyramidOutput:
    features: Tuple[torch.Tensor, ...]
    feature_names: Tuple[str, ...]
    strides: Tuple[int, ...]
    fused_feature: torch.Tensor


class MinimalViTAdapter(nn.Module):
    def __init__(
        self,
        in_dims: Sequence[int],
        hidden_dim: int = 256,
        out_dim: int = 256,
        fuse_mode: str = "mean",
        base_stride: int = 14,
    ):
        super().__init__()
        if not in_dims:
            raise ValueError("in_dims must not be empty.")
        self.in_dims = tuple(int(dim) for dim in in_dims)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.base_stride = int(base_stride)
        self.fuse_mode = str(fuse_mode or "mean").strip().lower()
        if self.fuse_mode not in {"sum", "mean"}:
            raise ValueError("Unsupported fuse_mode: {0}".format(self.fuse_mode))

        self.input_projections = nn.ModuleList([_ConvNormAct(dim, self.hidden_dim, kernel_size=1) for dim in self.in_dims])
        self.fuse_block = _ConvNormAct(self.hidden_dim, self.hidden_dim, kernel_size=3)
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, self.out_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(_resolve_group_count(self.out_dim), self.out_dim),
            nn.GELU(),
        )
        self.base_block = _ConvNormAct(self.hidden_dim, self.out_dim, kernel_size=3)
        self.down_block_1 = _ConvNormAct(self.hidden_dim, self.out_dim, kernel_size=3, stride=2)
        self.down_block_2 = _ConvNormAct(self.out_dim, self.out_dim, kernel_size=3, stride=2)
        self.feature_names = ("p2", "p3", "p4", "p5")
        self.strides = (
            max(1, self.base_stride // 2),
            self.base_stride,
            self.base_stride * 2,
            self.base_stride * 4,
        )

    @staticmethod
    def _flatten_leading_dims(feature_map: torch.Tensor) -> Tuple[torch.Tensor, Union[None, Tuple[int, int]]]:
        if feature_map.ndim == 4:
            return feature_map, None
        if feature_map.ndim != 5:
            raise ValueError("Expected feature map with shape (B, C, H, W) or (B, T, C, H, W).")
        batch, time, channels, height, width = feature_map.shape
        return feature_map.reshape(batch * time, channels, height, width), (batch, time)

    @staticmethod
    def _restore_leading_dims(feature_map: torch.Tensor, shape: Union[None, Tuple[int, int]]) -> torch.Tensor:
        if shape is None:
            return feature_map
        batch, time = shape
        return feature_map.reshape(batch, time, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    @staticmethod
    def _resize_like(feature_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        if feature_map.shape[-2:] == target_size:
            return feature_map
        return F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)

    @staticmethod
    def _downsample_size(height: int, width: int) -> Tuple[int, int]:
        return max(1, height // 2), max(1, width // 2)

    def forward(self, feature_maps: Sequence[torch.Tensor]) -> DensePyramidOutput:
        if len(feature_maps) != len(self.input_projections):
            raise ValueError(
                "Expected {0} feature maps, got {1}.".format(len(self.input_projections), len(feature_maps))
            )

        flattened = []
        leading_shape = None
        for feature_map in feature_maps:
            flat_map, current_shape = self._flatten_leading_dims(feature_map)
            if leading_shape is None:
                leading_shape = current_shape
            elif current_shape != leading_shape:
                raise ValueError("All feature maps must share the same leading dimensions.")
            flattened.append(flat_map)

        target_h = max(int(feature_map.shape[-2]) for feature_map in flattened)
        target_w = max(int(feature_map.shape[-1]) for feature_map in flattened)
        target_size = (target_h, target_w)

        fused = None
        for feature_map, projection in zip(flattened, self.input_projections):
            projected = projection(feature_map)
            projected = self._resize_like(projected, target_size)
            fused = projected if fused is None else fused + projected
        if fused is None:
            raise RuntimeError("No feature maps were provided for fusion.")
        if self.fuse_mode == "mean":
            fused = fused / float(len(flattened))
        fused = self.fuse_block(fused)

        up_feature = self.up_block(fused)
        base_feature = self.base_block(fused)
        down_feature_1 = self.down_block_1(fused)
        if min(int(down_feature_1.shape[-2]), int(down_feature_1.shape[-1])) < 2:
            down_feature_2 = down_feature_1
        else:
            down_feature_2 = self.down_block_2(down_feature_1)

        outputs = (up_feature, base_feature, down_feature_1, down_feature_2)
        outputs = tuple(self._restore_leading_dims(feature_map, leading_shape) for feature_map in outputs)
        fused = self._restore_leading_dims(fused, leading_shape)
        return DensePyramidOutput(
            features=outputs,
            feature_names=self.feature_names,
            strides=self.strides,
            fused_feature=fused,
        )
