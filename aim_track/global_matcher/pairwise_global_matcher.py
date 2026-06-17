from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_entropy(probabilities: torch.Tensor, dim: int = -1, eps: float = 1.0e-8) -> torch.Tensor:
    support = max(int(probabilities.shape[dim]), 1)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=dim)
    if support <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(float(support))


def _flatten_dense_features(dense_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    if dense_features.ndim == 4:
        return dense_features, None
    if dense_features.ndim != 5:
        raise ValueError("Expected dense features with shape (B, C, H, W) or (B, T, C, H, W).")
    batch, time, channels, height, width = dense_features.shape
    return dense_features.reshape(batch * time, channels, height, width), (batch, time)


def _restore_dense_features(tensor: torch.Tensor, shape: Optional[Tuple[int, int]]) -> torch.Tensor:
    if shape is None:
        return tensor
    batch, time = shape
    return tensor.reshape(batch, time, tensor.shape[1], tensor.shape[2], tensor.shape[3])


def _restore_points(tensor: torch.Tensor, shape: Optional[Tuple[int, int]]) -> torch.Tensor:
    if shape is None:
        return tensor
    batch, time = shape
    return tensor.reshape(batch, time, tensor.shape[1], tensor.shape[2], tensor.shape[3])


def _restore_values(tensor: torch.Tensor, shape: Optional[Tuple[int, int]]) -> torch.Tensor:
    if shape is None:
        return tensor
    batch, time = shape
    return tensor.reshape(batch, time, tensor.shape[1], tensor.shape[2])


@dataclass
class PairwiseMatcherOutput:
    points: torch.Tensor
    expected_points: torch.Tensor
    scores: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    heatmap: torch.Tensor
    logits: torch.Tensor


class PairwiseGlobalMatcher(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        topk: int = 5,
        temperature: float = 0.07,
        position_bias_strength: float = 0.0,
        position_bias_sigma: float = 0.25,
        use_context_mixer: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.position_bias_strength = float(position_bias_strength)
        self.position_bias_sigma = float(position_bias_sigma)
        self.anchor_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.key_proj = nn.Conv2d(self.feature_dim, self.hidden_dim, kernel_size=1, bias=False)
        if bool(use_context_mixer):
            self.context_mixer = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim, bias=False),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            )
        else:
            self.context_mixer = nn.Identity()

    @staticmethod
    def _aggregate_anchor_features(
        anchor_features: torch.Tensor,
        anchor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_features.ndim == 3:
            return anchor_features
        if anchor_features.ndim != 4:
            raise ValueError("Expected anchor features with shape (B, N, C) or (B, N, S, C).")
        if anchor_weights is None:
            anchor_weights = torch.ones(
                anchor_features.shape[:3],
                device=anchor_features.device,
                dtype=anchor_features.dtype,
            )
        anchor_weights = anchor_weights / anchor_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        return (anchor_weights.unsqueeze(-1) * anchor_features).sum(dim=2)

    @staticmethod
    def _aggregate_anchor_points(
        anchor_points: Optional[torch.Tensor],
        anchor_weights: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if anchor_points is None:
            return None
        if anchor_points.ndim == 3:
            return anchor_points
        if anchor_points.ndim != 4:
            raise ValueError("Expected anchor points with shape (B, N, 2) or (B, N, S, 2).")
        if anchor_weights is None:
            anchor_weights = torch.ones(anchor_points.shape[:3], device=anchor_points.device, dtype=anchor_points.dtype)
        anchor_weights = anchor_weights / anchor_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        return (anchor_weights.unsqueeze(-1) * anchor_points).sum(dim=2)

    @staticmethod
    def _coordinate_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        grid_x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        return torch.stack([mesh_y, mesh_x], dim=-1).reshape(1, 1, height * width, 2)

    def forward(
        self,
        anchor_features: torch.Tensor,
        dense_features: torch.Tensor,
        anchor_points: Optional[torch.Tensor] = None,
        anchor_weights: Optional[torch.Tensor] = None,
    ) -> PairwiseMatcherOutput:
        if self.topk <= 0:
            raise ValueError("topk must be positive for PairwiseGlobalMatcher.")

        dense_features, leading_shape = _flatten_dense_features(dense_features)
        batch, _, height, width = dense_features.shape
        aggregated_anchor_features = self._aggregate_anchor_features(anchor_features, anchor_weights)
        aggregated_anchor_points = self._aggregate_anchor_points(anchor_points, anchor_weights)

        if leading_shape is not None:
            if aggregated_anchor_features.ndim != 3:
                raise ValueError("Video dense features currently expect anchor features with shape (B, N, C).")
            batch_size, time = leading_shape
            aggregated_anchor_features = (
                aggregated_anchor_features.unsqueeze(1)
                .expand(batch_size, time, aggregated_anchor_features.shape[1], aggregated_anchor_features.shape[2])
                .reshape(batch_size * time, aggregated_anchor_features.shape[1], aggregated_anchor_features.shape[2])
            )
            if aggregated_anchor_points is not None:
                aggregated_anchor_points = (
                    aggregated_anchor_points.unsqueeze(1)
                    .expand(batch_size, time, aggregated_anchor_points.shape[1], aggregated_anchor_points.shape[2])
                    .reshape(batch_size * time, aggregated_anchor_points.shape[1], aggregated_anchor_points.shape[2])
                )

        if aggregated_anchor_features.shape[0] != batch:
            raise ValueError("Anchor features and dense features must share the same batch dimension after flattening.")

        query = F.normalize(self.anchor_proj(aggregated_anchor_features), dim=-1)
        dense = self.key_proj(dense_features)
        dense = dense + self.context_mixer(dense)
        dense = F.normalize(dense.flatten(2).transpose(1, 2), dim=-1)
        logits = torch.einsum("bnc,bkc->bnk", query, dense) / self.temperature

        if aggregated_anchor_points is not None and self.position_bias_strength > 0.0:
            flat_coords = self._coordinate_grid(height, width, dense_features.device, dense_features.dtype)
            sigma2 = max(self.position_bias_sigma * self.position_bias_sigma, 1.0e-6)
            delta = flat_coords - aggregated_anchor_points.unsqueeze(2)
            position_bias = -delta.square().sum(dim=-1) / (2.0 * sigma2)
            logits = logits + self.position_bias_strength * position_bias

        probabilities = torch.softmax(logits, dim=-1)
        topk = min(self.topk, int(probabilities.shape[-1]))
        scores, indices = torch.topk(probabilities, k=topk, dim=-1)

        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        if height > 1:
            y = y.to(dense_features.dtype) / float(height - 1)
        else:
            y = torch.zeros_like(y, dtype=dense_features.dtype)
        if width > 1:
            x = x.to(dense_features.dtype) / float(width - 1)
        else:
            x = torch.zeros_like(x, dtype=dense_features.dtype)
        points = torch.stack([y, x], dim=-1)

        flat_coords = self._coordinate_grid(height, width, dense_features.device, dense_features.dtype)
        expected_points = (probabilities.unsqueeze(-1) * flat_coords).sum(dim=2)
        confidence = scores[..., 0]
        entropy = _normalized_entropy(probabilities, dim=-1)
        heatmap = probabilities.view(batch, probabilities.shape[1], height, width)

        points = _restore_points(points, leading_shape)
        expected_points = _restore_values(expected_points, leading_shape)
        scores = _restore_values(scores, leading_shape)
        confidence = _restore_values(confidence.unsqueeze(-1), leading_shape).squeeze(-1)
        entropy = _restore_values(entropy.unsqueeze(-1), leading_shape).squeeze(-1)
        heatmap = _restore_dense_features(heatmap, leading_shape)
        logits = _restore_dense_features(logits.view(batch, logits.shape[1], height, width), leading_shape)

        return PairwiseMatcherOutput(
            points=points,
            expected_points=expected_points,
            scores=scores,
            confidence=confidence,
            entropy=entropy,
            heatmap=heatmap,
            logits=logits,
        )
