from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import normalized_entropy
from .memory_bank import VisibleMemoryBank


@dataclass
class GlobalRelocOutput:
    points: torch.Tensor
    expected_points: torch.Tensor
    scores: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    heatmap: torch.Tensor


class GlobalRelocator(nn.Module):
    def __init__(self, topk: int = 5, temperature: float = 0.07):
        super().__init__()
        self.topk = int(topk)
        self.temperature = float(temperature)

    def forward(self, frame_feat: torch.Tensor, memory: VisibleMemoryBank) -> GlobalRelocOutput:
        batch, channels, height, width = frame_feat.shape
        num_points = memory.descriptors.shape[1]
        if self.topk <= 0:
            empty_points = torch.zeros(batch, num_points, 0, 2, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_expected = torch.zeros(batch, num_points, 2, device=frame_feat.device, dtype=frame_feat.dtype)
            empty_scores = torch.zeros(batch, num_points, 0, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_conf = torch.zeros(batch, num_points, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_entropy = torch.zeros(batch, num_points, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_heatmap = torch.zeros(batch, num_points, height, width, device=frame_feat.device, dtype=frame_feat.dtype)
            return GlobalRelocOutput(
                points=empty_points,
                expected_points=zero_expected,
                scores=empty_scores,
                confidence=zero_conf,
                entropy=zero_entropy,
                heatmap=zero_heatmap,
            )
        flat_feat = frame_feat.flatten(2)
        flat_feat = F.normalize(flat_feat, dim=1)

        num_slots = memory.descriptors.shape[2]
        slot_desc = F.normalize(memory.descriptors, dim=-1)
        slot_logits = torch.einsum("bnmc,bch->bnmh", slot_desc, flat_feat) / self.temperature

        valid = memory.valid.to(dtype=frame_feat.dtype)
        visibility = memory.visibility.to(dtype=frame_feat.dtype).clamp_min(1.0e-4)
        if num_slots > 1:
            recency = torch.linspace(
                0.5,
                1.0,
                steps=num_slots,
                device=frame_feat.device,
                dtype=frame_feat.dtype,
            ).view(1, 1, num_slots)
        else:
            recency = torch.ones(1, 1, num_slots, device=frame_feat.device, dtype=frame_feat.dtype)

        slot_weight = (valid * visibility * recency).clamp_min(1.0e-6)
        slot_logits = slot_logits + slot_weight.log().unsqueeze(-1)
        invalid_fill = torch.full_like(slot_logits, -1.0e4)
        slot_logits = torch.where(valid.unsqueeze(-1) > 0, slot_logits, invalid_fill)

        logits = torch.logsumexp(slot_logits, dim=2)
        prob = torch.softmax(logits, dim=-1)
        topk = min(self.topk, prob.shape[-1])
        scores, indices = torch.topk(prob, k=topk, dim=-1)

        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        if height > 1:
            y = y.to(frame_feat.dtype) / float(height - 1)
        else:
            y = torch.zeros_like(y, dtype=frame_feat.dtype)
        if width > 1:
            x = x.to(frame_feat.dtype) / float(width - 1)
        else:
            x = torch.zeros_like(x, dtype=frame_feat.dtype)
        points = torch.stack([y, x], dim=-1)

        grid_y = torch.linspace(0.0, 1.0, steps=height, device=frame_feat.device, dtype=frame_feat.dtype)
        grid_x = torch.linspace(0.0, 1.0, steps=width, device=frame_feat.device, dtype=frame_feat.dtype)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        flat_coords = torch.stack([mesh_y, mesh_x], dim=-1).reshape(1, 1, height * width, 2)
        expected_points = (prob.unsqueeze(-1) * flat_coords).sum(dim=2)

        confidence = scores[..., 0]
        entropy = normalized_entropy(prob, dim=-1)
        heatmap = prob.view(batch, num_points, height, width)
        return GlobalRelocOutput(
            points=points,
            expected_points=expected_points,
            scores=scores,
            confidence=confidence,
            entropy=entropy,
            heatmap=heatmap,
        )


class PairwiseGlobalRelocator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        topk: int = 5,
        temperature: float = 0.07,
        hidden_dim: int = 256,
        position_bias_strength: float = 0.0,
        position_bias_sigma: float = 0.25,
    ):
        super().__init__()
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.position_bias_strength = float(position_bias_strength)
        self.position_bias_sigma = float(position_bias_sigma)
        self.query_proj = nn.Linear(int(feature_dim), int(hidden_dim))
        self.key_proj = nn.Conv2d(int(feature_dim), int(hidden_dim), kernel_size=1)
        self.context_mixer = nn.Sequential(
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1, groups=int(hidden_dim), bias=False),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=1, bias=False),
        )

    @staticmethod
    def _slot_weights(memory: VisibleMemoryBank, dtype: torch.dtype) -> torch.Tensor:
        valid = memory.valid.to(dtype=dtype)
        visibility = memory.visibility.to(dtype=dtype).clamp_min(1.0e-4)
        num_slots = memory.descriptors.shape[2]
        if num_slots > 1:
            recency = torch.linspace(
                0.5,
                1.0,
                steps=num_slots,
                device=memory.descriptors.device,
                dtype=dtype,
            ).view(1, 1, num_slots)
        else:
            recency = torch.ones(1, 1, num_slots, device=memory.descriptors.device, dtype=dtype)
        weights = (valid * visibility * recency).clamp_min(1.0e-6)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        return weights

    def forward(self, frame_feat: torch.Tensor, memory: VisibleMemoryBank) -> GlobalRelocOutput:
        batch, channels, height, width = frame_feat.shape
        num_points = memory.descriptors.shape[1]
        if self.topk <= 0:
            empty_points = torch.zeros(batch, num_points, 0, 2, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_expected = torch.zeros(batch, num_points, 2, device=frame_feat.device, dtype=frame_feat.dtype)
            empty_scores = torch.zeros(batch, num_points, 0, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_conf = torch.zeros(batch, num_points, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_entropy = torch.zeros(batch, num_points, device=frame_feat.device, dtype=frame_feat.dtype)
            zero_heatmap = torch.zeros(batch, num_points, height, width, device=frame_feat.device, dtype=frame_feat.dtype)
            return GlobalRelocOutput(
                points=empty_points,
                expected_points=zero_expected,
                scores=empty_scores,
                confidence=zero_conf,
                entropy=zero_entropy,
                heatmap=zero_heatmap,
            )

        slot_weights = self._slot_weights(memory, dtype=frame_feat.dtype)
        anchor_desc = (slot_weights.unsqueeze(-1) * memory.descriptors).sum(dim=2)
        anchor_pos = (slot_weights.unsqueeze(-1) * memory.positions).sum(dim=2)

        query = F.normalize(self.query_proj(anchor_desc), dim=-1)
        dense = self.key_proj(frame_feat)
        dense = dense + self.context_mixer(dense)
        dense = F.normalize(dense.flatten(2).transpose(1, 2), dim=-1)
        logits = torch.einsum("bnd,bhd->bnh", query, dense) / self.temperature

        if self.position_bias_strength > 0:
            grid_y = torch.linspace(0.0, 1.0, steps=height, device=frame_feat.device, dtype=frame_feat.dtype)
            grid_x = torch.linspace(0.0, 1.0, steps=width, device=frame_feat.device, dtype=frame_feat.dtype)
            mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            flat_coords = torch.stack([mesh_y, mesh_x], dim=-1).reshape(1, 1, height * width, 2)
            delta = flat_coords - anchor_pos.unsqueeze(2)
            sigma2 = max(self.position_bias_sigma * self.position_bias_sigma, 1.0e-6)
            pos_bias = -delta.square().sum(dim=-1) / (2.0 * sigma2)
            logits = logits + self.position_bias_strength * pos_bias

        prob = torch.softmax(logits, dim=-1)
        topk = min(self.topk, prob.shape[-1])
        scores, indices = torch.topk(prob, k=topk, dim=-1)

        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        if height > 1:
            y = y.to(frame_feat.dtype) / float(height - 1)
        else:
            y = torch.zeros_like(y, dtype=frame_feat.dtype)
        if width > 1:
            x = x.to(frame_feat.dtype) / float(width - 1)
        else:
            x = torch.zeros_like(x, dtype=frame_feat.dtype)
        points = torch.stack([y, x], dim=-1)

        grid_y = torch.linspace(0.0, 1.0, steps=height, device=frame_feat.device, dtype=frame_feat.dtype)
        grid_x = torch.linspace(0.0, 1.0, steps=width, device=frame_feat.device, dtype=frame_feat.dtype)
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        flat_coords = torch.stack([mesh_y, mesh_x], dim=-1).reshape(1, 1, height * width, 2)
        expected_points = (prob.unsqueeze(-1) * flat_coords).sum(dim=2)

        confidence = scores[..., 0]
        entropy = normalized_entropy(prob, dim=-1)
        heatmap = prob.view(batch, num_points, height, width)
        return GlobalRelocOutput(
            points=points,
            expected_points=expected_points,
            scores=scores,
            confidence=confidence,
            entropy=entropy,
            heatmap=heatmap,
        )
