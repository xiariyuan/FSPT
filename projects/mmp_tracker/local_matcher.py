from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import local_offset_grid, normalized_entropy, yx_to_grid_xy


@dataclass
class LocalMatchOutput:
    points: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    heatmap: torch.Tensor


class LocalMatcher(nn.Module):
    def __init__(self, radius: int = 4, temperature: float = 0.07):
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)

    def forward(
        self,
        frame_feat: torch.Tensor,
        query_feat: torch.Tensor,
        prior_points: torch.Tensor,
    ) -> LocalMatchOutput:
        batch, channels, height, width = frame_feat.shape
        num_points = query_feat.shape[1]
        kernel = 2 * self.radius + 1

        feat_bn = frame_feat.unsqueeze(1).expand(batch, num_points, channels, height, width)
        feat_bn = feat_bn.reshape(batch * num_points, channels, height, width)

        offsets = local_offset_grid(self.radius, height, width, frame_feat.device, frame_feat.dtype)
        offsets = offsets.view(1, kernel, kernel, 2)
        centers = prior_points.reshape(batch * num_points, 2)
        grid_yx = (centers[:, None, None, :] + offsets).clamp(0.0, 1.0)
        grid_xy = yx_to_grid_xy(grid_yx)

        patch = F.grid_sample(feat_bn, grid_xy, mode="bilinear", padding_mode="border", align_corners=True)
        patch = F.normalize(patch, dim=1)
        query = F.normalize(query_feat, dim=-1).reshape(batch * num_points, channels, 1, 1)

        corr = (patch * query).sum(dim=1) / self.temperature
        prob = torch.softmax(corr.flatten(1), dim=-1)
        offsets_flat = offsets.view(1, kernel * kernel, 2)
        delta = (prob.unsqueeze(-1) * offsets_flat).sum(dim=1)

        matched_points = (centers + delta).clamp(0.0, 1.0).view(batch, num_points, 2)
        confidence = prob.max(dim=-1).values.view(batch, num_points)
        entropy = normalized_entropy(prob, dim=-1).view(batch, num_points)
        heatmap = prob.view(batch, num_points, kernel, kernel)
        return LocalMatchOutput(points=matched_points, confidence=confidence, entropy=entropy, heatmap=heatmap)


class PatchMatcher(nn.Module):
    def __init__(self, radius: int = 4, temperature: float = 0.07, template_reduce: str = "logmeanexp"):
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.template_reduce = str(template_reduce or "logmeanexp").strip().lower()

    def forward(
        self,
        frame_feat: torch.Tensor,
        template_tokens: torch.Tensor,
        prior_points: torch.Tensor,
    ) -> LocalMatchOutput:
        if template_tokens.ndim != 4:
            raise ValueError("template_tokens must have shape (B, N, T, C).")

        batch, channels, height, width = frame_feat.shape
        num_points = template_tokens.shape[1]
        num_tokens = template_tokens.shape[2]
        if num_tokens <= 0:
            raise ValueError("template_tokens must have a positive token dimension.")
        kernel = 2 * self.radius + 1

        feat_bn = frame_feat.unsqueeze(1).expand(batch, num_points, channels, height, width)
        feat_bn = feat_bn.reshape(batch * num_points, channels, height, width)

        offsets = local_offset_grid(self.radius, height, width, frame_feat.device, frame_feat.dtype)
        offsets = offsets.view(1, kernel, kernel, 2)
        centers = prior_points.reshape(batch * num_points, 2)
        grid_yx = (centers[:, None, None, :] + offsets).clamp(0.0, 1.0)
        grid_xy = yx_to_grid_xy(grid_yx)

        patch = F.grid_sample(feat_bn, grid_xy, mode="bilinear", padding_mode="border", align_corners=True)
        patch = F.normalize(patch, dim=1).flatten(2)

        tokens = F.normalize(template_tokens, dim=-1).reshape(batch * num_points, num_tokens, channels)
        corr = torch.einsum("btc,bck->btk", tokens, patch) / self.temperature
        if self.template_reduce in {"max", "amax"}:
            corr = corr.max(dim=1).values
        else:
            corr = torch.logsumexp(corr, dim=1)
            if self.template_reduce in {"logmeanexp", "mean"}:
                corr = corr - corr.new_tensor(float(num_tokens)).log()

        prob = torch.softmax(corr, dim=-1)
        offsets_flat = offsets.view(1, kernel * kernel, 2)
        delta = (prob.unsqueeze(-1) * offsets_flat).sum(dim=1)

        matched_points = (centers + delta).clamp(0.0, 1.0).view(batch, num_points, 2)
        confidence = prob.max(dim=-1).values.view(batch, num_points)
        entropy = normalized_entropy(prob, dim=-1).view(batch, num_points)
        heatmap = prob.view(batch, num_points, kernel, kernel)
        return LocalMatchOutput(points=matched_points, confidence=confidence, entropy=entropy, heatmap=heatmap)
