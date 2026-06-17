from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PosteriorFusionOutput:
    points: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    hypothesis_weights: torch.Tensor
    candidate_points: torch.Tensor


class PosteriorFusionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        type_embed_dim: int = 8,
        min_visibility: float = 0.0,
        max_offset: float = 0.08,
        prior_residual_mix: float = 0.7,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(3, type_embed_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(6 + type_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.min_visibility = float(min_visibility)
        self.max_offset = float(max_offset)
        self.prior_residual_mix = float(prior_residual_mix)

    def forward(
        self,
        prior_points: torch.Tensor,
        prior_conf: torch.Tensor,
        local_points: torch.Tensor,
        local_conf: torch.Tensor,
        local_entropy: torch.Tensor,
        global_points: torch.Tensor,
        global_scores: torch.Tensor,
        global_entropy: torch.Tensor,
    ) -> PosteriorFusionOutput:
        batch, num_points, topk, _ = global_points.shape

        candidate_points = torch.cat(
            [prior_points.unsqueeze(2), local_points.unsqueeze(2), global_points],
            dim=2,
        )
        candidate_scores = torch.cat(
            [prior_conf.unsqueeze(2), local_conf.unsqueeze(2), global_scores],
            dim=2,
        )
        zero_entropy = torch.zeros_like(local_entropy)
        candidate_entropy = torch.cat(
            [
                zero_entropy.unsqueeze(2),
                local_entropy.unsqueeze(2),
                global_entropy.unsqueeze(2).expand(batch, num_points, topk),
            ],
            dim=2,
        )

        dist_prior = torch.norm(candidate_points - prior_points.unsqueeze(2), dim=-1)
        dist_local = torch.norm(candidate_points - local_points.unsqueeze(2), dim=-1)
        agreement = 1.0 - (dist_prior + dist_local) * 0.5
        candidate_quality = candidate_scores * (1.0 - candidate_entropy.clamp(0.0, 1.0))
        raw_features = torch.stack(
            [candidate_scores, candidate_quality, candidate_entropy, dist_prior, dist_local, agreement],
            dim=-1,
        )

        type_ids = torch.zeros(batch, num_points, topk + 2, device=global_points.device, dtype=torch.long)
        type_ids[:, :, 1] = 1
        if topk > 0:
            type_ids[:, :, 2:] = 2
        type_embed = self.type_embedding(type_ids)
        logits = self.score_mlp(torch.cat([raw_features, type_embed], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)

        fused_candidate = (candidate_points * weights.unsqueeze(-1)).sum(dim=2)
        local_prior_mid = self.prior_residual_mix * prior_points + (1.0 - self.prior_residual_mix) * local_points
        residual = (fused_candidate - local_prior_mid).clamp(min=-self.max_offset, max=self.max_offset)
        fused_points = (local_prior_mid + residual).clamp(0.0, 1.0)
        confidence = (candidate_quality * weights).sum(dim=2)
        best_global = global_scores[..., 0] if topk > 0 else confidence
        vis_features = torch.stack(
            [confidence, prior_conf, local_conf, best_global, local_entropy, global_entropy],
            dim=-1,
        )
        visibility = torch.sigmoid(self.visibility_head(vis_features)).squeeze(-1)
        visibility = visibility.clamp(min=self.min_visibility, max=1.0)
        return PosteriorFusionOutput(
            points=fused_points,
            visibility=visibility,
            confidence=confidence,
            hypothesis_weights=weights,
            candidate_points=candidate_points,
        )
