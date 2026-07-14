"""Learned hypothesis scoring primitives for Route-D.

The scorer is independent from the legacy selector. It predicts a categorical
score over already-generated candidate hypotheses and can be trained from
frozen candidate caches without updating the MMP backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


HYPOTHESIS_FEATURE_DIM = 12
HYPOTHESIS_FEATURE_NAMES = (
    "candidate_quality",
    "candidate_entropy",
    "distance_to_local",
    "distance_to_previous",
    "quality_gap_to_local",
    "previous_confidence",
    "is_global",
    "normalized_candidate_rank",
    "exp_negative_distance_to_local",
    "exp_negative_distance_to_previous",
    "quality_above_local",
    "bias",
)


class HypothesisScorer(nn.Module):
    def __init__(self, feature_dim: int = HYPOTHESIS_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, hypothesis_features: torch.Tensor) -> torch.Tensor:
        if hypothesis_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {hypothesis_features.shape[-1]}"
            )
        return self.network(hypothesis_features).squeeze(-1)


def build_hypothesis_features(
    candidate_points: torch.Tensor,
    candidate_quality: torch.Tensor,
    local_points: torch.Tensor,
    previous_points: torch.Tensor,
    previous_confidence: torch.Tensor,
    candidate_entropy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build candidate-level features with output shape ``(..., K, 12)``."""
    if candidate_points.shape[:-1] != candidate_quality.shape:
        raise ValueError("candidate_quality must match candidate_points without xy")
    if local_points.shape != candidate_points.shape[:-2] + (2,):
        raise ValueError("local_points shape mismatch")
    if previous_points.shape != local_points.shape:
        raise ValueError("previous_points shape mismatch")
    if previous_confidence.shape != local_points.shape[:-1]:
        raise ValueError("previous_confidence shape mismatch")

    local_delta = torch.norm(candidate_points - local_points.unsqueeze(-2), dim=-1)
    motion_delta = torch.norm(candidate_points - previous_points.unsqueeze(-2), dim=-1)
    local_quality = candidate_quality[..., :1].expand_as(candidate_quality)
    confidence_gap = candidate_quality - local_quality
    prior_conf = previous_confidence.unsqueeze(-1).expand_as(candidate_quality)
    if candidate_entropy is None:
        candidate_entropy = torch.zeros_like(candidate_quality)
    if candidate_entropy.shape != candidate_quality.shape:
        raise ValueError("candidate_entropy shape mismatch")

    k = candidate_points.shape[-2]
    rank = torch.arange(k, device=candidate_points.device, dtype=candidate_points.dtype)
    rank = rank / float(max(k - 1, 1))
    rank = rank.view(*([1] * (candidate_quality.ndim - 1)), k).expand_as(candidate_quality)
    is_global = (rank > 0).to(candidate_quality.dtype)

    features = torch.stack(
        [
            candidate_quality,
            candidate_entropy,
            local_delta,
            motion_delta,
            confidence_gap,
            prior_conf,
            is_global,
            rank,
            torch.exp(-local_delta),
            torch.exp(-motion_delta),
            (candidate_quality > local_quality).to(candidate_quality.dtype),
            torch.ones_like(candidate_quality),
        ],
        dim=-1,
    )
    return features


def oracle_candidate_target(
    candidate_points: torch.Tensor,
    gt_points: torch.Tensor,
    margin: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best candidate index and whether a global mode beats local."""
    error = torch.norm(candidate_points - gt_points.unsqueeze(-2), dim=-1)
    best_error, best_index = error.min(dim=-1)
    global_improves_local = (best_index > 0) & (best_error + float(margin) < error[..., 0])
    return best_index, global_improves_local


def hypothesis_ranking_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked categorical cross entropy over candidate hypotheses."""
    if logits.shape[:-1] != target_index.shape or target_index.shape != valid_mask.shape:
        raise ValueError("logits, target_index, and valid_mask shapes are incompatible")
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -log_prob.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
    weight = valid_mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
