"""Learned hypothesis scoring primitive for Route-D.

The scorer is intentionally independent from the legacy selector. It predicts a
categorical distribution over already-generated candidate hypotheses. Training
code can supervise it with oracle candidate assignments without changing the
MMP backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


HYPOTHESIS_FEATURE_DIM = 12


class HypothesisScorer(nn.Module):
    def __init__(self, feature_dim: int = HYPOTHESIS_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
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
    """Build candidate-level features.

    Output shape: ``(..., K, 12)``.
    """
    if candidate_points.shape[:-1] != candidate_quality.shape:
        raise ValueError("candidate_quality must match candidate_points without xy")
    if local_points.shape != candidate_points.shape[:-2] + (2,):
        raise ValueError("local_points shape mismatch")
    if previous_points.shape != local_points.shape:
        raise ValueError("previous_points shape mismatch")
    local_delta = torch.norm(candidate_points - local_points.unsqueeze(-2), dim=-1)
    motion_delta = torch.norm(candidate_points - previous_points.unsqueeze(-2), dim=-1)
    local_quality = candidate_quality[..., :1].expand_as(candidate_quality)
    confidence_gap = candidate_quality - local_quality
    prior_conf = previous_confidence.unsqueeze(-1).expand_as(candidate_quality)
    if candidate_entropy is None:
        candidate_entropy = torch.zeros_like(candidate_quality)
    features = torch.stack(
        [
            candidate_quality,
            candidate_entropy,
            local_delta,
            motion_delta,
            confidence_gap,
            prior_conf,
            torch.cos(local_delta),
            torch.exp(-local_delta),
            torch.exp(-motion_delta),
            (candidate_quality > local_quality).to(candidate_quality.dtype),
            torch.ones_like(candidate_quality),
            torch.zeros_like(candidate_quality),
        ],
        dim=-1,
    )
    return features


def oracle_candidate_target(
    candidate_points: torch.Tensor,
    gt_points: torch.Tensor,
    margin: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return oracle candidate indices and validity mask for training diagnostics."""
    error = torch.norm(candidate_points - gt_points.unsqueeze(-2), dim=-1)
    best_error, best_index = error.min(dim=-1)
    valid = best_error + float(margin) < error[..., 0]
    return best_index, valid


def hypothesis_ranking_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy over candidate hypotheses."""
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -log_prob.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
    weight = valid_mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
