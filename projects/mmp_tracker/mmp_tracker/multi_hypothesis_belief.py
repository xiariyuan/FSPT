"""Explicit multi-hypothesis belief contract for Route-D MMP tracking.

The existing tracker already produces local and global candidates, but those
candidates previously existed only as transient routing tensors.  This module
turns them into an auditable belief object without forcing a weighted-average
trajectory.  It deliberately separates:

1. candidate locations;
2. normalized hypothesis weights;
3. belief diagnostics; and
4. a conservative collapse decision.

The collapse helper is diagnostic by default.  Callers may keep all hypotheses
alive even when ``collapse_mask`` is true, and the current MMP integration does
not change the tracker output trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MultiHypothesisBelief:
    """A normalized categorical belief over point-location hypotheses.

    Shapes use ``(..., K, 2)`` for points and ``(..., K)`` for all candidate
    attributes.  Invalid candidates have zero probability.
    """

    points: torch.Tensor
    weights: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.points.ndim < 2 or self.points.shape[-1] != 2:
            raise ValueError("points must have shape (..., K, 2)")
        if self.weights.shape != self.points.shape[:-1]:
            raise ValueError("weights must have shape points.shape[:-1]")
        if self.valid_mask.shape != self.weights.shape:
            raise ValueError("valid_mask must match weights")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be boolean")
        if not torch.isfinite(self.points).all():
            raise ValueError("points contain non-finite values")
        if not torch.isfinite(self.weights).all():
            raise ValueError("weights contain non-finite values")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")
        invalid_mass = self.weights.masked_select(~self.valid_mask)
        if invalid_mass.numel() and not torch.allclose(
            invalid_mass, torch.zeros_like(invalid_mass), atol=1e-7, rtol=0.0
        ):
            raise ValueError("invalid hypotheses must have zero weight")
        total = self.weights.sum(dim=-1)
        has_valid = self.valid_mask.any(dim=-1)
        if has_valid.any() and not torch.allclose(
            total[has_valid], torch.ones_like(total[has_valid]), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("valid belief rows must sum to one")
        if (~has_valid).any() and not torch.allclose(
            total[~has_valid], torch.zeros_like(total[~has_valid]), atol=1e-7, rtol=0.0
        ):
            raise ValueError("empty belief rows must have zero total mass")


@dataclass(frozen=True)
class BeliefDiagnostics:
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    effective_hypotheses: torch.Tensor
    valid_count: torch.Tensor
    top1_index: torch.Tensor
    top1_weight: torch.Tensor
    top1_margin: torch.Tensor
    map_points: torch.Tensor
    expected_points: torch.Tensor


@dataclass(frozen=True)
class BeliefCollapseDecision:
    collapse_mask: torch.Tensor
    selected_points: torch.Tensor
    diagnostics: BeliefDiagnostics


def belief_from_probabilities(
    points: torch.Tensor,
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> MultiHypothesisBelief:
    """Create a normalized belief from non-negative candidate probabilities."""
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape (..., K, 2)")
    if probabilities.shape != points.shape[:-1]:
        raise ValueError("probabilities must have shape points.shape[:-1]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if (probabilities < 0).any():
        raise ValueError("probabilities must be non-negative")
    if valid_mask is None:
        valid_mask = torch.ones_like(probabilities, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=probabilities.device, dtype=torch.bool)
        if valid_mask.shape != probabilities.shape:
            raise ValueError("valid_mask must match probabilities")

    masked = probabilities * valid_mask.to(probabilities.dtype)
    denom = masked.sum(dim=-1, keepdim=True)
    has_valid_mass = denom > eps
    valid_count = valid_mask.sum(dim=-1, keepdim=True)
    uniform = valid_mask.to(probabilities.dtype) / valid_count.clamp_min(1).to(
        probabilities.dtype
    )
    normalized = torch.where(
        has_valid_mass,
        masked / denom.clamp_min(eps),
        torch.where(valid_count > 0, uniform, torch.zeros_like(masked)),
    )
    return MultiHypothesisBelief(
        points=points,
        weights=normalized,
        valid_mask=valid_mask,
    )


def belief_from_logits(
    points: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> MultiHypothesisBelief:
    """Create a belief from masked categorical logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape != points.shape[:-1]:
        raise ValueError("logits must have shape points.shape[:-1]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        if valid_mask.shape != logits.shape:
            raise ValueError("valid_mask must match logits")

    masked_logits = torch.where(
        valid_mask,
        logits / float(temperature),
        torch.full_like(logits, -torch.inf),
    )
    has_valid = valid_mask.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits, dim=-1)
    probabilities = torch.where(
        valid_mask & has_valid,
        probabilities,
        torch.zeros_like(probabilities),
    )
    return belief_from_probabilities(points, probabilities, valid_mask)


def diagnose_belief(
    belief: MultiHypothesisBelief,
    *,
    eps: float = 1e-8,
) -> BeliefDiagnostics:
    weights = belief.weights
    valid_count = belief.valid_mask.sum(dim=-1)
    entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
    max_entropy = valid_count.clamp_min(1).to(weights.dtype).log()
    normalized_entropy = torch.where(
        valid_count > 1,
        entropy / max_entropy.clamp_min(eps),
        torch.zeros_like(entropy),
    )
    effective_hypotheses = torch.where(
        valid_count > 0,
        entropy.exp(),
        torch.zeros_like(entropy),
    )

    top1_weight, top1_index = weights.max(dim=-1)
    if weights.shape[-1] > 1:
        top2_weight = torch.topk(weights, k=2, dim=-1).values[..., 1]
    else:
        top2_weight = torch.zeros_like(top1_weight)
    top1_margin = top1_weight - top2_weight
    gather_index = top1_index.unsqueeze(-1).unsqueeze(-1).expand(
        *top1_index.shape, 1, 2
    )
    map_points = belief.points.gather(-2, gather_index).squeeze(-2)
    expected_points = (belief.points * weights.unsqueeze(-1)).sum(dim=-2)
    has_valid = valid_count > 0
    map_points = torch.where(has_valid.unsqueeze(-1), map_points, torch.zeros_like(map_points))
    expected_points = torch.where(
        has_valid.unsqueeze(-1), expected_points, torch.zeros_like(expected_points)
    )
    return BeliefDiagnostics(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        effective_hypotheses=effective_hypotheses,
        valid_count=valid_count,
        top1_index=top1_index,
        top1_weight=top1_weight,
        top1_margin=top1_margin,
        map_points=map_points,
        expected_points=expected_points,
    )


def conservative_collapse(
    belief: MultiHypothesisBelief,
    *,
    min_top1_weight: float = 0.65,
    max_normalized_entropy: float = 0.45,
    min_top1_margin: float = 0.20,
) -> BeliefCollapseDecision:
    """Return a conservative MAP-collapse decision without discarding belief state.

    A single valid hypothesis always collapses.  Multi-hypothesis rows collapse
    only when all three concentration gates pass.  ``selected_points`` is the
    MAP point; callers should inspect ``collapse_mask`` before replacing a
    maintained belief with that point.
    """
    if not 0.0 <= min_top1_weight <= 1.0:
        raise ValueError("min_top1_weight must be in [0, 1]")
    if not 0.0 <= max_normalized_entropy <= 1.0:
        raise ValueError("max_normalized_entropy must be in [0, 1]")
    if not 0.0 <= min_top1_margin <= 1.0:
        raise ValueError("min_top1_margin must be in [0, 1]")

    diagnostics = diagnose_belief(belief)
    has_valid = diagnostics.valid_count > 0
    single = diagnostics.valid_count == 1
    concentrated = (
        (diagnostics.top1_weight >= min_top1_weight)
        & (diagnostics.normalized_entropy <= max_normalized_entropy)
        & (diagnostics.top1_margin >= min_top1_margin)
    )
    collapse_mask = has_valid & (single | concentrated)
    return BeliefCollapseDecision(
        collapse_mask=collapse_mask,
        selected_points=diagnostics.map_points,
        diagnostics=diagnostics,
    )
