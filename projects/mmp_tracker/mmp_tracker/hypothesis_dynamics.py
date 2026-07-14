"""Candidate generation and temporal belief propagation for Route-D MMP.

This module converts local/global tracker proposals into a single explicit
hypothesis set and propagates categorical belief mass across time using a
spatial transition kernel.  It is designed to preserve multiple plausible
modes rather than immediately averaging them into one point.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .multi_hypothesis_belief import (
    MultiHypothesisBelief,
    belief_from_logits,
)

LOCAL_SOURCE_ID = 0
GLOBAL_SOURCE_ID = 1


@dataclass(frozen=True)
class HypothesisCandidates:
    """Unified local/global candidate set.

    ``points`` has shape ``(..., K, 2)``. ``evidence_logits``, ``valid_mask``
    and ``source_ids`` have shape ``(..., K)``.
    """

    points: torch.Tensor
    evidence_logits: torch.Tensor
    valid_mask: torch.Tensor
    source_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.points.ndim < 2 or self.points.shape[-1] != 2:
            raise ValueError("points must have shape (..., K, 2)")
        expected = self.points.shape[:-1]
        if self.evidence_logits.shape != expected:
            raise ValueError("evidence_logits must match points.shape[:-1]")
        if self.valid_mask.shape != expected:
            raise ValueError("valid_mask must match points.shape[:-1]")
        if self.source_ids.shape != expected:
            raise ValueError("source_ids must match points.shape[:-1]")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be boolean")
        if not torch.isfinite(self.points).all():
            raise ValueError("points contain non-finite values")
        if not torch.isfinite(self.evidence_logits).all():
            raise ValueError("evidence_logits contain non-finite values")


@dataclass(frozen=True)
class TemporalBeliefUpdate:
    belief: MultiHypothesisBelief
    evidence_belief: MultiHypothesisBelief
    predicted_prior: torch.Tensor
    transition_probabilities: torch.Tensor


def generate_local_global_hypotheses(
    local_points: torch.Tensor,
    local_evidence: torch.Tensor,
    global_points: torch.Tensor,
    global_evidence: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    min_evidence: float = 1e-8,
) -> HypothesisCandidates:
    """Concatenate one local candidate and K global candidates.

    Evidence inputs are non-negative quality/probability-like scores. They are
    converted to log evidence after clamping, while candidate validity remains
    controlled by finiteness and ``active_mask`` rather than score magnitude.
    """
    if min_evidence <= 0:
        raise ValueError("min_evidence must be positive")
    if local_points.shape[-1] != 2:
        raise ValueError("local_points must end in dimension 2")
    if local_evidence.shape != local_points.shape[:-1]:
        raise ValueError("local_evidence must match local_points.shape[:-1]")
    if global_points.ndim != local_points.ndim + 1 or global_points.shape[-1] != 2:
        raise ValueError("global_points must have shape local_points.shape[:-1] + (K, 2)")
    if global_points.shape[:-2] != local_points.shape[:-1]:
        raise ValueError("global_points leading dimensions must match local_points")
    if global_evidence.shape != global_points.shape[:-1]:
        raise ValueError("global_evidence must match global_points.shape[:-1]")
    if (local_evidence < 0).any() or (global_evidence < 0).any():
        raise ValueError("evidence must be non-negative")
    if not torch.isfinite(local_evidence).all() or not torch.isfinite(global_evidence).all():
        raise ValueError("evidence contains non-finite values")

    points = torch.cat([local_points.unsqueeze(-2), global_points], dim=-2)
    evidence = torch.cat([local_evidence.unsqueeze(-1), global_evidence], dim=-1)
    evidence_logits = evidence.clamp_min(float(min_evidence)).log()

    local_valid = torch.isfinite(local_points).all(dim=-1).unsqueeze(-1)
    global_valid = torch.isfinite(global_points).all(dim=-1)
    valid_mask = torch.cat([local_valid, global_valid], dim=-1)
    if active_mask is not None:
        if active_mask.shape != local_points.shape[:-1]:
            raise ValueError("active_mask must match local_points.shape[:-1]")
        valid_mask = valid_mask & active_mask.to(dtype=torch.bool, device=points.device).unsqueeze(-1)

    source_ids = torch.full(
        valid_mask.shape,
        GLOBAL_SOURCE_ID,
        device=points.device,
        dtype=torch.long,
    )
    source_ids[..., 0] = LOCAL_SOURCE_ID
    return HypothesisCandidates(
        points=points,
        evidence_logits=evidence_logits,
        valid_mask=valid_mask,
        source_ids=source_ids,
    )


def _masked_transition_probabilities(
    previous: MultiHypothesisBelief,
    candidates: HypothesisCandidates,
    *,
    transition_sigma: float,
) -> torch.Tensor:
    if transition_sigma <= 0:
        raise ValueError("transition_sigma must be positive")
    if previous.points.shape[:-2] != candidates.points.shape[:-2]:
        raise ValueError("previous belief and current candidates must share leading dimensions")

    delta = previous.points.unsqueeze(-2) - candidates.points.unsqueeze(-3)
    dist2 = delta.square().sum(dim=-1)
    transition_logits = -dist2 / (2.0 * float(transition_sigma) ** 2)

    valid_pair = previous.valid_mask.unsqueeze(-1) & candidates.valid_mask.unsqueeze(-2)
    masked_logits = torch.where(
        valid_pair,
        transition_logits,
        torch.full_like(transition_logits, -torch.inf),
    )
    has_destination = valid_pair.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_destination, masked_logits, torch.zeros_like(masked_logits))
    probabilities = torch.softmax(safe_logits, dim=-1)
    probabilities = torch.where(valid_pair & has_destination, probabilities, torch.zeros_like(probabilities))
    return probabilities


def update_temporal_belief(
    candidates: HypothesisCandidates,
    previous: MultiHypothesisBelief | None = None,
    *,
    transition_sigma: float = 0.08,
    prior_strength: float = 1.0,
    evidence_temperature: float = 1.0,
    birth_mass: float = 0.05,
    eps: float = 1e-8,
) -> TemporalBeliefUpdate:
    """Combine current candidate evidence with a propagated previous belief.

    Previous mass is transported to current candidates through a Gaussian
    spatial kernel. ``birth_mass`` reserves a small uniform prior for new modes,
    preventing temporal persistence from making hypothesis birth impossible.
    """
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    if evidence_temperature <= 0:
        raise ValueError("evidence_temperature must be positive")
    if not 0.0 <= birth_mass <= 1.0:
        raise ValueError("birth_mass must be in [0, 1]")

    evidence_belief = belief_from_logits(
        candidates.points,
        candidates.evidence_logits,
        candidates.valid_mask,
        temperature=float(evidence_temperature),
    )

    valid_count = candidates.valid_mask.sum(dim=-1, keepdim=True)
    uniform_birth = candidates.valid_mask.to(candidates.points.dtype) / valid_count.clamp_min(1).to(
        candidates.points.dtype
    )
    uniform_birth = torch.where(valid_count > 0, uniform_birth, torch.zeros_like(uniform_birth))

    if previous is None:
        empty_transition = torch.zeros(
            *candidates.points.shape[:-2],
            0,
            candidates.points.shape[-2],
            device=candidates.points.device,
            dtype=candidates.points.dtype,
        )
        return TemporalBeliefUpdate(
            belief=evidence_belief,
            evidence_belief=evidence_belief,
            predicted_prior=uniform_birth,
            transition_probabilities=empty_transition,
        )

    transition_probabilities = _masked_transition_probabilities(
        previous,
        candidates,
        transition_sigma=float(transition_sigma),
    )
    transported_prior = (
        previous.weights.unsqueeze(-1) * transition_probabilities
    ).sum(dim=-2)
    has_previous_mass = previous.weights.sum(dim=-1, keepdim=True) > eps
    predicted_prior = torch.where(
        has_previous_mass,
        (1.0 - float(birth_mass)) * transported_prior + float(birth_mass) * uniform_birth,
        uniform_birth,
    )
    predicted_prior = predicted_prior * candidates.valid_mask.to(predicted_prior.dtype)
    predicted_prior = predicted_prior / predicted_prior.sum(dim=-1, keepdim=True).clamp_min(eps)
    predicted_prior = torch.where(
        valid_count > 0,
        predicted_prior,
        torch.zeros_like(predicted_prior),
    )

    posterior_logits = (
        candidates.evidence_logits / float(evidence_temperature)
        + float(prior_strength) * predicted_prior.clamp_min(eps).log()
    )
    belief = belief_from_logits(
        candidates.points,
        posterior_logits,
        candidates.valid_mask,
    )
    return TemporalBeliefUpdate(
        belief=belief,
        evidence_belief=evidence_belief,
        predicted_prior=predicted_prior,
        transition_probabilities=transition_probabilities,
    )
