"""Trainable multi-hypothesis utility-conditioned state recovery for Route-D.

This module is deliberately stronger than the historical per-candidate MLP/tree
controller.  It jointly reasons over a causal candidate set, predicts a
monotonic multi-threshold correctness profile for every candidate, estimates
catastrophic risk, and emits an abstention-aware bounded state update.

Candidate index 0 is the native backbone continuation.  All other candidates
must be causally available alternatives.  The module never generates future
information and does not mutate a backbone internally; integration code applies
its returned write strengths to an audited tracker state interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_THRESHOLDS_PX = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_STATE_FIELDS = ("coordinate", "visibility", "confidence", "memory")


@dataclass(frozen=True)
class RecoveryNetworkConfig:
    """Architecture and safety contract for the recovery network."""

    candidate_feature_dim: int
    state_feature_dim: int
    hidden_dim: int = 128
    num_attention_heads: int = 4
    num_attention_layers: int = 2
    feedforward_multiplier: int = 4
    num_candidate_sources: int = 16
    thresholds_px: tuple[float, ...] = DEFAULT_THRESHOLDS_PX
    threshold_utility_weights: tuple[float, ...] = (0.28, 0.24, 0.20, 0.16, 0.12)
    catastrophe_risk_weight: float = 0.75
    max_contextual_selector_bias: float = 0.25
    coordinate_scale_px: float = 32.0
    max_coordinate_update_px: float = 24.0
    selection_temperature: float = 0.20
    dropout: float = 0.0
    state_fields: tuple[str, ...] = DEFAULT_STATE_FIELDS
    native_safe_initialization: bool = True
    initial_abstention_bias: float = 2.0
    initial_coordinate_write_bias: float = -2.0
    initial_other_write_bias: float = -4.0

    def __post_init__(self) -> None:
        if self.candidate_feature_dim <= 0 or self.state_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if self.num_attention_layers <= 0:
            raise ValueError("num_attention_layers must be positive")
        if self.num_candidate_sources <= 0:
            raise ValueError("num_candidate_sources must be positive")
        if not self.thresholds_px or any(value <= 0.0 for value in self.thresholds_px):
            raise ValueError("thresholds_px must be positive and non-empty")
        if tuple(sorted(self.thresholds_px)) != tuple(self.thresholds_px):
            raise ValueError("thresholds_px must be sorted in ascending order")
        if len(self.threshold_utility_weights) != len(self.thresholds_px):
            raise ValueError("threshold_utility_weights must match thresholds_px")
        if any(value < 0.0 for value in self.threshold_utility_weights):
            raise ValueError("threshold_utility_weights must be non-negative")
        if sum(self.threshold_utility_weights) <= 0.0:
            raise ValueError("threshold_utility_weights must have positive mass")
        if self.max_contextual_selector_bias < 0.0:
            raise ValueError("max_contextual_selector_bias must be non-negative")
        if self.coordinate_scale_px <= 0.0 or self.max_coordinate_update_px <= 0.0:
            raise ValueError("coordinate scales must be positive")
        if self.selection_temperature <= 0.0:
            raise ValueError("selection_temperature must be positive")
        if not self.state_fields:
            raise ValueError("state_fields must be non-empty")


@dataclass(frozen=True)
class RecoveryLossConfig:
    """Training objective weights for utility, ranking, risk, and safe rollout."""

    threshold_bce_weight: float = 1.0
    ranking_weight: float = 0.50
    catastrophe_weight: float = 0.50
    coordinate_weight: float = 1.0
    no_harm_weight: float = 1.0
    abstention_weight: float = 0.50
    tail_weight: float = 0.25
    catastrophe_threshold_px: float = 16.0
    tail_start_px: float = 8.0
    min_candidate_gain_px: float = 0.50
    no_harm_margin_px: float = 0.0


class MultiHypothesisStateRecoveryNetwork(nn.Module):
    """Joint candidate reasoning and risk-aware bounded state recovery.

    Inputs use shape ``(B, P, K, *)`` where ``P`` is the number of tracked
    points and ``K`` is the causal candidate count. Candidate 0 is always the
    native backbone continuation. The model is permutation-equivariant over the
    non-native candidates because it uses no candidate-rank positional encoding.
    Candidate source identity is supplied explicitly through ``source_ids``.
    """

    def __init__(self, config: RecoveryNetworkConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        threshold_count = len(config.thresholds_px)

        self.candidate_feature_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_feature_dim),
            nn.Linear(config.candidate_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_feature_dim),
            nn.Linear(config.state_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.source_embedding = nn.Embedding(config.num_candidate_sources, hidden)
        self.native_role_embedding = nn.Parameter(torch.zeros(1, 1, hidden))
        self.candidate_role_embedding = nn.Parameter(torch.zeros(1, 1, hidden))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_attention_heads,
            dim_feedforward=hidden * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_attention_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )

        self.threshold_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, threshold_count),
        )
        self.catastrophe_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.selector_bias_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        decision_dim = hidden * 3
        self.abstention_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.state_write_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(config.state_fields)),
        )

        weights = torch.tensor(config.threshold_utility_weights, dtype=torch.float32)
        weights = weights / weights.sum()
        self.register_buffer("threshold_utility_weights", weights, persistent=True)

        nn.init.normal_(self.native_role_embedding, std=0.02)
        nn.init.normal_(self.candidate_role_embedding, std=0.02)
        if config.native_safe_initialization:
            self._initialize_native_safe_heads()

    @staticmethod
    def _zero_final_linear(module: nn.Sequential, *, bias: float = 0.0) -> None:
        final = module[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("expected final Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(bias))

    def _initialize_native_safe_heads(self) -> None:
        """Start at native selection with conservative state writes.

        Equal candidate scores make deterministic argmax choose candidate 0.
        The coordinate write is initially small and abstention is initially high,
        but exact zero-step parity comes from selecting the native coordinate.
        """
        self._zero_final_linear(self.threshold_head, bias=0.0)
        self._zero_final_linear(self.catastrophe_head, bias=0.0)
        self._zero_final_linear(self.selector_bias_head, bias=0.0)
        self._zero_final_linear(
            self.abstention_head, bias=float(self.config.initial_abstention_bias)
        )
        final = self.state_write_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("expected final state-write Linear layer")
        nn.init.zeros_(final.weight)
        bias = torch.full(
            (len(self.config.state_fields),),
            float(self.config.initial_other_write_bias),
        )
        bias[0] = float(self.config.initial_coordinate_write_bias)
        with torch.no_grad():
            final.bias.copy_(bias)

    @staticmethod
    def _validate_inputs(
        candidate_features: torch.Tensor,
        candidate_coords_px: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        state_features: torch.Tensor,
        source_ids: torch.Tensor,
        config: RecoveryNetworkConfig,
    ) -> tuple[int, int, int]:
        if candidate_features.ndim != 4:
            raise ValueError("candidate_features must have shape (B,P,K,F)")
        batch, points, candidates, feature_dim = candidate_features.shape
        if feature_dim != config.candidate_feature_dim:
            raise ValueError(
                f"expected candidate feature dim {config.candidate_feature_dim}, got {feature_dim}"
            )
        if candidate_coords_px.shape != (batch, points, candidates, 2):
            raise ValueError("candidate_coords_px must have shape (B,P,K,2)")
        if candidate_valid_mask.shape != (batch, points, candidates):
            raise ValueError("candidate_valid_mask must have shape (B,P,K)")
        if state_features.shape != (batch, points, config.state_feature_dim):
            raise ValueError("state_features must have shape (B,P,S)")
        if source_ids.shape != (batch, points, candidates):
            raise ValueError("source_ids must have shape (B,P,K)")
        if candidates <= 0:
            raise ValueError("at least the native candidate is required")
        if not torch.all(candidate_valid_mask[..., 0]):
            raise ValueError("candidate 0 is the native continuation and must always be valid")
        if source_ids.numel() and (
            int(source_ids.min().item()) < 0
            or int(source_ids.max().item()) >= config.num_candidate_sources
        ):
            raise ValueError("source_ids are outside the configured embedding range")
        return batch, points, candidates

    @staticmethod
    def _monotonic_threshold_logits(raw: torch.Tensor) -> torch.Tensor:
        """Parameterize non-decreasing threshold probabilities by construction."""
        first = raw[..., :1]
        if raw.shape[-1] == 1:
            return first
        increments = F.softplus(raw[..., 1:])
        outputs = [first]
        running = first
        for index in range(increments.shape[-1]):
            running = running + increments[..., index : index + 1]
            outputs.append(running)
        return torch.cat(outputs, dim=-1)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_coords_px: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        state_features: torch.Tensor,
        source_ids: torch.Tensor,
        *,
        use_hard_selection: bool = False,
        use_straight_through_selection: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch, points, candidates = self._validate_inputs(
            candidate_features,
            candidate_coords_px,
            candidate_valid_mask,
            state_features,
            source_ids,
            self.config,
        )
        flat = batch * points
        valid = candidate_valid_mask.reshape(flat, candidates).bool()
        coords = candidate_coords_px.reshape(flat, candidates, 2)
        native_coord = coords[:, 0]
        relative_coord = (coords - native_coord.unsqueeze(1)) / float(
            self.config.coordinate_scale_px
        )
        relative_norm = torch.linalg.vector_norm(relative_coord, dim=-1, keepdim=True)
        coordinate_input = torch.cat([relative_coord, relative_norm, relative_norm.square()], dim=-1)

        candidate_tokens = self.candidate_feature_projection(
            candidate_features.reshape(flat, candidates, self.config.candidate_feature_dim)
        )
        candidate_tokens = candidate_tokens + self.coordinate_projection(coordinate_input)
        candidate_tokens = candidate_tokens + self.source_embedding(
            source_ids.reshape(flat, candidates).long()
        )
        role = self.candidate_role_embedding.expand(flat, candidates, -1).clone()
        role[:, :1] = self.native_role_embedding.expand(flat, 1, -1)
        candidate_tokens = candidate_tokens + role

        state_token = self.state_projection(
            state_features.reshape(flat, self.config.state_feature_dim)
        ).unsqueeze(1)
        sequence = torch.cat([state_token, candidate_tokens], dim=1)
        padding_mask = torch.cat(
            [torch.zeros((flat, 1), dtype=torch.bool, device=valid.device), ~valid],
            dim=1,
        )
        encoded = self.candidate_set_encoder(sequence, src_key_padding_mask=padding_mask)
        encoded_state = encoded[:, 0]
        encoded_candidates = encoded[:, 1:]

        raw_threshold_logits = self.threshold_head(encoded_candidates)
        threshold_logits = self._monotonic_threshold_logits(raw_threshold_logits)
        threshold_probabilities = torch.sigmoid(threshold_logits)
        expected_utility = (
            threshold_probabilities * self.threshold_utility_weights.to(
                threshold_probabilities.dtype
            )
        ).sum(dim=-1)
        catastrophe_logit = self.catastrophe_head(encoded_candidates).squeeze(-1)
        catastrophe_probability = torch.sigmoid(catastrophe_logit)
        selector_bias = float(self.config.max_contextual_selector_bias) * torch.tanh(
            self.selector_bias_head(encoded_candidates).squeeze(-1)
        )
        candidate_score = (
            expected_utility
            - float(self.config.catastrophe_risk_weight) * catastrophe_probability
            + selector_bias
        )
        candidate_score = candidate_score.masked_fill(~valid, torch.finfo(candidate_score.dtype).min)
        candidate_probability = torch.softmax(
            candidate_score / float(self.config.selection_temperature), dim=-1
        )
        candidate_probability = candidate_probability * valid.to(candidate_probability.dtype)
        candidate_probability = candidate_probability / candidate_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        selected_candidate_index = candidate_score.argmax(dim=-1)
        if use_hard_selection and use_straight_through_selection:
            raise ValueError(
                "hard selection and straight-through selection are mutually exclusive"
            )

        if use_hard_selection:
            selection_weight = F.one_hot(
                selected_candidate_index, num_classes=candidates
            ).to(candidate_probability.dtype)
        elif use_straight_through_selection:
            hard_weight = F.one_hot(
                selected_candidate_index, num_classes=candidates
            ).to(candidate_probability.dtype)
            selection_weight = hard_weight + candidate_probability - candidate_probability.detach()
        else:
            selection_weight = candidate_probability
        selected_coord = (selection_weight.unsqueeze(-1) * coords).sum(dim=1)
        selected_context = (
            selection_weight.unsqueeze(-1) * encoded_candidates
        ).sum(dim=1)

        decision_context = torch.cat(
            [encoded_state, selected_context, encoded_candidates[:, 0]], dim=-1
        )
        abstention_logit = self.abstention_head(decision_context).squeeze(-1)
        abstention_probability = torch.sigmoid(abstention_logit)
        state_write_strength = torch.sigmoid(self.state_write_head(decision_context))
        state_write_strength = state_write_strength * (1.0 - abstention_probability).unsqueeze(-1)

        raw_delta = selected_coord - native_coord
        raw_delta_norm = torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True)
        trust_scale = torch.clamp(
            float(self.config.max_coordinate_update_px) / raw_delta_norm.clamp_min(1.0e-12),
            max=1.0,
        )
        bounded_delta = raw_delta * trust_scale
        coordinate_strength = state_write_strength[:, 0:1]
        updated_coord = native_coord + coordinate_strength * bounded_delta

        output_shape = (batch, points)
        return {
            "threshold_logits": threshold_logits.reshape(
                batch, points, candidates, len(self.config.thresholds_px)
            ),
            "threshold_probabilities": threshold_probabilities.reshape(
                batch, points, candidates, len(self.config.thresholds_px)
            ),
            "catastrophe_logit": catastrophe_logit.reshape(batch, points, candidates),
            "catastrophe_probability": catastrophe_probability.reshape(
                batch, points, candidates
            ),
            "expected_utility": expected_utility.reshape(batch, points, candidates),
            "candidate_score": candidate_score.reshape(batch, points, candidates),
            "candidate_probability": candidate_probability.reshape(
                batch, points, candidates
            ),
            "selected_candidate_index": selected_candidate_index.reshape(output_shape),
            "abstention_logit": abstention_logit.reshape(output_shape),
            "abstention_probability": abstention_probability.reshape(output_shape),
            "state_write_strength": state_write_strength.reshape(
                batch, points, len(self.config.state_fields)
            ),
            "native_coord_px": native_coord.reshape(batch, points, 2),
            "selected_coord_px": selected_coord.reshape(batch, points, 2),
            "bounded_coordinate_delta_px": bounded_delta.reshape(batch, points, 2),
            "updated_coord_px": updated_coord.reshape(batch, points, 2),
        }


def threshold_hit_targets(
    candidate_error_px: torch.Tensor, thresholds_px: Sequence[float]
) -> torch.Tensor:
    threshold_tensor = torch.as_tensor(
        tuple(float(value) for value in thresholds_px),
        dtype=candidate_error_px.dtype,
        device=candidate_error_px.device,
    )
    return (candidate_error_px.unsqueeze(-1) <= threshold_tensor).to(
        candidate_error_px.dtype
    )


def recovery_training_loss(
    outputs: Dict[str, torch.Tensor],
    candidate_coords_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    gt_coords_px: torch.Tensor,
    network_config: RecoveryNetworkConfig,
    loss_config: RecoveryLossConfig = RecoveryLossConfig(),
    point_valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Composite objective for utility calibration and safe coordinate rollout."""
    if candidate_coords_px.ndim != 4 or candidate_coords_px.shape[-1] != 2:
        raise ValueError("candidate_coords_px must have shape (B,P,K,2)")
    if candidate_valid_mask.shape != candidate_coords_px.shape[:-1]:
        raise ValueError("candidate_valid_mask must match candidate coordinates")
    if gt_coords_px.shape != candidate_coords_px.shape[:2] + (2,):
        raise ValueError("gt_coords_px must have shape (B,P,2)")
    if point_valid_mask is None:
        point_valid_mask = torch.ones(
            gt_coords_px.shape[:-1], dtype=torch.bool, device=gt_coords_px.device
        )
    if point_valid_mask.shape != gt_coords_px.shape[:-1]:
        raise ValueError("point_valid_mask must have shape (B,P)")

    candidate_error = torch.linalg.vector_norm(
        candidate_coords_px - gt_coords_px.unsqueeze(-2), dim=-1
    )
    targets = threshold_hit_targets(candidate_error, network_config.thresholds_px)
    threshold_logits = outputs["threshold_logits"]
    valid_candidate_weight = (
        candidate_valid_mask & point_valid_mask.unsqueeze(-1)
    ).unsqueeze(-1).to(threshold_logits.dtype)
    threshold_bce = F.binary_cross_entropy_with_logits(
        threshold_logits, targets, reduction="none"
    )
    threshold_loss = (threshold_bce * valid_candidate_weight).sum() / (
        valid_candidate_weight.sum().clamp_min(1.0) * threshold_logits.shape[-1]
    )

    masked_error = candidate_error.masked_fill(~candidate_valid_mask, float("inf"))
    best_candidate = masked_error.argmin(dim=-1)
    rank_loss_all = F.cross_entropy(
        outputs["candidate_score"].reshape(-1, candidate_coords_px.shape[-2]),
        best_candidate.reshape(-1),
        reduction="none",
    ).reshape_as(best_candidate)
    point_weight = point_valid_mask.to(rank_loss_all.dtype)
    ranking_loss = (rank_loss_all * point_weight).sum() / point_weight.sum().clamp_min(1.0)

    catastrophe_target = (
        candidate_error >= float(loss_config.catastrophe_threshold_px)
    ).to(outputs["catastrophe_logit"].dtype)
    catastrophe_bce = F.binary_cross_entropy_with_logits(
        outputs["catastrophe_logit"], catastrophe_target, reduction="none"
    )
    candidate_weight = (
        candidate_valid_mask & point_valid_mask.unsqueeze(-1)
    ).to(catastrophe_bce.dtype)
    catastrophe_loss = (catastrophe_bce * candidate_weight).sum() / candidate_weight.sum().clamp_min(1.0)

    updated_error = torch.linalg.vector_norm(
        outputs["updated_coord_px"] - gt_coords_px, dim=-1
    )
    native_error = candidate_error[..., 0]
    coordinate_loss = F.smooth_l1_loss(
        outputs["updated_coord_px"], gt_coords_px, reduction="none"
    ).sum(dim=-1)
    coordinate_loss = (coordinate_loss * point_weight).sum() / point_weight.sum().clamp_min(1.0)
    no_harm_loss = F.relu(
        updated_error - native_error - float(loss_config.no_harm_margin_px)
    )
    no_harm_loss = (no_harm_loss * point_weight).sum() / point_weight.sum().clamp_min(1.0)

    best_error = masked_error.min(dim=-1).values
    abstain_target = (
        best_error + float(loss_config.min_candidate_gain_px) >= native_error
    ).to(outputs["abstention_logit"].dtype)
    abstention_bce = F.binary_cross_entropy_with_logits(
        outputs["abstention_logit"], abstain_target, reduction="none"
    )
    abstention_loss = (abstention_bce * point_weight).sum() / point_weight.sum().clamp_min(1.0)

    tail_excess = F.relu(updated_error - float(loss_config.tail_start_px))
    tail_loss = ((tail_excess.square()) * point_weight).sum() / point_weight.sum().clamp_min(1.0)

    total = (
        float(loss_config.threshold_bce_weight) * threshold_loss
        + float(loss_config.ranking_weight) * ranking_loss
        + float(loss_config.catastrophe_weight) * catastrophe_loss
        + float(loss_config.coordinate_weight) * coordinate_loss
        + float(loss_config.no_harm_weight) * no_harm_loss
        + float(loss_config.abstention_weight) * abstention_loss
        + float(loss_config.tail_weight) * tail_loss
    )
    return {
        "loss": total,
        "threshold_bce": threshold_loss,
        "ranking": ranking_loss,
        "catastrophe": catastrophe_loss,
        "coordinate": coordinate_loss,
        "no_harm": no_harm_loss,
        "abstention": abstention_loss,
        "tail": tail_loss,
        "mean_updated_error_px": (
            updated_error * point_weight
        ).sum() / point_weight.sum().clamp_min(1.0),
        "mean_native_error_px": (
            native_error * point_weight
        ).sum() / point_weight.sum().clamp_min(1.0),
    }
