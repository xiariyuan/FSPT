"""Causal rank-invariant temporal candidate selector for Route-D MUSR P0f."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routeD_recovery_network import RecoveryNetworkConfig


@dataclass(frozen=True)
class TemporalSelectorConfig:
    window_frames: int = 6
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_feedforward_multiplier: int = 4
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.window_frames <= 1:
            raise ValueError("window_frames must exceed one")
        if self.temporal_layers <= 0 or self.temporal_heads <= 0:
            raise ValueError("temporal layer/head counts must be positive")
        if self.temporal_feedforward_multiplier <= 0:
            raise ValueError("temporal feedforward multiplier must be positive")


class CausalSetEvidenceTemporalSelector(nn.Module):
    """Current-candidate scoring conditioned on rank-invariant causal history.

    Inputs are ``(B,L,K,*)`` where ``L`` is the fixed causal window. Candidate
    zero is native in every valid frame. Past candidate sets are summarized by
    attention pooling and therefore do not rely on non-native rank identity.
    """

    def __init__(
        self,
        network_config: RecoveryNetworkConfig,
        temporal_config: TemporalSelectorConfig = TemporalSelectorConfig(),
    ) -> None:
        super().__init__()
        if network_config.hidden_dim % temporal_config.temporal_heads != 0:
            raise ValueError("hidden_dim must be divisible by temporal_heads")
        self.config = network_config
        self.temporal_config = temporal_config
        hidden = network_config.hidden_dim
        threshold_count = len(network_config.thresholds_px)

        self.candidate_feature_projection = nn.Sequential(
            nn.LayerNorm(network_config.candidate_feature_dim),
            nn.Linear(network_config.candidate_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(network_config.state_feature_dim),
            nn.Linear(network_config.state_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.source_embedding = nn.Embedding(network_config.num_candidate_sources, hidden)
        self.native_role_embedding = nn.Parameter(torch.zeros(1, 1, hidden))
        self.candidate_role_embedding = nn.Parameter(torch.zeros(1, 1, hidden))

        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=network_config.num_attention_heads,
            dim_feedforward=hidden * network_config.feedforward_multiplier,
            dropout=network_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=network_config.num_attention_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Linear(hidden, hidden, bias=False)
        self.pool_key = nn.Linear(hidden, hidden, bias=False)
        self.frame_evidence_projection = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=temporal_config.temporal_heads,
            dim_feedforward=hidden * temporal_config.temporal_feedforward_multiplier,
            dropout=temporal_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_config.temporal_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.temporal_position_embedding = nn.Parameter(
            torch.zeros(1, temporal_config.window_frames, hidden)
        )
        self.current_candidate_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
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
        weights = torch.tensor(network_config.threshold_utility_weights, dtype=torch.float32)
        self.register_buffer(
            "threshold_utility_weights", weights / weights.sum(), persistent=True
        )
        nn.init.normal_(self.native_role_embedding, std=0.02)
        nn.init.normal_(self.candidate_role_embedding, std=0.02)
        nn.init.normal_(self.temporal_position_embedding, std=0.02)
        self._initialize_native_safe_heads()

    def _initialize_native_safe_heads(self) -> None:
        """Make every candidate score identical before training."""
        for module in (self.threshold_head, self.catastrophe_head, self.selector_bias_head):
            final = module[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("selector heads must end in Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _monotonic_threshold_logits(raw: torch.Tensor) -> torch.Tensor:
        first = raw[..., :1]
        if raw.shape[-1] == 1:
            return first
        increments = F.softplus(raw[..., 1:])
        # Explicit recurrence is algebraically identical to cumsum but remains
        # deterministic on CUDA in the pinned PyTorch environment.
        parts = [first]
        running = first
        for index in range(increments.shape[-1]):
            running = running + increments[..., index : index + 1]
            parts.append(running)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_coords_px: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        state_features: torch.Tensor,
        source_ids: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if candidate_features.ndim != 4:
            raise ValueError("candidate_features must have shape (B,L,K,F)")
        batch, frames, candidates, feature_dim = candidate_features.shape
        if frames != self.temporal_config.window_frames:
            raise ValueError("temporal window length mismatch")
        if feature_dim != self.config.candidate_feature_dim:
            raise ValueError("candidate feature dimension mismatch")
        if candidate_coords_px.shape != (batch, frames, candidates, 2):
            raise ValueError("candidate coordinate shape mismatch")
        if candidate_valid_mask.shape != (batch, frames, candidates):
            raise ValueError("candidate valid shape mismatch")
        if state_features.shape != (batch, frames, self.config.state_feature_dim):
            raise ValueError("state feature shape mismatch")
        if source_ids.shape != (batch, frames, candidates):
            raise ValueError("source id shape mismatch")
        if frame_valid_mask.shape != (batch, frames):
            raise ValueError("frame valid shape mismatch")
        if not torch.all(frame_valid_mask[:, -1]):
            raise ValueError("current frame must be valid")
        if not torch.all(candidate_valid_mask[:, -1, 0]):
            raise ValueError("current native candidate must be valid")
        historical_native_ok = (~frame_valid_mask) | candidate_valid_mask[..., 0]
        if not torch.all(historical_native_ok):
            raise ValueError("native candidate must be valid in every valid frame")

        flat = batch * frames
        valid = candidate_valid_mask.reshape(flat, candidates).bool()
        coords = candidate_coords_px.reshape(flat, candidates, 2)
        native = coords[:, 0]
        relative = (coords - native[:, None]) / float(self.config.coordinate_scale_px)
        norm = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        coordinate_input = torch.cat([relative, norm, norm.square()], dim=-1)

        candidate_tokens = self.candidate_feature_projection(
            candidate_features.reshape(flat, candidates, feature_dim)
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

        # Keep the state token unmasked even for padded history to avoid an
        # all-masked Transformer row. The resulting frame evidence is zeroed.
        sequence = torch.cat([state_token, candidate_tokens], dim=1)
        padding = torch.cat(
            [torch.zeros(flat, 1, dtype=torch.bool, device=valid.device), ~valid], dim=1
        )
        encoded = self.candidate_set_encoder(sequence, src_key_padding_mask=padding)
        encoded_state = encoded[:, 0]
        encoded_candidates = encoded[:, 1:]

        pool_valid = valid.clone()
        invalid_frame = ~frame_valid_mask.reshape(flat)
        pool_valid[invalid_frame, 0] = True
        query = self.pool_query(encoded_state)[:, None]
        key = self.pool_key(encoded_candidates)
        logits = (query * key).sum(dim=-1) / math.sqrt(float(key.shape[-1]))
        logits = logits.masked_fill(~pool_valid, torch.finfo(logits.dtype).min)
        probability = torch.softmax(logits, dim=-1)
        pooled = (probability[..., None] * encoded_candidates).sum(dim=1)
        frame_evidence = self.frame_evidence_projection(
            torch.cat([encoded_state, pooled, encoded_candidates[:, 0]], dim=-1)
        ).reshape(batch, frames, -1)
        frame_evidence = frame_evidence * frame_valid_mask[..., None].to(frame_evidence.dtype)

        # Build a batch-specific causal mask. Valid queries cannot attend to
        # left-padding keys. Invalid query rows are allowed to attend only to
        # themselves, preventing all-masked softmax rows while keeping their
        # content isolated from every valid frame.
        causal_mask = torch.triu(
            torch.ones(frames, frames, dtype=torch.bool, device=frame_evidence.device),
            diagonal=1,
        ).unsqueeze(0).expand(batch, -1, -1).clone()
        invalid_key = (~frame_valid_mask.bool())[:, None, :].expand(batch, frames, frames)
        causal_mask |= invalid_key
        invalid_query = ~frame_valid_mask.bool()
        for frame_index in range(frames):
            rows = invalid_query[:, frame_index]
            if rows.any():
                causal_mask[rows, frame_index, :] = True
                causal_mask[rows, frame_index, frame_index] = False
        temporal_mask = causal_mask[:, None].expand(
            batch, self.temporal_config.temporal_heads, frames, frames
        ).reshape(batch * self.temporal_config.temporal_heads, frames, frames)

        temporal_input = frame_evidence + (
            self.temporal_position_embedding
            * frame_valid_mask[..., None].to(frame_evidence.dtype)
        )
        temporal = self.temporal_encoder(
            temporal_input,
            mask=temporal_mask,
        )
        temporal = temporal * frame_valid_mask[..., None].to(temporal.dtype)
        context = temporal[:, -1]
        current_candidates = encoded_candidates.reshape(batch, frames, candidates, -1)[:, -1]
        fused = self.current_candidate_fusion(
            torch.cat(
                [current_candidates, context[:, None].expand(-1, candidates, -1)], dim=-1
            )
        ) + current_candidates

        raw_threshold = self.threshold_head(fused)
        threshold_logits = self._monotonic_threshold_logits(raw_threshold)
        threshold_probability = torch.sigmoid(threshold_logits)
        expected_utility = (
            threshold_probability
            * self.threshold_utility_weights.to(threshold_probability.dtype)
        ).sum(dim=-1)
        catastrophe_logit = self.catastrophe_head(fused).squeeze(-1)
        catastrophe_probability = torch.sigmoid(catastrophe_logit)
        selector_bias = float(self.config.max_contextual_selector_bias) * torch.tanh(
            self.selector_bias_head(fused).squeeze(-1)
        )
        score = (
            expected_utility
            - float(self.config.catastrophe_risk_weight) * catastrophe_probability
            + selector_bias
        )
        current_valid = candidate_valid_mask[:, -1]
        score = score.masked_fill(~current_valid, torch.finfo(score.dtype).min)
        candidate_probability = torch.softmax(
            score / float(self.config.selection_temperature), dim=-1
        )
        selected = score.argmax(dim=-1)
        current_coords = candidate_coords_px[:, -1]
        selected_coords = current_coords.gather(
            1, selected[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        return {
            "threshold_logits": threshold_logits[:, None],
            "threshold_probabilities": threshold_probability[:, None],
            "catastrophe_logit": catastrophe_logit[:, None],
            "catastrophe_probability": catastrophe_probability[:, None],
            "expected_utility": expected_utility[:, None],
            "candidate_score": score[:, None],
            "candidate_probability": candidate_probability[:, None],
            "selected_candidate_index": selected[:, None],
            "selected_coord_px": selected_coords[:, None],
            "native_coord_px": current_coords[:, :1],
            "temporal_context": context,
            "frame_evidence": frame_evidence,
        }
