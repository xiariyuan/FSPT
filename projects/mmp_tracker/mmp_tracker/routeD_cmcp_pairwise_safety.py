"""Local native-vs-proposal safety comparison for frozen Route-D CMCP candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION = "routeD_cmcp_local_pairwise_safety_v0"
CMCP_PAIRWISE_LOCAL_TOKEN_DIM = 88
CMCP_PAIRWISE_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class CMCPLocalSafetyConfig:
    token_dim: int = CMCP_PAIRWISE_LOCAL_TOKEN_DIM
    hidden_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    feedforward_multiplier: int = 4
    dropout: float = 0.10
    risk_weight: float = 0.75
    preference_weight: float = 1.0
    initial_abstention_bias: float = 2.0
    thresholds_px: tuple[float, ...] = CMCP_PAIRWISE_THRESHOLDS
    utility_weights: tuple[float, ...] = (0.28, 0.24, 0.20, 0.16, 0.12)

    def __post_init__(self) -> None:
        if self.token_dim != CMCP_PAIRWISE_LOCAL_TOKEN_DIM:
            raise ValueError("P0h token dimension is frozen at 88")
        if self.hidden_dim <= 0 or self.attention_heads <= 0 or self.attention_layers <= 0:
            raise ValueError("invalid comparator dimensions")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if len(self.thresholds_px) != 5 or len(self.utility_weights) != 5:
            raise ValueError("P0h requires five frozen utility thresholds")


def _sample_map_at_xy(
    fmap: torch.Tensor,
    xy_px: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Bilinearly sample a batch of maps without feature normalization."""
    if fmap.ndim != 4:
        raise ValueError("fmap must have shape (B,C,H,W)")
    if xy_px.ndim != 3 or xy_px.shape[0] != fmap.shape[0] or xy_px.shape[-1] != 2:
        raise ValueError("xy_px must have shape (B,K,2)")
    xy = xy_px.to(device=fmap.device, dtype=fmap.dtype)
    gx = 2.0 * xy[..., 0] / float(max(input_width - 1, 1)) - 1.0
    gy = 2.0 * xy[..., 1] / float(max(input_height - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        fmap,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


def build_cmcp_local_candidate_tokens(
    *,
    hidden_map: torch.Tensor,
    recurrent_input: torch.Tensor,
    utility_logit: torch.Tensor,
    risk_logit: torch.Tensor,
    proposal_score: torch.Tensor,
    candidate_coords_xy_px: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    native_visibility_probability: torch.Tensor,
    native_confidence_probability: torch.Tensor,
    native_joint_probability: torch.Tensor,
    previous_decision_summary: torch.Tensor,
    input_height: int = 256,
    input_width: int = 256,
) -> torch.Tensor:
    """Build the frozen 88-D local token contract for one causal frame."""
    batch, candidates = candidate_valid_mask.shape
    if hidden_map.shape[:2] != (batch, 64):
        raise ValueError("hidden_map must have shape (B,64,H,W)")
    if recurrent_input.shape[:2] != (batch, 9):
        raise ValueError("recurrent_input must have shape (B,9,H,W)")
    for name, tensor in {
        "utility_logit": utility_logit,
        "risk_logit": risk_logit,
        "proposal_score": proposal_score,
    }.items():
        if tensor.shape[:2] != (batch, 1):
            raise ValueError(f"{name} must have shape (B,1,H,W)")
    if candidate_coords_xy_px.shape != (batch, candidates, 2):
        raise ValueError("candidate coordinate shape mismatch")
    if candidate_scores.shape != (batch, candidates):
        raise ValueError("candidate score shape mismatch")
    if previous_decision_summary.shape != (batch, 4):
        raise ValueError("previous_decision_summary must have shape (B,4)")
    if not candidate_valid_mask[:, 0].all():
        raise ValueError("native candidate must always be valid")

    hidden = _sample_map_at_xy(
        hidden_map, candidate_coords_xy_px,
        input_height=input_height, input_width=input_width,
    )
    evidence = _sample_map_at_xy(
        recurrent_input, candidate_coords_xy_px,
        input_height=input_height, input_width=input_width,
    )
    dense = torch.cat(
        [
            _sample_map_at_xy(utility_logit, candidate_coords_xy_px, input_height=input_height, input_width=input_width),
            _sample_map_at_xy(risk_logit, candidate_coords_xy_px, input_height=input_height, input_width=input_width),
            _sample_map_at_xy(proposal_score, candidate_coords_xy_px, input_height=input_height, input_width=input_width),
        ],
        dim=-1,
    )
    native_stats = torch.stack(
        [native_visibility_probability, native_confidence_probability, native_joint_probability], dim=-1
    )[:, None].expand(-1, candidates, -1)
    native = candidate_coords_xy_px[:, :1]
    delta = candidate_coords_xy_px - native
    scale_x = float(max(input_width - 1, 1))
    scale_y = float(max(input_height - 1, 1))
    dx = delta[..., 0] / scale_x
    dy = delta[..., 1] / scale_y
    magnitude = torch.sqrt(dx.square() + dy.square())
    displacement = torch.stack([dx, dy, magnitude], dim=-1)
    rank = torch.arange(candidates, device=hidden.device, dtype=hidden.dtype)
    rank = (rank / float(max(candidates - 1, 1))).view(1, candidates, 1).expand(batch, -1, -1)
    score_margin = (candidate_scores - candidate_scores[:, :1]).unsqueeze(-1)
    previous = previous_decision_summary[:, None].expand(-1, candidates, -1)
    token = torch.cat(
        [hidden, evidence, dense, native_stats, displacement, rank, score_margin, previous], dim=-1
    )
    if token.shape[-1] != CMCP_PAIRWISE_LOCAL_TOKEN_DIM:
        raise RuntimeError(f"local token dimension drift: {token.shape[-1]}")
    return torch.where(candidate_valid_mask.unsqueeze(-1), token, torch.zeros_like(token))


class CMCPLocalPairwiseSafetyComparator(nn.Module):
    """Candidate-set comparator with exact native-safe initialization."""

    def __init__(self, config: CMCPLocalSafetyConfig = CMCPLocalSafetyConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.token_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_set = nn.TransformerEncoder(layer, num_layers=config.attention_layers)
        self.utility_head = nn.Linear(config.hidden_dim, 5)
        self.risk_head = nn.Linear(config.hidden_dim, 1)
        self.preference_head = nn.Linear(config.hidden_dim, 1)
        self.abstention_head = nn.Linear(config.hidden_dim, 1)
        self._initialize_native_safe()

    def _initialize_native_safe(self) -> None:
        for head in (self.utility_head, self.risk_head, self.preference_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.abstention_head.weight)
        nn.init.constant_(self.abstention_head.bias, self.config.initial_abstention_bias)

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        candidate_coords_xy_px: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        if candidate_tokens.ndim != 3 or candidate_tokens.shape[-1] != self.config.token_dim:
            raise ValueError("candidate_tokens must have shape (B,K,88)")
        if candidate_valid_mask.shape != candidate_tokens.shape[:2]:
            raise ValueError("candidate_valid_mask shape mismatch")
        if candidate_coords_xy_px.shape != candidate_tokens.shape[:2] + (2,):
            raise ValueError("candidate_coords_xy_px shape mismatch")
        if not candidate_valid_mask[:, 0].all():
            raise ValueError("native candidate must be valid")
        encoded = self.encoder(candidate_tokens)
        encoded = self.candidate_set(encoded, src_key_padding_mask=~candidate_valid_mask)
        encoded = torch.where(candidate_valid_mask.unsqueeze(-1), encoded, torch.zeros_like(encoded))
        raw_utility = self.utility_head(encoded)
        base = raw_utility[..., :1]
        increments = F.softplus(raw_utility[..., 1:])
        cumulative = []
        running = base
        for threshold_index in range(increments.shape[-1]):
            running = running + increments[..., threshold_index : threshold_index + 1]
            cumulative.append(running)
        utility_logits = torch.cat([base, *cumulative], dim=-1)
        utility_probability = torch.sigmoid(utility_logits)
        weights = torch.tensor(self.config.utility_weights, device=encoded.device, dtype=encoded.dtype)
        weights = weights / weights.sum()
        weighted_utility = (utility_probability * weights).sum(dim=-1)
        risk_probability = torch.sigmoid(self.risk_head(encoded).squeeze(-1))
        preference_logit = self.preference_head(encoded).squeeze(-1)
        score = (
            weighted_utility
            - float(self.config.risk_weight) * risk_probability
            + float(self.config.preference_weight) * preference_logit
        )
        score = score.masked_fill(~candidate_valid_mask, float("-inf"))
        proposed_index = score.argmax(dim=-1)
        pooled = encoded[:, 0]
        abstention_probability = torch.sigmoid(self.abstention_head(pooled).squeeze(-1))
        selected_index = torch.where(
            abstention_probability >= 0.5,
            torch.zeros_like(proposed_index),
            proposed_index,
        )
        selected_coord = candidate_coords_xy_px.gather(
            1, selected_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        return {
            "utility_logits": utility_logits,
            "utility_probability": utility_probability,
            "risk_probability": risk_probability,
            "preference_logit": preference_logit,
            "candidate_score": score,
            "abstention_probability": abstention_probability,
            "selected_candidate_index": selected_index,
            "selected_coord_xy_px": selected_coord,
            "candidate_coords_xy_px": candidate_coords_xy_px,
        }
