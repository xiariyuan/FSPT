"""Small temporal model for learned re-entry visibility calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn


@dataclass
class ReEntryVisCalibratorConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReEntryVisCalibratorConfig":
        return ReEntryVisCalibratorConfig(**d)


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,L,H)
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + y)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class ReEntryVisCalibrator(nn.Module):
    """Lightweight per-frame visible-recovery scorer.

    Input:  (B,L,F) feature sequence.
    Output: (B,L) logits. Higher means safer to recover visible using base coords.
    """

    def __init__(self, config: ReEntryVisCalibratorConfig):
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(config.hidden_dim, config.kernel_size, config.dropout) for _ in range(config.num_layers)]
        )
        self.output = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input(x)
        if valid_mask is not None:
            h = h * valid_mask.unsqueeze(-1).to(dtype=h.dtype)
        for block in self.blocks:
            h = block(h, valid_mask=valid_mask)
        logits = self.output(h).squeeze(-1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.bool(), -20.0)
        return logits


def build_model_from_checkpoint_payload(payload: Dict[str, Any]) -> ReEntryVisCalibrator:
    cfg = ReEntryVisCalibratorConfig.from_dict(payload["model_config"])
    model = ReEntryVisCalibrator(cfg)
    model.load_state_dict(payload["model_state_dict"])
    return model
