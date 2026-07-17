"""Zero-initialized late feature metric residual adapter for Route-D P0i."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


LMRA_SCHEMA_VERSION = "routeD_cmcp_late_metric_residual_adapter_v0"
LMRA_FEATURE_DIM = 128
LMRA_RANK = 32
LMRA_TRAINABLE_PARAMETERS = 8352


@dataclass(frozen=True)
class LMRAConfig:
    feature_dim: int = LMRA_FEATURE_DIM
    rank: int = LMRA_RANK

    def __post_init__(self) -> None:
        if self.feature_dim != LMRA_FEATURE_DIM:
            raise ValueError("P0i feature dimension is frozen at 128")
        if self.rank != LMRA_RANK:
            raise ValueError("P0i adapter rank is frozen at 32")


class LateMetricResidualAdapter(nn.Module):
    """Rank-32 residual metric adapter with exact identity initialization.

    Correlation construction performs the final feature normalization. Keeping
    normalization outside this module makes a zero residual exactly preserve the
    cached float32 feature tensor and therefore the formal P0h outputs.
    """

    def __init__(self, config: LMRAConfig = LMRAConfig()) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Conv2d(config.feature_dim, config.rank, 1, bias=True)
        self.up = nn.Conv2d(config.rank, config.feature_dim, 1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        parameters = sum(p.numel() for p in self.parameters())
        if parameters != LMRA_TRAINABLE_PARAMETERS:
            raise RuntimeError(f"LMRA parameter-count drift: {parameters}")

    def residual(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.config.feature_dim:
            raise ValueError("feature_map must have shape (B,128,H,W)")
        return self.up(F.gelu(self.down(feature_map)))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        residual = self.residual(feature_map)
        updated = feature_map + residual
        # Preserve the exact signed-zero bit pattern at identity initialization.
        # Gradients still flow through every nonzero feature location and through
        # zero locations as soon as the learned residual becomes nonzero.
        preserve_zero = (feature_map == 0) & (residual == 0)
        return torch.where(preserve_zero, feature_map, updated)


def lmra_distortion_loss(adapted: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
    if adapted.shape != frozen.shape:
        raise ValueError("adapted/frozen feature shape mismatch")
    return (adapted - frozen).square().mean()
