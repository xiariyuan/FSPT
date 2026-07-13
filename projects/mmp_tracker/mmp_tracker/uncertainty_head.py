from typing import Dict, Optional

import torch
import torch.nn as nn


class DiagonalGaussianUncertaintyHead(nn.Module):
    """MVP-1 uncertainty head.

    Predicts only diagonal log variances. It must not modify tracker means or
    tracking decisions. Inputs are detached diagnostics from the tracker.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 min_std_px: float = 0.25, max_std_px: float = 256.0):
        super().__init__()
        self.min_log_var = float(2.0 * torch.log(torch.tensor(min_std_px)).item())
        self.max_log_var = float(2.0 * torch.log(torch.tensor(max_std_px)).item())
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_var = self.net(features)
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        return {
            "log_var": log_var,
            "variance": log_var.exp(),
            "std": (0.5 * log_var).exp(),
        }


def build_mmp_uncertainty_features(
    info: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build detached uncertainty features from existing MMP diagnostics.

    Output shape keeps the leading tracker dimensions and appends feature dim.
    Missing diagnostics are replaced by zeros plus no hidden sentinel semantics.
    """
    keys = [
        "confidence",
        "commit_probability",
        "selector_probability",
        "local_confidence",
        "global_confidence",
        "local_entropy",
        "global_entropy",
        "global_local_agreement_px",
        "coarse_refine_consistency_px",
    ]
    feats = []
    for key in keys:
        value = info.get(key)
        if isinstance(value, torch.Tensor):
            feats.append(value.detach().unsqueeze(-1) if value.ndim == 3 else value.detach())
        else:
            ref = next(v for v in info.values() if isinstance(v, torch.Tensor))
            feats.append(torch.zeros_like(ref).detach().unsqueeze(-1))
    out = torch.cat(feats, dim=-1)
    return out.clamp_min(-1e6).clamp_max(1e6) + eps * 0.0
