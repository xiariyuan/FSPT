from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


MMP_UNCERTAINTY_VALUE_NAMES: Tuple[str, ...] = (
    "tracker_confidence",
    "pred_visibility",
    "local_confidence",
    "local_entropy",
    "global_confidence",
    "global_entropy",
    "selector_probability",
    "global_local_disagreement",
    "coarse_refine_disagreement",
    "selected_global",
    "active",
    "elapsed_since_query",
    "predicted_occluded_duration",
)
MMP_UNCERTAINTY_FEATURE_DIM = 2 * len(MMP_UNCERTAINTY_VALUE_NAMES)


class DiagonalGaussianUncertaintyHead(nn.Module):
    """Strict MVP-1 uncertainty head.

    The module predicts only diagonal log variances in pixel coordinates. It
    never receives or returns a replacement trajectory mean, so it cannot
    change point coordinates, visibility, routing, commit decisions, or memory.
    """

    def __init__(
        self,
        input_dim: int = MMP_UNCERTAINTY_FEATURE_DIM,
        hidden_dim: int = 128,
        min_std_px: float = 0.25,
        max_std_px: float = 256.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if min_std_px <= 0 or max_std_px <= min_std_px:
            raise ValueError("Expected 0 < min_std_px < max_std_px.")
        self.input_dim = int(input_dim)
        self.min_log_var = float(2.0 * math.log(float(min_std_px)))
        self.max_log_var = float(2.0 * math.log(float(max_std_px)))
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}."
            )
        log_var = self.net(features)
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        return {
            "log_var": log_var,
            "variance": log_var.exp(),
            "std": (0.5 * log_var).exp(),
        }


def diagonal_gaussian_nll(
    errors_px: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log likelihood for a 2D diagonal Gaussian in pixel units."""
    if errors_px.shape != log_var.shape or errors_px.shape[-1] != 2:
        raise ValueError("errors_px and log_var must have identical (..., 2) shapes.")
    per_dim = log_var + errors_px.square() * torch.exp(-log_var) + math.log(2.0 * math.pi)
    nll = 0.5 * per_dim.sum(dim=-1)
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def _lookup_tensor(info: Dict[str, object], key: str) -> Optional[torch.Tensor]:
    value = info.get(key)
    if isinstance(value, torch.Tensor):
        return value
    debug = info.get("debug")
    if isinstance(debug, dict):
        value = debug.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _series_or_zero(
    info: Dict[str, object],
    key: str,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value = _lookup_tensor(info, key)
    if value is None:
        return torch.zeros_like(reference), torch.zeros_like(reference)
    if value.shape != reference.shape:
        raise ValueError(
            f"Diagnostic {key!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(reference.shape)}."
        )
    return value.detach().to(reference.dtype), torch.ones_like(reference)


def consecutive_true_duration(mask: torch.Tensor) -> torch.Tensor:
    """Consecutive run length ending at each frame for a (B,N,T) bool mask."""
    if mask.ndim != 3:
        raise ValueError("mask must have shape (B, N, T).")
    out = torch.zeros_like(mask, dtype=torch.long)
    running = torch.zeros(mask.shape[:2], dtype=torch.long, device=mask.device)
    for t in range(mask.shape[-1]):
        running = torch.where(mask[..., t], running + 1, torch.zeros_like(running))
        out[..., t] = running
    return out


def build_mmp_uncertainty_features(
    info: Dict[str, object],
    pred_visibility: torch.Tensor,
    visibility_threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build detached strict-MVP features and temporal metadata.

    Values are concatenated with one availability bit per value. This avoids
    encoding missing diagnostics with an undocumented sentinel. The result has
    shape (B, N, T, 26).
    """
    if pred_visibility.ndim != 3:
        raise ValueError("pred_visibility must have shape (B, N, T).")
    reference = pred_visibility.detach().float()
    batch, points, time = reference.shape

    query_t = info.get("query_t")
    if isinstance(query_t, torch.Tensor):
        if query_t.shape != (batch, points):
            raise ValueError("query_t must have shape (B, N).")
        query_t = query_t.detach().long().clamp(0, time - 1)
    else:
        query_t = torch.zeros(batch, points, dtype=torch.long, device=reference.device)

    frame_index = torch.arange(time, device=reference.device).view(1, 1, time)
    elapsed_raw = (frame_index - query_t.unsqueeze(-1)).clamp_min(0)
    active_derived = frame_index >= query_t.unsqueeze(-1)

    predicted_occluded = reference < float(visibility_threshold)
    pred_occ_duration = consecutive_true_duration(predicted_occluded)

    values = []
    masks = []

    def add(value: torch.Tensor, available: torch.Tensor) -> None:
        values.append(value.detach().float())
        masks.append(available.detach().float())

    tracker_conf, tracker_conf_mask = _series_or_zero(info, "confidence", reference)
    add(tracker_conf, tracker_conf_mask)
    add(reference.clamp(0.0, 1.0), torch.ones_like(reference))

    for key in (
        "local_confidence",
        "local_entropy",
        "global_confidence",
        "global_entropy",
        "selector_probability",
    ):
        value, available = _series_or_zero(info, key, reference)
        add(value, available)

    for key in ("global_local_agreement_px", "coarse_refine_consistency_px"):
        value, available = _series_or_zero(info, key, reference)
        value = torch.log1p(value.clamp_min(0.0)) / math.log1p(256.0)
        add(value, available)

    selected_global, selected_global_mask = _series_or_zero(
        info, "selected_global_mask", reference
    )
    add(selected_global, selected_global_mask)

    active_tensor = _lookup_tensor(info, "active_mask")
    if active_tensor is not None:
        if active_tensor.shape != reference.shape:
            raise ValueError("active_mask must have shape (B, N, T).")
        active = active_tensor.detach().float()
        active_mask = torch.ones_like(reference)
    else:
        active = active_derived.float()
        active_mask = torch.ones_like(reference)
    add(active, active_mask)

    time_denom = max(math.log1p(max(time - 1, 1)), 1.0)
    add(torch.log1p(elapsed_raw.float()) / time_denom, torch.ones_like(reference))
    add(torch.log1p(pred_occ_duration.float()) / time_denom, torch.ones_like(reference))

    value_tensor = torch.stack(values, dim=-1)
    mask_tensor = torch.stack(masks, dim=-1)
    features = torch.cat([value_tensor, mask_tensor], dim=-1).detach()
    if features.shape[-1] != MMP_UNCERTAINTY_FEATURE_DIM:
        raise RuntimeError("Unexpected uncertainty feature dimension.")

    metadata = {
        "active_mask": active.bool().detach(),
        "elapsed_since_query": elapsed_raw.detach(),
        "predicted_occluded": predicted_occluded.detach(),
        "predicted_occluded_duration": pred_occ_duration.detach(),
    }
    return features, metadata
