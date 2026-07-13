from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


CHI2_DF2_THRESHOLDS = {
    0.50: -2.0 * math.log(1.0 - 0.50),
    0.68: -2.0 * math.log(1.0 - 0.68),
    0.90: -2.0 * math.log(1.0 - 0.90),
    0.95: -2.0 * math.log(1.0 - 0.95),
}


def _masked_flatten(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return tensor.reshape(-1, *tensor.shape[tensor.ndim - 1 :]) if tensor.ndim > 1 else tensor.reshape(-1)
    if mask.shape != tensor.shape[:-1] and mask.shape != tensor.shape:
        raise ValueError("mask shape is incompatible with tensor shape.")
    if mask.shape == tensor.shape:
        return tensor[mask]
    return tensor[mask]


def mahalanobis_sq(errors_px: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    if errors_px.shape != variance.shape or errors_px.shape[-1] != 2:
        raise ValueError("errors_px and variance must have identical (..., 2) shapes.")
    return (errors_px.square() / variance.clamp_min(1e-8)).sum(dim=-1)


def gaussian_nll(errors_px: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    if errors_px.shape != variance.shape or errors_px.shape[-1] != 2:
        raise ValueError("errors_px and variance must have identical (..., 2) shapes.")
    return 0.5 * (
        variance.clamp_min(1e-8).log()
        + errors_px.square() / variance.clamp_min(1e-8)
        + math.log(2.0 * math.pi)
    ).sum(dim=-1)


def gaussian_ellipse_coverage(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    threshold = -2.0 * math.log(1.0 - float(quantile))
    return (mahalanobis_sq(errors_px, variance) <= threshold).float().mean()


def risk_coverage_curve(
    point_error_px: torch.Tensor,
    uncertainty_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Risk after retaining the least-uncertain fraction of predictions."""
    point_error_px = point_error_px.reshape(-1)
    uncertainty_score = uncertainty_score.reshape(-1)
    if point_error_px.numel() != uncertainty_score.numel():
        raise ValueError("risk and uncertainty must have the same number of elements.")
    if point_error_px.numel() == 0:
        raise ValueError("Cannot compute risk-coverage on an empty tensor.")
    order = torch.argsort(uncertainty_score, stable=True)
    sorted_risk = point_error_px[order]
    cumulative_risk = sorted_risk.cumsum(0) / torch.arange(
        1, sorted_risk.numel() + 1, device=sorted_risk.device, dtype=sorted_risk.dtype
    )
    coverage = torch.arange(
        1, sorted_risk.numel() + 1, device=sorted_risk.device, dtype=sorted_risk.dtype
    ) / float(sorted_risk.numel())
    return coverage, cumulative_risk


def area_under_risk_coverage(
    point_error_px: torch.Tensor,
    uncertainty_score: torch.Tensor,
) -> torch.Tensor:
    coverage, risk = risk_coverage_curve(point_error_px, uncertainty_score)
    if coverage.numel() == 1:
        return risk[0]
    zero = torch.zeros(1, dtype=coverage.dtype, device=coverage.device)
    coverage_aug = torch.cat([zero, coverage])
    risk_aug = torch.cat([risk[:1], risk])
    return torch.trapz(risk_aug, coverage_aug)


def calibration_report(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if mask is not None:
        errors_px = errors_px[mask]
        variance = variance[mask]
    errors_px = errors_px.reshape(-1, 2)
    variance = variance.reshape(-1, 2).clamp_min(1e-8)
    if errors_px.shape[0] == 0:
        return {"count": 0.0}

    point_error = torch.linalg.vector_norm(errors_px, dim=-1)
    uncertainty_score = torch.sqrt(variance.prod(dim=-1))
    mahal = mahalanobis_sq(errors_px, variance)
    nll = gaussian_nll(errors_px, variance)

    result: Dict[str, float] = {
        "count": float(errors_px.shape[0]),
        "nll": float(nll.mean().item()),
        "mean_point_error_px": float(point_error.mean().item()),
        "median_point_error_px": float(point_error.median().item()),
        "mean_sharpness": float(uncertainty_score.mean().item()),
        "mean_95_ellipse_area": float(
            (math.pi * CHI2_DF2_THRESHOLDS[0.95] * uncertainty_score).mean().item()
        ),
        "mahalanobis_median": float(mahal.median().item()),
        "mahalanobis_p95": float(torch.quantile(mahal, 0.95).item()),
        "aurc": float(area_under_risk_coverage(point_error, uncertainty_score).item()),
    }
    for q in (0.50, 0.68, 0.90, 0.95):
        coverage = gaussian_ellipse_coverage(errors_px, variance, q)
        label = int(round(q * 100))
        result[f"coverage_{label}"] = float(coverage.item())
        result[f"coverage_gap_{label}"] = float(abs(coverage.item() - q))
    return result
