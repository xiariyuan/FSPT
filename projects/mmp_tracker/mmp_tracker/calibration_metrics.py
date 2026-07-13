from typing import Dict

import torch


def gaussian_ellipse_coverage(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Empirical 2D diagonal Gaussian ellipse coverage.

    Uses chi-square(df=2) threshold: -2 ln(1-q).
    """
    threshold = -2.0 * torch.log(torch.tensor(1.0 - quantile, device=errors_px.device))
    mahal = (errors_px.square() / variance.clamp_min(1e-8)).sum(dim=-1)
    return (mahal <= threshold).float().mean()


def calibration_report(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
) -> Dict[str, float]:
    result = {}
    for q in (0.5, 0.68, 0.9, 0.95):
        cov = gaussian_ellipse_coverage(errors_px, variance, q)
        result[f"coverage_{int(q*100)}"] = float(cov.item())
        result[f"coverage_gap_{int(q*100)}"] = float(abs(cov - q).item())
    result["mean_area"] = float(torch.sqrt(variance[:, 0] * variance[:, 1]).mean().item())
    return result
