from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from .uncertainty_head import MMP_UNCERTAINTY_FEATURE_DIM, diagonal_gaussian_nll


CONDITIONAL_CALIBRATION_STEPS = 500
CONDITIONAL_CALIBRATION_LEARNING_RATE = 0.01
CONDITIONAL_CALIBRATION_L2 = 1e-3
CONDITIONAL_CALIBRATION_MIN_SCALE = 1.0 / 64.0
CONDITIONAL_CALIBRATION_MAX_SCALE = 64.0
CONDITIONAL_SELECTION_RELATIVE_NLL = 0.01


class ConditionalSharedScaleCalibrator(nn.Module):
    """Affine row-wise variance scale for the preregistered MVP-1C extension.

    The module outputs one shared log scale per cache row. It cannot change the
    predictive mean or the y/x variance ratio supplied by the frozen learned
    uncertainty head.
    """

    def __init__(
        self,
        input_dim: int = MMP_UNCERTAINTY_FEATURE_DIM + 2,
        min_scale: float = CONDITIONAL_CALIBRATION_MIN_SCALE,
        max_scale: float = CONDITIONAL_CALIBRATION_MAX_SCALE,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if min_scale <= 0.0 or max_scale <= min_scale:
            raise ValueError("Expected 0 < min_scale < max_scale")
        self.input_dim = int(input_dim)
        self.min_log_scale = float(math.log(float(min_scale)))
        self.max_log_scale = float(math.log(float(max_scale)))
        self.affine = nn.Linear(self.input_dim, 1)

    def forward(self, standardized_inputs: torch.Tensor) -> torch.Tensor:
        if standardized_inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected conditional input dimension {self.input_dim}, "
                f"got {standardized_inputs.shape[-1]}"
            )
        return self.affine(standardized_inputs).squeeze(-1).clamp(
            self.min_log_scale, self.max_log_scale
        )


def _validate_cache_and_variance(
    cache: Mapping[str, torch.Tensor], raw_variance: torch.Tensor
) -> None:
    for key in ("features", "errors_px"):
        if key not in cache:
            raise ValueError(f"Cache is missing {key!r}")
    features = cache["features"]
    errors = cache["errors_px"]
    if features.ndim != 2 or features.shape[-1] != MMP_UNCERTAINTY_FEATURE_DIM:
        raise ValueError(
            f"features must have shape (M,{MMP_UNCERTAINTY_FEATURE_DIM})"
        )
    if errors.ndim != 2 or errors.shape[-1] != 2:
        raise ValueError("errors_px must have shape (M,2)")
    if raw_variance.shape != errors.shape:
        raise ValueError("raw_variance and errors_px must have identical (M,2) shapes")
    if not torch.isfinite(features).all() or not torch.isfinite(raw_variance).all():
        raise ValueError("Conditional calibration inputs must be finite")


def build_conditional_calibration_inputs(
    cache: Mapping[str, torch.Tensor],
    raw_variance: torch.Tensor,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
) -> torch.Tensor:
    """Build the fixed 28D A2 input from frozen cache features and raw variance."""
    _validate_cache_and_variance(cache, raw_variance)
    min_variance = float(min_std_px) ** 2
    max_variance = float(max_std_px) ** 2
    safe_variance = raw_variance.detach().float().clamp(min_variance, max_variance)
    log_variance = safe_variance.log()
    log_geometric_mean = 0.5 * log_variance.sum(dim=-1, keepdim=True)
    log_anisotropy = (log_variance[:, :1] - log_variance[:, 1:2])
    return torch.cat(
        [
            cache["features"].detach().float(),
            log_geometric_mean,
            log_anisotropy,
        ],
        dim=-1,
    )


def apply_log_scale_to_variance(
    raw_variance: torch.Tensor,
    log_scale: torch.Tensor,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
) -> torch.Tensor:
    if raw_variance.ndim != 2 or raw_variance.shape[-1] != 2:
        raise ValueError("raw_variance must have shape (M,2)")
    if log_scale.shape != raw_variance.shape[:-1]:
        raise ValueError("log_scale must have shape (M,)")
    min_variance = float(min_std_px) ** 2
    max_variance = float(max_std_px) ** 2
    safe_variance = raw_variance.float().clamp(min_variance, max_variance)
    return (safe_variance * log_scale.exp().unsqueeze(-1)).clamp(
        min_variance, max_variance
    )


def _standardize(
    inputs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (inputs - mean) / std.clamp_min(1e-6)


def fit_conditional_shared_scale_calibration(
    calibration_cache: Mapping[str, torch.Tensor],
    raw_variance: torch.Tensor,
    initial_scale: float,
    seed: int,
    device: str = "cpu",
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
    steps: int = CONDITIONAL_CALIBRATION_STEPS,
    learning_rate: float = CONDITIONAL_CALIBRATION_LEARNING_RATE,
    l2_penalty: float = CONDITIONAL_CALIBRATION_L2,
    min_scale: float = CONDITIONAL_CALIBRATION_MIN_SCALE,
    max_scale: float = CONDITIONAL_CALIBRATION_MAX_SCALE,
) -> Dict[str, object]:
    """Fit the fixed A2 affine conditional calibrator on calibration rows only."""
    _validate_cache_and_variance(calibration_cache, raw_variance)
    if not math.isfinite(float(initial_scale)) or initial_scale <= 0.0:
        raise ValueError("initial_scale must be finite and positive")
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0 or l2_penalty < 0.0:
        raise ValueError("learning_rate must be positive and l2_penalty nonnegative")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch_device = torch.device(device)

    raw_inputs = build_conditional_calibration_inputs(
        calibration_cache,
        raw_variance,
        min_std_px=min_std_px,
        max_std_px=max_std_px,
    )
    input_mean = raw_inputs.mean(dim=0)
    input_std = raw_inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    inputs = _standardize(raw_inputs, input_mean, input_std).to(torch_device)
    errors = calibration_cache["errors_px"].detach().float().to(torch_device)
    raw_variance_device = raw_variance.detach().float().to(torch_device)

    model = ConditionalSharedScaleCalibrator(
        input_dim=raw_inputs.shape[-1],
        min_scale=min_scale,
        max_scale=max_scale,
    ).to(torch_device)
    with torch.no_grad():
        model.affine.weight.zero_()
        initial_log_scale = math.log(
            min(max(float(initial_scale), float(min_scale)), float(max_scale))
        )
        model.affine.bias.fill_(initial_log_scale)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=0.0,
    )

    def calibration_nll() -> torch.Tensor:
        log_scale = model(inputs)
        variance = apply_log_scale_to_variance(
            raw_variance_device,
            log_scale,
            min_std_px=min_std_px,
            max_std_px=max_std_px,
        )
        return diagonal_gaussian_nll(errors, variance.log(), reduction="mean")

    model.eval()
    with torch.no_grad():
        initial_nll = float(calibration_nll().item())
    best_nll = initial_nll
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    trace = [{"step": 0, "calibration_nll": initial_nll}]

    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        nll = calibration_nll()
        penalty = float(l2_penalty) * model.affine.weight.square().sum()
        loss = nll + penalty
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            current_nll = float(calibration_nll().item())
        if current_nll < best_nll:
            best_nll = current_nll
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        if step == 1 or step % 25 == 0 or step == int(steps):
            trace.append({"step": step, "calibration_nll": current_nll})
        model.train()

    model.load_state_dict(best_state)
    cpu_state = {key: value.detach().cpu() for key, value in best_state.items()}
    return {
        "format_version": 1,
        "kind": "conditional_affine_shared_variance_scale",
        "seed": int(seed),
        "input_dim": int(raw_inputs.shape[-1]),
        "input_mean": input_mean.cpu(),
        "input_std": input_std.cpu(),
        "model_state": cpu_state,
        "min_std_px": float(min_std_px),
        "max_std_px": float(max_std_px),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "weight_decay": 0.0,
        "l2_penalty": float(l2_penalty),
        "initial_scale": float(initial_scale),
        "initial_calibration_nll": float(initial_nll),
        "best_calibration_nll": float(best_nll),
        "best_step": int(best_step),
        "training_trace": trace,
    }


def apply_conditional_shared_scale_calibration(
    cache: Mapping[str, torch.Tensor],
    raw_variance: torch.Tensor,
    state: Mapping[str, object],
    device: str = "cpu",
) -> torch.Tensor:
    """Apply a frozen A2 calibrator without fitting or reading labels."""
    _validate_cache_and_variance(cache, raw_variance)
    torch_device = torch.device(device)
    inputs = build_conditional_calibration_inputs(
        cache,
        raw_variance,
        min_std_px=float(state["min_std_px"]),
        max_std_px=float(state["max_std_px"]),
    )
    standardized = _standardize(
        inputs,
        state["input_mean"].float(),
        state["input_std"].float(),
    ).to(torch_device)
    model = ConditionalSharedScaleCalibrator(
        input_dim=int(state["input_dim"]),
        min_scale=float(state["min_scale"]),
        max_scale=float(state["max_scale"]),
    ).to(torch_device)
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        log_scale = model(standardized)
        calibrated = apply_log_scale_to_variance(
            raw_variance.float().to(torch_device),
            log_scale,
            min_std_px=float(state["min_std_px"]),
            max_std_px=float(state["max_std_px"]),
        )
    return calibrated.cpu()


def select_conditional_candidate(
    scalar_validation_nll: float,
    conditional_validation_nll: float,
    required_relative_improvement: float = CONDITIONAL_SELECTION_RELATIVE_NLL,
) -> Dict[str, object]:
    """Apply the fixed A2 1% validation-NLL selection rule."""
    reference = float(scalar_validation_nll)
    candidate = float(conditional_validation_nll)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("Validation NLL values must be finite")
    denominator = max(abs(reference), 1e-12)
    relative_improvement = (reference - candidate) / denominator
    selected = (
        "conditional_affine"
        if relative_improvement >= float(required_relative_improvement)
        else "shared_scalar"
    )
    return {
        "selected": selected,
        "scalar_validation_nll": reference,
        "conditional_validation_nll": candidate,
        "relative_nll_improvement": float(relative_improvement),
        "required_relative_improvement": float(required_relative_improvement),
    }
