from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .calibration_metrics import calibration_report
from .uncertainty_head import (
    MMP_UNCERTAINTY_FEATURE_DIM,
    MMP_UNCERTAINTY_VALUE_NAMES,
    DiagonalGaussianUncertaintyHead,
    build_mmp_uncertainty_features,
    diagonal_gaussian_nll,
)


CACHE_TENSOR_KEYS: Tuple[str, ...] = (
    "features",
    "mu_px",
    "gt_px",
    "errors_px",
    "pred_visibility",
    "gt_visible",
    "predicted_occluded_duration",
    "gt_occluded_duration",
    "reentry_age",
    "sample_id",
)


def set_beliefcal_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def ground_truth_temporal_state(occluded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return current GT occlusion duration and visible re-entry age."""
    if occluded.ndim != 3:
        raise ValueError("occluded must have shape (B, N, T).")
    occ_duration = torch.zeros_like(occluded, dtype=torch.long)
    reentry_age = torch.zeros_like(occluded, dtype=torch.long)
    running_occ = torch.zeros(occluded.shape[:2], dtype=torch.long, device=occluded.device)
    running_reentry = torch.zeros_like(running_occ)
    for t in range(occluded.shape[-1]):
        occ_t = occluded[..., t].bool()
        just_reentered = (~occ_t) & (running_occ > 0)
        running_reentry = torch.where(
            just_reentered,
            torch.ones_like(running_reentry),
            torch.where(
                (~occ_t) & (running_reentry > 0),
                running_reentry + 1,
                torch.zeros_like(running_reentry),
            ),
        )
        running_occ = torch.where(occ_t, running_occ + 1, torch.zeros_like(running_occ))
        occ_duration[..., t] = running_occ
        reentry_age[..., t] = running_reentry
    return occ_duration, reentry_age


def _resolution_scale(info: Mapping[str, object], reference: torch.Tensor) -> torch.Tensor:
    height = info.get("input_height")
    width = info.get("input_width")
    if height is None or width is None:
        raise ValueError("MMP info must contain input_height and input_width.")
    return torch.tensor(
        [max(float(height) - 1.0, 1.0), max(float(width) - 1.0, 1.0)],
        dtype=reference.dtype,
        device=reference.device,
    )


def extract_beliefcal_cache_batch(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    info: Dict[str, object],
    gt_tracks: torch.Tensor,
    occluded: torch.Tensor,
    sample_ids: Optional[torch.Tensor] = None,
    visibility_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Convert one frozen MMP batch into flattened strict-MVP cache rows."""
    if pred_tracks.shape != gt_tracks.shape or pred_tracks.shape[-1] != 2:
        raise ValueError("pred_tracks and gt_tracks must have identical (B,N,T,2) shapes.")
    if pred_visibility.shape != pred_tracks.shape[:-1]:
        raise ValueError("pred_visibility must have shape (B,N,T).")
    if occluded.shape != pred_visibility.shape:
        raise ValueError("occluded must have shape (B,N,T).")

    features, temporal = build_mmp_uncertainty_features(
        info,
        pred_visibility,
        visibility_threshold=visibility_threshold,
    )
    scale = _resolution_scale(info, pred_tracks)
    mu_px = pred_tracks.detach() * scale
    gt_px = gt_tracks.detach() * scale
    errors_px = gt_px - mu_px
    gt_occ_duration, reentry_age = ground_truth_temporal_state(occluded.bool())

    active = temporal["active_mask"].bool()
    if active.shape != pred_visibility.shape:
        raise ValueError("active mask shape mismatch.")

    batch, points, time = pred_visibility.shape
    if sample_ids is None:
        sample_ids = torch.arange(batch, device=pred_tracks.device, dtype=torch.long)
    if sample_ids.shape != (batch,):
        raise ValueError("sample_ids must have shape (B,).")
    sample_grid = sample_ids[:, None, None].expand(batch, points, time)

    cache = {
        "features": features[active].detach().cpu(),
        "mu_px": mu_px[active].detach().cpu(),
        "gt_px": gt_px[active].detach().cpu(),
        "errors_px": errors_px[active].detach().cpu(),
        "pred_visibility": pred_visibility.detach()[active].float().cpu(),
        "gt_visible": (~occluded.bool())[active].float().cpu(),
        "predicted_occluded_duration": temporal["predicted_occluded_duration"][active].long().cpu(),
        "gt_occluded_duration": gt_occ_duration[active].long().cpu(),
        "reentry_age": reentry_age[active].long().cpu(),
        "sample_id": sample_grid[active].long().cpu(),
    }
    validate_beliefcal_cache(cache)
    return cache


def merge_beliefcal_cache_batches(batches: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("At least one cache batch is required.")
    merged = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in CACHE_TENSOR_KEYS}
    validate_beliefcal_cache(merged)
    return merged


def validate_beliefcal_cache(cache: Mapping[str, torch.Tensor]) -> None:
    missing = [key for key in CACHE_TENSOR_KEYS if key not in cache]
    if missing:
        raise ValueError(f"BeliefCal cache is missing keys: {missing}")
    count = int(cache["features"].shape[0])
    for key in CACHE_TENSOR_KEYS:
        if int(cache[key].shape[0]) != count:
            raise ValueError(f"Cache key {key!r} has inconsistent row count.")
    if cache["features"].ndim != 2 or cache["features"].shape[-1] != MMP_UNCERTAINTY_FEATURE_DIM:
        raise ValueError(
            f"features must have shape (M, {MMP_UNCERTAINTY_FEATURE_DIM})."
        )
    for key in ("mu_px", "gt_px", "errors_px"):
        if cache[key].ndim != 2 or cache[key].shape[-1] != 2:
            raise ValueError(f"{key} must have shape (M, 2).")
    if not torch.isfinite(cache["features"]).all():
        raise ValueError("Cache contains non-finite features.")
    if not torch.isfinite(cache["errors_px"]).all():
        raise ValueError("Cache contains non-finite errors.")


def save_beliefcal_cache(
    cache: Mapping[str, torch.Tensor],
    path: str | Path,
    manifest: Optional[Mapping[str, object]] = None,
) -> None:
    validate_beliefcal_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "manifest": dict(manifest or {}),
        "cache": {key: cache[key].cpu() for key in CACHE_TENSOR_KEYS},
    }
    torch.save(payload, path)


def load_beliefcal_cache(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "cache" not in payload:
        raise ValueError("Invalid BeliefCal cache payload.")
    cache = payload["cache"]
    validate_beliefcal_cache(cache)
    return dict(cache), dict(payload.get("manifest", {}))


def _clamp_variance(value: torch.Tensor, min_std_px: float, max_std_px: float) -> torch.Tensor:
    return value.clamp(float(min_std_px) ** 2, float(max_std_px) ** 2)


def fit_grouped_isotropic_variance(
    errors_px: torch.Tensor,
    groups: torch.Tensor,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
) -> Dict[str, object]:
    errors_px = errors_px.reshape(-1, 2).float()
    groups = groups.reshape(-1).long()
    if errors_px.shape[0] != groups.shape[0]:
        raise ValueError("errors and groups must have the same row count.")
    global_var = _clamp_variance(errors_px.square().mean(), min_std_px, max_std_px)
    values: Dict[int, float] = {}
    for group in torch.unique(groups).tolist():
        mask = groups == int(group)
        if int(mask.sum()) < 2:
            values[int(group)] = float(global_var.item())
        else:
            values[int(group)] = float(
                _clamp_variance(errors_px[mask].square().mean(), min_std_px, max_std_px).item()
            )
    return {
        "global_variance": float(global_var.item()),
        "group_variance": values,
        "min_std_px": float(min_std_px),
        "max_std_px": float(max_std_px),
    }


def apply_grouped_isotropic_variance(
    groups: torch.Tensor,
    state: Mapping[str, object],
) -> torch.Tensor:
    groups = groups.reshape(-1).long()
    default = float(state["global_variance"])
    mapping = {int(k): float(v) for k, v in dict(state["group_variance"]).items()}
    variance = torch.full((groups.shape[0],), default, dtype=torch.float32)
    for group, value in mapping.items():
        variance[groups == group] = float(value)
    return variance[:, None].expand(-1, 2).clone()


def visibility_groups(cache: Mapping[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
    return (cache["pred_visibility"] >= float(threshold)).long()


def predicted_occlusion_duration_groups(cache: Mapping[str, torch.Tensor]) -> torch.Tensor:
    duration = cache["predicted_occluded_duration"].long()
    groups = torch.zeros_like(duration)
    groups[(duration >= 1) & (duration <= 5)] = 1
    groups[(duration >= 6) & (duration <= 20)] = 2
    groups[duration >= 21] = 3
    return groups


def baseline_support_diagnostics(
    cache: Mapping[str, torch.Tensor],
    visibility_threshold: float = 0.5,
) -> Dict[str, object]:
    """Summarize whether the preregistered grouped baselines have support."""
    validate_beliefcal_cache(cache)
    visibility = cache["pred_visibility"].reshape(-1).float()
    duration = cache["predicted_occluded_duration"].reshape(-1).long()
    visibility_group = visibility_groups(cache, visibility_threshold)
    duration_group = predicted_occlusion_duration_groups(cache)

    def counts(groups: torch.Tensor) -> Dict[str, int]:
        values, value_counts = torch.unique(groups.long(), return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(values.tolist(), value_counts.tolist())}

    visibility_counts = counts(visibility_group)
    duration_counts = counts(duration_group)
    warnings: List[str] = []
    if len(visibility_counts) < 2:
        warnings.append(
            "visibility_conditioned baseline is degenerate: calibration rows occupy one group"
        )
    if len(duration_counts) < 2:
        warnings.append(
            "pred_occ_duration_binned baseline is degenerate: calibration rows occupy one group"
        )
    return {
        "rows": int(visibility.numel()),
        "visibility_threshold": float(visibility_threshold),
        "pred_visibility_min": float(visibility.min().item()),
        "pred_visibility_max": float(visibility.max().item()),
        "pred_visibility_mean": float(visibility.mean().item()),
        "visibility_group_counts": visibility_counts,
        "predicted_occluded_duration_min": int(duration.min().item()),
        "predicted_occluded_duration_max": int(duration.max().item()),
        "predicted_occluded_duration_mean": float(duration.float().mean().item()),
        "duration_group_counts": duration_counts,
        "warnings": warnings,
    }


def fit_required_posthoc_baselines(
    calibration_cache: Mapping[str, torch.Tensor],
    visibility_threshold: float = 0.5,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
) -> Dict[str, Dict[str, object]]:
    validate_beliefcal_cache(calibration_cache)
    errors = calibration_cache["errors_px"]
    global_groups = torch.zeros(errors.shape[0], dtype=torch.long)
    return {
        "global_scalar": fit_grouped_isotropic_variance(
            errors, global_groups, min_std_px, max_std_px
        ),
        "visibility_conditioned": fit_grouped_isotropic_variance(
            errors,
            visibility_groups(calibration_cache, visibility_threshold),
            min_std_px,
            max_std_px,
        ),
        "pred_occ_duration_binned": fit_grouped_isotropic_variance(
            errors,
            predicted_occlusion_duration_groups(calibration_cache),
            min_std_px,
            max_std_px,
        ),
    }


def predict_posthoc_baseline_variance(
    cache: Mapping[str, torch.Tensor],
    method: str,
    state: Mapping[str, object],
    visibility_threshold: float = 0.5,
) -> torch.Tensor:
    if method == "global_scalar":
        groups = torch.zeros(cache["errors_px"].shape[0], dtype=torch.long)
    elif method == "visibility_conditioned":
        groups = visibility_groups(cache, visibility_threshold)
    elif method == "pred_occ_duration_binned":
        groups = predicted_occlusion_duration_groups(cache)
    else:
        raise ValueError(f"Unsupported post-hoc method: {method}")
    return apply_grouped_isotropic_variance(groups, state)


def _standardize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) / std.clamp_min(1e-6)


def uncertainty_feature_names() -> Tuple[str, ...]:
    values = tuple(MMP_UNCERTAINTY_VALUE_NAMES)
    return values + tuple(f"{name}_available" for name in values)


def resolve_uncertainty_feature_selection(
    profile: str = "full",
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Return a fixed, auditable feature selection for a diagnostic ablation.

    ``drop_inert_mmp`` removes only the three value channels observed to be
    constant in the current frozen localglobal+heuristic execution path, plus
    their matching availability bits. It is an engineering ablation, not a
    replacement for the preregistered full feature contract.
    """
    profile = str(profile).strip().lower()
    names = uncertainty_feature_names()
    if profile == "full":
        indices = tuple(range(len(names)))
    elif profile == "drop_inert_mmp":
        dropped_values = {
            "selected_global",
            "active",
            "predicted_occluded_duration",
        }
        dropped = dropped_values | {f"{name}_available" for name in dropped_values}
        indices = tuple(index for index, name in enumerate(names) if name not in dropped)
    else:
        raise ValueError(
            f"Unsupported feature profile {profile!r}; expected full|drop_inert_mmp"
        )
    return indices, tuple(names[index] for index in indices)


def _select_features(
    features: torch.Tensor, feature_indices: Sequence[int]
) -> torch.Tensor:
    indices = torch.as_tensor(tuple(int(index) for index in feature_indices), dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("At least one uncertainty feature must be selected.")
    if int(indices.min()) < 0 or int(indices.max()) >= int(features.shape[-1]):
        raise ValueError("Feature selection index is outside the cache feature dimension.")
    return features.index_select(-1, indices)


def train_uncertainty_head(
    train_cache: Mapping[str, torch.Tensor],
    validation_cache: Mapping[str, torch.Tensor],
    seed: int,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
    device: str = "cpu",
    feature_profile: str = "full",
) -> Dict[str, object]:
    validate_beliefcal_cache(train_cache)
    validate_beliefcal_cache(validation_cache)
    set_beliefcal_seed(seed)
    torch_device = torch.device(device)

    feature_indices, selected_feature_names = resolve_uncertainty_feature_selection(
        feature_profile
    )
    train_features = _select_features(
        train_cache["features"].float(), feature_indices
    )
    train_errors = train_cache["errors_px"].float()
    val_features = _select_features(
        validation_cache["features"].float(), feature_indices
    )
    val_errors = validation_cache["errors_px"].float()
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    head = DiagonalGaussianUncertaintyHead(
        input_dim=train_features.shape[-1],
        hidden_dim=hidden_dim,
        min_std_px=min_std_px,
        max_std_px=max_std_px,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    best_state = copy.deepcopy(head.state_dict())
    best_val_nll = float("inf")
    history: List[Dict[str, float]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batch_size = max(1, int(batch_size))

    for epoch in range(int(epochs)):
        head.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        train_total = 0.0
        train_count = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            x = _standardize_features(
                train_features[index], feature_mean, feature_std
            ).to(torch_device)
            error = train_errors[index].to(torch_device)
            output = head(x)
            loss = diagonal_gaussian_nll(error, output["log_var"], reduction="mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * int(index.numel())
            train_count += int(index.numel())

        head.eval()
        with torch.no_grad():
            val_x = _standardize_features(val_features, feature_mean, feature_std).to(torch_device)
            val_error = val_errors.to(torch_device)
            val_log_var = head(val_x)["log_var"]
            val_nll = float(
                diagonal_gaussian_nll(val_error, val_log_var, reduction="mean").item()
            )
        train_nll = train_total / max(train_count, 1)
        history.append({"epoch": float(epoch), "train_nll": train_nll, "val_nll": val_nll})
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = copy.deepcopy(head.state_dict())

    return {
        "seed": int(seed),
        "feature_profile": str(feature_profile),
        "feature_indices": tuple(int(index) for index in feature_indices),
        "selected_feature_names": tuple(selected_feature_names),
        "input_dim": int(train_features.shape[-1]),
        "hidden_dim": int(hidden_dim),
        "min_std_px": float(min_std_px),
        "max_std_px": float(max_std_px),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model_state": best_state,
        "best_val_nll": float(best_val_nll),
        "history": history,
    }


def predict_learned_variance(
    cache: Mapping[str, torch.Tensor],
    state: Mapping[str, object],
    device: str = "cpu",
) -> torch.Tensor:
    validate_beliefcal_cache(cache)
    torch_device = torch.device(device)
    head = DiagonalGaussianUncertaintyHead(
        input_dim=int(state["input_dim"]),
        hidden_dim=int(state["hidden_dim"]),
        min_std_px=float(state["min_std_px"]),
        max_std_px=float(state["max_std_px"]),
    ).to(torch_device)
    head.load_state_dict(state["model_state"])
    head.eval()
    with torch.no_grad():
        feature_indices = state.get(
            "feature_indices", tuple(range(cache["features"].shape[-1]))
        )
        selected_features = _select_features(
            cache["features"].float(), feature_indices
        )
        x = _standardize_features(
            selected_features,
            state["feature_mean"].float(),
            state["feature_std"].float(),
        ).to(torch_device)
        variance = head(x)["variance"].cpu()
    return variance


def fit_scalar_variance_calibration(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
    min_std_px: float = 0.25,
    max_std_px: float = 256.0,
) -> Dict[str, float]:
    """Fit one positive variance scale on the calibration split.

    For an unconstrained diagonal Gaussian, the NLL-optimal shared scale is
    mean(error^2 / variance) across samples and coordinate dimensions.  The
    final variance is clipped to the preregistered standard-deviation bounds.
    """
    if errors_px.shape != variance.shape or errors_px.ndim != 2 or errors_px.shape[-1] != 2:
        raise ValueError("errors_px and variance must have identical (M, 2) shapes.")
    if errors_px.numel() == 0:
        raise ValueError("Cannot calibrate variance on an empty tensor.")
    min_variance = float(min_std_px) ** 2
    max_variance = float(max_std_px) ** 2
    safe_variance = variance.detach().float().clamp(min_variance, max_variance)
    ratio = errors_px.detach().float().square() / safe_variance
    candidate_scale = float(ratio.mean().clamp(1e-6, 1e6).item())
    before_log_var = safe_variance.log()
    candidate_variance = (safe_variance * candidate_scale).clamp(
        min_variance, max_variance
    )
    before_nll = float(
        diagonal_gaussian_nll(errors_px.float(), before_log_var, reduction="mean").item()
    )
    candidate_nll = float(
        diagonal_gaussian_nll(
            errors_px.float(), candidate_variance.log(), reduction="mean"
        ).item()
    )
    # Clipping can make the unconstrained closed-form candidate suboptimal.
    # Never replace the raw head with a calibration transform that worsens NLL
    # on the calibration split.
    if candidate_nll <= before_nll:
        scale = candidate_scale
        after_nll = candidate_nll
    else:
        scale = 1.0
        after_nll = before_nll
    return {
        "kind": "shared_variance_scale",
        "candidate_scale": candidate_scale,
        "scale": scale,
        "min_std_px": float(min_std_px),
        "max_std_px": float(max_std_px),
        "calibration_nll_before": before_nll,
        "calibration_nll_after": after_nll,
    }


def apply_scalar_variance_calibration(
    variance: torch.Tensor,
    state: Mapping[str, object],
) -> torch.Tensor:
    if variance.ndim != 2 or variance.shape[-1] != 2:
        raise ValueError("variance must have shape (M, 2).")
    min_variance = float(state["min_std_px"]) ** 2
    max_variance = float(state["max_std_px"]) ** 2
    safe_variance = variance.float().clamp(min_variance, max_variance)
    return (safe_variance * float(state["scale"])).clamp(
        min_variance, max_variance
    )


def evaluation_strata(cache: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gt_visible = cache["gt_visible"] > 0.5
    reentry = cache["reentry_age"].long()
    gt_occ = cache["gt_occluded_duration"].long()
    return {
        "overall": torch.ones_like(gt_visible, dtype=torch.bool),
        "visible": gt_visible,
        "occluded": ~gt_visible,
        "short_occlusion_1_5": (~gt_visible) & (gt_occ >= 1) & (gt_occ <= 5),
        "medium_occlusion_6_20": (~gt_visible) & (gt_occ >= 6) & (gt_occ <= 20),
        "long_occlusion_21_plus": (~gt_visible) & (gt_occ >= 21),
        "reentry_1_5": gt_visible & (reentry >= 1) & (reentry <= 5),
        "reentry_6_20": gt_visible & (reentry >= 6) & (reentry <= 20),
    }


def evaluate_variance_method(
    cache: Mapping[str, torch.Tensor],
    variance: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    validate_beliefcal_cache(cache)
    if variance.shape != cache["errors_px"].shape:
        raise ValueError("variance must have shape (M, 2).")
    report = {}
    for name, mask in evaluation_strata(cache).items():
        report[name] = calibration_report(cache["errors_px"], variance, mask=mask)
    return report


def run_beliefcal_mvp1_experiment(
    train_cache: Mapping[str, torch.Tensor],
    calibration_cache: Mapping[str, torch.Tensor],
    validation_cache: Mapping[str, torch.Tensor],
    test_cache: Mapping[str, torch.Tensor],
    seeds: Sequence[int] = (17, 29, 43),
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    feature_profile: str = "full",
) -> Dict[str, object]:
    for cache in (train_cache, calibration_cache, validation_cache, test_cache):
        validate_beliefcal_cache(cache)

    baselines = fit_required_posthoc_baselines(calibration_cache)
    baseline_reports: Dict[str, object] = {}
    for name, state in baselines.items():
        variance = predict_posthoc_baseline_variance(test_cache, name, state)
        baseline_reports[name] = {
            "state": state,
            "metrics": evaluate_variance_method(test_cache, variance),
        }

    learned_runs = []
    for seed in seeds:
        state = train_uncertainty_head(
            train_cache,
            validation_cache,
            seed=int(seed),
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            feature_profile=feature_profile,
        )
        calibration_variance = predict_learned_variance(
            calibration_cache, state, device=device
        )
        calibration_state = fit_scalar_variance_calibration(
            calibration_cache["errors_px"],
            calibration_variance,
            min_std_px=float(state["min_std_px"]),
            max_std_px=float(state["max_std_px"]),
        )
        raw_test_variance = predict_learned_variance(test_cache, state, device=device)
        calibrated_test_variance = apply_scalar_variance_calibration(
            raw_test_variance, calibration_state
        )
        learned_runs.append(
            {
                "seed": int(seed),
                "feature_profile": state["feature_profile"],
                "selected_feature_names": state["selected_feature_names"],
                "best_val_nll": float(state["best_val_nll"]),
                "raw_metrics": evaluate_variance_method(
                    test_cache, raw_test_variance
                ),
                "calibration_state": calibration_state,
                "metrics": evaluate_variance_method(
                    test_cache, calibrated_test_variance
                ),
                "state": state,
            }
        )

    return {
        "format_version": 3,
        "feature_profile": str(feature_profile),
        "support_diagnostics": {
            "calibration": baseline_support_diagnostics(calibration_cache),
            "test": baseline_support_diagnostics(test_cache),
        },
        "baselines": baseline_reports,
        "learned_runs": learned_runs,
    }


def synthetic_beliefcal_cache(
    count: int,
    seed: int,
    sample_offset: int = 0,
) -> Dict[str, torch.Tensor]:
    """Deterministic heteroscedastic cache for end-to-end pipeline tests."""
    generator = torch.Generator().manual_seed(int(seed))
    count = int(count)
    confidence = torch.rand(count, generator=generator)
    pred_visibility = torch.rand(count, generator=generator)
    pred_occ_duration = torch.randint(0, 31, (count,), generator=generator)
    gt_visible = (torch.rand(count, generator=generator) > 0.3).float()
    gt_occ_duration = torch.where(
        gt_visible > 0.5,
        torch.zeros_like(pred_occ_duration),
        torch.randint(1, 31, (count,), generator=generator),
    )
    reentry_age = torch.where(
        gt_visible > 0.5,
        torch.randint(0, 21, (count,), generator=generator),
        torch.zeros_like(pred_occ_duration),
    )

    sigma_y = 0.6 + 5.0 * (1.0 - confidence) + 0.15 * pred_occ_duration.float()
    sigma_x = 0.8 + 3.0 * (1.0 - pred_visibility) + 0.08 * pred_occ_duration.float()
    errors = torch.stack(
        [
            torch.randn(count, generator=generator) * sigma_y,
            torch.randn(count, generator=generator) * sigma_x,
        ],
        dim=-1,
    )

    values = torch.zeros(count, MMP_UNCERTAINTY_FEATURE_DIM // 2)
    values[:, 0] = confidence
    values[:, 1] = pred_visibility
    values[:, 11] = torch.rand(count, generator=generator)
    values[:, 12] = torch.log1p(pred_occ_duration.float()) / math.log1p(30.0)
    availability = torch.ones_like(values)
    features = torch.cat([values, availability], dim=-1)
    mu = torch.rand(count, 2, generator=generator) * 255.0
    gt = mu + errors
    cache = {
        "features": features,
        "mu_px": mu,
        "gt_px": gt,
        "errors_px": errors,
        "pred_visibility": pred_visibility,
        "gt_visible": gt_visible,
        "predicted_occluded_duration": pred_occ_duration,
        "gt_occluded_duration": gt_occ_duration,
        "reentry_age": reentry_age,
        "sample_id": torch.arange(sample_offset, sample_offset + count, dtype=torch.long) // 32,
    }
    validate_beliefcal_cache(cache)
    return cache


def json_safe_experiment_summary(result: Mapping[str, object]) -> Dict[str, object]:
    """Drop tensor-heavy learned states for compact JSON result reporting."""
    summary = {
        "format_version": result.get("format_version", 1),
        "feature_profile": result.get("feature_profile", "full"),
        "support_diagnostics": result.get("support_diagnostics"),
        "baselines": {},
        "learned_runs": [],
    }
    for name, payload in dict(result["baselines"]).items():
        summary["baselines"][name] = {
            "state": payload["state"],
            "metrics": payload["metrics"],
        }
    for run in list(result["learned_runs"]):
        summary["learned_runs"].append(
            {
                "seed": run["seed"],
                "feature_profile": run.get("feature_profile", "full"),
                "selected_feature_names": run.get("selected_feature_names"),
                "best_val_nll": run["best_val_nll"],
                "raw_metrics": run.get("raw_metrics"),
                "calibration_state": run.get("calibration_state"),
                "metrics": run["metrics"],
            }
        )
    return summary


def save_experiment_result(result: Mapping[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(result), output_dir / "beliefcal_result.pt")
    with open(output_dir / "beliefcal_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe_experiment_summary(result), handle, indent=2, ensure_ascii=False)
