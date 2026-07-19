"""Mechanistic coordinate basin audit for Route-D state reinstantiation.

The helpers in this module are deliberately model-free.  They define the
pre-registered offset geometry and aggregate future-rollout rows by video so
that a large number of point/direction rows from one video cannot masquerade
as independent evidence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch


BASIN_SCHEMA_VERSION = "routeD_state_reinstantiation_basin_gate2_5_v0"


def build_offset_hypotheses(
    radii_px: Sequence[float],
    directions_xy: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """Create a deterministic center-plus-radial offset schedule.

    Direction vectors are normalized here, so the configured radius is the
    Euclidean commit error away from image boundaries.  Radius zero appears
    exactly once and is named ``center``.
    """
    radii = [float(value) for value in radii_px]
    if not radii or radii[0] != 0.0 or any(value < 0.0 for value in radii):
        raise ValueError("radii must be non-negative and begin with zero")
    if radii != sorted(set(radii)):
        raise ValueError("radii must be strictly increasing and unique")
    normalized: list[tuple[str, np.ndarray]] = []
    for name, vector in directions_xy.items():
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"invalid direction vector: {name}")
        norm = float(np.linalg.norm(value))
        if norm <= 0.0:
            raise ValueError(f"zero direction vector: {name}")
        normalized.append((str(name), value / norm))
    if not normalized:
        raise ValueError("at least one nonzero direction is required")

    rows: list[dict[str, Any]] = [
        {
            "hypothesis_id": "r0_center",
            "nominal_radius_px": 0.0,
            "direction": "center",
            "offset_xy_px": [0.0, 0.0],
        }
    ]
    for radius in radii[1:]:
        for name, unit in normalized:
            offset = radius * unit
            rows.append(
                {
                    "hypothesis_id": f"r{radius:g}_{name}",
                    "nominal_radius_px": radius,
                    "direction": name,
                    "offset_xy_px": [float(offset[0]), float(offset[1])],
                }
            )
    return rows


def offset_and_clip_coordinates(
    teacher_coordinates_xy: torch.Tensor,
    offset_xy_px: Sequence[float],
    *,
    input_height: int = 256,
    input_width: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an offset, clip to the raster, and report realized displacement.

    Returns ``candidate_xy``, per-row realized error, and a per-row boundary
    clipping mask.  Reporting realized error is essential for points close to
    the image boundary, where nominal and actual radii differ.
    """
    if teacher_coordinates_xy.ndim != 2 or teacher_coordinates_xy.shape[-1] != 2:
        raise ValueError("teacher coordinates must have shape [N,2]")
    offset = teacher_coordinates_xy.new_tensor(offset_xy_px)
    if offset.shape != (2,):
        raise ValueError("offset must contain x and y")
    raw = teacher_coordinates_xy + offset
    candidate = raw.clone()
    candidate[:, 0].clamp_(0.0, float(input_width - 1))
    candidate[:, 1].clamp_(0.0, float(input_height - 1))
    realized = torch.linalg.vector_norm(candidate - teacher_coordinates_xy, dim=-1)
    clipped = (candidate != raw).any(dim=-1)
    return candidate, realized, clipped


def video_cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap equal-weighted video means instead of treating points as IID."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(float(row[value_key]))
    if not grouped:
        raise ValueError("cluster bootstrap requires non-empty rows")
    video_means = np.asarray(
        [np.mean(grouped[key], dtype=np.float64) for key in sorted(grouped)],
        dtype=np.float64,
    )
    if not np.isfinite(video_means).all():
        raise ValueError("cluster bootstrap values must be finite")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, video_means.size, size=(int(samples), video_means.size)
    )
    means = video_means[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(video_means.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "videos": int(video_means.size),
        "samples": int(samples),
        "seed": int(seed),
    }


def aggregate_basin_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Aggregate future outcomes for every probability variant and radius."""
    if not rows:
        raise ValueError("basin rows must be non-empty")
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        row["error_reduction_px"] = float(row["native_mean_l2_error_px"]) - float(
            row["mean_l2_error_px"]
        )
        row["utility_gain"] = float(row["threshold_utility"]) - float(
            row["native_threshold_utility"]
        )
        row["severe_16px_rate_reduction"] = float(
            row["native_severe_16px_rate"]
        ) - float(row["severe_16px_rate"])
        row["positive"] = float(row["error_reduction_px"] > 0.0)
        grouped[(str(row["variant"]), float(row["nominal_radius_px"]))].append(row)

    variants: dict[str, dict[str, Any]] = defaultdict(dict)
    ordered = sorted(grouped, key=lambda value: (value[0], value[1]))
    for ordinal, (variant, radius) in enumerate(ordered):
        group = grouped[(variant, radius)]
        mean = lambda key: float(np.mean([float(row[key]) for row in group]))
        variants[variant][f"{radius:g}"] = {
            "nominal_radius_px": radius,
            "rows": len(group),
            "videos": len({int(row["source_index"]) for row in group}),
            "boundary_clipped_fraction": mean("boundary_clipped"),
            "realized_commit_error_px_mean": mean("realized_commit_error_px"),
            "realized_commit_error_px_median": float(
                np.median([float(row["realized_commit_error_px"]) for row in group])
            ),
            "mean_l2_error_px": mean("mean_l2_error_px"),
            "severe_16px_rate": mean("severe_16px_rate"),
            "threshold_utility": mean("threshold_utility"),
            "mean_error_reduction_vs_native_px": mean("error_reduction_px"),
            "threshold_utility_gain_vs_native": mean("utility_gain"),
            "severe_16px_rate_reduction_vs_native": mean(
                "severe_16px_rate_reduction"
            ),
            "positive_row_fraction": mean("positive"),
            "error_reduction_video_cluster_CI": video_cluster_bootstrap_ci(
                group,
                value_key="error_reduction_px",
                seed=int(bootstrap_seed) + 2 * ordinal,
                samples=int(bootstrap_samples),
            ),
            "utility_gain_video_cluster_CI": video_cluster_bootstrap_ci(
                group,
                value_key="utility_gain",
                seed=int(bootstrap_seed) + 2 * ordinal + 1,
                samples=int(bootstrap_samples),
            ),
        }
    return {"variants": dict(variants)}


def evaluate_radius_gates(
    aggregate: Mapping[str, Any], gates: Mapping[str, float]
) -> dict[str, Any]:
    """Find the largest radius satisfying the frozen downstream utility gates."""
    output: dict[str, Any] = {}
    for variant, radius_rows in aggregate["variants"].items():
        checks_by_radius: dict[str, dict[str, bool]] = {}
        passing: list[float] = []
        ordered_rows = sorted(
            radius_rows.items(), key=lambda item: float(item[1]["nominal_radius_px"])
        )
        for radius_key, row in ordered_rows:
            checks = {
                "mean_error_reduction_px_min": float(
                    row["mean_error_reduction_vs_native_px"]
                )
                >= float(gates["mean_error_reduction_px_min"]),
                "error_reduction_CI_lower_px_min": float(
                    row["error_reduction_video_cluster_CI"]["lower"]
                )
                >= float(gates["error_reduction_CI_lower_px_min"]),
                "utility_gain_min": float(row["threshold_utility_gain_vs_native"])
                >= float(gates["utility_gain_min"]),
                "utility_gain_CI_lower_min": float(
                    row["utility_gain_video_cluster_CI"]["lower"]
                )
                >= float(gates["utility_gain_CI_lower_min"]),
                "positive_fraction_min": float(row["positive_row_fraction"])
                >= float(gates["positive_fraction_min"]),
                "severe_rate_reduction_min": float(
                    row["severe_16px_rate_reduction_vs_native"]
                )
                >= float(gates["severe_rate_reduction_min"]),
            }
            checks["pass"] = all(checks.values())
            checks_by_radius[radius_key] = checks
            if checks["pass"]:
                passing.append(float(row["nominal_radius_px"]))
        contiguous: list[float] = []
        for radius_key, row in ordered_rows:
            if not checks_by_radius[radius_key]["pass"]:
                break
            contiguous.append(float(row["nominal_radius_px"]))
        output[variant] = {
            "checks_by_radius": checks_by_radius,
            "passing_nominal_radii_px": passing,
            "contiguous_passing_nominal_radii_px": contiguous,
            "largest_passing_nominal_radius_px": (
                max(contiguous) if contiguous else None
            ),
            "nonmonotonic_passing_radii_excluded": [
                value for value in passing if value not in contiguous
            ],
        }
    return output
