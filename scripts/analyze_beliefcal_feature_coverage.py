#!/usr/bin/env python3
"""Compare BeliefCal cache feature support and error relationships.

This is a diagnostic utility. It does not fit a model or inspect test labels for
selection; use it only on development/calibration caches unless the protocol
explicitly permits test-set descriptive reporting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    MMP_UNCERTAINTY_FEATURE_DIM,
    MMP_UNCERTAINTY_VALUE_NAMES,
)


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_cache_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH for --cache")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("Cache label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Cache does not exist: {path}")
    return label, path


def finite_float(value: torch.Tensor | float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def tensor_quantiles(values: torch.Tensor) -> Dict[str, float | None]:
    values = values.reshape(-1).float()
    if values.numel() == 0:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    probs = torch.tensor(QUANTILES, dtype=torch.float32)
    result = torch.quantile(values, probs)
    return {
        f"q{int(q * 100):02d}": finite_float(v)
        for q, v in zip(QUANTILES, result)
    }


def pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom) <= 1e-12:
        return None
    return finite_float((x * y).sum() / denom)


def summarize_cache(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "cache" not in payload:
        raise ValueError(f"Invalid BeliefCal cache: {path}")
    cache = payload["cache"]
    features = cache["features"].float()
    if features.ndim != 2 or features.shape[1] != MMP_UNCERTAINTY_FEATURE_DIM:
        raise ValueError(
            f"Expected features (M,{MMP_UNCERTAINTY_FEATURE_DIM}), got {tuple(features.shape)}"
        )
    value_count = len(MMP_UNCERTAINTY_VALUE_NAMES)
    values = features[:, :value_count]
    availability = features[:, value_count:] > 0.5
    error_norm = cache["errors_px"].float().norm(dim=-1)
    manifest = payload.get("manifest", {})
    manifest_names = manifest.get("feature_names") if isinstance(manifest, dict) else None
    expected_names = list(MMP_UNCERTAINTY_VALUE_NAMES) + [
        f"{name}_available" for name in MMP_UNCERTAINTY_VALUE_NAMES
    ]

    feature_rows = []
    for index, name in enumerate(MMP_UNCERTAINTY_VALUE_NAMES):
        mask = availability[:, index]
        observed = values[mask, index]
        observed_error = error_norm[mask]
        std = observed.std(unbiased=False) if observed.numel() else torch.tensor(float("nan"))
        feature_rows.append(
            {
                "index": index,
                "name": name,
                "available_rows": int(mask.sum()),
                "availability_rate": finite_float(mask.float().mean()),
                "constant_on_available_rows": bool(
                    observed.numel() <= 1 or float(std) <= 1e-12
                ),
                "mean": finite_float(observed.mean()) if observed.numel() else None,
                "std": finite_float(std) if observed.numel() else None,
                "quantiles": tensor_quantiles(observed),
                "pearson_with_error_norm": pearson(observed, observed_error),
            }
        )

    dataset = manifest.get("dataset", {}) if isinstance(manifest, dict) else {}
    return {
        "path": str(path),
        "rows": int(features.shape[0]),
        "videos": int(torch.unique(cache["sample_id"]).numel()),
        "dataset_family": dataset.get("dataset_family") if isinstance(dataset, dict) else None,
        "feature_schema_matches_current": manifest_names in (None, expected_names),
        "error_norm_px": {
            "mean": finite_float(error_norm.mean()),
            "std": finite_float(error_norm.std(unbiased=False)),
            "quantiles": tensor_quantiles(error_norm),
        },
        "gt_visible_rate": finite_float(cache["gt_visible"].float().mean()),
        "pred_visibility": {
            "mean": finite_float(cache["pred_visibility"].float().mean()),
            "quantiles": tensor_quantiles(cache["pred_visibility"].float()),
        },
        "predicted_occluded_duration": {
            "nonzero_rate": finite_float(
                (cache["predicted_occluded_duration"] > 0).float().mean()
            ),
            "maximum": int(cache["predicted_occluded_duration"].max()),
        },
        "features": feature_rows,
    }


def pairwise_comparison(
    label_a: str, summary_a: Dict[str, object], label_b: str, summary_b: Dict[str, object]
) -> Dict[str, object]:
    features_a = {row["name"]: row for row in summary_a["features"]}
    features_b = {row["name"]: row for row in summary_b["features"]}
    shifts = []
    for name in MMP_UNCERTAINTY_VALUE_NAMES:
        a = features_a[name]
        b = features_b[name]
        mean_a, mean_b = a["mean"], b["mean"]
        std_a, std_b = a["std"], b["std"]
        pooled = None
        standardized = None
        if None not in (mean_a, mean_b, std_a, std_b):
            pooled = math.sqrt((float(std_a) ** 2 + float(std_b) ** 2) / 2.0)
            if pooled > 1e-12:
                standardized = (float(mean_b) - float(mean_a)) / pooled
        shifts.append(
            {
                "name": name,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "standardized_mean_shift_b_minus_a": standardized,
                "pearson_error_a": a["pearson_with_error_norm"],
                "pearson_error_b": b["pearson_with_error_norm"],
                "constant_a": a["constant_on_available_rows"],
                "constant_b": b["constant_on_available_rows"],
            }
        )
    shifts.sort(
        key=lambda row: abs(row["standardized_mean_shift_b_minus_a"] or 0.0),
        reverse=True,
    )
    return {
        "labels": [label_a, label_b],
        "feature_shifts": shifts,
        "constant_in_both": [
            row["name"] for row in shifts if row["constant_a"] and row["constant_b"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        type=parse_cache_spec,
        help="Repeatable LABEL=PATH cache specification.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summaries = {label: summarize_cache(path) for label, path in args.cache}
    labels = list(summaries)
    report: Dict[str, object] = {
        "format_version": 1,
        "caches": summaries,
    }
    if len(labels) == 2:
        report["comparison"] = pairwise_comparison(
            labels[0], summaries[labels[0]], labels[1], summaries[labels[1]]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
