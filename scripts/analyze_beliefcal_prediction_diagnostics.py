#!/usr/bin/env python3
"""Descriptive diagnostics for frozen BeliefCal experiment bundles.

The script reconstructs raw and scalar-calibrated variances from saved learned
states, then reports uncertainty-score quantiles and equal-count risk deciles on
calibration and test caches. It does not fit or select any parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    apply_scalar_variance_calibration,
    load_beliefcal_cache,
    predict_learned_variance,
)
from projects.mmp_tracker.mmp_tracker.calibration_metrics import (
    CHI2_DF2_THRESHOLDS,
    calibration_report,
    mahalanobis_sq,
)

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def parse_label_path(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
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
        f"q{int(q * 100):02d}": finite_float(value)
        for q, value in zip(QUANTILES, result)
    }


def equal_count_bin_indices(score: torch.Tensor, bins: int = 10) -> List[torch.Tensor]:
    score = score.reshape(-1)
    if score.numel() == 0:
        raise ValueError("Cannot bin an empty score tensor")
    bins = max(1, min(int(bins), int(score.numel())))
    order = torch.argsort(score, stable=True)
    boundaries = torch.linspace(0, order.numel(), bins + 1).round().long()
    result: List[torch.Tensor] = []
    for index in range(bins):
        start = int(boundaries[index])
        end = int(boundaries[index + 1])
        if end > start:
            result.append(order[start:end])
    return result


def risk_decile_table(
    errors_px: torch.Tensor,
    variance: torch.Tensor,
    bins: int = 10,
) -> List[Dict[str, float | int | None]]:
    errors_px = errors_px.reshape(-1, 2).float()
    variance = variance.reshape(-1, 2).float().clamp_min(1e-8)
    if errors_px.shape != variance.shape:
        raise ValueError("errors_px and variance must have identical (M,2) shapes")
    point_error = torch.linalg.vector_norm(errors_px, dim=-1)
    uncertainty_score = torch.sqrt(variance.prod(dim=-1))
    mahal = mahalanobis_sq(errors_px, variance)
    threshold95 = CHI2_DF2_THRESHOLDS[0.95]

    rows: List[Dict[str, float | int | None]] = []
    for bin_number, index in enumerate(
        equal_count_bin_indices(uncertainty_score, bins=bins), start=1
    ):
        score_bin = uncertainty_score[index]
        error_bin = point_error[index]
        mahal_bin = mahal[index]
        rows.append(
            {
                "bin": bin_number,
                "count": int(index.numel()),
                "score_min": finite_float(score_bin.min()),
                "score_max": finite_float(score_bin.max()),
                "score_mean": finite_float(score_bin.mean()),
                "error_mean_px": finite_float(error_bin.mean()),
                "error_median_px": finite_float(error_bin.median()),
                "error_p90_px": finite_float(torch.quantile(error_bin, 0.90)),
                "coverage_95": finite_float((mahal_bin <= threshold95).float().mean()),
            }
        )
    return rows


def summarize_prediction(
    cache: Mapping[str, torch.Tensor], variance: torch.Tensor
) -> Dict[str, object]:
    errors = cache["errors_px"].float()
    score = torch.sqrt(variance.float().clamp_min(1e-8).prod(dim=-1))
    std_y = variance[:, 0].sqrt()
    std_x = variance[:, 1].sqrt()
    sample_ids = cache["sample_id"].long()
    unique_videos = int(torch.unique(sample_ids).numel())
    return {
        "rows": int(errors.shape[0]),
        "videos": unique_videos,
        "per_video_inference_supported": unique_videos >= 2,
        "uncertainty_score_quantiles": tensor_quantiles(score),
        "std_y_quantiles": tensor_quantiles(std_y),
        "std_x_quantiles": tensor_quantiles(std_x),
        "point_error_quantiles": tensor_quantiles(torch.linalg.vector_norm(errors, dim=-1)),
        "metrics": calibration_report(errors, variance),
        "risk_deciles": risk_decile_table(errors, variance),
    }


def load_first_learned_run(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid experiment result: {path}")
    learned_runs = payload.get("learned_runs")
    if not isinstance(learned_runs, Sequence) or not learned_runs:
        raise ValueError(f"No learned run in experiment result: {path}")
    return learned_runs[0]


def write_decile_csv(
    path: Path,
    report: Mapping[str, object],
) -> None:
    fieldnames = [
        "run",
        "split",
        "stage",
        "bin",
        "count",
        "score_min",
        "score_max",
        "score_mean",
        "error_mean_px",
        "error_median_px",
        "error_p90_px",
        "coverage_95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_label, run_payload in report["runs"].items():
            for split_name, split_payload in run_payload["splits"].items():
                for stage in ("raw", "calibrated"):
                    for row in split_payload[stage]["risk_deciles"]:
                        writer.writerow(
                            {
                                "run": run_label,
                                "split": split_name,
                                "stage": stage,
                                **row,
                            }
                        )


def markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# BeliefCal prediction diagnostics",
        "",
        "This report is descriptive only. It does not fit parameters or use test statistics for model selection.",
        "",
    ]
    for run_label, run_payload in report["runs"].items():
        lines.extend(
            [
                f"## {run_label}",
                "",
                f"- Seed: `{run_payload['seed']}`",
                f"- Feature profile: `{run_payload['feature_profile']}`",
                f"- Calibration scale: `{run_payload['calibration_scale']:.6g}`",
                "",
            ]
        )
        for split_name, split_payload in run_payload["splits"].items():
            raw = split_payload["raw"]["metrics"]
            calibrated = split_payload["calibrated"]["metrics"]
            videos = split_payload["raw"]["videos"]
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"Rows: `{split_payload['raw']['rows']}`; unique videos: `{videos}`.",
                    "",
                    "| Stage | NLL | AURC | Coverage 95 | Mean 95 ellipse area |",
                    "|---|---:|---:|---:|---:|",
                    (
                        f"| Raw | {raw['nll']:.6g} | {raw['aurc']:.6g} | "
                        f"{raw['coverage_95']:.6g} | {raw['mean_95_ellipse_area']:.6g} |"
                    ),
                    (
                        f"| Calibrated | {calibrated['nll']:.6g} | "
                        f"{calibrated['aurc']:.6g} | {calibrated['coverage_95']:.6g} | "
                        f"{calibrated['mean_95_ellipse_area']:.6g} |"
                    ),
                    "",
                ]
            )
            if videos < 2:
                lines.extend(
                    [
                        "> Per-video uncertainty or confidence intervals are not estimable from this split because it contains fewer than two unique videos.",
                        "",
                    ]
                )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Equal-count risk bins describe association between predicted uncertainty and error; they do not establish causality.",
            "- A shared positive variance scale usually preserves ranking, except where clipping creates or removes ties.",
            "- Cross-domain calibration/test results do not provide distribution-free coverage guarantees.",
            "- The one-video test pilot is not paper-level evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_label_path)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    calibration_cache, calibration_manifest = load_beliefcal_cache(args.calibration_cache)
    test_cache, test_manifest = load_beliefcal_cache(args.test_cache)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "format_version": 1,
        "scope": "descriptive_diagnostic_only",
        "caches": {
            "calibration": {
                "path": str(Path(args.calibration_cache).resolve()),
                "manifest": calibration_manifest,
            },
            "test": {
                "path": str(Path(args.test_cache).resolve()),
                "manifest": test_manifest,
            },
        },
        "runs": {},
    }

    for label, result_path in args.run:
        learned_run = load_first_learned_run(result_path)
        state = learned_run["state"]
        calibration_state = learned_run["calibration_state"]
        split_report: Dict[str, object] = {}
        for split_name, cache in (
            ("calibration", calibration_cache),
            ("test", test_cache),
        ):
            raw_variance = predict_learned_variance(cache, state, device="cpu")
            calibrated_variance = apply_scalar_variance_calibration(
                raw_variance, calibration_state
            )
            split_report[split_name] = {
                "raw": summarize_prediction(cache, raw_variance),
                "calibrated": summarize_prediction(cache, calibrated_variance),
            }
        report["runs"][label] = {
            "result_path": str(result_path),
            "seed": int(learned_run["seed"]),
            "feature_profile": learned_run.get("feature_profile", "full"),
            "calibration_scale": float(calibration_state["scale"]),
            "splits": split_report,
        }

    json_path = output_dir / "prediction_diagnostics.json"
    csv_path = output_dir / "risk_deciles.csv"
    markdown_path = output_dir / "prediction_diagnostics.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_decile_csv(csv_path, report)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
