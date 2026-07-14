#!/usr/bin/env python3
"""Post-test descriptive transfer diagnostics for frozen BeliefCal A2 bundles.

This utility never fits, selects, or modifies a model. It reconstructs the
already-frozen selected variance for calibration, validation, and test caches,
then reports row-weighted, stratum-level, and video-cluster summaries. Any test
analysis produced by this script is post-test descriptive evidence only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    apply_scalar_variance_calibration,
    evaluate_variance_method,
    load_beliefcal_cache,
    predict_learned_variance,
)
from projects.mmp_tracker.mmp_tracker.calibration_metrics import (
    CHI2_DF2_THRESHOLDS,
    calibration_report,
    mahalanobis_sq,
)
from projects.mmp_tracker.mmp_tracker.conditional_calibration import (
    apply_conditional_shared_scale_calibration,
)

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
VIDEO_METRICS = (
    "nll",
    "mean_point_error_px",
    "mean_sharpness",
    "mean_95_ellipse_area",
    "coverage_95",
    "aurc",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_bundle(path: Path) -> Mapping[str, object]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, Mapping):
        raise ValueError(f"Invalid bundle: {path}")
    if bundle.get("kind") != "beliefcal_mvp1c_frozen_calibration_bundle":
        raise ValueError(f"Unexpected bundle kind: {bundle.get('kind')!r}")
    if bundle.get("test_loaded_during_fit") is not False:
        raise RuntimeError("Bundle does not certify test-free fitting")
    return bundle


def selected_variance(
    cache: Mapping[str, torch.Tensor],
    bundle: Mapping[str, object],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = predict_learned_variance(cache, bundle["learned_state"], device=device)
    selected = bundle["selection"]["selected"]
    if selected == "conditional_affine":
        calibrated = apply_conditional_shared_scale_calibration(
            cache,
            raw,
            bundle["conditional_state"],
            device=device,
        )
    elif selected == "shared_scalar":
        calibrated = apply_scalar_variance_calibration(raw, bundle["scalar_state"])
    else:
        raise ValueError(f"Unsupported frozen selection: {selected!r}")
    return raw, calibrated


def per_video_reports(
    cache: Mapping[str, torch.Tensor], variance: torch.Tensor
) -> list[Dict[str, float | int]]:
    sample_ids = cache["sample_id"].long()
    rows: list[Dict[str, float | int]] = []
    for sample_id in torch.unique(sample_ids, sorted=True).tolist():
        mask = sample_ids == int(sample_id)
        report = calibration_report(cache["errors_px"], variance, mask=mask)
        rows.append({"sample_id": int(sample_id), **report})
    return rows


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> Dict[str, float | int | None]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    count = int(tensor.numel())
    if count == 0:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(tensor.mean())
    if count == 1 or replicates <= 0:
        return {"count": count, "mean": mean, "ci95_low": None, "ci95_high": None}
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randint(0, count, (int(replicates), count), generator=generator)
    means = tensor[index].mean(dim=1)
    interval = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "count": count,
        "mean": mean,
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
    }


def summarize_video_reports(
    rows: Sequence[Mapping[str, float | int]],
    *,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {"videos": len(rows), "metrics": {}}
    for offset, metric in enumerate(VIDEO_METRICS):
        values = [float(row[metric]) for row in rows if metric in row]
        tensor = torch.tensor(values, dtype=torch.float64)
        summary["metrics"][metric] = {
            "bootstrap_macro_mean": bootstrap_mean_ci(
                values,
                replicates=replicates,
                seed=seed + 1009 * offset,
            ),
            "distribution": {
                "min": float(tensor.min()) if tensor.numel() else None,
                "q10": float(torch.quantile(tensor, 0.10)) if tensor.numel() else None,
                "median": float(torch.quantile(tensor, 0.50)) if tensor.numel() else None,
                "q90": float(torch.quantile(tensor, 0.90)) if tensor.numel() else None,
                "max": float(tensor.max()) if tensor.numel() else None,
            },
        }
    return summary


def summarize_split(
    cache: Mapping[str, torch.Tensor],
    variance: torch.Tensor,
    *,
    replicates: int,
    seed: int,
) -> Tuple[Dict[str, object], list[Dict[str, float | int]]]:
    errors = cache["errors_px"].float()
    variance = variance.float().clamp_min(1e-8)
    mahal = mahalanobis_sq(errors, variance)
    point_error = torch.linalg.vector_norm(errors, dim=-1)
    uncertainty = torch.sqrt(variance.prod(dim=-1))
    video_rows = per_video_reports(cache, variance)
    threshold95 = CHI2_DF2_THRESHOLDS[0.95]
    return (
        {
            "rows": int(errors.shape[0]),
            "videos": len(video_rows),
            "row_weighted_metrics": evaluate_variance_method(cache, variance),
            "mahalanobis_sq_quantiles": tensor_quantiles(mahal),
            "point_error_px_quantiles": tensor_quantiles(point_error),
            "uncertainty_score_quantiles": tensor_quantiles(uncertainty),
            "mahalanobis_tail_rate_above_nominal_95": float(
                (mahal > threshold95).float().mean()
            ),
            "video_cluster_summary": summarize_video_reports(
                video_rows,
                replicates=replicates,
                seed=seed,
            ),
        },
        video_rows,
    )


def transfer_summary(split_reports: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    reference = split_reports.get("calibration")
    test = split_reports.get("test")
    if reference is None or test is None:
        return {}
    cal = reference["row_weighted_metrics"]["overall"]
    tst = test["row_weighted_metrics"]["overall"]
    return {
        "test_minus_calibration": {
            "nll": float(tst["nll"] - cal["nll"]),
            "coverage_95": float(tst["coverage_95"] - cal["coverage_95"]),
            "mean_point_error_px": float(
                tst["mean_point_error_px"] - cal["mean_point_error_px"]
            ),
            "mean_95_ellipse_area": float(
                tst["mean_95_ellipse_area"] - cal["mean_95_ellipse_area"]
            ),
        },
        "test_to_calibration_ratio": {
            "nll": float(tst["nll"] / max(cal["nll"], 1e-12)),
            "mean_point_error_px": float(
                tst["mean_point_error_px"] / max(cal["mean_point_error_px"], 1e-12)
            ),
            "mean_95_ellipse_area": float(
                tst["mean_95_ellipse_area"]
                / max(cal["mean_95_ellipse_area"], 1e-12)
            ),
        },
    }


def write_video_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "bundle",
        "seed",
        "split",
        "sample_id",
        "count",
        "nll",
        "mean_point_error_px",
        "median_point_error_px",
        "mean_sharpness",
        "mean_95_ellipse_area",
        "mahalanobis_median",
        "mahalanobis_p95",
        "aurc",
        "coverage_50",
        "coverage_68",
        "coverage_90",
        "coverage_95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# Frozen BeliefCal A2 transfer diagnostics",
        "",
        "> Post-test descriptive analysis only. No parameter was fitted, selected, or modified.",
        "",
    ]
    for label, payload in report["bundles"].items():
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Seed: `{payload['seed']}`",
                f"- Frozen selection: `{payload['selected']}`",
                f"- Bundle SHA256: `{payload['bundle_sha256']}`",
                "",
                "| Split | NLL | AURC | Coverage 95 | Mean error px | Mean 95 ellipse area |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for split_name, split_payload in payload["splits"].items():
            overall = split_payload["row_weighted_metrics"]["overall"]
            lines.append(
                f"| {split_name} | {overall['nll']:.6g} | {overall['aurc']:.6g} | "
                f"{overall['coverage_95']:.6g} | {overall['mean_point_error_px']:.6g} | "
                f"{overall['mean_95_ellipse_area']:.6g} |"
            )
        lines.extend(["", "### Test strata", "", "| Stratum | Count | NLL | Coverage 95 | AURC |", "|---|---:|---:|---:|---:|"])
        test_metrics = payload["splits"].get("test", {}).get("row_weighted_metrics", {})
        for name, metrics in test_metrics.items():
            lines.append(
                f"| {name} | {metrics.get('count', 0):.0f} | {metrics.get('nll', float('nan')):.6g} | "
                f"{metrics.get('coverage_95', float('nan')):.6g} | {metrics.get('aurc', float('nan')):.6g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Test-set diagnostics are descriptive and cannot be used to tune A2.",
            "- Video-cluster bootstrap intervals quantify variation across the observed videos; they are not distribution-free guarantees.",
            "- Shared-family Kubric train/calibration/validation splits and DAVIS test imply a synthetic-to-real transfer setting.",
            "- Any conformal extension requires a separate prospective amendment before execution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", required=True, type=parse_label_path)
    parser.add_argument("--cache", action="append", required=True, type=parse_label_path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    caches: Dict[str, Mapping[str, torch.Tensor]] = {}
    cache_records: Dict[str, object] = {}
    for label, path in args.cache:
        cache, manifest = load_beliefcal_cache(path)
        caches[label] = cache
        cache_records[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "manifest": manifest,
        }

    report: Dict[str, object] = {
        "format_version": 1,
        "scope": "posttest_descriptive_only_no_fitting_no_selection",
        "bootstrap": {
            "unit": "video/sample_id cluster",
            "replicates": int(args.bootstrap_replicates),
            "interval": "percentile 95%",
        },
        "caches": cache_records,
        "bundles": {},
    }
    csv_rows: list[Dict[str, object]] = []

    for label, bundle_path in args.bundle:
        bundle = load_bundle(bundle_path)
        seed = int(bundle["learned_seed"])
        split_reports: Dict[str, object] = {}
        for split_index, (split_name, cache) in enumerate(caches.items()):
            _, variance = selected_variance(cache, bundle, args.device)
            split_report, video_rows = summarize_split(
                cache,
                variance,
                replicates=int(args.bootstrap_replicates),
                seed=seed * 10007 + split_index,
            )
            split_reports[split_name] = split_report
            for row in video_rows:
                csv_rows.append(
                    {
                        "bundle": label,
                        "seed": seed,
                        "split": split_name,
                        **row,
                    }
                )
        report["bundles"][label] = {
            "bundle_path": str(bundle_path),
            "bundle_sha256": sha256_file(bundle_path),
            "seed": seed,
            "selected": bundle["selection"]["selected"],
            "manifest_contract_passed_before_test": bool(
                bundle.get(
                    "manifest_contract_passed_before_test",
                    bundle.get("formal_eligible_before_test", False),
                )
            ),
            "evidence_status": bundle.get(
                "evidence_status", "legacy_status_unspecified"
            ),
            "paper_claim_eligible": bool(bundle.get("paper_claim_eligible", False)),
            "splits": split_reports,
            "transfer_summary": transfer_summary(split_reports),
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_transfer_diagnostics.json"
    csv_path = output_dir / "per_video_metrics.csv"
    markdown_path = output_dir / "frozen_transfer_diagnostics.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_video_csv(csv_path, csv_rows)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
