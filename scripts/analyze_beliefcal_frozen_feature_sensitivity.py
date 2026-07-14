#!/usr/bin/env python3
"""Permutation sensitivity for already-frozen BeliefCal A2 bundles.

For each semantic MMP uncertainty feature, the value/availability pair is
jointly permuted within a split. The frozen uncertainty head and frozen selected
calibrator are then re-applied. This is post-test descriptive sensitivity only:
it does not fit, select, or establish causal feature importance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

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
from projects.mmp_tracker.mmp_tracker.conditional_calibration import (
    apply_conditional_shared_scale_calibration,
)
from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    MMP_UNCERTAINTY_VALUE_NAMES,
)


def parse_label_path(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip() or not path.exists():
        raise argparse.ArgumentTypeError(f"Invalid LABEL=PATH: {spec}")
    return label.strip(), path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> torch.Tensor:
    raw = predict_learned_variance(cache, bundle["learned_state"], device=device)
    selected = bundle["selection"]["selected"]
    if selected == "conditional_affine":
        return apply_conditional_shared_scale_calibration(
            cache, raw, bundle["conditional_state"], device=device
        )
    if selected == "shared_scalar":
        return apply_scalar_variance_calibration(raw, bundle["scalar_state"])
    raise ValueError(f"Unsupported selection: {selected!r}")


def clone_with_joint_permutation(
    cache: Mapping[str, torch.Tensor],
    feature_index: int,
    permutation: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    modified = dict(cache)
    features = cache["features"].clone()
    value_count = len(MMP_UNCERTAINTY_VALUE_NAMES)
    features[:, feature_index] = features[permutation, feature_index]
    features[:, feature_index + value_count] = features[
        permutation, feature_index + value_count
    ]
    modified["features"] = features
    return modified


def metric_delta(
    observed: Mapping[str, float], baseline: Mapping[str, float]
) -> Dict[str, float]:
    return {
        "nll": float(observed["nll"] - baseline["nll"]),
        "aurc": float(observed["aurc"] - baseline["aurc"]),
        "coverage_95": float(observed["coverage_95"] - baseline["coverage_95"]),
        "coverage_gap_95": float(
            observed["coverage_gap_95"] - baseline["coverage_gap_95"]
        ),
        "mean_95_ellipse_area": float(
            observed["mean_95_ellipse_area"] - baseline["mean_95_ellipse_area"]
        ),
        "mean_sharpness": float(
            observed["mean_sharpness"] - baseline["mean_sharpness"]
        ),
    }


def aggregate_repeats(rows: Sequence[Mapping[str, Mapping[str, float]]]) -> Dict[str, object]:
    metrics = rows[0]["delta"].keys()
    result: Dict[str, object] = {"repeats": len(rows), "delta": {}}
    for metric in metrics:
        values = torch.tensor(
            [float(row["delta"][metric]) for row in rows], dtype=torch.float64
        )
        result["delta"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def standardized_shift(
    calibration_cache: Mapping[str, torch.Tensor],
    test_cache: Mapping[str, torch.Tensor],
    index: int,
) -> float | None:
    a = calibration_cache["features"][:, index].float()
    b = test_cache["features"][:, index].float()
    pooled = math.sqrt((float(a.var(unbiased=False)) + float(b.var(unbiased=False))) / 2.0)
    if pooled <= 1e-12:
        return None
    return float((b.mean() - a.mean()) / pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", required=True, type=parse_label_path)
    parser.add_argument("--cache", action="append", required=True, type=parse_label_path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
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
        "scope": "posttest_descriptive_permutation_sensitivity_no_fitting_no_selection",
        "permutation": {
            "unit": "row",
            "joint_columns": "semantic value plus availability flag",
            "repeats": int(args.repeats),
            "seed": int(args.seed),
        },
        "caches": cache_records,
        "bundles": {},
    }

    for bundle_offset, (bundle_label, bundle_path) in enumerate(args.bundle):
        bundle = load_bundle(bundle_path)
        bundle_payload: Dict[str, object] = {
            "bundle_path": str(bundle_path),
            "bundle_sha256": sha256_file(bundle_path),
            "seed": int(bundle["learned_seed"]),
            "selected": bundle["selection"]["selected"],
            "splits": {},
        }
        for split_offset, (split_label, cache) in enumerate(caches.items()):
            baseline_variance = selected_variance(cache, bundle, args.device)
            baseline = evaluate_variance_method(cache, baseline_variance)["overall"]
            feature_rows = []
            for feature_index, feature_name in enumerate(MMP_UNCERTAINTY_VALUE_NAMES):
                repeat_rows = []
                for repeat in range(int(args.repeats)):
                    generator = torch.Generator().manual_seed(
                        int(args.seed)
                        + 1000003 * bundle_offset
                        + 10007 * split_offset
                        + 101 * feature_index
                        + repeat
                    )
                    permutation = torch.randperm(
                        cache["features"].shape[0], generator=generator
                    )
                    modified = clone_with_joint_permutation(
                        cache, feature_index, permutation
                    )
                    variance = selected_variance(modified, bundle, args.device)
                    observed = evaluate_variance_method(modified, variance)["overall"]
                    repeat_rows.append(
                        {
                            "repeat": repeat,
                            "delta": metric_delta(observed, baseline),
                        }
                    )
                row = {
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    **aggregate_repeats(repeat_rows),
                }
                if "calibration" in caches and "test" in caches:
                    row["standardized_mean_shift_test_minus_calibration"] = standardized_shift(
                        caches["calibration"], caches["test"], feature_index
                    )
                feature_rows.append(row)
            feature_rows.sort(
                key=lambda row: abs(row["delta"]["nll"]["mean"]), reverse=True
            )
            bundle_payload["splits"][split_label] = {
                "baseline_metrics": baseline,
                "features_ranked_by_abs_nll_delta": feature_rows,
            }
        report["bundles"][bundle_label] = bundle_payload

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
