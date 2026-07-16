#!/usr/bin/env python3
"""Paired video bootstrap and failure audit for Route-D TAP evaluations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def paired_bootstrap(values: np.ndarray, seed: int, resamples: int):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "videos": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "positive_videos": int((values > 1.0e-12).sum()),
        "negative_videos": int((values < -1.0e-12).sum()),
        "tie_videos": int((np.abs(values) <= 1.0e-12).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "bootstrap_resamples": int(resamples),
    }


def safe_correlation(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_correlation(x: np.ndarray, y: np.ndarray):
    x_order = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    y_order = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    return safe_correlation(x_order, y_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    rows = payload["per_sample"]
    comparisons = {
        "open_vs_baseline": ("routeD_open", "baseline"),
        "closed_vs_baseline": ("routeD_closed", "baseline"),
        "closed_vs_open": ("routeD_closed", "routeD_open"),
    }
    metrics = ("AJ", "OA", "delta_avg")
    paired = {}
    per_video = []
    for row in rows:
        record = {
            "sample": int(row["sample"]),
            "global_selection_rate": float(row["global_selection_rate"]),
            "open_global_selection_rate": float(row["open_global_selection_rate"]),
            "trajectory_changed_rate": float(row["trajectory_changed_rate"]),
            "mean_trajectory_diff_px": float(row["mean_trajectory_diff_px"]),
            "mean_closed_vs_open_diff_px": float(row["mean_closed_vs_open_diff_px"]),
        }
        for comparison, (left, right) in comparisons.items():
            for metric in metrics:
                record[f"{comparison}_{metric}"] = float(
                    row[left][metric] - row[right][metric]
                )
        per_video.append(record)

    for comparison, (left, right) in comparisons.items():
        paired[comparison] = {}
        for metric_index, metric in enumerate(metrics):
            values = np.array(
                [float(row[left][metric] - row[right][metric]) for row in rows],
                dtype=np.float64,
            )
            paired[comparison][metric] = paired_bootstrap(
                values,
                args.seed + 100 * list(comparisons).index(comparison) + metric_index,
                args.resamples,
            )

    closed_aj = np.array(
        [row["closed_vs_baseline_AJ"] for row in per_video], dtype=np.float64
    )
    closed_delta = np.array(
        [row["closed_vs_baseline_delta_avg"] for row in per_video], dtype=np.float64
    )
    selection = np.array(
        [row["global_selection_rate"] for row in per_video], dtype=np.float64
    )
    trajectory = np.array(
        [row["mean_trajectory_diff_px"] for row in per_video], dtype=np.float64
    )
    correlation = {
        "selection_rate_vs_closed_AJ_gain_pearson": safe_correlation(
            selection, closed_aj
        ),
        "selection_rate_vs_closed_AJ_gain_rank": rank_correlation(
            selection, closed_aj
        ),
        "trajectory_diff_vs_closed_AJ_gain_pearson": safe_correlation(
            trajectory, closed_aj
        ),
        "trajectory_diff_vs_closed_AJ_gain_rank": rank_correlation(
            trajectory, closed_aj
        ),
        "selection_rate_vs_closed_delta_gain_pearson": safe_correlation(
            selection, closed_delta
        ),
        "trajectory_diff_vs_closed_delta_gain_pearson": safe_correlation(
            trajectory, closed_delta
        ),
    }

    ordered = sorted(per_video, key=lambda row: row["closed_vs_baseline_AJ"])
    failure_audit = {
        "worst_closed_AJ_videos": ordered[:8],
        "best_closed_AJ_videos": list(reversed(ordered[-8:])),
        "negative_closed_AJ_video_ids": [
            row["sample"]
            for row in per_video
            if row["closed_vs_baseline_AJ"] < -1.0e-12
        ],
        "negative_closed_delta_video_ids": [
            row["sample"]
            for row in per_video
            if row["closed_vs_baseline_delta_avg"] < -1.0e-12
        ],
    }

    reference = None
    if args.reference:
        reference_payload = json.loads(Path(args.reference).read_text())
        reference = {
            "path": str(args.reference),
            "aggregate": reference_payload.get("aggregate"),
            "delta_routeD_vs_baseline": reference_payload.get(
                "delta_routeD_vs_baseline"
            ),
        }

    result = {
        "evidence_tier": payload.get("evidence_tier"),
        "paper_claim_eligible": payload.get("paper_claim_eligible", False),
        "source": str(args.input),
        "protocol": "paired video bootstrap over the 30 fixed DAVIS videos",
        "paired_bootstrap": paired,
        "correlations": correlation,
        "failure_audit": failure_audit,
        "per_video": per_video,
        "reference": reference,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
