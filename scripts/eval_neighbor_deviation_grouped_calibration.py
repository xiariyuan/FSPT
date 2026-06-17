#!/usr/bin/env python3
"""
Grouped leave-one-video-out calibration evaluation for trajectory reliability.

This script evaluates whether risk calibration generalizes across videos.
It fits each calibrator on train videos and evaluates on the held-out video.

Usage:
  python scripts/eval_neighbor_deviation_grouped_calibration.py \
    --input-jsonl outputs/trajectory_manifold_verifier_v1/per_sample_predictions.jsonl \
    --output-dir outputs/neighbor_deviation_grouped_calibration_v1 \
    --error-threshold 16.0 \
    --long-occ-threshold 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def compute_ece(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece / max(1, len(labels)))


def brier_score(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def minmax_apply(train_scores, test_scores):
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    s_min = float(train_scores.min())
    s_max = float(train_scores.max())
    if s_max > s_min:
        return np.clip((test_scores - s_min) / (s_max - s_min), 0.0, 1.0)
    return np.ones_like(test_scores) * 0.5


def fit_platt(train_scores, train_labels, test_scores):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(train_scores.reshape(-1, 1), train_labels)
    return lr.predict_proba(test_scores.reshape(-1, 1))[:, 1], lr


def fit_isotonic(train_scores, train_labels, test_scores):
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_scores, train_labels)
    return ir.predict(test_scores), ir


def orient_scores(raw_scores, orientation):
    raw_scores = np.asarray(raw_scores, dtype=float)
    if orientation == "risk":
        return raw_scores
    if orientation == "reliability":
        return -raw_scores
    raise ValueError(f"Unknown orientation: {orientation}")


def summarize_probs(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return {
        "ece": round(compute_ece(probs, labels), 4),
        "brier": round(brier_score(probs, labels), 4),
        "mean_predicted": round(float(probs.mean()), 4),
        "actual_positive": round(float(labels.mean()), 4),
        "roc_auc": round(compute_auc(labels, probs), 4) if compute_auc(labels, probs) is not None else None,
    }


def summarize_subset(rows, subset_mask):
    labels = np.asarray([r["is_high_error"] for r in rows], dtype=float)
    out = {}
    for key in [
        "visibility_raw_risk",
        "visibility_platt_risk",
        "visibility_isotonic_risk",
        "neighbor_deviation_raw_risk",
        "neighbor_deviation_platt_risk",
        "neighbor_deviation_isotonic_risk",
    ]:
        probs = np.asarray([r[key] for r in rows], dtype=float)
        probs = probs[subset_mask]
        sub_labels = labels[subset_mask]
        if len(probs) == 0:
            continue
        out[key] = summarize_probs(probs, sub_labels)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--error-threshold", type=float, default=16.0)
    parser.add_argument("--long-occ-threshold", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(args.input_jsonl) if line.strip()]
    rows = [r for r in rows if r.get("neighbor_deviation") is not None]
    videos = sorted({r["video_name"] for r in rows})

    specs = [
        ("visibility", "visibility_score", "reliability"),
        ("neighbor_deviation", "neighbor_deviation", "risk"),
    ]

    heldout_rows = []
    per_video = []

    for video in videos:
        train_rows = [r for r in rows if r["video_name"] != video]
        test_rows = [r for r in rows if r["video_name"] == video]
        y_train = np.asarray([r["is_high_error"] for r in train_rows], dtype=float)
        y_test = np.asarray([r["is_high_error"] for r in test_rows], dtype=float)

        video_summary = {
            "test_video": video,
            "train_n": len(train_rows),
            "test_n": len(test_rows),
            "test_high_error_n": int(y_test.sum()),
            "scores": {},
        }

        per_score_predictions = {}
        for score_name, field_name, orientation in specs:
            train_raw = np.asarray([r[field_name] for r in train_rows], dtype=float)
            test_raw = np.asarray([r[field_name] for r in test_rows], dtype=float)
            train_fit = orient_scores(train_raw, orientation)
            test_fit = orient_scores(test_raw, orientation)

            raw_risk = minmax_apply(train_fit, test_fit)
            platt_risk, platt_model = fit_platt(train_fit, y_train, test_fit)
            isotonic_risk, _ = fit_isotonic(train_fit, y_train, test_fit)

            per_score_predictions[score_name] = {
                "raw_risk": raw_risk,
                "platt_risk": platt_risk,
                "isotonic_risk": isotonic_risk,
                "platt_a": round(float(platt_model.coef_[0][0]), 4),
                "platt_b": round(float(platt_model.intercept_[0]), 4),
            }

            video_summary["scores"][score_name] = {
                "raw": summarize_probs(raw_risk, y_test),
                "platt": summarize_probs(platt_risk, y_test),
                "isotonic": summarize_probs(isotonic_risk, y_test),
                "input_orientation": orientation,
                "platt_a": round(float(platt_model.coef_[0][0]), 4),
                "platt_b": round(float(platt_model.intercept_[0]), 4),
            }

        for idx, sample in enumerate(test_rows):
            record = dict(sample)
            for score_name in per_score_predictions:
                record[f"{score_name}_raw_risk"] = round(float(per_score_predictions[score_name]["raw_risk"][idx]), 6)
                record[f"{score_name}_platt_risk"] = round(float(per_score_predictions[score_name]["platt_risk"][idx]), 6)
                record[f"{score_name}_isotonic_risk"] = round(
                    float(per_score_predictions[score_name]["isotonic_risk"][idx]), 6
                )
            heldout_rows.append(record)

        per_video.append(video_summary)

    labels = np.asarray([r["is_high_error"] for r in heldout_rows], dtype=float)
    occ = np.asarray([r.get("occ_length", 0.0) for r in heldout_rows], dtype=float)
    overall_mask = np.ones(len(heldout_rows), dtype=bool)
    long_occ_mask = occ > args.long_occ_threshold

    pooled = {
        "all": summarize_subset(heldout_rows, overall_mask),
        "long_occ": summarize_subset(heldout_rows, long_occ_mask),
    }

    summary = {
        "n_samples": len(rows),
        "n_valid": len(heldout_rows),
        "n_sequences": len(videos),
        "error_threshold": args.error_threshold,
        "long_occ_threshold": args.long_occ_threshold,
        "videos_with_high_error": sum(1 for v in per_video if v["test_high_error_n"] > 0),
        "pooled": pooled,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_video_metrics.json", "w") as f:
        json.dump(per_video, f, indent=2)
    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for row in heldout_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Loaded {len(rows)} valid samples across {len(videos)} videos")
    print(f"Videos with high-error positives: {summary['videos_with_high_error']}")
    for subset_name, subset_metrics in pooled.items():
        print(f"\n--- {subset_name} ---")
        for key, metrics in subset_metrics.items():
            print(
                f"{key:32s} "
                f"ECE={metrics['ece']:.4f} "
                f"Brier={metrics['brier']:.4f} "
                f"AUC={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
            )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
