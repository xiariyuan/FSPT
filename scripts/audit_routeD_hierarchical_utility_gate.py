#!/usr/bin/env python3
"""Nested audit of point-only and hierarchical Route-D utility controllers.

For every outer fold, the point utility regressor and frame recovery prior are
trained on fit videos, early-stopped on model-validation videos, calibrated on
posthoc videos, and policy-selected on policy-calibration videos. Outer-test
videos are evaluated once. DAVIS has already been used for development, so all
results remain development diagnostics rather than untouched benchmark claims.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_hierarchical_prior import (
    FramePriorConfig,
    aggregate_frame_mean_target,
    build_causal_frame_prior_features,
    frame_prior_metrics,
    infer_frame_prior,
    train_frame_prior,
)
from projects.mmp_tracker.mmp_tracker.routeD_utility_regression import (
    UtilityRegressionConfig,
    build_utility_targets,
    conformal_lower_bound,
    conformal_upper_residual_quantile,
    infer_utility_regressor,
    regression_metrics,
    train_utility_regressor,
)
from scripts.audit_routeD_nested_posthoc_calibration import (
    bootstrap_video_means,
    make_inner_split,
    make_outer_folds,
    mask_for_ids,
    per_video_rows,
    subset_examples,
)
from scripts.audit_routeD_nested_profile_risk_gate import (
    clopper_pearson_upper,
    harmful_counts,
)
from scripts.train_routeD_domain_action_calibrator import (
    build_action_examples,
    evaluate_selection,
    infer_threshold_logits,
    load_cache,
    load_frozen_scorer,
    policy_prediction_from_profile,
)


def frame_mask_for_ids(frame_sample_id: torch.Tensor, ids: Sequence[int]):
    return mask_for_ids(frame_sample_id, ids)


def row_frame_prediction(
    frame_prediction: torch.Tensor,
    row_to_frame: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    return frame_prediction[row_to_frame[row_mask]]


def build_joint_scores(
    point_prediction: torch.Tensor,
    frame_prediction_rows: torch.Tensor,
    *,
    point_utility_q: float,
    point_p1_q: float,
    frame_q: float,
    frame_weight: float,
):
    point_utility_lcb = conformal_lower_bound(
        point_prediction[:, 0], point_utility_q
    )
    point_p1_lcb = conformal_lower_bound(point_prediction[:, 1], point_p1_q)
    frame_lcb = conformal_lower_bound(frame_prediction_rows, frame_q)
    joint_score = point_utility_lcb + float(frame_weight) * frame_lcb
    return {
        "joint_score": joint_score,
        "point_utility_lcb": point_utility_lcb,
        "point_p1_lcb": point_p1_lcb,
        "frame_lcb": frame_lcb,
    }


def predict_policy(
    examples: Mapping[str, torch.Tensor],
    scores: Mapping[str, torch.Tensor],
    policy: Mapping[str, float | bool],
):
    if not bool(policy.get("enabled", False)):
        return torch.zeros_like(examples["best_global_index"])
    selected = scores["joint_score"] >= float(policy["joint_threshold"])
    selected &= scores["point_p1_lcb"] >= float(policy["min_point_p1_lcb"])
    selected &= scores["frame_lcb"] >= float(policy["min_frame_lcb"])
    selected &= examples["predicted_p1_margin"] >= float(
        policy["min_profile_p1_margin"]
    )
    selected &= examples["predicted_coarse_gain"] >= float(
        policy["min_profile_coarse_gain"]
    )
    return torch.where(
        selected,
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )


def video_risk_summary(
    examples: Mapping[str, torch.Tensor],
    prediction: torch.Tensor,
    sample_ids: torch.Tensor,
    thresholds_px: Sequence[float],
):
    rows = per_video_rows(examples, prediction, sample_ids, thresholds_px)
    utility = torch.tensor(
        [row["mean_gain_over_local_threshold_utility"] for row in rows],
        dtype=torch.float64,
    )
    delta1 = torch.tensor(
        [row["gain_delta_1"] for row in rows], dtype=torch.float64
    )
    pixel = torch.tensor(
        [row["mean_gain_over_local_px"] for row in rows], dtype=torch.float64
    )
    return {
        "videos": len(rows),
        "mean_utility_gain": float(utility.mean()),
        "mean_delta1_gain": float(delta1.mean()),
        "mean_pixel_gain": float(pixel.mean()),
        "negative_utility_videos": int((utility < -1.0e-12).sum()),
        "negative_delta1_videos": int((delta1 < -1.0e-12).sum()),
        "positive_utility_videos": int((utility > 1.0e-12).sum()),
        "positive_delta1_videos": int((delta1 > 1.0e-12).sum()),
    }


def threshold_for_target_coverage(
    score: torch.Tensor,
    eligible: torch.Tensor,
    target_coverage: float,
):
    total_rows = int(score.numel())
    desired = max(1, int(math.ceil(float(target_coverage) * total_rows)))
    eligible_score = score[eligible]
    if int(eligible_score.numel()) < desired:
        return None
    sorted_score = torch.sort(eligible_score, descending=True).values
    return float(sorted_score[desired - 1])


def select_controller_policy(
    *,
    variant: str,
    examples: Mapping[str, torch.Tensor],
    point_prediction: torch.Tensor,
    frame_prediction_rows: torch.Tensor,
    sample_ids: torch.Tensor,
    thresholds_px: Sequence[float],
    residual_quantiles: Mapping[str, float],
    args,
):
    if variant not in {"point_only", "hierarchical"}:
        raise ValueError(f"Unknown controller variant: {variant}")
    frame_weights = [0.0] if variant == "point_only" else args.frame_weights
    frame_minimums = [-1.0e9] if variant == "point_only" else args.frame_lcb_minimums
    rows = []
    for frame_weight in frame_weights:
        scores = build_joint_scores(
            point_prediction,
            frame_prediction_rows,
            point_utility_q=residual_quantiles["point_utility"],
            point_p1_q=residual_quantiles["point_p1"],
            frame_q=residual_quantiles["frame"],
            frame_weight=frame_weight,
        )
        for min_frame_lcb in frame_minimums:
            for min_point_p1_lcb in args.point_p1_lcb_minimums:
                for min_profile_p1 in args.profile_p1_minimums:
                    for min_profile_coarse in args.profile_coarse_minimums:
                        eligible = (
                            (scores["point_p1_lcb"] >= min_point_p1_lcb)
                            & (scores["frame_lcb"] >= min_frame_lcb)
                            & (
                                examples["predicted_p1_margin"]
                                >= min_profile_p1
                            )
                            & (
                                examples["predicted_coarse_gain"]
                                >= min_profile_coarse
                            )
                        )
                        for target_coverage in args.target_coverages:
                            joint_threshold = threshold_for_target_coverage(
                                scores["joint_score"],
                                eligible,
                                target_coverage,
                            )
                            if joint_threshold is None:
                                continue
                            policy = {
                                "enabled": True,
                                "variant": variant,
                                "frame_weight": float(frame_weight),
                                "joint_threshold": joint_threshold,
                                "min_frame_lcb": float(min_frame_lcb),
                                "min_point_p1_lcb": float(min_point_p1_lcb),
                                "min_profile_p1_margin": float(min_profile_p1),
                                "min_profile_coarse_gain": float(
                                    min_profile_coarse
                                ),
                                "target_coverage": float(target_coverage),
                            }
                            prediction = predict_policy(examples, scores, policy)
                            metrics = evaluate_selection(
                                examples,
                                prediction,
                                thresholds_px,
                                f"{variant}_policy_calibration",
                            )
                            coverage = float(metrics["global_selection_rate"])
                            if not (
                                float(args.min_actual_coverage)
                                <= coverage
                                <= float(args.max_actual_coverage)
                            ):
                                continue
                            harmful, selected = harmful_counts(examples, prediction)
                            harmful_upper = clopper_pearson_upper(
                                harmful, selected, args.risk_alpha
                            )
                            metrics.update(policy)
                            metrics.update(
                                {
                                    "selected_count": selected,
                                    "harmful_count": harmful,
                                    "harmful_rate_upper_bound": harmful_upper,
                                }
                            )
                            row_feasible = (
                                metrics[
                                    "mean_gain_over_local_threshold_utility"
                                ]
                                > 0.0
                                and metrics["mean_gain_over_local_px"] >= 0.0
                                and metrics["gain_delta_1"]
                                >= -float(args.delta1_tolerance)
                                and harmful_upper
                                <= float(args.max_harmful_rate)
                            )
                            if row_feasible:
                                video_summary = video_risk_summary(
                                    examples,
                                    prediction,
                                    sample_ids,
                                    thresholds_px,
                                )
                                max_negative = int(
                                    math.floor(
                                        float(args.max_negative_video_fraction)
                                        * int(video_summary["videos"])
                                    )
                                )
                                video_feasible = (
                                    video_summary["mean_utility_gain"] > 0.0
                                    and video_summary["mean_delta1_gain"]
                                    >= -float(args.delta1_tolerance)
                                    and video_summary[
                                        "negative_utility_videos"
                                    ]
                                    <= max_negative
                                    and video_summary["negative_delta1_videos"]
                                    <= max_negative
                                )
                            else:
                                video_summary = None
                                video_feasible = False
                            metrics["video_summary"] = video_summary
                            metrics["risk_feasible"] = bool(
                                row_feasible and video_feasible
                            )
                            metrics["objective"] = (
                                metrics[
                                    "mean_gain_over_local_threshold_utility"
                                ]
                                - float(args.utility_loss_penalty)
                                * metrics[
                                    "mean_harmful_selection_utility_loss"
                                ]
                                + float(args.delta4_reward)
                                * metrics["gain_delta_4"]
                                + float(args.coverage_reward) * coverage
                            )
                            rows.append(metrics)
    feasible = [row for row in rows if row["risk_feasible"]]
    if not feasible:
        return {
            "enabled": False,
            "variant": variant,
            "frame_weight": 0.0,
            "joint_threshold": 1.0e9,
            "min_frame_lcb": -1.0e9,
            "min_point_p1_lcb": -1.0e9,
            "min_profile_p1_margin": -1.0e9,
            "min_profile_coarse_gain": -1.0e9,
            "target_coverage": 0.0,
        }, None, rows, True
    best = max(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mean_gain_over_local_threshold_utility"],
            row["gain_delta_4"],
            row["mean_gain_over_local_px"],
            row["global_selection_rate"],
            -row["harmful_rate_upper_bound"],
        ),
    )
    policy_keys = [
        "enabled",
        "variant",
        "frame_weight",
        "joint_threshold",
        "min_frame_lcb",
        "min_point_p1_lcb",
        "min_profile_p1_margin",
        "min_profile_coarse_gain",
        "target_coverage",
    ]
    return {key: best[key] for key in policy_keys}, best, rows, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--point-conformal-alpha", type=float, default=0.2)
    parser.add_argument("--p1-conformal-alpha", type=float, default=0.2)
    parser.add_argument("--frame-conformal-alpha", type=float, default=0.2)
    parser.add_argument("--risk-alpha", type=float, default=0.05)
    parser.add_argument("--max-harmful-rate", type=float, default=0.25)
    parser.add_argument("--min-actual-coverage", type=float, default=0.02)
    parser.add_argument("--max-actual-coverage", type=float, default=0.15)
    parser.add_argument("--delta1-tolerance", type=float, default=0.0)
    parser.add_argument("--max-negative-video-fraction", type=float, default=0.5)
    parser.add_argument("--utility-loss-penalty", type=float, default=2.0)
    parser.add_argument("--delta4-reward", type=float, default=0.25)
    parser.add_argument("--coverage-reward", type=float, default=0.01)
    parser.add_argument(
        "--frame-weights",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--frame-lcb-minimums",
        type=float,
        nargs="+",
        default=[-0.05, -0.02, 0.0, 0.02],
    )
    parser.add_argument(
        "--point-p1-lcb-minimums",
        type=float,
        nargs="+",
        default=[-0.1, -0.05, 0.0],
    )
    parser.add_argument(
        "--profile-p1-minimums",
        type=float,
        nargs="+",
        default=[-0.05, 0.0],
    )
    parser.add_argument(
        "--profile-coarse-minimums",
        type=float,
        nargs="+",
        default=[0.0, 0.01],
    )
    parser.add_argument(
        "--target-coverages",
        type=float,
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scorer_bundle = torch.load(
        args.scorer_bundle, map_location="cpu", weights_only=False
    )
    cache, cache_metadata = load_cache(args.cache)
    scorer, thresholds_px = load_frozen_scorer(scorer_bundle, device)
    scorer_logits = infer_threshold_logits(
        scorer, cache, scorer_bundle, device, args.batch_size
    )
    examples = build_action_examples(cache, scorer_logits, thresholds_px)
    utility_target, p1_target = build_utility_targets(examples)
    sample_id = cache["sample_id"].long()
    unique_ids = sorted(set(int(value) for value in sample_id.tolist()))

    frame_data = build_causal_frame_prior_features(
        sample_id=sample_id,
        frame_id=cache["frame_id"].long(),
        predicted_p1_margin=examples["predicted_p1_margin"],
        predicted_coarse_gain=examples["predicted_coarse_gain"],
        predicted_total_gain=examples["predicted_total_gain"],
        global_distance_to_local=examples["best_global_distance_to_local"],
        candidate_entropy=cache["features"][:, :, 1].float().mean(dim=1),
        previous_confidence=cache["features"][:, :, 5].float().mean(dim=1),
    )
    frame_target = aggregate_frame_mean_target(
        utility_target,
        frame_data["row_to_frame"],
        int(frame_data["features"].shape[0]),
    )
    outer_folds = make_outer_folds(unique_ids, args.folds, args.seed)
    variants = ["point_only", "hierarchical"]
    oof_prediction = {
        variant: torch.zeros_like(examples["best_global_index"])
        for variant in variants
    }
    oof_point_prediction = torch.full(
        (int(sample_id.numel()), 2), float("nan"), dtype=torch.float32
    )
    oof_frame_prediction = torch.full(
        (int(frame_data["features"].shape[0]),),
        float("nan"),
        dtype=torch.float32,
    )
    assigned_rows = torch.zeros_like(sample_id, dtype=torch.bool)
    assigned_frames = torch.zeros_like(
        frame_data["sample_id"], dtype=torch.bool
    )
    fold_reports = []

    for fold_index, test_ids in enumerate(outer_folds):
        outer_train_ids = [
            value for value in unique_ids if value not in set(test_ids)
        ]
        inner = make_inner_split(
            outer_train_ids, args.seed + 1000 + fold_index
        )
        row_masks = {
            name: mask_for_ids(sample_id, ids)
            for name, ids in inner.items()
        }
        row_test_mask = mask_for_ids(sample_id, test_ids)
        frame_masks = {
            name: frame_mask_for_ids(frame_data["sample_id"], ids)
            for name, ids in inner.items()
        }
        frame_test_mask = frame_mask_for_ids(
            frame_data["sample_id"], test_ids
        )
        if (assigned_rows & row_test_mask).any():
            raise RuntimeError("Outer row test folds overlap")
        if (assigned_frames & frame_test_mask).any():
            raise RuntimeError("Outer frame test folds overlap")
        assigned_rows |= row_test_mask
        assigned_frames |= frame_test_mask

        fit_examples = subset_examples(examples, row_masks["fit"])
        validation_examples = subset_examples(
            examples, row_masks["model_validation"]
        )
        posthoc_examples = subset_examples(
            examples, row_masks["posthoc_fit"]
        )
        policy_examples = subset_examples(
            examples, row_masks["policy_calibration"]
        )
        test_examples = subset_examples(examples, row_test_mask)

        utility_config = UtilityRegressionConfig(seed=args.seed + fold_index)
        utility_result = train_utility_regressor(
            fit_examples["features"],
            utility_target[row_masks["fit"]],
            p1_target[row_masks["fit"]],
            validation_examples["features"],
            utility_target[row_masks["model_validation"]],
            p1_target[row_masks["model_validation"]],
            config=utility_config,
            device=device,
        )
        point_predictions = {}
        for name, row_mask in {
            "posthoc_fit": row_masks["posthoc_fit"],
            "policy_calibration": row_masks["policy_calibration"],
            "test": row_test_mask,
        }.items():
            point_predictions[name] = infer_utility_regressor(
                utility_result["model"],
                examples["features"][row_mask],
                utility_result["feature_mean"],
                utility_result["feature_std"],
                device=device,
                batch_size=args.batch_size,
            )

        frame_config = FramePriorConfig(seed=args.seed + 100 + fold_index)
        frame_result = train_frame_prior(
            frame_data["features"][frame_masks["fit"]],
            frame_target[frame_masks["fit"]],
            frame_data["features"][frame_masks["model_validation"]],
            frame_target[frame_masks["model_validation"]],
            config=frame_config,
            device=device,
        )
        frame_predictions = {}
        for name, frame_mask in {
            "posthoc_fit": frame_masks["posthoc_fit"],
            "policy_calibration": frame_masks["policy_calibration"],
            "test": frame_test_mask,
        }.items():
            frame_predictions[name] = infer_frame_prior(
                frame_result["model"],
                frame_data["features"][frame_mask],
                frame_result["feature_mean"],
                frame_result["feature_std"],
                device=device,
            )

        point_utility_q = conformal_upper_residual_quantile(
            point_predictions["posthoc_fit"][:, 0],
            utility_target[row_masks["posthoc_fit"]],
            alpha=args.point_conformal_alpha,
        )
        point_p1_q = conformal_upper_residual_quantile(
            point_predictions["posthoc_fit"][:, 1],
            p1_target[row_masks["posthoc_fit"]],
            alpha=args.p1_conformal_alpha,
        )
        frame_q = conformal_upper_residual_quantile(
            frame_predictions["posthoc_fit"],
            frame_target[frame_masks["posthoc_fit"]],
            alpha=args.frame_conformal_alpha,
        )
        residual_quantiles = {
            "point_utility": point_utility_q,
            "point_p1": point_p1_q,
            "frame": frame_q,
        }

        frame_prediction_full = torch.full(
            (int(frame_data["features"].shape[0]),),
            float("nan"),
            dtype=torch.float32,
        )
        frame_prediction_full[frame_masks["policy_calibration"]] = (
            frame_predictions["policy_calibration"]
        )
        frame_prediction_full[frame_test_mask] = frame_predictions["test"]
        policy_frame_rows = row_frame_prediction(
            frame_prediction_full,
            frame_data["row_to_frame"],
            row_masks["policy_calibration"],
        )
        test_frame_rows = row_frame_prediction(
            frame_prediction_full,
            frame_data["row_to_frame"],
            row_test_mask,
        )
        if not torch.isfinite(policy_frame_rows).all():
            raise RuntimeError("Missing policy frame predictions")
        if not torch.isfinite(test_frame_rows).all():
            raise RuntimeError("Missing test frame predictions")

        variant_reports = {}
        for variant in variants:
            policy, selected_policy, search_rows, fallback = (
                select_controller_policy(
                    variant=variant,
                    examples=policy_examples,
                    point_prediction=point_predictions["policy_calibration"],
                    frame_prediction_rows=policy_frame_rows,
                    sample_ids=sample_id[row_masks["policy_calibration"]],
                    thresholds_px=thresholds_px,
                    residual_quantiles=residual_quantiles,
                    args=args,
                )
            )
            test_scores = build_joint_scores(
                point_predictions["test"],
                test_frame_rows,
                point_utility_q=point_utility_q,
                point_p1_q=point_p1_q,
                frame_q=frame_q,
                frame_weight=float(policy["frame_weight"]),
            )
            test_prediction = predict_policy(
                test_examples, test_scores, policy
            )
            oof_prediction[variant][row_test_mask] = test_prediction
            variant_reports[variant] = {
                "fallback_to_local": fallback,
                "frozen_policy": policy,
                "selected_policy_calibration": selected_policy,
                "candidate_policy_count": len(search_rows),
                "feasible_policy_count": int(
                    sum(bool(row["risk_feasible"]) for row in search_rows)
                ),
                "test_metrics": evaluate_selection(
                    test_examples,
                    test_prediction,
                    thresholds_px,
                    variant,
                ),
                "test_video_summary": video_risk_summary(
                    test_examples,
                    test_prediction,
                    sample_id[row_test_mask],
                    thresholds_px,
                ),
            }

        oof_point_prediction[row_test_mask] = point_predictions["test"]
        oof_frame_prediction[frame_test_mask] = frame_predictions["test"]
        fold_reports.append(
            {
                "fold": fold_index,
                "outer_test_sample_ids": test_ids,
                "inner_sample_partitions": inner,
                "partition_rows": {
                    name: int(mask.sum()) for name, mask in row_masks.items()
                }
                | {"test": int(row_test_mask.sum())},
                "partition_frames": {
                    name: int(mask.sum()) for name, mask in frame_masks.items()
                }
                | {"test": int(frame_test_mask.sum())},
                "point_regressor_best_epoch": utility_result["best_epoch"],
                "frame_prior_best_epoch": frame_result["best_epoch"],
                "residual_quantiles": residual_quantiles,
                "point_test_regression": regression_metrics(
                    point_predictions["test"],
                    utility_target[row_test_mask],
                    p1_target[row_test_mask],
                ),
                "frame_test_regression": frame_prior_metrics(
                    frame_predictions["test"],
                    frame_target[frame_test_mask],
                ),
                "variants": variant_reports,
            }
        )

    if not assigned_rows.all() or not assigned_frames.all():
        raise RuntimeError("Incomplete OOF fold assignment")
    if not torch.isfinite(oof_point_prediction).all():
        raise RuntimeError("Incomplete OOF point prediction")
    if not torch.isfinite(oof_frame_prediction).all():
        raise RuntimeError("Incomplete OOF frame prediction")

    aggregate = {}
    paired_stats = {}
    per_video = {}
    statistic_metrics = [
        "mean_gain_over_local_px",
        "mean_gain_over_local_threshold_utility",
        "gain_delta_1",
        "gain_delta_2",
        "gain_delta_4",
        "gain_delta_8",
        "gain_delta_16",
        "global_selection_rate",
    ]
    for variant_index, variant in enumerate(variants):
        aggregate[variant] = evaluate_selection(
            examples,
            oof_prediction[variant],
            thresholds_px,
            f"{variant}_oof",
        )
        video_rows = per_video_rows(
            examples,
            oof_prediction[variant],
            sample_id,
            thresholds_px,
        )
        per_video[variant] = video_rows
        paired_stats[variant] = {
            metric: bootstrap_video_means(
                video_rows,
                metric,
                args.seed + 100 * variant_index + metric_index,
            )
            for metric_index, metric in enumerate(statistic_metrics)
        }

    old_policy = {
        "p1_tolerance": 0.0,
        "min_coarse_gain": 0.05,
        "min_total_gain": -0.05,
    }
    old_prediction = policy_prediction_from_profile(examples, old_policy)
    oracle_prediction = torch.where(
        examples["true_gain"] > 0,
        examples["best_global_index"],
        torch.zeros_like(examples["best_global_index"]),
    )
    references = {
        "frozen_kubric_profile_gate": evaluate_selection(
            examples,
            old_prediction,
            thresholds_px,
            "frozen_kubric_profile_gate",
        ),
        "oracle_gate_on_frozen_ranking": evaluate_selection(
            examples,
            oracle_prediction,
            thresholds_px,
            "oracle_gate_on_frozen_ranking",
        ),
    }

    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "protocol": (
            "5-fold nested video-level CV with disjoint fit, model-validation, "
            "posthoc residual calibration, policy calibration, and outer test"
        ),
        "conformal_caveat": (
            "Residual bounds are row-level split-conformal diagnostics. Rows "
            "within a video are correlated, so no formal clustered-coverage "
            "guarantee is claimed."
        ),
        "important_caveat": (
            "DAVIS has already been used for development. OOF results are not "
            "untouched final-test evidence."
        ),
        "config": vars(args),
        "utility_regression_config": asdict(
            UtilityRegressionConfig(seed=args.seed)
        ),
        "frame_prior_config": asdict(FramePriorConfig(seed=args.seed)),
        "cache_metadata": cache_metadata,
        "outer_folds": outer_folds,
        "fold_reports": fold_reports,
        "aggregate_oof": aggregate,
        "paired_video_bootstrap": paired_stats,
        "per_video_oof": per_video,
        "point_oof_regression": regression_metrics(
            oof_point_prediction, utility_target, p1_target
        ),
        "frame_oof_regression": frame_prior_metrics(
            oof_frame_prediction, frame_target
        ),
        "references_full_cache": references,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    artifact = (
        Path(args.artifact_output)
        if args.artifact_output
        else output.with_suffix(".pt")
    )
    torch.save(
        {
            "kind": "routeD_hierarchical_utility_oof",
            "evidence_tier": "development_diagnostic_only",
            "paper_claim_eligible": False,
            "sample_id": sample_id,
            "frame_id": cache["frame_id"].long(),
            "point_id": cache["point_id"].long(),
            "frame_sample_id": frame_data["sample_id"],
            "frame_number": frame_data["frame_id"],
            "row_to_frame": frame_data["row_to_frame"],
            "utility_target": utility_target,
            "p1_target": p1_target,
            "frame_target": frame_target,
            "oof_point_prediction": oof_point_prediction,
            "oof_frame_prediction": oof_frame_prediction,
            "oof_prediction": oof_prediction,
            "outer_folds": outer_folds,
            "fold_policies": {
                variant: [
                    fold["variants"][variant]["frozen_policy"]
                    for fold in fold_reports
                ]
                for variant in variants
            },
        },
        artifact,
    )
    print(
        json.dumps(
            {
                "aggregate_oof": aggregate,
                "paired_video_bootstrap": paired_stats,
                "point_oof_regression": result["point_oof_regression"],
                "frame_oof_regression": result["frame_oof_regression"],
                "fold_summary": [
                    {
                        "fold": fold["fold"],
                        "test_ids": fold["outer_test_sample_ids"],
                        "point_regression": fold[
                            "point_test_regression"
                        ],
                        "frame_regression": fold[
                            "frame_test_regression"
                        ],
                        "variants": {
                            variant: {
                                "fallback": fold["variants"][variant][
                                    "fallback_to_local"
                                ],
                                "feasible_policy_count": fold["variants"][
                                    variant
                                ]["feasible_policy_count"],
                                "policy": fold["variants"][variant][
                                    "frozen_policy"
                                ],
                                "test_metrics": fold["variants"][variant][
                                    "test_metrics"
                                ],
                            }
                            for variant in variants
                        },
                    }
                    for fold in fold_reports
                ],
                "references_full_cache": references,
                "artifact": str(artifact),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
