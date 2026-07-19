#!/usr/bin/env python3
"""Formal fit-only training runner for Route-D CSRR Gate 2."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = EXTERNAL_ROOT / "baselines/cotracker"
for path in (COTRACKER_ROOT, EXTERNAL_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    restore_cotracker_online_state,
    snapshot_cotracker_online_state,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    CSRR_SCHEMA_VERSION,
    CounterfactualStructuredReextractionRestorer,
    apply_reextracted_state_action,
    model_xy_to_input_xy,
    reextract_cotracker_memory,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    load_complete_csrr_cache_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    CSRRBatchLossConfig,
    aggregate_point_rows,
    csrr_batch_loss,
    exact_snapshot_from_artifact,
    load_csrr_cache_video,
    point_future_rows,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import paired_video_bootstrap_ci
from scripts.audit_routeD_cotracker3_interface import set_deterministic

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_counterfactual_state_restorer_gate2_v0.yaml"
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs/routeD_counterfactual_state_restorer_cache_20260719"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_counterfactual_state_restorer_training_20260719/seed17"
VARIANTS = ("native", "coordinate_only", "coordinate_probability", "full_state")


def _loss_config(config: Mapping[str, Any]) -> CSRRBatchLossConfig:
    loss = config["training"]["loss"]
    return CSRRBatchLossConfig(**{key: float(value) for key, value in loss.items()})


def _model_batch(
    model: CounterfactualStructuredReextractionRestorer,
    tensors: Mapping[str, Any],
    row_indices: torch.Tensor,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    rows = row_indices.long()
    trajectory = tensors["trajectory_features"][rows].to(device=device, dtype=torch.float32)
    native_support = [
        value[rows].to(device=device, dtype=torch.float32)
        for value in tensors["native_track_support"]
    ]
    frame_feature = [
        value[None].to(device=device, dtype=torch.float32)
        for value in tensors["frame_feature_pyramid"]
    ]
    native_commit_normalized = tensors["native_commit_coordinates_normalized_xy"][rows].to(
        device=device, dtype=torch.float32
    )
    output = model(
        trajectory_features=trajectory,
        native_support_pyramid=native_support,
        frame_feature_pyramid=frame_feature,
        native_commit_coordinates_xy=native_commit_normalized * 255.0,
        input_height=256,
        input_width=256,
    )
    feature_for_sampling = [value[:, None] for value in frame_feature]
    re_feat_native, re_support_native = reextract_cotracker_memory(
        _model_batch.frozen_cotracker_model,
        feature_for_sampling,
        output["predicted_coordinates_xy"][None],
        input_height=256,
        input_width=256,
    )
    # Loss code treats rows as the batch dimension.
    re_feat_batch = [value.permute(2, 1, 0, 3) for value in re_feat_native]
    re_support_batch = [value.permute(2, 1, 0, 3) for value in re_support_native]
    batch = {
        "apply_target": tensors["apply_target"][rows].to(device=device),
        "teacher_commit_coordinates_normalized_xy": tensors[
            "teacher_commit_coordinates_normalized_xy"
        ][rows].to(device=device, dtype=torch.float32),
        "native_commit_coordinates_normalized_xy": native_commit_normalized,
        "native_visibility_probability": tensors["native_visibility_probability"][rows].to(
            device=device, dtype=torch.float32
        ),
        "native_confidence_probability": tensors["native_confidence_probability"][rows].to(
            device=device, dtype=torch.float32
        ),
        "teacher_visibility_probability": tensors["teacher_visibility_probability"][rows].to(
            device=device, dtype=torch.float32
        ),
        "teacher_confidence_probability": tensors["teacher_confidence_probability"][rows].to(
            device=device, dtype=torch.float32
        ),
        "teacher_track_feat": [
            value[rows].to(device=device, dtype=torch.float32)
            for value in tensors["teacher_track_feat"]
        ],
        "teacher_track_support": [
            value[rows].to(device=device, dtype=torch.float32)
            for value in tensors["teacher_track_support"]
        ],
        "native_track_feat": [
            value[rows].to(device=device, dtype=torch.float32)
            for value in tensors["native_track_feat"]
        ],
        "native_track_support": native_support,
    }
    return output, re_feat_batch, re_support_batch, batch


# Assigned once in main so helper stays easy to test without a wrapper class.
_model_batch.frozen_cotracker_model = None


def _native_format_memory(
    re_feat_batch: Sequence[torch.Tensor], re_support_batch: Sequence[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    feature = [value.permute(2, 1, 0, 3).contiguous() for value in re_feat_batch]
    support = [value.permute(2, 1, 0, 3).contiguous() for value in re_support_batch]
    return feature, support


def _continue_cached(
    predictor: CoTrackerOnlinePredictor,
    initial,
    continuation_video_u8: torch.Tensor,
    device: str,
):
    restore_cotracker_online_state(predictor, initial)
    predictor(
        video_chunk=continuation_video_u8[None].to(device=device, dtype=torch.float32),
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    final = snapshot_cotracker_online_state(predictor)
    if final.online_coords_predicted.shape[1] != 24:
        raise RuntimeError("incomplete cached continuation")
    return final


def _future_coordinates_input(final, point_indices: torch.Tensor, predictor) -> torch.Tensor:
    indices = point_indices.to(final.online_coords_predicted.device)
    model_xy = final.online_coords_predicted[0, 16:24, indices].permute(1, 0, 2)
    return model_xy_to_input_xy(
        model_xy,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    ).detach().cpu()


def _compare_rows(
    left: Sequence[Mapping[str, float]], right: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("comparison row mismatch")
    error_reduction = [
        float(r["mean_l2_error_px"]) - float(l["mean_l2_error_px"])
        for l, r in zip(left, right)
    ]
    utility_gain = [
        float(l["threshold_utility"]) - float(r["threshold_utility"])
        for l, r in zip(left, right)
    ]
    return {
        "mean_error_reduction_px": float(sum(error_reduction) / len(error_reduction)),
        "threshold_utility_gain": float(sum(utility_gain) / len(utility_gain)),
        "positive_point_fraction": float(
            sum(value > 0 for value in error_reduction) / len(error_reduction)
        ),
        "error_reduction_rows": error_reduction,
        "utility_gain_rows": utility_gain,
    }


def evaluate_model(
    *,
    model: CounterfactualStructuredReextractionRestorer,
    predictor: CoTrackerOnlinePredictor,
    validation_index: Mapping[str, Any],
    config_sha256: str,
    device: str,
    bootstrap_seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    model.eval()
    all_failure_rows = {name: [] for name in VARIANTS}
    all_union_native = []
    all_union_gate = []
    all_failure_apply = []
    all_clean_apply = []
    all_clean_harmful = []
    video_reports = []
    zero_action_exact_all = True
    with torch.no_grad():
        for index_row in validation_index["rows"]:
            artifact = load_csrr_cache_video(
                index_row,
                expected_partition="fit_internal_validation",
                expected_config_sha256=config_sha256,
            )
            tensors = artifact["model_tensors"]
            count = int(tensors["point_indices"].numel())
            row_indices = torch.arange(count)
            output, re_feat_batch, re_support_batch, _ = _model_batch(
                model, tensors, row_indices, device
            )
            re_feat_native, re_support_native = _native_format_memory(
                re_feat_batch, re_support_batch
            )
            point_indices = tensors["point_indices"].long()
            apply_target = tensors["apply_target"].float()
            failure_local = torch.where(apply_target > 0.5)[0]
            clean_local = torch.where(apply_target <= 0.5)[0]
            exact_initial = exact_snapshot_from_artifact(artifact, device)
            zero_state = apply_reextracted_state_action(
                exact_initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=output["predicted_coordinates_xy"],
                apply_mask=torch.zeros(count, dtype=torch.bool),
                reextracted_track_features=re_feat_native,
                reextracted_track_supports=re_support_native,
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                visibility_residual=output["visibility_residual"],
                confidence_residual=output["confidence_residual"],
            )
            zero_action_exact = snapshots_exact(exact_initial, zero_state)
            zero_action_exact_all = zero_action_exact_all and zero_action_exact

            gt = tensors["future_gt_coordinates_xy"].float()
            visible = tensors["future_visible_mask"].bool()
            native_coords = tensors["native_future_coordinates_xy"].float()
            native_union_rows = point_future_rows(native_coords, gt, visible)
            failure_native_rows = [native_union_rows[i] for i in failure_local.tolist()]
            all_failure_rows["native"].extend(failure_native_rows)
            all_union_native.extend(native_union_rows)

            failure_points = point_indices[failure_local]
            failure_predicted = output["predicted_coordinates_xy"][failure_local]
            failure_apply_mask = torch.ones(failure_local.numel(), dtype=torch.bool)
            failure_feat = [value[:, :, failure_local] for value in re_feat_native]
            failure_support = [value[:, :, failure_local] for value in re_support_native]
            variants = {}
            for name, write_probability, write_memory in (
                ("coordinate_only", False, False),
                ("coordinate_probability", True, False),
                ("full_state", True, True),
            ):
                initial = apply_reextracted_state_action(
                    exact_initial,
                    point_indices=failure_points,
                    predicted_coordinates_input_xy=failure_predicted,
                    apply_mask=failure_apply_mask,
                    reextracted_track_features=failure_feat,
                    reextracted_track_supports=failure_support,
                    input_height=256,
                    input_width=256,
                    model_height=int(predictor.interp_shape[0]),
                    model_width=int(predictor.interp_shape[1]),
                    visibility_residual=output["visibility_residual"][failure_local],
                    confidence_residual=output["confidence_residual"][failure_local],
                    write_probability=write_probability,
                    write_memory=write_memory,
                )
                final = _continue_cached(
                    predictor, initial, tensors["continuation_video_u8"], device
                )
                coords = _future_coordinates_input(final, failure_points, predictor)
                rows = point_future_rows(
                    coords, gt[failure_local], visible[failure_local]
                )
                all_failure_rows[name].extend(rows)
                variants[name] = rows

            apply_mask = torch.sigmoid(output["apply_logit"]) >= 0.5
            gate_initial = apply_reextracted_state_action(
                exact_initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=output["predicted_coordinates_xy"],
                apply_mask=apply_mask.cpu(),
                reextracted_track_features=re_feat_native,
                reextracted_track_supports=re_support_native,
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                visibility_residual=output["visibility_residual"],
                confidence_residual=output["confidence_residual"],
                write_probability=True,
                write_memory=True,
            )
            gate_final = _continue_cached(
                predictor, gate_initial, tensors["continuation_video_u8"], device
            )
            gate_coords = _future_coordinates_input(gate_final, point_indices, predictor)
            gate_rows = point_future_rows(gate_coords, gt, visible)
            all_union_gate.extend(gate_rows)
            for local in failure_local.tolist():
                all_failure_apply.append(bool(apply_mask[local].item()))
            for local in clean_local.tolist():
                applied = bool(apply_mask[local].item())
                all_clean_apply.append(applied)
                all_clean_harmful.append(
                    applied
                    and float(gate_rows[local]["mean_l2_error_px"])
                    > float(native_union_rows[local]["mean_l2_error_px"])
                )

            full_vs_native = _compare_rows(variants["full_state"], failure_native_rows)
            full_vs_coordinate = _compare_rows(
                variants["full_state"], variants["coordinate_only"]
            )
            full_vs_probability = _compare_rows(
                variants["full_state"], variants["coordinate_probability"]
            )
            video_reports.append(
                {
                    "source_index": int(index_row["source_index"]),
                    "video_name": artifact["video_name"],
                    "failure_points": int(failure_local.numel()),
                    "clean_points": int(clean_local.numel()),
                    "zero_action_state_exact": zero_action_exact,
                    "forced_failure": {
                        "native": aggregate_point_rows(failure_native_rows),
                        **{
                            name: aggregate_point_rows(rows)
                            for name, rows in variants.items()
                        },
                        "full_state_vs_native": full_vs_native,
                        "full_state_vs_coordinate_only": full_vs_coordinate,
                        "full_state_vs_coordinate_probability": full_vs_probability,
                    },
                    "learned_gate": {
                        "applied_rows": int(apply_mask.sum().item()),
                        "failure_applied": int(apply_mask[failure_local].sum().item()),
                        "clean_applied": int(apply_mask[clean_local].sum().item()),
                    },
                }
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    aggregates = {name: aggregate_point_rows(rows) for name, rows in all_failure_rows.items()}
    full_vs_native = _compare_rows(
        all_failure_rows["full_state"], all_failure_rows["native"]
    )
    full_vs_coordinate = _compare_rows(
        all_failure_rows["full_state"], all_failure_rows["coordinate_only"]
    )
    full_vs_probability = _compare_rows(
        all_failure_rows["full_state"], all_failure_rows["coordinate_probability"]
    )
    native_severe = float(aggregates["native"]["severe_16px_rate"])
    full_severe = float(aggregates["full_state"]["severe_16px_rate"])
    full_video_better_coordinate = [
        report["forced_failure"]["full_state_vs_coordinate_only"][
            "mean_error_reduction_px"
        ]
        > 0
        for report in video_reports
    ]
    union_native = aggregate_point_rows(all_union_native)
    union_gate = aggregate_point_rows(all_union_gate)
    failure_recall = float(sum(all_failure_apply) / len(all_failure_apply))
    clean_false_apply = float(sum(all_clean_apply) / len(all_clean_apply))
    clean_harmful = float(sum(all_clean_harmful) / len(all_clean_harmful))
    result = {
        "forced_failure": {
            "variants": aggregates,
            "full_state_vs_native": {
                **full_vs_native,
                "error_reduction_CI": paired_video_bootstrap_ci(
                    full_vs_native["error_reduction_rows"],
                    seed=bootstrap_seed,
                    samples=bootstrap_samples,
                ),
                "utility_gain_CI": paired_video_bootstrap_ci(
                    full_vs_native["utility_gain_rows"],
                    seed=bootstrap_seed + 1,
                    samples=bootstrap_samples,
                ),
                "severe_16px_rate_reduction": native_severe - full_severe,
            },
            "full_state_vs_coordinate_only": {
                **full_vs_coordinate,
                "better_video_fraction": float(
                    sum(full_video_better_coordinate)
                    / len(full_video_better_coordinate)
                ),
            },
            "full_state_vs_coordinate_probability": full_vs_probability,
        },
        "learned_gate": {
            "failure_apply_recall": failure_recall,
            "clean_false_apply_rate": clean_false_apply,
            "clean_harmful_rate": clean_harmful,
            "union_native": union_native,
            "union_selected": union_gate,
            "union_threshold_utility_gain_vs_native": float(
                union_gate["threshold_utility"] - union_native["threshold_utility"]
            ),
            "union_mean_error_reduction_vs_native_px": float(
                union_native["mean_l2_error_px"] - union_gate["mean_l2_error_px"]
            ),
        },
        "zero_action_native_parity_exact": zero_action_exact_all,
        "video_reports": video_reports,
    }
    result["checkpoint_score"] = float(
        result["forced_failure"]["full_state_vs_native"]["threshold_utility_gain"]
        - 0.5 * clean_false_apply
        - 0.5 * clean_harmful
    )
    return result


def _candidate_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_tuple = (
        float(candidate["checkpoint_score"]),
        -float(candidate["forced_failure"]["variants"]["full_state"]["mean_l2_error_px"]),
        -float(candidate["learned_gate"]["clean_false_apply_rate"]),
    )
    best_tuple = (
        float(best["checkpoint_score"]),
        -float(best["forced_failure"]["variants"]["full_state"]["mean_l2_error_px"]),
        -float(best["learned_gate"]["clean_false_apply_rate"]),
    )
    return candidate_tuple > best_tuple


def _scientific_checks(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    gates = config["scientific_gates"]
    forced = metrics["forced_failure"]
    native = forced["full_state_vs_native"]
    coord = forced["full_state_vs_coordinate_only"]
    probability = forced["full_state_vs_coordinate_probability"]
    learned = metrics["learned_gate"]
    return {
        "validation_failure_points_min": int(forced["variants"]["native"]["points"])
        >= int(gates["validation_failure_points_min"]),
        "validation_failure_videos_min": sum(
            report["failure_points"] > 0 for report in metrics["video_reports"]
        )
        >= int(gates["validation_failure_videos_min"]),
        "validation_clean_points_min": int(learned["union_native"]["points"])
        - int(forced["variants"]["native"]["points"])
        >= int(gates["validation_clean_points_min"]),
        "validation_clean_videos_min": sum(
            report["clean_points"] > 0 for report in metrics["video_reports"]
        )
        >= int(gates["validation_clean_videos_min"]),
        "forced_full_state_mean_error_reduction_vs_native_px_min": float(
            native["mean_error_reduction_px"]
        )
        >= float(gates["forced_full_state_mean_error_reduction_vs_native_px_min"]),
        "forced_full_state_error_reduction_CI_lower_px_min": float(
            native["error_reduction_CI"]["lower"]
        )
        >= float(gates["forced_full_state_error_reduction_CI_lower_px_min"]),
        "forced_full_state_threshold_utility_gain_vs_native_min": float(
            native["threshold_utility_gain"]
        )
        >= float(gates["forced_full_state_threshold_utility_gain_vs_native_min"]),
        "forced_full_state_utility_gain_CI_lower_min": float(
            native["utility_gain_CI"]["lower"]
        )
        >= float(gates["forced_full_state_utility_gain_CI_lower_min"]),
        "forced_full_state_positive_point_fraction_min": float(
            native["positive_point_fraction"]
        )
        >= float(gates["forced_full_state_positive_point_fraction_min"]),
        "forced_full_state_severe_16px_rate_reduction_min": float(
            native["severe_16px_rate_reduction"]
        )
        >= float(gates["forced_full_state_severe_16px_rate_reduction_min"]),
        "forced_full_state_mean_error_reduction_vs_coordinate_only_px_min": float(
            coord["mean_error_reduction_px"]
        )
        >= float(
            gates["forced_full_state_mean_error_reduction_vs_coordinate_only_px_min"]
        ),
        "forced_full_state_threshold_utility_gain_vs_coordinate_only_min": float(
            coord["threshold_utility_gain"]
        )
        >= float(
            gates["forced_full_state_threshold_utility_gain_vs_coordinate_only_min"]
        ),
        "forced_full_state_better_than_coordinate_only_video_fraction_min": float(
            coord["better_video_fraction"]
        )
        >= float(
            gates[
                "forced_full_state_better_than_coordinate_only_video_fraction_min"
            ]
        ),
        "forced_full_state_mean_error_reduction_vs_coordinate_probability_px_min": float(
            probability["mean_error_reduction_px"]
        )
        >= float(
            gates[
                "forced_full_state_mean_error_reduction_vs_coordinate_probability_px_min"
            ]
        ),
        "forced_full_state_threshold_utility_gain_vs_coordinate_probability_min": float(
            probability["threshold_utility_gain"]
        )
        >= float(
            gates[
                "forced_full_state_threshold_utility_gain_vs_coordinate_probability_min"
            ]
        ),
        "learned_gate_failure_apply_recall_min": float(
            learned["failure_apply_recall"]
        )
        >= float(gates["learned_gate_failure_apply_recall_min"]),
        "learned_gate_clean_false_apply_rate_max": float(
            learned["clean_false_apply_rate"]
        )
        <= float(gates["learned_gate_clean_false_apply_rate_max"]),
        "learned_gate_clean_harmful_rate_max": float(learned["clean_harmful_rate"])
        <= float(gates["learned_gate_clean_harmful_rate_max"]),
        "learned_gate_union_threshold_utility_gain_vs_native_min": float(
            learned["union_threshold_utility_gain_vs_native"]
        )
        >= float(gates["learned_gate_union_threshold_utility_gain_vs_native_min"]),
        "learned_gate_union_mean_error_reduction_vs_native_px_min": float(
            learned["union_mean_error_reduction_vs_native_px"]
        )
        >= float(gates["learned_gate_union_mean_error_reduction_vs_native_px_min"]),
        "zero_action_native_parity_exact": bool(
            metrics["zero_action_native_parity_exact"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interface-smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != CSRR_SCHEMA_VERSION:
        raise ValueError("unexpected Gate 2 training config")
    config_sha256 = file_sha256(config_path)
    cache_root = Path(args.cache_root).resolve()
    train_index = load_complete_csrr_cache_index(
        cache_root / "train/cache_index.json", expected_partition="train"
    )
    validation_index = load_complete_csrr_cache_index(
        cache_root / "fit_internal_validation/cache_index.json",
        expected_partition="fit_internal_validation",
    )
    if train_index["config_sha256"] != config_sha256 or validation_index["config_sha256"] != config_sha256:
        raise ValueError("cache/config hash mismatch")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("locked-data flags must remain false")

    seed = int(config["training"]["seed"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(checkpoint=config["backbone"]["checkpoint"]).to(args.device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    _model_batch.frozen_cotracker_model = predictor.model
    model = CounterfactualStructuredReextractionRestorer().to(args.device)
    if model.trainable_parameter_count > int(config["model"]["trainable_parameter_ceiling"]):
        raise ValueError("CSRR parameter ceiling exceeded")

    train_videos = []
    for row in train_index["rows"]:
        artifact = load_csrr_cache_video(
            row,
            expected_partition="train",
            expected_config_sha256=config_sha256,
        )
        train_videos.append(
            {
                "source_index": int(row["source_index"]),
                "tensors": artifact["model_tensors"],
            }
        )
    if args.interface_smoke:
        tensors = train_videos[0]["tensors"]
        rows = torch.arange(min(4, int(tensors["point_indices"].numel())))
        model.train()
        output, re_feat, re_support, batch = _model_batch(model, tensors, rows, args.device)
        losses = csrr_batch_loss(
            output=output,
            reextracted_track_features=re_feat,
            reextracted_track_supports=re_support,
            batch=batch,
            config=_loss_config(config),
        )
        losses["loss"].backward()
        result = {
            "status": "interface_smoke_pass",
            "source_index": train_videos[0]["source_index"],
            "rows": int(rows.numel()),
            "trainable_parameters": model.trainable_parameter_count,
            "loss": float(losses["loss"].detach().cpu()),
            "all_parameter_gradients_finite": all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ),
            "model_validation_read": False,
        }
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    loss_config = _loss_config(config)
    max_epochs = int(config["training"]["max_epochs"])
    logical_batch_size = int(config["training"]["batch_size"])
    patience = int(config["training"]["early_stopping"]["patience"])
    min_score_improvement = float(
        config["training"]["early_stopping"]["minimum_score_improvement"]
    )
    entries = [
        (video_index, row_index)
        for video_index, video in enumerate(train_videos)
        for row_index in range(int(video["tensors"]["point_indices"].numel()))
    ]

    history = []
    best_metrics = None
    best_epoch = -1
    best_score_for_stale = -float("inf")
    stale = 0
    initial_metrics = evaluate_model(
        model=model,
        predictor=predictor,
        validation_index=validation_index,
        config_sha256=config_sha256,
        device=args.device,
        bootstrap_seed=seed * 1000,
    )
    best_metrics = initial_metrics
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history.append({"epoch": -1, "train": None, "validation": initial_metrics})
    best_score_for_stale = float(initial_metrics["checkpoint_score"])

    for epoch in range(max_epochs):
        model.train()
        if epoch == 0:
            lr_scale = 1.0
        else:
            progress = (epoch - 1) / max(max_epochs - 1, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = float(optimizer_cfg["learning_rate"]) * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(entries), generator=generator).tolist()
        epoch_sums: dict[str, float] = defaultdict(float)
        epoch_rows = 0
        for start in range(0, len(order), logical_batch_size):
            selection = [entries[order[index]] for index in range(start, min(start + logical_batch_size, len(order)))]
            grouped: dict[int, list[int]] = defaultdict(list)
            for video_index, row_index in selection:
                grouped[video_index].append(row_index)
            optimizer.zero_grad(set_to_none=True)
            logical_rows = len(selection)
            for video_index, local_rows in grouped.items():
                tensors = train_videos[video_index]["tensors"]
                row_tensor = torch.tensor(local_rows, dtype=torch.long)
                output, re_feat, re_support, batch = _model_batch(
                    model, tensors, row_tensor, args.device
                )
                losses = csrr_batch_loss(
                    output=output,
                    reextracted_track_features=re_feat,
                    reextracted_track_supports=re_support,
                    batch=batch,
                    config=loss_config,
                )
                weight = float(len(local_rows)) / float(logical_rows)
                (losses["loss"] * weight).backward()
                for key, value in losses.items():
                    epoch_sums[key] += float(value.detach().cpu()) * len(local_rows)
                epoch_rows += len(local_rows)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
        train_metrics = {
            key: value / max(epoch_rows, 1) for key, value in epoch_sums.items()
        }
        train_metrics.update({"rows": epoch_rows, "learning_rate": current_lr})
        validation_metrics = evaluate_model(
            model=model,
            predictor=predictor,
            validation_index=validation_index,
            config_sha256=config_sha256,
            device=args.device,
            bootstrap_seed=seed * 1000 + epoch + 1,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        if _candidate_better(validation_metrics, best_metrics):
            best_metrics = validation_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        score = float(validation_metrics["checkpoint_score"])
        if score > best_score_for_stale + min_score_improvement:
            best_score_for_stale = score
            stale = 0
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "checkpoint_score": score,
                    "forced_full_vs_native_error_reduction": validation_metrics[
                        "forced_failure"
                    ]["full_state_vs_native"]["mean_error_reduction_px"],
                    "forced_full_vs_native_utility_gain": validation_metrics[
                        "forced_failure"
                    ]["full_state_vs_native"]["threshold_utility_gain"],
                    "clean_false_apply_rate": validation_metrics["learned_gate"][
                        "clean_false_apply_rate"
                    ],
                    "stale": stale,
                }
            ),
            flush=True,
        )
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(
        model=model,
        predictor=predictor,
        validation_index=validation_index,
        config_sha256=config_sha256,
        device=args.device,
        bootstrap_seed=seed * 1000 + 999,
    )
    checks = _scientific_checks(final_metrics, config)
    local_pass = all(checks.values())
    checkpoint = {
        "schema_version": "routeD_csrr_gate2_checkpoint_v0",
        "epoch": best_epoch,
        "model_state": best_state,
        "model_state_hashes": {key: tensor_sha256(value) for key, value in best_state.items()},
        "model_state_digest": canonical_json_sha256(
            {key: tensor_sha256(value) for key, value in best_state.items()}
        ),
        "config_sha256": config_sha256,
        "train_cache_index_sha256": train_index["_index_sha256"],
        "validation_cache_index_sha256": validation_index["_index_sha256"],
        "validation": final_metrics,
    }
    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema_version": "routeD_csrr_gate2_training_result_v0",
        "status": "completed_pass" if local_pass else "completed_fail",
        "decision": (
            "AUTHORIZE_INDEPENDENT_REPLAY_AND_PACKAGE"
            if local_pass
            else "STOP_LEARNED_STATE_RESTORER_BEFORE_MODEL_VALIDATION"
        ),
        "local_pass": local_pass,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "seed": seed,
        "train_cache_index": train_index["_index_path"],
        "train_cache_index_sha256": train_index["_index_sha256"],
        "validation_cache_index": validation_index["_index_path"],
        "validation_cache_index_sha256": validation_index["_index_sha256"],
        "trainable_parameters": model.trainable_parameter_count,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_state_digest": checkpoint["model_state_digest"],
        "history": history,
        "validation": final_metrics,
        "checks": checks,
        "integrity": {
            "gradient_train_source_indices": list(range(8, 32)),
            "fit_internal_validation_source_indices": list(range(32, 48)),
            "model_validation_read": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
            "official_kinetics_1144_rerun": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir / "metrics.json"),
                "status": report["status"],
                "decision": report["decision"],
                "best_epoch": best_epoch,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "model_state_digest": report["model_state_digest"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
