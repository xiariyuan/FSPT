#!/usr/bin/env python3
"""Run the fit-only Route-D coordinate basin audit on design-exposed rows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    restore_cotracker_online_state,
    snapshot_cotracker_online_state,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    apply_reextracted_state_action,
    deterministic_reextract_cotracker_memory,
    model_xy_to_input_xy,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    load_complete_csrr_cache_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    exact_snapshot_from_artifact,
    load_csrr_cache_video,
    point_future_rows,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_state_reinstantiation_basin import (
    BASIN_SCHEMA_VERSION,
    aggregate_basin_rows,
    build_offset_hypotheses,
    evaluate_radius_gates,
    offset_and_clip_coordinates,
)
from scripts.audit_routeD_cotracker3_interface import set_deterministic


DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_state_reinstantiation_basin_gate2_5_v0.yaml"
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs/routeD_counterfactual_state_restorer_cache_20260719"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_state_reinstantiation_basin_gate2_5_20260719/primary.json"


def _resolve_repo_fallback(path: str | Path) -> Path:
    configured = Path(path)
    if configured.exists():
        return configured.resolve()
    fallback = REPO_ROOT / configured.name
    return fallback.resolve()


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != BASIN_SCHEMA_VERSION:
        raise ValueError("unexpected Gate 2.5 basin schema")
    parent_path = _resolve_repo_fallback(config["frozen_parent"]["gate2_config"])
    if not parent_path.exists():
        raise FileNotFoundError(f"missing frozen Gate 2 config: {parent_path}")
    expected_hash = str(config["frozen_parent"]["gate2_config_sha256"])
    if file_sha256(parent_path) != expected_hash:
        raise ValueError("frozen Gate 2 config hash drift")
    parent = yaml.safe_load(parent_path.read_text())
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 2.5 locked-data flags must remain false")
    if bool(config["frozen_parent"]["learned_gate2_checkpoint_used"]):
        raise ValueError("Gate 2.5 must not load the learned Gate 2 checkpoint")
    if int(config["frozen_parent"]["trainable_parameters"]) != 0:
        raise ValueError("Gate 2.5 is nonparametric")
    return config, parent, parent_path


def _continue_cached(
    predictor: CoTrackerOnlinePredictor,
    initial,
    continuation_video_u8: torch.Tensor,
    device: str,
):
    restore_cotracker_online_state(predictor, initial)
    predictor(
        video_chunk=continuation_video_u8[None].to(
            device=device, dtype=torch.float32
        ),
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    final = snapshot_cotracker_online_state(predictor)
    if final.online_coords_predicted.shape[1] != 24:
        raise RuntimeError("incomplete Gate 2.5 continuation")
    return final


def _future_coordinates_input(
    final, point_indices: torch.Tensor, predictor: CoTrackerOnlinePredictor
) -> torch.Tensor:
    indices = point_indices.to(final.online_coords_predicted.device)
    model_xy = final.online_coords_predicted[0, 16:24, indices].permute(1, 0, 2)
    return model_xy_to_input_xy(
        model_xy,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    ).detach().cpu()


def _native_format_memory(
    reextracted_track_features: Sequence[torch.Tensor],
    reextracted_track_supports: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # Deterministic extraction treats hypotheses as the batch dimension;
    # CoTracker online state stores point identity in dimension two.
    feature = [
        value.permute(2, 1, 0, 3).contiguous()
        for value in reextracted_track_features
    ]
    support = [
        value.permute(2, 1, 0, 3).contiguous()
        for value in reextracted_track_supports
    ]
    return feature, support


def _rowwise_cosine_min(
    predicted: Sequence[torch.Tensor],
    teacher: Sequence[torch.Tensor],
) -> float:
    minima = []
    for left, right in zip(predicted, teacher):
        # Predicted is native state layout [1,S,R,C]; cache is [R,S,C].
        left_rows = left[0].permute(1, 0, 2).float().flatten(1).cpu()
        right_rows = right.float().flatten(1).cpu()
        minima.append(float(F.cosine_similarity(left_rows, right_rows, dim=-1).min()))
    return min(minima)


def _direction_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["variant"]),
                float(row["nominal_radius_px"]),
                str(row["direction"]),
            )
        ].append(row)
    output: dict[str, Any] = defaultdict(dict)
    for (variant, radius, direction), group in sorted(grouped.items()):
        error_reduction = [
            float(row["native_mean_l2_error_px"]) - float(row["mean_l2_error_px"])
            for row in group
        ]
        utility_gain = [
            float(row["threshold_utility"])
            - float(row["native_threshold_utility"])
            for row in group
        ]
        output[variant][f"{radius:g}_{direction}"] = {
            "nominal_radius_px": radius,
            "direction": direction,
            "rows": len(group),
            "mean_error_reduction_vs_native_px": float(np.mean(error_reduction)),
            "threshold_utility_gain_vs_native": float(np.mean(utility_gain)),
            "positive_row_fraction": float(
                np.mean([value > 0.0 for value in error_reduction])
            ),
        }
    return dict(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_path = Path(args.config).resolve()
    config, parent, parent_path = _load_protocol(config_path)
    cache_cfg = config["cache"]
    cache_index_path = Path(args.cache_root).resolve() / str(cache_cfg["index"])
    cache_index = load_complete_csrr_cache_index(
        cache_index_path, expected_partition=str(cache_cfg["expected_partition"])
    )
    expected_sources = [int(value) for value in cache_cfg["expected_source_indices"]]
    if cache_index["expected_source_indices"] != expected_sources:
        raise ValueError("Gate 2.5 design-exposed source membership drift")
    if cache_index["config_sha256"] != str(
        cache_cfg["expected_gate2_config_sha256"]
    ):
        raise ValueError("Gate 2.5 cache parent-config drift")
    if any(index >= 8 for index in expected_sources):
        raise ValueError("Gate 2.5 may read design-exposed indices 0-7 only")

    bootstrap = config["metrics"]["video_cluster_bootstrap"]
    seed = int(bootstrap["seed"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=str(parent["backbone"]["checkpoint"])
    ).to(args.device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)

    hypotheses = build_offset_hypotheses(
        config["offset_schedule"]["nominal_radii_px"],
        config["offset_schedule"]["directions_xy"],
    )
    variants = config["state_action"]["variants"]
    all_rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    center_feature_cosines: list[float] = []
    center_support_cosines: list[float] = []

    with torch.no_grad():
        for index_row in cache_index["rows"]:
            sidecar = Path(index_row["sidecar"])
            if file_sha256(sidecar) != str(index_row["sidecar_sha256"]):
                raise ValueError(f"Gate 2.5 sidecar hash drift: {sidecar}")
            artifact = load_csrr_cache_video(
                index_row,
                expected_partition=str(cache_cfg["expected_partition"]),
                expected_config_sha256=str(cache_cfg["expected_gate2_config_sha256"]),
            )
            tensors = artifact["model_tensors"]
            failure_local = torch.where(tensors["apply_target"].float() > 0.5)[0]
            if failure_local.numel() == 0:
                videos.append(
                    {
                        "source_index": int(index_row["source_index"]),
                        "video_name": str(artifact["video_name"]),
                        "failure_points": 0,
                    }
                )
                continue
            point_indices = tensors["point_indices"][failure_local].long()
            exact_teacher_xy = tensors.get(
                "teacher_commit_coordinates_xy_float32"
            )
            if not isinstance(exact_teacher_xy, torch.Tensor):
                raise ValueError(
                    "Gate 2.5 requires the design-exposed float32 teacher coordinate"
                )
            if exact_teacher_xy.dtype != torch.float32:
                raise ValueError("Gate 2.5 teacher coordinate must remain float32")
            teacher_xy = exact_teacher_xy[failure_local].to(device=args.device)
            feature_pyramid = [
                value[None, None].to(device=args.device, dtype=torch.float32)
                for value in tensors["frame_feature_pyramid"]
            ]
            exact_initial = exact_snapshot_from_artifact(artifact, args.device)
            gt = tensors["future_gt_coordinates_xy"][failure_local].float()
            visible = tensors["future_visible_mask"][failure_local].bool()
            native_rows = point_future_rows(
                tensors["native_future_coordinates_xy"][failure_local].float(),
                gt,
                visible,
            )
            exact_teacher_visibility = tensors.get(
                "teacher_visibility_probability_float32"
            )
            exact_teacher_confidence = tensors.get(
                "teacher_confidence_probability_float32"
            )
            if not isinstance(exact_teacher_visibility, torch.Tensor) or not isinstance(
                exact_teacher_confidence, torch.Tensor
            ):
                raise ValueError(
                    "Gate 2.5 requires design-exposed float32 teacher probabilities"
                )
            if (
                exact_teacher_visibility.dtype != torch.float32
                or exact_teacher_confidence.dtype != torch.float32
            ):
                raise ValueError("Gate 2.5 teacher probabilities must remain float32")
            teacher_visibility = exact_teacher_visibility[failure_local]
            teacher_confidence = exact_teacher_confidence[failure_local]
            point_device = point_indices.to(exact_initial.online_vis_predicted.device)
            native_visibility = torch.sigmoid(
                exact_initial.online_vis_predicted[0, 15, point_device]
            ).detach().cpu()
            native_confidence = torch.sigmoid(
                exact_initial.online_conf_predicted[0, 15, point_device]
            ).detach().cpu()

            for hypothesis in hypotheses:
                candidate_xy, realized_error, clipped = offset_and_clip_coordinates(
                    teacher_xy,
                    hypothesis["offset_xy_px"],
                    input_height=int(config["offset_schedule"]["input_raster"]),
                    input_width=int(config["offset_schedule"]["input_raster"]),
                )
                re_feat_batch, re_support_batch = (
                    deterministic_reextract_cotracker_memory(
                        feature_pyramid,
                        candidate_xy,
                        input_height=256,
                        input_width=256,
                        model_height=int(predictor.interp_shape[0]),
                        model_width=int(predictor.interp_shape[1]),
                        stride=int(parent["backbone"]["stride"]),
                        support_radius=int(config["state_action"]["support_radius"]),
                    )
                )
                re_feat, re_support = _native_format_memory(
                    re_feat_batch, re_support_batch
                )
                if float(hypothesis["nominal_radius_px"]) == 0.0:
                    center_feature_cosines.append(
                        _rowwise_cosine_min(
                            re_feat,
                            [value[failure_local] for value in tensors["teacher_track_feat"]],
                        )
                    )
                    center_support_cosines.append(
                        _rowwise_cosine_min(
                            re_support,
                            [
                                value[failure_local]
                                for value in tensors["teacher_track_support"]
                            ],
                        )
                    )

                for variant_name, variant_cfg in variants.items():
                    use_teacher_probability = (
                        variant_cfg["visibility_confidence_write"] is not False
                    )
                    initial = apply_reextracted_state_action(
                        exact_initial,
                        point_indices=point_indices,
                        predicted_coordinates_input_xy=candidate_xy,
                        apply_mask=torch.ones(point_indices.numel(), dtype=torch.bool),
                        reextracted_track_features=re_feat,
                        reextracted_track_supports=re_support,
                        input_height=256,
                        input_width=256,
                        model_height=int(predictor.interp_shape[0]),
                        model_width=int(predictor.interp_shape[1]),
                        visibility_residual=(
                            teacher_visibility - native_visibility
                            if use_teacher_probability
                            else None
                        ),
                        confidence_residual=(
                            teacher_confidence - native_confidence
                            if use_teacher_probability
                            else None
                        ),
                        write_probability=use_teacher_probability,
                        write_memory=True,
                    )
                    final = _continue_cached(
                        predictor,
                        initial,
                        tensors["continuation_video_u8"],
                        args.device,
                    )
                    outcome_rows = point_future_rows(
                        _future_coordinates_input(final, point_indices, predictor),
                        gt,
                        visible,
                    )
                    for local_index, (outcome, native) in enumerate(
                        zip(outcome_rows, native_rows)
                    ):
                        all_rows.append(
                            {
                                "source_index": int(index_row["source_index"]),
                                "video_name": str(artifact["video_name"]),
                                "point_index": int(point_indices[local_index]),
                                "variant": str(variant_name),
                                "hypothesis_id": str(hypothesis["hypothesis_id"]),
                                "nominal_radius_px": float(
                                    hypothesis["nominal_radius_px"]
                                ),
                                "direction": str(hypothesis["direction"]),
                                "offset_xy_px": list(hypothesis["offset_xy_px"]),
                                "realized_commit_error_px": float(
                                    realized_error[local_index].cpu()
                                ),
                                "boundary_clipped": bool(clipped[local_index].cpu()),
                                **outcome,
                                "native_mean_l2_error_px": float(
                                    native["mean_l2_error_px"]
                                ),
                                "native_severe_16px_rate": float(
                                    native["severe_16px_rate"]
                                ),
                                "native_threshold_utility": float(
                                    native["threshold_utility"]
                                ),
                            }
                        )
            videos.append(
                {
                    "source_index": int(index_row["source_index"]),
                    "video_name": str(artifact["video_name"]),
                    "failure_points": int(failure_local.numel()),
                }
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    failure_points = sum(int(row["failure_points"]) for row in videos)
    failure_videos = sum(int(row["failure_points"]) > 0 for row in videos)
    row_cfg = config["rows"]
    if failure_points < int(row_cfg["expected_failure_points_min"]):
        raise RuntimeError("Gate 2.5 has too few design-exposed failure points")
    if failure_videos < int(row_cfg["expected_failure_videos_min"]):
        raise RuntimeError("Gate 2.5 has too few design-exposed failure videos")

    aggregate = aggregate_basin_rows(
        all_rows,
        bootstrap_seed=seed,
        bootstrap_samples=int(bootstrap["samples"]),
    )
    gate_calibration = evaluate_radius_gates(
        aggregate, config["downstream_gate_calibration"]
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": BASIN_SCHEMA_VERSION,
        "status": "completed",
        "decision": config["formal_decisions"]["always"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "frozen_gate2_config": str(parent_path),
        "frozen_gate2_config_sha256": file_sha256(parent_path),
        "cache_index": str(cache_index_path),
        "cache_index_sha256": file_sha256(cache_index_path),
        "cache_combined_tensor_hash_digest": cache_index[
            "combined_tensor_hash_digest"
        ],
        "videos": videos,
        "failure_points": failure_points,
        "failure_videos": failure_videos,
        "hypotheses": hypotheses,
        "center_reextraction_integrity": {
            "minimum_track_feature_cosine": min(center_feature_cosines),
            "minimum_track_support_cosine": min(center_support_cosines),
        },
        "aggregate": aggregate,
        "direction_summary": _direction_summary(all_rows),
        "downstream_gate_calibration": gate_calibration,
        "rows": all_rows,
        "locked_data": config["locked_data"],
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": report["decision"],
                "failure_points": failure_points,
                "failure_videos": failure_videos,
                "center_reextraction_integrity": report[
                    "center_reextraction_integrity"
                ],
                "downstream_gate_calibration": gate_calibration,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
