#!/usr/bin/env python3
"""Teacher-nearest shortlist future rollout audit for Route-D Gate 3C1C."""
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
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    apply_reextracted_state_action,
    deterministic_reextract_cotracker_memory,
    extract_cotracker_observed_feature_pyramid,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    aggregate_point_rows,
    point_future_rows,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_state_reinstantiation_basin import (
    video_cluster_bootstrap_ci,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_selector import (
    QueryClosureIdentitySelectorConfig,
    select_query_closure_identity_candidates,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)
from scripts.audit_routeD_oracle_state_transplant_gate1 import (
    _continue_second_window,
    _coords_to_input,
    _initialize_and_first_window,
    _original_queries,
)
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
    verify_temporal_identity_cache_payload,
)

SCHEMA = "routeD_temporal_identity_future_rollout_gate3c1c_v0"
RESULT_SCHEMA = "routeD_temporal_identity_future_rollout_result_gate3c1c_v0"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/routeD_temporal_identity_future_rollout_gate3c1c_v0.yaml"
)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _selector_config(payload: Mapping[str, Any]) -> QueryClosureIdentitySelectorConfig:
    mechanism = payload["mechanism"]
    selection = payload["selection"]
    return QueryClosureIdentitySelectorConfig(
        cycle_weight=float(mechanism["cycle_weight"]),
        query_frame_identity_weight=float(
            mechanism["query_frame_identity_weight"]
        ),
        mean_identity_weight=float(mechanism["mean_identity_weight"]),
        minimum_identity_weight=float(mechanism["minimum_identity_weight"]),
        retained_nonnative=int(selection["retained_nonnative"]),
        zscore_epsilon=float(
            mechanism["candidate_relative_zscore_epsilon"]
        ),
    )


def freeze_shortlist(
    tensors: Mapping[str, torch.Tensor],
    selector_config: QueryClosureIdentitySelectorConfig,
) -> dict[str, torch.Tensor]:
    """Compute the causal shortlist without accepting teacher tensors."""
    return select_query_closure_identity_candidates(
        temporal_features=tensors["temporal_features"].float(),
        static_features=tensors["static_features"].float(),
        query_frames=tensors["query_frames"].long(),
        valid_mask=tensors["candidate_valid_mask"].bool(),
        config=selector_config,
    )


def teacher_nearest_in_frozen_shortlist(
    *,
    selected_indices: torch.Tensor,
    candidate_coordinates_xy: torch.Tensor,
    teacher_candidate_distance_px: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the teacher only inside an already frozen shortlist."""
    if selected_indices.ndim != 2:
        raise ValueError("selected indices must have shape [R,S]")
    rows, shortlist = selected_indices.shape
    if candidate_coordinates_xy.ndim != 3 or candidate_coordinates_xy.shape[:2] != (
        rows,
        teacher_candidate_distance_px.shape[1],
    ):
        raise ValueError("candidate-coordinate shape mismatch")
    if teacher_candidate_distance_px.shape != candidate_coordinates_xy.shape[:2]:
        raise ValueError("teacher-distance shape mismatch")
    gather_xy = selected_indices[..., None].expand(rows, shortlist, 2)
    shortlist_xy = candidate_coordinates_xy.gather(1, gather_xy)
    shortlist_distance = teacher_candidate_distance_px.gather(1, selected_indices)
    if torch.any(shortlist_distance < 0.0):
        raise ValueError("frozen shortlist contains invalid teacher distance")
    nearest_position = shortlist_distance.argmin(dim=1)
    row_index = torch.arange(rows, device=selected_indices.device)
    selected_candidate_index = selected_indices[row_index, nearest_position]
    selected_xy = shortlist_xy[row_index, nearest_position]
    selected_distance = shortlist_distance[row_index, nearest_position]
    return {
        "shortlist_coordinates_xy": shortlist_xy,
        "shortlist_teacher_distance_px": shortlist_distance,
        "nearest_shortlist_position": nearest_position,
        "selected_candidate_index": selected_candidate_index,
        "selected_coordinates_xy": selected_xy,
        "selected_teacher_distance_px": selected_distance,
    }


def _native_format_memory(
    features: Sequence[torch.Tensor], supports: Sequence[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        [value.permute(2, 1, 0, 3).contiguous() for value in features],
        [value.permute(2, 1, 0, 3).contiguous() for value in supports],
    )


def _future_coordinates(
    snapshot: Any,
    point_indices: torch.Tensor,
    predictor: CoTrackerOnlinePredictor,
) -> torch.Tensor:
    all_coordinates = _coords_to_input(
        snapshot,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    )
    return all_coordinates[point_indices.long(), 16:24].contiguous()


def _comparison_rows(
    left_rows: Sequence[Mapping[str, float]],
    right_rows: Sequence[Mapping[str, float]],
    *,
    source_index: int,
    point_indices: torch.Tensor,
    prefix: str,
) -> list[dict[str, Any]]:
    if len(left_rows) != len(right_rows) or len(left_rows) != int(point_indices.numel()):
        raise ValueError("comparison row count mismatch")
    output = []
    for local, (left, right) in enumerate(zip(left_rows, right_rows)):
        output.append(
            {
                "source_index": int(source_index),
                "point_index": int(point_indices[local]),
                f"{prefix}_error_reduction_px": float(left["mean_l2_error_px"])
                - float(right["mean_l2_error_px"]),
                f"{prefix}_utility_gain": float(right["threshold_utility"])
                - float(left["threshold_utility"]),
                f"{prefix}_severe_rate_reduction": float(left["severe_16px_rate"])
                - float(right["severe_16px_rate"]),
            }
        )
    return output


def _aggregate_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    error_key = f"{prefix}_error_reduction_px"
    utility_key = f"{prefix}_utility_gain"
    severe_key = f"{prefix}_severe_rate_reduction"
    return {
        "rows": len(rows),
        "mean_error_reduction_px": float(np.mean([float(row[error_key]) for row in rows])),
        "threshold_utility_gain": float(np.mean([float(row[utility_key]) for row in rows])),
        "severe_16px_rate_reduction": float(np.mean([float(row[severe_key]) for row in rows])),
        "positive_point_fraction": float(np.mean([float(row[error_key]) > 0.0 for row in rows])),
        "error_reduction_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key=error_key, seed=seed, samples=samples
        ),
        "utility_gain_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key=utility_key, seed=seed + 1, samples=samples
        ),
    }


def _memory_better_video_fraction(
    rows: Sequence[Mapping[str, Any]], prefix: str
) -> float:
    key = f"{prefix}_error_reduction_px"
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(float(row[key]))
    if not grouped:
        raise ValueError("no videos for memory comparison")
    return float(np.mean([np.mean(values) > 0.0 for values in grouped.values()]))


def gate_checks(
    *,
    commit: Mapping[str, float],
    full_vs_native: Mapping[str, Any],
    memory_vs_coordinate: Mapping[str, Any],
    memory_better_video_fraction: float,
    native_replay_max_abs_px: float,
    exact_replay: bool,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "shortlist_recall_within_12px": float(commit["recall_within_12px"])
        >= float(gates["shortlist_recall_within_12px_min"]),
        "shortlist_median_commit_error": float(commit["median_error_px"])
        <= float(gates["shortlist_median_commit_error_px_max"]),
        "full_state_error_reduction": float(full_vs_native["mean_error_reduction_px"])
        >= float(gates["full_state_mean_error_reduction_vs_native_px_min"]),
        "full_state_error_CI_lower": float(
            full_vs_native["error_reduction_video_cluster_CI"]["lower"]
        )
        >= float(gates["full_state_error_reduction_CI_lower_px_min"]),
        "full_state_utility_gain": float(full_vs_native["threshold_utility_gain"])
        >= float(gates["full_state_threshold_utility_gain_vs_native_min"]),
        "full_state_utility_CI_lower": float(
            full_vs_native["utility_gain_video_cluster_CI"]["lower"]
        )
        >= float(gates["full_state_utility_gain_CI_lower_min"]),
        "full_state_positive_fraction": float(full_vs_native["positive_point_fraction"])
        >= float(gates["full_state_positive_point_fraction_min"]),
        "full_state_severe_reduction": float(full_vs_native["severe_16px_rate_reduction"])
        >= float(gates["full_state_severe_16px_rate_reduction_min"]),
        "memory_incremental_error": float(memory_vs_coordinate["mean_error_reduction_px"])
        >= float(gates["memory_incremental_error_reduction_vs_coordinate_only_px_min"]),
        "memory_incremental_utility": float(memory_vs_coordinate["threshold_utility_gain"])
        >= float(gates["memory_incremental_utility_gain_vs_coordinate_only_min"]),
        "memory_better_video_fraction": float(memory_better_video_fraction)
        >= float(gates["memory_better_video_fraction_min"]),
        "native_replay_parity": float(native_replay_max_abs_px)
        <= float(gates["native_replay_max_abs_px"]),
        "exact_replay": bool(exact_replay),
    }


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    for path_key, sha_key in (
        ("gate3c1b_result", "gate3c1b_result_sha256"),
        ("gate3c1b_summary", "gate3c1b_summary_sha256"),
        ("audit_replay", "audit_replay_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[sha_key]:
            raise ValueError(f"Gate 3C1C parent hash drift: {path_key}")
    summary = json.loads(Path(parent["gate3c1b_summary"]).read_text())
    if summary.get("summary_payload_sha256") != parent[
        "gate3c1b_summary_payload_sha256"
    ]:
        raise ValueError("Gate 3C1B summary payload drift")
    if summary.get("status") != parent["required_gate3c1b_status"] or summary.get(
        "formal_decision"
    ) != parent["required_gate3c1b_decision"]:
        raise ValueError("Gate 3C1B did not authorize future rollout preregistration")
    replay = json.loads(Path(parent["audit_replay"]).read_text())
    if bool(replay.get("exact_replay")) is not bool(
        parent["required_audit_exact_replay"]
    ) or bool(replay.get("gate", {}).get("pass")) is not bool(
        parent["required_audit_gate_pass"]
    ):
        raise ValueError("Gate 3C1B audit replay did not authorize Gate 3C1C")
    authorization = config.get("model_validation_authorization")
    if authorization is not None:
        for path_key, sha_key in (
            ("gate3c1c_result", "gate3c1c_result_sha256"),
            ("gate3c1c_summary", "gate3c1c_summary_sha256"),
            ("gate3c1c_replay", "gate3c1c_replay_sha256"),
        ):
            if file_sha256(authorization[path_key]) != authorization[sha_key]:
                raise ValueError(f"Gate 3C1C model-validation authorization hash drift: {path_key}")
        gate3c1c_summary = json.loads(
            Path(authorization["gate3c1c_summary"]).read_text()
        )
        if gate3c1c_summary.get("summary_payload_sha256") != authorization[
            "gate3c1c_summary_payload_sha256"
        ]:
            raise ValueError("Gate 3C1C authorization summary payload drift")
        gate3c1c_replay = json.loads(
            Path(authorization["gate3c1c_replay"]).read_text()
        )
        if (
            not bool(gate3c1c_replay.get("exact_replay"))
            or not bool(gate3c1c_replay.get("gate", {}).get("pass"))
            or gate3c1c_replay.get("gate", {}).get("decision")
            != authorization["required_decision"]
        ):
            raise ValueError("Gate 3C1C did not authorize original model validation")


def _expected_read_state(config: Mapping[str, Any]) -> dict[str, bool]:
    partition = config["partition"]
    payload = partition.get("expected_read_state")
    if payload is None:
        payload = {
            "checkpoint_selection_read": True,
            "fit_only_internal_audit_read": True,
            "original_model_validation_read": False,
            "external_read": False,
        }
    required = (
        "checkpoint_selection_read",
        "fit_only_internal_audit_read",
        "original_model_validation_read",
        "external_read",
    )
    if set(payload) != set(required):
        raise ValueError("Gate 3C1C expected read-state keys drift")
    return {key: bool(payload[key]) for key in required}


def _load_candidate_index(config: Mapping[str, Any]) -> dict[str, Any]:
    partition = config["partition"]
    index_path = Path(partition["candidate_cache_index"])
    expected_index_sha = partition.get("candidate_cache_index_sha256")
    if expected_index_sha is not None and file_sha256(index_path) != expected_index_sha:
        raise ValueError("Gate 3C1C candidate index file hash drift")
    index = json.loads(index_path.read_text())
    expected_payload_sha = partition.get("candidate_cache_index_payload_sha256")
    if expected_payload_sha is not None and index.get("index_payload_sha256") != expected_payload_sha:
        raise ValueError("Gate 3C1C candidate index payload drift")
    cache_config = partition.get("candidate_cache_config")
    if cache_config is not None:
        cache_config_path = Path(cache_config)
        if file_sha256(cache_config_path) != partition["candidate_cache_config_sha256"]:
            raise ValueError("Gate 3C1C candidate cache config hash drift")
        if index.get("config_sha256") != partition["candidate_cache_config_sha256"]:
            raise ValueError("Gate 3C1C candidate index/config hash mismatch")
    expected = list(
        range(
            int(partition["source_indices"][0]),
            int(partition["source_indices"][1]) + 1,
        )
    )
    if index.get("partition") != partition["name"]:
        raise ValueError("Gate 3C1C candidate partition drift")
    if index.get("completed_source_indices") != expected:
        raise ValueError("Gate 3C1C candidate membership drift")
    if int(index.get("videos", -1)) != int(partition["expected_videos"]):
        raise ValueError("Gate 3C1C candidate video-count drift")
    expected_rows = partition.get("expected_failure_rows")
    if expected_rows is not None and int(index.get("failure_rows", -1)) != int(expected_rows):
        raise ValueError("Gate 3C1C candidate failure-row drift")
    expected_state = _expected_read_state(config)
    actual_state = {
        key: bool(index.get("read_state", {}).get(key)) for key in expected_state
    }
    if actual_state != expected_state:
        raise ValueError("Gate 3C1C candidate read-state drift")
    if actual_state["external_read"]:
        raise ValueError("Gate 3C1C candidate index reports external contamination")
    index["_index_file_sha256"] = file_sha256(index_path)
    return index


def evaluate(
    *,
    config_path: Path,
    output_path: Path,
    reference_path: Path | None,
    device: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1C config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1C locked-data flags must be false")
    _validate_parent(config)
    if file_sha256(config["partition"]["manifest"]) != config["partition"][
        "manifest_sha256"
    ]:
        raise ValueError("Gate 3C1C manifest hash drift")
    if file_sha256(config["selector"]["config"]) != config["selector"][
        "config_sha256"
    ]:
        raise ValueError("Gate 3C1C selector config hash drift")
    if file_sha256(config["backbone"]["parent_config"]) != config["backbone"][
        "parent_config_sha256"
    ]:
        raise ValueError("Gate 3C1C backbone parent hash drift")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"][
        "checkpoint_sha256"
    ]:
        raise ValueError("Gate 3C1C backbone checkpoint hash drift")

    candidate_index = _load_candidate_index(config)
    selector_payload = yaml.safe_load(Path(config["selector"]["config"]).read_text())
    selector_config = _selector_config(selector_payload)
    seed = int(config["metrics"]["bootstrap_seed"])
    samples = int(config["metrics"]["bootstrap_samples"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=config["backbone"]["checkpoint"]
    ).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)

    action_rows: dict[str, list[dict[str, float]]] = {
        "native": [],
        "coordinate_only": [],
        "coordinate_plus_memory": [],
    }
    full_vs_native_rows: list[dict[str, Any]] = []
    coordinate_vs_native_rows: list[dict[str, Any]] = []
    memory_vs_coordinate_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    selected_index_hashes: list[str] = []
    teacher_selected_hashes: list[str] = []
    future_coordinate_hashes: dict[str, list[str]] = {
        name: [] for name in action_rows
    }
    native_replay_differences: list[float] = []

    with torch.no_grad():
        for index_row in candidate_index["rows"]:
            source_index = int(index_row["source_index"])
            sidecar_path = Path(index_row["sidecar"])
            if file_sha256(sidecar_path) != index_row["sidecar_sha256"]:
                raise ValueError("Gate 3C1C candidate sidecar hash drift")
            payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            verify_temporal_identity_cache_payload(
                payload,
                expected_partition=config["partition"]["name"],
                expected_source_index=source_index,
                expected_config_sha256=candidate_index["config_sha256"],
                expected_read_state=_expected_read_state(config),
            )
            tensors = payload["tensors"]
            point_indices = tensors["point_indices"].long()
            rows = int(point_indices.numel())
            if rows == 0:
                video_records.append(
                    {
                        "source_index": source_index,
                        "video_name": str(payload["video_name"]),
                        "failure_rows": 0,
                    }
                )
                continue

            causal = freeze_shortlist(tensors, selector_config)
            selected_indices = causal["selected_indices"].cpu().contiguous()
            selected_index_hashes.append(tensor_sha256(selected_indices))
            # Teacher labels are first accessed after the selected-index tensor is frozen.
            oracle = teacher_nearest_in_frozen_shortlist(
                selected_indices=selected_indices,
                candidate_coordinates_xy=tensors["candidate_coordinates_xy"].float(),
                teacher_candidate_distance_px=tensors[
                    "teacher_candidate_distance_px"
                ].float(),
            )
            selected_candidate_index = oracle["selected_candidate_index"].cpu().contiguous()
            selected_xy = oracle["selected_coordinates_xy"].float().cpu().contiguous()
            selected_error = oracle["selected_teacher_distance_px"].float().cpu().contiguous()
            teacher_selected_hashes.append(
                canonical_json_sha256(
                    {
                        "candidate_index": tensor_sha256(selected_candidate_index),
                        "coordinates": tensor_sha256(selected_xy),
                        "distance": tensor_sha256(selected_error),
                    }
                )
            )

            sample, metadata = load_manifest_sample(
                Path(config["partition"]["manifest"]), source_index
            )
            prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
            if str(prepared["video_name"]) != str(payload["video_name"]):
                raise ValueError("Gate 3C1C video identity drift")
            video = prepared["video"].to(device)
            set_deterministic(seed + source_index)
            initial = _initialize_and_first_window(
                predictor, video, _original_queries(prepared, video)
            )
            native_frame15 = _coords_to_input(
                initial,
                interp_height=int(predictor.interp_shape[0]),
                interp_width=int(predictor.interp_shape[1]),
            )[point_indices, 15]
            candidate_zero = tensors["candidate_coordinates_xy"][:, 0].float()
            native_commit_parity = float((native_frame15 - candidate_zero).abs().max())
            if native_commit_parity > 1.0e-4:
                raise ValueError("Gate 3C1C native/candidate-zero coordinate drift")

            observed = extract_cotracker_observed_feature_pyramid(
                predictor.model, video[:, :16]
            )
            frame15_pyramid = [value[:, 15:16].contiguous() for value in observed]
            native_final = _continue_second_window(predictor, video, initial)

            coordinate_initial = apply_reextracted_state_action(
                initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=selected_xy.to(device),
                apply_mask=torch.ones(rows, dtype=torch.bool),
                reextracted_track_features=(),
                reextracted_track_supports=(),
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                write_probability=False,
                write_memory=False,
            )
            coordinate_final = _continue_second_window(
                predictor, video, coordinate_initial
            )

            re_feat_batch, re_support_batch = deterministic_reextract_cotracker_memory(
                frame15_pyramid,
                selected_xy.to(device),
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                stride=int(config["state_actions"]["coordinate_plus_memory"]["model_stride"]),
                support_radius=int(
                    config["state_actions"]["coordinate_plus_memory"]["support_radius"]
                ),
            )
            re_feat, re_support = _native_format_memory(
                re_feat_batch, re_support_batch
            )
            full_initial = apply_reextracted_state_action(
                initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=selected_xy.to(device),
                apply_mask=torch.ones(rows, dtype=torch.bool),
                reextracted_track_features=re_feat,
                reextracted_track_supports=re_support,
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                write_probability=False,
                write_memory=True,
            )
            full_final = _continue_second_window(predictor, video, full_initial)

            gt_future = (
                prepared["gt_tracks_yx"][point_indices, 16:24][..., [1, 0]].float()
                * 255.0
            )
            visible_future = ~prepared["gt_occluded"][point_indices, 16:24]
            future_coordinates = {
                "native": _future_coordinates(native_final, point_indices, predictor),
                "coordinate_only": _future_coordinates(
                    coordinate_final, point_indices, predictor
                ),
                "coordinate_plus_memory": _future_coordinates(
                    full_final, point_indices, predictor
                ),
            }
            local_rows = {
                name: point_future_rows(coords, gt_future, visible_future)
                for name, coords in future_coordinates.items()
            }
            for name in action_rows:
                action_rows[name].extend(local_rows[name])
                future_coordinate_hashes[name].append(
                    tensor_sha256(future_coordinates[name].contiguous())
                )

            cached_native = tensors["native_future_mean_error_px"].float()
            recomputed_native = torch.tensor(
                [row["mean_l2_error_px"] for row in local_rows["native"]]
            )
            native_difference = (cached_native - recomputed_native).abs()
            native_replay_differences.extend(native_difference.tolist())

            local_full_vs_native = _comparison_rows(
                local_rows["native"],
                local_rows["coordinate_plus_memory"],
                source_index=source_index,
                point_indices=point_indices,
                prefix="full_vs_native",
            )
            local_coordinate_vs_native = _comparison_rows(
                local_rows["native"],
                local_rows["coordinate_only"],
                source_index=source_index,
                point_indices=point_indices,
                prefix="coordinate_vs_native",
            )
            local_memory_vs_coordinate = _comparison_rows(
                local_rows["coordinate_only"],
                local_rows["coordinate_plus_memory"],
                source_index=source_index,
                point_indices=point_indices,
                prefix="memory_vs_coordinate",
            )
            full_vs_native_rows.extend(local_full_vs_native)
            coordinate_vs_native_rows.extend(local_coordinate_vs_native)
            memory_vs_coordinate_rows.extend(local_memory_vs_coordinate)

            for local in range(rows):
                commit = {
                    "source_index": source_index,
                    "point_index": int(point_indices[local]),
                    "selected_candidate_index": int(selected_candidate_index[local]),
                    "commit_error_px": float(selected_error[local]),
                    "within_4px": float(selected_error[local] <= 4.0),
                    "within_8px": float(selected_error[local] <= 8.0),
                    "within_12px": float(selected_error[local] <= 12.0),
                }
                commit_rows.append(commit)
                point_records.append(
                    {
                        **commit,
                        "native": local_rows["native"][local],
                        "coordinate_only": local_rows["coordinate_only"][local],
                        "coordinate_plus_memory": local_rows[
                            "coordinate_plus_memory"
                        ][local],
                        **local_full_vs_native[local],
                        **local_coordinate_vs_native[local],
                        **local_memory_vs_coordinate[local],
                        "native_cached_replay_abs_px": float(native_difference[local]),
                    }
                )
            video_records.append(
                {
                    "source_index": source_index,
                    "video_name": str(payload["video_name"]),
                    "failure_rows": rows,
                    "native_candidate_zero_max_abs_px": native_commit_parity,
                    "commit_mean_error_px": float(selected_error.mean()),
                    "commit_recall_within_12px": float(
                        (selected_error <= 12.0).float().mean()
                    ),
                    "actions": {
                        name: aggregate_point_rows(local_rows[name])
                        for name in action_rows
                    },
                    "sample_metadata": metadata,
                }
            )
            del video
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    expected_rows = config["partition"].get("expected_failure_rows")
    if expected_rows is None:
        expected_rows = int(candidate_index["failure_rows"])
    if len(commit_rows) != int(expected_rows):
        raise ValueError("Gate 3C1C completed failure-row drift")
    commit_error = [float(row["commit_error_px"]) for row in commit_rows]
    commit = {
        "rows": len(commit_rows),
        "mean_error_px": float(np.mean(commit_error)),
        "median_error_px": float(np.median(commit_error)),
        "recall_within_4px": float(np.mean([row["within_4px"] for row in commit_rows])),
        "recall_within_8px": float(np.mean([row["within_8px"] for row in commit_rows])),
        "recall_within_12px": float(np.mean([row["within_12px"] for row in commit_rows])),
    }
    action_summary = {
        name: aggregate_point_rows(rows) for name, rows in action_rows.items()
    }
    comparisons = {
        "full_vs_native": _aggregate_comparison(
            full_vs_native_rows,
            prefix="full_vs_native",
            seed=seed,
            samples=samples,
        ),
        "coordinate_vs_native": _aggregate_comparison(
            coordinate_vs_native_rows,
            prefix="coordinate_vs_native",
            seed=seed + 10,
            samples=samples,
        ),
        "memory_vs_coordinate": _aggregate_comparison(
            memory_vs_coordinate_rows,
            prefix="memory_vs_coordinate",
            seed=seed + 20,
            samples=samples,
        ),
    }
    memory_better = _memory_better_video_fraction(
        memory_vs_coordinate_rows, "memory_vs_coordinate"
    )
    native_replay_max_abs = float(max(native_replay_differences, default=0.0))
    scientific = {
        "commit": commit,
        "actions": action_summary,
        "comparisons": comparisons,
        "memory_better_video_fraction": memory_better,
        "native_replay_max_abs_px": native_replay_max_abs,
        "selected_indices_digest": canonical_json_sha256(selected_index_hashes),
        "teacher_selected_digest": canonical_json_sha256(teacher_selected_hashes),
        "future_coordinate_digest": {
            name: canonical_json_sha256(values)
            for name, values in future_coordinate_hashes.items()
        },
        "point_records_digest": canonical_json_sha256(point_records),
        "video_records_digest": canonical_json_sha256(video_records),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)

    reference = None
    comparison = None
    exact_replay = False
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        reference_scientific = reference["scientific"]
        comparison = {
            "selected_indices_digest": scientific["selected_indices_digest"]
            == reference_scientific["selected_indices_digest"],
            "teacher_selected_digest": scientific["teacher_selected_digest"]
            == reference_scientific["teacher_selected_digest"],
            "future_coordinate_digest": scientific["future_coordinate_digest"]
            == reference_scientific["future_coordinate_digest"],
            "point_records_digest": scientific["point_records_digest"]
            == reference_scientific["point_records_digest"],
            "video_records_digest": scientific["video_records_digest"]
            == reference_scientific["video_records_digest"],
            "scientific_payload_sha256": scientific["scientific_payload_sha256"]
            == reference_scientific["scientific_payload_sha256"],
        }
        exact_replay = all(comparison.values())

    checks = gate_checks(
        commit=commit,
        full_vs_native=comparisons["full_vs_native"],
        memory_vs_coordinate=comparisons["memory_vs_coordinate"],
        memory_better_video_fraction=memory_better,
        native_replay_max_abs_px=native_replay_max_abs,
        exact_replay=exact_replay,
        gates=config["pass_gates"],
    )
    if reference_path is None:
        passed = False
        decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    else:
        passed = all(checks.values())
        decision = config["pass_gates"][
            "decision_pass" if passed else "decision_fail"
        ]
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "partition": config["partition"]["name"],
        "source_indices": config["partition"]["source_indices"],
        "videos": int(config["partition"]["expected_videos"]),
        "failure_rows": len(commit_rows),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "candidate_cache_index": config["partition"]["candidate_cache_index"],
        "candidate_cache_index_sha256": candidate_index["_index_file_sha256"],
        "scientific": scientific,
        "point_records": point_records,
        "video_records": video_records,
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": comparison,
        "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision},
        "read_state": _expected_read_state(config),
        "claim_boundary": config["claim_scope"],
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    result = evaluate(
        config_path=Path(args.config).resolve(),
        output_path=Path(args.output).resolve(),
        reference_path=(
            None if args.reference is None else Path(args.reference).resolve()
        ),
        device=str(args.device),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scientific": result["scientific"],
                "gate": result["gate"],
                "exact_replay": result["exact_replay"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
