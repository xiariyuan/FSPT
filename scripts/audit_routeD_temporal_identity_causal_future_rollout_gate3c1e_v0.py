#!/usr/bin/env python3
"""Deployable causal top-1 future rollout for Route-D Gate 3C1E v0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
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
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1 import (
    Top1FeatureConfig,
    build_shortlist_top1_features,
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
from scripts.run_routeD_temporal_identity_top1_gate3c1d_v1 import (
    _predict_bundle,
    _selector_config,
)

SCHEMA = "routeD_temporal_identity_causal_future_rollout_gate3c1e_v0"
RESULT_SCHEMA = "routeD_temporal_identity_causal_future_rollout_result_gate3c1e_v0"
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/routeD_temporal_identity_causal_future_rollout_gate3c1e_v0.yaml"
)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _result_payload_sha256(result: Mapping[str, Any]) -> str:
    value = dict(result)
    value.pop("result_payload_sha256", None)
    return canonical_json_sha256(value)


def _feature_config(config: Mapping[str, Any]) -> Top1FeatureConfig:
    payload = config["feature_contract"]
    return Top1FeatureConfig(
        shortlist_size=int(payload["shortlist_size"]),
        temporal_summary_channels=tuple(
            int(value) for value in payload["temporal_summary_channels"]
        ),
        zscore_epsilon=float(payload["candidate_relative_zscore_epsilon"]),
    )


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


def freeze_deployable_policy_action(
    *,
    tensors: Mapping[str, torch.Tensor],
    bundle: Mapping[str, Any],
    policy: Mapping[str, float],
    feature_config: Top1FeatureConfig,
) -> dict[str, Any]:
    """Freeze all deployable decisions without reading teacher or future tensors."""
    causal = build_shortlist_top1_features(
        temporal_features=tensors["temporal_features"].float(),
        static_features=tensors["static_features"].float(),
        query_frames=tensors["query_frames"].long(),
        valid_mask=tensors["candidate_valid_mask"].bool(),
        selector_config=_selector_config(),
        feature_config=feature_config,
    )
    features = causal["candidate_features"].cpu().contiguous()
    shortlist_indices = causal["selected_indices"].cpu().contiguous()
    predictions = _predict_bundle(bundle, {"features": features})
    selected_slot = predictions["selected"].astype(np.int64)
    row = np.arange(len(selected_slot))
    selected_support = predictions["support"][row, selected_slot]
    action = (
        (selected_slot != 0)
        & (selected_support >= float(policy["support_min"]))
        & (predictions["value"] >= float(policy["value_min_px"]))
        & (predictions["harm"] <= float(policy["harm_max"]))
    )
    output_slot = np.where(action, selected_slot, 0).astype(np.int64)
    shortlist_np = shortlist_indices.numpy()
    selected_candidate_index = shortlist_np[row, selected_slot]
    output_candidate_index = shortlist_np[row, output_slot]
    coordinates = tensors["candidate_coordinates_xy"].float().cpu()
    output_xy = coordinates[torch.arange(len(output_slot)), torch.from_numpy(output_candidate_index)]
    return {
        "candidate_features": features,
        "shortlist_indices": shortlist_indices,
        "predictions": predictions,
        "selected_slot": selected_slot,
        "selected_candidate_index": selected_candidate_index.astype(np.int64),
        "selected_support": selected_support.astype(np.float64),
        "action": action.astype(bool),
        "output_slot": output_slot,
        "output_candidate_index": output_candidate_index.astype(np.int64),
        "output_coordinates_xy": output_xy.contiguous(),
    }


def _causal_records(
    *,
    source_index: int,
    video_name: str,
    point_indices: torch.Tensor,
    action: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    predictions = action["predictions"]
    for local, point_index in enumerate(point_indices.tolist()):
        records.append(
            {
                "source_index": int(source_index),
                "video_name": str(video_name),
                "point_index": int(point_index),
                "selected_shortlist_slot": int(action["selected_slot"][local]),
                "selected_candidate_index": int(action["selected_candidate_index"][local]),
                "selected_support_probability": float(action["selected_support"][local]),
                "predicted_value_px": float(predictions["value"][local]),
                "predicted_harm_probability": float(predictions["harm"][local]),
                "action": bool(action["action"][local]),
                "output_candidate_index": int(action["output_candidate_index"][local]),
            }
        )
    return records


def _parent_causal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source_index",
        "video_name",
        "point_index",
        "selected_shortlist_slot",
        "selected_candidate_index",
        "selected_support_probability",
        "predicted_value_px",
        "predicted_harm_probability",
        "action",
        "output_candidate_index",
    )
    return {key: record[key] for key in keys}


def _comparison_rows(
    left_rows: Sequence[Mapping[str, float]],
    right_rows: Sequence[Mapping[str, float]],
    *,
    source_index: int,
    point_indices: torch.Tensor,
    action_mask: np.ndarray,
    prefix: str,
) -> list[dict[str, Any]]:
    if (
        len(left_rows) != len(right_rows)
        or len(left_rows) != int(point_indices.numel())
        or action_mask.shape != (len(left_rows),)
    ):
        raise ValueError("Gate 3C1E comparison row count mismatch")
    output = []
    for local, (left, right) in enumerate(zip(left_rows, right_rows)):
        output.append(
            {
                "source_index": int(source_index),
                "point_index": int(point_indices[local]),
                "action": bool(action_mask[local]),
                f"{prefix}_error_reduction_px": float(left["mean_l2_error_px"])
                - float(right["mean_l2_error_px"]),
                f"{prefix}_utility_gain": float(right["threshold_utility"])
                - float(left["threshold_utility"]),
                f"{prefix}_severe_rate_reduction": float(left["severe_16px_rate"])
                - float(right["severe_16px_rate"]),
                f"{prefix}_harmful_gt_4px": bool(
                    float(right["mean_l2_error_px"])
                    > float(left["mean_l2_error_px"]) + 4.0
                ),
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
    if not rows:
        raise ValueError("Gate 3C1E cannot aggregate an empty comparison")
    error_key = f"{prefix}_error_reduction_px"
    utility_key = f"{prefix}_utility_gain"
    severe_key = f"{prefix}_severe_rate_reduction"
    harm_key = f"{prefix}_harmful_gt_4px"
    return {
        "rows": len(rows),
        "mean_error_reduction_px": float(np.mean([float(row[error_key]) for row in rows])),
        "threshold_utility_gain": float(np.mean([float(row[utility_key]) for row in rows])),
        "severe_16px_rate_reduction": float(np.mean([float(row[severe_key]) for row in rows])),
        "positive_point_fraction": float(np.mean([float(row[error_key]) > 0.0 for row in rows])),
        "nonnegative_point_fraction": float(np.mean([float(row[error_key]) >= 0.0 for row in rows])),
        "harmful_gt_4px_fraction": float(np.mean([bool(row[harm_key]) for row in rows])),
        "error_reduction_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key=error_key, seed=seed, samples=samples
        ),
        "utility_gain_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key=utility_key, seed=seed + 1, samples=samples
        ),
    }


def _video_nonnegative_fraction(rows: Sequence[Mapping[str, Any]], prefix: str) -> float:
    key = f"{prefix}_error_reduction_px"
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(float(row[key]))
    if not grouped:
        raise ValueError("Gate 3C1E has no videos for comparison")
    return float(np.mean([np.mean(values) >= 0.0 for values in grouped.values()]))


def _validate_parent(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = config["authorized_parent"]
    for path_key, sha_key in (
        ("result", "result_sha256"),
        ("model_validation_replay", "model_validation_replay_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[sha_key]:
            raise ValueError(f"Gate 3C1E parent hash drift: {path_key}")
    replay = json.loads(Path(parent["model_validation_replay"]).read_text())
    if replay.get("result_payload_sha256") != _result_payload_sha256(replay):
        raise ValueError("Gate 3C1E parent replay payload drift")
    if (
        not bool(replay.get("exact_replay"))
        or not bool(replay.get("gate", {}).get("pass"))
        or replay.get("gate", {}).get("decision") != parent["required_decision"]
        or replay.get("partition") != parent["required_partition"]
    ):
        raise ValueError("Gate 3C1D v1 did not authorize Gate 3C1E")
    bundle = replay["frozen_bundle_artifact"]
    if (
        bundle["path"] != parent["bundle_path"]
        or bundle["file_sha256"] != parent["bundle_sha256"]
        or file_sha256(bundle["path"]) != parent["bundle_sha256"]
    ):
        raise ValueError("Gate 3C1E frozen bundle drift")
    if replay["frozen_policy"] != parent["frozen_policy"]:
        raise ValueError("Gate 3C1E frozen policy drift")
    return replay, joblib.load(bundle["path"])


def _validate_implementation(config: Mapping[str, Any]) -> None:
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1E implementation hash drift: {name}")


def _load_candidate_index(config: Mapping[str, Any]) -> dict[str, Any]:
    partition = config["partition"]
    path = Path(partition["candidate_cache_index"])
    if file_sha256(path) != partition["candidate_cache_index_sha256"]:
        raise ValueError("Gate 3C1E candidate index file drift")
    index = json.loads(path.read_text())
    without_hash = dict(index)
    payload_hash = without_hash.pop("index_payload_sha256", None)
    expected = list(
        range(int(partition["source_indices"][0]), int(partition["source_indices"][1]) + 1)
    )
    if (
        payload_hash != partition["candidate_cache_index_payload_sha256"]
        or payload_hash != canonical_json_sha256(without_hash)
        or not bool(index.get("pass"))
        or index.get("partition") != partition["name"]
        or index.get("completed_source_indices") != expected
        or int(index.get("videos", -1)) != int(partition["expected_videos"])
        or int(index.get("failure_rows", -1)) != int(partition["expected_failure_rows"])
        or index.get("read_state") != partition["expected_read_state"]
        or not all(index.get("checks", {}).values())
    ):
        raise ValueError("Gate 3C1E candidate index drift")
    index["_index_file_sha256"] = file_sha256(path)
    return index


def gate_checks(
    *,
    selector_consistency_exact: bool,
    commit: Mapping[str, Any],
    full_vs_native: Mapping[str, Any],
    full_vs_native_action: Mapping[str, Any],
    memory_vs_coordinate: Mapping[str, Any],
    video_nonnegative_fraction: float,
    memory_better_video_fraction: float,
    native_replay_max_abs_px: float,
    exact_replay: bool,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "selector_consistency_exact": bool(selector_consistency_exact),
        "commit_error_reduction": float(commit["mean_error_reduction_px"])
        >= float(gates["commit_mean_error_reduction_px_min"]),
        "full_future_error_reduction": float(full_vs_native["mean_error_reduction_px"])
        >= float(gates["full_future_mean_error_reduction_px_min"]),
        "full_future_error_CI_lower": float(
            full_vs_native["error_reduction_video_cluster_CI"]["lower"]
        )
        >= float(gates["full_future_error_reduction_CI_lower_px_min"]),
        "full_future_utility_gain": float(full_vs_native["threshold_utility_gain"])
        >= float(gates["full_future_threshold_utility_gain_min"]),
        "full_future_utility_CI_lower": float(
            full_vs_native["utility_gain_video_cluster_CI"]["lower"]
        )
        >= float(gates["full_future_utility_gain_CI_lower_min"]),
        "action_future_error_reduction": float(
            full_vs_native_action["mean_error_reduction_px"]
        )
        >= float(gates["action_future_mean_error_reduction_px_min"]),
        "action_positive_fraction": float(full_vs_native_action["positive_point_fraction"])
        >= float(gates["action_future_positive_point_fraction_min"]),
        "full_future_severe_reduction": float(full_vs_native["severe_16px_rate_reduction"])
        >= float(gates["full_future_severe_16px_rate_reduction_min"]),
        "full_future_harm_all": float(full_vs_native["harmful_gt_4px_fraction"])
        <= float(gates["full_future_harmful_all_rows_max"]),
        "full_future_harm_action": float(full_vs_native_action["harmful_gt_4px_fraction"])
        <= float(gates["full_future_harmful_action_rows_max"]),
        "video_nonnegative_fraction": float(video_nonnegative_fraction)
        >= float(gates["full_future_nonnegative_video_fraction_min"]),
        "memory_incremental_error": float(memory_vs_coordinate["mean_error_reduction_px"])
        >= float(gates["memory_incremental_error_reduction_px_min"]),
        "memory_incremental_error_CI_lower": float(
            memory_vs_coordinate["error_reduction_video_cluster_CI"]["lower"]
        )
        >= float(gates["memory_incremental_error_reduction_CI_lower_px_min"]),
        "memory_incremental_utility": float(memory_vs_coordinate["threshold_utility_gain"])
        >= float(gates["memory_incremental_utility_gain_min"]),
        "memory_better_video_fraction": float(memory_better_video_fraction)
        >= float(gates["memory_better_video_fraction_min"]),
        "native_replay_parity": float(native_replay_max_abs_px)
        <= float(gates["native_replay_max_abs_px"]),
        "exact_replay": bool(exact_replay),
    }


def evaluate(
    *,
    config_path: Path,
    output_path: Path,
    reference_path: Path | None,
    device: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1E config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1E locked data flags must remain false")
    _validate_implementation(config)
    parent_replay, bundle = _validate_parent(config)
    if file_sha256(config["partition"]["manifest"]) != config["partition"]["manifest_sha256"]:
        raise ValueError("Gate 3C1E manifest drift")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("Gate 3C1E backbone checkpoint drift")
    candidate_index = _load_candidate_index(config)
    feature_config = _feature_config(config)
    policy = config["authorized_parent"]["frozen_policy"]
    parent_records = {
        (int(record["source_index"]), int(record["point_index"])): _parent_causal_record(record)
        for record in parent_replay["records"]
    }
    if len(parent_records) != int(config["partition"]["expected_failure_rows"]):
        raise ValueError("Gate 3C1E parent record membership drift")

    # Complete causal-selector preflight before CoTracker initialization or rollout.
    preflight_causal_records: list[dict[str, Any]] = []
    for index_row in candidate_index["rows"]:
        source_index = int(index_row["source_index"])
        sidecar_path = Path(index_row["sidecar"])
        if file_sha256(sidecar_path) != index_row["sidecar_sha256"]:
            raise ValueError("Gate 3C1E preflight sidecar hash drift")
        payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        verify_temporal_identity_cache_payload(
            payload,
            expected_partition=config["partition"]["name"],
            expected_source_index=source_index,
            expected_config_sha256=candidate_index["config_sha256"],
            expected_read_state=config["partition"]["expected_read_state"],
        )
        tensors = payload["tensors"]
        point_indices = tensors["point_indices"].long()
        if int(point_indices.numel()) == 0:
            continue
        frozen = freeze_deployable_policy_action(
            tensors=tensors,
            bundle=bundle,
            policy=policy,
            feature_config=feature_config,
        )
        local_causal = _causal_records(
            source_index=source_index,
            video_name=str(payload["video_name"]),
            point_indices=point_indices,
            action=frozen,
        )
        local_parent = [parent_records[(source_index, int(point))] for point in point_indices]
        if canonical_json_sha256(local_causal) != canonical_json_sha256(local_parent):
            raise ValueError("Gate 3C1E causal selector preflight drift")
        preflight_causal_records.extend(local_causal)
    parent_causal_records = [_parent_causal_record(record) for record in parent_replay["records"]]
    preflight_causal_digest = canonical_json_sha256(preflight_causal_records)
    if (
        len(preflight_causal_records) != int(config["partition"]["expected_failure_rows"])
        or preflight_causal_digest != canonical_json_sha256(parent_causal_records)
    ):
        raise ValueError("Gate 3C1E complete causal preflight drift")

    seed = int(config["metrics"]["bootstrap_seed"])
    samples = int(config["metrics"]["bootstrap_samples"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(checkpoint=config["backbone"]["checkpoint"]).to(device).eval()
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
    commit_records: list[dict[str, Any]] = []
    causal_records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    shortlist_hashes: list[str] = []
    action_hashes: list[str] = []
    output_candidate_hashes: list[str] = []
    future_coordinate_hashes: dict[str, list[str]] = {name: [] for name in action_rows}
    native_replay_differences: list[float] = []

    with torch.no_grad():
        for index_row in candidate_index["rows"]:
            source_index = int(index_row["source_index"])
            sidecar_path = Path(index_row["sidecar"])
            if file_sha256(sidecar_path) != index_row["sidecar_sha256"]:
                raise ValueError("Gate 3C1E sidecar hash drift")
            payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            verify_temporal_identity_cache_payload(
                payload,
                expected_partition=config["partition"]["name"],
                expected_source_index=source_index,
                expected_config_sha256=candidate_index["config_sha256"],
                expected_read_state=config["partition"]["expected_read_state"],
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
                        "action_rows": 0,
                    }
                )
                continue

            frozen = freeze_deployable_policy_action(
                tensors=tensors,
                bundle=bundle,
                policy=policy,
                feature_config=feature_config,
            )
            local_causal = _causal_records(
                source_index=source_index,
                video_name=str(payload["video_name"]),
                point_indices=point_indices,
                action=frozen,
            )
            local_parent = [parent_records[(source_index, int(point))] for point in point_indices]
            if canonical_json_sha256(local_causal) != canonical_json_sha256(local_parent):
                raise ValueError("Gate 3C1E causal selector output drift before rollout")
            causal_records.extend(local_causal)
            shortlist_hashes.append(tensor_sha256(frozen["shortlist_indices"]))
            action_hashes.append(tensor_sha256(torch.from_numpy(frozen["action"])))
            output_candidate_hashes.append(
                tensor_sha256(torch.from_numpy(frozen["output_candidate_index"]))
            )

            # Teacher labels become accessible only after deployable action identity is frozen.
            output_candidate_tensor = torch.from_numpy(frozen["output_candidate_index"]).long()
            row_tensor = torch.arange(rows)
            output_commit_error = tensors["teacher_candidate_distance_px"].float()[
                row_tensor, output_candidate_tensor
            ]
            native_commit_error = tensors["teacher_candidate_distance_px"].float()[:, 0]
            action_mask = frozen["action"]
            for local, point_index in enumerate(point_indices.tolist()):
                commit_records.append(
                    {
                        "source_index": source_index,
                        "point_index": int(point_index),
                        "action": bool(action_mask[local]),
                        "native_commit_error_px": float(native_commit_error[local]),
                        "policy_commit_error_px": float(output_commit_error[local]),
                        "commit_error_reduction_px": float(
                            native_commit_error[local] - output_commit_error[local]
                        ),
                        "policy_within_12px": bool(output_commit_error[local] <= 12.0),
                    }
                )

            sample, metadata = load_manifest_sample(
                Path(config["partition"]["manifest"]), source_index
            )
            prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
            if str(prepared["video_name"]) != str(payload["video_name"]):
                raise ValueError("Gate 3C1E video identity drift")
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
                raise ValueError("Gate 3C1E native/candidate-zero coordinate drift")

            observed = extract_cotracker_observed_feature_pyramid(predictor.model, video[:, :16])
            frame15_pyramid = [value[:, 15:16].contiguous() for value in observed]
            native_final = _continue_second_window(predictor, video, initial)
            output_xy = frozen["output_coordinates_xy"].to(device)
            apply_mask = torch.from_numpy(action_mask).to(device=device, dtype=torch.bool)

            coordinate_initial = apply_reextracted_state_action(
                initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=output_xy,
                apply_mask=apply_mask,
                reextracted_track_features=(),
                reextracted_track_supports=(),
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                write_probability=False,
                write_memory=False,
            )
            coordinate_final = _continue_second_window(predictor, video, coordinate_initial)

            re_feat_batch, re_support_batch = deterministic_reextract_cotracker_memory(
                frame15_pyramid,
                output_xy,
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
                stride=int(config["state_actions"]["coordinate_plus_memory"]["model_stride"]),
                support_radius=int(
                    config["state_actions"]["coordinate_plus_memory"]["support_radius"]
                ),
            )
            re_feat, re_support = _native_format_memory(re_feat_batch, re_support_batch)
            full_initial = apply_reextracted_state_action(
                initial,
                point_indices=point_indices,
                predicted_coordinates_input_xy=output_xy,
                apply_mask=apply_mask,
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
                prepared["gt_tracks_yx"][point_indices, 16:24][..., [1, 0]].float() * 255.0
            )
            visible_future = ~prepared["gt_occluded"][point_indices, 16:24]
            future_coordinates = {
                "native": _future_coordinates(native_final, point_indices, predictor),
                "coordinate_only": _future_coordinates(coordinate_final, point_indices, predictor),
                "coordinate_plus_memory": _future_coordinates(full_final, point_indices, predictor),
            }
            local_rows = {
                name: point_future_rows(coordinates, gt_future, visible_future)
                for name, coordinates in future_coordinates.items()
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

            local_full = _comparison_rows(
                local_rows["native"],
                local_rows["coordinate_plus_memory"],
                source_index=source_index,
                point_indices=point_indices,
                action_mask=action_mask,
                prefix="full_vs_native",
            )
            local_coordinate = _comparison_rows(
                local_rows["native"],
                local_rows["coordinate_only"],
                source_index=source_index,
                point_indices=point_indices,
                action_mask=action_mask,
                prefix="coordinate_vs_native",
            )
            local_memory = _comparison_rows(
                local_rows["coordinate_only"],
                local_rows["coordinate_plus_memory"],
                source_index=source_index,
                point_indices=point_indices,
                action_mask=action_mask,
                prefix="memory_vs_coordinate",
            )
            full_vs_native_rows.extend(local_full)
            coordinate_vs_native_rows.extend(local_coordinate)
            memory_vs_coordinate_rows.extend(local_memory)

            for local, point_index in enumerate(point_indices.tolist()):
                point_records.append(
                    {
                        **local_causal[local],
                        "native_commit_error_px": float(native_commit_error[local]),
                        "policy_commit_error_px": float(output_commit_error[local]),
                        "native": local_rows["native"][local],
                        "coordinate_only": local_rows["coordinate_only"][local],
                        "coordinate_plus_memory": local_rows["coordinate_plus_memory"][local],
                        **local_full[local],
                        **local_coordinate[local],
                        **local_memory[local],
                        "native_cached_replay_abs_px": float(native_difference[local]),
                    }
                )
            video_records.append(
                {
                    "source_index": source_index,
                    "video_name": str(payload["video_name"]),
                    "failure_rows": rows,
                    "action_rows": int(action_mask.sum()),
                    "native_candidate_zero_max_abs_px": native_commit_parity,
                    "commit_native_mean_error_px": float(native_commit_error.mean()),
                    "commit_policy_mean_error_px": float(output_commit_error.mean()),
                    "actions": {
                        name: aggregate_point_rows(local_rows[name]) for name in action_rows
                    },
                    "sample_metadata": metadata,
                }
            )
            del video
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    expected_rows = int(config["partition"]["expected_failure_rows"])
    if len(commit_records) != expected_rows or len(causal_records) != expected_rows:
        raise ValueError("Gate 3C1E completed row drift")
    selector_consistency = canonical_json_sha256(causal_records) == canonical_json_sha256(
        [_parent_causal_record(record) for record in parent_replay["records"]]
    )
    action_count = int(sum(bool(record["action"]) for record in commit_records))
    action_commit = [record for record in commit_records if bool(record["action"])]
    commit = {
        "rows": len(commit_records),
        "action_rows": action_count,
        "action_coverage": action_count / len(commit_records),
        "mean_native_error_px": float(
            np.mean([float(record["native_commit_error_px"]) for record in commit_records])
        ),
        "mean_policy_error_px": float(
            np.mean([float(record["policy_commit_error_px"]) for record in commit_records])
        ),
        "mean_error_reduction_px": float(
            np.mean([float(record["commit_error_reduction_px"]) for record in commit_records])
        ),
        "policy_within_12px": float(
            np.mean([bool(record["policy_within_12px"]) for record in commit_records])
        ),
        "action_precision_within_12px": float(
            np.mean([bool(record["policy_within_12px"]) for record in action_commit])
        ),
    }
    actions = {name: aggregate_point_rows(rows) for name, rows in action_rows.items()}
    full_action_rows = [row for row in full_vs_native_rows if bool(row["action"])]
    coordinate_action_rows = [row for row in coordinate_vs_native_rows if bool(row["action"])]
    memory_action_rows = [row for row in memory_vs_coordinate_rows if bool(row["action"])]
    comparisons = {
        "full_vs_native": _aggregate_comparison(
            full_vs_native_rows, prefix="full_vs_native", seed=seed, samples=samples
        ),
        "full_vs_native_action_rows": _aggregate_comparison(
            full_action_rows,
            prefix="full_vs_native",
            seed=seed + 10,
            samples=samples,
        ),
        "coordinate_vs_native": _aggregate_comparison(
            coordinate_vs_native_rows,
            prefix="coordinate_vs_native",
            seed=seed + 20,
            samples=samples,
        ),
        "coordinate_vs_native_action_rows": _aggregate_comparison(
            coordinate_action_rows,
            prefix="coordinate_vs_native",
            seed=seed + 30,
            samples=samples,
        ),
        "memory_vs_coordinate": _aggregate_comparison(
            memory_vs_coordinate_rows,
            prefix="memory_vs_coordinate",
            seed=seed + 40,
            samples=samples,
        ),
        "memory_vs_coordinate_action_rows": _aggregate_comparison(
            memory_action_rows,
            prefix="memory_vs_coordinate",
            seed=seed + 50,
            samples=samples,
        ),
    }
    video_nonnegative = _video_nonnegative_fraction(
        full_vs_native_rows, "full_vs_native"
    )
    memory_better = _video_nonnegative_fraction(
        memory_vs_coordinate_rows, "memory_vs_coordinate"
    )
    native_replay_max_abs = float(max(native_replay_differences, default=0.0))
    scientific = {
        "selector_consistency_exact": selector_consistency,
        "commit": commit,
        "actions": actions,
        "comparisons": comparisons,
        "full_future_nonnegative_video_fraction": video_nonnegative,
        "memory_better_video_fraction": memory_better,
        "native_replay_max_abs_px": native_replay_max_abs,
        "preflight_causal_record_digest": preflight_causal_digest,
        "causal_record_digest": canonical_json_sha256(causal_records),
        "shortlist_digest": canonical_json_sha256(shortlist_hashes),
        "action_digest": canonical_json_sha256(action_hashes),
        "output_candidate_digest": canonical_json_sha256(output_candidate_hashes),
        "future_coordinate_digest": {
            name: canonical_json_sha256(values)
            for name, values in future_coordinate_hashes.items()
        },
        "point_records_digest": canonical_json_sha256(point_records),
        "video_records_digest": canonical_json_sha256(video_records),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)

    comparison = None
    exact_replay = False
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        reference_scientific = reference["scientific"]
        keys = (
            "selector_consistency_exact",
            "commit",
            "actions",
            "comparisons",
            "full_future_nonnegative_video_fraction",
            "memory_better_video_fraction",
            "native_replay_max_abs_px",
            "preflight_causal_record_digest",
            "causal_record_digest",
            "shortlist_digest",
            "action_digest",
            "output_candidate_digest",
            "future_coordinate_digest",
            "point_records_digest",
            "video_records_digest",
            "scientific_payload_sha256",
        )
        comparison = {key: scientific[key] == reference_scientific[key] for key in keys}
        exact_replay = all(comparison.values())

    checks = gate_checks(
        selector_consistency_exact=selector_consistency,
        commit=commit,
        full_vs_native=comparisons["full_vs_native"],
        full_vs_native_action=comparisons["full_vs_native_action_rows"],
        memory_vs_coordinate=comparisons["memory_vs_coordinate"],
        video_nonnegative_fraction=video_nonnegative,
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
        decision = config["pass_gates"]["decision_pass" if passed else "decision_fail"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "partition": config["partition"]["name"],
        "source_indices": config["partition"]["source_indices"],
        "videos": int(config["partition"]["expected_videos"]),
        "failure_rows": len(commit_records),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "candidate_cache_index": config["partition"]["candidate_cache_index"],
        "candidate_cache_index_sha256": candidate_index["_index_file_sha256"],
        "frozen_bundle_artifact": parent_replay["frozen_bundle_artifact"],
        "frozen_policy": policy,
        "scientific": scientific,
        "point_records": point_records,
        "video_records": video_records,
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": comparison,
        "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision},
        "read_state": config["partition"]["expected_read_state"],
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    result["result_payload_sha256"] = _result_payload_sha256(result)
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(
        config_path=Path(args.config).resolve(),
        output_path=Path(args.output).resolve(),
        reference_path=None if args.reference is None else Path(args.reference).resolve(),
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
