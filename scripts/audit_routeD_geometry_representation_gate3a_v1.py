#!/usr/bin/env python3
"""Teacher audit and future rollout for frozen Route-D Gate 3A v1 candidates."""
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
    aggregate_point_rows,
    exact_snapshot_from_artifact,
    load_csrr_cache_video,
    point_future_rows,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_state_reinstantiation_basin import (
    video_cluster_bootstrap_ci,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)


SCHEMA = "routeD_geometry_representation_audit_gate3a_v1"
CACHE_SCHEMA = "routeD_geometry_representation_candidate_cache_gate3a_v1"
INDEX_SCHEMA = "routeD_geometry_representation_candidate_index_gate3a_v1"
REPRESENTATIONS = (
    "M0_pooled_native_gate2",
    "M1_geometry_native",
    "M2_geometry_immutable_query",
)
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_geometry_representation_audit_gate3a_v1.yaml"


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
        raise RuntimeError("incomplete Gate 3A v1 continuation")
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
    features: Sequence[torch.Tensor], supports: Sequence[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        [value.permute(2, 1, 0, 3).contiguous() for value in features],
        [value.permute(2, 1, 0, 3).contiguous() for value in supports],
    )


def _load_candidate_index(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != INDEX_SCHEMA or not value.get("complete"):
        raise ValueError("Gate 3A v1 candidate index incomplete")
    expected = [int(item) for item in config["partition"]["source_indices"]]
    if value.get("expected_source_indices") != expected or value.get(
        "completed_source_indices"
    ) != expected:
        raise ValueError("Gate 3A v1 candidate index membership drift")
    if int(value.get("completed_failure_points", -1)) != int(
        config["partition"]["expected_failure_points"]
    ):
        raise ValueError("Gate 3A v1 candidate point-count drift")
    integrity = value.get("integrity", {})
    if not bool(integrity.get("candidate_sets_frozen_before_teacher_read")):
        raise ValueError("Gate 3A v1 candidates were not sealed before teacher audit")
    return value


def _teacher_coordinates(
    config: Mapping[str, Any], source_index: int, point_indices: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    # This function is called only after the complete candidate index is loaded
    # and verified. It is the first Gate 3A v1 access to the commit teacher.
    sample, metadata = load_manifest_sample(
        Path(config["partition"]["manifest"]), int(source_index)
    )
    prepared = prepare_sample(sample, 256)
    gt = prepared["gt_tracks_yx"][point_indices, 15]
    visible = ~prepared["gt_occluded"][point_indices, 15]
    if not bool(visible.all()):
        raise ValueError("Gate 3A v1 failure row is not visible at commit")
    return torch.stack([gt[:, 1], gt[:, 0]], dim=-1).float() * 255.0, metadata


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    error = [float(row["error_reduction_px"]) for row in rows]
    utility = [float(row["utility_gain"]) for row in rows]
    return {
        "mean_error_reduction_px": float(np.mean(error)),
        "threshold_utility_gain": float(np.mean(utility)),
        "positive_point_fraction": float(np.mean([value > 0.0 for value in error])),
        "error_reduction_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key="error_reduction_px", seed=seed, samples=samples
        ),
        "utility_gain_video_cluster_CI": video_cluster_bootstrap_ci(
            rows, value_key="utility_gain", seed=seed + 1, samples=samples
        ),
    }


def _representation_checks(
    summary: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "candidate_recall_within_12px_min": float(summary["commit"]["recall_within_12px"])
        >= float(gates["candidate_recall_within_12px_min"]),
        "median_nearest_candidate_error_px_max": float(
            summary["commit"]["median_nearest_candidate_error_px"]
        )
        <= float(gates["median_nearest_candidate_error_px_max"]),
        "future_mean_error_reduction_vs_native_px_min": float(
            summary["future_vs_native"]["mean_error_reduction_px"]
        )
        >= float(gates["future_mean_error_reduction_vs_native_px_min"]),
        "future_error_reduction_CI_lower_px_min": float(
            summary["future_vs_native"]["error_reduction_video_cluster_CI"]["lower"]
        )
        >= float(gates["future_error_reduction_CI_lower_px_min"]),
        "future_threshold_utility_gain_vs_native_min": float(
            summary["future_vs_native"]["threshold_utility_gain"]
        )
        >= float(gates["future_threshold_utility_gain_vs_native_min"]),
        "future_utility_gain_CI_lower_min": float(
            summary["future_vs_native"]["utility_gain_video_cluster_CI"]["lower"]
        )
        >= float(gates["future_utility_gain_CI_lower_min"]),
        "future_positive_point_fraction_min": float(
            summary["future_vs_native"]["positive_point_fraction"]
        )
        >= float(gates["future_positive_point_fraction_min"]),
        "future_severe_16px_rate_reduction_min": float(
            summary["future_vs_native"]["severe_16px_rate_reduction"]
        )
        >= float(gates["future_severe_16px_rate_reduction_min"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--candidate-qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3A v1 audit config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3A v1 locked-data flags must remain false")
    candidate_index_path = Path(args.candidate_index).resolve()
    candidate_index = _load_candidate_index(candidate_index_path, config)
    if candidate_index["config_sha256"] != file_sha256(config_path):
        raise ValueError("Gate 3A v1 candidate/config hash mismatch")
    qualification_path = Path(args.candidate_qualification).resolve()
    qualification = json.loads(qualification_path.read_text())
    if qualification.get("schema_version") != "routeD_geometry_candidate_cache_qualification_gate3a_v1":
        raise ValueError("Gate 3A v1 candidate qualification schema drift")
    if not bool(qualification.get("pass")):
        raise ValueError("Gate 3A v1 candidate cache replay did not pass")
    if qualification.get("primary_index_sha256") != file_sha256(candidate_index_path):
        raise ValueError("Gate 3A v1 audited index was not the qualified primary")

    gate2_index_path = Path(config["cache"]["gate2_root"]) / config["cache"]["gate2_fit_index"]
    gate2_index = load_complete_csrr_cache_index(
        gate2_index_path, expected_partition="fit_internal_validation"
    )
    gate2_rows = {int(row["source_index"]): row for row in gate2_index["rows"]}
    parent = yaml.safe_load(Path(config["frozen_parent"]["gate2_config"]).read_text())
    seed = int(config["metrics"]["bootstrap_seed"])
    samples = int(config["metrics"]["bootstrap_samples"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=parent["backbone"]["checkpoint"]
    ).to(args.device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)

    all_native: list[dict[str, float]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in REPRESENTATIONS
    }
    commit_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in REPRESENTATIONS
    }
    video_reports = []
    with torch.no_grad():
        for candidate_row in candidate_index["rows"]:
            source_index = int(candidate_row["source_index"])
            sidecar = Path(candidate_row["sidecar"])
            if file_sha256(sidecar) != candidate_row["sidecar_sha256"]:
                raise ValueError("Gate 3A v1 candidate sidecar hash drift")
            frozen = torch.load(sidecar, map_location="cpu", weights_only=False)
            if frozen.get("schema_version") != CACHE_SCHEMA:
                raise ValueError("Gate 3A v1 candidate sidecar schema drift")
            artifact = load_csrr_cache_video(
                gate2_rows[source_index],
                expected_partition="fit_internal_validation",
                expected_config_sha256=str(config["frozen_parent"]["gate2_config_sha256"]),
            )
            tensors = artifact["model_tensors"]
            failure_local = torch.where(tensors["apply_target"].float() > 0.5)[0]
            point_indices = tensors["point_indices"][failure_local].long()
            if not torch.equal(point_indices, frozen["point_indices"]):
                raise ValueError("Gate 3A v1 point identity drift")

            teacher_xy, sample_metadata = _teacher_coordinates(
                config, source_index, point_indices
            )
            gt_future = tensors["future_gt_coordinates_xy"][failure_local].float()
            visible_future = tensors["future_visible_mask"][failure_local].bool()
            native_future = tensors["native_future_coordinates_xy"][failure_local].float()
            native_metrics = point_future_rows(
                native_future, gt_future, visible_future
            )
            all_native.extend(native_metrics)
            exact_initial = exact_snapshot_from_artifact(artifact, args.device)
            per_video = {
                "source_index": source_index,
                "video_name": str(artifact["video_name"]),
                "failure_points": int(point_indices.numel()),
                "sample_metadata": sample_metadata,
                "representations": {},
            }
            frame_pyramid = [
                value[None, None].to(device=args.device, dtype=torch.float32)
                for value in tensors["frame_feature_pyramid"]
            ]
            for representation in REPRESENTATIONS:
                candidate = frozen["candidates"][representation]
                coordinates = candidate["candidate_coordinates_xy"].float()
                valid = candidate["candidate_valid_mask"].bool()
                distance = torch.linalg.vector_norm(
                    coordinates - teacher_xy[:, None], dim=-1
                )
                distance[~valid] = float("inf")
                nearest_distance, nearest_index = distance.min(dim=1)
                rows = torch.arange(point_indices.numel())
                selected_xy = coordinates[rows, nearest_index]
                if not bool(torch.isfinite(nearest_distance).all()):
                    raise RuntimeError("Gate 3A v1 row has no valid candidate")
                for local in range(point_indices.numel()):
                    commit_rows[representation].append(
                        {
                            "source_index": source_index,
                            "point_index": int(point_indices[local]),
                            "nearest_error_px": float(nearest_distance[local]),
                            "nearest_candidate_rank": int(nearest_index[local]) + 1,
                            "within_4px": float(nearest_distance[local] <= 4.0),
                            "within_8px": float(nearest_distance[local] <= 8.0),
                            "within_12px": float(nearest_distance[local] <= 12.0),
                        }
                    )
                re_feat_batch, re_support_batch = deterministic_reextract_cotracker_memory(
                    frame_pyramid,
                    selected_xy.to(args.device),
                    input_height=256,
                    input_width=256,
                    model_height=int(predictor.interp_shape[0]),
                    model_width=int(predictor.interp_shape[1]),
                    stride=int(parent["backbone"]["stride"]),
                    support_radius=int(parent["backbone"]["corr_radius"]),
                )
                re_feat, re_support = _native_format_memory(
                    re_feat_batch, re_support_batch
                )
                initial = apply_reextracted_state_action(
                    exact_initial,
                    point_indices=point_indices,
                    predicted_coordinates_input_xy=selected_xy.to(args.device),
                    apply_mask=torch.ones(point_indices.numel(), dtype=torch.bool),
                    reextracted_track_features=re_feat,
                    reextracted_track_supports=re_support,
                    input_height=256,
                    input_width=256,
                    model_height=int(predictor.interp_shape[0]),
                    model_width=int(predictor.interp_shape[1]),
                    write_probability=False,
                    write_memory=True,
                )
                final = _continue_cached(
                    predictor, initial, tensors["continuation_video_u8"], args.device
                )
                outcome = point_future_rows(
                    _future_coordinates_input(final, point_indices, predictor),
                    gt_future,
                    visible_future,
                )
                enriched = []
                for local, (row, native) in enumerate(zip(outcome, native_metrics)):
                    value = {
                        **row,
                        "source_index": source_index,
                        "point_index": int(point_indices[local]),
                        "error_reduction_px": float(native["mean_l2_error_px"])
                        - float(row["mean_l2_error_px"]),
                        "utility_gain": float(row["threshold_utility"])
                        - float(native["threshold_utility"]),
                        "severe_rate_reduction": float(native["severe_16px_rate"])
                        - float(row["severe_16px_rate"]),
                    }
                    enriched.append(value)
                    all_rows[representation].append(value)
                per_video["representations"][representation] = {
                    "candidate_count_min": int(candidate["candidate_count"].min()),
                    "candidate_count_max": int(candidate["candidate_count"].max()),
                    "commit_mean_error_px": float(nearest_distance.mean()),
                    "commit_recall_within_12px": float(
                        (nearest_distance <= 12.0).float().mean()
                    ),
                    "future": aggregate_point_rows(outcome),
                }
            video_reports.append(per_video)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    native_aggregate = aggregate_point_rows(all_native)
    representation_summary = {}
    checks = {}
    gates = config["representation_pass_gates"]
    for ordinal, representation in enumerate(REPRESENTATIONS):
        commits = commit_rows[representation]
        outcomes = all_rows[representation]
        future_aggregate = aggregate_point_rows(outcomes)
        comparison = _comparison_rows(
            outcomes, seed=seed + 10 * ordinal, samples=samples
        )
        comparison["severe_16px_rate_reduction"] = float(
            np.mean([row["severe_rate_reduction"] for row in outcomes])
        )
        summary = {
            "commit": {
                "mean_nearest_candidate_error_px": float(
                    np.mean([row["nearest_error_px"] for row in commits])
                ),
                "median_nearest_candidate_error_px": float(
                    np.median([row["nearest_error_px"] for row in commits])
                ),
                "recall_within_4px": float(np.mean([row["within_4px"] for row in commits])),
                "recall_within_8px": float(np.mean([row["within_8px"] for row in commits])),
                "recall_within_12px": float(np.mean([row["within_12px"] for row in commits])),
            },
            "future": future_aggregate,
            "future_vs_native": comparison,
        }
        representation_summary[representation] = summary
        local_checks = _representation_checks(summary, gates)
        local_checks["pass"] = all(local_checks.values())
        checks[representation] = local_checks

    m0, m1, m2 = [checks[name]["pass"] for name in REPRESENTATIONS]
    decisions = config["formal_decisions"]
    if m0:
        decision = decisions["M0_pass"]
        selected_representation = REPRESENTATIONS[0]
    elif m1:
        decision = decisions["M0_fail_M1_pass"]
        selected_representation = REPRESENTATIONS[1]
    elif m2:
        decision = decisions["M0_M1_fail_M2_pass"]
        selected_representation = REPRESENTATIONS[2]
    else:
        decision = decisions["all_fail"]
        selected_representation = None

    report = {
        "schema_version": SCHEMA,
        "status": "completed",
        "formal_decision": decision,
        "selected_representation": selected_representation,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "candidate_index": str(candidate_index_path),
        "candidate_index_sha256": file_sha256(candidate_index_path),
        "candidate_qualification": str(qualification_path),
        "candidate_qualification_sha256": file_sha256(qualification_path),
        "combined_candidate_hash_digest": candidate_index[
            "combined_candidate_hash_digest"
        ],
        "failure_points": len(all_native),
        "failure_videos": len(video_reports),
        "native": native_aggregate,
        "representations": representation_summary,
        "checks": checks,
        "video_reports": video_reports,
        "commit_rows": commit_rows,
        "outcome_rows": all_rows,
        "integrity": {
            "candidate_index_complete_before_teacher_read": True,
            "candidate_builder_accessed_teacher": False,
            "model_validation_read": False,
            "external_read": False,
        },
        "locked_data": config["locked_data"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "formal_decision": decision,
                "selected_representation": selected_representation,
                "failure_points": len(all_native),
                "representations": representation_summary,
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
