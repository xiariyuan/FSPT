#!/usr/bin/env python3
"""Build the gradient-train-only Route-D temporal identity feature cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

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
from transformers import AutoModel

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    extract_cotracker_observed_feature_pyramid,
    model_xy_to_input_xy,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    select_gate2_rows,
)
from projects.mmp_tracker.mmp_tracker.routeD_discrete_candidates import (
    extract_discrete_candidates,
)
from projects.mmp_tracker.mmp_tracker.routeD_geometry_preserving_relocalization import (
    geometry_preserving_pyramid_score,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
    assemble_candidate_temporal_identity_features,
    sample_spatiotemporal_descriptors,
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
from scripts.probe_routeD_temporal_identity_features_gate3c1_v0 import (
    _dinov3_preprocess,
)


SCHEMA = "routeD_temporal_identity_train_cache_gate3c1a_v0"
CACHE_SCHEMA = "routeD_temporal_identity_train_cache_video_gate3c1a_v0"
INDEX_SCHEMA = "routeD_temporal_identity_train_cache_index_gate3c1a_v0"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/routeD_temporal_identity_train_cache_gate3c1a_v0.yaml"
)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _selected_native_support(
    predictor_state: Any, point_indices: torch.Tensor
) -> list[torch.Tensor]:
    output = []
    for value in predictor_state.online_track_support:
        if value is None:
            raise ValueError("cache requires all CoTracker support levels")
        selected = point_indices.to(value.device)
        output.append(
            value[0, :, selected]
            .permute(1, 0, 2)
            .detach()
            .float()
            .contiguous()
        )
    return output


def _dino_feature_video(
    dino: torch.nn.Module, video: torch.Tensor, device: str
) -> torch.Tensor:
    with torch.no_grad():
        encoded = dino(pixel_values=_dinov3_preprocess(video[0, :16], device))
    register_tokens = int(getattr(dino.config, "num_register_tokens", 4))
    grid_size = int(dino.config.image_size) // int(dino.config.patch_size)
    tokens = encoded.last_hidden_state[:, 1 + register_tokens :]
    return tokens.reshape(
        16, grid_size, grid_size, tokens.shape[-1]
    ).permute(0, 3, 1, 2).contiguous()


def _empty_tensors() -> dict[str, torch.Tensor]:
    return {
        "point_indices": torch.empty(0, dtype=torch.long),
        "query_frames": torch.empty(0, dtype=torch.long),
        "candidate_coordinates_xy": torch.empty(0, 129, 2),
        "candidate_scores": torch.empty(0, 129),
        "candidate_valid_mask": torch.empty(0, 129, dtype=torch.bool),
        "temporal_features": torch.empty(0, 129, 16, 9),
        "static_features": torch.empty(0, 129, 14),
        "teacher_candidate_distance_px": torch.empty(0, 129),
        "teacher_positive_within_12px": torch.empty(
            0, 129, dtype=torch.bool
        ),
        "native_commit_error_px": torch.empty(0),
        "native_future_mean_error_px": torch.empty(0),
    }


def _tensor_hashes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: tensor_sha256(value.detach().cpu().contiguous())
        for name, value in tensors.items()
    }


def verify_temporal_identity_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_partition: str,
    expected_source_index: int,
    expected_config_sha256: str,
    expected_read_state: Mapping[str, bool],
) -> None:
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("Gate 3C1A sidecar schema drift")
    if payload.get("partition") != expected_partition:
        raise ValueError("Gate 3C1A sidecar partition drift")
    if int(payload.get("source_index", -1)) != int(expected_source_index):
        raise ValueError("Gate 3C1A sidecar source drift")
    integrity = payload.get("integrity", {})
    for key, expected in expected_read_state.items():
        if bool(integrity.get(key)) is not bool(expected):
            raise ValueError(f"Gate 3C1A read-state drift: {key}")
    if payload.get("provenance", {}).get("config_sha256") != expected_config_sha256:
        raise ValueError("Gate 3C1A sidecar config drift")
    tensors = payload.get("tensors", {})
    rows = int(tensors["point_indices"].numel())
    expected = {
        "query_frames": (rows,),
        "candidate_coordinates_xy": (rows, 129, 2),
        "candidate_scores": (rows, 129),
        "candidate_valid_mask": (rows, 129),
        "temporal_features": (rows, 129, 16, 9),
        "static_features": (rows, 129, 14),
        "teacher_candidate_distance_px": (rows, 129),
        "teacher_positive_within_12px": (rows, 129),
        "native_commit_error_px": (rows,),
        "native_future_mean_error_px": (rows,),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"Gate 3C1A tensor shape drift: {name}")
    if rows and not bool(tensors["candidate_valid_mask"][:, 0].all()):
        raise ValueError("Gate 3C1A mandatory native candidate is invalid")
    for name in (
        "candidate_coordinates_xy",
        "candidate_scores",
        "temporal_features",
        "static_features",
        "native_commit_error_px",
        "native_future_mean_error_px",
    ):
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError(f"Gate 3C1A non-finite tensor: {name}")
    valid = tensors["candidate_valid_mask"]
    label_distance = tensors["teacher_candidate_distance_px"]
    if rows:
        if not bool((label_distance[~valid] == -1.0).all()):
            raise ValueError("Gate 3C1A invalid label sentinel drift")
        if not bool((label_distance[valid] >= 0.0).all()):
            raise ValueError("Gate 3C1A valid label distance drift")
    actual_hashes = _tensor_hashes(tensors)
    if actual_hashes != payload.get("tensor_hashes"):
        raise ValueError("Gate 3C1A tensor hash drift")
    candidate_hashes = [
        tensor_sha256(
            tensors["candidate_coordinates_xy"][row].contiguous()
        )
        for row in range(rows)
    ]
    if candidate_hashes != payload.get(
        "candidate_coordinate_hashes_before_teacher"
    ):
        raise ValueError("Gate 3C1A candidate pre-teacher hash drift")
    integrity = payload.get("integrity", {})
    mandatory_true = (
        "all_feature_tensors_finite",
        "candidate_hashes_frozen_before_teacher_labels",
    )
    if not all(bool(integrity.get(key)) for key in mandatory_true):
        raise ValueError("Gate 3C1A integrity flag failed")
    forbidden_true = (
        "teacher_available_to_candidate_coordinates_or_features",
        "future_frames_available_to_model_features",
    )
    if any(bool(integrity.get(key)) for key in forbidden_true):
        raise ValueError("Gate 3C1A locked-data integrity failed")


def verify_train_cache_payload(
    payload: Mapping[str, Any],
    *,
    expected_source_index: int,
    expected_config_sha256: str,
) -> None:
    verify_temporal_identity_cache_payload(
        payload,
        expected_partition="gradient_train",
        expected_source_index=expected_source_index,
        expected_config_sha256=expected_config_sha256,
        expected_read_state={
            "checkpoint_selection_read": False,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        },
    )


def _build_video(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    predictor: CoTrackerOnlinePredictor,
    dino: torch.nn.Module,
    source_index: int,
    device: str,
    partition_name: str = "gradient_train",
    read_state: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    if read_state is None:
        read_state = {
            "checkpoint_selection_read": False,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        }
    manifest = Path(config["partition"]["manifest"])
    sample, sample_metadata = load_manifest_sample(manifest, source_index)
    prepared = prepare_sample(
        sample, int(config["backbones"]["cotracker3"]["input_raster"])
    )
    video = prepared["video"].to(device)
    if tuple(video.shape[1:]) != (24, 3, 256, 256):
        raise ValueError("Gate 3C1A source video shape drift")
    set_deterministic(193721 + int(source_index))

    with torch.no_grad():
        native_initial = _initialize_and_first_window(
            predictor, video, _original_queries(prepared, video)
        )
        native_final = _continue_second_window(
            predictor, video, native_initial
        )
        native_coords = _coords_to_input(
            native_final,
            interp_height=int(predictor.interp_shape[0]),
            interp_width=int(predictor.interp_shape[1]),
        )
    membership = config["failure_membership"]
    selected = select_gate2_rows(
        native_future_coords_xy_px=native_coords[:, 16:24],
        gt_tracks_yx=prepared["gt_tracks_yx"],
        gt_occluded=prepared["gt_occluded"],
        original_query_frames=prepared["query_points_tyx"][:, 0],
        failure_error_min_px=float(
            membership["native_future_mean_error_px_min"]
        ),
        clean_error_max_px=float(
            membership["clean_future_mean_error_px_max"]
        ),
        min_future_visible_frames=int(
            membership["gt_visible_in_future_min_frames"]
        ),
        per_class_cap=int(membership["per_video_cap"]),
    )
    point_indices = selected["failure_point_indices"].long()
    rows = int(point_indices.numel())
    if rows == 0:
        tensors = _empty_tensors()
        candidate_hashes: list[str] = []
    else:
        with torch.no_grad():
            observed = extract_cotracker_observed_feature_pyramid(
                predictor.model, video[:, :16]
            )
            frame15 = [value[0, 15].contiguous() for value in observed]
            support = _selected_native_support(
                native_initial, point_indices
            )
            score_maps = geometry_preserving_pyramid_score(
                [value[None] for value in frame15],
                support,
                common_height=64,
                common_width=64,
                support_radius=int(
                    config["candidate_bank"]["geometry_support_radius"]
                ),
                trim_fraction=float(
                    config["candidate_bank"][
                        "geometry_token_trim_fraction"
                    ]
                ),
            )["fused_score_map"]
            native_model_xy = native_initial.online_coords_predicted[
                0, 15, point_indices.to(native_initial.online_coords_predicted.device)
            ]
            native_input_xy = model_xy_to_input_xy(
                native_model_xy,
                input_height=256,
                input_width=256,
                model_height=int(predictor.interp_shape[0]),
                model_width=int(predictor.interp_shape[1]),
            )
            bank_config = config["candidate_bank"]
            bank = extract_discrete_candidates(
                score_maps,
                native_input_xy,
                top_k=int(bank_config["nonnative_top_k"]),
                nms_radius_grid_cells=int(
                    bank_config["nms_radius_grid_cells"]
                ),
                local_refinement_window_grid_cells=int(
                    bank_config["local_refinement_window_grid_cells"]
                ),
                local_softmax_temperature=float(
                    bank_config["local_softmax_temperature"]
                ),
                deduplicate_radius_input_px=float(
                    bank_config["deduplicate_radius_input_px"]
                ),
                input_height=256,
                input_width=256,
            )
        coordinates = bank["candidate_coordinates_xy"].detach().float()
        candidate_scores = bank["candidate_scores"].detach().float()
        valid_mask = bank["candidate_valid_mask"].detach().bool()
        coordinates[~valid_mask] = 0.0
        candidate_scores[~valid_mask] = 0.0
        candidate_hashes = [
            tensor_sha256(coordinates[row].cpu().contiguous())
            for row in range(rows)
        ]

        tracklet_rows = []
        visibility_rows = []
        confidence_rows = []
        for row in range(rows):
            valid_indices = torch.where(valid_mask[row])[0]
            reverse_queries = torch.zeros(
                1, valid_indices.numel(), 3, device=device
            )
            reverse_queries[0, :, 1:3] = coordinates[
                row, valid_indices
            ].to(device)
            with torch.no_grad():
                reverse_state = _initialize_and_first_window(
                    predictor, video[:, :16].flip(1), reverse_queries
                )
            reverse_valid = _coords_to_input(
                reverse_state,
                interp_height=int(predictor.interp_shape[0]),
                interp_width=int(predictor.interp_shape[1]),
            ).flip(1)
            reverse_visibility = (
                torch.sigmoid(reverse_state.online_vis_predicted[0])
                .permute(1, 0)
                .flip(1)
                .detach()
                .float()
                .cpu()
            )
            reverse_confidence = (
                torch.sigmoid(reverse_state.online_conf_predicted[0])
                .permute(1, 0)
                .flip(1)
                .detach()
                .float()
                .cpu()
            )
            tracklet = torch.zeros(129, 16, 2)
            visibility = torch.zeros(129, 16)
            confidence = torch.zeros(129, 16)
            valid_cpu = valid_indices.cpu()
            tracklet[valid_cpu] = reverse_valid
            visibility[valid_cpu] = reverse_visibility
            confidence[valid_cpu] = reverse_confidence
            tracklet_rows.append(tracklet)
            visibility_rows.append(visibility)
            confidence_rows.append(confidence)

        feature_video = _dino_feature_video(dino, video, device)
        temporal_rows = []
        static_rows = []
        query_frames = []
        for row, point_index_tensor in enumerate(point_indices):
            point_index = int(point_index_tensor.item())
            descriptors = sample_spatiotemporal_descriptors(
                feature_video,
                tracklet_rows[row].to(device),
                input_height=256,
                input_width=256,
            )
            query_tyx = prepared["query_points_tyx"][point_index]
            query_frame = int(round(float(query_tyx[0].item())))
            if query_frame >= int(
                membership["original_query_frame_less_than"]
            ):
                raise ValueError("Gate 3C1A query-frame membership drift")
            query_xy = (
                torch.stack([query_tyx[2], query_tyx[1]]).float() * 255.0
            )
            query_track = query_xy[None, None].expand(1, 16, 2)
            query_sequence = sample_spatiotemporal_descriptors(
                feature_video,
                query_track.to(device),
                input_height=256,
                input_width=256,
            )
            feature = assemble_candidate_temporal_identity_features(
                descriptor_sequence=descriptors,
                query_descriptor=query_sequence[0, query_frame],
                tracklets_xy=tracklet_rows[row].to(device),
                visibility_probability=visibility_rows[row].to(device),
                confidence_probability=confidence_rows[row].to(device),
                candidate_scores=candidate_scores[row].to(device),
                valid_mask=valid_mask[row].to(device),
                query_frame=query_frame,
                query_coordinate_xy=query_xy.to(device),
                input_height=256,
                input_width=256,
            )
            temporal_rows.append(
                feature["temporal_features"].detach().cpu().contiguous()
            )
            static_rows.append(
                feature["static_features"].detach().cpu().contiguous()
            )
            query_frames.append(query_frame)

        temporal_features = torch.stack(temporal_rows)
        static_features = torch.stack(static_rows)
        coordinates_cpu = coordinates.cpu().contiguous()
        scores_cpu = candidate_scores.cpu().contiguous()
        valid_cpu = valid_mask.cpu().contiguous()

        # Teacher labels are constructed only after all candidate coordinates,
        # hashes, reverse tracklets, and causal feature tensors are frozen.
        teacher_yx = prepared["gt_tracks_yx"][point_indices, 15]
        teacher_xy = teacher_yx[..., [1, 0]].float() * 255.0
        teacher_distance = torch.linalg.vector_norm(
            coordinates_cpu - teacher_xy[:, None], dim=-1
        )
        teacher_distance[~valid_cpu] = -1.0
        positive = (teacher_distance >= 0.0) & (
            teacher_distance
            <= float(config["feature_contract"]["label_positive_radius_px"])
        )
        native_commit_error = teacher_distance[:, 0].clone()
        future_gt_xy = (
            prepared["gt_tracks_yx"][point_indices, 16:24][
                ..., [1, 0]
            ].float()
            * 255.0
        )
        future_visible = ~prepared["gt_occluded"][point_indices, 16:24]
        future_error_all = torch.linalg.vector_norm(
            native_coords[point_indices, 16:24] - future_gt_xy, dim=-1
        )
        native_future_mean_error = torch.stack(
            [
                future_error_all[row][future_visible[row]].mean()
                for row in range(rows)
            ]
        )
        tensors = {
            "point_indices": point_indices.cpu().contiguous(),
            "query_frames": torch.tensor(
                query_frames, dtype=torch.long
            ),
            "candidate_coordinates_xy": coordinates_cpu,
            "candidate_scores": scores_cpu,
            "candidate_valid_mask": valid_cpu,
            "temporal_features": temporal_features,
            "static_features": static_features,
            "teacher_candidate_distance_px": teacher_distance.contiguous(),
            "teacher_positive_within_12px": positive.contiguous(),
            "native_commit_error_px": native_commit_error.contiguous(),
            "native_future_mean_error_px": (
                native_future_mean_error.contiguous()
            ),
        }

    hashes = _tensor_hashes(tensors)
    payload = {
        "schema_version": CACHE_SCHEMA,
        "partition": partition_name,
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "tensors": tensors,
        "candidate_coordinate_hashes_before_teacher": candidate_hashes,
        "tensor_hashes": hashes,
        "tensor_hash_digest": canonical_json_sha256(hashes),
        "selection": {
            "eligible_failure_count": int(
                selected["failure_eligible_count"].item()
            ),
            "cached_failure_count": rows,
            "per_video_cap": int(membership["per_video_cap"]),
        },
        "integrity": {
            "all_feature_tensors_finite": True,
            "candidate_hashes_frozen_before_teacher_labels": True,
            "teacher_available_to_candidate_coordinates_or_features": False,
            "future_frames_available_to_model_features": False,
            "observed_frames": [0, 15],
            "checkpoint_selection_read": bool(
                read_state["checkpoint_selection_read"]
            ),
            "fit_only_internal_audit_read": bool(
                read_state["fit_only_internal_audit_read"]
            ),
            "original_model_validation_read": bool(
                read_state["original_model_validation_read"]
            ),
            "external_read": bool(read_state["external_read"]),
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": config_sha256,
            "manifest": str(manifest),
            "manifest_sha256": config["partition"]["manifest_sha256"],
            "sample_metadata": sample_metadata,
            "cotracker_checkpoint_sha256": config["backbones"][
                "cotracker3"
            ]["checkpoint_sha256"],
            "dinov3_weights_sha256": config["backbones"]["dinov3"][
                "weights_sha256"
            ],
        },
    }
    verify_temporal_identity_cache_payload(
        payload,
        expected_partition=partition_name,
        expected_source_index=source_index,
        expected_config_sha256=config_sha256,
        expected_read_state=read_state,
    )
    del video
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    probe_path = Path(parent["probe_result"])
    replay_path = Path(parent["replay_result"])
    if file_sha256(probe_path) != parent["probe_result_sha256"]:
        raise ValueError("Gate 3C1A probe-result hash drift")
    if file_sha256(replay_path) != parent["replay_result_sha256"]:
        raise ValueError("Gate 3C1A replay-result hash drift")
    probe = json.loads(probe_path.read_text())
    replay = json.loads(replay_path.read_text())
    if (
        probe.get("status") != parent["required_probe_status"]
        or probe.get("summary_payload_sha256")
        != parent["probe_summary_payload_sha256"]
        or replay.get("status") != parent["required_replay_status"]
        or not all(replay.get("exact_checks", {}).values())
    ):
        raise ValueError("Gate 3C1A parent probe did not authorize cache")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1A config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1A locked-data flags must remain false")
    _validate_parent(config)
    manifest = Path(config["partition"]["manifest"])
    if file_sha256(manifest) != config["partition"]["manifest_sha256"]:
        raise ValueError("Gate 3C1A manifest hash drift")
    for values in config["backbones"].values():
        weight = values.get("checkpoint") or values.get("weights")
        expected = values.get("checkpoint_sha256") or values.get(
            "weights_sha256"
        )
        if file_sha256(weight) != expected:
            raise ValueError("Gate 3C1A frozen weight hash drift")
    config_sha256 = file_sha256(config_path)
    bounds = [int(value) for value in config["partition"]["source_indices"]]
    expected_sources = list(range(bounds[0], bounds[1] + 1))
    if expected_sources != list(range(64, 384)):
        raise ValueError("Gate 3C1A gradient-train membership drift")

    device = str(args.device)
    set_deterministic(193721)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=config["backbones"]["cotracker3"]["checkpoint"]
    ).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    dino = AutoModel.from_pretrained(
        config["backbones"]["dinov3"]["model_dir"],
        local_files_only=True,
    ).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)

    output_root = Path(
        args.output_root or config["cache"]["output_root"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for source_index in expected_sources:
        sidecar = output_root / f"video_{source_index:05d}.pt"
        if args.resume and sidecar.exists():
            payload = torch.load(
                sidecar, map_location="cpu", weights_only=False
            )
            verify_train_cache_payload(
                payload,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
            )
            stage = "resume_skip"
        else:
            payload = _build_video(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                predictor=predictor,
                dino=dino,
                source_index=source_index,
                device=device,
            )
            _atomic_torch_save(payload, sidecar)
            payload = torch.load(
                sidecar, map_location="cpu", weights_only=False
            )
            verify_train_cache_payload(
                payload,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
            )
            stage = "video_complete"
        tensors = payload["tensors"]
        failure_rows = int(tensors["point_indices"].numel())
        if failure_rows:
            distance = tensors["teacher_candidate_distance_px"].clone()
            distance[~tensors["candidate_valid_mask"]] = float("inf")
            oracle = distance.min(dim=1).values
            static_top8 = distance[:, :9].min(dim=1).values
            native = distance[:, 0]
        else:
            oracle = torch.empty(0)
            static_top8 = torch.empty(0)
            native = torch.empty(0)
        row = {
            "source_index": source_index,
            "video_name": payload["video_name"],
            "sidecar": str(sidecar),
            "sidecar_sha256": file_sha256(sidecar),
            "failure_rows": failure_rows,
            "oracle_supported_12px_rows": int((oracle <= 12.0).sum()),
            "static_top8_supported_12px_rows": int(
                (static_top8 <= 12.0).sum()
            ),
            "native_supported_12px_rows": int((native <= 12.0).sum()),
            "tensor_hash_digest": payload["tensor_hash_digest"],
        }
        index_rows.append(row)
        print(json.dumps({"stage": stage, **row}), flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    completed_sources = [int(row["source_index"]) for row in index_rows]
    total_rows = sum(int(row["failure_rows"]) for row in index_rows)
    oracle_supported = sum(
        int(row["oracle_supported_12px_rows"]) for row in index_rows
    )
    static_supported = sum(
        int(row["static_top8_supported_12px_rows"]) for row in index_rows
    )
    native_supported = sum(
        int(row["native_supported_12px_rows"]) for row in index_rows
    )
    checks = {
        "exact_source_membership": completed_sources == expected_sources,
        "exact_video_count": len(index_rows)
        == int(config["partition"]["expected_videos"]),
        "all_sidecars_hash_verified": all(
            file_sha256(row["sidecar"]) == row["sidecar_sha256"]
            for row in index_rows
        ),
        "locked_data_unread": all(
            value is False for value in config["locked_data"].values()
        ),
    }
    passed = all(checks.values())
    index = {
        "schema_version": INDEX_SCHEMA,
        "date": "2026-07-19",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "formal_decision": config["completion_gate"][
            "decision_pass" if passed else "decision_fail"
        ],
        "partition": "gradient_train",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "manifest": str(manifest),
        "manifest_sha256": config["partition"]["manifest_sha256"],
        "expected_source_indices": expected_sources,
        "completed_source_indices": completed_sources,
        "videos": len(index_rows),
        "videos_with_failures": sum(
            int(row["failure_rows"] > 0) for row in index_rows
        ),
        "failure_rows": total_rows,
        "training_only_descriptive_support": {
            "top128_oracle_supported_12px_rows": oracle_supported,
            "top128_oracle_recall_12px": (
                0.0 if total_rows == 0 else oracle_supported / total_rows
            ),
            "static_native_plus_top8_supported_12px_rows": static_supported,
            "static_native_plus_top8_recall_12px": (
                0.0 if total_rows == 0 else static_supported / total_rows
            ),
            "native_supported_12px_rows": native_supported,
            "native_recall_12px": (
                0.0 if total_rows == 0 else native_supported / total_rows
            ),
        },
        "checks": checks,
        "combined_sidecar_sha256": canonical_json_sha256(
            [row["sidecar_sha256"] for row in index_rows]
        ),
        "combined_tensor_hash_digest": canonical_json_sha256(
            [row["tensor_hash_digest"] for row in index_rows]
        ),
        "rows": index_rows,
        "locked_data": config["locked_data"],
    }
    index["index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_root / "cache_index.json"
    _atomic_json_save(index, index_path)
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
