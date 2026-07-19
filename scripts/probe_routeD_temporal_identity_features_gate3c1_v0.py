#!/usr/bin/env python3
"""Probe causal candidate-conditioned temporal identity shapes and memory."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
    TEMPORAL_IDENTITY_FEATURE_SCHEMA_VERSION,
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


SCHEMA = "routeD_temporal_identity_feature_probe_gate3c1_v0"
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/routeD_temporal_identity_feature_probe_gate3c1_v0.yaml"
)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_mebibytes(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / float(1024**2)


def _dinov3_preprocess(video: torch.Tensor, device: str) -> torch.Tensor:
    value = F.interpolate(
        video.float() / 255.0,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    mean = value.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = value.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((value - mean) / std).to(device)


def _selected_native_support(
    predictor_state: Any, point_index: int
) -> list[torch.Tensor]:
    output = []
    for value in predictor_state.online_track_support:
        if value is None:
            raise ValueError("probe requires all CoTracker support levels")
        output.append(
            value[0, :, int(point_index)]
            .unsqueeze(0)
            .detach()
            .float()
            .contiguous()
        )
    return output


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _validate_authority(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    parent = config["authorized_parent"]
    parent_path = Path(parent["summary"]).resolve()
    if file_sha256(parent_path) != parent["summary_file_sha256"]:
        raise ValueError("Gate 3C1 parent summary file hash drift")
    summary = json.loads(parent_path.read_text())
    if (
        not bool(summary.get("pass"))
        or summary.get("formal_decision") != parent["required_decision"]
        or summary.get("summary_payload_sha256")
        != parent["summary_payload_sha256"]
    ):
        raise ValueError("Gate 3C0 did not authorize the Gate 3C1 probe")
    return parent_path, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1 probe config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1 probe locked-data flags must remain false")

    parent_path, parent_summary = _validate_authority(config)
    partition = config["partition"]
    manifest_path = Path(partition["manifest"]).resolve()
    if file_sha256(manifest_path) != partition["manifest_sha256"]:
        raise ValueError("Gate 3C1 expanded manifest hash drift")
    source_index = int(partition["probe_source_index"])
    allowed = [int(value) for value in partition["allowed_gradient_train_range"]]
    if not allowed[0] <= source_index <= allowed[1]:
        raise ValueError("Gate 3C1 probe source is outside gradient train")
    for name, values in config["backbones"].items():
        weight = values.get("checkpoint") or values.get("weights")
        expected = values.get("checkpoint_sha256") or values.get("weights_sha256")
        if file_sha256(weight) != expected:
            raise ValueError(f"Gate 3C1 frozen {name} weight hash drift")

    set_deterministic(193721 + source_index)
    device = str(args.device)
    sample, sample_metadata = load_manifest_sample(manifest_path, source_index)
    prepared = prepare_sample(
        sample, int(config["backbones"]["cotracker3"]["input_raster"])
    )
    video = prepared["video"].to(device)
    if tuple(video.shape[1:]) != (24, 3, 256, 256):
        raise ValueError("Gate 3C1 probe video shape drift")

    predictor = CoTrackerOnlinePredictor(
        checkpoint=config["backbones"]["cotracker3"]["checkpoint"]
    ).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)

    stage: dict[str, dict[str, float]] = {}
    _reset_peak(device)
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        native_initial = _initialize_and_first_window(
            predictor, video, _original_queries(prepared, video)
        )
        native_final = _continue_second_window(predictor, video, native_initial)
        native_coords = _coords_to_input(
            native_final,
            interp_height=int(predictor.interp_shape[0]),
            interp_width=int(predictor.interp_shape[1]),
        )
        membership = config["failure_membership"]
        rows = select_gate2_rows(
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
        failure = rows["failure_point_indices"].long()
        if failure.numel() == 0:
            raise RuntimeError("Gate 3C1 probe index 64 has no eligible failure")
        point_index = int(failure[0].item())

        observed = extract_cotracker_observed_feature_pyramid(
            predictor.model, video[:, :16]
        )
        frame15 = [value[0, 15].contiguous() for value in observed]
        frame15_shapes = [list(value.shape) for value in frame15]
        native_support = _selected_native_support(native_initial, point_index)
        score = geometry_preserving_pyramid_score(
            [value[None] for value in frame15],
            native_support,
            common_height=64,
            common_width=64,
            support_radius=int(
                config["candidate_bank"]["geometry_support_radius"]
            ),
            trim_fraction=float(
                config["candidate_bank"]["geometry_token_trim_fraction"]
            ),
        )["fused_score_map"]
        native_model_xy = native_initial.online_coords_predicted[
            0, 15, point_index
        ][None]
        native_input_xy = model_xy_to_input_xy(
            native_model_xy,
            input_height=256,
            input_width=256,
            model_height=int(predictor.interp_shape[0]),
            model_width=int(predictor.interp_shape[1]),
        )
        bank = config["candidate_bank"]
        candidate = extract_discrete_candidates(
            score,
            native_input_xy,
            top_k=int(bank["nonnative_top_k"]),
            nms_radius_grid_cells=int(bank["nms_radius_grid_cells"]),
            local_refinement_window_grid_cells=int(
                bank["local_refinement_window_grid_cells"]
            ),
            local_softmax_temperature=float(
                bank["local_softmax_temperature"]
            ),
            deduplicate_radius_input_px=float(
                bank["deduplicate_radius_input_px"]
            ),
            input_height=256,
            input_width=256,
        )
    _sync(device)
    stage["native_and_candidate"] = {
        "seconds": float(time.perf_counter() - start),
        "peak_cuda_mib": _peak_mebibytes(device),
    }

    coordinates = candidate["candidate_coordinates_xy"][0].float()
    scores = candidate["candidate_scores"][0].float()
    valid = candidate["candidate_valid_mask"][0].bool()
    candidate_coordinate_hash_before_teacher = tensor_sha256(
        coordinates.detach().cpu().contiguous()
    )
    valid_indices = torch.where(valid)[0]
    if int(valid_indices.numel()) < 2:
        raise RuntimeError("Gate 3C1 probe candidate bank collapsed")

    _reset_peak(device)
    _sync(device)
    start = time.perf_counter()
    reverse_queries = torch.zeros(1, valid_indices.numel(), 3, device=device)
    reverse_queries[0, :, 1:3] = coordinates[valid_indices].to(device)
    reverse_video = video[:, :16].flip(1)
    with torch.no_grad():
        reverse_state = _initialize_and_first_window(
            predictor, reverse_video, reverse_queries
        )
    reverse_track_valid = _coords_to_input(
        reverse_state,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    ).flip(1)
    reverse_visibility_valid = (
        torch.sigmoid(reverse_state.online_vis_predicted[0])
        .permute(1, 0)
        .flip(1)
        .detach()
        .float()
        .cpu()
    )
    reverse_confidence_valid = (
        torch.sigmoid(reverse_state.online_conf_predicted[0])
        .permute(1, 0)
        .flip(1)
        .detach()
        .float()
        .cpu()
    )
    _sync(device)
    stage["reverse_candidate_tracklets"] = {
        "seconds": float(time.perf_counter() - start),
        "peak_cuda_mib": _peak_mebibytes(device),
    }

    candidates = int(coordinates.shape[0])
    tracklets = torch.zeros(candidates, 16, 2)
    visibility = torch.zeros(candidates, 16)
    confidence = torch.zeros(candidates, 16)
    tracklets[valid_indices.cpu()] = reverse_track_valid
    visibility[valid_indices.cpu()] = reverse_visibility_valid
    confidence[valid_indices.cpu()] = reverse_confidence_valid

    del predictor, native_initial, native_final, reverse_state, observed, frame15
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    _reset_peak(device)
    _sync(device)
    start = time.perf_counter()
    from transformers import AutoModel

    dino = AutoModel.from_pretrained(
        config["backbones"]["dinov3"]["model_dir"], local_files_only=True
    ).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        encoded = dino(pixel_values=_dinov3_preprocess(video[0, :16], device))
    register_tokens = int(getattr(dino.config, "num_register_tokens", 4))
    grid_size = int(dino.config.image_size) // int(dino.config.patch_size)
    tokens = encoded.last_hidden_state[:, 1 + register_tokens :]
    dino_feature = tokens.reshape(
        16, grid_size, grid_size, tokens.shape[-1]
    ).permute(0, 3, 1, 2).contiguous()
    descriptor_sequence = sample_spatiotemporal_descriptors(
        dino_feature,
        tracklets.to(device),
        input_height=256,
        input_width=256,
    )
    query_tyx = prepared["query_points_tyx"][point_index]
    query_frame = int(round(float(query_tyx[0].item())))
    if query_frame >= int(membership["original_query_frame_less_than"]):
        raise ValueError("Gate 3C1 selected failure violates query-frame rule")
    query_coordinate_xy = (
        torch.stack([query_tyx[2], query_tyx[1]]).float() * 255.0
    )
    query_track = query_coordinate_xy[None, None].expand(1, 16, 2)
    query_sequence = sample_spatiotemporal_descriptors(
        dino_feature,
        query_track.to(device),
        input_height=256,
        input_width=256,
    )
    features = assemble_candidate_temporal_identity_features(
        descriptor_sequence=descriptor_sequence,
        query_descriptor=query_sequence[0, query_frame],
        tracklets_xy=tracklets.to(device),
        visibility_probability=visibility.to(device),
        confidence_probability=confidence.to(device),
        candidate_scores=scores.to(device),
        valid_mask=valid.to(device),
        query_frame=query_frame,
        query_coordinate_xy=query_coordinate_xy.to(device),
        input_height=256,
        input_width=256,
    )
    temporal = features["temporal_features"].detach().cpu().contiguous()
    static = features["static_features"].detach().cpu().contiguous()
    dino_feature_shape = list(dino_feature.shape)
    descriptor_shape = list(descriptor_sequence.shape)
    _sync(device)
    stage["dinov3_and_feature_assembly"] = {
        "seconds": float(time.perf_counter() - start),
        "peak_cuda_mib": _peak_mebibytes(device),
    }

    # Teacher is accessed only after candidates and causal feature tensors are
    # frozen in memory.
    teacher_yx = prepared["gt_tracks_yx"][point_index, 15]
    teacher_xy = torch.stack([teacher_yx[1], teacher_yx[0]]).float() * 255.0
    distance = torch.linalg.vector_norm(
        coordinates.detach().cpu() - teacher_xy[None], dim=-1
    )
    distance[~valid.detach().cpu()] = float("inf")
    native_future = native_coords[point_index, 16:24]
    future_gt_xy = (
        prepared["gt_tracks_yx"][point_index, 16:24][..., [1, 0]].float()
        * 255.0
    )
    future_visible = ~prepared["gt_occluded"][point_index, 16:24]
    native_future_error = torch.linalg.vector_norm(
        native_future - future_gt_xy, dim=-1
    )[future_visible]

    valid_cpu = valid.detach().cpu().contiguous()
    output = {
        "schema_version": (
            "routeD_temporal_identity_feature_probe_gate3c1_v0_result"
        ),
        "feature_schema_version": TEMPORAL_IDENTITY_FEATURE_SCHEMA_VERSION,
        "date": "2026-07-19",
        "status": "completed_shape_probe",
        "claim_scope": config["claim_scope"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "authorized_parent": {
            "summary": str(parent_path),
            "summary_file_sha256": file_sha256(parent_path),
            "summary_payload_sha256": parent_summary[
                "summary_payload_sha256"
            ],
            "formal_decision": parent_summary["formal_decision"],
        },
        "sample": {
            "source_index": source_index,
            "video_name": str(prepared["video_name"]),
            "sample_metadata": sample_metadata,
            "eligible_failure_count": int(
                rows["failure_eligible_count"].item()
            ),
            "selected_point_index": point_index,
            "query_frame": query_frame,
        },
        "candidate_bank": {
            "total_candidates": candidates,
            "valid_candidates": int(valid.sum().item()),
            "candidate_coordinate_hash_before_teacher": (
                candidate_coordinate_hash_before_teacher
            ),
            "native_commit_error_px": float(distance[0].item()),
            "oracle_min_commit_error_px": float(distance.min().item()),
            "oracle_support_within_12px": bool(distance.min().item() <= 12.0),
            "native_future_mean_error_px": float(
                native_future_error.mean().item()
            ),
        },
        "shapes": {
            "cotracker_feature_pyramid_frame15": frame15_shapes,
            "reverse_tracklets": list(tracklets.shape),
            "dinov3_feature_video": dino_feature_shape,
            "descriptor_sequence": descriptor_shape,
            "temporal_features": list(temporal.shape),
            "static_features": list(static.shape),
            "valid_mask": list(valid_cpu.shape),
        },
        "channels": {
            "temporal": list(TEMPORAL_FEATURE_CHANNELS),
            "static": list(STATIC_FEATURE_CHANNELS),
        },
        "storage_bytes": {
            "temporal_float32": _tensor_bytes(temporal),
            "static_float32": _tensor_bytes(static),
            "valid_bool": _tensor_bytes(valid_cpu),
            "total": _tensor_bytes(temporal)
            + _tensor_bytes(static)
            + _tensor_bytes(valid_cpu),
        },
        "tensor_hashes": {
            "temporal_features": tensor_sha256(temporal),
            "static_features": tensor_sha256(static),
            "valid_mask": tensor_sha256(valid_cpu),
        },
        "stage_measurements": stage,
        "integrity": {
            "all_feature_tensors_finite": bool(
                torch.isfinite(temporal).all()
                and torch.isfinite(static).all()
            ),
            "candidate_hash_frozen_before_teacher_distance": True,
            "teacher_available_to_candidate_coordinates_or_features": False,
            "future_frames_available_to_model_features": False,
            "observed_frames": [0, 15],
            "checkpoint_selection_read": False,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        },
        "locked_data": config["locked_data"],
    }
    output["tensor_hash_digest"] = canonical_json_sha256(
        output["tensor_hashes"]
    )
    output["summary_payload_sha256"] = canonical_json_sha256(output)
    output_path = Path(
        args.output or config["feature_probe"]["output"]
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
