#!/usr/bin/env python3
"""Build exposed post-writeback visibility cache for Route-D Gate 3C1G0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml
from transformers import AutoModel

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

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_visibility_coupling_v0 import (
    VISIBILITY_FEATURE_CHANNELS,
    VISIBILITY_FEATURE_SCHEMA_VERSION,
    assemble_visibility_features,
    flatten_visibility_rows,
)
from scripts import diagnose_routeD_temporal_identity_full_population_gate3c1f2_v0 as diagnosis
from scripts import run_routeD_temporal_identity_full_population_gate3c1f2_v0 as formal
from scripts.probe_routeD_temporal_identity_features_gate3c1_v0 import (
    _dinov3_preprocess,
)

SCHEMA = "routeD_visibility_coupling_cache_gate3c1g0_v0"
VIDEO_SCHEMA = "routeD_visibility_coupling_cache_video_gate3c1g0_v0"
INDEX_SCHEMA = "routeD_visibility_coupling_cache_index_gate3c1g0_v0"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_visibility_coupling_cache_gate3c1g0_v0.yaml"


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _dino_feature_video_all(
    dino: torch.nn.Module, video: torch.Tensor, device: str
) -> torch.Tensor:
    frames = int(video.shape[1])
    with torch.no_grad():
        encoded = dino(pixel_values=_dinov3_preprocess(video[0], device))
    register_tokens = int(getattr(dino.config, "num_register_tokens", 4))
    grid_size = int(dino.config.image_size) // int(dino.config.patch_size)
    tokens = encoded.last_hidden_state[:, 1 + register_tokens :]
    output = tokens.reshape(
        frames, grid_size, grid_size, tokens.shape[-1]
    ).permute(0, 3, 1, 2).contiguous()
    if output.shape[0] != frames:
        raise ValueError("Gate 3C1G0 DINO frame-count drift")
    return output


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left.float(), dim=-1, eps=1.0e-8)
    right = torch.nn.functional.normalize(right.float(), dim=-1, eps=1.0e-8)
    return (left * right).sum(dim=-1)


def _speed(coordinates: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(coordinates)
    delta[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]
    return torch.linalg.vector_norm(delta / 255.0, dim=-1)


def _acceleration(coordinates: torch.Tensor) -> torch.Tensor:
    velocity = torch.zeros_like(coordinates)
    velocity[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]
    acceleration = torch.zeros_like(coordinates)
    acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
    return torch.linalg.vector_norm(acceleration / 255.0, dim=-1)


def _running_query_statistics(
    query_cosine: torch.Tensor, query_frames: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    actions, frames = query_cosine.shape
    mean = torch.empty_like(query_cosine)
    minimum = torch.empty_like(query_cosine)
    for action in range(actions):
        start = int(query_frames[action])
        for frame in range(frames):
            left = min(start, frame)
            values = query_cosine[action, left : frame + 1]
            mean[action, frame] = values.mean()
            minimum[action, frame] = values.amin()
    return mean, minimum


def _validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1G0 cache config")
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1G0 cache implementation drift: {name}")
    diagnostic_path = Path(config["authorized_parent"]["diagnostic_result"])
    if file_sha256(diagnostic_path) != config["authorized_parent"]["diagnostic_result_sha256"]:
        raise ValueError("Gate 3C1G0 diagnostic-result file drift")
    diagnostic = json.loads(diagnostic_path.read_text())
    without_hash = dict(diagnostic)
    payload = without_hash.pop("result_payload_sha256", None)
    if (
        payload != config["authorized_parent"]["diagnostic_result_payload_sha256"]
        or payload != canonical_json_sha256(without_hash)
        or diagnostic["scientific"]["scientific_payload_sha256"]
        != config["authorized_parent"]["diagnostic_scientific_payload_sha256"]
        or not bool(diagnostic["scientific"]["sealed_pipeline_exact"])
    ):
        raise ValueError("Gate 3C1G0 diagnostic-result payload drift")
    diagnostic_config_path = Path(config["authorized_parent"]["diagnostic_config"])
    if file_sha256(diagnostic_config_path) != config["authorized_parent"]["diagnostic_config_sha256"]:
        raise ValueError("Gate 3C1G0 diagnostic config drift")
    diagnostic_config, loaded, reference = diagnosis._validate_config(
        diagnostic_config_path
    )
    expected_sources = [int(value) for value in config["source_indices"]]
    if expected_sources != [int(value) for value in diagnostic_config["action_source_indices"]]:
        raise ValueError("Gate 3C1G0 source membership drift")
    if tuple(config["feature_contract"]["frames"]) != (15, 23):
        raise ValueError("Gate 3C1G0 feature-frame contract drift")
    if int(config["feature_contract"]["feature_dim"]) != len(VISIBILITY_FEATURE_CHANNELS):
        raise ValueError("Gate 3C1G0 feature dimension drift")
    if config["feature_contract"]["schema_version"] != VISIBILITY_FEATURE_SCHEMA_VERSION:
        raise ValueError("Gate 3C1G0 feature schema drift")
    if any(bool(value) for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1G0 locked-data flags must remain false")
    return config, loaded["formal"], loaded["top1"], reference


def _decision_projection(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return diagnosis._decision_projection(records)


def _build_video(
    *,
    config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    top1_config: Mapping[str, Any],
    reference: Mapping[str, Any],
    entry_bundle: Mapping[str, Any],
    top1_bundle: Mapping[str, Any],
    predictor: CoTrackerOnlinePredictor,
    dino: torch.nn.Module,
    source_index: int,
    device: str,
) -> dict[str, Any]:
    sample, metadata = formal.load_manifest_sample(
        Path(formal_config["data"]["manifest"]), source_index
    )
    prepared = formal.prepare_sample(
        sample, int(formal_config["backbone"]["input_raster"])
    )
    video = prepared["video"].to(device)
    formal.set_deterministic(
        int(formal_config["determinism"]["seed"]) + int(source_index)
    )
    initial = formal._initialize_and_first_window(
        predictor, video, formal._original_queries(prepared, video)
    )
    query_frames_all = prepared["query_points_tyx"][:, 0].round().long()
    eligible = torch.where(
        query_frames_all
        < int(formal_config["entry_contract"]["query_frame_less_than"])
    )[0]
    trajectory = formal.build_csrr_trajectory_features(
        initial, point_indices=eligible
    ).float().cpu()
    commit_visibility = torch.sigmoid(
        initial.online_vis_predicted[
            0, 15, eligible.to(initial.online_vis_predicted.device)
        ]
    ).float().cpu()
    commit_confidence = torch.sigmoid(
        initial.online_conf_predicted[
            0, 15, eligible.to(initial.online_conf_predicted.device)
        ]
    ).float().cpu()
    native_features = formal._selected_memory(
        initial.online_track_feat, eligible, support=False
    )
    native_supports = formal._selected_memory(
        initial.online_track_support, eligible, support=True
    )
    entry_features = formal.build_entry_features(
        trajectory_features=trajectory,
        visibility_probability=commit_visibility,
        confidence_probability=commit_confidence,
        native_track_features=native_features,
        native_track_supports=native_supports,
    )
    entry_probability = entry_bundle["model"].predict_proba(entry_features)[:, 1]
    operating = formal_config["entry_contract"]["operating_point"]
    entry_mask = formal.entry_action_mask(
        entry_probability=entry_probability,
        native_joint_probability=(commit_visibility * commit_confidence).numpy(),
        probability_min=float(operating["entry_probability_min"]),
        joint_probability_max=float(operating["native_joint_probability_max"]),
    )
    trigger_positions = np.where(entry_mask)[0]
    trigger_indices = eligible[torch.from_numpy(entry_mask)]
    candidates = formal._build_candidates(
        predictor=predictor,
        dino=dino,
        video=video,
        prepared=prepared,
        initial=initial,
        point_indices=trigger_indices,
        config=formal_config,
        device=device,
    )
    causal = formal.build_shortlist_top1_features(
        temporal_features=candidates["temporal"].float(),
        static_features=candidates["static"].float(),
        query_frames=candidates["query_frames"],
        valid_mask=candidates["valid"],
        selector_config=formal._selector_config(),
        feature_config=formal._feature_config(top1_config),
    )
    predictions = formal._predict_bundle(
        top1_bundle, {"features": causal["candidate_features"].cpu()}
    )
    selected_slot = predictions["selected"].astype(np.int64)
    row = np.arange(len(selected_slot))
    selected_support = predictions["support"][row, selected_slot].astype(np.float64)
    predicted_value = predictions["value"].astype(np.float64)
    predicted_harm = predictions["harm"].astype(np.float64)
    policy = formal_config["top1_contract"]["frozen_policy"]
    top1_action = (
        (selected_slot != 0)
        & (selected_support >= float(policy["support_min"]))
        & (predicted_value >= float(policy["value_min_px"]))
        & (predicted_harm <= float(policy["harm_max"]))
    )
    action_positions = np.where(top1_action)[0]
    if not len(action_positions):
        raise ValueError("Gate 3C1G0 action-video membership drift")
    output_slot = np.where(top1_action, selected_slot, 0)
    shortlist = causal["selected_indices"].numpy()
    output_candidate = shortlist[row, output_slot].astype(np.int64)
    output_xy = candidates["coordinates"][
        torch.arange(len(output_candidate)), torch.from_numpy(output_candidate)
    ].to(device)
    re_feat_batch, re_support_batch = formal.deterministic_reextract_cotracker_memory(
        candidates["frame15_pyramid"],
        output_xy,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
        stride=int(formal_config["backbone"]["model_stride"]),
        support_radius=int(formal_config["backbone"]["support_radius"]),
    )
    re_feat, re_support = formal._native_format_memory(
        re_feat_batch, re_support_batch
    )
    modified_initial = formal.apply_reextracted_state_action(
        initial,
        point_indices=trigger_indices,
        predicted_coordinates_input_xy=output_xy,
        apply_mask=torch.from_numpy(top1_action).to(device),
        reextracted_track_features=re_feat,
        reextracted_track_supports=re_support,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
        write_probability=False,
        write_memory=True,
    )
    native_final = formal._continue_second_window(predictor, video, initial)
    modified_final = formal._continue_second_window(
        predictor, video, modified_initial
    )
    native_xy = formal._coords_to_input(
        native_final,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    )
    modified_xy = formal._coords_to_input(
        modified_final,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    )
    native_vis_prob, native_conf_prob, native_visible = formal._visible_probabilities(
        native_final
    )
    modified_vis_prob, modified_conf_prob, modified_visible = formal._visible_probabilities(
        modified_final
    )
    ref_scientific = reference["scientific"]
    ref_digests = ref_scientific["digests"]
    actual_digests = {
        "eligible_indices": formal.tensor_sha256(eligible.contiguous()),
        "entry_features": formal.tensor_sha256(torch.from_numpy(entry_features)),
        "entry_probability": formal.tensor_sha256(torch.from_numpy(entry_probability)),
        "entry_mask": formal.tensor_sha256(torch.from_numpy(entry_mask)),
        "coordinates": formal.tensor_sha256(candidates["coordinates"]),
        "temporal": formal.tensor_sha256(candidates["temporal"]),
        "static": formal.tensor_sha256(candidates["static"]),
        "shortlist": formal.tensor_sha256(causal["selected_indices"]),
        "candidate_features": formal.tensor_sha256(causal["candidate_features"]),
        "selected_slot": formal.tensor_sha256(torch.from_numpy(selected_slot)),
        "top1_action": formal.tensor_sha256(torch.from_numpy(top1_action)),
        "output_candidate": formal.tensor_sha256(torch.from_numpy(output_candidate)),
        "native_coordinates": formal.tensor_sha256(native_xy.contiguous()),
        "modified_coordinates": formal.tensor_sha256(modified_xy.contiguous()),
        "native_visibility": formal.tensor_sha256(native_visible.contiguous()),
        "modified_visibility": formal.tensor_sha256(modified_visible.contiguous()),
    }
    exact_checks = {
        name: actual_digests[name] == ref_digests[name] for name in actual_digests
    }
    if not all(exact_checks.values()):
        raise ValueError(f"Gate 3C1G0 sealed digest drift: {exact_checks}")
    action_points = trigger_indices[torch.from_numpy(top1_action)]
    decision_records = []
    for local, point_index in zip(action_positions.tolist(), action_points.tolist()):
        decision_records.append(
            {
                "point_index": int(point_index),
                "entry_probability": float(
                    entry_probability[trigger_positions[int(local)]]
                ),
                "native_joint_probability": float(
                    (commit_visibility * commit_confidence)[
                        trigger_positions[int(local)]
                    ].item()
                ),
                "selected_slot": int(selected_slot[int(local)]),
                "output_candidate_index": int(output_candidate[int(local)]),
                "selected_support_probability": float(selected_support[int(local)]),
                "predicted_value_px": float(predicted_value[int(local)]),
                "predicted_harm_probability": float(predicted_harm[int(local)]),
            }
        )
    if _decision_projection(reference["action_records"]) != decision_records:
        raise ValueError("Gate 3C1G0 action-decision drift")

    frames = torch.arange(
        int(config["feature_contract"]["frames"][0]),
        int(config["feature_contract"]["frames"][1]) + 1,
        dtype=torch.long,
    )
    action_native_xy = native_xy[action_points]
    action_modified_xy = modified_xy[action_points]
    action_query_frames = query_frames_all[action_points]
    dino_video = _dino_feature_video_all(dino, video, device)
    dino_native = formal.sample_spatiotemporal_descriptors(
        dino_video,
        action_native_xy.to(device),
        input_height=256,
        input_width=256,
    ).cpu()
    dino_modified = formal.sample_spatiotemporal_descriptors(
        dino_video,
        action_modified_xy.to(device),
        input_height=256,
        input_width=256,
    ).cpu()
    query_xy = (
        prepared["query_points_tyx"][action_points][:, [2, 1]].float() * 255.0
    )
    query_tracks = query_xy[:, None].expand(-1, 24, -1).contiguous()
    dino_query_sequence = formal.sample_spatiotemporal_descriptors(
        dino_video,
        query_tracks.to(device),
        input_height=256,
        input_width=256,
    ).cpu()
    query_descriptor = dino_query_sequence[
        torch.arange(action_points.numel()), action_query_frames
    ]
    dino_mod_query = _cosine(dino_modified, query_descriptor[:, None])
    dino_native_query = _cosine(dino_native, query_descriptor[:, None])
    dino_mod_prev = torch.ones_like(dino_mod_query)
    dino_native_prev = torch.ones_like(dino_native_query)
    dino_mod_prev[:, 1:] = _cosine(dino_modified[:, 1:], dino_modified[:, :-1])
    dino_native_prev[:, 1:] = _cosine(dino_native[:, 1:], dino_native[:, :-1])
    dino_mod_commit = _cosine(
        dino_modified, dino_modified[:, 15:16]
    )
    dino_native_commit = _cosine(
        dino_native, dino_native[:, 15:16]
    )
    dino_mod_native = _cosine(dino_modified, dino_native)
    dino_running_mean, dino_running_min = _running_query_statistics(
        dino_mod_query, action_query_frames
    )

    cotracker_pyramid = formal.extract_cotracker_observed_feature_pyramid(
        predictor.model, video
    )
    cotracker_channels: dict[str, torch.Tensor] = {}
    for level, feature in enumerate(cotracker_pyramid):
        native_descriptor = formal.sample_spatiotemporal_descriptors(
            feature[0],
            action_native_xy.to(device),
            input_height=256,
            input_width=256,
        ).cpu()
        modified_descriptor = formal.sample_spatiotemporal_descriptors(
            feature[0],
            action_modified_xy.to(device),
            input_height=256,
            input_width=256,
        ).cpu()
        cotracker_channels[f"cotracker_level{level}_modified_commit_cosine"] = _cosine(
            modified_descriptor, modified_descriptor[:, 15:16]
        )[:, frames]
        cotracker_channels[f"cotracker_level{level}_native_commit_cosine"] = _cosine(
            native_descriptor, native_descriptor[:, 15:16]
        )[:, frames]
        cotracker_channels[f"cotracker_level{level}_modified_native_commit_cosine"] = _cosine(
            modified_descriptor, native_descriptor[:, 15:16]
        )[:, frames]
        cotracker_channels[f"cotracker_level{level}_modified_native_current_cosine"] = _cosine(
            modified_descriptor, native_descriptor
        )[:, frames]
        modified_previous = torch.ones_like(dino_mod_query)
        native_previous = torch.ones_like(dino_native_query)
        modified_previous[:, 1:] = _cosine(
            modified_descriptor[:, 1:], modified_descriptor[:, :-1]
        )
        native_previous[:, 1:] = _cosine(
            native_descriptor[:, 1:], native_descriptor[:, :-1]
        )
        cotracker_channels[f"cotracker_level{level}_modified_previous_cosine"] = modified_previous[:, frames]
        cotracker_channels[f"cotracker_level{level}_native_previous_cosine"] = native_previous[:, frames]

    native_speed = _speed(action_native_xy)
    modified_speed = _speed(action_modified_xy)
    native_acceleration = _acceleration(action_native_xy)
    modified_acceleration = _acceleration(action_modified_xy)
    modified_from_commit = torch.linalg.vector_norm(
        (action_modified_xy - action_modified_xy[:, 15:16]) / 255.0, dim=-1
    )
    border = torch.stack(
        [
            action_modified_xy[..., 0],
            action_modified_xy[..., 1],
            255.0 - action_modified_xy[..., 0],
            255.0 - action_modified_xy[..., 1],
        ],
        dim=-1,
    ).amin(dim=-1) / 255.0
    native_action_vis = native_vis_prob[action_points]
    native_action_conf = native_conf_prob[action_points]
    modified_action_vis = modified_vis_prob[action_points]
    modified_action_conf = modified_conf_prob[action_points]
    native_joint = native_action_vis * native_action_conf
    modified_joint = modified_action_vis * modified_action_conf
    action_entry = torch.tensor(
        [record["entry_probability"] for record in decision_records], dtype=torch.float32
    )
    action_commit_joint = torch.tensor(
        [record["native_joint_probability"] for record in decision_records], dtype=torch.float32
    )
    action_support = torch.tensor(
        [record["selected_support_probability"] for record in decision_records], dtype=torch.float32
    )
    action_value = torch.tensor(
        [record["predicted_value_px"] / 64.0 for record in decision_records], dtype=torch.float32
    )
    action_harm = torch.tensor(
        [record["predicted_harm_probability"] for record in decision_records], dtype=torch.float32
    )
    action_slot = torch.tensor(
        [record["selected_slot"] / 8.0 for record in decision_records], dtype=torch.float32
    )
    action_candidate = torch.tensor(
        [record["output_candidate_index"] / 128.0 for record in decision_records], dtype=torch.float32
    )
    repeated = lambda value: value[:, None].expand(-1, frames.numel())
    native_frame_xy = action_native_xy[:, frames] / 255.0
    modified_frame_xy = action_modified_xy[:, frames] / 255.0
    delta_xy = modified_frame_xy - native_frame_xy
    channels: dict[str, torch.Tensor] = {
        "frame_fraction": (frames.float() / 23.0)[None].expand(action_points.numel(), -1),
        "native_x": native_frame_xy[..., 0],
        "native_y": native_frame_xy[..., 1],
        "modified_x": modified_frame_xy[..., 0],
        "modified_y": modified_frame_xy[..., 1],
        "coordinate_delta_x": delta_xy[..., 0],
        "coordinate_delta_y": delta_xy[..., 1],
        "coordinate_delta_norm": torch.linalg.vector_norm(delta_xy, dim=-1),
        "native_speed": native_speed[:, frames],
        "modified_speed": modified_speed[:, frames],
        "native_acceleration": native_acceleration[:, frames],
        "modified_acceleration": modified_acceleration[:, frames],
        "modified_distance_from_commit": modified_from_commit[:, frames],
        "modified_border_margin": border[:, frames],
        "native_visibility_probability": native_action_vis[:, frames],
        "native_confidence_probability": native_action_conf[:, frames],
        "native_joint_probability": native_joint[:, frames],
        "modified_visibility_probability": modified_action_vis[:, frames],
        "modified_confidence_probability": modified_action_conf[:, frames],
        "modified_joint_probability": modified_joint[:, frames],
        "visibility_probability_delta": (modified_action_vis - native_action_vis)[:, frames],
        "confidence_probability_delta": (modified_action_conf - native_action_conf)[:, frames],
        "joint_probability_delta": (modified_joint - native_joint)[:, frames],
        "native_visible_binary": native_visible[action_points][:, frames].float(),
        "modified_visible_binary": modified_visible[action_points][:, frames].float(),
        "dino_modified_query_cosine": dino_mod_query[:, frames],
        "dino_native_query_cosine": dino_native_query[:, frames],
        "dino_query_cosine_delta": (dino_mod_query - dino_native_query)[:, frames],
        "dino_modified_previous_cosine": dino_mod_prev[:, frames],
        "dino_native_previous_cosine": dino_native_prev[:, frames],
        "dino_modified_commit_cosine": dino_mod_commit[:, frames],
        "dino_native_commit_cosine": dino_native_commit[:, frames],
        "dino_modified_native_cosine": dino_mod_native[:, frames],
        "dino_modified_query_running_mean": dino_running_mean[:, frames],
        "dino_modified_query_running_minimum": dino_running_min[:, frames],
        "entry_probability": repeated(action_entry),
        "commit_native_joint_probability": repeated(action_commit_joint),
        "selected_support_probability": repeated(action_support),
        "predicted_value_normalized": repeated(action_value),
        "predicted_harm_probability": repeated(action_harm),
        "selected_slot_normalized": repeated(action_slot),
        "output_candidate_index_normalized": repeated(action_candidate),
        **cotracker_channels,
    }
    feature_tensor = assemble_visibility_features(
        channels, rows=action_points.numel(), frames=frames.numel()
    )
    flat = flatten_visibility_rows(
        feature_tensor,
        source_index=source_index,
        point_indices=action_points,
        frame_indices=frames,
    )
    gt_visible = (~prepared["gt_occluded"])[action_points][:, frames].bool()
    gt_xy = prepared["gt_tracks_yx"][..., [1, 0]].float() * 255.0
    native_error = torch.linalg.vector_norm(
        action_native_xy[:, frames] - gt_xy[action_points][:, frames], dim=-1
    )
    modified_error = torch.linalg.vector_norm(
        action_modified_xy[:, frames] - gt_xy[action_points][:, frames], dim=-1
    )
    threshold_hits = torch.stack(
        [(modified_error < threshold) & gt_visible for threshold in (1, 2, 4, 8, 16)],
        dim=-1,
    )
    categories = formal._category_rows(
        prepared=prepared, native_xy=native_xy, eligible=eligible
    )
    trigger_categories = [categories[int(position)] for position in trigger_positions]
    action_categories = [trigger_categories[int(position)] for position in action_positions]
    tensors = {
        **flat,
        "features_action_frame": feature_tensor,
        "action_point_indices": action_points.long().contiguous(),
        "action_frame_indices": frames.long().contiguous(),
        "gt_visible_action_frame": gt_visible.contiguous(),
        "native_error_px_action_frame": native_error.float().contiguous(),
        "modified_error_px_action_frame": modified_error.float().contiguous(),
        "modified_threshold_hits_action_frame": threshold_hits.contiguous(),
        "native_coordinates_xy": native_xy.float().contiguous(),
        "modified_coordinates_xy": modified_xy.float().contiguous(),
        "native_visibility_probability": native_vis_prob.float().contiguous(),
        "native_confidence_probability": native_conf_prob.float().contiguous(),
        "modified_visibility_probability": modified_vis_prob.float().contiguous(),
        "modified_confidence_probability": modified_conf_prob.float().contiguous(),
        "native_visibility": native_visible.contiguous(),
        "modified_visibility": modified_visible.contiguous(),
        "gt_tracks_yx": prepared["gt_tracks_yx"].float().contiguous(),
        "gt_occluded": prepared["gt_occluded"].bool().contiguous(),
        "query_points_tyx": prepared["query_points_tyx"].float().contiguous(),
    }
    tensor_hashes = {
        name: formal.tensor_sha256(value) for name, value in tensors.items()
    }
    payload = {
        "schema_version": VIDEO_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "feature_schema_version": VISIBILITY_FEATURE_SCHEMA_VERSION,
        "feature_channels": list(VISIBILITY_FEATURE_CHANNELS),
        "feature_dim": len(VISIBILITY_FEATURE_CHANNELS),
        "action_rows": int(action_points.numel()),
        "frame_rows": int(action_points.numel() * frames.numel()),
        "action_categories": action_categories,
        "sealed_digest_checks": exact_checks,
        "action_decisions_exact": True,
        "tensors": tensors,
        "tensor_hashes": tensor_hashes,
        "tensor_hash_digest": canonical_json_sha256(tensor_hashes),
        "sample_metadata": metadata,
    }
    payload["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "tensors"}
    )
    del video, dino_video, cotracker_pyramid
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def verify_cache_payload(
    payload: Mapping[str, Any], *, source_index: int
) -> None:
    if payload.get("schema_version") != VIDEO_SCHEMA:
        raise ValueError("Gate 3C1G0 cache video schema drift")
    if int(payload.get("source_index", -1)) != int(source_index):
        raise ValueError("Gate 3C1G0 cache source drift")
    if payload.get("feature_schema_version") != VISIBILITY_FEATURE_SCHEMA_VERSION:
        raise ValueError("Gate 3C1G0 cache feature schema drift")
    if payload.get("feature_channels") != list(VISIBILITY_FEATURE_CHANNELS):
        raise ValueError("Gate 3C1G0 cache feature channels drift")
    if not all(payload.get("sealed_digest_checks", {}).values()) or not bool(
        payload.get("action_decisions_exact")
    ):
        raise ValueError("Gate 3C1G0 cache sealed pipeline mismatch")
    tensors = payload["tensors"]
    actions = int(payload["action_rows"])
    frames = 9
    if tensors["features_action_frame"].shape != (
        actions,
        frames,
        len(VISIBILITY_FEATURE_CHANNELS),
    ):
        raise ValueError("Gate 3C1G0 cache feature tensor shape drift")
    actual_hashes = {
        name: formal.tensor_sha256(value) for name, value in tensors.items()
    }
    if actual_hashes != payload.get("tensor_hashes"):
        raise ValueError("Gate 3C1G0 cache tensor hash drift")
    if canonical_json_sha256(actual_hashes) != payload.get("tensor_hash_digest"):
        raise ValueError("Gate 3C1G0 cache tensor digest drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-index", action="append", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config, formal_config, top1_config, reference = _validate_config(config_path)
    output_root = Path(args.output_root or config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entry_bundle = joblib.load(formal_config["bundles"]["entry_bundle"]["path"])
    top1_bundle = joblib.load(formal_config["bundles"]["top1_bundle"]["path"])
    predictor = CoTrackerOnlinePredictor(
        checkpoint=formal_config["backbone"]["checkpoint"]
    ).to(args.device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    dino = AutoModel.from_pretrained(
        formal_config["dinov3"]["model_dir"], local_files_only=True
    ).to(args.device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    selected_sources = (
        [int(value) for value in config["source_indices"]]
        if args.source_index is None
        else [int(value) for value in args.source_index]
    )
    if any(value not in config["source_indices"] for value in selected_sources):
        raise ValueError("Gate 3C1G0 requested source outside frozen membership")
    rows = []
    total_actions = 0
    total_frame_rows = 0
    for source_index in selected_sources:
        source_index = int(source_index)
        sidecar = output_root / f"video_{source_index:05d}.pt"
        if args.resume and sidecar.exists():
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            verify_cache_payload(payload, source_index=source_index)
        else:
            payload = _build_video(
                config=config,
                formal_config=formal_config,
                top1_config=top1_config,
                reference=reference[source_index],
                entry_bundle=entry_bundle,
                top1_bundle=top1_bundle,
                predictor=predictor,
                dino=dino,
                source_index=source_index,
                device=str(args.device),
            )
            _atomic_torch_save(payload, sidecar)
            verify_cache_payload(payload, source_index=source_index)
        sidecar_sha = file_sha256(sidecar)
        rows.append(
            {
                "source_index": source_index,
                "video_name": payload["video_name"],
                "sidecar": str(sidecar),
                "sidecar_sha256": sidecar_sha,
                "action_rows": int(payload["action_rows"]),
                "frame_rows": int(payload["frame_rows"]),
                "tensor_hash_digest": payload["tensor_hash_digest"],
                "payload_sha256": payload["payload_sha256"],
            }
        )
        total_actions += int(payload["action_rows"])
        total_frame_rows += int(payload["frame_rows"])
        print(
            json.dumps(
                {
                    "stage": "video_complete",
                    "source_index": source_index,
                    "video_name": payload["video_name"],
                    "actions": payload["action_rows"],
                    "frame_rows": payload["frame_rows"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    index = {
        "schema_version": INDEX_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "feature_schema_version": VISIBILITY_FEATURE_SCHEMA_VERSION,
        "feature_channels": list(VISIBILITY_FEATURE_CHANNELS),
        "feature_dim": len(VISIBILITY_FEATURE_CHANNELS),
        "source_indices": selected_sources,
        "videos": len(rows),
        "actions": total_actions,
        "frame_rows": total_frame_rows,
        "rows": rows,
        "combined_sidecar_digest": canonical_json_sha256(
            [row["sidecar_sha256"] for row in rows]
        ),
        "combined_tensor_digest": canonical_json_sha256(
            [row["tensor_hash_digest"] for row in rows]
        ),
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    index["index_payload_sha256"] = canonical_json_sha256(index)
    _atomic_json_save(index, output_root / "cache_index.json")
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
