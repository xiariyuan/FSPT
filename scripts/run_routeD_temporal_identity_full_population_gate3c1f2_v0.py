#!/usr/bin/env python3
"""Run frozen causal full-population confirmation on raw-record-disjoint Kubric."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
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
from datasets.metrics import compute_tapvid_metrics

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    apply_reextracted_state_action,
    build_csrr_trajectory_features,
    deterministic_reextract_cotracker_memory,
    extract_cotracker_observed_feature_pyramid,
    model_xy_to_input_xy,
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
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import _tracks_from_xy
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_entry_v0 import (
    ENTRY_FEATURE_SCHEMA_VERSION,
    build_entry_features,
    entry_action_mask,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_features import (
    assemble_candidate_temporal_identity_features,
    sample_spatiotemporal_descriptors,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1 import (
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
from scripts.audit_routeD_temporal_identity_causal_future_rollout_gate3c1e_v0 import (
    _native_format_memory,
)
from scripts.build_routeD_counterfactual_state_restorer_cache import _selected_memory
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
    _dino_feature_video,
    _selected_native_support,
)
from scripts.run_routeD_temporal_identity_top1_gate3c1d_v1 import (
    _feature_config,
    _predict_bundle,
    _selector_config,
)

SCHEMA = "routeD_temporal_identity_full_population_confirmation_gate3c1f2_v0"
RESULT_SCHEMA = "routeD_temporal_identity_full_population_confirmation_result_gate3c1f2_v0"
VIDEO_SCHEMA = "routeD_temporal_identity_full_population_confirmation_video_gate3c1f2_v0"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_temporal_identity_full_population_gate3c1f2_v0.yaml"


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _metric_dict(value: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, item in value.items():
        array = np.asarray(item)
        if array.size == 1:
            output[str(key)] = float(array.reshape(-1)[0])
    for required in ("AJ", "<avg", "OA"):
        if required not in output:
            raise ValueError(f"Gate 3C1F2 metric missing: {required}")
    return output


def _runtime_contract(config: Mapping[str, Any]) -> None:
    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    expected = {key: str(value) for key, value in config["runtime_contract"].items()}
    if actual != expected:
        raise ValueError(f"Gate 3C1F2 runtime drift: {actual} != {expected}")


def _validate_json_authority(
    authority: Mapping[str, Any], *, decision_path: Sequence[str] | None = None
) -> dict[str, Any]:
    path = Path(authority["path"])
    if file_sha256(path) != authority["file_sha256"]:
        raise ValueError(f"Gate 3C1F2 JSON authority file drift: {path}")
    payload = json.loads(path.read_text())
    without_hash = dict(payload)
    embedded = without_hash.pop(authority.get("payload_field", "result_payload_sha256"), None)
    if embedded != authority["payload_sha256"] or embedded != canonical_json_sha256(without_hash):
        raise ValueError(f"Gate 3C1F2 JSON authority payload drift: {path}")
    if decision_path is not None:
        current: Any = payload
        for key in decision_path:
            current = current[key]
        if current != authority["required_decision"]:
            raise ValueError(f"Gate 3C1F2 authority decision drift: {path}")
    return payload


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1F2 config schema")
    _runtime_contract(config)
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1F2 implementation hash drift: {name}")
    if file_sha256(config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise ValueError("Gate 3C1F2 manifest hash drift")
    data_parent = _validate_json_authority(
        config["authorized_parents"]["data_qualification"],
        decision_path=("formal_decision",),
    )
    entry_parent = _validate_json_authority(
        config["authorized_parents"]["entry_replay"],
        decision_path=("gate", "decision"),
    )
    top1_parent = _validate_json_authority(
        config["authorized_parents"]["top1_replay"],
        decision_path=("gate", "decision"),
    )
    future_parent = _validate_json_authority(
        config["authorized_parents"]["future_rollout_replay"],
        decision_path=("gate", "decision"),
    )
    if not bool(entry_parent.get("exact_replay")) or not bool(top1_parent.get("exact_replay")) or not bool(future_parent.get("exact_replay")):
        raise ValueError("Gate 3C1F2 parent exact replay missing")
    if data_parent["renewal_manifest_sha256"] != config["data"]["manifest_sha256"]:
        raise ValueError("Gate 3C1F2 data-parent manifest drift")
    if data_parent["selected_identity_digest"] != config["data"]["selected_identity_digest"]:
        raise ValueError("Gate 3C1F2 data identity digest drift")
    if entry_parent["scientific"]["operating_point"] != config["entry_contract"]["operating_point"]:
        raise ValueError("Gate 3C1F2 entry operating-point drift")
    if top1_parent["frozen_policy"] != config["top1_contract"]["frozen_policy"]:
        raise ValueError("Gate 3C1F2 top1 frozen-policy drift")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("Gate 3C1F2 CoTracker checkpoint drift")
    if file_sha256(config["dinov3"]["weights"]) != config["dinov3"]["weights_sha256"]:
        raise ValueError("Gate 3C1F2 DINOv3 weights drift")
    if any(bool(value) for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1F2 locked-data flags must remain false")
    for name in ("entry_bundle", "top1_bundle"):
        authority = config["bundles"][name]
        if file_sha256(authority["path"]) != authority["file_sha256"]:
            raise ValueError(f"Gate 3C1F2 bundle file drift: {name}")
    top1_config_path = Path(config["top1_contract"]["config"])
    if file_sha256(top1_config_path) != config["top1_contract"]["config_sha256"]:
        raise ValueError("Gate 3C1F2 top1 config drift")
    top1_config = yaml.safe_load(top1_config_path.read_text())
    return config, top1_config, {
        "entry": entry_parent,
        "top1": top1_parent,
        "future": future_parent,
        "data": data_parent,
    }


def _visible_probabilities(snapshot: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visibility = torch.sigmoid(snapshot.online_vis_predicted[0]).permute(1, 0).float().cpu()
    confidence = torch.sigmoid(snapshot.online_conf_predicted[0]).permute(1, 0).float().cpu()
    return visibility, confidence, visibility * confidence > 0.6


def _build_candidates(
    *,
    predictor: CoTrackerOnlinePredictor,
    dino: torch.nn.Module,
    video: torch.Tensor,
    prepared: Mapping[str, Any],
    initial: Any,
    point_indices: torch.Tensor,
    config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    rows = int(point_indices.numel())
    if rows <= 0:
        raise ValueError("Gate 3C1F2 candidate builder requires nonempty rows")
    observed = extract_cotracker_observed_feature_pyramid(predictor.model, video[:, :16])
    frame15 = [value[0, 15].contiguous() for value in observed]
    support = _selected_native_support(initial, point_indices)
    bank_cfg = config["candidate_bank"]
    score = geometry_preserving_pyramid_score(
        [value[None] for value in frame15],
        support,
        common_height=int(bank_cfg["common_height"]),
        common_width=int(bank_cfg["common_width"]),
        support_radius=int(bank_cfg["geometry_support_radius"]),
        trim_fraction=float(bank_cfg["geometry_token_trim_fraction"]),
    )["fused_score_map"]
    native_model = initial.online_coords_predicted[
        0, 15, point_indices.to(initial.online_coords_predicted.device)
    ]
    native_xy = model_xy_to_input_xy(
        native_model,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    )
    bank = extract_discrete_candidates(
        score,
        native_xy,
        top_k=int(bank_cfg["nonnative_top_k"]),
        nms_radius_grid_cells=int(bank_cfg["nms_radius_grid_cells"]),
        local_refinement_window_grid_cells=int(bank_cfg["local_refinement_window_grid_cells"]),
        local_softmax_temperature=float(bank_cfg["local_softmax_temperature"]),
        deduplicate_radius_input_px=float(bank_cfg["deduplicate_radius_input_px"]),
        input_height=256,
        input_width=256,
    )
    coordinates = bank["candidate_coordinates_xy"].detach().float()
    scores = bank["candidate_scores"].detach().float()
    valid = bank["candidate_valid_mask"].detach().bool()
    coordinates[~valid] = 0.0
    scores[~valid] = 0.0
    tracklets: list[torch.Tensor] = []
    visibility_rows: list[torch.Tensor] = []
    confidence_rows: list[torch.Tensor] = []
    for row in range(rows):
        valid_indices = torch.where(valid[row])[0]
        reverse_queries = torch.zeros(1, valid_indices.numel(), 3, device=device)
        reverse_queries[0, :, 1:3] = coordinates[row, valid_indices].to(device)
        reverse = _initialize_and_first_window(
            predictor, video[:, :16].flip(1), reverse_queries
        )
        reverse_tracks = _coords_to_input(
            reverse,
            interp_height=int(predictor.interp_shape[0]),
            interp_width=int(predictor.interp_shape[1]),
        ).flip(1)
        reverse_visibility = (
            torch.sigmoid(reverse.online_vis_predicted[0])
            .permute(1, 0)
            .flip(1)
            .float()
            .cpu()
        )
        reverse_confidence = (
            torch.sigmoid(reverse.online_conf_predicted[0])
            .permute(1, 0)
            .flip(1)
            .float()
            .cpu()
        )
        row_tracklet = torch.zeros(129, 16, 2)
        row_visibility = torch.zeros(129, 16)
        row_confidence = torch.zeros(129, 16)
        selected = valid_indices.cpu()
        row_tracklet[selected] = reverse_tracks
        row_visibility[selected] = reverse_visibility
        row_confidence[selected] = reverse_confidence
        tracklets.append(row_tracklet)
        visibility_rows.append(row_visibility)
        confidence_rows.append(row_confidence)
    feature_video = _dino_feature_video(dino, video, device)
    temporal: list[torch.Tensor] = []
    static: list[torch.Tensor] = []
    query_frames: list[int] = []
    for row, point_index in enumerate(point_indices.tolist()):
        descriptors = sample_spatiotemporal_descriptors(
            feature_video,
            tracklets[row].to(device),
            input_height=256,
            input_width=256,
        )
        query_tyx = prepared["query_points_tyx"][point_index]
        query_frame = int(round(float(query_tyx[0].item())))
        if query_frame >= 8:
            raise ValueError("Gate 3C1F2 entry emitted an ineligible query")
        query_xy = torch.stack([query_tyx[2], query_tyx[1]]).float() * 255.0
        query_track = query_xy[None, None].expand(1, 16, 2)
        query_sequence = sample_spatiotemporal_descriptors(
            feature_video,
            query_track.to(device),
            input_height=256,
            input_width=256,
        )
        features = assemble_candidate_temporal_identity_features(
            descriptor_sequence=descriptors,
            query_descriptor=query_sequence[0, query_frame],
            tracklets_xy=tracklets[row].to(device),
            visibility_probability=visibility_rows[row].to(device),
            confidence_probability=confidence_rows[row].to(device),
            candidate_scores=scores[row],
            valid_mask=valid[row],
            query_frame=query_frame,
            query_coordinate_xy=query_xy.to(device),
            input_height=256,
            input_width=256,
        )
        temporal.append(features["temporal_features"].detach().cpu().contiguous())
        static.append(features["static_features"].detach().cpu().contiguous())
        query_frames.append(query_frame)
    return {
        "coordinates": coordinates.cpu().contiguous(),
        "valid": valid.cpu().contiguous(),
        "temporal": torch.stack(temporal).contiguous(),
        "static": torch.stack(static).contiguous(),
        "query_frames": torch.tensor(query_frames, dtype=torch.long),
        "frame15_pyramid": [value[:, 15:16].contiguous() for value in observed],
    }


def _video_bootstrap(values: np.ndarray, *, seed: int, samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Gate 3C1F2 bootstrap requires nonempty video values")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = rng.choice(array, array.size, replace=True).mean()
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "videos": int(array.size),
        "samples": int(samples),
        "seed": int(seed),
    }


def _category_rows(
    *,
    prepared: Mapping[str, Any],
    native_xy: torch.Tensor,
    eligible: torch.Tensor,
) -> list[str]:
    gt_xy = prepared["gt_tracks_yx"][..., [1, 0]].float() * 255.0
    occluded = prepared["gt_occluded"].bool()
    future_visible = ~occluded[eligible, 16:24]
    error = torch.linalg.vector_norm(
        native_xy[eligible, 16:24] - gt_xy[eligible, 16:24], dim=-1
    )
    visible_count = future_visible.sum(dim=1)
    mean_error = (error * future_visible.float()).sum(dim=1) / visible_count.clamp_min(1)
    commit_visible = ~occluded[eligible, 15]
    evaluable = commit_visible & (visible_count >= 4)
    failure = evaluable & (mean_error >= 16.0)
    clean = evaluable & (mean_error <= 4.0)
    ambiguous = evaluable & (~failure) & (~clean)
    output = []
    for index in range(int(eligible.numel())):
        if bool(failure[index]):
            output.append("failure")
        elif bool(clean[index]):
            output.append("clean")
        elif bool(ambiguous[index]):
            output.append("ambiguous")
        else:
            output.append("other")
    return output


def _run_video(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    top1_config: Mapping[str, Any],
    entry_bundle: Mapping[str, Any],
    top1_bundle: Mapping[str, Any],
    predictor: CoTrackerOnlinePredictor,
    dino: torch.nn.Module,
    source_index: int,
    device: str,
) -> dict[str, Any]:
    sample, metadata = load_manifest_sample(Path(config["data"]["manifest"]), source_index)
    prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
    video = prepared["video"].to(device)
    set_deterministic(int(config["determinism"]["seed"]) + int(source_index))
    initial = _initialize_and_first_window(predictor, video, _original_queries(prepared, video))
    query_frames = prepared["query_points_tyx"][:, 0].round().long()
    eligible = torch.where(query_frames < int(config["entry_contract"]["query_frame_less_than"]))[0]
    trajectory = build_csrr_trajectory_features(initial, point_indices=eligible).float().cpu()
    visibility = torch.sigmoid(
        initial.online_vis_predicted[0, 15, eligible.to(initial.online_vis_predicted.device)]
    ).float().cpu()
    confidence = torch.sigmoid(
        initial.online_conf_predicted[0, 15, eligible.to(initial.online_conf_predicted.device)]
    ).float().cpu()
    native_features = _selected_memory(initial.online_track_feat, eligible, support=False)
    native_supports = _selected_memory(initial.online_track_support, eligible, support=True)
    entry_features = build_entry_features(
        trajectory_features=trajectory,
        visibility_probability=visibility,
        confidence_probability=confidence,
        native_track_features=native_features,
        native_track_supports=native_supports,
    )
    if entry_bundle.get("feature_schema_version") != ENTRY_FEATURE_SCHEMA_VERSION:
        raise ValueError("Gate 3C1F2 entry feature schema drift")
    entry_probability = entry_bundle["model"].predict_proba(entry_features)[:, 1]
    operating = config["entry_contract"]["operating_point"]
    entry_mask = entry_action_mask(
        entry_probability=entry_probability,
        native_joint_probability=(visibility * confidence).numpy(),
        probability_min=float(operating["entry_probability_min"]),
        joint_probability_max=float(operating["native_joint_probability_max"]),
    )
    trigger_indices = eligible[torch.from_numpy(entry_mask)]
    top1_action = np.zeros(int(trigger_indices.numel()), dtype=bool)
    output_candidate = np.zeros(int(trigger_indices.numel()), dtype=np.int64)
    selected_slot = np.zeros(int(trigger_indices.numel()), dtype=np.int64)
    selected_support = np.zeros(int(trigger_indices.numel()), dtype=np.float64)
    predicted_value = np.zeros(int(trigger_indices.numel()), dtype=np.float64)
    predicted_harm = np.zeros(int(trigger_indices.numel()), dtype=np.float64)
    candidate_digests: dict[str, Any] = {
        "coordinates": canonical_json_sha256([]),
        "temporal": canonical_json_sha256([]),
        "static": canonical_json_sha256([]),
        "shortlist": canonical_json_sha256([]),
    }
    if trigger_indices.numel() > 0:
        candidates = _build_candidates(
            predictor=predictor,
            dino=dino,
            video=video,
            prepared=prepared,
            initial=initial,
            point_indices=trigger_indices,
            config=config,
            device=device,
        )
        causal = build_shortlist_top1_features(
            temporal_features=candidates["temporal"].float(),
            static_features=candidates["static"].float(),
            query_frames=candidates["query_frames"],
            valid_mask=candidates["valid"],
            selector_config=_selector_config(),
            feature_config=_feature_config(top1_config),
        )
        predictions = _predict_bundle(
            top1_bundle, {"features": causal["candidate_features"].cpu()}
        )
        selected_slot = predictions["selected"].astype(np.int64)
        row = np.arange(len(selected_slot))
        selected_support = predictions["support"][row, selected_slot].astype(np.float64)
        predicted_value = predictions["value"].astype(np.float64)
        predicted_harm = predictions["harm"].astype(np.float64)
        policy = config["top1_contract"]["frozen_policy"]
        top1_action = (
            (selected_slot != 0)
            & (selected_support >= float(policy["support_min"]))
            & (predicted_value >= float(policy["value_min_px"]))
            & (predicted_harm <= float(policy["harm_max"]))
        )
        output_slot = np.where(top1_action, selected_slot, 0)
        shortlist = causal["selected_indices"].numpy()
        output_candidate = shortlist[row, output_slot].astype(np.int64)
        output_xy = candidates["coordinates"][
            torch.arange(len(output_candidate)), torch.from_numpy(output_candidate)
        ].to(device)
        re_feat_batch, re_support_batch = deterministic_reextract_cotracker_memory(
            candidates["frame15_pyramid"],
            output_xy,
            input_height=256,
            input_width=256,
            model_height=int(predictor.interp_shape[0]),
            model_width=int(predictor.interp_shape[1]),
            stride=int(config["backbone"]["model_stride"]),
            support_radius=int(config["backbone"]["support_radius"]),
        )
        re_feat, re_support = _native_format_memory(re_feat_batch, re_support_batch)
        modified_initial = apply_reextracted_state_action(
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
        candidate_digests = {
            "coordinates": tensor_sha256(candidates["coordinates"]),
            "temporal": tensor_sha256(candidates["temporal"]),
            "static": tensor_sha256(candidates["static"]),
            "shortlist": tensor_sha256(causal["selected_indices"]),
            "candidate_features": tensor_sha256(causal["candidate_features"]),
        }
    else:
        modified_initial = initial
    native_final = _continue_second_window(predictor, video, initial)
    modified_final = _continue_second_window(predictor, video, modified_initial)
    native_xy = _coords_to_input(
        native_final,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    )
    modified_xy = _coords_to_input(
        modified_final,
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
    )
    _, _, native_visible = _visible_probabilities(native_final)
    _, _, modified_visible = _visible_probabilities(modified_final)
    native_metrics = _metric_dict(
        compute_tapvid_metrics(
            _tracks_from_xy(native_xy, 256),
            prepared["gt_tracks_yx"],
            native_visible,
            ~prepared["gt_occluded"],
            prepared["query_points_tyx"],
            resolution=256,
            query_mode="first",
        )
    )
    modified_metrics = _metric_dict(
        compute_tapvid_metrics(
            _tracks_from_xy(modified_xy, 256),
            prepared["gt_tracks_yx"],
            modified_visible,
            ~prepared["gt_occluded"],
            prepared["query_points_tyx"],
            resolution=256,
            query_mode="first",
        )
    )
    categories = _category_rows(prepared=prepared, native_xy=native_xy, eligible=eligible)
    trigger_positions = np.where(entry_mask)[0]
    trigger_categories = [categories[int(position)] for position in trigger_positions]
    action_positions = np.where(top1_action)[0]
    action_categories = [trigger_categories[int(position)] for position in action_positions]
    action_points = (
        trigger_indices[torch.from_numpy(top1_action)]
        if trigger_indices.numel()
        else torch.empty(0, dtype=torch.long)
    )
    gt_xy = prepared["gt_tracks_yx"][..., [1, 0]].float() * 255.0
    occluded = prepared["gt_occluded"].bool()
    action_records: list[dict[str, Any]] = []
    future_reductions: list[float] = []
    no_future_visible = 0
    for local, point_index in zip(action_positions.tolist(), action_points.tolist()):
        future_mask = ~occluded[point_index, 16:24]
        reduction: float | None
        if bool(future_mask.any()):
            native_error = torch.linalg.vector_norm(
                native_xy[point_index, 16:24][future_mask]
                - gt_xy[point_index, 16:24][future_mask],
                dim=-1,
            ).mean()
            modified_error = torch.linalg.vector_norm(
                modified_xy[point_index, 16:24][future_mask]
                - gt_xy[point_index, 16:24][future_mask],
                dim=-1,
            ).mean()
            reduction = float(native_error - modified_error)
            future_reductions.append(reduction)
        else:
            reduction = None
            no_future_visible += 1
        action_records.append(
            {
                "point_index": int(point_index),
                "category": trigger_categories[int(local)],
                "entry_probability": float(entry_probability[trigger_positions[int(local)]]),
                "native_joint_probability": float(
                    (visibility * confidence)[trigger_positions[int(local)]].item()
                ),
                "selected_slot": int(selected_slot[int(local)]),
                "output_candidate_index": int(output_candidate[int(local)]),
                "selected_support_probability": float(selected_support[int(local)]),
                "predicted_value_px": float(predicted_value[int(local)]),
                "predicted_harm_probability": float(predicted_harm[int(local)]),
                "future_error_reduction_px": reduction,
            }
        )
    gains = {
        "AJ": float(modified_metrics["AJ"] - native_metrics["AJ"]),
        "delta_avg": float(modified_metrics["<avg"] - native_metrics["<avg"]),
        "OA": float(modified_metrics["OA"] - native_metrics["OA"]),
    }
    scientific = {
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "eligible_rows": int(eligible.numel()),
        "entry_trigger_rows": int(entry_mask.sum()),
        "top1_action_rows": int(top1_action.sum()),
        "entry_category_counts": dict(Counter(trigger_categories)),
        "action_category_counts": dict(Counter(action_categories)),
        "native_metrics": native_metrics,
        "modified_metrics": modified_metrics,
        "gains": gains,
        "evaluable_action_rows": len(future_reductions),
        "no_future_visible_action_rows": int(no_future_visible),
        "action_future_reductions_px": future_reductions,
        "digests": {
            "eligible_indices": tensor_sha256(eligible.contiguous()),
            "entry_features": tensor_sha256(torch.from_numpy(entry_features)),
            "entry_probability": tensor_sha256(torch.from_numpy(entry_probability)),
            "entry_mask": tensor_sha256(torch.from_numpy(entry_mask)),
            **candidate_digests,
            "selected_slot": tensor_sha256(torch.from_numpy(selected_slot)),
            "top1_action": tensor_sha256(torch.from_numpy(top1_action)),
            "output_candidate": tensor_sha256(torch.from_numpy(output_candidate)),
            "native_coordinates": tensor_sha256(native_xy.contiguous()),
            "modified_coordinates": tensor_sha256(modified_xy.contiguous()),
            "native_visibility": tensor_sha256(native_visible.contiguous()),
            "modified_visibility": tensor_sha256(modified_visible.contiguous()),
            "action_records": canonical_json_sha256(action_records),
        },
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    result = {
        "schema_version": VIDEO_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "sample_metadata": metadata,
        "scientific": scientific,
        "action_records": action_records,
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    del video
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _aggregate(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    video_records: Sequence[Mapping[str, Any]],
    reference_path: Path | None,
) -> dict[str, Any]:
    expected_videos = int(config["data"]["expected_videos"])
    if len(video_records) != expected_videos:
        raise ValueError("Gate 3C1F2 completed video count drift")
    metric_keys = {"AJ": "AJ", "delta_avg": "<avg", "OA": "OA"}
    mean_native: dict[str, float] = {}
    mean_modified: dict[str, float] = {}
    mean_gain: dict[str, float] = {}
    intervals: dict[str, Any] = {}
    nonnegative: dict[str, float] = {}
    positive: dict[str, int] = {}
    negative: dict[str, int] = {}
    for ordinal, (name, key) in enumerate(metric_keys.items()):
        native = np.array(
            [row["scientific"]["native_metrics"][key] for row in video_records],
            dtype=np.float64,
        )
        modified = np.array(
            [row["scientific"]["modified_metrics"][key] for row in video_records],
            dtype=np.float64,
        )
        gain = modified - native
        mean_native[name] = float(native.mean())
        mean_modified[name] = float(modified.mean())
        mean_gain[name] = float(gain.mean())
        intervals[name] = _video_bootstrap(
            gain,
            seed=int(config["metrics"]["bootstrap_seed"]) + ordinal,
            samples=int(config["metrics"]["bootstrap_samples"]),
        )
        nonnegative[name] = float((gain >= 0.0).mean())
        positive[name] = int((gain > 0.0).sum())
        negative[name] = int((gain < 0.0).sum())
    future_reductions = np.array(
        [
            value
            for row in video_records
            for value in row["scientific"]["action_future_reductions_px"]
        ],
        dtype=np.float64,
    )
    entry_categories = sum(
        (Counter(row["scientific"]["entry_category_counts"]) for row in video_records),
        Counter(),
    )
    action_categories = sum(
        (Counter(row["scientific"]["action_category_counts"]) for row in video_records),
        Counter(),
    )
    metrics = {
        "videos": len(video_records),
        "eligible_rows": int(sum(row["scientific"]["eligible_rows"] for row in video_records)),
        "entry_trigger_rows": int(
            sum(row["scientific"]["entry_trigger_rows"] for row in video_records)
        ),
        "top1_action_rows": int(
            sum(row["scientific"]["top1_action_rows"] for row in video_records)
        ),
        "action_videos": int(
            sum(row["scientific"]["top1_action_rows"] > 0 for row in video_records)
        ),
        "evaluable_action_rows": int(future_reductions.size),
        "no_future_visible_action_rows": int(
            sum(
                row["scientific"]["no_future_visible_action_rows"]
                for row in video_records
            )
        ),
        "entry_category_counts": dict(entry_categories),
        "action_category_counts": dict(action_categories),
        "native_mean": mean_native,
        "modified_mean": mean_modified,
        "mean_gain": mean_gain,
        "paired_video_CI": intervals,
        "nonnegative_video_fraction": nonnegative,
        "positive_videos": positive,
        "negative_videos": negative,
        "action_future_mean_reduction_px": (
            0.0 if future_reductions.size == 0 else float(future_reductions.mean())
        ),
        "action_future_positive_fraction": (
            0.0 if future_reductions.size == 0 else float((future_reductions > 0.0).mean())
        ),
        "action_future_harm_gt4_fraction": (
            0.0 if future_reductions.size == 0 else float((future_reductions < -4.0).mean())
        ),
    }
    scientific = {
        "data_manifest_sha256": config["data"]["manifest_sha256"],
        "selected_identity_digest": config["data"]["selected_identity_digest"],
        "entry_operating_point": config["entry_contract"]["operating_point"],
        "top1_policy": config["top1_contract"]["frozen_policy"],
        "metrics": metrics,
        "video_scientific_digests": [
            row["scientific"]["scientific_payload_sha256"] for row in video_records
        ],
        "video_result_digests": [row["result_payload_sha256"] for row in video_records],
        "video_records_digest": canonical_json_sha256(
            [row["scientific"] for row in video_records]
        ),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    reference = None
    replay_comparison = None
    exact_replay = False
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        reference_scientific = reference["scientific"]
        replay_comparison = {
            "video_scientific_digests": scientific["video_scientific_digests"]
            == reference_scientific["video_scientific_digests"],
            "video_records_digest": scientific["video_records_digest"]
            == reference_scientific["video_records_digest"],
            "metrics": scientific["metrics"] == reference_scientific["metrics"],
            "scientific_payload_sha256": scientific["scientific_payload_sha256"]
            == reference_scientific["scientific_payload_sha256"],
        }
        exact_replay = all(replay_comparison.values())
    gates = config["pass_gates"]
    checks = {
        "videos_complete": metrics["videos"] == int(gates["videos_exact"]),
        "entry_support": metrics["entry_trigger_rows"] >= int(gates["entry_trigger_rows_min"]),
        "action_support": metrics["top1_action_rows"] >= int(gates["top1_action_rows_min"]),
        "action_video_support": metrics["action_videos"] >= int(gates["action_videos_min"]),
        "evaluable_action_support": metrics["evaluable_action_rows"]
        >= int(gates["evaluable_action_rows_min"]),
        "AJ_gain": metrics["mean_gain"]["AJ"] >= float(gates["AJ_gain_min"]),
        "AJ_CI_lower": metrics["paired_video_CI"]["AJ"]["lower"]
        > float(gates["AJ_CI_lower_min"]),
        "delta_avg_gain": metrics["mean_gain"]["delta_avg"]
        >= float(gates["delta_avg_gain_min"]),
        "delta_avg_CI_lower": metrics["paired_video_CI"]["delta_avg"]["lower"]
        > float(gates["delta_avg_CI_lower_min"]),
        "OA_gain": metrics["mean_gain"]["OA"] >= float(gates["OA_gain_min"]),
        "OA_CI_lower": metrics["paired_video_CI"]["OA"]["lower"]
        > float(gates["OA_CI_lower_min"]),
        "AJ_nonnegative_videos": metrics["nonnegative_video_fraction"]["AJ"]
        >= float(gates["AJ_nonnegative_video_fraction_min"]),
        "action_future_reduction": metrics["action_future_mean_reduction_px"]
        >= float(gates["action_future_mean_reduction_px_min"]),
        "action_future_positive": metrics["action_future_positive_fraction"]
        >= float(gates["action_future_positive_fraction_min"]),
        "action_future_harm": metrics["action_future_harm_gt4_fraction"]
        <= float(gates["action_future_harm_gt4_fraction_max"]),
        "exact_replay": exact_replay,
    }
    if reference_path is None:
        passed = False
        decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    else:
        passed = all(checks.values())
        decision = gates["decision_pass" if passed else "decision_fail"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "scientific": scientific,
        "video_records": list(video_records),
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": replay_comparison,
        "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision},
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    return result


def evaluate(
    *,
    config_path: Path,
    output_path: Path,
    work_root: Path,
    reference_path: Path | None,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    config, top1_config, _ = _validate_config(config_path)
    set_deterministic(int(config["determinism"]["seed"]))
    entry_bundle = joblib.load(config["bundles"]["entry_bundle"]["path"])
    top1_bundle = joblib.load(config["bundles"]["top1_bundle"]["path"])
    if entry_bundle.get("schema_version") != config["bundles"]["entry_bundle"]["schema_version"]:
        raise ValueError("Gate 3C1F2 entry bundle schema drift")
    if top1_bundle.get("schema_version") != config["bundles"]["top1_bundle"]["schema_version"]:
        raise ValueError("Gate 3C1F2 top1 bundle schema drift")
    predictor = CoTrackerOnlinePredictor(checkpoint=config["backbone"]["checkpoint"]).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    dino = AutoModel.from_pretrained(
        config["dinov3"]["model_dir"], local_files_only=True
    ).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    work_root.mkdir(parents=True, exist_ok=True)
    source_start, source_end = config["data"]["source_indices"]
    records: list[dict[str, Any]] = []
    for source_index in range(int(source_start), int(source_end) + 1):
        sidecar = work_root / f"video_{source_index:05d}.json"
        if resume and sidecar.exists():
            record = json.loads(sidecar.read_text())
            if (
                record.get("schema_version") != VIDEO_SCHEMA
                or record.get("config_sha256") != file_sha256(config_path)
                or int(record["scientific"]["source_index"]) != source_index
            ):
                raise ValueError("Gate 3C1F2 resume sidecar drift")
        else:
            record = _run_video(
                config=config,
                config_path=config_path,
                top1_config=top1_config,
                entry_bundle=entry_bundle,
                top1_bundle=top1_bundle,
                predictor=predictor,
                dino=dino,
                source_index=source_index,
                device=device,
            )
            _atomic_json_save(record, sidecar)
        records.append(record)
        print(
            json.dumps(
                {
                    "stage": "video_complete",
                    "source_index": source_index,
                    "video_name": record["scientific"]["video_name"],
                    "eligible_rows": record["scientific"]["eligible_rows"],
                    "entry_trigger_rows": record["scientific"]["entry_trigger_rows"],
                    "top1_action_rows": record["scientific"]["top1_action_rows"],
                    "gains": record["scientific"]["gains"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    result = _aggregate(
        config=config,
        config_path=config_path,
        video_records=records,
        reference_path=reference_path,
    )
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        config_path=Path(args.config).resolve(),
        output_path=Path(args.output).resolve(),
        work_root=Path(args.work_root).resolve(),
        reference_path=None if args.reference is None else Path(args.reference).resolve(),
        device=str(args.device),
        resume=bool(args.resume),
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
