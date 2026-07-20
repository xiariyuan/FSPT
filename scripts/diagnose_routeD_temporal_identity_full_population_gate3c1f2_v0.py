#!/usr/bin/env python3
"""Diagnose coordinate/visibility coupling after frozen Gate 3C1F2 failure."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from scripts import run_routeD_temporal_identity_full_population_gate3c1f2_v0 as formal

SCHEMA = "routeD_temporal_identity_full_population_failure_diagnostic_gate3c1f2_v0"
RESULT_SCHEMA = "routeD_temporal_identity_full_population_failure_diagnostic_result_gate3c1f2_v0"
VIDEO_SCHEMA = "routeD_temporal_identity_full_population_failure_diagnostic_video_gate3c1f2_v0"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_temporal_identity_full_population_gate3c1f2_failure_diagnostic_v0.yaml"
THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _validate_parent(authority: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(authority["path"])
    if file_sha256(path) != authority["file_sha256"]:
        raise ValueError("Gate 3C1F2 diagnostic parent file drift")
    payload = json.loads(path.read_text())
    without_hash = dict(payload)
    embedded = without_hash.pop("result_payload_sha256", None)
    if embedded != authority["payload_sha256"] or embedded != canonical_json_sha256(without_hash):
        raise ValueError("Gate 3C1F2 diagnostic parent payload drift")
    if (
        payload.get("exact_replay") is not True
        or payload.get("gate", {}).get("decision") != authority["required_decision"]
        or payload.get("scientific", {}).get("scientific_payload_sha256")
        != authority["scientific_payload_sha256"]
    ):
        raise ValueError("Gate 3C1F2 diagnostic parent status drift")
    return payload


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1F2 diagnostic config")
    parent = _validate_parent(config["authorized_parent"])
    formal_path = Path(config["formal_config"]["path"])
    if file_sha256(formal_path) != config["formal_config"]["sha256"]:
        raise ValueError("Gate 3C1F2 diagnostic formal config drift")
    formal_config, top1_config, _ = formal._validate_config(formal_path)
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1F2 diagnostic implementation drift: {name}")
    if any(bool(value) for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1F2 diagnostic locked-data flags must remain false")
    reference = {
        int(row["scientific"]["source_index"]): row for row in parent["video_records"]
    }
    actual_sources = sorted(
        source_index
        for source_index, row in reference.items()
        if int(row["scientific"]["top1_action_rows"]) > 0
    )
    expected_sources = [int(value) for value in config["action_source_indices"]]
    if actual_sources != expected_sources:
        raise ValueError("Gate 3C1F2 diagnostic action-video membership drift")
    return config, {"formal": formal_config, "top1": top1_config}, reference


def _metric_view(
    coordinates_xy: torch.Tensor,
    visibility: torch.Tensor,
    prepared: Mapping[str, Any],
) -> dict[str, float]:
    return formal._metric_dict(
        formal.compute_tapvid_metrics(
            formal._tracks_from_xy(coordinates_xy, 256),
            prepared["gt_tracks_yx"],
            visibility,
            ~prepared["gt_occluded"],
            prepared["query_points_tyx"],
            resolution=256,
            query_mode="first",
        )
    )


def jaccard_components(
    *,
    gt_visible: bool,
    predicted_visible: bool,
    error_px: float,
    threshold_px: float,
) -> dict[str, int]:
    within = float(error_px) < float(threshold_px)
    correct = bool(gt_visible) and within
    true_positive = correct and bool(predicted_visible)
    false_positive = bool(predicted_visible) and (
        (not bool(gt_visible)) or (not within)
    )
    return {
        "gt_positive": int(bool(gt_visible)),
        "point_correct": int(correct),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
    }


def _decision_projection(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "point_index",
        "entry_probability",
        "native_joint_probability",
        "selected_slot",
        "output_candidate_index",
        "selected_support_probability",
        "predicted_value_px",
        "predicted_harm_probability",
    )
    return [{key: row[key] for key in keys} for row in records]


def _run_video(
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
    query_frames = prepared["query_points_tyx"][:, 0].round().long()
    eligible = torch.where(
        query_frames < int(formal_config["entry_contract"]["query_frame_less_than"])
    )[0]
    trajectory = formal.build_csrr_trajectory_features(
        initial, point_indices=eligible
    ).float().cpu()
    visibility = torch.sigmoid(
        initial.online_vis_predicted[
            0, 15, eligible.to(initial.online_vis_predicted.device)
        ]
    ).float().cpu()
    confidence = torch.sigmoid(
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
        visibility_probability=visibility,
        confidence_probability=confidence,
        native_track_features=native_features,
        native_track_supports=native_supports,
    )
    entry_probability = entry_bundle["model"].predict_proba(entry_features)[:, 1]
    operating = formal_config["entry_contract"]["operating_point"]
    entry_mask = formal.entry_action_mask(
        entry_probability=entry_probability,
        native_joint_probability=(visibility * confidence).numpy(),
        probability_min=float(operating["entry_probability_min"]),
        joint_probability_max=float(operating["native_joint_probability_max"]),
    )
    trigger_indices = eligible[torch.from_numpy(entry_mask)]
    if trigger_indices.numel() == 0:
        raise ValueError("Gate 3C1F2 diagnostic action video lost all entry rows")
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
    if not bool(top1_action.any()):
        raise ValueError("Gate 3C1F2 diagnostic action video lost all top1 actions")
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
        raise ValueError(f"Gate 3C1F2 diagnostic sealed digest drift: {exact_checks}")
    categories = formal._category_rows(
        prepared=prepared, native_xy=native_xy, eligible=eligible
    )
    trigger_positions = np.where(entry_mask)[0]
    trigger_categories = [categories[int(position)] for position in trigger_positions]
    action_positions = np.where(top1_action)[0]
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
                    (visibility * confidence)[trigger_positions[int(local)]].item()
                ),
                "selected_slot": int(selected_slot[int(local)]),
                "output_candidate_index": int(output_candidate[int(local)]),
                "selected_support_probability": float(selected_support[int(local)]),
                "predicted_value_px": float(predicted_value[int(local)]),
                "predicted_harm_probability": float(predicted_harm[int(local)]),
            }
        )
    decision_exact = _decision_projection(reference["action_records"]) == decision_records
    if not decision_exact:
        raise ValueError("Gate 3C1F2 diagnostic action decision drift")
    gt_visible = ~prepared["gt_occluded"]
    views = {
        "native": (native_xy, native_visible),
        "actual_modified": (modified_xy, modified_visible),
        "modified_coordinates_native_visibility": (modified_xy, native_visible),
        "native_coordinates_modified_visibility": (native_xy, modified_visible),
        "native_coordinates_gt_visibility_oracle": (native_xy, gt_visible),
        "modified_coordinates_gt_visibility_oracle": (modified_xy, gt_visible),
    }
    view_metrics = {
        name: _metric_view(coordinates, predicted_visibility, prepared)
        for name, (coordinates, predicted_visibility) in views.items()
    }
    if (
        view_metrics["native"] != ref_scientific["native_metrics"]
        or view_metrics["actual_modified"] != ref_scientific["modified_metrics"]
    ):
        raise ValueError("Gate 3C1F2 diagnostic formal metric drift")
    gt_xy = prepared["gt_tracks_yx"][..., [1, 0]].float() * 255.0
    frame_records: list[dict[str, Any]] = []
    for action_ordinal, (local, point_index) in enumerate(
        zip(action_positions.tolist(), action_points.tolist())
    ):
        category = trigger_categories[int(local)]
        for frame in range(
            int(config["affected_frames"][0]), int(config["affected_frames"][1]) + 1
        ):
            visible = bool(gt_visible[point_index, frame])
            native_error = float(
                torch.linalg.vector_norm(
                    native_xy[point_index, frame] - gt_xy[point_index, frame]
                )
            )
            modified_error = float(
                torch.linalg.vector_norm(
                    modified_xy[point_index, frame] - gt_xy[point_index, frame]
                )
            )
            native_binary = bool(native_visible[point_index, frame])
            modified_binary = bool(modified_visible[point_index, frame])
            components: dict[str, Any] = {}
            for view_name, (coordinates, predicted_visibility) in views.items():
                error = float(
                    torch.linalg.vector_norm(
                        coordinates[point_index, frame] - gt_xy[point_index, frame]
                    )
                )
                components[view_name] = {
                    str(int(threshold)): jaccard_components(
                        gt_visible=visible,
                        predicted_visible=bool(predicted_visibility[point_index, frame]),
                        error_px=error,
                        threshold_px=threshold,
                    )
                    for threshold in THRESHOLDS
                }
            frame_records.append(
                {
                    "source_index": int(source_index),
                    "video_name": str(prepared["video_name"]),
                    "video_AJ_sign": (
                        "positive"
                        if ref_scientific["gains"]["AJ"] > 0.0
                        else "negative"
                        if ref_scientific["gains"]["AJ"] < 0.0
                        else "zero"
                    ),
                    "action_ordinal": int(action_ordinal),
                    "point_index": int(point_index),
                    "category": category,
                    "frame": int(frame),
                    "gt_visible": visible,
                    "native_visibility_probability": float(
                        native_vis_prob[point_index, frame]
                    ),
                    "native_confidence_probability": float(
                        native_conf_prob[point_index, frame]
                    ),
                    "native_joint_probability": float(
                        native_vis_prob[point_index, frame]
                        * native_conf_prob[point_index, frame]
                    ),
                    "native_visible": native_binary,
                    "modified_visibility_probability": float(
                        modified_vis_prob[point_index, frame]
                    ),
                    "modified_confidence_probability": float(
                        modified_conf_prob[point_index, frame]
                    ),
                    "modified_joint_probability": float(
                        modified_vis_prob[point_index, frame]
                        * modified_conf_prob[point_index, frame]
                    ),
                    "modified_visible": modified_binary,
                    "native_error_px": native_error,
                    "modified_error_px": modified_error,
                    "error_reduction_px": native_error - modified_error,
                    "recovered_visible_false_negative": visible
                    and (not native_binary)
                    and modified_binary,
                    "new_visible_false_negative": visible
                    and native_binary
                    and (not modified_binary),
                    "removed_occluded_false_positive": (not visible)
                    and native_binary
                    and (not modified_binary),
                    "new_occluded_false_positive": (not visible)
                    and (not native_binary)
                    and modified_binary,
                    "jaccard_components": components,
                }
            )
    scientific = {
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "sealed_digest_checks": exact_checks,
        "action_decisions_exact": decision_exact,
        "action_rows": int(top1_action.sum()),
        "view_metrics": view_metrics,
        "frame_records_digest": canonical_json_sha256(frame_records),
        "frame_records": frame_records,
        "formal_video_scientific_sha256": ref_scientific[
            "scientific_payload_sha256"
        ],
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    result = {
        "schema_version": VIDEO_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "source_index": int(source_index),
        "sample_metadata": metadata,
        "scientific": scientific,
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    del video
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _paired_ci(values: np.ndarray, *, seed: int, samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
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


def _aggregate(
    *, config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    views = list(config["trajectory_views"])
    metric_alias = {"AJ": "AJ", "delta_avg": "<avg", "OA": "OA"}
    equal_video_means: dict[str, dict[str, float]] = {}
    for view in views:
        equal_video_means[view] = {
            name: float(
                np.mean(
                    [
                        row["scientific"]["view_metrics"][view][metric]
                        for row in records
                    ]
                )
            )
            for name, metric in metric_alias.items()
        }
    comparisons = {
        "actual_modified_minus_native": ("actual_modified", "native"),
        "modified_coordinates_native_visibility_minus_native": (
            "modified_coordinates_native_visibility",
            "native",
        ),
        "native_coordinates_modified_visibility_minus_native": (
            "native_coordinates_modified_visibility",
            "native",
        ),
        "modified_minus_native_coordinates_under_gt_visibility": (
            "modified_coordinates_gt_visibility_oracle",
            "native_coordinates_gt_visibility_oracle",
        ),
    }
    paired: dict[str, Any] = {}
    for comparison_ordinal, (name, (left, right)) in enumerate(comparisons.items()):
        paired[name] = {}
        for metric_ordinal, (metric_name, metric_key) in enumerate(metric_alias.items()):
            values = np.array(
                [
                    row["scientific"]["view_metrics"][left][metric_key]
                    - row["scientific"]["view_metrics"][right][metric_key]
                    for row in records
                ],
                dtype=np.float64,
            )
            paired[name][metric_name] = _paired_ci(
                values,
                seed=int(config["bootstrap"]["seed"])
                + comparison_ordinal * 10
                + metric_ordinal,
                samples=int(config["bootstrap"]["samples"]),
            )
    all_frames = [
        frame
        for row in records
        for frame in row["scientific"]["frame_records"]
    ]
    transitions = Counter()
    transitions_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    coordinate_hits: dict[str, Counter[str]] = defaultdict(Counter)
    pooled_components: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for frame in all_frames:
        category = str(frame["category"])
        for key in (
            "recovered_visible_false_negative",
            "new_visible_false_negative",
            "removed_occluded_false_positive",
            "new_occluded_false_positive",
        ):
            if bool(frame[key]):
                transitions[key] += 1
                transitions_by_category[category][key] += 1
        if bool(frame["gt_visible"]):
            for threshold in THRESHOLDS:
                token = str(int(threshold))
                if float(frame["native_error_px"]) < threshold:
                    coordinate_hits[category][f"native_{token}"] += 1
                if float(frame["modified_error_px"]) < threshold:
                    coordinate_hits[category][f"modified_{token}"] += 1
            coordinate_hits[category]["visible_frames"] += 1
        for view, threshold_rows in frame["jaccard_components"].items():
            for threshold, components in threshold_rows.items():
                pooled_components[category][view][threshold].update(components)
                pooled_components["all"][view][threshold].update(components)
    pooled_jaccard: dict[str, Any] = {}
    for category, view_rows in pooled_components.items():
        pooled_jaccard[category] = {}
        for view, threshold_rows in view_rows.items():
            per_threshold: dict[str, Any] = {}
            values = []
            for threshold, counts in threshold_rows.items():
                denominator = counts["gt_positive"] + counts["false_positive"]
                jaccard = 0.0 if denominator == 0 else counts["true_positive"] / denominator
                point_accuracy = (
                    0.0
                    if counts["gt_positive"] == 0
                    else counts["point_correct"] / counts["gt_positive"]
                )
                per_threshold[threshold] = {
                    **dict(counts),
                    "jaccard": float(jaccard),
                    "point_accuracy": float(point_accuracy),
                }
                values.append(jaccard)
            pooled_jaccard[category][view] = {
                "thresholds": per_threshold,
                "average_jaccard": float(np.mean(values)) if values else 0.0,
            }
    exact = all(
        all(row["scientific"]["sealed_digest_checks"].values())
        and bool(row["scientific"]["action_decisions_exact"])
        for row in records
    )
    scientific = {
        "videos": len(records),
        "actions": int(sum(row["scientific"]["action_rows"] for row in records)),
        "sealed_pipeline_exact": exact,
        "equal_video_means": equal_video_means,
        "paired_comparisons": paired,
        "visibility_transitions": dict(transitions),
        "visibility_transitions_by_category": {
            key: dict(value) for key, value in transitions_by_category.items()
        },
        "coordinate_hits_by_category": {
            key: dict(value) for key, value in coordinate_hits.items()
        },
        "pooled_affected_frame_jaccard": pooled_jaccard,
        "video_scientific_digests": [
            row["scientific"]["scientific_payload_sha256"] for row in records
        ],
        "frame_records_digest": canonical_json_sha256(
            [frame for row in records for frame in row["scientific"]["frame_records"]]
        ),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "scientific": scientific,
        "video_records": list(records),
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
    device: str,
    resume: bool,
) -> dict[str, Any]:
    config, loaded, reference = _validate_config(config_path)
    formal_config = loaded["formal"]
    top1_config = loaded["top1"]
    formal.set_deterministic(int(formal_config["determinism"]["seed"]))
    entry_bundle = joblib.load(formal_config["bundles"]["entry_bundle"]["path"])
    top1_bundle = joblib.load(formal_config["bundles"]["top1_bundle"]["path"])
    predictor = CoTrackerOnlinePredictor(
        checkpoint=formal_config["backbone"]["checkpoint"]
    ).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    dino = AutoModel.from_pretrained(
        formal_config["dinov3"]["model_dir"], local_files_only=True
    ).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    work_root.mkdir(parents=True, exist_ok=True)
    records = []
    for source_index in config["action_source_indices"]:
        source_index = int(source_index)
        sidecar = work_root / f"video_{source_index:05d}.json"
        if resume and sidecar.exists():
            record = json.loads(sidecar.read_text())
            if (
                record.get("schema_version") != VIDEO_SCHEMA
                or int(record.get("source_index", -1)) != source_index
            ):
                raise ValueError("Gate 3C1F2 diagnostic resume sidecar drift")
        else:
            record = _run_video(
                config=config,
                formal_config=formal_config,
                top1_config=top1_config,
                reference=reference[source_index],
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
                    "action_rows": record["scientific"]["action_rows"],
                    "sealed_exact": all(
                        record["scientific"]["sealed_digest_checks"].values()
                    )
                    and record["scientific"]["action_decisions_exact"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    result = _aggregate(config=config, records=records)
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        config_path=Path(args.config).resolve(),
        output_path=Path(args.output).resolve(),
        work_root=Path(args.work_root).resolve(),
        device=str(args.device),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scientific": result["scientific"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
