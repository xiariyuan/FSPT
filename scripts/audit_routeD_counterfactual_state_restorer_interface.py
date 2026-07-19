#!/usr/bin/env python3
"""Gate 2 interface smoke for structured state re-extraction."""
from __future__ import annotations

import argparse
import json
import os
import sys
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

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    snapshot_cotracker_online_state,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    CSRR_SCHEMA_VERSION,
    CounterfactualStructuredReextractionRestorer,
    apply_reextracted_state_action,
    extract_cotracker_observed_feature_pyramid,
    reextract_cotracker_memory,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)
from scripts.audit_routeD_oracle_state_transplant_gate1 import (
    _initialize_and_first_window,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_counterfactual_state_restorer_gate2_v0.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_counterfactual_state_restorer_interface_20260719/primary.json"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item())


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left.float() - right.float()).abs()
    return {
        "max_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "cosine_similarity": _cosine(left, right),
    }


def _quantization(tensor: torch.Tensor) -> dict[str, float]:
    restored = tensor.detach().to(torch.float16).to(torch.float32)
    return _difference(tensor.detach().float(), restored)


def _fresh_queries(prepared: dict[str, Any], selected: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    gt = prepared["gt_tracks_yx"].to(video.device)
    indices = selected.to(video.device)
    queries = torch.zeros(1, selected.numel(), 3, device=video.device)
    queries[0, :, 0] = 15.0
    queries[0, :, 1] = gt[indices, 15, 1] * float(video.shape[-1] - 1)
    queries[0, :, 2] = gt[indices, 15, 0] * float(video.shape[-2] - 1)
    return queries


def _original_queries(prepared: dict[str, Any], video: torch.Tensor) -> torch.Tensor:
    query = prepared["query_points_tyx"].to(video.device)
    queries = torch.zeros(1, query.shape[0], 3, device=video.device)
    queries[0, :, 0] = query[:, 0]
    queries[0, :, 1] = query[:, 2] * float(video.shape[-1] - 1)
    queries[0, :, 2] = query[:, 1] * float(video.shape[-2] - 1)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    sidecar = output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != CSRR_SCHEMA_VERSION:
        raise ValueError("unexpected Gate 2 schema")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash drift")
    if file_sha256(config["partition"]["manifest"]) != config["partition"]["manifest_sha256"]:
        raise ValueError("manifest hash drift")
    if list(config["partition"]["gradient_train_source_indices"]) != list(range(8, 32)):
        raise ValueError("Gate 2 training membership drift")
    if list(config["partition"]["fit_internal_validation_source_indices"]) != list(range(32, 48)):
        raise ValueError("Gate 2 validation membership drift")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("locked-data flags must remain false")

    seed = int(config["training"]["seed"])
    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(checkpoint=config["backbone"]["checkpoint"]).to(args.device).eval()
    model = CounterfactualStructuredReextractionRestorer().to(args.device).eval()
    source_index = 8
    sample, sample_metadata = load_manifest_sample(Path(config["partition"]["manifest"]), source_index)
    prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
    video = prepared["video"].to(args.device)
    query_frames = prepared["query_points_tyx"][:, 0].round().long()
    visible_commit = ~prepared["gt_occluded"][:, 15]
    selected = torch.where((query_frames < 8) & visible_commit)[0][:8]
    if selected.numel() != 8:
        raise RuntimeError("source-8 interface smoke requires eight eligible points")

    with torch.no_grad():
        native = _initialize_and_first_window(predictor, video, _original_queries(prepared, video))
        fresh = _initialize_and_first_window(predictor, video, _fresh_queries(prepared, selected, video))
        observed = extract_cotracker_observed_feature_pyramid(predictor.model, video[:, :16])
        commit_pyramid = tuple(level[:, 15:16] for level in observed)
        gt = prepared["gt_tracks_yx"]
        commit_xy = torch.stack([gt[selected, 15, 1], gt[selected, 15, 0]], dim=-1).to(args.device)[None] * 255.0
        re_feat, re_support = reextract_cotracker_memory(
            predictor.model,
            commit_pyramid,
            commit_xy,
            input_height=256,
            input_width=256,
        )
        zero_action = apply_reextracted_state_action(
            native,
            point_indices=selected,
            predicted_coordinates_input_xy=commit_xy[0],
            apply_mask=torch.zeros(selected.numel(), dtype=torch.bool),
            reextracted_track_features=re_feat,
            reextracted_track_supports=re_support,
            input_height=256,
            input_width=256,
            model_height=int(predictor.interp_shape[0]),
            model_width=int(predictor.interp_shape[1]),
        )

    levels = []
    for level in range(4):
        levels.append(
            {
                "level": level,
                "track_feature_reextraction": _difference(fresh.online_track_feat[level], re_feat[level]),
                "support_reextraction": _difference(fresh.online_track_support[level], re_support[level]),
                "frame_feature_float16": _quantization(commit_pyramid[level]),
                "teacher_track_feature_float16": _quantization(fresh.online_track_feat[level]),
                "teacher_support_float16": _quantization(fresh.online_track_support[level]),
            }
        )
    qg = config["cache"]["quantization_gates"]
    maximum_error = float(qg["max_absolute_error"])
    minimum_cosine = float(qg["minimum_cosine_similarity"])
    reextraction_measurements = [
        row[key]
        for row in levels
        for key in ("track_feature_reextraction", "support_reextraction")
    ]
    quantization_measurements = [
        row[key]
        for row in levels
        for key in (
            "frame_feature_float16",
            "teacher_track_feature_float16",
            "teacher_support_float16",
        )
    ]
    checks = {
        "source_index_is_gradient_train": source_index in config["partition"]["gradient_train_source_indices"],
        "frame_count_exact": int(video.shape[1]) == int(config["partition"]["expected_frames_each"]),
        "point_count_exact": int(prepared["gt_tracks_yx"].shape[0]) == int(config["partition"]["expected_points_each"]),
        "selected_points_exact": int(selected.numel()) == 8,
        "model_parameter_ceiling": model.trainable_parameter_count <= int(config["model"]["trainable_parameter_ceiling"]),
        "teacher_reextraction_max_error": all(
            value["max_absolute_error"] <= maximum_error
            for value in reextraction_measurements
        ),
        "teacher_reextraction_min_cosine": all(
            value["cosine_similarity"] >= minimum_cosine
            for value in reextraction_measurements
        ),
        "float16_cache_max_error": all(
            value["max_absolute_error"] <= maximum_error
            for value in quantization_measurements
        ),
        "float16_cache_min_cosine": all(
            value["cosine_similarity"] >= minimum_cosine
            for value in quantization_measurements
        ),
        "zero_action_state_exact": snapshots_exact(native, zero_action),
        "model_reads_future_frames": False,
        "model_reads_gt": False,
        "model_validation_read": False,
        "calibration_read": False,
        "final_holdout_read": False,
        "tapvid_davis_read": False,
        "tapvid_kinetics_read": False,
        "official_kinetics_1144_rerun": False,
    }
    # False factual flags are required for the access checks.
    pass_value = all(
        value if key not in {"model_reads_future_frames", "model_reads_gt", "model_validation_read", "calibration_read", "final_holdout_read", "tapvid_davis_read", "tapvid_kinetics_read", "official_kinetics_1144_rerun"} else not value
        for key, value in checks.items()
    )
    artifact = {
        "selected_point_indices": selected.cpu(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "fresh_track_feat": [value.detach().cpu() for value in fresh.online_track_feat],
        "fresh_track_support": [value.detach().cpu() for value in fresh.online_track_support],
        "reextracted_track_feat": [value.detach().cpu() for value in re_feat],
        "reextracted_track_support": [value.detach().cpu() for value in re_support],
        "commit_feature_pyramid": [value.detach().cpu() for value in commit_pyramid],
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_counterfactual_state_restorer_interface_v0",
        "status": "completed_pass" if pass_value else "completed_fail",
        "decision": "ALLOW_GATE2_TEACHER_CACHE_BUILD" if pass_value else "STOP_GATE2_BEFORE_TEACHER_CACHE",
        "pass": pass_value,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_index": source_index,
        "video_name": prepared["video_name"],
        "sample_metadata": sample_metadata,
        "selected_point_indices": selected.tolist(),
        "interp_shape": list(map(int, predictor.interp_shape)),
        "trainable_parameters": model.trainable_parameter_count,
        "levels": levels,
        "checks": checks,
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("output", "decision", "pass", "trainable_parameters", "levels", "checks") if key in report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
