"""Pure P0m protocol, DAVIS conversion, and gate helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .routeD_kubric_cache import file_sha256

DAVIS_EXTERNAL_CACHE_SCHEMA = "routeD_cmcp_davis_external_video_cache_v0"
DAVIS_EXTERNAL_INDEX_SCHEMA = "routeD_cmcp_davis_external_cache_index_v0"
DAVIS_EXTERNAL_RESULT_SCHEMA = "routeD_cmcp_davis_external_result_v0"


def load_davis_external_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_davis_external_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = file_sha256(config_path)
    return config


def validate_davis_external_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment", {}).get("stage") != "P0m":
        raise ValueError("expected P0m config")
    if config["claim_boundary"]["untouched_final_test_claim"] != "forbidden":
        raise ValueError("DAVIS untouched-test claims are forbidden")
    if not config["claim_boundary"]["zero_shot_for_frozen_variant_C"]:
        raise ValueError("P0m must retain frozen zero-shot contract")
    if config["frozen_model"]["state_writeback"] != "disabled":
        raise ValueError("P0m is output-only")
    if config["frozen_model"]["calibration"] != "none":
        raise ValueError("P0m forbids calibration")
    if int(config["frozen_model"]["new_trainable_parameters"]) != 0:
        raise ValueError("P0m cannot add parameters")
    dataset = config["external_dataset"]
    if int(dataset["expected_videos"]) != 30 or dataset["subset_allowed"]:
        raise ValueError("P0m requires the complete 30-video DAVIS set")
    if dataset["loader_args"] != {
        "dataset_type": "davis",
        "resize_to": [256, 256],
        "queried_first": True,
        "fast_eval": False,
    }:
        raise ValueError("DAVIS loader contract drift")
    cache = config["cache_contract"]
    if not cache["cache_before_metrics"] or not cache["complete_30_video_cache_required"]:
        raise ValueError("P0m requires a complete sealed cache")
    if not cache["partial_performance_forbidden"]:
        raise ValueError("P0m forbids partial performance")
    if list(cache["extraction_replay_video_indices"]) != [0, 14, 29]:
        raise ValueError("P0m replay anchors drift")
    gates = config["formal_gates"]
    if float(gates["direct_AJ_gain_points_min"]) != 0.30:
        raise ValueError("P0m direct AJ gate drift")
    if int(gates["positive_video_count_min"]) != 18:
        raise ValueError("P0m positive-video gate drift")
    if config["kinetics_lock"]["rerun"] != "forbidden":
        raise ValueError("official Kinetics rerun must remain forbidden")


def verify_davis_external_files(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_config_path"]).parents[1]
    verified: dict[str, Any] = {}
    model = Path(str(config["frozen_model"]["checkpoint"]))
    if not model.is_absolute():
        model = (root / model).resolve()
    rows = [
        ("model", model, config["frozen_model"]["checkpoint_sha256"]),
        ("backbone", Path(config["backbone"]["checkpoint"]).resolve(), config["backbone"]["checkpoint_sha256"]),
        ("dataset", Path(config["external_dataset"]["path"]).resolve(), config["external_dataset"]["sha256"]),
    ]
    for name, path, expected in rows:
        if not path.exists() or file_sha256(path) != str(expected):
            raise ValueError(f"{name} path/hash mismatch")
        verified[name] = {"path": str(path), "sha256": str(expected)}
    dataset = Path(verified["dataset"]["path"])
    if dataset.stat().st_size != int(config["external_dataset"]["size_bytes"]):
        raise ValueError("DAVIS dataset size mismatch")
    for alias, row in config["pinned_implementation"].items():
        if alias == "repository_parent_commit":
            continue
        path = (root / row["path"]).resolve()
        if not path.exists() or file_sha256(path) != str(row["sha256"]):
            raise ValueError(f"pinned source mismatch: {alias}")
        verified[alias] = {"path": str(path), "sha256": str(row["sha256"])}
    return verified


def load_davis_sample_preserving_points(dataset: Any, index: int) -> Any:
    """Call the pinned loader while undoing its in-place point scaling exactly."""
    video_name = dataset.video_names[int(index)]
    original = dataset.points_dataset[video_name]["points"].copy()
    try:
        sample = dataset[int(index)]
    finally:
        dataset.points_dataset[video_name]["points"] = original
    return sample


def prepare_davis_sample(sample: Any, *, raster: int = 256) -> dict[str, Any]:
    """Convert pinned CoTrackerData to the P0j-C base-cache tensor contract."""
    if tuple(sample.video.shape[-2:]) != (raster, raster):
        raise ValueError("DAVIS sample raster mismatch")
    if sample.video.ndim != 4 or sample.trajectory.ndim != 3:
        raise ValueError("unexpected DAVIS sample rank")
    if sample.visibility.ndim != 2 or sample.query_points.ndim != 2:
        raise ValueError("unexpected DAVIS visibility/query rank")
    frames = int(sample.video.shape[0])
    points = int(sample.trajectory.shape[1])
    if sample.trajectory.shape != (frames, points, 2):
        raise ValueError("DAVIS trajectory shape mismatch")
    if sample.visibility.shape != (frames, points):
        raise ValueError("DAVIS visibility shape mismatch")
    if sample.query_points.shape != (points, 3):
        raise ValueError("DAVIS query shape mismatch")
    denom = float(raster - 1)
    query = sample.query_points.float().clone()
    query[:, 1:] /= denom
    gt_xy = sample.trajectory.float().permute(1, 0, 2).contiguous()
    gt_yx = gt_xy[..., [1, 0]] / denom
    gt_occluded = (~sample.visibility.bool()).permute(1, 0).contiguous()
    query_frames = query[:, 0].round().long()
    if (query_frames < 0).any() or (query_frames >= frames).any():
        raise ValueError("DAVIS query frame outside video")
    row = torch.arange(points)
    if gt_occluded[row, query_frames].any():
        raise ValueError("first-query point is occluded")
    return {
        "video": sample.video.float().unsqueeze(0),
        "query_points_tyx": query,
        "gt_tracks_yx": gt_yx,
        "gt_occluded": gt_occluded,
        "video_name": str(sample.seq_name),
        "frames": frames,
        "points": points,
    }


def davis_video_order_sha256(video_names: list[str]) -> str:
    payload = "\n".join(str(name) for name in video_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_davis_external_gate(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    exact_replay: bool,
) -> dict[str, Any]:
    gates = config["formal_gates"]
    per_video = metrics["per_video"]
    checks = {
        "complete_30_video_partition": int(metrics["videos"]) == 30 and len(per_video) == 30,
        "native_candidate_parity_all": bool(metrics["native_candidate_parity_all"]),
        "candidate_oracle_AJ_gain_at_least_3": float(metrics["oracle_gain_points"]["AJ"])
        >= float(gates["candidate_oracle_AJ_gain_points_min"]),
        "direct_AJ_gain_at_least_0_30": float(metrics["selected_gain_points"]["AJ"])
        >= float(gates["direct_AJ_gain_points_min"]),
        "paired_direct_AJ_CI_lower_positive": float(
            metrics["paired_video_selected_AJ_gain_CI"]["lower"]
        )
        > float(gates["paired_video_direct_AJ_CI_lower_gt"]),
        "delta_gain_positive": float(metrics["selected_gain_points"]["delta_average"])
        > float(gates["direct_delta_average_gain_gt"]),
        "severe_16px_not_worse": float(metrics["severe_16px_rate"]["selected_delta"])
        <= 0.0,
        "harmful_non_native_rate_at_most_0_01": float(
            metrics["behavior"]["harmful_non_native_rate"]
        )
        <= float(gates["harmful_non_native_rate_max"]),
        "positive_videos_at_least_18": sum(
            float(row["selected_AJ_gain_points"]) > 0.0 for row in per_video
        )
        >= int(gates["positive_video_count_min"]),
        "exact_primary_replay": bool(exact_replay),
    }
    checks["pass"] = all(checks.values())
    checks["decision"] = config["formal_decisions"]["pass" if checks["pass"] else "fail"]
    return checks
