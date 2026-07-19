"""Pure P0m protocol, DAVIS conversion, and gate helpers."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from datasets.metrics import compute_tapvid_metrics

from .cotracker3_stage0_adapter import tensor_sha256

from .routeD_cmcp_feature_cache import load_complete_feature_index
from .routeD_cmcp_lmra_training import predict_lmra_video
from .routeD_cmcp_training import load_cmcp_video
from .routeD_kubric_cache import canonical_json_sha256, file_sha256
from .routeD_musr_training import (
    _tracks_from_xy,
    _visible_error_stats,
    paired_video_bootstrap_ci,
)

DAVIS_EXTERNAL_CACHE_SCHEMA = "routeD_cmcp_davis_external_video_cache_v0"
DAVIS_EXTERNAL_INDEX_SCHEMA = "routeD_cmcp_davis_external_cache_index_v0"
DAVIS_EXTERNAL_RESULT_SCHEMA = "routeD_cmcp_davis_external_result_v0"
DAVIS_NATIVE_STATE_KEYS = (
    "native_coords_xy_px",
    "native_visibility_probability",
    "native_confidence_probability",
    "native_joint_probability",
    "native_visibility",
)


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


def validated_davis_native_state_hashes(
    base_sidecar: str | Path, feature_sidecar: str | Path
) -> dict[str, str]:
    """Validate and return the frozen native-state provenance for one cache row."""
    base_path = Path(base_sidecar).resolve()
    feature_path = Path(feature_sidecar).resolve()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    tensors = base.get("tensors", {})
    stored = feature.get("native_state_hashes")
    if not isinstance(stored, dict) or set(stored) != set(DAVIS_NATIVE_STATE_KEYS):
        raise ValueError("DAVIS feature artifact native-state hash contract mismatch")
    observed = {}
    for key in DAVIS_NATIVE_STATE_KEYS:
        value = tensors.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"DAVIS base artifact missing native tensor: {key}")
        observed[key] = tensor_sha256(value)
    if observed != stored:
        raise ValueError("DAVIS native-state hashes do not match base tensors")
    return observed

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


def mean_davis_metric_dicts(rows: list[Mapping[str, float]]) -> dict[str, float]:
    """Equal-video mean used by the official CoTracker TAP-Vid evaluator."""
    if not rows:
        raise ValueError("DAVIS metric rows must be non-empty")
    keys = tuple(rows[0].keys())
    if any(tuple(row.keys()) != keys for row in rows):
        raise ValueError("DAVIS metric-key drift across videos")
    output: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite DAVIS metric: {key}")
        output[key] = float(np.mean(values))
    return output


def evaluate_davis_lmra_index(
    adapter,
    cmcp,
    comparator,
    feature_index_path: str | Path,
    normalization,
    *,
    device: str,
    point_batch_size: int = 4,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20019,
) -> dict[str, Any]:
    """Evaluate frozen variant C on variable-length DAVIS videos."""
    index = load_complete_feature_index(
        feature_index_path, expected_partition="davis_external"
    )
    rows = index["videos"]
    per_video: list[dict[str, Any]] = []
    coordinate_hashes: list[str] = []
    totals = {
        "rows": 0,
        "selected": 0.0,
        "harmful": 0.0,
        "available": 0.0,
        "beneficial": 0.0,
    }
    native_parity_all = True
    for row in rows:
        bundle = load_cmcp_video(row)
        prediction, behavior = predict_lmra_video(
            adapter,
            cmcp,
            comparator,
            bundle,
            normalization,
            device=device,
            point_batch_size=point_batch_size,
        )
        tensors = bundle.tensors
        native = tensors["native_coords_xy_px"].float()
        selected = prediction["selected_coords_xy_px"]
        oracle = prediction["oracle_coords_xy_px"]
        candidates = prediction["candidate_coords_xy_px"]
        native_parity = torch.equal(candidates[..., 0, :], native)
        native_parity_all = native_parity_all and native_parity
        coordinate_hash = tensor_sha256(candidates)
        coordinate_hashes.append(coordinate_hash)
        visibility = tensors["native_visibility"]
        gt_visibility = ~tensors["gt_occluded"]
        metric_args = (
            tensors["gt_tracks_yx"],
            visibility,
            gt_visibility,
            tensors["query_points_tyx"],
        )
        native_metrics = compute_tapvid_metrics(
            _tracks_from_xy(native, 256),
            *metric_args,
            resolution=256,
            query_mode="first",
        )
        selected_metrics = compute_tapvid_metrics(
            _tracks_from_xy(selected, 256),
            *metric_args,
            resolution=256,
            query_mode="first",
        )
        oracle_metrics = compute_tapvid_metrics(
            _tracks_from_xy(oracle, 256),
            *metric_args,
            resolution=256,
            query_mode="first",
        )
        native_error = _visible_error_stats(native, tensors, raster=256)
        selected_error = _visible_error_stats(selected, tensors, raster=256)
        oracle_error = _visible_error_stats(oracle, tensors, raster=256)
        per_video.append(
            {
                "source_index": bundle.source_index,
                "video_name": bundle.video_name,
                "frames": int(native.shape[1]),
                "points": int(native.shape[0]),
                "native_metrics": native_metrics,
                "selected_metrics": selected_metrics,
                "oracle_metrics": oracle_metrics,
                "native_AJ": float(native_metrics["AJ"]),
                "selected_AJ": float(selected_metrics["AJ"]),
                "oracle_AJ": float(oracle_metrics["AJ"]),
                "selected_AJ_gain_points": 100.0
                * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
                "oracle_AJ_gain_points": 100.0
                * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
                "native_delta_average": float(native_metrics["<avg"]),
                "selected_delta_average": float(selected_metrics["<avg"]),
                "oracle_delta_average": float(oracle_metrics["<avg"]),
                "selected_delta_gain_points": 100.0
                * (
                    float(selected_metrics["<avg"])
                    - float(native_metrics["<avg"])
                ),
                "oracle_delta_gain_points": 100.0
                * (float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])),
                "native_error": native_error,
                "selected_error": selected_error,
                "oracle_error": oracle_error,
                "behavior": behavior,
                "candidate_coordinate_sha256": coordinate_hash,
                "native_candidate_parity": native_parity,
                "adapted_feature_sha256": prediction["adapted_feature_sha256"],
            }
        )
        count = int(behavior["rows"])
        totals["rows"] += count
        totals["selected"] += float(behavior["selected_non_native_rate"]) * count
        totals["harmful"] += float(behavior["harmful_non_native_rate"]) * count
        totals["available"] += (
            float(behavior["beneficial_candidate_available_rate"]) * count
        )
        totals["beneficial"] += (
            float(behavior["beneficial_candidate_recall"])
            * float(behavior["beneficial_candidate_available_rate"])
            * count
        )

    native_metrics = mean_davis_metric_dicts(
        [row["native_metrics"] for row in per_video]
    )
    selected_metrics = mean_davis_metric_dicts(
        [row["selected_metrics"] for row in per_video]
    )
    oracle_metrics = mean_davis_metric_dicts(
        [row["oracle_metrics"] for row in per_video]
    )
    selected_aj = [row["selected_AJ_gain_points"] for row in per_video]
    oracle_aj = [row["oracle_AJ_gain_points"] for row in per_video]
    selected_delta = [row["selected_delta_gain_points"] for row in per_video]
    weights = [row["native_error"]["rows"] for row in per_video]
    native_severe = float(
        np.average(
            [row["native_error"]["severe_16px_rate"] for row in per_video],
            weights=weights,
        )
    )
    selected_severe = float(
        np.average(
            [row["selected_error"]["severe_16px_rate"] for row in per_video],
            weights=weights,
        )
    )
    oracle_severe = float(
        np.average(
            [row["oracle_error"]["severe_16px_rate"] for row in per_video],
            weights=weights,
        )
    )
    count = max(int(totals["rows"]), 1)
    return {
        "partition": "davis_external",
        "aggregation": "official_equal_video_mean",
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "selected_metrics": selected_metrics,
        "oracle_metrics": oracle_metrics,
        "selected_gain_points": {
            "AJ": 100.0
            * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0
            * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0
            * (float(selected_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "oracle_gain_points": {
            "AJ": 100.0
            * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0
            * (float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0
            * (float(oracle_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "paired_video_selected_AJ_gain_CI": paired_video_bootstrap_ci(
            selected_aj, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "paired_video_oracle_AJ_gain_CI": paired_video_bootstrap_ci(
            oracle_aj, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        "paired_video_selected_delta_gain_CI": paired_video_bootstrap_ci(
            selected_delta, seed=bootstrap_seed + 2, samples=bootstrap_samples
        ),
        "severe_16px_rate": {
            "native": native_severe,
            "selected": selected_severe,
            "oracle": oracle_severe,
            "selected_delta": selected_severe - native_severe,
            "oracle_delta": oracle_severe - native_severe,
        },
        "behavior": {
            "rows": totals["rows"],
            "selected_non_native_rate": totals["selected"] / count,
            "harmful_non_native_rate": totals["harmful"] / count,
            "beneficial_candidate_available_rate": totals["available"] / count,
            "beneficial_candidate_recall": totals["beneficial"]
            / max(totals["available"], 1.0),
        },
        "candidate_coordinate_combined_sha256": canonical_json_sha256(
            coordinate_hashes
        ),
        "native_candidate_parity_all": native_parity_all,
        "per_video": per_video,
        "feature_cache_index_sha256": index["_index_sha256"],
    }

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
