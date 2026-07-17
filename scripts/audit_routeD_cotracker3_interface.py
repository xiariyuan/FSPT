#!/usr/bin/env python3
"""One-video Kubric audit for the Route-D MUSR x CoTracker3-online interface.

This is a non-training gate. Candidate generation is causal and ground-truth
free. Ground truth is used only after the candidate cache is frozen to compute
an oracle upper bound. The completed TAP-Vid-Kinetics evaluation is never read.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = REPO_ROOT / "baselines" / "cotracker"
for path in (EXTERNAL_ROOT, COTRACKER_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cotracker.predictor import CoTrackerOnlinePredictor

from datasets.metrics import compute_tapvid_metrics
from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import (
    CANDIDATE_FEATURE_DIM,
    LOCAL_CORRELATION_SOURCE_ID,
    NATIVE_SOURCE_ID,
    STATE_FEATURE_DIM,
    build_state_feature,
    file_sha256,
    gather_candidate_coordinates,
    local_correlation_candidates,
    make_first_visible_queries,
    normalize_feature_maps,
    oracle_candidate_indices,
    sample_feature_at_xy,
    tensor_sha256,
)

DEFAULT_MANIFEST = EXTERNAL_ROOT / (
    "outputs/beliefcal_mvp1_a2_protocol_run_20260714/"
    "kubric_validation/test.index.json"
)
DEFAULT_CHECKPOINT = EXTERNAL_ROOT / "baselines/cotracker/checkpoints/scaled_online.pth"
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/routeD_strong_backbone_20260717/cotracker3_interface_smoke.json"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    repo_candidate = (REPO_ROOT / value).resolve()
    if repo_candidate.exists():
        return repo_candidate
    external_candidate = (EXTERNAL_ROOT / value).resolve()
    if external_candidate.exists():
        return external_candidate
    return repo_candidate


def load_manifest_sample(manifest_path: Path, video_index: int) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest has no shards: {manifest_path}")
    target = int(video_index)
    if target < 0:
        raise ValueError("video-index must be non-negative")
    offset = 0
    for shard_index, entry in enumerate(shards):
        if isinstance(entry, str):
            relative = entry
            declared_count = 0
        else:
            relative = entry.get("path") or entry.get("file")
            declared_count = int(entry.get("num_samples", 0) or 0)
        shard_path = Path(relative)
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        if declared_count > 0 and target >= offset + declared_count:
            offset += declared_count
            continue
        with shard_path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            samples = list(payload.values())
        elif isinstance(payload, list):
            samples = payload
        else:
            raise ValueError(f"unsupported shard payload: {type(payload)}")
        local_index = target - offset
        if 0 <= local_index < len(samples):
            sample = samples[local_index]
            metadata = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(str(manifest_path)),
                "manifest_split": manifest.get("split"),
                "manifest_num_samples": manifest.get("num_samples"),
                "global_video_index": target,
                "shard_index": shard_index,
                "shard_local_index": local_index,
                "shard_path": str(shard_path.resolve()),
                "shard_sha256": file_sha256(str(shard_path)),
            }
            return sample, metadata
        offset += len(samples)
    raise IndexError(f"video-index {target} is outside manifest")


def prepare_sample(sample: dict, input_raster: int) -> dict[str, Any]:
    video = np.asarray(sample["video"])
    if video.ndim != 4:
        raise ValueError(f"video must have shape (T,H,W,3), got {video.shape}")
    if video.shape[-1] != 3 and video.shape[1] == 3:
        video = np.transpose(video, (0, 2, 3, 1))
    tracks = np.asarray(sample["target_points"], dtype=np.float32)
    occluded = np.asarray(sample["occluded"], dtype=bool)
    queries, tracks, occluded = make_first_visible_queries(tracks, occluded)
    if np.nanmax(np.abs(tracks)) > 1.5:
        source_h, source_w = int(video.shape[1]), int(video.shape[2])
        tracks = tracks.copy()
        tracks[..., 0] /= float(max(source_h - 1, 1))
        tracks[..., 1] /= float(max(source_w - 1, 1))
        queries = queries.copy()
        queries[:, 1] /= float(max(source_h - 1, 1))
        queries[:, 2] /= float(max(source_w - 1, 1))

    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    if tuple(video_tensor.shape[-2:]) != (input_raster, input_raster):
        video_tensor = F.interpolate(
            video_tensor,
            size=(input_raster, input_raster),
            mode="bilinear",
            align_corners=True,
        )
    return {
        "video": video_tensor.unsqueeze(0),
        "query_points_tyx": torch.from_numpy(queries).float(),
        "gt_tracks_yx": torch.from_numpy(tracks).float(),
        "gt_occluded": torch.from_numpy(occluded),
        "video_name": str(sample.get("video_name", "unknown")),
        "source_tfrecord": sample.get("source_tfrecord"),
        "source_record_index": sample.get("source_record_index"),
    }


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_true_streaming(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    queries_tyx: torch.Tensor,
) -> None:
    _, frames, _, height, width = video.shape
    queries_xy = torch.zeros(
        (1, queries_tyx.shape[0], 3), device=video.device, dtype=torch.float32
    )
    queries_xy[0, :, 0] = queries_tyx[:, 0].to(video.device)
    queries_xy[0, :, 1] = queries_tyx[:, 2].to(video.device) * float(width - 1)
    queries_xy[0, :, 2] = queries_tyx[:, 1].to(video.device) * float(height - 1)
    predictor(
        video_chunk=video,
        is_first_step=True,
        queries=queries_xy,
        add_support_grid=False,
        grid_size=0,
    )
    with torch.no_grad():
        for start in range(0, frames - predictor.step, predictor.step):
            chunk = video[:, start : start + predictor.step * 2]
            predictor(
                video_chunk=chunk,
                is_first_step=False,
                add_support_grid=False,
                grid_size=0,
            )
    state = predictor.model.online_coords_predicted
    if state is None or state.shape[1] < frames:
        raise RuntimeError(
            f"incomplete online state: {None if state is None else tuple(state.shape)}"
        )


def compute_fmaps(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    *,
    chunk_size: int = 8,
) -> torch.Tensor:
    _, frames, channels, _, _ = video.shape
    interp_height, interp_width = predictor.interp_shape
    resized = F.interpolate(
        video.reshape(-1, channels, video.shape[-2], video.shape[-1]),
        size=(interp_height, interp_width),
        mode="bilinear",
        align_corners=True,
    ).reshape(1, frames, channels, interp_height, interp_width)
    normalized = 2.0 * (resized / 255.0) - 1.0
    outputs = []
    with torch.no_grad():
        for start in range(0, frames, chunk_size):
            part = normalized[:, start : start + chunk_size]
            fmap = predictor.model.fnet(
                part.reshape(-1, channels, interp_height, interp_width)
            )
            outputs.append(fmap.reshape(part.shape[1], *fmap.shape[1:]))
    return normalize_feature_maps(torch.cat(outputs, dim=0))


def extract_online_track_features(
    predictor: CoTrackerOnlinePredictor,
    point_count: int,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    state = getattr(predictor.model, "online_track_feat", None)
    if isinstance(state, (list, tuple)) and state:
        state = state[0]
    if isinstance(state, torch.Tensor):
        value = state.detach().float()
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        matching_axes = [axis for axis, size in enumerate(value.shape) if size == point_count]
        if matching_axes:
            point_axis = matching_axes[0]
            value = value.movedim(point_axis, 0).reshape(point_count, -1)
            return value, {
                "source": "cotracker3.model.online_track_feat[0]",
                "raw_shape": list(state.shape),
                "flattened_shape": list(value.shape),
            }
    return fallback.detach().float(), {
        "source": "query_frame_fnet_support_fallback",
        "raw_shape": None,
        "flattened_shape": list(fallback.shape),
    }


def build_adapter_tensors(
    predictor: CoTrackerOnlinePredictor,
    prepared: dict[str, Any],
    *,
    input_raster: int,
    candidate_topk: int,
    search_radius_px: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    video = prepared["video"].to(next(predictor.parameters()).device)
    queries = prepared["query_points_tyx"].to(video.device)
    gt_tracks = prepared["gt_tracks_yx"].to(video.device)
    gt_occluded = prepared["gt_occluded"].to(video.device)
    point_count, frame_count = gt_tracks.shape[:2]
    candidate_count = 1 + candidate_topk

    raw_coords = predictor.model.online_coords_predicted[0, :frame_count, :point_count].float()
    raw_vis = predictor.model.online_vis_predicted[0, :frame_count, :point_count].float()
    raw_conf = predictor.model.online_conf_predicted[0, :frame_count, :point_count].float()
    interp_height, interp_width = predictor.interp_shape
    native_xy = raw_coords.clone()
    native_xy[..., 0] *= float(input_raster - 1) / float(max(interp_width - 1, 1))
    native_xy[..., 1] *= float(input_raster - 1) / float(max(interp_height - 1, 1))
    native_xy = native_xy.permute(1, 0, 2).contiguous()  # N,T,2
    visibility_probability = torch.sigmoid(raw_vis).permute(1, 0).contiguous()
    confidence_probability = torch.sigmoid(raw_conf).permute(1, 0).contiguous()
    native_joint = visibility_probability * confidence_probability
    native_visible = native_joint > 0.6

    fmaps = compute_fmaps(predictor, video)
    query_xy = torch.stack(
        [queries[:, 2] * float(input_raster - 1), queries[:, 1] * float(input_raster - 1)],
        dim=-1,
    )
    query_frames = queries[:, 0].round().long().clamp(0, frame_count - 1)
    support_features = torch.stack(
        [
            sample_feature_at_xy(
                fmaps[int(query_frames[index].item())],
                query_xy[index],
                input_height=input_raster,
                input_width=input_raster,
            )
            for index in range(point_count)
        ]
    )
    online_track_feature, track_feature_info = extract_online_track_features(
        predictor, point_count, support_features
    )

    candidate_coords = torch.zeros(
        point_count, frame_count, candidate_count, 2, device=video.device
    )
    candidate_features = torch.zeros(
        point_count,
        frame_count,
        candidate_count,
        CANDIDATE_FEATURE_DIM,
        device=video.device,
    )
    candidate_scores = torch.full(
        (point_count, frame_count, candidate_count),
        float("-inf"),
        device=video.device,
    )
    candidate_valid = torch.zeros(
        point_count, frame_count, candidate_count, dtype=torch.bool, device=video.device
    )
    source_ids = torch.full(
        (point_count, frame_count, candidate_count),
        LOCAL_CORRELATION_SOURCE_ID,
        dtype=torch.long,
        device=video.device,
    )
    source_ids[..., 0] = NATIVE_SOURCE_ID
    state_features = torch.zeros(
        point_count, frame_count, STATE_FEATURE_DIM, device=video.device
    )
    peak = torch.zeros(point_count, frame_count, device=video.device)
    margin = torch.zeros_like(peak)
    entropy = torch.zeros_like(peak)
    search_cells = torch.zeros(
        point_count, frame_count, dtype=torch.int32, device=video.device
    )

    for point_index in range(point_count):
        query_frame = int(query_frames[point_index].item())
        for frame_index in range(frame_count):
            current = native_xy[point_index, frame_index]
            previous = native_xy[point_index, max(frame_index - 1, 0)]
            previous_previous = native_xy[point_index, max(frame_index - 2, 0)]
            candidate_coords[point_index, frame_index, 0] = current
            candidate_valid[point_index, frame_index, 0] = True
            state_features[point_index, frame_index] = build_state_feature(
                native_xy_px=current,
                previous_native_xy_px=previous,
                previous_previous_native_xy_px=previous_previous,
                visibility_probability=float(
                    visibility_probability[point_index, frame_index].item()
                ),
                confidence_probability=float(
                    confidence_probability[point_index, frame_index].item()
                ),
                frame_index=frame_index,
                frame_count=frame_count,
                query_frame=query_frame,
                online_track_feature=online_track_feature[point_index],
                input_height=input_raster,
                input_width=input_raster,
                radius_px=search_radius_px,
            )
            if frame_index < query_frame:
                continue
            result = local_correlation_candidates(
                fmap=fmaps[frame_index],
                support_feature=support_features[point_index],
                native_xy_px=current,
                previous_native_xy_px=previous,
                native_visibility_probability=float(
                    visibility_probability[point_index, frame_index].item()
                ),
                native_confidence_probability=float(
                    confidence_probability[point_index, frame_index].item()
                ),
                input_height=input_raster,
                input_width=input_raster,
                radius_px=search_radius_px,
                topk=candidate_topk,
            )
            candidate_coords[point_index, frame_index] = result.candidate_xy_px
            candidate_features[point_index, frame_index] = result.candidate_features
            candidate_scores[point_index, frame_index] = result.candidate_scores
            candidate_valid[point_index, frame_index] = result.candidate_valid
            source_ids[point_index, frame_index] = result.source_ids
            peak[point_index, frame_index] = result.peak
            margin[point_index, frame_index] = result.margin
            entropy[point_index, frame_index] = result.entropy
            search_cells[point_index, frame_index] = result.search_cell_count

    gt_xy_px = torch.stack(
        [
            gt_tracks[..., 1] * float(input_raster - 1),
            gt_tracks[..., 0] * float(input_raster - 1),
        ],
        dim=-1,
    )
    oracle_index, oracle_error, native_error = oracle_candidate_indices(
        candidate_coords, candidate_valid, gt_xy_px
    )
    oracle_xy = gather_candidate_coordinates(candidate_coords, oracle_index)

    tensors = {
        "candidate_coords_xy_px": candidate_coords.detach().cpu(),
        "candidate_features": candidate_features.detach().cpu(),
        "candidate_scores": candidate_scores.detach().cpu(),
        "candidate_valid_mask": candidate_valid.detach().cpu(),
        "source_ids": source_ids.detach().cpu(),
        "state_features": state_features.detach().cpu(),
        "native_coords_xy_px": native_xy.detach().cpu(),
        "native_visibility_probability": visibility_probability.detach().cpu(),
        "native_confidence_probability": confidence_probability.detach().cpu(),
        "native_joint_probability": native_joint.detach().cpu(),
        "native_visibility": native_visible.detach().cpu(),
        "query_points_tyx": queries.detach().cpu(),
        "gt_tracks_yx": gt_tracks.detach().cpu(),
        "gt_occluded": gt_occluded.detach().cpu(),
        "oracle_candidate_index": oracle_index.detach().cpu(),
        "oracle_coords_xy_px": oracle_xy.detach().cpu(),
        "oracle_error_px": oracle_error.detach().cpu(),
        "native_error_px": native_error.detach().cpu(),
        "local_peak": peak.detach().cpu(),
        "local_margin": margin.detach().cpu(),
        "local_entropy": entropy.detach().cpu(),
        "search_cell_count": search_cells.detach().cpu(),
    }
    runtime = {
        "interp_shape": [int(interp_height), int(interp_width)],
        "online_step": int(predictor.step),
        "fmap_shape": list(fmaps.shape),
        "online_track_feature": track_feature_info,
    }
    return tensors, runtime


def compute_metrics_and_oracle(
    tensors: dict[str, torch.Tensor],
    *,
    metric_raster: int,
) -> dict[str, Any]:
    scale = float(metric_raster - 1)
    native_xy = tensors["native_coords_xy_px"]
    oracle_xy = tensors["oracle_coords_xy_px"]
    native_tracks_yx = torch.stack(
        [native_xy[..., 1] / scale, native_xy[..., 0] / scale], dim=-1
    )
    oracle_tracks_yx = torch.stack(
        [oracle_xy[..., 1] / scale, oracle_xy[..., 0] / scale], dim=-1
    )
    gt_visibility = ~tensors["gt_occluded"]
    native_metrics = compute_tapvid_metrics(
        native_tracks_yx,
        tensors["gt_tracks_yx"],
        tensors["native_visibility"],
        gt_visibility,
        tensors["query_points_tyx"],
        resolution=metric_raster,
        query_mode="first",
    )
    oracle_metrics = compute_tapvid_metrics(
        oracle_tracks_yx,
        tensors["gt_tracks_yx"],
        tensors["native_visibility"],
        gt_visibility,
        tensors["query_points_tyx"],
        resolution=metric_raster,
        query_mode="first",
    )
    points, frames = gt_visibility.shape
    frame_index = torch.arange(frames).view(1, frames)
    query_frame = tensors["query_points_tyx"][:, 0].round().long().view(points, 1)
    visible_eval = gt_visibility & (frame_index > query_frame)
    native_error = tensors["native_error_px"][visible_eval]
    oracle_error = tensors["oracle_error_px"][visible_eval]
    oracle_index = tensors["oracle_candidate_index"][visible_eval]
    threshold_stats = {}
    for threshold in (1.0, 2.0, 4.0, 8.0, 16.0):
        native_hit = (native_error < threshold).float().mean()
        oracle_hit = (oracle_error < threshold).float().mean()
        threshold_stats[str(int(threshold))] = {
            "native": float(native_hit.item()),
            "oracle": float(oracle_hit.item()),
            "gain": float((oracle_hit - native_hit).item()),
        }
    aj_gain_points = 100.0 * (
        float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])
    )
    delta_gain_points = 100.0 * (
        float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])
    )
    return {
        "native_metrics": native_metrics,
        "coordinate_oracle_same_visibility_metrics": oracle_metrics,
        "gain_points": {
            "AJ": aj_gain_points,
            "delta_average": delta_gain_points,
            "OA": 100.0 * (
                float(oracle_metrics["OA"]) - float(native_metrics["OA"])
            ),
        },
        "visible_evaluation_rows": int(visible_eval.sum().item()),
        "mean_native_error_px": float(native_error.mean().item()),
        "mean_oracle_error_px": float(oracle_error.mean().item()),
        "median_native_error_px": float(native_error.median().item()),
        "median_oracle_error_px": float(oracle_error.median().item()),
        "oracle_non_native_selection_rate": float((oracle_index > 0).float().mean().item()),
        "oracle_strict_improvement_rate": float((oracle_error < native_error).float().mean().item()),
        "threshold_stats": threshold_stats,
    }


def core_hashes(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    keys = (
        "candidate_coords_xy_px",
        "candidate_features",
        "candidate_scores",
        "candidate_valid_mask",
        "source_ids",
        "state_features",
        "native_coords_xy_px",
        "native_visibility_probability",
        "native_confidence_probability",
        "native_visibility",
    )
    return {key: tensor_sha256(tensors[key]) for key in keys}


def compare_replays(
    reference: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = {}
    exact = True
    for key in core_hashes(reference):
        left = reference[key]
        right = current[key]
        equal = torch.equal(left, right)
        exact = exact and equal
        max_abs = None
        if left.dtype.is_floating_point and left.shape == right.shape:
            if equal:
                max_abs = 0.0
            else:
                finite = torch.isfinite(left) & torch.isfinite(right)
                nonfinite_match = torch.equal(torch.isposinf(left), torch.isposinf(right)) and torch.equal(
                    torch.isneginf(left), torch.isneginf(right)
                ) and torch.equal(torch.isnan(left), torch.isnan(right))
                if finite.any():
                    max_abs = float((left[finite] - right[finite]).abs().max().item())
                else:
                    max_abs = 0.0 if nonfinite_match else float("inf")
                if not nonfinite_match:
                    max_abs = float("inf")
        rows[key] = {
            "exact": equal,
            "max_abs": max_abs,
            "reference_sha256": tensor_sha256(left),
            "current_sha256": tensor_sha256(right),
        }
    return {"exact": exact, "tensors": rows}


def build_provenance(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    sample_metadata: dict,
    prepared: dict[str, Any],
    runtime: dict,
) -> dict[str, Any]:
    return {
        "schema_version": "routeD_musr_cotracker3_stage0_adapter_v1",
        "backbone": "CoTracker3 scaled online true-streaming",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(str(checkpoint)),
        "dataset": "Kubric-only frozen validation partition",
        "sample": {
            **sample_metadata,
            "video_name": prepared["video_name"],
            "source_tfrecord": prepared["source_tfrecord"],
            "source_record_index": prepared["source_record_index"],
        },
        "query_sampler": "first visible frame per track; query format [t,y,x]",
        "coordinate_convention": "candidate/state coordinates are input-raster [x,y] pixels",
        "candidate_sources": {
            str(NATIVE_SOURCE_ID): "native CoTracker3 online continuation",
            str(LOCAL_CORRELATION_SOURCE_ID): (
                "query-memory local correlation: current-frame CoTracker fnet map, "
                "query-frame support feature, square search around native coordinate"
            ),
        },
        "candidate_generation": {
            "causal": True,
            "future_frames_used": False,
            "ground_truth_used": False,
            "support_rule": "per-point feature sampled at its first-visible query frame",
            "current_feature_rule": "frame-independent CoTracker3 fnet on current frame only",
            "similarity": "cosine correlation",
            "search_radius_input_px": float(args.search_radius_px),
            "topk": int(args.candidate_topk),
            "tie_break": "stable descending score, then ascending row-major feature-cell index",
            "candidate_zero_is_native": True,
        },
        "ground_truth_usage": (
            "GT is accessed only after candidate tensors are frozen, for the oracle upper-bound audit."
        ),
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "state_feature_dim": STATE_FEATURE_DIM,
        "runtime": runtime,
        "deterministic_replay_scope": "adapter export is recomputed independently from one frozen native backbone state",
        "external_data_lock": {
            "tapvid_kinetics_official_scale_1144_read": False,
            "tapvid_davis_read": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--kubric-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--query-mode", choices=["first"], default="first")
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--input-raster", type=int, default=256)
    parser.add_argument("--metric-raster", type=int, default=256)
    parser.add_argument("--deterministic-replays", type=int, default=2)
    parser.add_argument("--candidate-topk", type=int, default=5)
    parser.add_argument("--search-radius-px", type=float, default=64.0)
    parser.add_argument("--routing-disabled", action="store_true")
    parser.add_argument("--oracle-aj-gate-points", type=float, default=3.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.deterministic_replays < 1:
        raise ValueError("deterministic-replays must be at least 1")
    if args.input_raster <= 1 or args.metric_raster <= 1:
        raise ValueError("raster sizes must exceed 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = _resolve(args.checkpoint)
    manifest = _resolve(args.kubric_manifest)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".pt")
    print(json.dumps({"stage": "load_sample", "manifest": str(manifest), "video_index": args.video_index}), flush=True)
    sample, sample_metadata = load_manifest_sample(manifest, args.video_index)
    prepared = prepare_sample(sample, args.input_raster)
    print(json.dumps({"stage": "sample_ready", "frames": int(prepared["video"].shape[1]), "points": int(prepared["gt_tracks_yx"].shape[0])}), flush=True)

    replay_tensors = []
    replay_runtime = []
    replay_hashes = []
    replay_seconds = []

    # Run the frozen backbone once. Deterministic replays below independently
    # recompute the adapter export (fnet maps, local search, features, and hashes)
    # from this unchanged native online state. This isolates adapter determinism
    # from virtual-GPU allocation instability between separate processes.
    set_deterministic(17000)
    print(json.dumps({"stage": "backbone_start", "device": args.device}), flush=True)
    predictor = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(args.device).eval()
    print(json.dumps({"stage": "model_loaded", "interp_shape": list(predictor.interp_shape)}), flush=True)
    video = prepared["video"].to(args.device)
    run_true_streaming(
        predictor,
        video,
        prepared["query_points_tyx"].to(args.device),
    )
    print(json.dumps({"stage": "stream_complete"}), flush=True)

    for replay in range(args.deterministic_replays):
        set_deterministic(17000)
        start = time.time()
        print(json.dumps({"stage": "adapter_replay_start", "replay": replay}), flush=True)
        tensors, runtime = build_adapter_tensors(
            predictor,
            prepared,
            input_raster=args.input_raster,
            candidate_topk=args.candidate_topk,
            search_radius_px=args.search_radius_px,
        )
        print(json.dumps({"stage": "adapter_replay_complete", "replay": replay}), flush=True)
        replay_tensors.append(tensors)
        replay_runtime.append(runtime)
        replay_hashes.append(core_hashes(tensors))
        replay_seconds.append(round(time.time() - start, 3))

    del predictor, video
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    comparisons = [
        compare_replays(replay_tensors[0], replay_tensors[index])
        for index in range(1, len(replay_tensors))
    ]
    deterministic_exact = bool(comparisons) and all(item["exact"] for item in comparisons)
    tensors = replay_tensors[0]

    candidate_zero = tensors["candidate_coords_xy_px"][..., 0, :]
    native = tensors["native_coords_xy_px"]
    routing_disabled_max_abs = float((candidate_zero - native).abs().max().item())
    routing_disabled_exact = torch.equal(candidate_zero, native)
    candidate_zero_valid = bool(tensors["candidate_valid_mask"][..., 0].all().item())
    parity_pass = routing_disabled_exact and candidate_zero_valid
    if args.routing_disabled and not parity_pass:
        raise RuntimeError("routing-disabled native parity failed")

    audit = compute_metrics_and_oracle(tensors, metric_raster=args.metric_raster)
    oracle_gate_pass = audit["gain_points"]["AJ"] >= float(args.oracle_aj_gate_points)
    decision = (
        "PROCEED_TO_FROZEN_KUBRIC_MULTI_VIDEO_ORACLE_AUDIT"
        if oracle_gate_pass
        else "STOP_AND_REDESIGN_CANDIDATE_GENERATOR_BEFORE_TRAINING"
    )
    provenance = build_provenance(
        args=args,
        checkpoint=checkpoint,
        sample_metadata=sample_metadata,
        prepared=prepared,
        runtime=replay_runtime[0],
    )
    artifact = {
        "schema_version": provenance["schema_version"],
        "provenance": provenance,
        "tensors": tensors,
        "tensor_hashes": replay_hashes[0],
        "audit": audit,
    }
    torch.save(artifact, sidecar)

    report = {
        "script": "scripts/audit_routeD_cotracker3_interface.py",
        "output": str(output),
        "tensor_sidecar": str(sidecar),
        "provenance": provenance,
        "sample_shape": {
            "frames": int(prepared["video"].shape[1]),
            "points": int(prepared["gt_tracks_yx"].shape[0]),
            "candidate_count": int(args.candidate_topk + 1),
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "state_feature_dim": STATE_FEATURE_DIM,
        },
        "routing_disabled_native_parity": {
            "requested": bool(args.routing_disabled),
            "exact": routing_disabled_exact,
            "candidate_zero_valid_everywhere": candidate_zero_valid,
            "max_abs_xy_px": routing_disabled_max_abs,
            "pass": parity_pass,
        },
        "deterministic_replay": {
            "scope": "adapter_export_from_one_frozen_native_backbone_state",
            "replays": int(args.deterministic_replays),
            "exact": deterministic_exact if comparisons else None,
            "seconds": replay_seconds,
            "hashes": replay_hashes,
            "comparisons": comparisons,
        },
        "oracle_audit": audit,
        "stage0_gate": {
            "oracle_AJ_gain_points_required": float(args.oracle_aj_gate_points),
            "oracle_AJ_gain_points_observed": float(audit["gain_points"]["AJ"]),
            "pass": oracle_gate_pass,
            "decision": decision,
            "training_started": False,
        },
        "integrity": {
            "candidate_generation_ground_truth_free": True,
            "kinetics_read": False,
            "davis_read": False,
            "sidecar_sha256": None,
        },
    }
    output.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    report["integrity"]["sidecar_sha256"] = file_sha256(str(sidecar))
    output.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output),
        "tensor_sidecar": str(sidecar),
        "routing_disabled_native_parity": report["routing_disabled_native_parity"],
        "deterministic_exact": deterministic_exact,
        "native_AJ": audit["native_metrics"]["AJ"],
        "oracle_AJ": audit["coordinate_oracle_same_visibility_metrics"]["AJ"],
        "oracle_AJ_gain_points": audit["gain_points"]["AJ"],
        "oracle_delta_gain_points": audit["gain_points"]["delta_average"],
        "stage0_decision": decision,
    }
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    if comparisons and not deterministic_exact:
        raise RuntimeError("deterministic replay tensor hashes differ")


if __name__ == "__main__":
    main()
