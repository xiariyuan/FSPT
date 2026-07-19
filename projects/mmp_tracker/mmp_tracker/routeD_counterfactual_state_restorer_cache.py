"""Sealed teacher-cache contracts for Route-D CSRR Gate 2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .cotracker3_stage0_adapter import tensor_sha256
from .routeD_counterfactual_state_restoration import CoTrackerOnlineStateSnapshot
from .routeD_kubric_cache import canonical_json_sha256, file_sha256


CSRR_CACHE_SCHEMA_VERSION = "routeD_csrr_teacher_cache_v0"
CSRR_CACHE_INDEX_SCHEMA_VERSION = "routeD_csrr_teacher_cache_index_v0"


def snapshot_to_cache_dict(snapshot: CoTrackerOnlineStateSnapshot) -> dict[str, Any]:
    """Serialize the exact float32 state needed for future continuation."""
    tensors: dict[str, Any] = {
        "predictor_queries": snapshot.predictor_queries.detach().cpu().float().contiguous(),
        "online_track_feat": [
            None if value is None else value.detach().cpu().float().contiguous()
            for value in snapshot.online_track_feat
        ],
        "online_track_support": [
            None if value is None else value.detach().cpu().float().contiguous()
            for value in snapshot.online_track_support
        ],
        "online_coords_predicted": snapshot.online_coords_predicted.detach().cpu().float().contiguous(),
        "online_vis_predicted": snapshot.online_vis_predicted.detach().cpu().float().contiguous(),
        "online_conf_predicted": snapshot.online_conf_predicted.detach().cpu().float().contiguous(),
    }
    return {
        "predictor_n": int(snapshot.predictor_n),
        "online_ind": int(snapshot.online_ind),
        "tensors": tensors,
    }


def snapshot_from_cache_dict(payload: Mapping[str, Any]) -> CoTrackerOnlineStateSnapshot:
    tensors = payload["tensors"]
    return CoTrackerOnlineStateSnapshot(
        predictor_n=int(payload["predictor_n"]),
        predictor_queries=tensors["predictor_queries"].clone(),
        online_ind=int(payload["online_ind"]),
        online_track_feat=tuple(
            None if value is None else value.clone() for value in tensors["online_track_feat"]
        ),
        online_track_support=tuple(
            None if value is None else value.clone()
            for value in tensors["online_track_support"]
        ),
        online_coords_predicted=tensors["online_coords_predicted"].clone(),
        online_vis_predicted=tensors["online_vis_predicted"].clone(),
        online_conf_predicted=tensors["online_conf_predicted"].clone(),
    )


def iter_nested_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_nested_tensors(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_nested_tensors(item, child)


def nested_tensor_hashes(value: Any) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in iter_nested_tensors(value)}


def nested_tensor_hash_digest(value: Any) -> str:
    return canonical_json_sha256(nested_tensor_hashes(value))


def quantization_measurement(original: torch.Tensor, quantized: torch.Tensor) -> dict[str, float]:
    left = original.detach().float().cpu().contiguous()
    right = quantized.detach().float().cpu().contiguous()
    difference = (left - right).abs()
    cosine = torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0)
    return {
        "max_absolute_error": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_absolute_error": float(difference.mean().item()) if difference.numel() else 0.0,
        "cosine_similarity": float(cosine.item()) if difference.numel() else 1.0,
    }


def select_gate2_rows(
    *,
    native_future_coords_xy_px: torch.Tensor,
    gt_tracks_yx: torch.Tensor,
    gt_occluded: torch.Tensor,
    original_query_frames: torch.Tensor,
    failure_error_min_px: float,
    clean_error_max_px: float,
    min_future_visible_frames: int,
    per_class_cap: int,
) -> dict[str, torch.Tensor]:
    """Apply fixed Gate 2 failure and clean-noop selection at commit frame 15."""
    native = native_future_coords_xy_px.float().cpu()
    gt_xy = gt_tracks_yx[..., [1, 0]].float().cpu() * 255.0
    occluded = gt_occluded.bool().cpu()
    query = original_query_frames.round().long().cpu()
    if native.shape != gt_xy[:, 16:24].shape:
        raise ValueError("future/native shape mismatch")
    future_visible = ~occluded[:, 16:24]
    visible_count = future_visible.sum(dim=1)
    error = torch.linalg.vector_norm(native - gt_xy[:, 16:24], dim=-1)
    mean_error = (error * future_visible.float()).sum(dim=1) / visible_count.clamp_min(1)
    common = (query < 8) & (~occluded[:, 15]) & (
        visible_count >= int(min_future_visible_frames)
    )
    failure_eligible = common & (mean_error >= float(failure_error_min_px))
    clean_eligible = common & (mean_error <= float(clean_error_max_px))
    failure_order = sorted(
        torch.where(failure_eligible)[0].tolist(),
        key=lambda index: (-float(mean_error[index]), index),
    )
    clean_order = sorted(
        torch.where(clean_eligible)[0].tolist(),
        key=lambda index: (float(mean_error[index]), index),
    )
    failure = torch.tensor(failure_order[: int(per_class_cap)], dtype=torch.long)
    clean = torch.tensor(clean_order[: int(per_class_cap)], dtype=torch.long)
    return {
        "failure_point_indices": failure,
        "clean_point_indices": clean,
        "failure_native_future_mean_error_px": mean_error[failure],
        "clean_native_future_mean_error_px": mean_error[clean],
        "failure_eligible_count": torch.tensor(int(failure_eligible.sum()), dtype=torch.long),
        "clean_eligible_count": torch.tensor(int(clean_eligible.sum()), dtype=torch.long),
        "future_visible_count": visible_count,
    }


def _require_tensor(
    tensors: Mapping[str, Any], key: str, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    value = tensors.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"missing tensor: {key}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"unexpected dtype for {key}: {value.dtype}")
    return value


def verify_csrr_cache_artifact(
    path: str | Path,
    *,
    expected_partition: str | None = None,
    expected_source_index: int | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    sidecar = Path(path).resolve()
    artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != CSRR_CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected CSRR cache schema")
    if expected_partition is not None and artifact.get("partition") != expected_partition:
        raise ValueError("CSRR cache partition mismatch")
    if expected_source_index is not None and int(artifact.get("source_index", -1)) != int(
        expected_source_index
    ):
        raise ValueError("CSRR cache source-index mismatch")
    provenance = artifact.get("provenance", {})
    if expected_config_sha256 is not None and provenance.get("config_sha256") != expected_config_sha256:
        raise ValueError("CSRR config hash mismatch")
    exact = artifact.get("exact_native_state", {})
    restored = snapshot_from_cache_dict(exact)
    if restored.predictor_n != 64 or restored.online_ind != 8:
        raise ValueError("unexpected exact native state identity")
    for _, tensor in iter_nested_tensors(exact):
        if tensor.dtype != torch.float32:
            raise ValueError("exact rollout state must remain float32")
    tensors = artifact.get("model_tensors", {})
    video = _require_tensor(tensors, "continuation_video_u8", dtype=torch.uint8)
    if tuple(video.shape) != (16, 3, 256, 256):
        raise ValueError("unexpected continuation video shape")
    point_indices = _require_tensor(tensors, "point_indices", dtype=torch.int64)
    labels = _require_tensor(tensors, "apply_target", dtype=torch.float32)
    trajectory = _require_tensor(tensors, "trajectory_features", dtype=torch.float16)
    if trajectory.shape != (point_indices.numel(), 8, 9):
        raise ValueError("trajectory cache shape mismatch")
    if labels.shape != (point_indices.numel(),):
        raise ValueError("apply target shape mismatch")
    for key in (
        "native_commit_coordinates_xy",
        "teacher_commit_coordinates_xy",
        "native_visibility_logits",
        "native_confidence_logits",
        "teacher_visibility_logits",
        "teacher_confidence_logits",
    ):
        tensor = _require_tensor(tensors, key)
        if tensor.dtype != torch.float16:
            raise ValueError(f"model view must be float16: {key}")
    for key in (
        "frame_feature_pyramid",
        "native_track_feat",
        "native_track_support",
        "teacher_track_feat",
        "teacher_track_support",
    ):
        values = tensors.get(key)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"{key} must contain four levels")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float16 for value in values):
            raise ValueError(f"{key} levels must be float16 tensors")
    hashes = artifact.get("tensor_hashes", {})
    observed_hashes = nested_tensor_hashes(
        {"exact_native_state": exact, "model_tensors": tensors}
    )
    if hashes != observed_hashes:
        raise ValueError("CSRR nested tensor hash mismatch")
    quantization = artifact.get("quantization", {})
    if not bool(quantization.get("pass")):
        raise ValueError("CSRR cache quantization gate failed")
    return artifact


def load_complete_csrr_cache_index(
    path: str | Path, *, expected_partition: str | None = None
) -> dict[str, Any]:
    index_path = Path(path).resolve()
    payload = json.loads(index_path.read_text())
    if payload.get("schema_version") != CSRR_CACHE_INDEX_SCHEMA_VERSION:
        raise ValueError("unexpected CSRR cache-index schema")
    if expected_partition is not None and payload.get("partition") != expected_partition:
        raise ValueError("CSRR cache-index partition mismatch")
    if not payload.get("complete"):
        raise ValueError("CSRR cache index is incomplete")
    if payload.get("expected_source_indices") != payload.get("completed_source_indices"):
        raise ValueError("CSRR cache membership mismatch")
    rows = payload.get("rows", [])
    if len(rows) != len(payload["expected_source_indices"]):
        raise ValueError("CSRR cache row-count mismatch")
    payload["_index_path"] = str(index_path)
    payload["_index_sha256"] = file_sha256(index_path)
    return payload
