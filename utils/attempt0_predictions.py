from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np


SCHEMA_VERSION = "attempt0.prediction_cache.v1"


def safe_video_id(video_id: str) -> str:
    text = str(video_id).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or "unknown_video"


def _as_float32(name: str, value: Any, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={arr.shape}")
    return arr


def _as_bool(name: str, value: Any, ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={arr.shape}")
    if arr.dtype != np.bool_:
        arr = arr > 0.5
    return arr.astype(np.bool_, copy=False)


def xy_pixels_to_yx_normalized(points_xy: Any, original_size_hw: Tuple[int, int]) -> np.ndarray:
    arr = _as_float32("points_xy", points_xy, ndim=arr_ndim(points_xy))
    if arr.shape[-1] != 2:
        raise ValueError(f"points_xy last dim must be 2, got shape={arr.shape}")
    height, width = _validate_original_size(original_size_hw)
    out = arr[..., [1, 0]].copy()
    out[..., 0] /= float(height)
    out[..., 1] /= float(width)
    return out


def txy_pixels_to_tyx_normalized(query_points_txy: Any, original_size_hw: Tuple[int, int]) -> np.ndarray:
    arr = _as_float32("query_points_txy", query_points_txy, ndim=2)
    if arr.shape[1] != 3:
        raise ValueError(f"query_points_txy must have shape (N, 3), got {arr.shape}")
    height, width = _validate_original_size(original_size_hw)
    out = arr.copy()
    t = out[:, 0:1]
    x = out[:, 1:2] / float(width)
    y = out[:, 2:3] / float(height)
    return np.concatenate([t, y, x], axis=1).astype(np.float32, copy=False)


def _validate_original_size(original_size_hw: Any) -> Tuple[int, int]:
    if isinstance(original_size_hw, np.ndarray):
        size = original_size_hw.tolist()
    else:
        size = list(original_size_hw)
    if len(size) < 2:
        raise ValueError(f"original_size_hw must have length >= 2, got {original_size_hw}")
    height = int(size[0])
    width = int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"original_size_hw must be positive, got {(height, width)}")
    return height, width


def arr_ndim(value: Any) -> int:
    return np.asarray(value).ndim


def build_prediction_payload(
    *,
    model_name: str,
    dataset_name: str,
    split: str,
    protocol: str,
    video_id: str,
    sequence_index: int,
    frame_count: int,
    query_points_tyx_norm: Any,
    pred_tracks_yx_norm: Any,
    pred_visibility_bool: Any,
    original_size_hw: Any,
    repo_commit: str = "",
    checkpoint_path: str = "",
    checkpoint_sha256: str = "",
    model_input_size_hw: Optional[Any] = None,
    adapter_version: str = "",
    raw_coordinate_note: str = "",
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    query_points = _as_float32("query_points_tyx_norm", query_points_tyx_norm, ndim=2)
    pred_tracks = _as_float32("pred_tracks_yx_norm", pred_tracks_yx_norm, ndim=3)
    pred_visibility = _as_bool("pred_visibility_bool", pred_visibility_bool, ndim=2)
    if query_points.shape[1] != 3:
        raise ValueError(f"query_points must have shape (N, 3), got {query_points.shape}")
    if pred_tracks.shape[-1] != 2:
        raise ValueError(f"pred_tracks last dim must be 2, got {pred_tracks.shape}")
    if pred_tracks.shape[:2] != pred_visibility.shape:
        raise ValueError(
            "pred_tracks and pred_visibility shape mismatch: "
            f"{pred_tracks.shape[:2]} vs {pred_visibility.shape}"
        )
    if pred_tracks.shape[0] != query_points.shape[0]:
        raise ValueError(
            "query_points and pred_tracks query count mismatch: "
            f"{query_points.shape[0]} vs {pred_tracks.shape[0]}"
        )

    original_size = np.asarray(_validate_original_size(original_size_hw), dtype=np.int32)
    if model_input_size_hw is None:
        model_input_size = original_size.copy()
    else:
        model_input_size = np.asarray(_validate_original_size(model_input_size_hw), dtype=np.int32)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_name": str(model_name),
        "repo_commit": str(repo_commit),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_sha256),
        "dataset_name": str(dataset_name),
        "split": str(split),
        "protocol": str(protocol),
        "video_id": str(video_id),
        "sequence_index": int(sequence_index),
        "frame_count": int(frame_count),
        "original_size_hw": original_size,
        "model_input_size_hw": model_input_size,
        "query_points_tyx_norm": query_points.astype(np.float32, copy=False),
        "pred_tracks_yx_norm": pred_tracks.astype(np.float32, copy=False),
        "pred_visibility_bool": pred_visibility.astype(np.bool_, copy=False),
        "adapter_version": str(adapter_version),
        "raw_coordinate_note": str(raw_coordinate_note),
        "extra_meta": dict(extra_meta or {}),
    }
    validate_prediction_payload(payload)
    return payload


def validate_prediction_payload(payload: Mapping[str, Any]) -> None:
    required = [
        "schema_version",
        "model_name",
        "dataset_name",
        "split",
        "protocol",
        "video_id",
        "sequence_index",
        "frame_count",
        "original_size_hw",
        "model_input_size_hw",
        "query_points_tyx_norm",
        "pred_tracks_yx_norm",
        "pred_visibility_bool",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Prediction payload missing keys: {missing}")
    if str(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={payload['schema_version']}; expected {SCHEMA_VERSION}"
        )

    query_points = _as_float32("query_points_tyx_norm", payload["query_points_tyx_norm"], ndim=2)
    pred_tracks = _as_float32("pred_tracks_yx_norm", payload["pred_tracks_yx_norm"], ndim=3)
    pred_visibility = _as_bool("pred_visibility_bool", payload["pred_visibility_bool"], ndim=2)
    original_size = _validate_original_size(payload["original_size_hw"])
    _validate_original_size(payload["model_input_size_hw"])

    if query_points.shape[1] != 3:
        raise ValueError(f"query_points must have shape (N, 3), got {query_points.shape}")
    if pred_tracks.shape[-1] != 2:
        raise ValueError(f"pred_tracks must end with 2, got shape={pred_tracks.shape}")
    if pred_tracks.shape[:2] != pred_visibility.shape:
        raise ValueError(
            f"pred_tracks/pred_visibility mismatch: {pred_tracks.shape[:2]} vs {pred_visibility.shape}"
        )
    if query_points.shape[0] != pred_tracks.shape[0]:
        raise ValueError(
            f"query count mismatch: query_points={query_points.shape[0]} pred_tracks={pred_tracks.shape[0]}"
        )

    num_frames = pred_tracks.shape[1]
    if int(payload["frame_count"]) != num_frames:
        raise ValueError(f"frame_count={payload['frame_count']} != pred_tracks.shape[1]={num_frames}")

    query_t = np.round(query_points[:, 0]).astype(np.int64)
    if query_t.size > 0 and (query_t.min() < 0 or query_t.max() >= num_frames):
        raise ValueError(
            f"query frame indices out of range: min={query_t.min()} max={query_t.max()} num_frames={num_frames}"
        )

    for name, arr in [("query_points", query_points[:, 1:3]), ("pred_tracks", pred_tracks)]:
        if arr.size == 0:
            continue
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        if arr_min < -1e-4 or arr_max > 1.0001:
            raise ValueError(f"{name} expected normalized [0,1], got min={arr_min:.6f} max={arr_max:.6f}")

    height, width = original_size
    if height <= 1 or width <= 1:
        raise ValueError(f"original_size_hw too small: {(height, width)}")


def compute_query_alignment_report(payload: Mapping[str, Any], tolerance_px: float = 0.5) -> Dict[str, Any]:
    validate_prediction_payload(payload)
    query_points = np.asarray(payload["query_points_tyx_norm"], dtype=np.float32)
    pred_tracks = np.asarray(payload["pred_tracks_yx_norm"], dtype=np.float32)
    height, width = _validate_original_size(payload["original_size_hw"])
    if query_points.shape[0] == 0:
        return {
            "num_queries": 0,
            "mean_error_px": 0.0,
            "max_error_px": 0.0,
            "p95_error_px": 0.0,
            "tolerance_px": float(tolerance_px),
            "passes": True,
        }

    q_t = np.round(query_points[:, 0]).astype(np.int64)
    q_yx = query_points[:, 1:3]
    track_yx = pred_tracks[np.arange(pred_tracks.shape[0]), q_t]
    delta = track_yx - q_yx
    delta_px = delta * np.array([height, width], dtype=np.float32)
    err_px = np.sqrt(np.sum(np.square(delta_px), axis=-1))
    return {
        "num_queries": int(err_px.shape[0]),
        "mean_error_px": float(err_px.mean()) if err_px.size > 0 else 0.0,
        "max_error_px": float(err_px.max()) if err_px.size > 0 else 0.0,
        "p95_error_px": float(np.percentile(err_px, 95)) if err_px.size > 0 else 0.0,
        "tolerance_px": float(tolerance_px),
        "passes": bool(err_px.size == 0 or np.max(err_px) <= float(tolerance_px)),
    }


def save_prediction_npz(path: Any, payload: Mapping[str, Any]) -> Path:
    validate_prediction_payload(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "model_name": str(payload["model_name"]),
        "repo_commit": str(payload.get("repo_commit", "")),
        "checkpoint_path": str(payload.get("checkpoint_path", "")),
        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
        "dataset_name": str(payload["dataset_name"]),
        "split": str(payload["split"]),
        "protocol": str(payload["protocol"]),
        "video_id": str(payload["video_id"]),
        "sequence_index": int(payload["sequence_index"]),
        "frame_count": int(payload["frame_count"]),
        "adapter_version": str(payload.get("adapter_version", "")),
        "raw_coordinate_note": str(payload.get("raw_coordinate_note", "")),
        "extra_meta": dict(payload.get("extra_meta", {})),
    }
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps(meta, ensure_ascii=True), dtype=np.str_),
        original_size_hw=np.asarray(payload["original_size_hw"], dtype=np.int32),
        model_input_size_hw=np.asarray(payload["model_input_size_hw"], dtype=np.int32),
        query_points_tyx_norm=np.asarray(payload["query_points_tyx_norm"], dtype=np.float32),
        pred_tracks_yx_norm=np.asarray(payload["pred_tracks_yx_norm"], dtype=np.float32),
        pred_visibility_bool=np.asarray(payload["pred_visibility_bool"], dtype=np.bool_),
    )
    return path


def load_prediction_npz(path: Any) -> Dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta_raw = data["meta_json"]
        if isinstance(meta_raw, np.ndarray):
            meta_text = str(meta_raw.reshape(-1)[0])
        else:
            meta_text = str(meta_raw)
        meta = json.loads(meta_text)
        payload = {
            **meta,
            "original_size_hw": np.asarray(data["original_size_hw"], dtype=np.int32),
            "model_input_size_hw": np.asarray(data["model_input_size_hw"], dtype=np.int32),
            "query_points_tyx_norm": np.asarray(data["query_points_tyx_norm"], dtype=np.float32),
            "pred_tracks_yx_norm": np.asarray(data["pred_tracks_yx_norm"], dtype=np.float32),
            "pred_visibility_bool": np.asarray(data["pred_visibility_bool"], dtype=np.bool_),
        }
    validate_prediction_payload(payload)
    return payload


def write_json(path: Any, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def mean_numeric_dict(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        values = []
        for row in rows:
            value = row.get(key)
            try:
                if value is None or isinstance(value, (str, bool)):
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    values.append(value_f)
            except Exception:
                continue
        if values:
            out[key] = float(np.mean(values))
    return out
