from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


CACHE_SCHEMA_VERSION = 1


TOP_LEVEL_REQUIRED_KEYS = (
    "model_name",
    "repo_commit",
    "checkpoint_path",
    "dataset_name",
    "split",
    "protocol",
    "records",
)


RECORD_REQUIRED_KEYS = (
    "video_id",
    "sequence_index",
    "frame_count",
    "query_points",
    "pred_tracks",
    "pred_visibility",
    "original_size",
    "adapter_version",
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_int_tuple_2(value: Any) -> Tuple[int, int]:
    arr = _to_numpy(value).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected 2 values, got shape={_to_numpy(value).shape}")
    return int(arr[0]), int(arr[1])


def load_attempt0_cache(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
        return payload
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
        return payload
    raise ValueError(f"Unsupported cache extension: {path.suffix}")


def save_attempt0_cache(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        torch.save(payload, str(path))
        return
    if suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return
    raise ValueError(f"Unsupported cache extension: {path.suffix}")


def build_attempt0_status_skeleton(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "repo_url": "",
        "repo_commit": "",
        "checkpoint_path": "",
        "checkpoint_sha256": "",
        "dataset_name": "",
        "split": "",
        "run_date": "",
        "repo_native_protocol": "",
        "repo_native_metric_names": {},
        "repo_native_numbers": {},
        "official_reference_numbers": {},
        "delta_vs_official": {},
        "reproduction_status": "pending",
        "raw_coordinate_format": "",
        "raw_visibility_format": "",
        "query_format": "",
        "adapter_version": "",
        "adapter_sanity_status": "pending",
        "adapter_notes": [],
        "query_mode": "",
        "metric_resolution_mode": "",
        "AJ": None,
        "OA": None,
        "<avg": None,
        "<4px": None,
        "delta_vs_cotracker3_baseline": {},
        "rescoring_status": "pending",
        "long_occ_definition": "",
        "long_occ_AJ": None,
        "long_occ_<avg": None,
        "long_occ_<4px": None,
        "reentry_first_frame_error": None,
        "AJ_RD": None,
        "gpu_type": "",
        "batch_size": None,
        "peak_memory_gb": None,
        "eval_wall_time": None,
        "notes": [],
        "keep_for_main_ranking": None,
        "keep_as_teacher_candidate": None,
        "known_blockers": [],
        "next_action": "",
    }


def validate_attempt0_cache(
    payload: Dict[str, Any],
    *,
    require_gt: bool = False,
    normalized_coord_threshold: float = 1.5,
    query_anchor_tol_px: float = 2.0,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    per_record: List[Dict[str, Any]] = []

    for key in TOP_LEVEL_REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"Missing top-level key: {key}")

    records = payload.get("records", [])
    if not isinstance(records, list):
        errors.append("Top-level key 'records' must be a list.")
        records = []

    total_queries = 0

    for idx, record in enumerate(records):
        record_errors: List[str] = []
        record_warnings: List[str] = []
        for key in RECORD_REQUIRED_KEYS:
            if key not in record:
                record_errors.append(f"Missing record key: {key}")

        if record_errors:
            per_record.append(
                {
                    "record_index": idx,
                    "video_id": record.get("video_id", f"unknown_{idx}"),
                    "valid": False,
                    "errors": record_errors,
                    "warnings": record_warnings,
                }
            )
            errors.extend([f"[record {idx}] {msg}" for msg in record_errors])
            continue

        query_points = _to_numpy(record["query_points"]).astype(np.float32)
        pred_tracks = _to_numpy(record["pred_tracks"]).astype(np.float32)
        pred_visibility = _to_numpy(record["pred_visibility"])
        original_h, original_w = _as_int_tuple_2(record["original_size"])

        if query_points.ndim != 2 or query_points.shape[1] != 3:
            record_errors.append(f"query_points must be (N,3), got {query_points.shape}")
        if pred_tracks.ndim != 3 or pred_tracks.shape[-1] != 2:
            record_errors.append(f"pred_tracks must be (N,T,2), got {pred_tracks.shape}")
        if pred_visibility.ndim != 2:
            record_errors.append(f"pred_visibility must be (N,T), got {pred_visibility.shape}")

        n_queries = int(query_points.shape[0]) if query_points.ndim == 2 else 0
        t_frames = int(pred_tracks.shape[1]) if pred_tracks.ndim == 3 else 0
        total_queries += max(n_queries, 0)

        if pred_tracks.ndim == 3 and query_points.ndim == 2 and pred_tracks.shape[0] != query_points.shape[0]:
            record_errors.append(
                f"N mismatch: query_points={query_points.shape[0]} pred_tracks={pred_tracks.shape[0]}"
            )
        if pred_visibility.ndim == 2 and query_points.ndim == 2 and pred_visibility.shape[0] != query_points.shape[0]:
            record_errors.append(
                f"N mismatch: query_points={query_points.shape[0]} pred_visibility={pred_visibility.shape[0]}"
            )
        if pred_visibility.ndim == 2 and pred_tracks.ndim == 3 and pred_visibility.shape[1] != pred_tracks.shape[1]:
            record_errors.append(
                f"T mismatch: pred_tracks={pred_tracks.shape[1]} pred_visibility={pred_visibility.shape[1]}"
            )

        if pred_tracks.size > 0 and float(np.nanmax(np.abs(pred_tracks))) > normalized_coord_threshold:
            record_errors.append(
                f"pred_tracks appear non-normalized; max_abs={float(np.nanmax(np.abs(pred_tracks))):.3f}"
            )
        if query_points.size > 0 and float(np.nanmax(np.abs(query_points[:, 1:3]))) > normalized_coord_threshold:
            record_errors.append(
                f"query_points appear non-normalized; max_abs={float(np.nanmax(np.abs(query_points[:, 1:3]))):.3f}"
            )

        vis_arr = pred_visibility.astype(np.float32)
        if vis_arr.size > 0 and (float(np.nanmin(vis_arr)) < 0.0 or float(np.nanmax(vis_arr)) > 1.0):
            record_errors.append(
                f"pred_visibility out of [0,1]; min={float(np.nanmin(vis_arr)):.3f} max={float(np.nanmax(vis_arr)):.3f}"
            )

        anchor_mean_error_px = None
        anchor_max_error_px = None
        if not record_errors and n_queries > 0 and t_frames > 0:
            query_t = np.clip(np.round(query_points[:, 0]).astype(np.int64), 0, t_frames - 1)
            anchor_pred = pred_tracks[np.arange(n_queries), query_t]
            anchor_gt = query_points[:, 1:3]
            scale = np.array([float(original_h - 1), float(original_w - 1)], dtype=np.float32)
            anchor_err = np.linalg.norm((anchor_pred - anchor_gt) * scale[None, :], axis=-1)
            anchor_mean_error_px = float(anchor_err.mean()) if anchor_err.size > 0 else 0.0
            anchor_max_error_px = float(anchor_err.max()) if anchor_err.size > 0 else 0.0
            if anchor_max_error_px > query_anchor_tol_px:
                record_warnings.append(
                    f"query-anchor max error {anchor_max_error_px:.3f}px exceeds tolerance {query_anchor_tol_px:.3f}px"
                )

        if require_gt:
            if "gt_tracks" not in record or "gt_visibility" not in record:
                record_errors.append("GT fields required but missing: gt_tracks / gt_visibility")
            else:
                gt_tracks = _to_numpy(record["gt_tracks"]).astype(np.float32)
                gt_visibility = _to_numpy(record["gt_visibility"])
                if gt_tracks.shape != pred_tracks.shape:
                    record_errors.append(f"gt_tracks shape {gt_tracks.shape} != pred_tracks shape {pred_tracks.shape}")
                if gt_visibility.shape != pred_visibility.shape:
                    record_errors.append(
                        f"gt_visibility shape {gt_visibility.shape} != pred_visibility shape {pred_visibility.shape}"
                    )
                if gt_tracks.size > 0 and float(np.nanmax(np.abs(gt_tracks))) > normalized_coord_threshold:
                    record_errors.append(
                        f"gt_tracks appear non-normalized; max_abs={float(np.nanmax(np.abs(gt_tracks))):.3f}"
                    )

        record_valid = len(record_errors) == 0
        if record_errors:
            errors.extend([f"[record {idx}] {msg}" for msg in record_errors])
        if record_warnings:
            warnings.extend([f"[record {idx}] {msg}" for msg in record_warnings])

        per_record.append(
            {
                "record_index": idx,
                "video_id": str(record.get("video_id", f"unknown_{idx}")),
                "valid": record_valid,
                "errors": record_errors,
                "warnings": record_warnings,
                "num_queries": n_queries,
                "num_frames": t_frames,
                "original_size_hw": [original_h, original_w],
                "anchor_mean_error_px": anchor_mean_error_px,
                "anchor_max_error_px": anchor_max_error_px,
            }
        )

    return {
        "schema_version": int(payload.get("schema_version", CACHE_SCHEMA_VERSION)),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "num_records": len(records),
            "num_queries": total_queries,
            "require_gt": bool(require_gt),
        },
        "per_record": per_record,
    }


def write_json_report(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
