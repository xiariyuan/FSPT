"""Verified sparse event cache for Route-D safe re-detection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

SAFE_REDETECTION_CACHE_SCHEMA = "routeD_safe_redetection_event_cache_v0"
SAFE_REDETECTION_INDEX_SCHEMA = "routeD_safe_redetection_event_index_v0"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_event_cache(
    path: str | Path,
    *,
    expected_protocol_sha256: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SAFE_REDETECTION_CACHE_SCHEMA:
        raise ValueError(f"unexpected cache schema: {source}")
    provenance = payload.get("provenance", {})
    if expected_protocol_sha256 and provenance.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("protocol hash mismatch")
    if expected_role and provenance.get("role") != expected_role:
        raise ValueError("partition role mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("cache tensors missing")
    required = {
        "feature_maps_f16",
        "frame_indices_abs",
        "frame_indices_rel",
        "native_coords_xy_px",
        "native_visibility_probability",
        "native_confidence_probability",
        "native_visibility",
        "gt_coords_xy_px",
        "gt_visible",
        "occlusion_age",
        "query_frame_rel",
        "query_coord_xy_px",
        "reappearance_frame_rel",
        "last_visible_frame_rel",
    }
    missing = required - tensors.keys()
    if missing:
        raise ValueError(f"missing tensors: {sorted(missing)}")
    rows = tensors["frame_indices_abs"].numel()
    if rows <= 0:
        raise ValueError("empty cache")
    for key in (
        "frame_indices_rel",
        "native_visibility_probability",
        "native_confidence_probability",
        "native_visibility",
        "gt_visible",
        "occlusion_age",
    ):
        if tensors[key].shape != (rows,):
            raise ValueError(f"row shape mismatch: {key}")
    for key in ("native_coords_xy_px", "gt_coords_xy_px"):
        if tensors[key].shape != (rows, 2):
            raise ValueError(f"coordinate shape mismatch: {key}")
    if tensors["feature_maps_f16"].shape[0] != rows:
        raise ValueError("feature map row mismatch")
    hashes = payload.get("tensor_sha256", {})
    for key, tensor in tensors.items():
        if key not in hashes or hashes[key] != tensor_sha256(tensor):
            raise ValueError(f"tensor hash mismatch: {key}")
    if not torch.equal(tensors["frame_indices_abs"], torch.sort(tensors["frame_indices_abs"]).values):
        raise ValueError("frame indices are not sorted")
    return payload


def load_complete_event_index(
    path: str | Path,
    *,
    expected_role: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    index = json.loads(source.read_text())
    if index.get("schema_version") != SAFE_REDETECTION_INDEX_SCHEMA:
        raise ValueError("unexpected event index schema")
    if index.get("role") != expected_role:
        raise ValueError("index role mismatch")
    if not index.get("complete"):
        raise ValueError("event index incomplete")
    if index.get("completed_count") != index.get("expected_count"):
        raise ValueError("event count mismatch")
    for row in index.get("events", []):
        sidecar = Path(row["sidecar"])
        if file_sha256(sidecar) != row["sidecar_sha256"]:
            raise ValueError(f"sidecar hash mismatch: {sidecar}")
        verify_event_cache(
            sidecar,
            expected_protocol_sha256=index["protocol_sha256"],
            expected_role=expected_role,
        )
    index["_index_sha256"] = file_sha256(source)
    return index
