"""Frozen CoTracker feature-map cache utilities for Route-D CMCP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .cotracker3_stage0_adapter import tensor_sha256
from .routeD_kubric_cache import file_sha256


CMCP_FEATURE_CACHE_SCHEMA_VERSION = "routeD_cmcp_feature_map_cache_v1"
CMCP_FEATURE_INDEX_SCHEMA_VERSION = "routeD_cmcp_feature_map_cache_index_v1"


def feature_quantization_audit(
    feature_maps_float32: torch.Tensor,
    feature_maps_float16: torch.Tensor,
) -> dict[str, float]:
    if feature_maps_float32.dtype != torch.float32:
        raise ValueError("feature_maps_float32 must be float32")
    if feature_maps_float16.dtype != torch.float16:
        raise ValueError("feature_maps_float16 must be float16")
    if feature_maps_float32.shape != feature_maps_float16.shape:
        raise ValueError("feature map shape mismatch")
    reconstructed = feature_maps_float16.float()
    absolute = (feature_maps_float32 - reconstructed).abs()
    cosine = F.cosine_similarity(feature_maps_float32, reconstructed, dim=1, eps=1e-12)
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rms_abs": float(absolute.square().mean().sqrt().item()),
        "min_cosine": float(cosine.min().item()),
        "mean_cosine": float(cosine.mean().item()),
    }


def validate_quantization_audit(
    audit: Mapping[str, float],
    *,
    max_abs_max: float = 5.0e-4,
    min_cosine_min: float = 0.99999,
) -> None:
    if float(audit["max_abs"]) > float(max_abs_max):
        raise ValueError(f"feature-map fp16 max_abs too large: {audit['max_abs']}")
    if float(audit["min_cosine"]) < float(min_cosine_min):
        raise ValueError(f"feature-map fp16 min cosine too small: {audit['min_cosine']}")


def verify_feature_cache_artifact(
    sidecar_path: str | Path,
    *,
    expected_protocol_sha256: str | None = None,
    expected_base_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(sidecar_path).resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != CMCP_FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unexpected CMCP feature cache schema: {path}")
    feature = artifact.get("feature_maps_f16")
    if not isinstance(feature, torch.Tensor) or feature.dtype != torch.float16:
        raise ValueError("feature cache must contain float16 feature_maps_f16")
    provenance = artifact.get("provenance", {})
    if expected_protocol_sha256 is not None and provenance.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("feature cache protocol mismatch")
    if expected_base_sidecar_sha256 is not None and provenance.get("base_sidecar_sha256") != expected_base_sidecar_sha256:
        raise ValueError("feature cache base-sidecar mismatch")
    if artifact.get("feature_maps_f16_sha256") != tensor_sha256(feature):
        raise ValueError("feature cache tensor hash mismatch")
    return artifact


def load_complete_feature_index(
    index_path: str | Path,
    *,
    expected_partition: str | None = None,
) -> dict[str, Any]:
    path = Path(index_path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CMCP_FEATURE_INDEX_SCHEMA_VERSION:
        raise ValueError("unexpected CMCP feature index schema")
    if expected_partition is not None and payload.get("partition") != expected_partition:
        raise ValueError("CMCP feature index partition mismatch")
    if not payload.get("complete"):
        raise ValueError("CMCP feature index is incomplete")
    if payload.get("expected_source_indices") != payload.get("completed_source_indices"):
        raise ValueError("CMCP feature index membership mismatch")
    if int(payload.get("expected_count", -1)) != int(payload.get("completed_count", -2)):
        raise ValueError("CMCP feature index count mismatch")
    payload["_index_path"] = str(path)
    payload["_index_sha256"] = file_sha256(path)
    return payload
