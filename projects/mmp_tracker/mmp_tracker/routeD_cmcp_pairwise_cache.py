"""Frozen local-token cache utilities for Route-D CMCP P0h."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .cotracker3_stage0_adapter import tensor_sha256
from .routeD_cmcp_pairwise_safety import CMCP_PAIRWISE_LOCAL_TOKEN_DIM
from .routeD_kubric_cache import file_sha256


CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION = "routeD_cmcp_pairwise_token_cache_v0"
CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION = "routeD_cmcp_pairwise_token_cache_index_v0"
CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM = 4
CMCP_PAIRWISE_STATIC_TOKEN_DIM = CMCP_PAIRWISE_LOCAL_TOKEN_DIM - CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM


def verify_pairwise_token_artifact(
    path: str | Path,
    *,
    expected_generator_state_sha256: str | None = None,
    expected_base_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    sidecar=Path(path).resolve()
    artifact=torch.load(sidecar,map_location="cpu",weights_only=False)
    if artifact.get("schema_version")!=CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected pairwise token cache schema")
    provenance=artifact.get("provenance",{})
    if expected_generator_state_sha256 is not None and provenance.get("generator_model_state_sha256")!=expected_generator_state_sha256:
        raise ValueError("generator state hash mismatch")
    if expected_base_sidecar_sha256 is not None and provenance.get("base_sidecar_sha256")!=expected_base_sidecar_sha256:
        raise ValueError("base sidecar hash mismatch")
    tensors=artifact.get("tensors",{})
    token=tensors.get("candidate_tokens")
    if not isinstance(token,torch.Tensor) or token.dtype!=torch.float32 or token.shape[-1]!=CMCP_PAIRWISE_LOCAL_TOKEN_DIM:
        raise ValueError("candidate_tokens must be float32 (...,88)")
    if torch.count_nonzero(token[...,-CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM:]).item()!=0:
        raise ValueError("dynamic summary placeholder must be zero in frozen cache")
    for key,value in tensors.items():
        if artifact["tensor_hashes"].get(key)!=tensor_sha256(value):
            raise ValueError(f"tensor hash mismatch: {key}")
    if not torch.equal(tensors["candidate_coords_xy_px"][...,0,:],tensors["native_coords_xy_px"]):
        raise ValueError("candidate-0/native parity drift")
    return artifact


def load_complete_pairwise_token_index(
    path: str | Path,
    *,
    expected_partition: str | None=None,
) -> dict[str,Any]:
    index_path=Path(path).resolve(); payload=json.loads(index_path.read_text())
    if payload.get("schema_version")!=CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION:
        raise ValueError("unexpected pairwise token index schema")
    if expected_partition is not None and payload.get("partition")!=expected_partition:
        raise ValueError("partition mismatch")
    if not payload.get("complete"):
        raise ValueError("pairwise token index is incomplete")
    if payload.get("expected_source_indices")!=payload.get("completed_source_indices"):
        raise ValueError("pairwise token membership mismatch")
    payload["_index_path"]=str(index_path); payload["_index_sha256"]=file_sha256(index_path)
    return payload


def inject_dynamic_summary(candidate_tokens: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Return tokens with the final four causal summary fields replaced."""
    if candidate_tokens.shape[-1]!=CMCP_PAIRWISE_LOCAL_TOKEN_DIM:
        raise ValueError("candidate token dimension mismatch")
    if summary.shape!=candidate_tokens.shape[:-2]+(CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM,):
        raise ValueError("dynamic summary shape mismatch")
    result=candidate_tokens.clone()
    result[...,-CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM:]=summary.unsqueeze(-2).expand(*candidate_tokens.shape[:-2],candidate_tokens.shape[-2],CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM)
    return result
