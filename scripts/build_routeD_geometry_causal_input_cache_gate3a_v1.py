#!/usr/bin/env python3
"""Materialize the teacher-free causal input boundary for Route-D Gate 3A v1."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.append(str(EXTERNAL_ROOT))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    load_complete_csrr_cache_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    load_csrr_cache_video,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import load_manifest_sample


SCHEMA = "routeD_geometry_representation_audit_gate3a_v1"
CACHE_SCHEMA = "routeD_geometry_causal_input_cache_gate3a_v1"
INDEX_SCHEMA = "routeD_geometry_causal_input_index_gate3a_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_geometry_representation_audit_gate3a_v1.yaml"
ALLOWED_TENSOR_KEYS = (
    "observed_video_u8",
    "point_indices",
    "query_frames",
    "query_coordinates_model_xy",
    "native_commit_coordinates_model_xy",
    "trajectory_features",
    "native_commit_coordinates_normalized_xy",
    "frame_feature_pyramid",
    "native_track_support",
)
FORBIDDEN_KEY_TOKENS = ("teacher", "future", "gt_", "continuation", "target_points", "occluded")


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _observed_video_u8(sample: dict[str, Any], raster: int) -> torch.Tensor:
    # Deliberately access only the task-input video field. Query inputs come
    # from the frozen predictor state, never from target_points/occluded.
    video = np.asarray(sample["video"])
    if video.ndim != 4:
        raise ValueError("Gate 3A v1 video must have shape [T,H,W,3]")
    if video.shape[-1] != 3 and video.shape[1] == 3:
        video = np.transpose(video, (0, 2, 3, 1))
    value = torch.from_numpy(video[:16]).permute(0, 3, 1, 2)
    if value.shape[0] != 16:
        raise ValueError("Gate 3A v1 observed video must contain frames 0-15")
    if tuple(value.shape[-2:]) != (raster, raster):
        value = F.interpolate(
            value.float(), size=(raster, raster), mode="bilinear", align_corners=True
        ).round().clamp(0, 255)
    return value.to(dtype=torch.uint8).contiguous()


def _flatten_tensor_hashes(prefix: str, value: Any, output: dict[str, str]) -> None:
    if isinstance(value, torch.Tensor):
        output[prefix] = tensor_sha256(value)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_tensor_hashes(f"{prefix}.{index}", item, output)
    else:
        raise TypeError(f"non-tensor causal input at {prefix}")


def _assert_teacher_free(payload: dict[str, Any]) -> None:
    if tuple(payload["tensors"]) != ALLOWED_TENSOR_KEYS:
        raise ValueError("Gate 3A v1 causal tensor allowlist drift")
    paths: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                paths.append(path.lower())
                walk(path, child)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(f"{prefix}.{index}", child)

    # Claims/provenance may name the forbidden concepts; only tensor/data paths
    # are required to be physically free of them.
    walk("tensors", payload["tensors"])
    violations = [
        path
        for path in paths
        if any(token in path for token in FORBIDDEN_KEY_TOKENS)
    ]
    if violations:
        raise ValueError(f"teacher/future key crossed causal boundary: {violations}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3A v1 config schema")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3A v1 locked-data flags must remain false")
    config_sha256 = file_sha256(config_path)
    gate2_index_path = (
        Path(config["cache"]["gate2_root"]) / config["cache"]["gate2_fit_index"]
    )
    if file_sha256(gate2_index_path) != config["cache"]["gate2_fit_index_sha256"]:
        raise ValueError("Gate 3A v1 Gate 2 cache-index hash drift")
    gate2_index = load_complete_csrr_cache_index(
        gate2_index_path, expected_partition="fit_internal_validation"
    )
    expected_sources = [int(value) for value in config["partition"]["source_indices"]]
    gate2_rows = {int(row["source_index"]): row for row in gate2_index["rows"]}
    output_root = Path(
        args.output_root or config["cache"]["causal_input_output_root"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent = yaml.safe_load(Path(config["frozen_parent"]["gate2_config"]).read_text())
    raster = int(parent["backbone"]["input_raster"])
    rows = []
    for source_index in expected_sources:
        output_path = output_root / f"video_{source_index:05d}.pt"
        if args.resume and output_path.exists():
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            _assert_teacher_free(payload)
        else:
            row = gate2_rows[source_index]
            artifact = load_csrr_cache_video(
                row,
                expected_partition="fit_internal_validation",
                expected_config_sha256=str(config["frozen_parent"]["gate2_config_sha256"]),
            )
            source, sample_metadata = load_manifest_sample(
                Path(config["partition"]["manifest"]), source_index
            )
            source_tensors = artifact["model_tensors"]
            failure_local = torch.where(source_tensors["apply_target"].float() > 0.5)[0]
            point_indices = source_tensors["point_indices"][failure_local].long()
            state = artifact["exact_native_state"]["tensors"]
            queries = state["predictor_queries"][0, point_indices]
            tensors = {
                "observed_video_u8": _observed_video_u8(source, raster),
                "point_indices": point_indices.contiguous(),
                "query_frames": queries[:, 0].round().long().contiguous(),
                "query_coordinates_model_xy": queries[:, 1:3].float().contiguous(),
                "native_commit_coordinates_model_xy": state[
                    "online_coords_predicted"
                ][0, 15, point_indices].float().contiguous(),
                "trajectory_features": source_tensors["trajectory_features"][
                    failure_local
                ].contiguous(),
                "native_commit_coordinates_normalized_xy": source_tensors[
                    "native_commit_coordinates_normalized_xy"
                ][failure_local].contiguous(),
                "frame_feature_pyramid": [
                    value.contiguous() for value in source_tensors["frame_feature_pyramid"]
                ],
                "native_track_support": [
                    value[failure_local].contiguous()
                    for value in source_tensors["native_track_support"]
                ],
            }
            hashes: dict[str, str] = {}
            for key, value in tensors.items():
                _flatten_tensor_hashes(key, value, hashes)
            payload = {
                "schema_version": CACHE_SCHEMA,
                "partition": "fit_development_failure_conditioned",
                "source_index": source_index,
                "video_name": str(artifact["video_name"]),
                "tensors": tensors,
                "tensor_hashes": hashes,
                "tensor_hash_digest": canonical_json_sha256(hashes),
                "integrity": {
                    "allowed_tensor_keys": list(ALLOWED_TENSOR_KEYS),
                    "forbidden_key_tokens_absent": True,
                    "source_failure_membership_is_frozen_fit_only_diagnostic": True,
                    "candidate_builder_can_read_commit_teacher": False,
                    "candidate_builder_can_read_future_outcomes": False,
                    "observed_frames": [0, 15],
                    "model_validation_read": False,
                    "external_read": False,
                },
                "provenance": {
                    "config": str(config_path),
                    "config_sha256": config_sha256,
                    "gate2_cache_sidecar_sha256": str(row["sidecar_sha256"]),
                    "sample_manifest_sha256": str(sample_metadata["manifest_sha256"]),
                    "sample_shard_sha256": str(sample_metadata["shard_sha256"]),
                    "sample_global_video_index": int(
                        sample_metadata["global_video_index"]
                    ),
                },
            }
            _assert_teacher_free(payload)
            _atomic_torch_save(payload, output_path)
        rows.append(
            {
                "source_index": source_index,
                "video_name": payload["video_name"],
                "failure_points": int(payload["tensors"]["point_indices"].numel()),
                "sidecar": str(output_path),
                "sidecar_sha256": file_sha256(output_path),
                "tensor_hash_digest": payload["tensor_hash_digest"],
            }
        )
        print(json.dumps({"stage": "causal_input_complete", **rows[-1]}), flush=True)
    index = {
        "schema_version": INDEX_SCHEMA,
        "date": "2026-07-19",
        "partition": "fit_development_failure_conditioned",
        "complete": True,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "expected_source_indices": expected_sources,
        "completed_source_indices": [int(row["source_index"]) for row in rows],
        "expected_failure_points": int(config["partition"]["expected_failure_points"]),
        "completed_failure_points": sum(int(row["failure_points"]) for row in rows),
        "combined_tensor_hash_digest": canonical_json_sha256(
            [row["tensor_hash_digest"] for row in rows]
        ),
        "rows": rows,
        "integrity": {
            "all_sidecars_teacher_and_future_key_free": True,
            "model_validation_read": False,
            "external_read": False,
        },
    }
    if index["completed_source_indices"] != expected_sources:
        raise RuntimeError("Gate 3A v1 causal input membership mismatch")
    if index["completed_failure_points"] != index["expected_failure_points"]:
        raise RuntimeError("Gate 3A v1 causal input point-count mismatch")
    index["index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_root / "cache_index.json"
    _atomic_json_save(index, index_path)
    print(json.dumps({"stage": "causal_input_index_complete", "index": str(index_path),
                      "index_sha256": file_sha256(index_path),
                      "combined_tensor_hash_digest": index["combined_tensor_hash_digest"]},
                     indent=2))


if __name__ == "__main__":
    main()
