#!/usr/bin/env python3
"""Build GT-blind frozen candidate sets for Route-D Gate 3A v1."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = EXTERNAL_ROOT / "baselines/cotracker"
for path in (COTRACKER_ROOT, EXTERNAL_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_causal_anchor_memory import (
    extract_immutable_query_anchor_memory,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    CounterfactualStructuredReextractionRestorer,
    extract_cotracker_observed_feature_pyramid,
    model_xy_to_input_xy,
)
from projects.mmp_tracker.mmp_tracker.routeD_discrete_candidates import (
    extract_discrete_candidates,
)
from projects.mmp_tracker.mmp_tracker.routeD_geometry_preserving_relocalization import (
    geometry_preserving_pyramid_score,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import set_deterministic


SCHEMA = "routeD_geometry_representation_audit_gate3a_v1"
CACHE_SCHEMA = "routeD_geometry_representation_candidate_cache_gate3a_v1"
INDEX_SCHEMA = "routeD_geometry_representation_candidate_index_gate3a_v1"
CAUSAL_INDEX_SCHEMA = "routeD_geometry_causal_input_index_gate3a_v1"
CAUSAL_CACHE_SCHEMA = "routeD_geometry_causal_input_cache_gate3a_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_geometry_representation_audit_gate3a_v1.yaml"


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _candidate_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config["candidate_extraction"]
    return {
        "top_k": int(value["nonnative_top_k"]),
        "nms_radius_grid_cells": int(value["nms_radius_grid_cells"]),
        "local_refinement_window_grid_cells": int(
            value["local_refinement_window_grid_cells"]
        ),
        "local_softmax_temperature": float(value["local_softmax_temperature"]),
        "deduplicate_radius_input_px": float(
            value["deduplicate_radius_input_px"]
        ),
        "input_height": 256,
        "input_width": 256,
    }


def _gate2_output(
    model: CounterfactualStructuredReextractionRestorer,
    tensors: Mapping[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    return model(
        trajectory_features=tensors["trajectory_features"].to(
            device=device, dtype=torch.float32
        ),
        native_support_pyramid=[
            value.to(device=device, dtype=torch.float32)
            for value in tensors["native_track_support"]
        ],
        frame_feature_pyramid=[
            value[None].to(device=device, dtype=torch.float32)
            for value in tensors["frame_feature_pyramid"]
        ],
        native_commit_coordinates_xy=(
            tensors["native_commit_coordinates_normalized_xy"]
            .to(device=device, dtype=torch.float32)
            * 255.0
        ),
        input_height=256,
        input_width=256,
    )


def _native_coordinates(
    tensors: Mapping[str, Any],
    predictor: CoTrackerOnlinePredictor,
) -> torch.Tensor:
    return model_xy_to_input_xy(
        tensors["native_commit_coordinates_model_xy"].float(),
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    ).float()


def _query_anchor_support(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    query_frames: torch.Tensor,
    query_coordinates_model_xy: torch.Tensor,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    observed = extract_cotracker_observed_feature_pyramid(
        predictor.model, video
    )
    frames = query_frames.to(video.device).long()
    coordinates = model_xy_to_input_xy(
        query_coordinates_model_xy.to(video.device),
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    )
    feature, support = extract_immutable_query_anchor_memory(
        predictor.model,
        observed,
        frames,
        coordinates,
        input_height=256,
        input_width=256,
        support_radius=3,
    )
    feature_rows = [value[0].permute(1, 0, 2).contiguous() for value in feature]
    support_rows = [value[0].permute(1, 0, 2).contiguous() for value in support]
    frame15 = [value[0, 15].contiguous() for value in observed]
    return feature_rows, support_rows, frames, coordinates, frame15


def _build_video(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    causal_index_row: Mapping[str, Any],
    predictor: CoTrackerOnlinePredictor,
    gate2_model: CounterfactualStructuredReextractionRestorer,
    device: str,
) -> dict[str, Any]:
    source_index = int(causal_index_row["source_index"])
    causal_path = Path(causal_index_row["sidecar"])
    if file_sha256(causal_path) != causal_index_row["sidecar_sha256"]:
        raise ValueError("Gate 3A v1 causal-input sidecar hash drift")
    causal = torch.load(causal_path, map_location="cpu", weights_only=False)
    if causal.get("schema_version") != CAUSAL_CACHE_SCHEMA:
        raise ValueError("Gate 3A v1 causal-input sidecar schema drift")
    integrity = causal.get("integrity", {})
    if (
        not bool(integrity.get("forbidden_key_tokens_absent"))
        or bool(integrity.get("candidate_builder_can_read_commit_teacher"))
        or bool(integrity.get("candidate_builder_can_read_future_outcomes"))
    ):
        raise ValueError("Gate 3A v1 causal-input boundary failed")
    tensors = causal["tensors"]
    point_indices = tensors["point_indices"].long()
    if point_indices.numel() == 0:
        raise RuntimeError(f"Gate 3A v1 source {source_index} has no failure rows")
    video = tensors["observed_video_u8"][None].to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        anchor_feature, anchor_support, query_frames, query_coordinates, observed_frame15 = (
            _query_anchor_support(
                predictor,
                video,
                tensors["query_frames"],
                tensors["query_coordinates_model_xy"],
            )
        )
        cached_frame15 = [
            value.to(device=device, dtype=torch.float32)
            for value in tensors["frame_feature_pyramid"]
        ]
        frame15_max_abs = max(
            float((observed.float() - cached).abs().max().item())
            for observed, cached in zip(observed_frame15, cached_frame15)
        )
        output = _gate2_output(gate2_model, tensors, device)
        native_support = [
            value.to(device=device, dtype=torch.float32)
            for value in tensors["native_track_support"]
        ]
        frame_maps = [value[None] for value in cached_frame15]
        geometry_native = geometry_preserving_pyramid_score(
            frame_maps,
            native_support,
            common_height=64,
            common_width=64,
            support_radius=3,
            trim_fraction=float(
                config["representations"]["M1_geometry_native"][
                    "token_trim_fraction"
                ]
            ),
        )["fused_score_map"]
        geometry_query = geometry_preserving_pyramid_score(
            frame_maps,
            [value.float() for value in anchor_support],
            common_height=64,
            common_width=64,
            support_radius=3,
            trim_fraction=float(
                config["representations"]["M2_geometry_immutable_query"][
                    "token_trim_fraction"
                ]
            ),
        )["fused_score_map"]
        score_maps = {
            "M0_pooled_native_gate2": output["fused_logits"].float(),
            "M1_geometry_native": geometry_native.float(),
            "M2_geometry_immutable_query": geometry_query.float(),
        }
        native_xy = _native_coordinates(tensors, predictor).to(device)
        candidates = {
            name: extract_discrete_candidates(
                score, native_xy, **_candidate_kwargs(config)
            )
            for name, score in score_maps.items()
        }

    candidate_payload = {
        name: {
            key: value.detach().cpu().contiguous()
            for key, value in candidate.items()
        }
        for name, candidate in candidates.items()
    }
    map_payload = {
        name: value.detach().cpu().float().contiguous()
        for name, value in score_maps.items()
    }
    hashes = {
        "score_maps": {name: tensor_sha256(value) for name, value in map_payload.items()},
        "candidate_coordinates": {
            name: tensor_sha256(value["candidate_coordinates_xy"])
            for name, value in candidate_payload.items()
        },
        "candidate_valid_mask": {
            name: tensor_sha256(value["candidate_valid_mask"])
            for name, value in candidate_payload.items()
        },
    }
    payload = {
        "schema_version": CACHE_SCHEMA,
        "partition": "fit_development",
        "source_index": source_index,
        "video_name": str(causal["video_name"]),
        "point_indices": point_indices.cpu(),
        "query_frames": query_frames.cpu(),
        "query_coordinates_input_xy": query_coordinates.cpu().float(),
        "immutable_query_track_feat": [value.detach().cpu().half() for value in anchor_feature],
        "immutable_query_track_support": [value.detach().cpu().half() for value in anchor_support],
        "score_maps": map_payload,
        "candidates": candidate_payload,
        "gate2_soft_coordinates_xy": output["predicted_coordinates_xy"].detach().cpu().float(),
        "hashes": hashes,
        "candidate_hash_digest": canonical_json_sha256(hashes),
        "integrity": {
            "candidate_builder_accessed_commit_teacher": False,
            "candidate_builder_accessed_future_outcomes": False,
            "observed_frames": [0, 15],
            "frame15_feature_max_abs_vs_gate2_float16_cache": frame15_max_abs,
            "model_validation_read": False,
            "external_read": False,
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": config_sha256,
            "causal_input_sidecar": str(causal_path),
            "causal_input_sidecar_sha256": str(causal_index_row["sidecar_sha256"]),
            "gate2_cache_sidecar_sha256": str(
                causal["provenance"]["gate2_cache_sidecar_sha256"]
            ),
            "gate2_checkpoint_sha256": str(
                config["frozen_parent"]["gate2_checkpoint_sha256"]
            ),
            "sample_manifest_sha256": str(
                causal["provenance"]["sample_manifest_sha256"]
            ),
            "sample_shard_sha256": str(
                causal["provenance"]["sample_shard_sha256"]
            ),
        },
    }
    del video
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3A v1 config schema")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3A v1 locked-data flags must remain false")
    for path_key, hash_key in (
        ("gate2_config", "gate2_config_sha256"),
        ("gate2_checkpoint", "gate2_checkpoint_sha256"),
    ):
        if file_sha256(config["frozen_parent"][path_key]) != config["frozen_parent"][hash_key]:
            raise ValueError(f"Gate 3A v1 frozen parent hash drift: {path_key}")
    if file_sha256(config["partition"]["manifest"]) != config["partition"]["manifest_sha256"]:
        raise ValueError("Gate 3A v1 manifest hash drift")
    config_sha256 = file_sha256(config_path)
    causal_index_path = Path(config["cache"]["causal_input_index"])
    causal_index = json.loads(causal_index_path.read_text())
    if (
        causal_index.get("schema_version") != CAUSAL_INDEX_SCHEMA
        or not bool(causal_index.get("complete"))
        or not bool(
            causal_index.get("integrity", {}).get(
                "all_sidecars_teacher_and_future_key_free"
            )
        )
    ):
        raise ValueError("Gate 3A v1 causal input index incomplete")
    if causal_index.get("config_sha256") != config_sha256:
        raise ValueError("Gate 3A v1 causal input/config hash drift")
    expected_sources = [int(value) for value in config["partition"]["source_indices"]]
    if (
        causal_index["expected_source_indices"] != expected_sources
        or causal_index["completed_source_indices"] != expected_sources
    ):
        raise ValueError("Gate 3A v1 source membership drift")

    set_deterministic(int(config["metrics"]["bootstrap_seed"]))
    # Use the checkpoint path frozen by the parent Gate 2 protocol.
    parent = yaml.safe_load(Path(config["frozen_parent"]["gate2_config"]).read_text())
    predictor = CoTrackerOnlinePredictor(
        checkpoint=parent["backbone"]["checkpoint"]
    ).to(args.device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(
        config["frozen_parent"]["gate2_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    gate2_model = CounterfactualStructuredReextractionRestorer().to(args.device)
    gate2_model.load_state_dict(checkpoint["model_state"], strict=True)
    gate2_model.eval()
    for parameter in gate2_model.parameters():
        parameter.requires_grad_(False)

    output_root = Path(
        args.output_root or config["cache"]["candidate_output_root"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for causal_row in causal_index["rows"]:
        source_index = int(causal_row["source_index"])
        sidecar = output_root / f"video_{source_index:05d}.pt"
        if args.resume and sidecar.exists():
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != CACHE_SCHEMA:
                raise ValueError("Gate 3A v1 resume schema drift")
        else:
            payload = _build_video(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                causal_index_row=causal_row,
                predictor=predictor,
                gate2_model=gate2_model,
                device=args.device,
            )
            _atomic_torch_save(payload, sidecar)
        rows.append(
            {
                "source_index": source_index,
                "video_name": payload["video_name"],
                "failure_points": int(payload["point_indices"].numel()),
                "sidecar": str(sidecar),
                "sidecar_sha256": file_sha256(sidecar),
                "candidate_hash_digest": payload["candidate_hash_digest"],
            }
        )
        print(
            json.dumps(
                {
                    "stage": "video_complete",
                    "source_index": source_index,
                    "failure_points": rows[-1]["failure_points"],
                    "candidate_hash_digest": rows[-1]["candidate_hash_digest"],
                }
            ),
            flush=True,
        )
    index = {
        "schema_version": INDEX_SCHEMA,
        "date": "2026-07-19",
        "partition": "fit_development",
        "complete": True,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "causal_input_index": str(causal_index_path),
        "causal_input_index_sha256": file_sha256(causal_index_path),
        "causal_input_combined_tensor_hash_digest": causal_index[
            "combined_tensor_hash_digest"
        ],
        "expected_source_indices": expected_sources,
        "completed_source_indices": [int(row["source_index"]) for row in rows],
        "expected_failure_points": int(config["partition"]["expected_failure_points"]),
        "completed_failure_points": sum(int(row["failure_points"]) for row in rows),
        "combined_candidate_hash_digest": canonical_json_sha256(
            [row["candidate_hash_digest"] for row in rows]
        ),
        "rows": rows,
        "integrity": {
            "candidate_sets_frozen_before_teacher_read": True,
            "model_validation_read": False,
            "external_read": False,
        },
    }
    if index["completed_source_indices"] != expected_sources:
        raise RuntimeError("Gate 3A v1 candidate source completion mismatch")
    if index["completed_failure_points"] != index["expected_failure_points"]:
        raise RuntimeError("Gate 3A v1 candidate point-count mismatch")
    index["index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_root / "cache_index.json"
    _atomic_json_save(index, index_path)
    print(
        json.dumps(
            {
                "stage": "candidate_cache_complete",
                "index": str(index_path),
                "index_sha256": file_sha256(index_path),
                "failure_points": index["completed_failure_points"],
                "combined_candidate_hash_digest": index[
                    "combined_candidate_hash_digest"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
