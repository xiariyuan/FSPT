#!/usr/bin/env python3
"""Build sealed Gate 2 CSRR teacher caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    CSRR_SCHEMA_VERSION,
    build_csrr_trajectory_features,
    extract_cotracker_observed_feature_pyramid,
    model_xy_to_input_xy,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    CSRR_CACHE_INDEX_SCHEMA_VERSION,
    CSRR_CACHE_SCHEMA_VERSION,
    nested_tensor_hash_digest,
    nested_tensor_hashes,
    quantization_measurement,
    select_gate2_rows,
    snapshot_to_cache_dict,
    verify_csrr_cache_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)
from scripts.audit_routeD_oracle_state_transplant_gate1 import (
    _continue_second_window,
    _coords_to_input,
    _initialize_and_first_window,
    _original_queries,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_counterfactual_state_restorer_gate2_v0.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/routeD_counterfactual_state_restorer_cache_20260719"


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _partition_sources(config: dict[str, Any], partition: str) -> list[int]:
    if partition == "train":
        return [int(value) for value in config["partition"]["gradient_train_source_indices"]]
    if partition == "fit_internal_validation":
        return [int(value) for value in config["partition"]["fit_internal_validation_source_indices"]]
    if partition == "smoke":
        anchors = config["cache"]["primary_replay_fixed_source_indices"]
        return [int(anchors["train"][0]), int(anchors["train"][1]), int(anchors["validation"][0]), int(anchors["validation"][1])]
    raise ValueError(f"unsupported partition: {partition}")


def _per_class_cap(config: dict[str, Any], source_index: int) -> int:
    train = set(config["partition"]["gradient_train_source_indices"])
    if int(source_index) in train:
        return int(config["example_construction"]["failure"]["train_per_video_cap"])
    return int(config["example_construction"]["failure"]["validation_per_video_cap"])


def _fresh_queries_at_commit(
    prepared: dict[str, Any], failure: torch.Tensor, video: torch.Tensor
) -> torch.Tensor:
    gt = prepared["gt_tracks_yx"].to(video.device)
    indices = failure.to(video.device)
    queries = torch.zeros(1, failure.numel(), 3, device=video.device)
    queries[0, :, 0] = 15.0
    queries[0, :, 1] = gt[indices, 15, 1] * float(video.shape[-1] - 1)
    queries[0, :, 2] = gt[indices, 15, 0] * float(video.shape[-2] - 1)
    return queries


def _selected_memory(
    values: tuple[torch.Tensor | None, ...], indices: torch.Tensor, *, support: bool
) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for value in values:
        if value is None:
            raise ValueError("CSRR cache requires all four memory levels")
        selected = indices.to(value.device)
        if support:
            # [1,49,N,C] -> [R,49,C]
            output.append(value[0, :, selected].permute(1, 0, 2).detach().float().cpu())
        else:
            # [1,1,N,C] -> [R,1,C]
            output.append(value[0, :, selected].permute(1, 0, 2).detach().float().cpu())
    return output


def _quantize_list(values: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    quantized = [value.to(torch.float16).contiguous() for value in values]
    measurements = [quantization_measurement(value, qvalue) for value, qvalue in zip(values, quantized)]
    return quantized, measurements


def _quantize_tensor(value: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    quantized = value.detach().float().cpu().to(torch.float16).contiguous()
    return quantized, quantization_measurement(value, quantized)


def _balance_rows(failure: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if failure.numel() > 0 and clean.numel() > 0:
        count = min(int(failure.numel()), int(clean.numel()))
        return failure[:count], clean[:count]
    return failure, clean


def _build_video(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    predictor: CoTrackerOnlinePredictor,
    source_index: int,
    partition: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample, sample_metadata = load_manifest_sample(
        Path(config["partition"]["manifest"]), int(source_index)
    )
    prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
    video = prepared["video"].to(device)
    if tuple(video.shape[1:]) != (24, 3, 256, 256):
        raise ValueError("Gate 2 source video shape drift")
    if int(prepared["gt_tracks_yx"].shape[0]) != 64:
        raise ValueError("Gate 2 source point-count drift")

    seed = int(config["training"]["seed"]) + int(source_index)
    set_deterministic(seed)
    with torch.no_grad():
        native_initial = _initialize_and_first_window(
            predictor, video, _original_queries(prepared, video)
        )
        native_final = _continue_second_window(predictor, video, native_initial)
        native_coords = _coords_to_input(
            native_final,
            interp_height=int(predictor.interp_shape[0]),
            interp_width=int(predictor.interp_shape[1]),
        )
        cap = _per_class_cap(config, source_index)
        rows = select_gate2_rows(
            native_future_coords_xy_px=native_coords[:, 16:24],
            gt_tracks_yx=prepared["gt_tracks_yx"],
            gt_occluded=prepared["gt_occluded"],
            original_query_frames=prepared["query_points_tyx"][:, 0],
            failure_error_min_px=float(
                config["example_construction"]["failure"]["native_future_mean_error_px_min"]
            ),
            clean_error_max_px=float(
                config["example_construction"]["clean_noop"]["native_future_mean_error_px_max"]
            ),
            min_future_visible_frames=int(
                config["example_construction"]["failure"]["gt_visible_in_future_min_frames"]
            ),
            per_class_cap=cap,
        )
        failure, clean = _balance_rows(
            rows["failure_point_indices"], rows["clean_point_indices"]
        )
        point_indices = torch.cat([failure, clean], dim=0)
        apply_target = torch.cat(
            [torch.ones(failure.numel()), torch.zeros(clean.numel())], dim=0
        ).float()
        if point_indices.numel() == 0:
            raise RuntimeError(f"Gate 2 source {source_index} has no failure or clean rows")

        fresh = None
        if failure.numel() > 0:
            fresh = _initialize_and_first_window(
                predictor,
                video,
                _fresh_queries_at_commit(prepared, failure, video),
            )

        observed_pyramid = extract_cotracker_observed_feature_pyramid(
            predictor.model, video[:, :16]
        )
        commit_pyramid_float = [
            level[0, 15].detach().float().cpu().contiguous() for level in observed_pyramid
        ]

    exact_native_state = snapshot_to_cache_dict(native_initial)
    trajectory_float = build_csrr_trajectory_features(
        native_initial,
        point_indices=point_indices,
        frame_start=8,
        frame_end_inclusive=15,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    ).detach().float().cpu()

    selected_device = point_indices.to(native_initial.online_coords_predicted.device)
    native_commit_model = native_initial.online_coords_predicted[0, 15, selected_device]
    native_commit_input = model_xy_to_input_xy(
        native_commit_model,
        input_height=256,
        input_width=256,
        model_height=int(predictor.interp_shape[0]),
        model_width=int(predictor.interp_shape[1]),
    ).detach().float().cpu()
    teacher_commit_input = native_commit_input.clone()
    if failure.numel() > 0:
        gt = prepared["gt_tracks_yx"]
        teacher_commit_input[: failure.numel()] = (
            torch.stack([gt[failure, 15, 1], gt[failure, 15, 0]], dim=-1).float()
            * 255.0
        )

    native_feat_float = _selected_memory(
        native_initial.online_track_feat, point_indices, support=False
    )
    native_support_float = _selected_memory(
        native_initial.online_track_support, point_indices, support=True
    )
    teacher_feat_float = [value.clone() for value in native_feat_float]
    teacher_support_float = [value.clone() for value in native_support_float]
    if fresh is not None:
        for level in range(4):
            teacher_feat_float[level][: failure.numel()] = (
                fresh.online_track_feat[level][0].permute(1, 0, 2).detach().float().cpu()
            )
            teacher_support_float[level][: failure.numel()] = (
                fresh.online_track_support[level][0].permute(1, 0, 2).detach().float().cpu()
            )

    native_vis = native_initial.online_vis_predicted[0, 15, selected_device].detach().float().cpu()
    native_conf = native_initial.online_conf_predicted[0, 15, selected_device].detach().float().cpu()
    teacher_vis = native_vis.clone()
    teacher_conf = native_conf.clone()
    if fresh is not None:
        teacher_vis[: failure.numel()] = fresh.online_vis_predicted[0, 15].detach().float().cpu()
        teacher_conf[: failure.numel()] = fresh.online_conf_predicted[0, 15].detach().float().cpu()

    frame_half, frame_q = _quantize_list(commit_pyramid_float)
    trajectory_half, trajectory_q = _quantize_tensor(trajectory_float)
    native_feat_half, native_feat_q = _quantize_list(native_feat_float)
    native_support_half, native_support_q = _quantize_list(native_support_float)
    teacher_feat_half, teacher_feat_q = _quantize_list(teacher_feat_float)
    teacher_support_half, teacher_support_q = _quantize_list(teacher_support_float)
    native_commit_half, native_commit_q = _quantize_tensor(native_commit_input)
    teacher_commit_half, teacher_commit_q = _quantize_tensor(teacher_commit_input)
    native_vis_half, native_vis_q = _quantize_tensor(native_vis)
    native_conf_half, native_conf_q = _quantize_tensor(native_conf)
    teacher_vis_half, teacher_vis_q = _quantize_tensor(teacher_vis)
    teacher_conf_half, teacher_conf_q = _quantize_tensor(teacher_conf)

    quantization_rows = (
        frame_q
        + [trajectory_q]
        + native_feat_q
        + native_support_q
        + teacher_feat_q
        + teacher_support_q
        + [
            native_commit_q,
            teacher_commit_q,
            native_vis_q,
            native_conf_q,
            teacher_vis_q,
            teacher_conf_q,
        ]
    )
    maximum_error = max(row["max_absolute_error"] for row in quantization_rows)
    minimum_cosine = min(row["cosine_similarity"] for row in quantization_rows)
    qg = config["cache"]["quantization_gates"]
    quantization_pass = maximum_error <= float(qg["max_absolute_error"]) and minimum_cosine >= float(
        qg["minimum_cosine_similarity"]
    )

    future_gt_xy = (
        prepared["gt_tracks_yx"][point_indices, 16:24][..., [1, 0]].float() * 255.0
    )
    future_visible = (~prepared["gt_occluded"][point_indices, 16:24]).bool()
    native_future_selected = native_coords[point_indices, 16:24].float()
    continuation_video = video[0, 8:24].round().clamp(0, 255).to(torch.uint8).cpu()

    model_tensors = {
        "continuation_video_u8": continuation_video,
        "point_indices": point_indices.long(),
        "failure_point_indices": failure.long(),
        "clean_point_indices": clean.long(),
        "apply_target": apply_target,
        "trajectory_features": trajectory_half,
        "native_commit_coordinates_xy": native_commit_half,
        "teacher_commit_coordinates_xy": teacher_commit_half,
        "native_visibility_logits": native_vis_half,
        "native_confidence_logits": native_conf_half,
        "teacher_visibility_logits": teacher_vis_half,
        "teacher_confidence_logits": teacher_conf_half,
        "frame_feature_pyramid": frame_half,
        "native_track_feat": native_feat_half,
        "native_track_support": native_support_half,
        "teacher_track_feat": teacher_feat_half,
        "teacher_track_support": teacher_support_half,
        "future_gt_coordinates_xy": future_gt_xy,
        "future_visible_mask": future_visible,
        "native_future_coordinates_xy": native_future_selected,
    }
    hashed_value = {
        "exact_native_state": exact_native_state,
        "model_tensors": model_tensors,
    }
    artifact = {
        "schema_version": CSRR_CACHE_SCHEMA_VERSION,
        "partition": partition,
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "provenance": {
            "config": str(config_path),
            "config_sha256": config_sha256,
            "checkpoint": str(config["backbone"]["checkpoint"]),
            "checkpoint_sha256": str(config["backbone"]["checkpoint_sha256"]),
            "manifest": str(config["partition"]["manifest"]),
            "manifest_sha256": str(config["partition"]["manifest_sha256"]),
            "sample_metadata": sample_metadata,
        },
        "selection": {
            "failure_count": int(failure.numel()),
            "clean_count": int(clean.numel()),
            "failure_eligible_count": int(rows["failure_eligible_count"].item()),
            "clean_eligible_count": int(rows["clean_eligible_count"].item()),
            "balanced_when_both_available": True,
            "per_class_cap": int(cap),
        },
        "exact_native_state": exact_native_state,
        "model_tensors": model_tensors,
        "tensor_hashes": nested_tensor_hashes(hashed_value),
        "tensor_hash_digest": nested_tensor_hash_digest(hashed_value),
        "quantization": {
            "maximum_absolute_error": maximum_error,
            "minimum_cosine_similarity": minimum_cosine,
            "max_absolute_error_gate": float(qg["max_absolute_error"]),
            "minimum_cosine_gate": float(qg["minimum_cosine_similarity"]),
            "pass": quantization_pass,
            "measurements": quantization_rows,
        },
        "integrity": {
            "model_validation_read": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
            "official_kinetics_1144_rerun": False,
        },
    }
    report = {
        "schema_version": "routeD_csrr_teacher_cache_report_v0",
        "partition": partition,
        "source_index": int(source_index),
        "video_name": str(prepared["video_name"]),
        "failure_count": int(failure.numel()),
        "clean_count": int(clean.numel()),
        "failure_eligible_count": int(rows["failure_eligible_count"].item()),
        "clean_eligible_count": int(rows["clean_eligible_count"].item()),
        "point_indices": point_indices.tolist(),
        "quantization": artifact["quantization"],
        "tensor_hash_digest": artifact["tensor_hash_digest"],
        "sample_metadata": sample_metadata,
        "integrity": artifact["integrity"],
    }
    return artifact, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--partition", choices=("train", "fit_internal_validation", "smoke"), required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != CSRR_SCHEMA_VERSION:
        raise ValueError("unexpected Gate 2 config schema")
    config_sha256 = file_sha256(config_path)
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash drift")
    if file_sha256(config["partition"]["manifest"]) != config["partition"]["manifest_sha256"]:
        raise ValueError("manifest hash drift")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("locked-data flags must remain false")
    expected = _partition_sources(config, args.partition)
    if any(index >= 48 for index in expected):
        raise ValueError("Gate 2 cache builder may not read model-validation indices")

    output_dir = Path(args.output_root).resolve() / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(int(config["training"]["seed"]))
    predictor = CoTrackerOnlinePredictor(checkpoint=config["backbone"]["checkpoint"]).to(args.device).eval()
    rows: list[dict[str, Any]] = []
    for source_index in expected:
        sidecar = output_dir / f"video_{source_index:05d}.pt"
        report_path = output_dir / f"video_{source_index:05d}.json"
        if args.resume and sidecar.exists() and report_path.exists():
            artifact = verify_csrr_cache_artifact(
                sidecar,
                expected_partition=args.partition,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
            )
            report = json.loads(report_path.read_text())
            print(json.dumps({"stage": "resume_skip", "source_index": source_index}), flush=True)
        else:
            artifact, report = _build_video(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                predictor=predictor,
                source_index=source_index,
                partition=args.partition,
                device=args.device,
            )
            _atomic_torch_save(artifact, sidecar)
            artifact = verify_csrr_cache_artifact(
                sidecar,
                expected_partition=args.partition,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
            )
            report.update(
                {
                    "sidecar": str(sidecar),
                    "sidecar_sha256": file_sha256(sidecar),
                }
            )
            _atomic_json_save(report, report_path)
            print(
                json.dumps(
                    {
                        "stage": "video_complete",
                        "source_index": source_index,
                        "failure_count": report["failure_count"],
                        "clean_count": report["clean_count"],
                        "quantization_max_abs": report["quantization"]["maximum_absolute_error"],
                        "quantization_min_cosine": report["quantization"]["minimum_cosine_similarity"],
                    }
                ),
                flush=True,
            )
        rows.append(
            {
                "source_index": source_index,
                "video_name": artifact["video_name"],
                "sidecar": str(sidecar),
                "sidecar_sha256": file_sha256(sidecar),
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "failure_count": int(artifact["selection"]["failure_count"]),
                "clean_count": int(artifact["selection"]["clean_count"]),
                "tensor_hash_digest": artifact["tensor_hash_digest"],
                "quantization_maximum_absolute_error": float(
                    artifact["quantization"]["maximum_absolute_error"]
                ),
                "quantization_minimum_cosine_similarity": float(
                    artifact["quantization"]["minimum_cosine_similarity"]
                ),
            }
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    completed = [int(row["source_index"]) for row in rows]
    complete = completed == expected
    if not complete:
        raise RuntimeError("CSRR cache membership failed before index write")
    index = {
        "schema_version": CSRR_CACHE_INDEX_SCHEMA_VERSION,
        "date": "2026-07-19",
        "partition": args.partition,
        "complete": complete,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
        "manifest_sha256": config["partition"]["manifest_sha256"],
        "expected_source_indices": expected,
        "completed_source_indices": completed,
        "expected_count": len(expected),
        "completed_count": len(completed),
        "total_failure_rows": sum(row["failure_count"] for row in rows),
        "total_clean_rows": sum(row["clean_count"] for row in rows),
        "maximum_quantization_absolute_error": max(
            row["quantization_maximum_absolute_error"] for row in rows
        ),
        "minimum_quantization_cosine_similarity": min(
            row["quantization_minimum_cosine_similarity"] for row in rows
        ),
        "combined_sidecar_sha256": canonical_json_sha256(
            [row["sidecar_sha256"] for row in rows]
        ),
        "combined_tensor_hash_digest": canonical_json_sha256(
            [row["tensor_hash_digest"] for row in rows]
        ),
        "rows": rows,
        "integrity": {
            "model_validation_read": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
            "official_kinetics_1144_rerun": False,
        },
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_dir / "cache_index.json"
    _atomic_json_save(index, index_path)
    print(
        json.dumps(
            {
                "stage": "partition_complete",
                "partition": args.partition,
                "cache_index": str(index_path),
                "cache_index_sha256": file_sha256(index_path),
                "completed_count": len(completed),
                "total_failure_rows": index["total_failure_rows"],
                "total_clean_rows": index["total_clean_rows"],
                "maximum_quantization_absolute_error": index[
                    "maximum_quantization_absolute_error"
                ],
                "minimum_quantization_cosine_similarity": index[
                    "minimum_quantization_cosine_similarity"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
