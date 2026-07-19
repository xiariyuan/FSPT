#!/usr/bin/env python3
"""Build the complete sealed P0m DAVIS native/feature cache without metrics."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = REPO_ROOT / "baselines/cotracker"
for path in (EXTERNAL_ROOT, COTRACKER_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cotracker.datasets.tap_vid_datasets import TapVidDataset
from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    DAVIS_EXTERNAL_CACHE_SCHEMA,
    davis_video_order_sha256,
    load_davis_external_config,
    load_davis_sample_preserving_points,
    prepare_davis_sample,
    validated_davis_native_state_hashes,
    verify_davis_external_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    CMCP_FEATURE_CACHE_SCHEMA_VERSION,
    CMCP_FEATURE_INDEX_SCHEMA_VERSION,
    feature_quantization_audit,
    validate_quantization_audit,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import compute_fmaps, run_true_streaming

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_davis_external_v0.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_davis_external_20260719"


def _deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _paths(root: Path, source_index: int) -> tuple[Path, Path, Path]:
    folder = root / "davis_external"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"video_{source_index:05d}"
    return (
        folder / f"{stem}.base.pt",
        folder / f"{stem}.feature.pt",
        folder / f"{stem}.json",
    )


def _extract_native(
    predictor: CoTrackerOnlinePredictor,
    prepared: dict[str, Any],
) -> dict[str, torch.Tensor]:
    points = int(prepared["points"])
    frames = int(prepared["frames"])
    raw_coords = predictor.model.online_coords_predicted[0, :frames, :points].float()
    raw_vis = predictor.model.online_vis_predicted[0, :frames, :points].float()
    raw_conf = predictor.model.online_conf_predicted[0, :frames, :points].float()
    interp_height, interp_width = predictor.interp_shape
    coords = raw_coords.clone()
    coords[..., 0] *= 255.0 / float(max(interp_width - 1, 1))
    coords[..., 1] *= 255.0 / float(max(interp_height - 1, 1))
    coords = coords.permute(1, 0, 2).contiguous().cpu()
    visibility_probability = torch.sigmoid(raw_vis).permute(1, 0).contiguous().cpu()
    confidence_probability = torch.sigmoid(raw_conf).permute(1, 0).contiguous().cpu()
    joint_probability = visibility_probability * confidence_probability
    tensors = {
        "native_coords_xy_px": coords,
        "native_visibility_probability": visibility_probability,
        "native_confidence_probability": confidence_probability,
        "native_joint_probability": joint_probability,
        "native_visibility": joint_probability > 0.6,
        "query_points_tyx": prepared["query_points_tyx"].float().cpu(),
        "gt_tracks_yx": prepared["gt_tracks_yx"].float().cpu(),
        "gt_occluded": prepared["gt_occluded"].bool().cpu(),
        "candidate_coords_xy_px": coords.unsqueeze(2),
    }
    if not torch.equal(tensors["candidate_coords_xy_px"][..., 0, :], coords):
        raise RuntimeError("candidate 0/native parity failure")
    return tensors


def _extract_once(
    predictor: CoTrackerOnlinePredictor,
    prepared: dict[str, Any],
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    video = prepared["video"].to(device)
    queries = prepared["query_points_tyx"].to(device)
    run_true_streaming(predictor, video, queries)
    tensors = _extract_native(predictor, prepared)
    feature_maps = compute_fmaps(predictor, video).detach().float().cpu().contiguous()
    return tensors, feature_maps


def _tensor_dict_exact(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _load_valid_resume_row(report_path: Path) -> dict[str, Any] | None:
    row = json.loads(report_path.read_text())
    base = Path(row.get("base_sidecar", ""))
    feature = Path(row.get("sidecar", ""))
    if not (
        base.exists()
        and feature.exists()
        and row.get("base_sidecar_sha256") == file_sha256(base)
        and row.get("sidecar_sha256") == file_sha256(feature)
        and row.get("integrity", {}).get("performance_metrics_computed") is False
    ):
        return None
    native_hashes = validated_davis_native_state_hashes(base, feature)
    if row.get("native_state_hashes") != native_hashes:
        row["native_state_hashes"] = native_hashes
        report_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = load_davis_external_config(args.config)
    verified = verify_davis_external_files(config)
    output_root = Path(args.output_root).resolve()
    expected_indices = list(range(int(config["external_dataset"]["expected_videos"])))
    replay_indices = set(
        int(value) for value in config["cache_contract"]["extraction_replay_video_indices"]
    )

    reports: dict[int, dict[str, Any]] = {}
    if args.resume:
        for source_index in expected_indices:
            _, _, report_path = _paths(output_root, source_index)
            if report_path.exists():
                row = _load_valid_resume_row(report_path)
                if row is not None:
                    reports[source_index] = row

    dataset = TapVidDataset(
        data_root=verified["dataset"]["path"],
        **config["external_dataset"]["loader_args"],
    )
    if len(dataset) != len(expected_indices):
        raise RuntimeError(f"DAVIS video count mismatch: {len(dataset)}")
    video_names = [str(value) for value in dataset.video_names]
    if len(video_names) != len(set(video_names)):
        raise RuntimeError("duplicate DAVIS video names")
    order_sha = davis_video_order_sha256(video_names)

    todo = [index for index in expected_indices if index not in reports]
    predictor = None
    if todo:
        _deterministic(20000)
        predictor = CoTrackerOnlinePredictor(
            checkpoint=verified["backbone"]["path"]
        ).to(args.device).eval()

    print(
        json.dumps(
            {
                "stage": "cache_start",
                "partition": "davis_external",
                "expected_count": 30,
                "completed_before_start": len(reports),
                "video_order_sha256": order_sha,
                "performance_metrics_computed": False,
            }
        ),
        flush=True,
    )

    for source_index in expected_indices:
        if source_index in reports:
            print(
                json.dumps({"stage": "resume_skip", "source_index": source_index}),
                flush=True,
            )
            continue
        started = time.time()
        sample = load_davis_sample_preserving_points(dataset, source_index)
        prepared = prepare_davis_sample(
            sample, raster=int(config["backbone"]["input_raster"][0])
        )
        if prepared["video_name"] != video_names[source_index]:
            raise RuntimeError("DAVIS video-order identity mismatch")
        _deterministic(20000 + source_index)
        tensors, feature32 = _extract_once(predictor, prepared, device=args.device)
        replay_exact = None
        if source_index in replay_indices:
            _deterministic(20000 + source_index)
            second_tensors, second_feature = _extract_once(
                predictor, prepared, device=args.device
            )
            replay_exact = _tensor_dict_exact(tensors, second_tensors) and torch.equal(
                feature32, second_feature
            )
            if not replay_exact:
                raise RuntimeError(f"DAVIS extraction replay mismatch: {source_index}")
        feature16 = feature32.half().contiguous()
        quantization = feature_quantization_audit(feature32, feature16)
        validate_quantization_audit(
            quantization,
            max_abs_max=float(
                config["cache_contract"]["feature_map_quantization_max_abs_max"]
            ),
            min_cosine_min=float(
                config["cache_contract"]["feature_map_quantization_min_cosine_min"]
            ),
        )
        base_path, feature_path, report_path = _paths(output_root, source_index)
        identity = {
            "dataset_index": source_index,
            "video_name": prepared["video_name"],
            "frames": prepared["frames"],
            "points": prepared["points"],
        }
        base_artifact = {
            "schema_version": DAVIS_EXTERNAL_CACHE_SCHEMA,
            "provenance": {
                "partition": "davis_external",
                "source_index": source_index,
                "sample_identity": identity,
                "config_sha256": config["_config_sha256"],
                "dataset_sha256": config["external_dataset"]["sha256"],
                "video_order_sha256": order_sha,
                "candidate_generation_ground_truth_free": True,
            },
            "tensors": tensors,
            "tensor_hashes": {
                key: tensor_sha256(value) for key, value in tensors.items()
            },
        }
        torch.save(base_artifact, base_path)
        native_keys = (
            "native_coords_xy_px",
            "native_visibility_probability",
            "native_confidence_probability",
            "native_joint_probability",
            "native_visibility",
        )
        feature_artifact = {
            "schema_version": CMCP_FEATURE_CACHE_SCHEMA_VERSION,
            "provenance": {
                "partition": "davis_external",
                "source_index": source_index,
                "sample_identity": identity,
                "protocol_sha256": config["_config_sha256"],
                "base_sidecar": str(base_path),
                "base_sidecar_sha256": file_sha256(base_path),
                "backbone_checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
                "candidate_generation_ground_truth_free": True,
            },
            "feature_maps_f16": feature16,
            "feature_maps_f16_sha256": tensor_sha256(feature16),
            "feature_maps_float32_sha256": tensor_sha256(feature32),
            "quantization_audit": quantization,
            "native_state_hashes": {
                key: tensor_sha256(tensors[key]) for key in native_keys
            },
        }
        torch.save(feature_artifact, feature_path)
        row = {
            "schema_version": "routeD_cmcp_davis_external_video_report_v0",
            "partition": "davis_external",
            "source_index": source_index,
            "video_name": prepared["video_name"],
            "sample_identity": identity,
            "protocol_sha256": config["_config_sha256"],
            "dataset_sha256": config["external_dataset"]["sha256"],
            "video_order_sha256": order_sha,
            "base_sidecar": str(base_path),
            "base_sidecar_sha256": file_sha256(base_path),
            "sidecar": str(feature_path),
            "sidecar_sha256": file_sha256(feature_path),
            "feature_map_shape": list(feature16.shape),
            "feature_maps_f16_sha256": tensor_sha256(feature16),
            "feature_maps_float32_sha256": tensor_sha256(feature32),
            "quantization_audit": quantization,
            "native_state_hashes": feature_artifact["native_state_hashes"],
            "deterministic_full_extraction_replay_exact": replay_exact,
            "seconds": round(time.time() - started, 3),
            "integrity": {
                "performance_metrics_computed": False,
                "model_variant_C_executed": False,
                "calibration_read": False,
                "tapvid_kinetics_read": False,
            },
        }
        report_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
        reports[source_index] = row
        print(
            json.dumps(
                {
                    "stage": "video_cached",
                    "source_index": source_index,
                    "replay_exact": replay_exact,
                    "quantization": quantization,
                    "performance_metrics_computed": False,
                }
            ),
            flush=True,
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    completed = sorted(reports)
    if completed != expected_indices:
        raise RuntimeError("DAVIS cache must complete before index release")
    rows = [reports[index] for index in expected_indices]
    index = {
        "schema_version": CMCP_FEATURE_INDEX_SCHEMA_VERSION,
        "protocol_sha256": config["_config_sha256"],
        "partition": "davis_external",
        "expected_source_indices": expected_indices,
        "completed_source_indices": completed,
        "expected_count": 30,
        "completed_count": 30,
        "complete": True,
        "dataset_path": verified["dataset"]["path"],
        "dataset_sha256": config["external_dataset"]["sha256"],
        "video_names": video_names,
        "video_order_sha256": order_sha,
        "feature_dtype": "torch.float16",
        "feature_channels": 128,
        "videos": rows,
        "aggregate_quantization_audit": {
            "max_abs_max": max(row["quantization_audit"]["max_abs"] for row in rows),
            "min_cosine_min": min(
                row["quantization_audit"]["min_cosine"] for row in rows
            ),
        },
        "integrity": {
            "performance_metrics_computed": False,
            "model_variant_C_executed": False,
            "partial_performance_exposed": False,
            "calibration_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_root / "davis_external" / "cache_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "stage": "cache_complete",
                "index": str(index_path),
                "complete": True,
                "count": 30,
                "performance_metrics_computed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
