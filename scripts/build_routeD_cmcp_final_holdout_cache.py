#!/usr/bin/env python3
"""Build sealed P0l native/feature caches without computing model performance."""
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

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    CMCP_FEATURE_CACHE_SCHEMA_VERSION,
    CMCP_FEATURE_INDEX_SCHEMA_VERSION,
    feature_quantization_audit,
    validate_quantization_audit,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout import (
    FINAL_HOLDOUT_CACHE_SCHEMA,
    load_final_holdout_config,
    partition_indices,
    verify_frozen_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import (
    compute_fmaps,
    load_manifest_sample,
    prepare_sample,
    run_true_streaming,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_final_holdout_v0.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_final_holdout_20260719"
REFERENCE_BASE = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717/fit/video_00000.pt"
REFERENCE_FEATURE = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/video_00000.pt"


def _deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _paths(root: Path, partition: str, source_index: int):
    folder = root / partition
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"video_{source_index:05d}"
    return folder / f"{stem}.base.pt", folder / f"{stem}.feature.pt", folder / f"{stem}.json"


def _extract_native(predictor, prepared: dict[str, Any]) -> dict[str, torch.Tensor]:
    points = int(prepared["query_points_tyx"].shape[0])
    frames = int(prepared["video"].shape[1])
    raw_coords = predictor.model.online_coords_predicted[0, :frames, :points].float()
    raw_vis = predictor.model.online_vis_predicted[0, :frames, :points].float()
    raw_conf = predictor.model.online_conf_predicted[0, :frames, :points].float()
    interp_h, interp_w = predictor.interp_shape
    coords = raw_coords.clone()
    coords[..., 0] *= 255.0 / float(max(interp_w - 1, 1))
    coords[..., 1] *= 255.0 / float(max(interp_h - 1, 1))
    coords = coords.permute(1, 0, 2).contiguous().cpu()
    vis = torch.sigmoid(raw_vis).permute(1, 0).contiguous().cpu()
    conf = torch.sigmoid(raw_conf).permute(1, 0).contiguous().cpu()
    joint = vis * conf
    tensors = {
        "native_coords_xy_px": coords,
        "native_visibility_probability": vis,
        "native_confidence_probability": conf,
        "native_joint_probability": joint,
        "native_visibility": joint > 0.6,
        "query_points_tyx": prepared["query_points_tyx"].float().cpu(),
        "gt_tracks_yx": prepared["gt_tracks_yx"].float().cpu(),
        "gt_occluded": prepared["gt_occluded"].bool().cpu(),
        "candidate_coords_xy_px": coords.unsqueeze(2),
    }
    return tensors


def _extract_once(predictor, prepared, device: str):
    video = prepared["video"].to(device)
    queries = prepared["query_points_tyx"].to(device)
    run_true_streaming(predictor, video, queries)
    tensors = _extract_native(predictor, prepared)
    feature = compute_fmaps(predictor, video).detach().float().cpu().contiguous()
    return tensors, feature


def _tensor_dict_exact(left, right) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def _smoke_reference_parity(tensors, feature16):
    base = torch.load(REFERENCE_BASE, map_location="cpu", weights_only=False)["tensors"]
    ref_feature = torch.load(REFERENCE_FEATURE, map_location="cpu", weights_only=False)["feature_maps_f16"]
    keys = (
        "native_coords_xy_px", "native_visibility_probability",
        "native_confidence_probability", "native_joint_probability",
        "native_visibility", "query_points_tyx", "gt_tracks_yx", "gt_occluded",
    )
    checks = {key: torch.equal(tensors[key], base[key]) for key in keys}
    checks["candidate_zero"] = torch.equal(tensors["candidate_coords_xy_px"][..., 0, :], base["native_coords_xy_px"])
    checks["feature_maps_f16"] = torch.equal(feature16, ref_feature)
    if not all(checks.values()):
        raise RuntimeError(f"implementation smoke reference mismatch: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--partition", choices=("implementation_smoke", "final_holdout"), required=True)
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    config = load_final_holdout_config(args.config)
    verified = verify_frozen_files(config)
    indices = list(partition_indices(config, args.partition))
    selected = indices[: args.max_videos] if args.max_videos > 0 else indices
    if args.partition == "final_holdout" and args.max_videos > 0:
        raise ValueError("partial final-holdout jobs are forbidden")
    source_alias = config["partitions"][args.partition]["source"]
    manifest = Path(config["source_manifests"][source_alias]["path"])
    output_root = Path(args.output_root).resolve()
    replay_indices = set(config["cache_contract"]["deterministic_full_extraction_replay_indices"])
    reports = {}
    if args.resume:
        for index in indices:
            base_path, feature_path, report_path = _paths(output_root, args.partition, index)
            if base_path.exists() and feature_path.exists() and report_path.exists():
                row = json.loads(report_path.read_text())
                if row.get("base_sidecar_sha256") == file_sha256(base_path) and row.get("sidecar_sha256") == file_sha256(feature_path):
                    reports[index] = row
    todo = [index for index in selected if index not in reports]
    predictor = None
    if todo:
        _deterministic(19000)
        predictor = CoTrackerOnlinePredictor(checkpoint=verified["backbone"]["path"]).to(args.device).eval()
    print(json.dumps({"stage":"cache_start","partition":args.partition,"indices":selected,"performance_metrics_computed":False}), flush=True)
    for source_index in selected:
        if source_index in reports:
            print(json.dumps({"stage":"resume_skip","source_index":source_index}), flush=True); continue
        sample, metadata = load_manifest_sample(manifest, source_index)
        prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
        started = time.time(); _deterministic(19000 + source_index)
        tensors, feature32 = _extract_once(predictor, prepared, args.device)
        replay_exact = None
        if args.partition == "final_holdout" and source_index in replay_indices:
            _deterministic(19000 + source_index)
            second_tensors, second_feature = _extract_once(predictor, prepared, args.device)
            replay_exact = _tensor_dict_exact(tensors, second_tensors) and torch.equal(feature32, second_feature)
            if not replay_exact: raise RuntimeError(f"full extraction replay mismatch: {source_index}")
        feature16 = feature32.half().contiguous()
        quant = feature_quantization_audit(feature32, feature16)
        validate_quantization_audit(
            quant,
            max_abs_max=float(config["cache_contract"]["feature_map_quantization_max_abs_max"]),
            min_cosine_min=float(config["cache_contract"]["feature_map_quantization_min_cosine_min"]),
        )
        native_hashes = {key: tensor_sha256(tensors[key]) for key in (
            "native_coords_xy_px","native_visibility_probability","native_confidence_probability","native_joint_probability","native_visibility"
        )}
        smoke_checks = _smoke_reference_parity(tensors, feature16) if args.partition == "implementation_smoke" else None
        base_path, feature_path, report_path = _paths(output_root, args.partition, source_index)
        identity = {"video_name": prepared["video_name"], "source_tfrecord": prepared["source_tfrecord"], "source_record_index": prepared["source_record_index"]}
        base_artifact = {
            "schema_version": FINAL_HOLDOUT_CACHE_SCHEMA,
            "provenance": {"partition":args.partition,"source_index":source_index,"sample_identity":identity,"config_sha256":config["_config_sha256"],"candidate_generation_ground_truth_free":True},
            "tensors": tensors,
            "tensor_hashes": {key:tensor_sha256(value) for key,value in tensors.items()},
        }
        torch.save(base_artifact, base_path)
        feature_artifact = {
            "schema_version": CMCP_FEATURE_CACHE_SCHEMA_VERSION,
            "provenance": {"partition":args.partition,"source_index":source_index,"sample_identity":identity,"protocol_sha256":config["_config_sha256"],"base_sidecar":str(base_path),"base_sidecar_sha256":file_sha256(base_path),"backbone_checkpoint_sha256":config["backbone"]["checkpoint_sha256"],"candidate_generation_ground_truth_free":True},
            "feature_maps_f16": feature16,
            "feature_maps_f16_sha256": tensor_sha256(feature16),
            "feature_maps_float32_sha256": tensor_sha256(feature32),
            "quantization_audit": quant,
            "native_state_hashes": native_hashes,
        }
        torch.save(feature_artifact, feature_path)
        row = {
            "schema_version":"routeD_cmcp_final_holdout_video_report_v0","partition":args.partition,"source_index":source_index,"sample_identity":identity,
            "protocol_sha256":config["_config_sha256"],"base_sidecar":str(base_path),"base_sidecar_sha256":file_sha256(base_path),
            "sidecar":str(feature_path),"sidecar_sha256":file_sha256(feature_path),"feature_map_shape":list(feature16.shape),
            "feature_maps_f16_sha256":tensor_sha256(feature16),"feature_maps_float32_sha256":tensor_sha256(feature32),
            "quantization_audit":quant,"native_state_hashes":native_hashes,"deterministic_full_extraction_replay_exact":replay_exact,
            "smoke_reference_parity":smoke_checks,"seconds":round(time.time()-started,3),
            "integrity":{"performance_metrics_computed":False,"calibration_read":False,"tapvid_davis_read":False,"tapvid_kinetics_read":False}
        }
        report_path.write_text(json.dumps(row,indent=2,ensure_ascii=False)+"\n"); reports[source_index]=row
        print(json.dumps({"stage":"video_cached","source_index":source_index,"replay_exact":replay_exact,"quantization":quant,"performance_metrics_computed":False}),flush=True)
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
    completed = sorted(reports)
    complete = completed == indices
    if args.partition == "final_holdout" and not complete:
        raise RuntimeError("final holdout cache must complete atomically")
    rows = [reports[index] for index in completed]
    index = {
        "schema_version":CMCP_FEATURE_INDEX_SCHEMA_VERSION,"protocol_sha256":config["_config_sha256"],"partition":args.partition,
        "expected_source_indices":indices,"completed_source_indices":completed,"expected_count":len(indices),"completed_count":len(rows),"complete":complete,
        "feature_dtype":"torch.float16","feature_channels":128,"videos":rows,
        "aggregate_quantization_audit":{"max_abs_max":max(row["quantization_audit"]["max_abs"] for row in rows),"min_cosine_min":min(row["quantization_audit"]["min_cosine"] for row in rows)},
        "integrity":{"performance_metrics_computed":False,"calibration_read":False,"tapvid_davis_read":False,"tapvid_kinetics_read":False}
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_root / args.partition / "cache_index.json"
    index_path.write_text(json.dumps(index,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"cache_complete","index":str(index_path),"complete":complete,"count":len(rows),"performance_metrics_computed":False},indent=2),flush=True)

if __name__ == "__main__": main()
