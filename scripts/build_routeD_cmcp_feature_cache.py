#!/usr/bin/env python3
"""Build resumable frozen float16 CoTracker feature-map caches for CMCP."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

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
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    CMCP_FEATURE_CACHE_SCHEMA_VERSION,
    CMCP_FEATURE_INDEX_SCHEMA_VERSION,
    feature_quantization_audit,
    validate_quantization_audit,
    verify_feature_cache_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
    load_protocol,
    partition_indices,
    resolve_manifest_path,
    verify_protocol_files,
)
from scripts.audit_routeD_cotracker3_interface import (
    compute_fmaps,
    load_manifest_sample,
    prepare_sample,
    run_true_streaming,
    set_deterministic,
)
from scripts.build_routeD_cotracker3_raw_v1_cache import (
    _load_base_artifact,
    _load_base_index,
    _resolve_checkpoint,
    _verify_backbone_native_state,
)

DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_BASE_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717"
ALLOWED_PARTITIONS = ("fit", "model_validation")


def _paths(output_dir: Path, source_index: int) -> tuple[Path, Path]:
    stem = f"video_{source_index:05d}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}.json"


def _valid_existing(
    sidecar: Path,
    report_path: Path,
    *,
    protocol_sha256: str,
    base_sidecar_sha256: str,
) -> bool:
    if not sidecar.exists() or not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        if report.get("sidecar_sha256") != file_sha256(sidecar):
            return False
        verify_feature_cache_artifact(
            sidecar,
            expected_protocol_sha256=protocol_sha256,
            expected_base_sidecar_sha256=base_sidecar_sha256,
        )
        return True
    except Exception:
        return False


def _save_video(
    *,
    sidecar: Path,
    report_path: Path,
    partition: str,
    source_index: int,
    sample_identity: dict[str, Any],
    protocol: dict[str, Any],
    base_path: Path,
    base_row: dict[str, Any],
    feature_maps: torch.Tensor,
    native_state_hashes: dict[str, str],
    replay_exact: bool | None,
    seconds: float,
) -> dict[str, Any]:
    feature32 = feature_maps.detach().float().cpu().contiguous()
    feature16 = feature32.half().contiguous()
    quantization = feature_quantization_audit(feature32, feature16)
    validate_quantization_audit(quantization)
    artifact = {
        "schema_version": CMCP_FEATURE_CACHE_SCHEMA_VERSION,
        "provenance": {
            "partition": partition,
            "source_index": source_index,
            "sample_identity": sample_identity,
            "protocol_sha256": protocol["_protocol_sha256"],
            "base_sidecar": str(base_path),
            "base_sidecar_sha256": file_sha256(base_path),
            "backbone_checkpoint_sha256": protocol["backbone"]["checkpoint_sha256"],
            "candidate_generation_ground_truth_free": True,
        },
        "feature_maps_f16": feature16,
        "feature_maps_f16_sha256": tensor_sha256(feature16),
        "feature_maps_float32_sha256": tensor_sha256(feature32),
        "quantization_audit": quantization,
        "native_state_hashes": native_state_hashes,
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_cmcp_feature_map_cache_video_report_v1",
        "partition": partition,
        "source_index": source_index,
        "sample_identity": sample_identity,
        "protocol_sha256": protocol["_protocol_sha256"],
        "base_sidecar": str(base_path),
        "base_sidecar_sha256": file_sha256(base_path),
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "feature_map_shape": list(feature16.shape),
        "feature_maps_f16_sha256": artifact["feature_maps_f16_sha256"],
        "feature_maps_float32_sha256": artifact["feature_maps_float32_sha256"],
        "quantization_audit": quantization,
        "native_state_hashes": native_state_hashes,
        "deterministic_float32_replay_exact": replay_exact,
        "seconds": round(seconds, 3),
        "integrity": {
            "ground_truth_used_for_feature_map": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _write_index(
    *,
    protocol: dict[str, Any],
    partition: str,
    expected_indices: tuple[int, ...],
    reports: list[dict[str, Any]],
    base_index_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    completed = [int(row["source_index"]) for row in reports]
    complete = completed == list(expected_indices)
    max_abs = max(float(row["quantization_audit"]["max_abs"]) for row in reports)
    min_cosine = min(float(row["quantization_audit"]["min_cosine"]) for row in reports)
    index = {
        "schema_version": CMCP_FEATURE_INDEX_SCHEMA_VERSION,
        "protocol_sha256": protocol["_protocol_sha256"],
        "partition": partition,
        "base_cache_index": str(base_index_path),
        "base_cache_index_sha256": file_sha256(base_index_path),
        "expected_source_indices": list(expected_indices),
        "completed_source_indices": completed,
        "expected_count": len(expected_indices),
        "completed_count": len(completed),
        "complete": complete,
        "feature_dtype": "torch.float16",
        "feature_channels": 128,
        "videos": reports,
        "aggregate_quantization_audit": {
            "max_abs_max": max_abs,
            "min_cosine_min": min_cosine,
        },
        "integrity": {
            "ground_truth_used_for_feature_map": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    path = output_dir / "cache_index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--base-cache-root", default=str(DEFAULT_BASE_ROOT))
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--partition", choices=ALLOWED_PARTITIONS, required=True)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replay-indices", default="0,47,48,63")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = load_protocol(args.protocol)
    verify_protocol_files(protocol)
    expected_indices = partition_indices(protocol, args.partition)
    if args.start_offset < 0 or args.start_offset >= len(expected_indices):
        raise ValueError("start-offset outside partition")
    selected = list(expected_indices[args.start_offset :])
    if args.max_videos > 0:
        selected = selected[: args.max_videos]
    base_index_path = Path(args.base_cache_root).resolve() / args.partition / "cache_index.json"
    base_index = _load_base_index(base_index_path, partition=args.partition)
    if base_index["protocol_sha256"] != protocol["_protocol_sha256"]:
        raise ValueError("base cache protocol mismatch")
    base_by_index = {int(row["source_index"]): row for row in base_index["videos"]}
    if tuple(sorted(base_by_index)) != tuple(expected_indices):
        raise ValueError("base cache membership mismatch")

    output_dir = Path(args.output_root).resolve() / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    report_by_index: dict[int, dict[str, Any]] = {}
    if args.resume:
        for source_index in expected_indices:
            base_row = base_by_index[source_index]
            sidecar, report_path = _paths(output_dir, source_index)
            if _valid_existing(
                sidecar,
                report_path,
                protocol_sha256=protocol["_protocol_sha256"],
                base_sidecar_sha256=base_row["sidecar_sha256"],
            ):
                report_by_index[source_index] = json.loads(report_path.read_text())

    to_process = [index for index in selected if index not in report_by_index]
    predictor = None
    if to_process:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_deterministic(17000)
        predictor = CoTrackerOnlinePredictor(
            checkpoint=str(_resolve_checkpoint(protocol))
        ).to(args.device).eval()
    manifest_alias = protocol["partitions"][args.partition]["source"]
    manifest = resolve_manifest_path(protocol, manifest_alias)
    replay_indices = {int(value) for value in args.replay_indices.split(",") if value.strip()}

    print(json.dumps({
        "stage": "cmcp_feature_partition_start",
        "partition": args.partition,
        "selected_indices": selected,
        "protocol_sha256": protocol["_protocol_sha256"],
        "base_cache_index_sha256": file_sha256(base_index_path),
    }), flush=True)

    for ordinal, source_index in enumerate(selected, start=1):
        if source_index in report_by_index:
            print(json.dumps({"stage": "resume_skip", "source_index": source_index}), flush=True)
            continue
        if predictor is None:
            raise RuntimeError("missing predictor")
        started = time.time()
        base_row = base_by_index[source_index]
        base_path, base_artifact = _load_base_artifact(base_row)
        sample, _ = load_manifest_sample(manifest, source_index)
        prepared = prepare_sample(sample, int(protocol["candidate_generator"]["input_raster"]))
        if str(prepared["video_name"]) != str(base_row["sample_identity"]["video_name"]):
            raise ValueError("sample identity drift")
        print(json.dumps({
            "stage": "cmcp_feature_video_start",
            "ordinal": ordinal,
            "selected_count": len(selected),
            "source_index": source_index,
            "video_name": prepared["video_name"],
        }), flush=True)
        set_deterministic(17000 + source_index)
        video = prepared["video"].to(args.device)
        run_true_streaming(predictor, video, prepared["query_points_tyx"].to(args.device))
        base_tensors = base_artifact["tensors"]
        point_count, frame_count = base_tensors["native_visibility"].shape
        native_hashes = _verify_backbone_native_state(
            predictor,
            base_tensors,
            frame_count=frame_count,
            point_count=point_count,
            input_raster=int(protocol["candidate_generator"]["input_raster"]),
        )
        feature_maps = compute_fmaps(predictor, video)
        replay_exact = None
        if source_index in replay_indices:
            second = compute_fmaps(predictor, video)
            replay_exact = bool(torch.equal(feature_maps, second))
            if not replay_exact:
                raise RuntimeError(f"feature-map replay mismatch for {source_index}")
        sidecar, report_path = _paths(output_dir, source_index)
        report = _save_video(
            sidecar=sidecar,
            report_path=report_path,
            partition=args.partition,
            source_index=source_index,
            sample_identity=base_row["sample_identity"],
            protocol=protocol,
            base_path=base_path,
            base_row=base_row,
            feature_maps=feature_maps,
            native_state_hashes=native_hashes,
            replay_exact=replay_exact,
            seconds=time.time() - started,
        )
        report_by_index[source_index] = report
        print(json.dumps({
            "stage": "cmcp_feature_video_complete",
            "source_index": source_index,
            "max_abs": report["quantization_audit"]["max_abs"],
            "min_cosine": report["quantization_audit"]["min_cosine"],
            "replay_exact": replay_exact,
            "seconds": report["seconds"],
        }), flush=True)
        del video, feature_maps
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    reports = [report_by_index[index] for index in sorted(report_by_index)]
    index = _write_index(
        protocol=protocol,
        partition=args.partition,
        expected_indices=expected_indices,
        reports=reports,
        base_index_path=base_index_path,
        output_dir=output_dir,
    )
    print(json.dumps({
        "stage": "cmcp_feature_partition_complete",
        "cache_index": str(output_dir / "cache_index.json"),
        "completed_count": index["completed_count"],
        "expected_count": index["expected_count"],
        "complete": index["complete"],
        "aggregate_quantization_audit": index["aggregate_quantization_audit"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
