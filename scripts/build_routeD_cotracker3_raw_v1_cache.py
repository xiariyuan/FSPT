#!/usr/bin/env python3
"""Build raw-v1 MUSR representation caches from frozen stage-0 candidates."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import (
    sample_feature_at_xy,
    tensor_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    RAW_V1_VIDEO_SIDECAR_SCHEMA_VERSION,
    VIDEO_SIDECAR_SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    load_protocol,
    partition_indices,
    resolve_manifest_path,
    verify_protocol_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_raw_representation import (
    RAW_V1_CANDIDATE_FEATURE_DIM,
    RAW_V1_SCHEMA_VERSION,
    RAW_V1_STATE_FEATURE_DIM,
    build_raw_v1_candidate_features_for_frame,
    build_raw_v1_state_features,
    verify_raw_v1_frozen_contract,
)
from scripts.audit_routeD_cotracker3_interface import (
    _jsonable,
    compute_fmaps,
    extract_online_track_features,
    load_manifest_sample,
    prepare_sample,
    run_true_streaming,
    set_deterministic,
)

DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_BASE_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/routeD_musr_raw_v1_cache_20260717"
ALLOWED_PARTITIONS = ("fit", "model_validation")


def _resolve_checkpoint(protocol: dict[str, Any]) -> Path:
    path = Path(protocol["backbone"]["checkpoint"])
    return path if path.is_absolute() else (EXTERNAL_ROOT / path).resolve()


def _load_base_index(path: Path, *, partition: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("partition") != partition or not payload.get("complete"):
        raise ValueError(f"base cache is not a complete {partition} partition")
    if payload.get("expected_source_indices") != payload.get("completed_source_indices"):
        raise ValueError("base cache membership mismatch")
    payload["_index_path"] = str(path.resolve())
    payload["_index_sha256"] = file_sha256(path)
    return payload


def _load_base_artifact(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(row["sidecar"])).resolve()
    if file_sha256(path) != str(row["sidecar_sha256"]):
        raise ValueError(f"base sidecar hash mismatch: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != VIDEO_SIDECAR_SCHEMA_VERSION:
        raise ValueError(f"unexpected base sidecar schema: {path}")
    return path, artifact


def _verify_backbone_native_state(
    predictor: CoTrackerOnlinePredictor,
    base_tensors: dict[str, torch.Tensor],
    *,
    frame_count: int,
    point_count: int,
    input_raster: int,
) -> dict[str, str]:
    coords = predictor.model.online_coords_predicted[0, :frame_count, :point_count].float()
    raw_vis = predictor.model.online_vis_predicted[0, :frame_count, :point_count].float()
    raw_conf = predictor.model.online_conf_predicted[0, :frame_count, :point_count].float()
    interp_height, interp_width = predictor.interp_shape
    coords = coords.clone()
    coords[..., 0] *= float(input_raster - 1) / float(max(interp_width - 1, 1))
    coords[..., 1] *= float(input_raster - 1) / float(max(interp_height - 1, 1))
    coords = coords.permute(1, 0, 2).contiguous().cpu()
    visibility_probability = torch.sigmoid(raw_vis).permute(1, 0).contiguous().cpu()
    confidence_probability = torch.sigmoid(raw_conf).permute(1, 0).contiguous().cpu()
    joint = visibility_probability * confidence_probability
    visible = joint > 0.6
    observed = {
        "native_coords_xy_px": coords,
        "native_visibility_probability": visibility_probability,
        "native_confidence_probability": confidence_probability,
        "native_joint_probability": joint,
        "native_visibility": visible,
    }
    hashes = {}
    for key, value in observed.items():
        if not torch.equal(value, base_tensors[key]):
            max_abs = None
            if value.dtype.is_floating_point:
                max_abs = float((value - base_tensors[key]).abs().max().item())
            raise ValueError(f"frozen backbone state drift for {key}; max_abs={max_abs}")
        hashes[key] = tensor_sha256(value)
    return hashes


def build_raw_v1_tensors(
    predictor: CoTrackerOnlinePredictor,
    prepared: dict[str, Any],
    base_tensors: dict[str, torch.Tensor],
    *,
    input_raster: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Append raw-v1 representation while preserving every frozen tensor."""
    video = prepared["video"].to(next(predictor.parameters()).device)
    queries = prepared["query_points_tyx"].to(video.device)
    point_count, frame_count = base_tensors["native_visibility"].shape
    if base_tensors["candidate_features"].shape[-1] != 64:
        raise ValueError("base cache is not the frozen 64-D representation")
    if base_tensors["state_features"].shape[-1] != 32:
        raise ValueError("base cache is not the frozen 32-D representation")

    native_hashes = _verify_backbone_native_state(
        predictor,
        base_tensors,
        frame_count=frame_count,
        point_count=point_count,
        input_raster=input_raster,
    )
    fmaps = compute_fmaps(predictor, video)
    query_frames = queries[:, 0].round().long().clamp(0, frame_count - 1)
    query_xy = torch.stack(
        [queries[:, 2] * float(input_raster - 1), queries[:, 1] * float(input_raster - 1)],
        dim=-1,
    )
    support_features = torch.stack(
        [
            sample_feature_at_xy(
                fmaps[int(query_frames[index].item())],
                query_xy[index],
                input_height=input_raster,
                input_width=input_raster,
            )
            for index in range(point_count)
        ]
    )
    online_track_features, track_info = extract_online_track_features(
        predictor, point_count, support_features
    )
    if online_track_features.shape != (point_count, 128):
        raise ValueError(f"unexpected full online-track state {tuple(online_track_features.shape)}")

    candidates = int(base_tensors["candidate_coords_xy_px"].shape[-2])
    raw_candidate = torch.zeros(
        point_count,
        frame_count,
        candidates,
        RAW_V1_CANDIDATE_FEATURE_DIM,
        device=video.device,
        dtype=torch.float32,
    )
    for frame_index in range(frame_count):
        active = frame_index >= query_frames
        frame_feature = build_raw_v1_candidate_features_for_frame(
            structured_candidate_features=base_tensors["candidate_features"][:, frame_index].to(video.device),
            candidate_valid_mask=base_tensors["candidate_valid_mask"][:, frame_index].to(video.device),
            candidate_xy_px=base_tensors["candidate_coords_xy_px"][:, frame_index].to(video.device),
            support_features=support_features,
            fmap=fmaps[frame_index],
            input_height=input_raster,
            input_width=input_raster,
        )
        raw_candidate[:, frame_index] = torch.where(
            active[:, None, None], frame_feature, torch.zeros_like(frame_feature)
        )

    raw_state = build_raw_v1_state_features(
        base_tensors["state_features"].to(video.device), online_track_features
    )
    active_state = (
        torch.arange(frame_count, device=video.device)[None, :] >= query_frames[:, None]
    )
    raw_state[..., 32:] = torch.where(
        active_state.unsqueeze(-1), raw_state[..., 32:], torch.zeros_like(raw_state[..., 32:])
    )

    raw_tensors = {key: value.detach().cpu().clone() for key, value in base_tensors.items()}
    raw_tensors["candidate_features"] = raw_candidate.detach().cpu()
    raw_tensors["state_features"] = raw_state.detach().cpu()
    frozen_hashes = verify_raw_v1_frozen_contract(base_tensors, raw_tensors)
    runtime = {
        "representation_schema_version": RAW_V1_SCHEMA_VERSION,
        "candidate_feature_dim": RAW_V1_CANDIDATE_FEATURE_DIM,
        "state_feature_dim": RAW_V1_STATE_FEATURE_DIM,
        "fmap_shape": list(fmaps.shape),
        "support_feature_shape": list(support_features.shape),
        "online_track_feature": track_info,
        "native_state_hashes": native_hashes,
        "frozen_tensor_hashes": frozen_hashes,
        "pre_query_raw_append_zero": True,
    }
    return raw_tensors, runtime


def _paths(output_dir: Path, source_index: int) -> tuple[Path, Path]:
    stem = f"video_{source_index:05d}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}.json"


def _valid_existing(
    sidecar: Path,
    report_path: Path,
    *,
    base_sidecar_sha256: str,
    protocol_sha256: str,
) -> bool:
    if not sidecar.exists() or not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        if report.get("representation_schema_version") != RAW_V1_SCHEMA_VERSION:
            return False
        if report.get("protocol_sha256") != protocol_sha256:
            return False
        if report.get("base_sidecar_sha256") != base_sidecar_sha256:
            return False
        if report.get("sidecar_sha256") != file_sha256(sidecar):
            return False
        artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
        return artifact.get("schema_version") == RAW_V1_VIDEO_SIDECAR_SCHEMA_VERSION
    except Exception:
        return False


def _save(
    *,
    sidecar: Path,
    report_path: Path,
    partition: str,
    source_index: int,
    base_row: dict[str, Any],
    base_path: Path,
    base_artifact: dict[str, Any],
    raw_tensors: dict[str, torch.Tensor],
    runtime: dict[str, Any],
    protocol: dict[str, Any],
    replay: dict[str, Any] | None,
    seconds: float,
) -> dict[str, Any]:
    raw_hashes = {
        "candidate_features": tensor_sha256(raw_tensors["candidate_features"]),
        "state_features": tensor_sha256(raw_tensors["state_features"]),
        "candidate_coords_xy_px": tensor_sha256(raw_tensors["candidate_coords_xy_px"]),
        "candidate_valid_mask": tensor_sha256(raw_tensors["candidate_valid_mask"]),
        "source_ids": tensor_sha256(raw_tensors["source_ids"]),
    }
    provenance = dict(base_artifact.get("provenance", {}))
    provenance["raw_representation"] = {
        "schema_version": RAW_V1_SCHEMA_VERSION,
        "candidate_feature_dim": RAW_V1_CANDIDATE_FEATURE_DIM,
        "state_feature_dim": RAW_V1_STATE_FEATURE_DIM,
        "base_sidecar": str(base_path),
        "base_sidecar_sha256": file_sha256(base_path),
        "runtime": runtime,
    }
    artifact = {
        "schema_version": RAW_V1_VIDEO_SIDECAR_SCHEMA_VERSION,
        "provenance": provenance,
        "tensors": raw_tensors,
        "tensor_hashes": raw_hashes,
        "audit": base_artifact["audit"],
        "base_sidecar_sha256": file_sha256(base_path),
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_musr_cotracker3_raw_v1_video_report_v1",
        "representation_schema_version": RAW_V1_SCHEMA_VERSION,
        "protocol_sha256": protocol["_protocol_sha256"],
        "partition": partition,
        "source_alias": base_row["source_alias"],
        "source_index": source_index,
        "sample_identity": base_row["sample_identity"],
        "base_sidecar": str(base_path),
        "base_sidecar_sha256": file_sha256(base_path),
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "tensor_hashes": raw_hashes,
        "candidate_coordinate_hash_unchanged": (
            raw_hashes["candidate_coords_xy_px"]
            == base_row["tensor_hashes"]["candidate_coords_xy_px"]
        ),
        "frozen_contract_pass": True,
        "deterministic_raw_replay": replay,
        "oracle_audit": base_artifact["audit"],
        "seconds": round(seconds, 3),
        "integrity": {
            "candidate_coordinates_changed": False,
            "ground_truth_used_for_representation": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--base-cache-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--partition", choices=ALLOWED_PARTITIONS, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--replay-indices", default="0,47,48,63")
    args = parser.parse_args()
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
        raise ValueError("base index membership differs from frozen protocol")

    output_dir = Path(args.output_root).resolve() / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_indices = {int(value) for value in args.replay_indices.split(",") if value.strip()}
    report_by_index: dict[int, dict[str, Any]] = {}
    if args.resume:
        for source_index in expected_indices:
            base_row = base_by_index[source_index]
            sidecar, report_path = _paths(output_dir, source_index)
            if _valid_existing(
                sidecar,
                report_path,
                base_sidecar_sha256=str(base_row["sidecar_sha256"]),
                protocol_sha256=protocol["_protocol_sha256"],
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

    print(json.dumps({
        "stage": "raw_v1_partition_start",
        "partition": args.partition,
        "selected_indices": selected,
        "base_cache_index_sha256": base_index["_index_sha256"],
        "protocol_sha256": protocol["_protocol_sha256"],
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
            "stage": "raw_v1_video_start",
            "ordinal": ordinal,
            "selected_count": len(selected),
            "source_index": source_index,
            "video_name": prepared["video_name"],
        }), flush=True)
        set_deterministic(17000 + source_index)
        video = prepared["video"].to(args.device)
        run_true_streaming(predictor, video, prepared["query_points_tyx"].to(args.device))
        raw_tensors, runtime = build_raw_v1_tensors(
            predictor,
            prepared,
            base_artifact["tensors"],
            input_raster=int(protocol["candidate_generator"]["input_raster"]),
        )
        replay = None
        if source_index in replay_indices:
            second, _ = build_raw_v1_tensors(
                predictor,
                prepared,
                base_artifact["tensors"],
                input_raster=int(protocol["candidate_generator"]["input_raster"]),
            )
            replay = {
                "candidate_features_exact": torch.equal(
                    raw_tensors["candidate_features"], second["candidate_features"]
                ),
                "state_features_exact": torch.equal(
                    raw_tensors["state_features"], second["state_features"]
                ),
                "candidate_features_sha256": tensor_sha256(raw_tensors["candidate_features"]),
                "state_features_sha256": tensor_sha256(raw_tensors["state_features"]),
            }
            replay["exact"] = bool(
                replay["candidate_features_exact"] and replay["state_features_exact"]
            )
            if not replay["exact"]:
                raise RuntimeError(f"raw-v1 replay mismatch for {source_index}")
        sidecar, report_path = _paths(output_dir, source_index)
        report = _save(
            sidecar=sidecar,
            report_path=report_path,
            partition=args.partition,
            source_index=source_index,
            base_row=base_row,
            base_path=base_path,
            base_artifact=base_artifact,
            raw_tensors=raw_tensors,
            runtime=runtime,
            protocol=protocol,
            replay=replay,
            seconds=time.time() - started,
        )
        report_by_index[source_index] = report
        print(json.dumps({
            "stage": "raw_v1_video_complete",
            "source_index": source_index,
            "candidate_feature_sha256": report["tensor_hashes"]["candidate_features"],
            "coordinate_hash_unchanged": report["candidate_coordinate_hash_unchanged"],
            "replay_exact": None if replay is None else replay["exact"],
            "seconds": report["seconds"],
        }), flush=True)
        del video, raw_tensors
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    reports = [report_by_index[index] for index in sorted(report_by_index)]
    completed = [int(row["source_index"]) for row in reports]
    complete = completed == list(expected_indices)
    index = {
        "schema_version": "routeD_musr_cotracker3_raw_v1_cache_index_v1",
        "representation_schema_version": RAW_V1_SCHEMA_VERSION,
        "candidate_feature_dim": RAW_V1_CANDIDATE_FEATURE_DIM,
        "state_feature_dim": RAW_V1_STATE_FEATURE_DIM,
        "protocol_path": protocol["_protocol_path"],
        "protocol_sha256": protocol["_protocol_sha256"],
        "partition": args.partition,
        "source_alias": manifest_alias,
        "expected_source_indices": list(expected_indices),
        "completed_source_indices": completed,
        "expected_count": len(expected_indices),
        "completed_count": len(completed),
        "complete": complete,
        "base_cache_index": str(base_index_path),
        "base_cache_index_sha256": base_index["_index_sha256"],
        "videos": reports,
        "aggregate_oracle_audit": base_index["aggregate_oracle_audit"],
        "qualification_gate": None,
        "training_authorized": False,
        "integrity": {
            "candidate_coordinates_changed": False,
            "ground_truth_used_for_representation": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_dir / "cache_index.json"
    index_path.write_text(json.dumps(_jsonable(index), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "stage": "raw_v1_partition_complete",
        "cache_index": str(index_path),
        "completed_count": len(completed),
        "expected_count": len(expected_indices),
        "complete": complete,
        "payload_sha256": index["cache_index_payload_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
