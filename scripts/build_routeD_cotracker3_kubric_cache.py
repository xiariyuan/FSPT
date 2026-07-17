#!/usr/bin/env python3
"""Build resumable, identity-frozen CoTracker3 MUSR candidate caches on Kubric."""
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
# External CoTracker code is needed, but the active repository must remain
# first so an older /gemini/code/FSPT/projects package cannot shadow new code.
for path in (COTRACKER_ROOT, EXTERNAL_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    CACHE_INDEX_SCHEMA_VERSION,
    aggregate_partition_sidecars,
    canonical_json_sha256,
    collect_protocol_identities,
    evaluate_qualification_gates,
    file_sha256,
    identity_digest,
    load_protocol,
    partition_indices,
    resolve_manifest_path,
    verify_protocol_files,
)
from scripts.audit_routeD_cotracker3_interface import (
    _jsonable,
    build_adapter_tensors,
    build_provenance,
    compare_replays,
    compute_metrics_and_oracle,
    core_hashes,
    load_manifest_sample,
    prepare_sample,
    run_true_streaming,
    set_deterministic,
)

DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"


def _resolve_checkpoint(protocol: dict[str, Any]) -> Path:
    path = Path(protocol["backbone"]["checkpoint"])
    return path if path.is_absolute() else (EXTERNAL_ROOT / path).resolve()


def _partition_spec(protocol: dict[str, Any], partition: str) -> dict[str, Any]:
    if partition not in protocol["partitions"]:
        raise KeyError(f"unknown partition {partition}")
    return protocol["partitions"][partition]


def _video_paths(output_dir: Path, source_index: int) -> tuple[Path, Path]:
    stem = f"video_{source_index:05d}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}.json"


def _valid_existing(sidecar: Path, report_path: Path, protocol_sha: str) -> bool:
    if not sidecar.exists() or not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        if report.get("protocol_sha256") != protocol_sha:
            return False
        if report.get("sidecar_sha256") != file_sha256(sidecar):
            return False
        artifact = torch.load(sidecar, map_location="cpu", weights_only=False)
        tensors = artifact["tensors"]
        return bool(torch.equal(tensors["candidate_coords_xy_px"][..., 0, :], tensors["native_coords_xy_px"]))
    except Exception:
        return False


def _save_video(
    *,
    sidecar: Path,
    report_path: Path,
    protocol: dict[str, Any],
    partition: str,
    source_alias: str,
    source_index: int,
    sample_metadata: dict[str, Any],
    prepared: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    runtime: dict[str, Any],
    audit: dict[str, Any],
    replay: dict[str, Any] | None,
    checkpoint: Path,
    seconds: float,
) -> dict[str, Any]:
    generator = protocol["candidate_generator"]
    fake_args = argparse.Namespace(
        query_mode=generator["query_mode"],
        input_raster=generator["input_raster"],
        metric_raster=generator["metric_raster"],
        candidate_topk=generator["candidate_topk"],
        search_radius_px=generator["search_radius_px"],
        routing_disabled=True,
    )
    provenance = build_provenance(
        args=fake_args,
        checkpoint=checkpoint,
        sample_metadata=sample_metadata,
        prepared=prepared,
        runtime=runtime,
    )
    provenance["cache_protocol"] = {
        "path": protocol["_protocol_path"],
        "sha256": protocol["_protocol_sha256"],
        "partition": partition,
        "source_alias": source_alias,
        "source_index": source_index,
    }
    artifact = {
        "schema_version": provenance["schema_version"],
        "provenance": provenance,
        "tensors": tensors,
        "tensor_hashes": core_hashes(tensors),
        "audit": audit,
    }
    torch.save(artifact, sidecar)
    parity = bool(torch.equal(tensors["candidate_coords_xy_px"][..., 0, :], tensors["native_coords_xy_px"]))
    report = {
        "schema_version": "routeD_musr_cotracker3_kubric_video_report_v1",
        "protocol_sha256": protocol["_protocol_sha256"],
        "partition": partition,
        "source_alias": source_alias,
        "source_index": source_index,
        "sample_identity": {
            "video_name": prepared["video_name"],
            "source_tfrecord": prepared["source_tfrecord"],
            "source_record_index": prepared["source_record_index"],
        },
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "tensor_hashes": artifact["tensor_hashes"],
        "routing_disabled_native_parity": parity,
        "deterministic_adapter_replay": replay,
        "oracle_audit": audit,
        "seconds": round(seconds, 3),
        "candidate_generation_ground_truth_free": True,
        "kinetics_read": False,
        "davis_read": False,
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n")
    return report


def _build_index(
    *,
    protocol: dict[str, Any],
    partition: str,
    source_alias: str,
    requested_indices: list[int],
    expected_indices: tuple[int, ...],
    output_dir: Path,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    sidecars = [Path(row["sidecar"]) for row in reports]
    aggregate = aggregate_partition_sidecars(
        sidecars,
        metric_raster=int(protocol["candidate_generator"]["metric_raster"]),
    )
    complete = requested_indices == list(expected_indices) and len(reports) == len(expected_indices)
    gate = None
    if partition == protocol["qualification_partition"]:
        gate = evaluate_qualification_gates(
            aggregate,
            protocol["qualification_gates"],
            complete=complete,
        )
    identities = collect_protocol_identities(protocol)[partition]
    selected_identity_digest = identity_digest(
        [identities[list(expected_indices).index(index)] for index in requested_indices]
    )
    index = {
        "schema_version": CACHE_INDEX_SCHEMA_VERSION,
        "protocol_path": protocol["_protocol_path"],
        "protocol_sha256": protocol["_protocol_sha256"],
        "partition": partition,
        "source_alias": source_alias,
        "expected_source_indices": list(expected_indices),
        "completed_source_indices": requested_indices,
        "expected_count": len(expected_indices),
        "completed_count": len(reports),
        "complete": complete,
        "selected_identity_sha256": selected_identity_digest,
        "videos": reports,
        "aggregate_oracle_audit": aggregate,
        "qualification_gate": gate,
        "training_authorized": bool(gate and gate["pass"]),
        "integrity": {
            "candidate_generation_ground_truth_free": True,
            "kinetics_read": False,
            "davis_read": False,
            "final_holdout_read": partition == protocol["final_holdout_partition"],
        },
    }
    index["cache_index_payload_sha256"] = canonical_json_sha256(index)
    index_path = output_dir / "cache_index.json"
    index_path.write_text(json.dumps(_jsonable(index), indent=2, ensure_ascii=False) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--partition", default="candidate_qualification")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    verified = verify_protocol_files(protocol)
    all_identities = collect_protocol_identities(protocol)
    spec = _partition_spec(protocol, args.partition)
    source_alias = str(spec["source"])
    expected_indices = partition_indices(protocol, args.partition)
    if args.start_offset < 0 or args.start_offset >= len(expected_indices):
        raise ValueError("start-offset is outside the partition")
    selected = list(expected_indices[args.start_offset :])
    if args.max_videos > 0:
        selected = selected[: args.max_videos]
    if args.partition == protocol["final_holdout_partition"]:
        raise RuntimeError("final_holdout is locked until the learned system is frozen")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_root).resolve() / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = resolve_manifest_path(protocol, source_alias)
    checkpoint = _resolve_checkpoint(protocol)
    replay_indices = set(int(value) for value in protocol["deterministic_adapter_replay_indices"])
    generator = protocol["candidate_generator"]

    # Resume is partition-global, not only local to the requested chunk.  Scan
    # every expected member so start-offset/max-videos jobs can safely compose.
    report_by_index: dict[int, dict[str, Any]] = {}
    if args.resume:
        for existing_index in expected_indices:
            existing_sidecar, existing_report_path = _video_paths(output_dir, existing_index)
            if _valid_existing(
                existing_sidecar, existing_report_path, protocol["_protocol_sha256"]
            ):
                row = json.loads(existing_report_path.read_text())
                report_by_index[int(existing_index)] = row

    to_process = [index for index in selected if index not in report_by_index]
    predictor = None
    if to_process:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        set_deterministic(17000)
        predictor = (
            CoTrackerOnlinePredictor(checkpoint=str(checkpoint))
            .to(args.device)
            .eval()
        )

    print(json.dumps({
        "stage": "partition_start",
        "partition": args.partition,
        "selected_indices": selected,
        "protocol_sha256": protocol["_protocol_sha256"],
        "identity_sha256": identity_digest(all_identities[args.partition]),
        "verified": verified,
    }, ensure_ascii=False), flush=True)

    for ordinal, source_index in enumerate(selected, start=1):
        sidecar, report_path = _video_paths(output_dir, source_index)
        if source_index in report_by_index:
            print(json.dumps({"stage": "resume_skip", "source_index": source_index}), flush=True)
            continue
        if predictor is None:
            raise RuntimeError("internal error: missing predictor for uncached video")

        started = time.time()
        sample, sample_metadata = load_manifest_sample(manifest, source_index)
        prepared = prepare_sample(sample, int(generator["input_raster"]))
        print(json.dumps({
            "stage": "video_start",
            "ordinal": ordinal,
            "selected_count": len(selected),
            "source_index": source_index,
            "video_name": prepared["video_name"],
            "points": int(prepared["gt_tracks_yx"].shape[0]),
        }, ensure_ascii=False), flush=True)

        set_deterministic(17000 + source_index)
        video = prepared["video"].to(args.device)
        run_true_streaming(predictor, video, prepared["query_points_tyx"].to(args.device))
        tensors, runtime = build_adapter_tensors(
            predictor,
            prepared,
            input_raster=int(generator["input_raster"]),
            candidate_topk=int(generator["candidate_topk"]),
            search_radius_px=float(generator["search_radius_px"]),
        )
        replay = None
        if source_index in replay_indices:
            second_tensors, _ = build_adapter_tensors(
                predictor,
                prepared,
                input_raster=int(generator["input_raster"]),
                candidate_topk=int(generator["candidate_topk"]),
                search_radius_px=float(generator["search_radius_px"]),
            )
            replay = compare_replays(tensors, second_tensors)
            if not replay["exact"]:
                raise RuntimeError(f"adapter replay mismatch for source index {source_index}")
        audit = compute_metrics_and_oracle(
            tensors, metric_raster=int(generator["metric_raster"])
        )
        report = _save_video(
            sidecar=sidecar,
            report_path=report_path,
            protocol=protocol,
            partition=args.partition,
            source_alias=source_alias,
            source_index=source_index,
            sample_metadata=sample_metadata,
            prepared=prepared,
            tensors=tensors,
            runtime=runtime,
            audit=audit,
            replay=replay,
            checkpoint=checkpoint,
            seconds=time.time() - started,
        )
        report_by_index[int(source_index)] = report
        print(json.dumps({
            "stage": "video_complete",
            "source_index": source_index,
            "AJ_gain_points": audit["gain_points"]["AJ"],
            "delta_gain_points": audit["gain_points"]["delta_average"],
            "parity": report["routing_disabled_native_parity"],
            "replay_exact": None if replay is None else replay["exact"],
            "seconds": report["seconds"],
        }, ensure_ascii=False), flush=True)
        del video, tensors
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    reports = [report_by_index[index] for index in sorted(report_by_index)]
    completed_indices = [int(row["source_index"]) for row in reports]
    index = _build_index(
        protocol=protocol,
        partition=args.partition,
        source_alias=source_alias,
        requested_indices=completed_indices,
        expected_indices=expected_indices,
        output_dir=output_dir,
        reports=reports,
    )
    print(json.dumps(_jsonable({
        "stage": "partition_complete",
        "cache_index": str(output_dir / "cache_index.json"),
        "completed_count": index["completed_count"],
        "expected_count": index["expected_count"],
        "complete": index["complete"],
        "aggregate_gain_points": index["aggregate_oracle_audit"]["gain_points"],
        "qualification_gate": index["qualification_gate"],
        "training_authorized": index["training_authorized"],
    }), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
