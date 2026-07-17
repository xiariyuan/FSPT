"""Frozen Kubric cache protocol and partition-level oracle audit for Route-D MUSR.

The runtime exporter writes one sidecar per video.  This module contains the
pure, testable protocol/identity/aggregation logic so that cache qualification
cannot silently change sample membership, reuse the observed pilot, or report a
partial run as complete.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from datasets.metrics import compute_tapvid_metrics


PROTOCOL_SCHEMA_VERSION = "routeD_musr_kubric_cache_protocol_v0"
CACHE_INDEX_SCHEMA_VERSION = "routeD_musr_cotracker3_kubric_cache_index_v1"
VIDEO_SIDECAR_SCHEMA_VERSION = "routeD_musr_cotracker3_stage0_adapter_v1"
RAW_V1_VIDEO_SIDECAR_SCHEMA_VERSION = "routeD_musr_cotracker3_raw_v1_sidecar_v2"
SUPPORTED_VIDEO_SIDECAR_SCHEMA_VERSIONS = (
    VIDEO_SIDECAR_SCHEMA_VERSION,
    RAW_V1_VIDEO_SIDECAR_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class SampleIdentity:
    source_tfrecord: str
    source_record_index: int
    video_name: str

    @property
    def canonical(self) -> str:
        return f"{self.source_tfrecord}::{self.source_record_index}::{self.video_name}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    protocol["_protocol_path"] = str(protocol_path)
    protocol["_protocol_sha256"] = file_sha256(protocol_path)
    return protocol


def _expand_indices(spec: Mapping[str, Any]) -> tuple[int, ...]:
    if "indices" in spec:
        raw = spec["indices"]
        if not isinstance(raw, list):
            raise ValueError("partition indices must be a list")
        indices = [int(value) for value in raw]
    elif "inclusive_ranges" in spec:
        raw_ranges = spec["inclusive_ranges"]
        if not isinstance(raw_ranges, list):
            raise ValueError("inclusive_ranges must be a list")
        indices = []
        for pair in raw_ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each inclusive range must be [start, end]")
            start, end = int(pair[0]), int(pair[1])
            if start < 0 or end < start:
                raise ValueError(f"invalid inclusive range: {pair}")
            indices.extend(range(start, end + 1))
    else:
        raise ValueError("partition must define indices or inclusive_ranges")
    if len(indices) != len(set(indices)):
        raise ValueError("partition indices contain duplicates")
    return tuple(indices)


def partition_indices(protocol: Mapping[str, Any], partition_name: str) -> tuple[int, ...]:
    partitions = protocol.get("partitions", {})
    if partition_name not in partitions:
        raise KeyError(f"unknown protocol partition: {partition_name}")
    return _expand_indices(partitions[partition_name])


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected Kubric cache protocol schema")
    manifests = protocol.get("source_manifests")
    partitions = protocol.get("partitions")
    if not isinstance(manifests, dict) or not manifests:
        raise ValueError("source_manifests must be a non-empty object")
    if not isinstance(partitions, dict) or not partitions:
        raise ValueError("partitions must be a non-empty object")

    for alias, entry in manifests.items():
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {alias} must be an object")
        if not entry.get("path") or not entry.get("sha256"):
            raise ValueError(f"manifest entry {alias} must pin path and sha256")
        declared = int(entry.get("num_samples", -1))
        if declared <= 0:
            raise ValueError(f"manifest entry {alias} has invalid num_samples")

    memberships: dict[str, set[int]] = {}
    for name, spec in partitions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"partition {name} must be an object")
        source = str(spec.get("source", ""))
        if source not in manifests:
            raise ValueError(f"partition {name} references unknown source {source}")
        indices = set(_expand_indices(spec))
        declared_count = int(spec.get("expected_count", len(indices)))
        if declared_count != len(indices):
            raise ValueError(f"partition {name} expected_count mismatch")
        source_count = int(manifests[source]["num_samples"])
        if indices and (min(indices) < 0 or max(indices) >= source_count):
            raise ValueError(f"partition {name} index outside source manifest")
        prior = memberships.setdefault(source, set())
        overlap = prior & indices
        if overlap:
            raise ValueError(
                f"partitions sharing source {source} overlap at {sorted(overlap)[:5]}"
            )
        prior.update(indices)

    pilot_name = str(protocol.get("pilot_exclusion_partition", "pilot_excluded"))
    if pilot_name not in partitions:
        raise ValueError("pilot exclusion partition is missing")
    final_name = str(protocol.get("final_holdout_partition", "final_holdout"))
    if final_name not in partitions:
        raise ValueError("final holdout partition is missing")


def resolve_manifest_path(protocol: Mapping[str, Any], source_alias: str) -> Path:
    entry = protocol["source_manifests"][source_alias]
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        base = Path(str(protocol.get("_protocol_path", "."))).resolve().parent
        path = (base / path).resolve()
    return path


def verify_protocol_files(protocol: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for alias, entry in protocol["source_manifests"].items():
        path = resolve_manifest_path(protocol, alias)
        if not path.exists():
            raise FileNotFoundError(path)
        observed_hash = file_sha256(path)
        if observed_hash != str(entry["sha256"]):
            raise ValueError(f"manifest hash drift for {alias}: {observed_hash}")
        payload = json.loads(path.read_text())
        observed_count = int(payload.get("num_samples", 0) or 0)
        if observed_count != int(entry["num_samples"]):
            raise ValueError(f"manifest count drift for {alias}")
        verified[alias] = {
            "path": str(path),
            "sha256": observed_hash,
            "num_samples": observed_count,
            "split": payload.get("split"),
        }

    checkpoint = Path(str(protocol["backbone"]["checkpoint"]))
    if not checkpoint.is_absolute():
        roots = [
            Path(str(protocol.get("repository_root", "."))),
            Path("/gemini/code/FSPT"),
        ]
        candidates = [(root / checkpoint).resolve() for root in roots]
        checkpoint = next((item for item in candidates if item.exists()), candidates[0])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint_hash = file_sha256(checkpoint)
    if observed_checkpoint_hash != str(protocol["backbone"]["checkpoint_sha256"]):
        raise ValueError("backbone checkpoint hash drift")
    verified["backbone"] = {
        "path": str(checkpoint),
        "sha256": observed_checkpoint_hash,
    }
    return verified


def iter_manifest_samples(manifest_path: str | Path) -> Iterable[tuple[int, dict, dict]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    global_index = 0
    for shard_index, entry in enumerate(manifest["shards"]):
        relative = entry if isinstance(entry, str) else entry.get("path") or entry.get("file")
        shard_path = Path(relative)
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        with shard_path.open("rb") as handle:
            payload = pickle.load(handle)
        samples = list(payload.values()) if isinstance(payload, dict) else list(payload)
        for shard_local_index, sample in enumerate(samples):
            metadata = {
                "global_video_index": global_index,
                "shard_index": shard_index,
                "shard_local_index": shard_local_index,
                "shard_path": str(shard_path.resolve()),
            }
            yield global_index, sample, metadata
            global_index += 1


def sample_identity(sample: Mapping[str, Any]) -> SampleIdentity:
    if sample.get("source_tfrecord") is None or sample.get("source_record_index") is None:
        raise ValueError("sample lacks frozen source identity")
    return SampleIdentity(
        source_tfrecord=str(sample["source_tfrecord"]),
        source_record_index=int(sample["source_record_index"]),
        video_name=str(sample.get("video_name", "unknown")),
    )


def collect_protocol_identities(protocol: Mapping[str, Any]) -> dict[str, list[SampleIdentity]]:
    source_samples: dict[str, dict[int, SampleIdentity]] = {}
    for source_alias in protocol["source_manifests"]:
        manifest_path = resolve_manifest_path(protocol, source_alias)
        source_samples[source_alias] = {
            index: sample_identity(sample)
            for index, sample, _ in iter_manifest_samples(manifest_path)
        }

    output: dict[str, list[SampleIdentity]] = {}
    global_seen: dict[str, str] = {}
    for partition_name, spec in protocol["partitions"].items():
        source = str(spec["source"])
        identities = [source_samples[source][index] for index in _expand_indices(spec)]
        for identity in identities:
            key = identity.canonical
            if key in global_seen:
                raise ValueError(
                    f"identity overlap between {global_seen[key]} and {partition_name}: {key}"
                )
            global_seen[key] = partition_name
        output[partition_name] = identities
    return output


def identity_digest(identities: Sequence[SampleIdentity]) -> str:
    return canonical_json_sha256([item.canonical for item in identities])


def _normalised_tracks_from_xy(coords_xy_px: torch.Tensor, raster: int) -> torch.Tensor:
    scale = float(raster - 1)
    return torch.stack(
        [coords_xy_px[..., 1] / scale, coords_xy_px[..., 0] / scale], dim=-1
    )


def aggregate_partition_sidecars(
    sidecar_paths: Sequence[str | Path],
    *,
    metric_raster: int = 256,
) -> dict[str, Any]:
    if not sidecar_paths:
        raise ValueError("at least one sidecar is required")

    native_tracks = []
    oracle_tracks = []
    gt_tracks = []
    native_visibility = []
    gt_visibility = []
    queries = []
    per_video = []
    parity_all = True

    for path_like in sidecar_paths:
        path = Path(path_like)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("schema_version") not in SUPPORTED_VIDEO_SIDECAR_SCHEMA_VERSIONS:
            raise ValueError(f"unexpected sidecar schema: {path}")
        tensors = artifact["tensors"]
        candidate_zero = tensors["candidate_coords_xy_px"][..., 0, :]
        native_xy = tensors["native_coords_xy_px"]
        parity = bool(torch.equal(candidate_zero, native_xy)) and bool(
            tensors["candidate_valid_mask"][..., 0].all().item()
        )
        parity_all = parity_all and parity

        native_tracks.append(_normalised_tracks_from_xy(native_xy, metric_raster))
        oracle_tracks.append(
            _normalised_tracks_from_xy(tensors["oracle_coords_xy_px"], metric_raster)
        )
        gt_tracks.append(tensors["gt_tracks_yx"])
        native_visibility.append(tensors["native_visibility"])
        gt_visibility.append(~tensors["gt_occluded"])
        queries.append(tensors["query_points_tyx"])
        audit = artifact.get("audit", {})
        per_video.append(
            {
                "sidecar": str(path),
                "AJ_gain_points": float(audit.get("gain_points", {}).get("AJ", float("nan"))),
                "delta_gain_points": float(
                    audit.get("gain_points", {}).get("delta_average", float("nan"))
                ),
                "visible_evaluation_rows": int(audit.get("visible_evaluation_rows", 0)),
                "parity": parity,
            }
        )

    frame_counts = {int(value.shape[1]) for value in gt_tracks}
    if len(frame_counts) != 1:
        raise ValueError("sidecars have inconsistent frame counts")

    native_tracks_cat = torch.cat(native_tracks, dim=0)
    oracle_tracks_cat = torch.cat(oracle_tracks, dim=0)
    gt_tracks_cat = torch.cat(gt_tracks, dim=0)
    native_visibility_cat = torch.cat(native_visibility, dim=0)
    gt_visibility_cat = torch.cat(gt_visibility, dim=0)
    queries_cat = torch.cat(queries, dim=0)

    native_metrics = compute_tapvid_metrics(
        native_tracks_cat,
        gt_tracks_cat,
        native_visibility_cat,
        gt_visibility_cat,
        queries_cat,
        resolution=metric_raster,
        query_mode="first",
    )
    oracle_metrics = compute_tapvid_metrics(
        oracle_tracks_cat,
        gt_tracks_cat,
        native_visibility_cat,
        gt_visibility_cat,
        queries_cat,
        resolution=metric_raster,
        query_mode="first",
    )

    per_video_aj = torch.tensor([row["AJ_gain_points"] for row in per_video])
    per_video_delta = torch.tensor([row["delta_gain_points"] for row in per_video])
    threshold_gains = {
        str(threshold): 100.0
        * (
            float(oracle_metrics[f"pts_within_{threshold}"])
            - float(native_metrics[f"pts_within_{threshold}"])
        )
        for threshold in (1, 2, 4, 8, 16)
    }
    return {
        "videos": len(sidecar_paths),
        "points": int(gt_tracks_cat.shape[0]),
        "frames": int(gt_tracks_cat.shape[1]),
        "routing_disabled_native_parity_all": parity_all,
        "native_metrics": native_metrics,
        "coordinate_oracle_same_visibility_metrics": oracle_metrics,
        "gain_points": {
            "AJ": 100.0 * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0
            * (float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0 * (float(oracle_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "threshold_hit_gain_points": threshold_gains,
        "per_video": per_video,
        "per_video_AJ_gain_points": {
            "min": float(per_video_aj.min().item()),
            "median": float(per_video_aj.median().item()),
            "mean": float(per_video_aj.mean().item()),
            "positive_fraction": float((per_video_aj > 0.0).float().mean().item()),
            "gain_at_least_1_fraction": float((per_video_aj >= 1.0).float().mean().item()),
        },
        "per_video_delta_gain_points": {
            "min": float(per_video_delta.min().item()),
            "median": float(per_video_delta.median().item()),
            "mean": float(per_video_delta.mean().item()),
        },
    }


def evaluate_qualification_gates(
    summary: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    checks = {
        "complete_partition": bool(complete),
        "native_parity_all": bool(summary["routing_disabled_native_parity_all"]),
        "pooled_AJ_gain": float(summary["gain_points"]["AJ"])
        >= float(gates["pooled_AJ_gain_points_min"]),
        "pooled_delta_gain": float(summary["gain_points"]["delta_average"])
        >= float(gates["pooled_delta_gain_points_min"]),
        "median_video_AJ_gain": float(summary["per_video_AJ_gain_points"]["median"])
        >= float(gates["median_video_AJ_gain_points_min"]),
        "video_gain_at_least_1_fraction": float(
            summary["per_video_AJ_gain_points"]["gain_at_least_1_fraction"]
        )
        >= float(gates["video_AJ_gain_at_least_1_fraction_min"]),
        "all_thresholds_positive": all(
            float(value) > float(gates.get("each_threshold_gain_points_gt", 0.0))
            for value in summary["threshold_hit_gain_points"].values()
        ),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "pass": passed,
        "decision": (
            "ALLOW_FIT_MODEL_CACHE_EXPORT_AND_MUSR_TRAINING"
            if passed
            else "TRAINING_FORBIDDEN_UNTIL_FULL_QUALIFICATION_PASSES"
        ),
    }
