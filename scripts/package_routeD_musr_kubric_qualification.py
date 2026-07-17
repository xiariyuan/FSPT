#!/usr/bin/env python3
"""Validate and package the completed MUSR Kubric candidate qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    file_sha256,
    load_protocol,
    partition_indices,
)

DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_INDEX = (
    REPO_ROOT
    / "outputs/routeD_musr_kubric_cache_20260717/candidate_qualification/cache_index.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json"
)


def combined_hash(hashes: list[str]) -> str:
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()


def build_summary(protocol: dict, index: dict, index_path: Path) -> dict:
    qualification = protocol["qualification_partition"]
    expected_indices = list(partition_indices(protocol, qualification))
    if index.get("protocol_sha256") != protocol["_protocol_sha256"]:
        raise ValueError("cache index protocol hash mismatch")
    if index.get("partition") != qualification:
        raise ValueError("cache index is not the qualification partition")
    if index.get("expected_source_indices") != expected_indices:
        raise ValueError("qualification membership drift")
    if index.get("completed_source_indices") != expected_indices:
        raise ValueError("qualification partition is incomplete")
    if not index.get("complete") or not index.get("training_authorized"):
        raise ValueError("qualification gate did not authorize training")
    gate = index.get("qualification_gate") or {}
    if not gate.get("pass") or not all(gate.get("checks", {}).values()):
        raise ValueError("one or more qualification checks failed")

    replay_indices = set(int(value) for value in protocol["deterministic_adapter_replay_indices"])
    observed_replays = set()
    sidecar_hashes = []
    sidecar_bytes = 0
    per_video = []
    for row in index["videos"]:
        source_index = int(row["source_index"])
        sidecar = Path(row["sidecar"])
        observed_hash = file_sha256(sidecar)
        if observed_hash != row["sidecar_sha256"]:
            raise ValueError(f"sidecar hash mismatch at source index {source_index}")
        if not row["routing_disabled_native_parity"]:
            raise ValueError(f"native parity failed at source index {source_index}")
        replay = row.get("deterministic_adapter_replay")
        if source_index in replay_indices:
            if not replay or not replay.get("exact"):
                raise ValueError(f"missing exact replay at source index {source_index}")
            observed_replays.add(source_index)
        sidecar_hashes.append(observed_hash)
        sidecar_bytes += sidecar.stat().st_size
        per_video.append(
            {
                "source_index": source_index,
                "video_name": row["sample_identity"]["video_name"],
                "source_tfrecord": row["sample_identity"]["source_tfrecord"],
                "source_record_index": row["sample_identity"]["source_record_index"],
                "AJ_gain_points": row["oracle_audit"]["gain_points"]["AJ"],
                "delta_average_gain_points": row["oracle_audit"]["gain_points"][
                    "delta_average"
                ],
                "native_parity": True,
                "deterministic_replay_exact": (
                    True if source_index in replay_indices else None
                ),
                "sidecar_sha256": observed_hash,
            }
        )
    if observed_replays != replay_indices:
        raise ValueError("deterministic replay coverage mismatch")

    audit = index["aggregate_oracle_audit"]
    return {
        "schema_version": "routeD_musr_kubric_qualification_summary_v1",
        "date": "2026-07-17",
        "status": "PASS_CANDIDATE_QUALIFICATION_AUTHORIZE_TRAINING_CACHE_EXPORT",
        "claim_boundary": (
            "GT-only coordinate-oracle headroom on frozen Kubric qualification; "
            "not a learned MUSR result and not external benchmark evidence."
        ),
        "protocol": {
            "path": str(Path(protocol["_protocol_path"]).relative_to(REPO_ROOT)),
            "sha256": protocol["_protocol_sha256"],
            "qualification_partition": qualification,
            "source_indices": expected_indices,
            "identity_sha256": index["selected_identity_sha256"],
            "pilot_excluded_source_index": 0,
            "final_holdout_source_indices": list(
                partition_indices(protocol, protocol["final_holdout_partition"])
            ),
        },
        "candidate_interface": protocol["candidate_generator"],
        "qualification": {
            "videos": index["completed_count"],
            "points": audit["points"],
            "frames": audit["frames"],
            "native_metrics": audit["native_metrics"],
            "coordinate_oracle_same_visibility_metrics": audit[
                "coordinate_oracle_same_visibility_metrics"
            ],
            "gain_points": audit["gain_points"],
            "threshold_hit_gain_points": audit["threshold_hit_gain_points"],
            "per_video_AJ_gain_points": audit["per_video_AJ_gain_points"],
            "per_video_delta_gain_points": audit["per_video_delta_gain_points"],
            "routing_disabled_native_parity_all": audit[
                "routing_disabled_native_parity_all"
            ],
            "gate": gate,
            "training_authorized": True,
        },
        "deterministic_replay": {
            "source_indices": sorted(replay_indices),
            "exact_all": True,
            "scope": "adapter export repeated from each video's frozen native backbone state",
        },
        "per_video": per_video,
        "integrity": {
            "cache_index_path": str(index_path.relative_to(REPO_ROOT)),
            "cache_index_file_sha256": file_sha256(index_path),
            "cache_index_payload_sha256": index["cache_index_payload_sha256"],
            "sidecar_count": len(sidecar_hashes),
            "sidecar_total_bytes": sidecar_bytes,
            "ordered_sidecar_hashes_sha256": combined_hash(sidecar_hashes),
            "candidate_generation_ground_truth_free": True,
            "tapvid_kinetics_read": False,
            "tapvid_davis_read": False,
            "final_holdout_read": False,
        },
        "next_gate": (
            "Export fit/model_validation/calibration caches under the frozen protocol, "
            "then train MUSR without reading final_holdout, DAVIS, or Kinetics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--cache-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    index_path = Path(args.cache_index).resolve()
    index = json.loads(index_path.read_text())
    summary = build_summary(protocol, index, index_path)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "videos": summary["qualification"]["videos"],
                "AJ_gain_points": summary["qualification"]["gain_points"]["AJ"],
                "training_authorized": summary["qualification"]["training_authorized"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
