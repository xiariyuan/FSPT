#!/usr/bin/env python3
"""Package and verify the one-video CMCP interface audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCP_SCHEMA_VERSION,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--primary",
        default=str(REPO_ROOT / "outputs/routeD_cmcp_20260717/interface_smoke.json"),
    )
    ap.add_argument(
        "--replay",
        default=str(
            REPO_ROOT / "outputs/routeD_cmcp_20260717/interface_smoke_replay.json"
        ),
    )
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/routeD_cmcp_cotracker3_stage0.yaml"),
    )
    ap.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_CMCP_INTERFACE_SMOKE_SUMMARY_2026-07-17.json"
        ),
    )
    args = ap.parse_args()
    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    config_path = Path(args.config).resolve()
    primary = _read(primary_path)
    replay = _read(replay_path)

    exact_fields = (
        "proposal_schema_version",
        "source_partition",
        "source_index",
        "video_name",
        "points",
        "frames",
        "feature_map_shape",
        "config_sha256",
        "protocol_sha256",
        "base_cache_index_sha256",
        "base_sidecar_sha256",
        "checkpoint_sha256",
        "native_state_hashes",
        "trainable_parameters",
        "candidate_zero_native_parity",
        "zero_step_selected_native_all_active",
        "deterministic_first_chunk_replay",
        "tensor_hashes",
        "integrity",
    )
    checks = {key: primary[key] == replay[key] for key in exact_fields}
    if not all(checks.values()):
        raise RuntimeError(f"CMCP interface replay mismatch: {checks}")
    if primary["proposal_schema_version"] != CMCP_SCHEMA_VERSION:
        raise RuntimeError("CMCP schema mismatch")
    if primary["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("CMCP config hash mismatch")
    if not primary["candidate_zero_native_parity"]:
        raise RuntimeError("CMCP candidate-zero parity failed")
    if not primary["zero_step_selected_native_all_active"]:
        raise RuntimeError("CMCP native-safe initialization failed")
    if not primary["deterministic_first_chunk_replay"]["exact"]:
        raise RuntimeError("CMCP first-chunk replay failed")
    if any(bool(value) for value in primary["integrity"].values() if value):
        # The only true integrity field is candidate_generation_ground_truth_free.
        pass
    locked = (
        "calibration_read",
        "final_holdout_read",
        "tapvid_davis_read",
        "tapvid_kinetics_read",
    )
    if any(primary["integrity"][key] for key in locked):
        raise RuntimeError("locked data were read")
    if not primary["integrity"]["candidate_generation_ground_truth_free"]:
        raise RuntimeError("candidate generation used GT")

    summary = {
        "schema_version": "routeD_cmcp_interface_smoke_summary_v1",
        "date": "2026-07-17",
        "status": "completed_pass",
        "decision": "ALLOW_CMCP_FEATURE_MAP_CACHE_AND_FIT_ONLY_PROPOSAL_TRAINING",
        "claim_boundary": (
            "This is a zero-step interface audit on one authorized fit video; "
            "it is not a learned proposal result."
        ),
        "model": {
            "proposal_schema_version": primary["proposal_schema_version"],
            "trainable_parameters": primary["trainable_parameters"],
            "config_path": str(config_path),
            "config_sha256": primary["config_sha256"],
            "memory_sources": [
                "immutable_query_anchor",
                "previous_native_feature",
                "causal_fixed_alpha_ema",
            ],
            "candidate_count": 6,
            "native_candidate_index": 0,
        },
        "sample": {
            "partition": primary["source_partition"],
            "source_index": primary["source_index"],
            "video_name": primary["video_name"],
            "points": primary["points"],
            "frames": primary["frames"],
            "feature_map_shape": primary["feature_map_shape"],
        },
        "audit": {
            "frozen_native_state_exact": True,
            "candidate_zero_native_parity": True,
            "zero_step_selected_native_all_active": True,
            "first_chunk_replay_exact": True,
            "independent_full_video_tensor_replay_exact": True,
            "tensor_hashes": primary["tensor_hashes"],
            "native_state_hashes": primary["native_state_hashes"],
            "correlation_sha256": primary["deterministic_first_chunk_replay"][
                "correlation_sha256"
            ],
            "proposal_score_sha256": primary[
                "deterministic_first_chunk_replay"
            ]["proposal_score_sha256"],
            "replay_checks": checks,
        },
        "artifacts": {
            "primary_report_sha256": file_sha256(primary_path),
            "primary_sidecar_sha256": primary["sidecar_sha256"],
            "replay_report_sha256": file_sha256(replay_path),
            "replay_sidecar_sha256": replay["sidecar_sha256"],
            "sidecar_file_hash_expected_to_differ_by_embedded_path": True,
        },
        "next_gate": {
            "step": "export frozen float16 feature-map caches for fit and model_validation",
            "training": "fit-only dense CMCP proposal training",
            "model_validation": "proposal-only checkpoint selection and gates",
            "selector_training_authorized": False,
            "state_write_training_authorized": False,
        },
        "integrity": primary["integrity"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "trainable_parameters": primary["trainable_parameters"],
                "tensor_replay_exact": True,
                "decision": summary["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
