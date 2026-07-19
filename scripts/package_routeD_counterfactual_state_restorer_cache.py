#!/usr/bin/env python3
"""Package complete train/fit-internal-validation CSRR teacher caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    load_complete_csrr_cache_index,
    verify_csrr_cache_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _rows_by_source(index: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["source_index"]): row for row in index["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--validation-index", required=True)
    parser.add_argument("--smoke-primary-index", required=True)
    parser.add_argument("--smoke-replay-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    config_sha256 = file_sha256(config_path)
    train = load_complete_csrr_cache_index(args.train_index, expected_partition="train")
    validation = load_complete_csrr_cache_index(
        args.validation_index, expected_partition="fit_internal_validation"
    )
    smoke_primary = load_complete_csrr_cache_index(
        args.smoke_primary_index, expected_partition="smoke"
    )
    smoke_replay = load_complete_csrr_cache_index(
        args.smoke_replay_index, expected_partition="smoke"
    )
    expected_train = list(range(8, 32))
    expected_validation = list(range(32, 48))
    if train["completed_source_indices"] != expected_train:
        raise ValueError("complete train-cache membership mismatch")
    if validation["completed_source_indices"] != expected_validation:
        raise ValueError("complete validation-cache membership mismatch")
    if any(index >= 48 for index in train["completed_source_indices"] + validation["completed_source_indices"]):
        raise ValueError("locked model-validation source was read")

    verified_rows = []
    for partition, index in (("train", train), ("fit_internal_validation", validation)):
        for row in index["rows"]:
            artifact = verify_csrr_cache_artifact(
                row["sidecar"],
                expected_partition=partition,
                expected_source_index=int(row["source_index"]),
                expected_config_sha256=config_sha256,
            )
            verified_rows.append(
                {
                    "partition": partition,
                    "source_index": int(row["source_index"]),
                    "sidecar_sha256": row["sidecar_sha256"],
                    "tensor_hash_digest": artifact["tensor_hash_digest"],
                    "failure_count": int(artifact["selection"]["failure_count"]),
                    "clean_count": int(artifact["selection"]["clean_count"]),
                }
            )

    train_rows = _rows_by_source(train)
    validation_rows = _rows_by_source(validation)
    smoke_primary_rows = _rows_by_source(smoke_primary)
    smoke_replay_rows = _rows_by_source(smoke_replay)
    anchor_checks = []
    for source_index in (8, 31, 32, 47):
        formal = train_rows[source_index] if source_index < 32 else validation_rows[source_index]
        primary = smoke_primary_rows[source_index]
        replay = smoke_replay_rows[source_index]
        digest_exact = (
            formal["tensor_hash_digest"]
            == primary["tensor_hash_digest"]
            == replay["tensor_hash_digest"]
        )
        anchor_checks.append(
            {
                "source_index": source_index,
                "tensor_hash_digest_exact": digest_exact,
                "formal_tensor_hash_digest": formal["tensor_hash_digest"],
                "smoke_primary_tensor_hash_digest": primary["tensor_hash_digest"],
                "smoke_replay_tensor_hash_digest": replay["tensor_hash_digest"],
            }
        )

    gates = config["scientific_gates"]
    validation_failure_videos = sum(row["failure_count"] > 0 for row in validation["rows"])
    validation_clean_videos = sum(row["clean_count"] > 0 for row in validation["rows"])
    checks = {
        "train_complete": bool(train["complete"]),
        "validation_complete": bool(validation["complete"]),
        "all_40_sidecars_verified": len(verified_rows) == 40,
        "memberships_exact_and_disjoint": train["completed_source_indices"] == expected_train
        and validation["completed_source_indices"] == expected_validation,
        "all_anchor_tensor_digests_exact": all(row["tensor_hash_digest_exact"] for row in anchor_checks),
        "train_quantization_max_error": float(train["maximum_quantization_absolute_error"])
        <= float(config["cache"]["quantization_gates"]["max_absolute_error"]),
        "train_quantization_min_cosine": float(train["minimum_quantization_cosine_similarity"])
        >= float(config["cache"]["quantization_gates"]["minimum_cosine_similarity"]),
        "validation_quantization_max_error": float(validation["maximum_quantization_absolute_error"])
        <= float(config["cache"]["quantization_gates"]["max_absolute_error"]),
        "validation_quantization_min_cosine": float(validation["minimum_quantization_cosine_similarity"])
        >= float(config["cache"]["quantization_gates"]["minimum_cosine_similarity"]),
        "validation_failure_points_min": int(validation["total_failure_rows"])
        >= int(gates["validation_failure_points_min"]),
        "validation_failure_videos_min": validation_failure_videos
        >= int(gates["validation_failure_videos_min"]),
        "validation_clean_points_min": int(validation["total_clean_rows"])
        >= int(gates["validation_clean_points_min"]),
        "validation_clean_videos_min": validation_clean_videos
        >= int(gates["validation_clean_videos_min"]),
        "model_validation_read": False,
        "calibration_read": False,
        "final_holdout_read": False,
        "tapvid_davis_read": False,
        "tapvid_kinetics_read": False,
        "official_kinetics_1144_rerun": False,
    }
    negative_flags = {
        "model_validation_read",
        "calibration_read",
        "final_holdout_read",
        "tapvid_davis_read",
        "tapvid_kinetics_read",
        "official_kinetics_1144_rerun",
    }
    pass_value = all(not value if key in negative_flags else bool(value) for key, value in checks.items())
    summary = {
        "schema_version": "routeD_csrr_teacher_cache_complete_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass" if pass_value else "completed_fail",
        "formal_decision": "ALLOW_GATE2_RESTORER_TRAINING" if pass_value else "STOP_GATE2_BEFORE_RESTORER_TRAINING",
        "pass": pass_value,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "train": {
            "index": train["_index_path"],
            "index_sha256": train["_index_sha256"],
            "completed_count": train["completed_count"],
            "source_indices": train["completed_source_indices"],
            "failure_rows": train["total_failure_rows"],
            "clean_rows": train["total_clean_rows"],
            "maximum_quantization_absolute_error": train["maximum_quantization_absolute_error"],
            "minimum_quantization_cosine_similarity": train["minimum_quantization_cosine_similarity"],
            "combined_sidecar_sha256": train["combined_sidecar_sha256"],
            "combined_tensor_hash_digest": train["combined_tensor_hash_digest"],
        },
        "fit_internal_validation": {
            "index": validation["_index_path"],
            "index_sha256": validation["_index_sha256"],
            "completed_count": validation["completed_count"],
            "source_indices": validation["completed_source_indices"],
            "failure_rows": validation["total_failure_rows"],
            "clean_rows": validation["total_clean_rows"],
            "failure_support_videos": validation_failure_videos,
            "clean_support_videos": validation_clean_videos,
            "maximum_quantization_absolute_error": validation["maximum_quantization_absolute_error"],
            "minimum_quantization_cosine_similarity": validation["minimum_quantization_cosine_similarity"],
            "combined_sidecar_sha256": validation["combined_sidecar_sha256"],
            "combined_tensor_hash_digest": validation["combined_tensor_hash_digest"],
        },
        "fixed_anchor_replay": anchor_checks,
        "checks": checks,
        "verified_rows": verified_rows,
        "claim_boundary": "Complete fit-only teacher caches. No learned checkpoint, model-validation, holdout, or external claim.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": summary["formal_decision"],
                "pass": pass_value,
                "verified_sidecars": len(verified_rows),
                "validation_failure_rows": validation["total_failure_rows"],
                "validation_clean_rows": validation["total_clean_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
