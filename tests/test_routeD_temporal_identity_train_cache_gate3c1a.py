import torch

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import (
    tensor_sha256,
)
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
    CACHE_SCHEMA,
    verify_train_cache_payload,
)


def _payload(rows: int = 2):
    tensors = {
        "point_indices": torch.arange(rows, dtype=torch.long),
        "query_frames": torch.zeros(rows, dtype=torch.long),
        "candidate_coordinates_xy": torch.zeros(rows, 129, 2),
        "candidate_scores": torch.zeros(rows, 129),
        "candidate_valid_mask": torch.ones(rows, 129, dtype=torch.bool),
        "temporal_features": torch.zeros(rows, 129, 16, 9),
        "static_features": torch.zeros(rows, 129, 14),
        "teacher_candidate_distance_px": torch.ones(rows, 129),
        "teacher_positive_within_12px": torch.ones(
            rows, 129, dtype=torch.bool
        ),
        "native_commit_error_px": torch.ones(rows),
        "native_future_mean_error_px": torch.full((rows,), 20.0),
    }
    hashes = {
        name: tensor_sha256(value.contiguous())
        for name, value in tensors.items()
    }
    return {
        "schema_version": CACHE_SCHEMA,
        "partition": "gradient_train",
        "source_index": 64,
        "tensors": tensors,
        "candidate_coordinate_hashes_before_teacher": [
            tensor_sha256(tensors["candidate_coordinates_xy"][row].contiguous())
            for row in range(rows)
        ],
        "tensor_hashes": hashes,
        "integrity": {
            "all_feature_tensors_finite": True,
            "candidate_hashes_frozen_before_teacher_labels": True,
            "teacher_available_to_candidate_coordinates_or_features": False,
            "future_frames_available_to_model_features": False,
            "checkpoint_selection_read": False,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        },
        "provenance": {"config_sha256": "config"},
    }


def test_train_cache_payload_validation_accepts_nonempty_and_empty():
    for rows in (2, 0):
        verify_train_cache_payload(
            _payload(rows),
            expected_source_index=64,
            expected_config_sha256="config",
        )


def test_train_cache_payload_rejects_candidate_hash_drift():
    payload = _payload()
    payload["candidate_coordinate_hashes_before_teacher"][0] = "bad"
    try:
        verify_train_cache_payload(
            payload,
            expected_source_index=64,
            expected_config_sha256="config",
        )
    except ValueError as error:
        assert "candidate pre-teacher hash drift" in str(error)
    else:
        raise AssertionError("candidate hash drift was not rejected")


def test_generic_cache_payload_accepts_explicit_checkpoint_read_state():
    from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
        verify_temporal_identity_cache_payload,
    )

    payload = _payload()
    payload["partition"] = "checkpoint_selection"
    payload["integrity"]["checkpoint_selection_read"] = True
    verify_temporal_identity_cache_payload(
        payload,
        expected_partition="checkpoint_selection",
        expected_source_index=64,
        expected_config_sha256="config",
        expected_read_state={
            "checkpoint_selection_read": True,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        },
    )
