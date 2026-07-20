import torch

from scripts.audit_routeD_temporal_identity_future_rollout_gate3c1c_v0 import (
    gate_checks,
    teacher_nearest_in_frozen_shortlist,
)


def test_teacher_nearest_is_restricted_to_frozen_shortlist():
    selected = torch.tensor([[0, 2, 4], [0, 1, 3]])
    coordinates = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        ]
    )
    # Candidate 1 is globally best in row 0 but is not in the frozen shortlist.
    distance = torch.tensor([[8.0, 0.1, 3.0, 2.0, 4.0], [9.0, 5.0, 0.1, 2.0, 1.0]])
    result = teacher_nearest_in_frozen_shortlist(
        selected_indices=selected,
        candidate_coordinates_xy=coordinates,
        teacher_candidate_distance_px=distance,
    )
    assert torch.equal(result["selected_candidate_index"], torch.tensor([2, 3]))
    assert torch.allclose(
        result["selected_teacher_distance_px"], torch.tensor([3.0, 2.0])
    )


def test_teacher_nearest_rejects_invalid_distance_in_shortlist():
    selected = torch.tensor([[0, 1]])
    coordinates = torch.zeros(1, 2, 2)
    distance = torch.tensor([[1.0, -1.0]])
    try:
        teacher_nearest_in_frozen_shortlist(
            selected_indices=selected,
            candidate_coordinates_xy=coordinates,
            teacher_candidate_distance_px=distance,
        )
    except ValueError as error:
        assert "invalid teacher distance" in str(error)
    else:
        raise AssertionError("invalid shortlist teacher distance was accepted")


def test_gate_checks_require_exact_replay_and_memory_increment():
    gates = {
        "shortlist_recall_within_12px_min": 0.60,
        "shortlist_median_commit_error_px_max": 10.0,
        "full_state_mean_error_reduction_vs_native_px_min": 8.0,
        "full_state_error_reduction_CI_lower_px_min": 2.0,
        "full_state_threshold_utility_gain_vs_native_min": 0.08,
        "full_state_utility_gain_CI_lower_min": 0.02,
        "full_state_positive_point_fraction_min": 0.65,
        "full_state_severe_16px_rate_reduction_min": 0.15,
        "memory_incremental_error_reduction_vs_coordinate_only_px_min": 2.0,
        "memory_incremental_utility_gain_vs_coordinate_only_min": 0.02,
        "memory_better_video_fraction_min": 0.60,
        "native_replay_max_abs_px": 0.0001,
    }
    commit = {"recall_within_12px": 0.64, "median_error_px": 9.0}
    full = {
        "mean_error_reduction_px": 10.0,
        "error_reduction_video_cluster_CI": {"lower": 3.0},
        "threshold_utility_gain": 0.10,
        "utility_gain_video_cluster_CI": {"lower": 0.03},
        "positive_point_fraction": 0.70,
        "severe_16px_rate_reduction": 0.20,
    }
    memory = {"mean_error_reduction_px": 3.0, "threshold_utility_gain": 0.03}
    checks = gate_checks(
        commit=commit,
        full_vs_native=full,
        memory_vs_coordinate=memory,
        memory_better_video_fraction=0.7,
        native_replay_max_abs_px=0.0,
        exact_replay=False,
        gates=gates,
    )
    assert all(value for key, value in checks.items() if key != "exact_replay")
    assert not checks["exact_replay"]

    memory["mean_error_reduction_px"] = 1.0
    checks = gate_checks(
        commit=commit,
        full_vs_native=full,
        memory_vs_coordinate=memory,
        memory_better_video_fraction=0.7,
        native_replay_max_abs_px=0.0,
        exact_replay=True,
        gates=gates,
    )
    assert not checks["memory_incremental_error"]


def test_expected_read_state_supports_original_model_validation():
    from scripts.audit_routeD_temporal_identity_future_rollout_gate3c1c_v0 import (
        _expected_read_state,
    )

    config = {
        "partition": {
            "expected_read_state": {
                "checkpoint_selection_read": True,
                "fit_only_internal_audit_read": True,
                "original_model_validation_read": True,
                "external_read": False,
            }
        }
    }
    assert _expected_read_state(config)["original_model_validation_read"]
    assert not _expected_read_state(config)["external_read"]


def test_candidate_index_can_be_verified_by_preregistered_cache_config(tmp_path):
    import hashlib
    import json

    from scripts.audit_routeD_temporal_identity_future_rollout_gate3c1c_v0 import (
        _load_candidate_index,
    )

    cache_config = tmp_path / "cache.yaml"
    cache_config.write_text("schema_version: test\n")
    config_sha = hashlib.sha256(cache_config.read_bytes()).hexdigest()
    index_path = tmp_path / "cache_index.json"
    index_path.write_text(
        json.dumps(
            {
                "partition": "original_model_validation",
                "config_sha256": config_sha,
                "completed_source_indices": list(range(48, 64)),
                "videos": 16,
                "failure_rows": 123,
                "read_state": {
                    "checkpoint_selection_read": True,
                    "fit_only_internal_audit_read": True,
                    "original_model_validation_read": True,
                    "external_read": False,
                },
            }
        )
    )
    config = {
        "partition": {
            "candidate_cache_index": str(index_path),
            "candidate_cache_config": str(cache_config),
            "candidate_cache_config_sha256": config_sha,
            "name": "original_model_validation",
            "source_indices": [48, 63],
            "expected_videos": 16,
            "expected_read_state": {
                "checkpoint_selection_read": True,
                "fit_only_internal_audit_read": True,
                "original_model_validation_read": True,
                "external_read": False,
            },
        }
    }
    loaded = _load_candidate_index(config)
    assert loaded["failure_rows"] == 123
    assert loaded["_index_file_sha256"] == hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()
