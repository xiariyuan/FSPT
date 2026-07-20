import hashlib
import json
from pathlib import Path

import joblib
import pytest
import torch
import yaml

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256
from scripts.audit_routeD_temporal_identity_causal_future_rollout_gate3c1e_v0 import (
    _causal_records,
    _feature_config,
    _parent_causal_record,
    _result_payload_sha256,
    freeze_deployable_policy_action,
    gate_checks,
)
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
    verify_temporal_identity_cache_payload,
)

CONFIG_PATH = Path("configs/routeD_temporal_identity_causal_future_rollout_gate3c1e_v0.yaml")


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_preregistered_hashes_parent_and_locked_data_are_exact():
    config = _config()
    assert all(value is False for value in config["locked_data"].values())
    for authority in config["implementation"].values():
        actual = hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest()
        assert actual == authority["sha256"]
    parent = config["authorized_parent"]
    assert hashlib.sha256(Path(parent["result"]).read_bytes()).hexdigest() == parent["result_sha256"]
    assert (
        hashlib.sha256(Path(parent["model_validation_replay"]).read_bytes()).hexdigest()
        == parent["model_validation_replay_sha256"]
    )
    replay = json.loads(Path(parent["model_validation_replay"]).read_text())
    assert replay["result_payload_sha256"] == _result_payload_sha256(replay)
    assert replay["exact_replay"] is True
    assert replay["gate"]["pass"] is True
    assert replay["gate"]["decision"] == parent["required_decision"]
    assert hashlib.sha256(Path(parent["bundle_path"]).read_bytes()).hexdigest() == parent["bundle_sha256"]


def test_complete_causal_preflight_matches_gate3c1d_parent_records():
    config = _config()
    parent = config["authorized_parent"]
    replay = json.loads(Path(parent["model_validation_replay"]).read_text())
    parent_records = {
        (int(record["source_index"]), int(record["point_index"])): _parent_causal_record(record)
        for record in replay["records"]
    }
    bundle = joblib.load(parent["bundle_path"])
    index = json.loads(Path(config["partition"]["candidate_cache_index"]).read_text())
    feature_config = _feature_config(config)
    actual_records = []
    for index_row in index["rows"]:
        source_index = int(index_row["source_index"])
        payload = torch.load(index_row["sidecar"], map_location="cpu", weights_only=False)
        verify_temporal_identity_cache_payload(
            payload,
            expected_partition=config["partition"]["name"],
            expected_source_index=source_index,
            expected_config_sha256=index["config_sha256"],
            expected_read_state=config["partition"]["expected_read_state"],
        )
        tensors = payload["tensors"]
        point_indices = tensors["point_indices"].long()
        if int(point_indices.numel()) == 0:
            continue
        frozen = freeze_deployable_policy_action(
            tensors=tensors,
            bundle=bundle,
            policy=parent["frozen_policy"],
            feature_config=feature_config,
        )
        local = _causal_records(
            source_index=source_index,
            video_name=str(payload["video_name"]),
            point_indices=point_indices,
            action=frozen,
        )
        expected = [parent_records[(source_index, int(point))] for point in point_indices]
        assert canonical_json_sha256(local) == canonical_json_sha256(expected)
        actual_records.extend(local)
    expected_records = [_parent_causal_record(record) for record in replay["records"]]
    assert len(actual_records) == config["partition"]["expected_failure_rows"] == 1379
    assert canonical_json_sha256(actual_records) == canonical_json_sha256(expected_records)


def test_gate_checks_require_future_and_memory_improvement():
    gates = _config()["pass_gates"]
    commit = {"mean_error_reduction_px": 3.4}
    full = {
        "mean_error_reduction_px": 1.5,
        "threshold_utility_gain": 0.03,
        "severe_16px_rate_reduction": 0.05,
        "harmful_gt_4px_fraction": 0.01,
        "error_reduction_video_cluster_CI": {"lower": 0.8},
        "utility_gain_video_cluster_CI": {"lower": 0.01},
    }
    full_action = {
        "mean_error_reduction_px": 5.5,
        "positive_point_fraction": 0.65,
        "harmful_gt_4px_fraction": 0.05,
    }
    memory = {
        "mean_error_reduction_px": 1.0,
        "threshold_utility_gain": 0.01,
        "error_reduction_video_cluster_CI": {"lower": 0.2},
    }
    checks = gate_checks(
        selector_consistency_exact=True,
        commit=commit,
        full_vs_native=full,
        full_vs_native_action=full_action,
        memory_vs_coordinate=memory,
        video_nonnegative_fraction=0.85,
        memory_better_video_fraction=0.70,
        native_replay_max_abs_px=0.0,
        exact_replay=True,
        gates=gates,
    )
    assert all(checks.values())
    memory["mean_error_reduction_px"] = 0.4
    checks = gate_checks(
        selector_consistency_exact=True,
        commit=commit,
        full_vs_native=full,
        full_vs_native_action=full_action,
        memory_vs_coordinate=memory,
        video_nonnegative_fraction=0.85,
        memory_better_video_fraction=0.70,
        native_replay_max_abs_px=0.0,
        exact_replay=True,
        gates=gates,
    )
    assert checks["memory_incremental_error"] is False


def test_gate_checks_reject_future_harm_even_when_mean_gain_is_positive():
    gates = _config()["pass_gates"]
    full = {
        "mean_error_reduction_px": 2.0,
        "threshold_utility_gain": 0.03,
        "severe_16px_rate_reduction": 0.05,
        "harmful_gt_4px_fraction": 0.031,
        "error_reduction_video_cluster_CI": {"lower": 0.8},
        "utility_gain_video_cluster_CI": {"lower": 0.01},
    }
    full_action = {
        "mean_error_reduction_px": 6.0,
        "positive_point_fraction": 0.70,
        "harmful_gt_4px_fraction": 0.11,
    }
    memory = {
        "mean_error_reduction_px": 1.0,
        "threshold_utility_gain": 0.01,
        "error_reduction_video_cluster_CI": {"lower": 0.2},
    }
    checks = gate_checks(
        selector_consistency_exact=True,
        commit={"mean_error_reduction_px": 3.4},
        full_vs_native=full,
        full_vs_native_action=full_action,
        memory_vs_coordinate=memory,
        video_nonnegative_fraction=0.85,
        memory_better_video_fraction=0.70,
        native_replay_max_abs_px=0.0,
        exact_replay=True,
        gates=gates,
    )
    assert checks["full_future_harm_all"] is False
    assert checks["full_future_harm_action"] is False
