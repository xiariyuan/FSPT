import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1_v1 import (
    RowPolicyFeatureConfig,
    build_row_policy_features,
    select_minimum_expected_distance,
)
from scripts.run_routeD_temporal_identity_top1_gate3c1d_v1 import _checks, _validate_runtime


def test_expected_distance_selection_is_stable_on_ties():
    value = np.array([[4.0, 2.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    assert select_minimum_expected_distance(value).tolist() == [1]


def test_row_policy_features_have_frozen_shape_and_no_labels():
    rows = 3
    features = torch.arange(rows * 9 * 102, dtype=torch.float32).reshape(rows, 9, 102) / 1000
    support = np.linspace(.05, .95, rows * 9).reshape(rows, 9)
    expected = np.linspace(1, 40, rows * 9).reshape(rows, 9)
    slot = select_minimum_expected_distance(expected)
    output = build_row_policy_features(
        candidate_features=features,
        support_probability=support,
        expected_distance_px=expected,
        selected_slot=slot,
    )
    assert output.shape == (rows, RowPolicyFeatureConfig().output_dim) == (3, 164)
    assert np.isfinite(output).all()


def test_gate_checks_require_separate_support_value_and_harm_outcomes():
    gates = {
        "candidate_AUC_min": .78,
        "candidate_AP_min": .30,
        "raw_top1_within_12px_min": .32,
        "action_coverage_min": .20,
        "action_precision_within_12px_min": .65,
        "mean_commit_error_reduction_px_min": 2.0,
        "video_cluster_error_reduction_CI_lower_px_min": 1.0,
        "harmful_all_rows_gt_native_plus_4px_max": .02,
        "harmful_action_fraction_max": .08,
        "nonnegative_video_fraction_min": .70,
    }
    metrics = {
        "candidate_AUC": .84,
        "candidate_AP": .45,
        "raw_top1_within_12px": .36,
        "action_coverage": .30,
        "action_precision_within_12px": .66,
        "mean_commit_error_reduction_px": 4.0,
        "video_cluster_error_reduction_CI": {"lower": 3.0},
        "harmful_all_rows_gt_native_plus_4px": .008,
        "harmful_action_fraction": .03,
        "nonnegative_video_fraction": .98,
    }
    assert all(_checks(metrics, gates).values())
    metrics["action_precision_within_12px"] = .64
    assert not _checks(metrics, gates)["action_precision"]


def test_preregistered_implementation_and_cache_config_hashes_are_exact():
    config = yaml.safe_load(Path("configs/routeD_temporal_identity_top1_gate3c1d_v1.yaml").read_text())
    assert config["development_expected_rows"] == 4901
    assert config["oof_contract"]["candidate_predictions_for_row_training"] == "strict_group_out_of_fold"
    for authority in config["implementation"].values():
        actual = hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest()
        assert actual == authority["sha256"]
    for authority in config["renewed_cache_indices"].values():
        actual = hashlib.sha256(Path(authority["config"]).read_bytes()).hexdigest()
        assert actual == authority["config_sha256"]


def test_runtime_contract_is_exact():
    config = yaml.safe_load(Path("configs/routeD_temporal_identity_top1_gate3c1d_v1.yaml").read_text())
    _validate_runtime(config)
    config["runtime_contract"]["scikit_learn"] = "0.0.0"
    try:
        _validate_runtime(config)
    except ValueError as error:
        assert "runtime version drift" in str(error)
    else:
        raise AssertionError("runtime drift was not rejected")
