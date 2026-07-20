from pathlib import Path
import numpy as np
import torch

from scripts.run_routeD_temporal_identity_top1_gate3c1d_v0 import (
    _checkpoint_conditions,
    _confirmation_checks,
    _policy_metrics,
    _validate_implementation,
    select_checkpoint_threshold,
)


def _dataset():
    # Four rows, native slot zero and two non-native examples inside a nine-slot contract.
    distances = torch.tensor(
        [
            [30.0, 5.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            [25.0, 8.0, 22.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0],
            [20.0, 35.0, 6.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [18.0, 28.0, 26.0, 36.0, 46.0, 56.0, 66.0, 76.0, 86.0],
        ]
    )
    return {
        "distances": distances,
        "shortlist_indices": torch.arange(9).repeat(4, 1),
        "groups": torch.tensor([1, 1, 2, 2]),
        "point_indices": torch.arange(4),
        "video_names": ["a", "a", "b", "b"],
    }


def test_policy_abstains_to_native_below_threshold():
    probability = np.array(
        [
            [0.1, 0.9, 0.0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0],
            [0.1, 0.2, 0.7, 0, 0, 0, 0, 0, 0],
            [0.6, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    metrics, records, slot = _policy_metrics(
        dataset=_dataset(),
        probability=probability,
        threshold=0.75,
        bootstrap_seed=1,
        bootstrap_samples=50,
    )
    assert slot.tolist() == [1, 1, 0, 0]
    assert metrics["action_rows"] == 2
    assert metrics["action_precision_within_12px"] == 1.0
    assert records[2]["action"] is False
    assert records[2]["output_candidate_index"] == 0


def test_checkpoint_conditions_and_confirmation_require_safety():
    metrics = {
        "action_coverage": 0.3,
        "action_precision_within_12px": 0.7,
        "mean_commit_error_reduction_px": 3.0,
        "video_cluster_error_reduction_CI": {"lower": 1.5},
        "harmful_all_rows_gt_native_plus_4px": 0.01,
        "candidate_AUC": 0.8,
        "candidate_AP": 0.4,
    }
    checkpoint = {
        "action_coverage_min": 0.2,
        "action_precision_within_12px_min": 0.65,
        "mean_commit_error_reduction_px_min": 2.0,
        "video_cluster_error_reduction_CI_lower_px_min": 1.0,
        "harmful_all_rows_gt_native_plus_4px_max": 0.02,
    }
    assert all(_checkpoint_conditions(metrics, checkpoint).values())
    confirmation = {
        **checkpoint,
        "candidate_AUC_min": 0.75,
        "candidate_AP_min": 0.25,
    }
    checks = _confirmation_checks(metrics, confirmation, exact_replay=False)
    assert all(value for key, value in checks.items() if key != "exact_replay")
    assert not checks["exact_replay"]
    metrics["harmful_all_rows_gt_native_plus_4px"] = 0.03
    assert not _checkpoint_conditions(metrics, checkpoint)["harmful_all_rows"]


def test_threshold_selection_chooses_lowest_passing_grid():
    dataset = _dataset()
    probability = np.array(
        [
            [0.05, 0.70, 0.10, 0, 0, 0, 0, 0, 0],
            [0.05, 0.55, 0.10, 0, 0, 0, 0, 0, 0],
            [0.05, 0.10, 0.45, 0, 0, 0, 0, 0, 0],
            [0.60, 0.10, 0.10, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    config = {
        "threshold_selection": {
            "fixed_grid": [0.4, 0.5, 0.6],
            "gates": {
                "action_coverage_min": 0.5,
                "action_precision_within_12px_min": 1.0,
                "mean_commit_error_reduction_px_min": 5.0,
                "video_cluster_error_reduction_CI_lower_px_min": -100.0,
                "harmful_all_rows_gt_native_plus_4px_max": 0.0,
            },
        },
        "bootstrap": {"seed": 1, "samples": 50},
    }
    threshold, records = select_checkpoint_threshold(
        config=config, dataset=dataset, probability=probability
    )
    assert threshold == 0.4
    assert all(records[0]["checks"].values())


def test_frozen_implementation_hashes_validate():
    import yaml
    config = yaml.safe_load(
        Path("configs/routeD_temporal_identity_top1_gate3c1d_v0.yaml").read_text()
    )
    _validate_implementation(config)

