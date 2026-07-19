from __future__ import annotations

import copy

import pytest

from scripts.package_routeD_cmcp_davis_external_result import (
    verify_davis_eval_replay,
)


def _row():
    return {
        "schema_version": "x",
        "seed": 17,
        "config_sha256": "c",
        "cache_index_sha256": "i",
        "cache_index_payload_sha256": "p",
        "dataset_sha256": "d",
        "video_order_sha256": "v",
        "checkpoint_sha256": "k",
        "combined_model_state_sha256": "m",
        "adapter_state_sha256": "a",
        "cmcp_state_sha256": "g",
        "comparator_state_sha256": "q",
        "normalization": {"x": 1},
        "metrics": {"AJ": 1},
        "model_selection_on_DAVIS": False,
        "threshold_selection_on_DAVIS": False,
        "calibration_read": False,
        "kinetics_read_or_rerun": False,
        "claim_boundary": {"x": 1},
    }


def test_davis_eval_replay_exact_except_uncompared_paths():
    primary = _row()
    replay = copy.deepcopy(primary)
    primary["output"] = "a"
    replay["output"] = "b"
    assert all(verify_davis_eval_replay(primary, replay).values())


def test_davis_eval_replay_rejects_metric_drift():
    primary = _row()
    replay = copy.deepcopy(primary)
    replay["metrics"]["AJ"] = 2
    with pytest.raises(RuntimeError, match="replay mismatch"):
        verify_davis_eval_replay(primary, replay)
