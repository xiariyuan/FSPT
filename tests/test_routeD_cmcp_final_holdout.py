from __future__ import annotations

import copy
from pathlib import Path

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout import (
    evaluate_final_holdout_gate,
    load_final_holdout_config,
    partition_indices,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/routeD_cmcp_final_holdout_v0.yaml"


def _metrics():
    return {
        "videos": 16,
        "native_candidate_parity_all": True,
        "oracle_gain_points": {"AJ": 10.0},
        "selected_gain_points": {"AJ": 0.8, "delta_average": 1.0},
        "paired_video_selected_AJ_gain_CI": {"lower": 0.2},
        "severe_16px_rate": {"selected_delta": -0.01},
        "behavior": {"harmful_non_native_rate": 0.009},
        "per_video": [{"selected_AJ_gain_points": 0.1} for _ in range(16)],
    }


def test_protocol_membership_and_no_calibration():
    config = load_final_holdout_config(CONFIG)
    assert partition_indices(config, "implementation_smoke") == (0,)
    assert partition_indices(config, "final_holdout") == tuple(range(16, 32))
    assert config["calibration_policy"]["calibration_source_read"] is False
    assert config["frozen_model"]["state_writeback"] == "disabled"


def test_final_gate_passes_only_complete_exact_result():
    config = load_final_holdout_config(CONFIG)
    gate = evaluate_final_holdout_gate(_metrics(), config, exact_replay=True)
    assert gate["pass"]
    assert gate["decision"] == "AUTHORIZE_FROZEN_EXTERNAL_PROTOCOL_PREREGISTRATION"
    failed = copy.deepcopy(_metrics())
    failed["behavior"]["harmful_non_native_rate"] = 0.02
    gate = evaluate_final_holdout_gate(failed, config, exact_replay=True)
    assert not gate["pass"]
    assert gate["decision"] == "STOP_STRONG_BACKBONE_EXTERNAL_ROUTE_AND_RETAIN_INTERNAL_EVIDENCE_ONLY"


def test_positive_video_and_replay_are_mandatory():
    config = load_final_holdout_config(CONFIG)
    metrics = _metrics()
    for row in metrics["per_video"][:5]: row["selected_AJ_gain_points"] = -0.1
    assert not evaluate_final_holdout_gate(metrics, config, exact_replay=True)["pass"]
    assert not evaluate_final_holdout_gate(_metrics(), config, exact_replay=False)["pass"]
