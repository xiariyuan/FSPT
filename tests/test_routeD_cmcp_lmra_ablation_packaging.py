from __future__ import annotations

import copy

import pytest

from scripts.package_routeD_cmcp_lmra_ablation_result import (
    canonical_metrics,
    paired_delta_summary,
    verify_replay,
)


def test_canonical_metrics_ignores_only_checkpoint_path():
    payload = {"checkpoint_path": "/a/b.pt", "history": [1], "gate": {"pass": True}}
    other = copy.deepcopy(payload)
    other["checkpoint_path"] = "/other/b.pt"
    assert canonical_metrics(payload) == canonical_metrics(other)
    other["history"] = [2]
    assert canonical_metrics(payload) != canonical_metrics(other)


def test_verify_replay_requires_byte_identical_checkpoint_and_full_payload():
    base = {
        "checkpoint_path": "/a.pt",
        "checkpoint_sha256": "same",
        "combined_model_state_sha256": "state",
        "adapter_state_sha256": "a",
        "cmcp_state_sha256": "c",
        "comparator_state_sha256": "p",
        "best_epoch": 2,
        "history": [1],
        "final_validation": {"x": 1},
        "gate": {"pass": True},
    }
    replay = copy.deepcopy(base)
    replay["checkpoint_path"] = "/replay.pt"
    assert verify_replay(base, replay, label="X")["exact"]
    replay["checkpoint_sha256"] = "different"
    with pytest.raises(RuntimeError, match="byte-identical"):
        verify_replay(base, replay, label="X")


def test_paired_delta_summary_is_paired_and_deterministic():
    left = {(0, "a"): 0.5, (1, "b"): 0.7, (2, "c"): 0.9}
    right = {(0, "a"): 0.1, (1, "b"): 0.2, (2, "c"): 0.3}
    one = paired_delta_summary(left, right, seed=17, samples=1000)
    two = paired_delta_summary(left, right, seed=17, samples=1000)
    assert one == two
    assert one["positive_videos"] == 3
    assert one["left_minus_right_mean_AJ_points"] == pytest.approx(0.5)
    assert one["lower"] > 0


def test_paired_delta_summary_rejects_identity_mismatch():
    with pytest.raises(RuntimeError, match="identities"):
        paired_delta_summary({(0, "a"): 1.0}, {(1, "b"): 1.0}, seed=17)
