import torch

from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_selector import (
    QueryClosureIdentitySelectorConfig,
    candidate_relative_zscore,
    retain_native_plus_nonnative,
    select_query_closure_identity_candidates,
)


def _features(rows: int = 2, candidates: int = 12):
    temporal = torch.zeros(
        rows, candidates, 16, len(TEMPORAL_FEATURE_CHANNELS)
    )
    static = torch.zeros(rows, candidates, len(STATIC_FEATURE_CHANNELS))
    valid = torch.ones(rows, candidates, dtype=torch.bool)
    query = torch.tensor([0, 3], dtype=torch.long)[:rows]
    return temporal, static, valid, query


def test_candidate_relative_zscore_ignores_invalid_values():
    value = torch.tensor([[1.0, 3.0, 1000.0]])
    valid = torch.tensor([[True, True, False]])
    result = candidate_relative_zscore(value, valid)
    torch.testing.assert_close(result[0, :2], torch.tensor([-1.0, 1.0]))
    assert result[0, 2].item() == 0.0


def test_frozen_score_matches_explicit_formula_and_query_frame():
    temporal, static, valid, query = _features(rows=1)
    q_index = TEMPORAL_FEATURE_CHANNELS.index("query_descriptor_cosine")
    cycle = STATIC_FEATURE_CHANNELS.index(
        "reverse_cycle_error_at_query_normalized"
    )
    mean = STATIC_FEATURE_CHANNELS.index("mean_query_descriptor_cosine")
    minimum = STATIC_FEATURE_CHANNELS.index("minimum_query_descriptor_cosine")
    static[0, :, cycle] = torch.arange(12, dtype=torch.float32)
    static[0, :, mean] = torch.arange(12, dtype=torch.float32).flip(0)
    static[0, :, minimum] = torch.arange(12, dtype=torch.float32).flip(0)
    temporal[0, :, 0, q_index] = torch.arange(12, dtype=torch.float32).flip(0)
    output = select_query_closure_identity_candidates(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    expected = (
        -2.0 * output["cycle_z"]
        + output["query_frame_identity_z"]
        + output["mean_identity_z"]
        + output["minimum_identity_z"]
    )
    torch.testing.assert_close(output["score"], expected)
    assert output["selected_indices"][0, 0].item() == 0
    assert output["selected_indices"][0, 1].item() == 1


def test_stable_ties_keep_lower_frozen_rank():
    score = torch.zeros(1, 12)
    valid = torch.ones(1, 12, dtype=torch.bool)
    selected = retain_native_plus_nonnative(score, valid, retained_nonnative=8)
    assert selected.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8]]


def test_selector_rejects_insufficient_nonnative_candidates():
    temporal, static, valid, query = _features(rows=1)
    valid[:, 5:] = False
    try:
        select_query_closure_identity_candidates(
            temporal_features=temporal,
            static_features=static,
            query_frames=query,
            valid_mask=valid,
            config=QueryClosureIdentitySelectorConfig(retained_nonnative=8),
        )
    except ValueError as error:
        assert "insufficient valid non-native" in str(error)
    else:
        raise AssertionError("insufficient candidate list was accepted")


def test_empty_rows_preserve_shapes():
    temporal = torch.empty(0, 129, 16, len(TEMPORAL_FEATURE_CHANNELS))
    static = torch.empty(0, 129, len(STATIC_FEATURE_CHANNELS))
    valid = torch.empty(0, 129, dtype=torch.bool)
    query = torch.empty(0, dtype=torch.long)
    output = select_query_closure_identity_candidates(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    assert output["score"].shape == (0, 129)
    assert output["selected_indices"].shape == (0, 9)


def test_formal_gate_waits_for_replay_before_decision():
    from scripts.eval_routeD_temporal_identity_selector_gate3c1b_v0 import (
        _gate_checks,
    )

    config = {
        "gates": {
            "checkpoint_selection": {
                "minimum_pooled_support_gain_12px": 0.04,
                "minimum_absolute_support_recall_12px": 0.56,
                "minimum_median_distance_improvement_px": 0.75,
                "minimum_nonnegative_video_fraction": 0.60,
                "decision_pass": "PASS",
                "decision_fail": "FAIL",
            }
        }
    }
    primary = {
        "recall_12px": 0.62,
        "recall_8px": 0.40,
        "median_minimum_distance_px": 9.0,
    }
    static = {
        "recall_12px": 0.52,
        "recall_8px": 0.31,
        "median_minimum_distance_px": 11.0,
    }
    videos = [{"support_gain_12px": 0.1} for _ in range(10)]
    ci = {"lower": 0.05, "mean": 0.1, "upper": 0.15}
    primary_only = _gate_checks(
        config,
        "checkpoint_selection",
        primary,
        static,
        videos,
        ci,
        exact_replay=False,
        reference_provided=False,
    )
    assert not primary_only["pass"]
    assert primary_only["decision"] == "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    replay = _gate_checks(
        config,
        "checkpoint_selection",
        primary,
        static,
        videos,
        ci,
        exact_replay=True,
        reference_provided=True,
    )
    assert replay["pass"]
    assert replay["decision"] == "PASS"
