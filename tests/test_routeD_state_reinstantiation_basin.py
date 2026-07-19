import pytest
import torch

from projects.mmp_tracker.mmp_tracker.routeD_state_reinstantiation_basin import (
    aggregate_basin_rows,
    build_offset_hypotheses,
    evaluate_radius_gates,
    offset_and_clip_coordinates,
    video_cluster_bootstrap_ci,
)


def test_offset_schedule_has_one_center_and_normalized_diagonal():
    rows = build_offset_hypotheses(
        [0, 2, 4], {"east": [1, 0], "southeast": [1, 1]}
    )
    assert [row["hypothesis_id"] for row in rows] == [
        "r0_center",
        "r2_east",
        "r2_southeast",
        "r4_east",
        "r4_southeast",
    ]
    diagonal = torch.tensor(rows[2]["offset_xy_px"])
    assert torch.linalg.vector_norm(diagonal).item() == pytest.approx(2.0)


def test_offset_clipping_reports_realized_error_and_boundary_mask():
    teacher = torch.tensor([[254.0, 100.0], [100.0, 100.0]])
    candidate, realized, clipped = offset_and_clip_coordinates(
        teacher, [4.0, 0.0]
    )
    assert candidate.tolist() == [[255.0, 100.0], [104.0, 100.0]]
    assert realized.tolist() == [1.0, 4.0]
    assert clipped.tolist() == [True, False]


def test_video_cluster_bootstrap_uses_equal_video_weight():
    rows = [
        {"source_index": 0, "value": 0.0},
        {"source_index": 0, "value": 0.0},
        {"source_index": 0, "value": 0.0},
        {"source_index": 1, "value": 10.0},
    ]
    result = video_cluster_bootstrap_ci(
        rows, value_key="value", seed=3, samples=1000
    )
    assert result["mean"] == pytest.approx(5.0)
    assert result["videos"] == 2


def _basin_row(source_index, radius, error, utility, severe):
    return {
        "source_index": source_index,
        "variant": "native_probability",
        "nominal_radius_px": radius,
        "realized_commit_error_px": radius,
        "boundary_clipped": False,
        "mean_l2_error_px": error,
        "severe_16px_rate": severe,
        "threshold_utility": utility,
        "native_mean_l2_error_px": 30.0,
        "native_severe_16px_rate": 0.8,
        "native_threshold_utility": 0.1,
    }


def test_aggregate_and_radius_gate_find_largest_supported_radius():
    rows = [
        _basin_row(0, 0.0, 5.0, 0.8, 0.1),
        _basin_row(1, 0.0, 7.0, 0.7, 0.2),
        _basin_row(0, 4.0, 10.0, 0.5, 0.3),
        _basin_row(1, 4.0, 12.0, 0.5, 0.3),
        _basin_row(0, 8.0, 28.0, 0.12, 0.75),
        _basin_row(1, 8.0, 29.0, 0.11, 0.79),
    ]
    aggregate = aggregate_basin_rows(
        rows, bootstrap_seed=7, bootstrap_samples=1000
    )
    gates = evaluate_radius_gates(
        aggregate,
        {
            "mean_error_reduction_px_min": 8.0,
            "error_reduction_CI_lower_px_min": 2.0,
            "utility_gain_min": 0.12,
            "utility_gain_CI_lower_min": 0.03,
            "positive_fraction_min": 0.65,
            "severe_rate_reduction_min": 0.15,
        },
    )
    assert gates["native_probability"]["largest_passing_nominal_radius_px"] == 4.0
    assert gates["native_probability"]["checks_by_radius"]["8"]["pass"] is False


def test_offset_schedule_rejects_duplicate_radii():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_offset_hypotheses([0, 2, 2], {"east": [1, 0]})
