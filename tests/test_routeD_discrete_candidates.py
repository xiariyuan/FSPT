import torch

from projects.mmp_tracker.mmp_tracker.routeD_discrete_candidates import (
    extract_discrete_candidates,
)


def test_discrete_candidates_use_y_then_x_tie_break():
    score = torch.zeros(1, 8, 8)
    score[0, 1, 5] = 3.0
    score[0, 2, 1] = 3.0
    result = extract_discrete_candidates(
        score,
        torch.tensor([[0.0, 255.0]]),
        top_k=2,
        nms_radius_grid_cells=0,
        local_refinement_window_grid_cells=1,
        deduplicate_radius_input_px=0.0,
    )
    assert result["candidate_peak_yx"][0, 1].tolist() == [1, 5]
    assert result["candidate_peak_yx"][0, 2].tolist() == [2, 1]


def test_native_duplicate_is_suppressed_and_backfilled():
    score = torch.zeros(1, 8, 8)
    score[0, 0, 0] = 10.0
    score[0, 4, 4] = 9.0
    result = extract_discrete_candidates(
        score,
        torch.tensor([[0.0, 0.0]]),
        top_k=1,
        nms_radius_grid_cells=1,
        local_refinement_window_grid_cells=1,
        deduplicate_radius_input_px=4.0,
    )
    assert result["candidate_count"].item() == 2
    assert result["candidate_peak_yx"][0, 1].tolist() == [4, 4]


def test_local_refinement_uses_cropped_five_by_five_softmax():
    score = torch.full((1, 9, 9), -10.0)
    score[0, 4, 4] = 1.0
    score[0, 4, 5] = 1.0
    result = extract_discrete_candidates(
        score,
        torch.tensor([[0.0, 0.0]]),
        top_k=1,
        nms_radius_grid_cells=0,
        local_refinement_window_grid_cells=5,
        local_softmax_temperature=0.05,
        deduplicate_radius_input_px=0.0,
    )
    expected_x = 4.5 * 255.0 / 8.0
    expected_y = 4.0 * 255.0 / 8.0
    assert torch.allclose(
        result["candidate_coordinates_xy"][0, 1],
        torch.tensor([expected_x, expected_y]),
        atol=1.0e-4,
    )


def test_candidate_extraction_is_exactly_replayable():
    generator = torch.Generator().manual_seed(71)
    score = torch.randn(3, 16, 16, generator=generator)
    native = torch.tensor([[10.0, 20.0], [100.0, 120.0], [220.0, 30.0]])
    first = extract_discrete_candidates(score, native)
    second = extract_discrete_candidates(score, native)
    for key in first:
        left, right = first[key], second[key]
        if left.dtype.is_floating_point:
            assert torch.equal(torch.isnan(left), torch.isnan(right))
            left = torch.nan_to_num(left)
            right = torch.nan_to_num(right)
        assert torch.equal(left, right)
