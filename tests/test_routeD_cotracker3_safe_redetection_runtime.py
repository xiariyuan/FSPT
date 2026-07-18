from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_cotracker3_safe_redetection_runtime import (
    build_official_single_point_queries,
    points_on_grid,
)


def test_grid_matches_official_margin_and_row_major_order():
    grid = points_on_grid(5, (100.0, 200.0), None, device=torch.device("cpu"))
    assert grid.shape == (1, 25, 2)
    assert torch.allclose(grid[0, 0], torch.tensor([3.125, 3.125]))
    assert torch.allclose(grid[0, -1], torch.tensor([196.875, 96.875]))
    assert torch.all(grid[0, 1:, 0] >= 0)


def test_official_support_query_count_and_target_identity():
    target = torch.tensor([[[7.0, 100.0, 50.0]]])
    queries = build_official_single_point_queries(
        target,
        input_height=256,
        input_width=256,
        interp_height=384,
        interp_width=512,
    )
    assert queries.shape == (1, 1 + 64 + 25, 3)
    assert torch.equal(queries[:, :1], target)
    assert torch.equal(queries[:, 1:, 0], torch.zeros_like(queries[:, 1:, 0]))
