from types import SimpleNamespace

import torch

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    snapshot_cotracker_online_state,
)
from projects.mmp_tracker.mmp_tracker.routeD_oracle_state_transplant import (
    aggregate_variant_metrics,
    point_future_metrics,
    select_failure_points,
    support_delta_svd_energy,
    transplant_fresh_state,
)


def _snapshot(points: int, value: float):
    model = SimpleNamespace(
        online_ind=8,
        online_track_feat=[torch.full((1, 1, points, 4), value)],
        online_track_support=[torch.full((1, 3, points, 4), value)],
        online_coords_predicted=torch.full((1, 16, points, 2), value),
        online_vis_predicted=torch.full((1, 16, points), value),
        online_conf_predicted=torch.full((1, 16, points), value),
    )
    queries = torch.zeros(1, points, 3)
    return snapshot_cotracker_online_state(SimpleNamespace(N=points, queries=queries, model=model))


def test_transplant_copies_only_selected_time_and_requested_fields():
    native = _snapshot(3, 1.0)
    fresh = _snapshot(2, 5.0)
    result = transplant_fresh_state(
        native,
        fresh,
        selected_point_indices=torch.tensor([0, 2]),
        oracle_query_frames=torch.tensor([10, 12]),
        overlap_end_inclusive=15,
        copy_probability=False,
        copy_track_memory=False,
    )
    assert torch.equal(result.online_coords_predicted[:, :10, 0], native.online_coords_predicted[:, :10, 0])
    assert torch.all(result.online_coords_predicted[:, 10:, 0] == 5)
    assert torch.all(result.online_coords_predicted[:, 12:, 2] == 5)
    assert torch.equal(result.online_vis_predicted, native.online_vis_predicted)
    assert torch.equal(result.online_track_support[0], native.online_track_support[0])


def test_full_transplant_copies_probabilities_and_memory_mapping():
    native = _snapshot(3, 1.0)
    fresh = _snapshot(2, 5.0)
    result = transplant_fresh_state(
        native,
        fresh,
        selected_point_indices=torch.tensor([0, 2]),
        oracle_query_frames=torch.tensor([8, 8]),
        overlap_end_inclusive=15,
        copy_probability=True,
        copy_track_memory=True,
    )
    assert torch.all(result.online_vis_predicted[:, 8:, [0, 2]] == 5)
    assert torch.all(result.online_track_feat[0][:, :, [0, 2]] == 5)
    assert torch.all(result.online_track_support[0][:, :, [0, 2]] == 5)
    assert torch.all(result.online_track_support[0][:, :, 1] == 1)


def test_failure_selection_has_no_below_threshold_fallback():
    native = torch.zeros(3, 24, 2)
    gt = torch.zeros(3, 24, 2)
    gt[0, 16:, 1] = 20 / 255.0
    gt[1, 16:, 1] = 10 / 255.0
    occ = torch.zeros(3, 24, dtype=torch.bool)
    result = select_failure_points(
        native_coords_xy_px=native,
        gt_tracks_yx=gt,
        gt_occluded=occ,
        original_query_frames=torch.tensor([0, 0, 9]),
        overlap_start=8,
        overlap_end_inclusive=15,
        future_start=16,
        future_end_inclusive=23,
        min_overlap_visible_frames=1,
        min_future_visible_frames=4,
        min_native_future_mean_error_px=16,
        per_video_cap=8,
    )
    assert result["selected_point_indices"].tolist() == [0]
    assert result["oracle_query_frames"].tolist() == [15]


def test_metrics_and_support_svd_are_well_formed():
    coords = torch.zeros(1, 24, 2, 2)
    gt = torch.zeros(2, 24, 2)
    occ = torch.zeros(2, 24, dtype=torch.bool)
    rows = point_future_metrics(
        final_coords_xy_model=coords,
        selected_point_indices=torch.tensor([0, 1]),
        gt_tracks_yx=gt,
        gt_occluded=occ,
        future_start=16,
        future_end_inclusive=23,
        interp_height=256,
        interp_width=256,
    )
    aggregate = aggregate_variant_metrics(rows)
    assert aggregate["points"] == 2
    assert aggregate["mean_l2_error_px"] == 0
    native = _snapshot(3, 1.0)
    fresh = _snapshot(2, 2.0)
    svd = support_delta_svd_energy(
        native,
        fresh,
        selected_point_indices=torch.tensor([0, 2]),
        ranks=[1, 2],
    )
    assert 0 < svd["pooled_rank_energy"]["1"] <= 1
    assert svd["pooled_rank_energy"]["2"] >= svd["pooled_rank_energy"]["1"]
