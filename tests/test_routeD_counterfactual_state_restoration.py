from types import SimpleNamespace

import torch

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    composite_corrupt_snapshot,
    future_difference,
    restore_coordinates_only,
    restore_cotracker_online_state,
    snapshot_cotracker_online_state,
    snapshots_exact,
)


def _predictor() -> SimpleNamespace:
    points, frames, channels = 3, 16, 8
    model = SimpleNamespace(
        online_ind=8,
        online_track_feat=[torch.randn(1, 1, points, channels) for _ in range(2)],
        online_track_support=[torch.randn(1, 9, points, channels) for _ in range(2)],
        online_coords_predicted=torch.randn(1, frames, points, 2),
        online_vis_predicted=torch.randn(1, frames, points),
        online_conf_predicted=torch.randn(1, frames, points),
    )
    queries = torch.tensor([[[0.0, 1.0, 2.0], [15.0, 3.0, 4.0], [20.0, 5.0, 6.0]]])
    return SimpleNamespace(N=points, queries=queries, model=model)


def test_snapshot_restore_is_exact_and_has_no_aliases():
    torch.manual_seed(1)
    predictor = _predictor()
    snapshot = snapshot_cotracker_online_state(predictor)
    predictor.model.online_coords_predicted.add_(10)
    predictor.model.online_track_support[0].zero_()
    predictor.queries.add_(1)
    restore_cotracker_online_state(predictor, snapshot)
    restored = snapshot_cotracker_online_state(predictor)
    assert snapshots_exact(snapshot, restored)
    predictor.model.online_coords_predicted.add_(1)
    assert not torch.equal(
        predictor.model.online_coords_predicted, snapshot.online_coords_predicted
    )


def test_composite_corruption_respects_points_frames_and_channel_count():
    torch.manual_seed(2)
    clean = snapshot_cotracker_online_state(_predictor())
    corrupted = composite_corrupt_snapshot(
        clean,
        input_height=256,
        input_width=256,
        interp_height=384,
        interp_width=512,
        overlap_start=8,
        overlap_end_inclusive=15,
        active_before_frame=16,
        coordinate_shift_input_xy_px=(16.0, -12.0),
        visibility_logit_delta=-4.0,
        confidence_logit_delta=-4.0,
        support_channel_keep_fraction=0.5,
        seed=190719,
    )
    # Points 0 and 1 are active; point 2 has query frame 20 and is untouched.
    assert torch.equal(
        corrupted.online_coords_predicted[:, :8], clean.online_coords_predicted[:, :8]
    )
    assert not torch.equal(
        corrupted.online_coords_predicted[:, 8:, :2],
        clean.online_coords_predicted[:, 8:, :2],
    )
    assert torch.equal(
        corrupted.online_coords_predicted[:, 8:, 2],
        clean.online_coords_predicted[:, 8:, 2],
    )
    for before, after in zip(clean.online_track_support, corrupted.online_track_support):
        assert before is not None and after is not None
        # Exactly half the channels remain for active points.
        assert int((after[0, 0, 0] != 0).sum().item()) == 4
        assert torch.equal(after[:, :, 2], before[:, :, 2])


def test_coordinate_only_restore_leaves_other_corrupted_state():
    torch.manual_seed(3)
    clean = snapshot_cotracker_online_state(_predictor())
    corrupted = composite_corrupt_snapshot(
        clean,
        input_height=256,
        input_width=256,
        interp_height=384,
        interp_width=512,
        overlap_start=8,
        overlap_end_inclusive=15,
        active_before_frame=16,
        coordinate_shift_input_xy_px=(16.0, -12.0),
        visibility_logit_delta=-4.0,
        confidence_logit_delta=-4.0,
        support_channel_keep_fraction=0.5,
        seed=190719,
    )
    partial = restore_coordinates_only(corrupted, clean)
    assert torch.equal(partial.online_coords_predicted, clean.online_coords_predicted)
    assert not torch.equal(partial.online_vis_predicted, clean.online_vis_predicted)
    assert not torch.equal(partial.online_conf_predicted, clean.online_conf_predicted)
    assert not torch.equal(partial.online_track_support[0], clean.online_track_support[0])


def test_future_difference_uses_active_rows_and_input_raster():
    torch.manual_seed(4)
    predictor = _predictor()
    predictor.model.online_coords_predicted = torch.zeros(1, 24, 3, 2)
    predictor.model.online_vis_predicted = torch.zeros(1, 24, 3)
    predictor.model.online_conf_predicted = torch.zeros(1, 24, 3)
    clean = snapshot_cotracker_online_state(predictor)
    other = snapshot_cotracker_online_state(predictor)
    # Replace immutable snapshot with a shifted coordinate tensor.
    shifted = other.online_coords_predicted.clone()
    shifted[:, 16:24, :, 0] += (512 - 1) / (256 - 1)
    other = type(other)(
        predictor_n=other.predictor_n,
        predictor_queries=other.predictor_queries,
        online_ind=other.online_ind,
        online_track_feat=other.online_track_feat,
        online_track_support=other.online_track_support,
        online_coords_predicted=shifted,
        online_vis_predicted=other.online_vis_predicted,
        online_conf_predicted=other.online_conf_predicted,
    )
    metrics = future_difference(
        clean,
        other,
        future_start=16,
        future_end_inclusive=23,
        input_height=256,
        input_width=256,
        interp_height=384,
        interp_width=512,
    )
    assert metrics["rows"] == 20  # point 2 becomes active for frames 20--23.
    assert abs(metrics["coordinate_mean_l2_px"] - 1.0) < 1e-5
    assert abs(metrics["coordinate_max_l2_px"] - 1.0) < 1e-5
