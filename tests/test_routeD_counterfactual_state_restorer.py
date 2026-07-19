from types import SimpleNamespace

import torch

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    CoTrackerOnlineStateSnapshot,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
    CounterfactualStructuredReextractionRestorer,
    apply_reextracted_state_action,
    build_csrr_trajectory_features,
    input_xy_to_model_xy,
    model_xy_to_input_xy,
)


def _snapshot(points: int = 3) -> CoTrackerOnlineStateSnapshot:
    return CoTrackerOnlineStateSnapshot(
        predictor_n=points,
        predictor_queries=torch.zeros(1, points, 3),
        online_ind=8,
        online_track_feat=tuple(torch.randn(1, 1, points, 128) for _ in range(4)),
        online_track_support=tuple(torch.randn(1, 49, points, 128) for _ in range(4)),
        online_coords_predicted=torch.randn(1, 16, points, 2),
        online_vis_predicted=torch.randn(1, 16, points),
        online_conf_predicted=torch.randn(1, 16, points),
    )


def test_coordinate_roundtrip():
    xy = torch.tensor([[0.0, 0.0], [255.0, 255.0], [83.25, 129.5]])
    model = input_xy_to_model_xy(
        xy, input_height=256, input_width=256, model_height=384, model_width=512
    )
    recovered = model_xy_to_input_xy(
        model, input_height=256, input_width=256, model_height=384, model_width=512
    )
    torch.testing.assert_close(recovered, xy, rtol=0, atol=1.0e-5)


def test_model_shapes_and_parameter_ceiling():
    torch.manual_seed(3)
    model = CounterfactualStructuredReextractionRestorer()
    batch = 2
    output = model(
        trajectory_features=torch.randn(batch, 8, 9),
        native_support_pyramid=[torch.randn(batch, 49, 128) for _ in range(4)],
        frame_feature_pyramid=[
            torch.randn(batch, 128, 96, 128),
            torch.randn(batch, 128, 48, 64),
            torch.randn(batch, 128, 24, 32),
            torch.randn(batch, 128, 12, 16),
        ],
        native_commit_coordinates_xy=torch.rand(batch, 2) * 255.0,
    )
    assert model.trainable_parameter_count <= 250_000
    assert output["predicted_coordinates_xy"].shape == (batch, 2)
    assert output["apply_logit"].shape == (batch,)
    assert output["spatial_probability"].shape == (batch, 64, 64)
    torch.testing.assert_close(
        output["spatial_probability"].flatten(1).sum(dim=1),
        torch.ones(batch),
        rtol=0,
        atol=1.0e-5,
    )


def test_false_apply_is_exact_noop():
    torch.manual_seed(4)
    snapshot = _snapshot()
    updated = apply_reextracted_state_action(
        snapshot,
        point_indices=torch.tensor([0, 2]),
        predicted_coordinates_input_xy=torch.tensor([[20.0, 30.0], [80.0, 90.0]]),
        apply_mask=torch.tensor([False, False]),
        reextracted_track_features=[torch.randn(1, 1, 2, 128) for _ in range(4)],
        reextracted_track_supports=[torch.randn(1, 49, 2, 128) for _ in range(4)],
        input_height=256,
        input_width=256,
        model_height=384,
        model_width=512,
    )
    assert snapshots_exact(snapshot, updated)
    assert snapshot.online_coords_predicted.data_ptr() != updated.online_coords_predicted.data_ptr()


def test_true_apply_changes_only_requested_point_memory():
    torch.manual_seed(5)
    snapshot = _snapshot()
    new_feat = [torch.randn(1, 1, 2, 128) for _ in range(4)]
    new_support = [torch.randn(1, 49, 2, 128) for _ in range(4)]
    updated = apply_reextracted_state_action(
        snapshot,
        point_indices=torch.tensor([0, 2]),
        predicted_coordinates_input_xy=torch.tensor([[20.0, 30.0], [80.0, 90.0]]),
        apply_mask=torch.tensor([True, False]),
        reextracted_track_features=new_feat,
        reextracted_track_supports=new_support,
        input_height=256,
        input_width=256,
        model_height=384,
        model_width=512,
    )
    assert not torch.equal(snapshot.online_coords_predicted[:, 15, 0], updated.online_coords_predicted[:, 15, 0])
    torch.testing.assert_close(snapshot.online_coords_predicted[:, :, 1:], updated.online_coords_predicted[:, :, 1:])
    for level in range(4):
        torch.testing.assert_close(updated.online_track_feat[level][:, :, 0], new_feat[level][:, :, 0])
        torch.testing.assert_close(updated.online_track_support[level][:, :, 0], new_support[level][:, :, 0])
        torch.testing.assert_close(updated.online_track_feat[level][:, :, 2], snapshot.online_track_feat[level][:, :, 2])
        torch.testing.assert_close(updated.online_track_support[level][:, :, 2], snapshot.online_track_support[level][:, :, 2])


def test_real_trajectory_feature_contract_is_nine_dimensional():
    torch.manual_seed(6)
    snapshot = _snapshot(points=3)
    snapshot = CoTrackerOnlineStateSnapshot(
        predictor_n=snapshot.predictor_n,
        predictor_queries=torch.tensor([[[0.0, 100.0, 120.0], [0.0, 200.0, 220.0], [0.0, 300.0, 320.0]]]),
        online_ind=snapshot.online_ind,
        online_track_feat=snapshot.online_track_feat,
        online_track_support=snapshot.online_track_support,
        online_coords_predicted=snapshot.online_coords_predicted,
        online_vis_predicted=snapshot.online_vis_predicted,
        online_conf_predicted=snapshot.online_conf_predicted,
    )
    features = build_csrr_trajectory_features(
        snapshot, point_indices=torch.tensor([0, 2])
    )
    assert features.shape == (2, 8, 9)
    torch.testing.assert_close(features[:, 0, 2:4], torch.zeros(2, 2))
    assert torch.all((features[..., 4:6] >= 0) & (features[..., 4:6] <= 1))


def test_probability_residual_is_applied_in_probability_space():
    torch.manual_seed(7)
    snapshot = _snapshot()
    point = torch.tensor([0])
    feat = [torch.randn(1, 1, 1, 128) for _ in range(4)]
    support = [torch.randn(1, 49, 1, 128) for _ in range(4)]
    before_vis = torch.sigmoid(snapshot.online_vis_predicted[0, 15, 0])
    before_conf = torch.sigmoid(snapshot.online_conf_predicted[0, 15, 0])
    updated = apply_reextracted_state_action(
        snapshot,
        point_indices=point,
        predicted_coordinates_input_xy=torch.tensor([[30.0, 40.0]]),
        apply_mask=torch.tensor([True]),
        reextracted_track_features=feat,
        reextracted_track_supports=support,
        input_height=256,
        input_width=256,
        model_height=384,
        model_width=512,
        visibility_residual=torch.tensor([0.1]),
        confidence_residual=torch.tensor([-0.1]),
    )
    after_vis = torch.sigmoid(updated.online_vis_predicted[0, 15, 0])
    after_conf = torch.sigmoid(updated.online_conf_predicted[0, 15, 0])
    torch.testing.assert_close(after_vis, (before_vis + 0.1).clamp(1e-5, 1 - 1e-5))
    torch.testing.assert_close(after_conf, (before_conf - 0.1).clamp(1e-5, 1 - 1e-5))


def test_shared_frame_feature_batch_matches_explicit_repeat():
    torch.manual_seed(8)
    model = CounterfactualStructuredReextractionRestorer()
    batch = 3
    support = [torch.randn(batch, 49, 128) for _ in range(4)]
    shared = [
        torch.randn(1, 128, 12, 16),
        torch.randn(1, 128, 8, 10),
        torch.randn(1, 128, 6, 8),
        torch.randn(1, 128, 4, 5),
    ]
    common = dict(
        trajectory_features=torch.randn(batch, 8, 9),
        native_support_pyramid=support,
        native_commit_coordinates_xy=torch.rand(batch, 2) * 255,
    )
    shared_out = model(frame_feature_pyramid=shared, **common)
    repeated_out = model(
        frame_feature_pyramid=[value.repeat(batch, 1, 1, 1) for value in shared],
        **common,
    )
    torch.testing.assert_close(
        shared_out["fused_logits"], repeated_out["fused_logits"], rtol=0, atol=2e-6
    )


def test_coordinate_only_control_does_not_write_probability_or_memory():
    torch.manual_seed(9)
    snapshot = _snapshot()
    feat = [torch.randn(1, 1, 1, 128) for _ in range(4)]
    support = [torch.randn(1, 49, 1, 128) for _ in range(4)]
    updated = apply_reextracted_state_action(
        snapshot,
        point_indices=torch.tensor([0]),
        predicted_coordinates_input_xy=torch.tensor([[50.0, 60.0]]),
        apply_mask=torch.tensor([True]),
        reextracted_track_features=feat,
        reextracted_track_supports=support,
        input_height=256,
        input_width=256,
        model_height=384,
        model_width=512,
        visibility_residual=torch.tensor([0.2]),
        confidence_residual=torch.tensor([0.2]),
        write_probability=False,
        write_memory=False,
    )
    torch.testing.assert_close(updated.online_vis_predicted, snapshot.online_vis_predicted)
    torch.testing.assert_close(updated.online_conf_predicted, snapshot.online_conf_predicted)
    for left, right in zip(updated.online_track_feat, snapshot.online_track_feat):
        torch.testing.assert_close(left, right)
    for left, right in zip(updated.online_track_support, snapshot.online_track_support):
        torch.testing.assert_close(left, right)
    assert not torch.equal(updated.online_coords_predicted[:, 15, 0], snapshot.online_coords_predicted[:, 15, 0])


def test_deterministic_sampler_known_plane_and_repeatable_gradient():
    from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import (
        deterministic_reextract_cotracker_memory,
    )

    yy, xx = torch.meshgrid(torch.arange(4.0), torch.arange(5.0), indexing="ij")
    feature = (xx + 10.0 * yy)[None, None]
    gradients = []
    values = []
    for _ in range(2):
        coordinate = torch.tensor([[2.5, 1.5]], requires_grad=True)
        track, support = deterministic_reextract_cotracker_memory(
            [feature],
            coordinate,
            input_height=4,
            input_width=5,
            model_height=4,
            model_width=5,
            stride=1,
            support_radius=0,
        )
        value = track[0][0, 0, 0, 0]
        value.backward()
        values.append(value.detach())
        gradients.append(coordinate.grad.detach().clone())
        assert support[0].shape == (1, 1, 1, 1)
    torch.testing.assert_close(values[0], torch.tensor(17.5))
    torch.testing.assert_close(gradients[0], torch.tensor([[1.0, 10.0]]))
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
