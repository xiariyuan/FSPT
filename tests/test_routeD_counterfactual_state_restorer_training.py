import torch

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    CSRRBatchLossConfig,
    aggregate_point_rows,
    csrr_batch_loss,
    gaussian_heatmap_targets,
    point_future_rows,
)


def test_gaussian_heatmap_peaks_near_target():
    target = gaussian_heatmap_targets(torch.tensor([[0.5, 0.25]]), height=64, width=64)
    flat = target[0].argmax().item()
    y, x = divmod(flat, 64)
    assert abs(x - 31.5) <= 1
    assert abs(y - 15.75) <= 1


def test_point_future_rows_and_aggregate():
    coords = torch.tensor([[[0.0, 0.0], [2.0, 0.0]], [[20.0, 0.0], [20.0, 0.0]]])
    gt = torch.zeros_like(coords)
    visible = torch.ones(2, 2, dtype=torch.bool)
    rows = point_future_rows(coords, gt, visible)
    agg = aggregate_point_rows(rows)
    assert rows[0]["mean_l2_error_px"] == 1.0
    assert rows[1]["severe_16px_rate"] == 1.0
    assert agg["points"] == 2


def test_batch_loss_is_finite_and_backpropagates():
    torch.manual_seed(12)
    batch_size = 4
    logits = torch.randn(batch_size, 64, 64, requires_grad=True)
    predicted_xy = torch.rand(batch_size, 2, requires_grad=True) * 255
    output = {
        "fused_logits": logits,
        "predicted_coordinates_xy": predicted_xy,
        "apply_logit": torch.randn(batch_size, requires_grad=True),
        "visibility_residual": torch.randn(batch_size, requires_grad=True),
        "confidence_residual": torch.randn(batch_size, requires_grad=True),
    }
    re_feat = [torch.randn(batch_size, 1, 1, 128, requires_grad=True) for _ in range(4)]
    re_support = [torch.randn(batch_size, 49, 1, 128, requires_grad=True) for _ in range(4)]
    batch = {
        "apply_target": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "teacher_commit_coordinates_normalized_xy": torch.rand(batch_size, 2),
        "native_commit_coordinates_normalized_xy": torch.rand(batch_size, 2),
        "native_visibility_probability": torch.rand(batch_size),
        "native_confidence_probability": torch.rand(batch_size),
        "teacher_visibility_probability": torch.rand(batch_size),
        "teacher_confidence_probability": torch.rand(batch_size),
        "teacher_track_feat": [torch.randn(batch_size, 1, 128) for _ in range(4)],
        "teacher_track_support": [torch.randn(batch_size, 49, 128) for _ in range(4)],
        "native_track_feat": [torch.randn(batch_size, 1, 128) for _ in range(4)],
        "native_track_support": [torch.randn(batch_size, 49, 128) for _ in range(4)],
    }
    losses = csrr_batch_loss(
        output=output,
        reextracted_track_features=re_feat,
        reextracted_track_supports=re_support,
        batch=batch,
        config=CSRRBatchLossConfig(),
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert logits.grad is not None
