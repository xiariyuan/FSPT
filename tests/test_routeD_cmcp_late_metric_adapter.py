import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRA_TRAINABLE_PARAMETERS,
    LateMetricResidualAdapter,
    LMRAConfig,
    lmra_distortion_loss,
)


def test_lmra_parameter_count_and_zero_identity():
    torch.manual_seed(1)
    model = LateMetricResidualAdapter(LMRAConfig())
    value = torch.randn(3, 128, 7, 9)
    output = model(value)
    assert sum(p.numel() for p in model.parameters()) == LMRA_TRAINABLE_PARAMETERS
    assert torch.equal(output, value)
    assert not model.residual(value).any()


def test_lmra_gradients_reach_both_projections_after_up_warmup():
    torch.manual_seed(2)
    model = LateMetricResidualAdapter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    value = torch.randn(2, 128, 4, 5)
    for _ in range(2):
        loss = model(value).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert model.up.weight.grad is not None and model.up.weight.grad.abs().sum() > 0
    assert model.down.weight.grad is not None and model.down.weight.grad.abs().sum() > 0


def test_distortion_loss_zero_at_initialization():
    value = torch.randn(1, 128, 4, 4)
    model = LateMetricResidualAdapter()
    assert lmra_distortion_loss(model(value), value).item() == 0.0


def test_config_rejects_rank_sweep():
    try:
        LMRAConfig(rank=16)
    except ValueError:
        pass
    else:
        raise AssertionError("rank sweep must be rejected")
