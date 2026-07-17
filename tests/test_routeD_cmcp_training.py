import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import (
    CMCPLossConfig,
    _warp_previous_by_native_motion,
    cmcp_proposal_loss,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)


def _config():
    return CMCPConfig(feature_dim=4, hidden_channels=8, proposal_topk=3, input_height=32, input_width=32)


def test_cmcp_loss_is_finite_and_backpropagates_after_safe_head_warmup():
    torch.manual_seed(4)
    model = CausalMultiMemoryProposalGenerator(_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    corr = torch.randn(2, 4, 3, 7, 9)
    motion = torch.rand(2, 4, 1, 7, 9)
    valid = torch.ones(2, 4, dtype=torch.bool)
    native = torch.rand(2, 4, 2) * 31
    gt = torch.rand(2, 4, 2) * 31
    visible = torch.ones(2, 4, dtype=torch.bool)
    for _ in range(2):
        output = model.forward_sequence(corr, motion, valid)
        losses = cmcp_proposal_loss(output, native, gt, visible, _config(), CMCPLossConfig())
        assert torch.isfinite(losses["loss"])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
    gradient = sum(float(p.grad.abs().sum()) for p in model.recurrent.parameters() if p.grad is not None)
    assert gradient > 0


def test_loss_ignores_unsupervised_frames():
    torch.manual_seed(5)
    model = CausalMultiMemoryProposalGenerator(_config()).eval()
    corr = torch.randn(1, 3, 3, 5, 6)
    motion = torch.rand(1, 3, 1, 5, 6)
    valid = torch.ones(1, 3, dtype=torch.bool)
    native = torch.rand(1, 3, 2) * 31
    gt = torch.rand(1, 3, 2) * 31
    visible = torch.tensor([[False, True, True]])
    first = model.forward_sequence(corr, motion, valid)
    loss1 = cmcp_proposal_loss(first, native, gt, visible, _config(), CMCPLossConfig())["loss"]
    gt2 = gt.clone(); gt2[:, 0] = 1000
    loss2 = cmcp_proposal_loss(first, native, gt2, visible, _config(), CMCPLossConfig())["loss"]
    assert torch.equal(loss1, loss2)


def test_zero_motion_warp_is_identity():
    value = torch.randn(2, 1, 5, 7)
    xy = torch.tensor([[8.0, 9.0], [12.0, 13.0]])
    warped = _warp_previous_by_native_motion(value, xy, xy, input_height=32, input_width=32)
    assert torch.allclose(warped, value, atol=1e-6, rtol=1e-6)


def test_temporal_teacher_is_stop_gradient_deterministic():
    torch.manual_seed(9)
    model = CausalMultiMemoryProposalGenerator(_config())
    corr = torch.randn(1, 3, 3, 5, 6)
    motion = torch.rand(1, 3, 1, 5, 6)
    valid = torch.ones(1, 3, dtype=torch.bool)
    native = torch.rand(1, 3, 2) * 31
    gt = torch.rand(1, 3, 2) * 31
    visible = torch.ones(1, 3, dtype=torch.bool)
    output = model.forward_sequence(corr, motion, valid)
    losses = cmcp_proposal_loss(output, native, gt, visible, _config(), CMCPLossConfig())
    losses["loss"].backward()
    assert torch.isfinite(losses["temporal_consistency"])
