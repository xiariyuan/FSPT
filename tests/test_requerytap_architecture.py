from dataclasses import dataclass

import torch
from torch import nn

from mmp_tracker.requerytap import ReQueryTAP


@dataclass
class DummyState:
    step: int
    query_points: torch.Tensor


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size=(16,16)
        self.patch_size=(4,4)
        self.lin_proj=nn.Conv2d(3,8,kernel_size=4,stride=4)
        self.image_pos_emb=nn.Parameter(torch.zeros(1,16,8))
        self.head=nn.Linear(8,3)

    def forward(self,video,query_points=None,state=None):
        if state is None:
            assert query_points is not None
            query=query_points
            step=0
        else:
            query=state.query_points
            step=state.step
        pooled=video.mean(dim=(1,2,3,4),keepdim=False)[:,None,None]
        base=query[...,1:3]+pooled
        tracks=base[:,None]
        logits=torch.cat([base.repeat(1,1,4),base.repeat(1,1,4)],dim=-1)[:,None]
        vis=(pooled+1.0).reshape(video.shape[0],1,1,1)
        return tracks,logits,vis,DummyState(step+1,query)


def make_inputs():
    video=torch.randn(2,1,16,16,3)
    query=torch.tensor([[[0.,4.,5.]],[[0.,8.,9.]]])
    return video,query


def test_native_off_is_exact():
    torch.manual_seed(3)
    base=DummyBackbone(); model=ReQueryTAP(base,identity_dim=4)
    video,query=make_inputs(); initialized=model.initialize(video,query)
    next_video=torch.randn_like(video)
    direct=base(next_video,state=initialized.state.track_state)
    output=model.step(next_video,initialized.state,allow_respawn=False)
    assert torch.equal(output.selected_tracks,direct[0])
    assert torch.equal(output.selected_track_logits,direct[1])
    assert torch.equal(output.selected_visibility_logits,direct[2])
    assert output.state.generation==0


def test_forced_respawn_equals_standalone_fresh_path():
    torch.manual_seed(4)
    base=DummyBackbone(); model=ReQueryTAP(base,identity_dim=4)
    video,query=make_inputs(); initialized=model.initialize(video,query)
    next_video=torch.randn_like(video)
    forced=torch.tensor([[[6.,7.]],[[10.,11.]]])
    fresh_query=torch.cat([torch.zeros(2,1,1),forced],dim=-1)
    direct=base(next_video,query_points=fresh_query)
    output=model.step(next_video,initialized.state,force_respawn=True,forced_query_coordinate_yx=forced)
    assert torch.equal(output.selected_tracks,direct[0])
    assert torch.equal(output.selected_track_logits,direct[1])
    assert output.respawn_applied
    assert output.state.generation==1


def test_predicted_fresh_path_is_differentiable():
    torch.manual_seed(5)
    base=DummyBackbone(); model=ReQueryTAP(base,identity_dim=4)
    model.train(); video,query=make_inputs(); initialized=model.initialize(video,query)
    output=model.step(torch.randn_like(video),initialized.state,force_respawn=True)
    loss=output.selected_tracks.square().mean()+output.rebind_coordinate_yx.square().mean()+output.respawn_logit.square().mean()
    loss.backward()
    gradients=[p.grad for p in model.locator.parameters()]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0 for grad in gradients)


def test_rejects_multi_query_input():
    base=DummyBackbone(); model=ReQueryTAP(base,identity_dim=4)
    video=torch.randn(1,1,16,16,3); query=torch.zeros(1,2,3)
    try:model.initialize(video,query)
    except ValueError as error:assert 'one independent query' in str(error)
    else:raise AssertionError('multi-query input must fail')
