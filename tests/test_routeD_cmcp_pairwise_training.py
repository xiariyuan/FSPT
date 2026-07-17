import torch
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import CMCPLocalPairwiseSafetyComparator,CMCPLocalSafetyConfig
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
 PairwiseSafetyLossConfig,StaticTokenNormalization,normalize_static_tokens,pairwise_safety_frame_loss,safety_feasible,
)


def test_normalization_only_changes_static_dimensions():
 t=torch.randn(2,6,88); t[...,-4:]=torch.randn(2,6,4); valid=torch.ones(2,6,dtype=torch.bool)
 n=StaticTokenNormalization(torch.randn(84),torch.rand(84)+.1); out=normalize_static_tokens(t,valid,n)
 assert torch.equal(out[...,-4:],t[...,-4:])


def test_frame_loss_is_finite_and_backpropagates():
 torch.manual_seed(4); model=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0))
 tokens=torch.randn(3,6,88); valid=torch.ones(3,6,dtype=torch.bool); coords=torch.rand(3,6,2)*255; gt=torch.rand(3,2)*255
 output=model(tokens,valid,coords); losses=pairwise_safety_frame_loss(output,coords,valid,gt,torch.ones(3,dtype=torch.bool),model.config,PairwiseSafetyLossConfig())
 assert torch.isfinite(losses['loss']); losses['loss'].backward(); assert any(p.grad is not None for p in model.parameters())


def test_safety_feasible_requires_both_constraints():
 base={'behavior':{'harmful_non_native_rate':0.01},'severe_16px_rate':{'selected_delta':0.0}}
 assert safety_feasible(base)
 bad={'behavior':{'harmful_non_native_rate':0.011},'severe_16px_rate':{'selected_delta':0.0}}
 assert not safety_feasible(bad)
