import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCP_PAIRWISE_LOCAL_TOKEN_DIM,
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
    build_cmcp_local_candidate_tokens,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)


def _inputs(batch=2, candidates=6):
    torch.manual_seed(3)
    h,w=7,9
    coords=torch.rand(batch,candidates,2)*31
    valid=torch.ones(batch,candidates,dtype=torch.bool)
    scores=torch.randn(batch,candidates)
    return dict(
        hidden_map=torch.randn(batch,64,h,w),
        recurrent_input=torch.randn(batch,9,h,w),
        utility_logit=torch.randn(batch,1,h,w),
        risk_logit=torch.randn(batch,1,h,w),
        proposal_score=torch.randn(batch,1,h,w),
        candidate_coords_xy_px=coords,
        candidate_scores=scores,
        candidate_valid_mask=valid,
        native_visibility_probability=torch.rand(batch),
        native_confidence_probability=torch.rand(batch),
        native_joint_probability=torch.rand(batch),
        previous_decision_summary=torch.zeros(batch,4),
        input_height=32,
        input_width=32,
    )


def test_local_token_contract_and_invalid_zero():
    values=_inputs(); values['candidate_valid_mask'][0,-1]=False
    token=build_cmcp_local_candidate_tokens(**values)
    assert token.shape==(2,6,CMCP_PAIRWISE_LOCAL_TOKEN_DIM)
    assert not token[0,-1].any()


def test_zero_step_is_exact_native_and_coords_bypass():
    values=_inputs()
    token=build_cmcp_local_candidate_tokens(**values)
    model=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0)).eval()
    out=model(token, values['candidate_valid_mask'], values['candidate_coords_xy_px'])
    assert torch.equal(out['selected_candidate_index'], torch.zeros(2,dtype=torch.long))
    assert torch.equal(out['selected_coord_xy_px'], values['candidate_coords_xy_px'][:,0])
    assert torch.equal(out['candidate_coords_xy_px'], values['candidate_coords_xy_px'])


def test_utility_probabilities_are_monotonic():
    values=_inputs(); token=build_cmcp_local_candidate_tokens(**values)
    model=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0)).eval()
    p=model(token,values['candidate_valid_mask'],values['candidate_coords_xy_px'])['utility_probability']
    assert torch.all(p[...,1:] >= p[...,:-1])


def test_nonnative_permutation_equivariance():
    values=_inputs(); token=build_cmcp_local_candidate_tokens(**values)
    model=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0)).eval()
    first=model(token,values['candidate_valid_mask'],values['candidate_coords_xy_px'])
    perm=torch.tensor([0,3,1,5,2,4])
    second=model(token[:,perm],values['candidate_valid_mask'][:,perm],values['candidate_coords_xy_px'][:,perm])
    inverse=torch.argsort(perm)
    assert torch.allclose(first['candidate_score'], second['candidate_score'][:,inverse])


def test_invalid_candidate_cannot_be_selected_after_abstention_disabled():
    values=_inputs(); values['candidate_valid_mask'][:,-1]=False
    token=build_cmcp_local_candidate_tokens(**values)
    model=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0))
    with torch.no_grad():
        model.abstention_head.bias.fill_(-10)
        model.preference_head.bias.fill_(0)
    out=model(token,values['candidate_valid_mask'],values['candidate_coords_xy_px'])
    assert not torch.any(out['selected_candidate_index']==5)


def test_frozen_generator_receives_no_gradient():
    generator=CausalMultiMemoryProposalGenerator(CMCPConfig(feature_dim=4,hidden_channels=8,input_height=32,input_width=32))
    for p in generator.parameters(): p.requires_grad_(False)
    comparator=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(dropout=0.0))
    values=_inputs(); token=build_cmcp_local_candidate_tokens(**values).detach().requires_grad_(True)
    out=comparator(token,values['candidate_valid_mask'],values['candidate_coords_xy_px'])
    out['candidate_score'][torch.isfinite(out['candidate_score'])].sum().backward()
    assert all(p.grad is None for p in generator.parameters())
    assert token.grad is not None
