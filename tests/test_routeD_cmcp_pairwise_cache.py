import json
from pathlib import Path
import torch

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import (
    CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION,
    CMCP_PAIRWISE_LOCAL_TOKEN_DIM,
    inject_dynamic_summary,
    verify_pairwise_token_artifact,
)


def _artifact(path:Path):
    tokens=torch.zeros(2,3,6,CMCP_PAIRWISE_LOCAL_TOKEN_DIM)
    coords=torch.randn(2,3,6,2); native=coords[...,0,:].clone()
    tensors={'candidate_tokens':tokens,'candidate_coords_xy_px':coords,'candidate_scores':torch.randn(2,3,6),'candidate_valid_mask':torch.ones(2,3,6,dtype=torch.bool),'frame_valid':torch.ones(2,3,dtype=torch.bool),'native_coords_xy_px':native}
    artifact={'schema_version':CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION,'provenance':{'generator_model_state_sha256':'g','base_sidecar_sha256':'b'},'tensors':tensors,'tensor_hashes':{k:tensor_sha256(v) for k,v in tensors.items()}}
    torch.save(artifact,path)


def test_verify_pairwise_token_artifact(tmp_path):
    p=tmp_path/'x.pt'; _artifact(p)
    out=verify_pairwise_token_artifact(p,expected_generator_state_sha256='g',expected_base_sidecar_sha256='b')
    assert out['tensors']['candidate_tokens'].shape[-1]==88


def test_dynamic_summary_injection_preserves_static_fields():
    token=torch.zeros(2,3,6,88); token[...,:84]=torch.randn(2,3,6,84)
    summary=torch.randn(2,3,4)
    out=inject_dynamic_summary(token,summary)
    assert torch.equal(out[...,:84],token[...,:84])
    assert torch.equal(out[...,-4:],summary.unsqueeze(-2).expand(-1,-1,6,-1))
