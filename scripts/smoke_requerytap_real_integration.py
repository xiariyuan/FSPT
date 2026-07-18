#!/usr/bin/env python3
"""Real-checkpoint architecture integrity smoke for ReQueryTAP Gate B."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from audit_tapnextpp_query_state_bridge_static import clone_tracking_state, prepare_frame
from mmp_tracker.requerytap import ReQueryTAP, ReQueryTAPState

DEFAULT_REPO=Path('/gemini/code/FSPT/external/tapnextpp/repo')
DEFAULT_CKPT=Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_SCENE=Path('/gemini/code/FSPT/datasets/pointodyssey/train/ani13_new_f')


def clone_lifecycle(state:ReQueryTAPState)->ReQueryTAPState:
    return ReQueryTAPState(
        track_state=clone_tracking_state(state.track_state),
        identity_memory=state.identity_memory,
        original_query_points=state.original_query_points,
        generation=state.generation,
        frames_since_respawn=state.frames_since_respawn,
    )


def state_max_abs(left,right)->float:
    values=[]
    for a,b in zip(left.hidden_state,right.hidden_state):
        values.append((a.rg_lru_state-b.rg_lru_state).abs().max())
        values.append((a.conv1d_state-b.conv1d_state).abs().max())
    values.append((left.query_points-right.query_points).abs().max())
    return float(torch.stack(values).max().item())


def load_inputs(scene:Path,device:str):
    images=[]
    for index in [0,1]:
        arr=np.asarray(Image.open(scene/'rgbs'/f'rgb_{index:05d}.jpg').convert('RGB'),dtype=np.uint8)
        images.append(prepare_frame(arr,device=device))
    with np.load(scene/'anno.npz',allow_pickle=False) as data:
        coords=np.asarray(data['trajs_2d'][:2],dtype=np.float32)
        valid=np.asarray(data['visibs'][:2]&data['valids'][:2],dtype=bool).all(0)
    h,w=np.asarray(Image.open(scene/'rgbs'/'rgb_00000.jpg')).shape[:2]
    finite=np.isfinite(coords).all((0,2)); inside=(coords[...,0]>=64).all(0)&(coords[...,0]<w-64).all(0)&(coords[...,1]>=64).all(0)&(coords[...,1]<h-64).all(0)
    ids=np.flatnonzero(valid&finite&inside)
    if ids.size==0:raise RuntimeError('no valid point')
    point=int(ids[len(ids)//2]); xy=coords[:,point]
    scaled=xy*np.asarray([256.0/w,256.0/h],dtype=np.float32)
    yx=torch.from_numpy(scaled[:,::-1].copy()).to(device)
    query=torch.tensor([[[0.0,float(yx[0,0]),float(yx[0,1])]]],device=device)
    return images,query,yx,point


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=DEFAULT_REPO);ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT);ap.add_argument('--scene',type=Path,default=DEFAULT_SCENE);ap.add_argument('--device',default='cuda');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if 'test' in args.scene.parts:raise ValueError('test data forbidden')
    sys.path.insert(0,str(args.repo))
    from tapnet.tapnext.tapnext_torch import TAPNext
    device=torch.device(args.device); frames,query,target_yx,point=load_inputs(args.scene,args.device)
    backbone=TAPNext(image_size=(256,256),use_checkpointing=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    backbone.load_state_dict({k.replace('tapnext.',''):v for k,v in payload['state_dict'].items()})
    backbone.to(device).eval()
    for parameter in backbone.parameters():parameter.requires_grad_(False)
    model=ReQueryTAP(backbone,identity_dim=128).to(device).eval()
    with torch.no_grad():
        initialized=model.initialize(frames[0],query)
        direct_native=backbone(video=frames[1],state=clone_tracking_state(initialized.state.track_state))
        wrapped_native=model.step(frames[1],clone_lifecycle(initialized.state),allow_respawn=False)
        native_diffs={
          'tracks':float((direct_native[0]-wrapped_native.selected_tracks).abs().max().item()),
          'track_logits':float((direct_native[1]-wrapped_native.selected_track_logits).abs().max().item()),
          'visibility_logits':float((direct_native[2]-wrapped_native.selected_visibility_logits).abs().max().item()),
          'state':state_max_abs(direct_native[3],wrapped_native.state.track_state),
        }
        forced=target_yx[1].reshape(1,1,2)
        fresh_query=torch.cat([torch.zeros(1,1,1,device=device),forced],dim=-1)
        direct_fresh=backbone(video=frames[1],query_points=fresh_query)
        wrapped_fresh=model.step(frames[1],clone_lifecycle(initialized.state),force_respawn=True,forced_query_coordinate_yx=forced)
        fresh_diffs={
          'tracks':float((direct_fresh[0]-wrapped_fresh.selected_tracks).abs().max().item()),
          'track_logits':float((direct_fresh[1]-wrapped_fresh.selected_track_logits).abs().max().item()),
          'visibility_logits':float((direct_fresh[2]-wrapped_fresh.selected_visibility_logits).abs().max().item()),
          'state':state_max_abs(direct_fresh[3],wrapped_fresh.state.track_state),
        }
    # Gradient through the locator and differentiable fresh-query coordinate.
    model.locator.train(); model.backbone.eval(); torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    initialized=model.initialize(frames[0],query)
    output=model.step(frames[1],initialized.state,force_respawn=True)
    target=target_yx[1].reshape(1,1,2)
    patch_width=256//8
    patch_y=int(torch.clamp((target[0,0,0]/8).floor(),0,31).item());patch_x=int(torch.clamp((target[0,0,1]/8).floor(),0,31).item());patch_target=torch.tensor([patch_y*patch_width+patch_x],device=device)
    locator_loss=F.smooth_l1_loss(output.rebind_coordinate_yx,target)
    patch_loss=F.cross_entropy(output.rebind_patch_logits[:,0].float(),patch_target)
    gate_loss=F.binary_cross_entropy_with_logits(output.respawn_logit,torch.ones_like(output.respawn_logit))
    rollout_loss=F.smooth_l1_loss(output.selected_tracks[:,-1],target)
    loss=locator_loss+0.1*patch_loss+0.1*gate_loss+rollout_loss
    model.zero_grad(set_to_none=True);loss.backward()
    gradient_groups={
      'identity_projection':sum(float(p.grad.float().norm().item()) for p in model.locator.identity_projection.parameters() if p.grad is not None),
      'image_projection':sum(float(p.grad.float().norm().item()) for p in model.locator.image_projection.parameters() if p.grad is not None),
      'offset_head':sum(float(p.grad.float().norm().item()) for p in model.locator.offset_head.parameters() if p.grad is not None),
      'respawn_head':sum(float(p.grad.float().norm().item()) for p in model.locator.respawn_head.parameters() if p.grad is not None),
    }
    finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.locator.parameters())
    peak=int(torch.cuda.max_memory_allocated(device)); max_native=max(native_diffs.values()); max_fresh=max(fresh_diffs.values())
    gate={
      'native_off_exact':max_native==0.0,
      'forced_fresh_matches_standalone_le_1e6':max_fresh<=1e-6,
      'finite_loss':bool(torch.isfinite(loss).item()),
      'finite_locator_gradients':finite_gradients,
      'identity_gradient_nonzero':gradient_groups['identity_projection']>0,
      'image_gradient_nonzero':gradient_groups['image_projection']>0,
      'offset_gradient_nonzero':gradient_groups['offset_head']>0,
      'respawn_gradient_nonzero':gradient_groups['respawn_head']>0,
      'peak_allocated_below_22GiB':peak<22*1024**3,
      'single_query_contract':True,
    };gate['pass']=all(gate.values())
    result={'schema_version':'requerytap_real_architecture_smoke_v0','source':{'scene':str(args.scene.resolve()),'point_index':point,'split':'fit/train','locked_data_read':{'pointodyssey_model_validation':False,'pointodyssey_internal_holdout':False,'pointodyssey_test':False,'davis_method_eval':False,'kinetics_1144':False}},'native_off_max_abs':native_diffs,'forced_fresh_max_abs':fresh_diffs,'loss':float(loss.detach().item()),'loss_components':{'locator':float(locator_loss.detach().item()),'patch':float(patch_loss.detach().item()),'gate':float(gate_loss.detach().item()),'fresh_rollout_query_frame':float(rollout_loss.detach().item())},'gradient_norm_sums':gradient_groups,'locator_parameter_count':sum(p.numel() for p in model.locator.parameters()),'peak_allocated_bytes':peak,'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)),'gate':gate}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
