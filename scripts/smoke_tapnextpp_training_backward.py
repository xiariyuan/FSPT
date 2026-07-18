#!/usr/bin/env python3
"""One-step TAPNext++ PyTorch backward smoke on a real fit clip."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DEFAULT_REPO = Path('/gemini/code/FSPT/external/tapnextpp/repo')
DEFAULT_CKPT = Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_SCENE = Path('/gemini/code/FSPT/datasets/pointodyssey/train/ani13_new_f')


def load_clip(scene: Path, frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    for index in range(frames):
        image = np.asarray(Image.open(scene/'rgbs'/f'rgb_{index:05d}.jpg').convert('RGB'), dtype=np.uint8)
        tensor = torch.from_numpy(image.copy()).float().permute(2,0,1)[None]
        tensor = F.interpolate(tensor, (256,256), mode='bilinear', align_corners=False)
        images.append(tensor[0].permute(1,2,0) / 127.5 - 1.0)
    video = torch.stack(images, dim=0)[None]
    with np.load(scene/'anno.npz', allow_pickle=False) as data:
        coords = np.asarray(data['trajs_2d'][:frames], dtype=np.float32)
        visible = np.asarray(data['visibs'][:frames] & data['valids'][:frames], dtype=bool)
    height, width = Image.open(scene/'rgbs'/'rgb_00000.jpg').size[1], Image.open(scene/'rgbs'/'rgb_00000.jpg').size[0]
    valid = visible.all(0) & np.isfinite(coords).all((0,2))
    valid &= (coords[...,0]>=64).all(0)&(coords[...,0]<width-64).all(0)
    valid &= (coords[...,1]>=64).all(0)&(coords[...,1]<height-64).all(0)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        raise RuntimeError('no valid point for gradient smoke')
    point = int(ids[len(ids)//2])
    xy = coords[:,point]
    scaled_xy = xy * np.asarray([256.0/width, 256.0/height], dtype=np.float32)
    target_yx = torch.from_numpy(scaled_xy[:, ::-1].copy())[None,:,None]
    query = torch.tensor([[[0.0, float(scaled_xy[0,1]), float(scaled_xy[0,0])]]])
    target_vis = torch.ones(1,frames,1,1)
    return video, query, target_yx, target_vis


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT)
    ap.add_argument('--scene',type=Path,default=DEFAULT_SCENE)
    ap.add_argument('--frames',type=int,default=4)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if 'test' in args.scene.parts:
        raise ValueError('test data is forbidden')
    sys.path.insert(0,str(args.repo))
    from tapnet.tapnext.tapnext_torch import TAPNext

    video, query, target_yx, target_vis = load_clip(args.scene,args.frames)
    device=torch.device(args.device)
    model=TAPNext(image_size=(256,256),use_checkpointing=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict({k.replace('tapnext.',''):v for k,v in payload['state_dict'].items()})
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules=[model.blocks[-1],model.encoder_norm,model.coordinate_head,model.visible_head]
    for module in trainable_modules:
        for parameter in module.parameters(): parameter.requires_grad_(True)
    model.point_query_token.requires_grad_(True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=1e-6,weight_decay=0.0)
    video=video.to(device)
    query=query.to(device)
    target_yx=target_yx.to(device)
    target_vis=target_vis.to(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    before={name:param.detach().clone() for name,param in model.named_parameters() if param.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    tracks,_,visibility_logits,_=model(video=video,query_points=query)
    coord_loss=F.smooth_l1_loss(tracks.float(),target_yx.float())
    vis_loss=F.binary_cross_entropy_with_logits(visibility_logits.float(),target_vis.float())
    loss=coord_loss+vis_loss
    loss.backward()
    grad_groups={
        'last_block':sum(float(p.grad.float().norm().item()) for p in model.blocks[-1].parameters() if p.grad is not None),
        'coordinate_head':sum(float(p.grad.float().norm().item()) for p in model.coordinate_head.parameters() if p.grad is not None),
        'visibility_head':sum(float(p.grad.float().norm().item()) for p in model.visible_head.parameters() if p.grad is not None),
        'point_query_token':float(model.point_query_token.grad.float().norm().item()) if model.point_query_token.grad is not None else 0.0,
    }
    finite_loss=bool(torch.isfinite(loss).item())
    finite_grads=all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in trainable)
    optimizer.step()
    changes={name:float((param.detach()-before[name]).float().norm().item()) for name,param in model.named_parameters() if param.requires_grad}
    changed_count=sum(value>0.0 for value in changes.values())
    peak_allocated=int(torch.cuda.max_memory_allocated(device))
    peak_reserved=int(torch.cuda.max_memory_reserved(device))
    gate={
        'forward_backward_optimizer_step_complete':True,
        'finite_loss':finite_loss,
        'finite_gradients':finite_grads,
        'last_block_gradient_nonzero':grad_groups['last_block']>0.0,
        'coordinate_head_gradient_nonzero':grad_groups['coordinate_head']>0.0,
        'visibility_head_gradient_nonzero':grad_groups['visibility_head']>0.0,
        'parameter_changed':changed_count>0,
        'peak_allocated_below_22GiB':peak_allocated < 22*1024**3,
    }
    gate['pass']=all(gate.values())
    output={
        'schema_version':'requerytap_tapnextpp_backward_smoke_v0',
        'source':{'scene':str(args.scene.resolve()),'split':'fit/train','frames':args.frames,'checkpoint':str(args.checkpoint.resolve()),'locked_data_read':{'pointodyssey_model_validation':False,'pointodyssey_internal_holdout':False,'pointodyssey_test':False,'davis_method_eval':False,'kinetics_1144':False}},
        'trainable_parameter_count':sum(p.numel() for p in trainable),
        'loss':float(loss.detach().item()),
        'coordinate_loss':float(coord_loss.detach().item()),
        'visibility_loss':float(vis_loss.detach().item()),
        'gradient_norm_sums':grad_groups,
        'changed_parameter_tensors':changed_count,
        'trainable_parameter_tensors':len(changes),
        'maximum_parameter_change_norm':max(changes.values()),
        'peak_allocated_bytes':peak_allocated,
        'peak_reserved_bytes':peak_reserved,
        'gate':gate,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
