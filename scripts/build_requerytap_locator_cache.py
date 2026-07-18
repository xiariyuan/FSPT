#!/usr/bin/env python3
"""Build fit-only TAPNext++ patch-token cache for ReQueryTAP locator training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DEFAULT_CKPT=Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_DATA=Path('/gemini/code/FSPT/datasets/pointodyssey/train')
DEFAULT_MANIFEST=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_LOCATOR_TRAINING_MANIFEST_2026-07-18.json')


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def select_points(scene:str,valid:np.ndarray,limit:int,seed:int)->np.ndarray:
    ids=np.flatnonzero(valid)
    ranked=sorted((hashlib.sha256(f'{seed}|{scene}|{int(i)}'.encode()).hexdigest(),int(i)) for i in ids)
    return np.asarray([i for _,i in ranked[:limit]],dtype=np.int64)


def load_image(path:Path)->tuple[torch.Tensor,tuple[int,int]]:
    image=np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)
    h,w=image.shape[:2]
    tensor=torch.from_numpy(image.copy()).float().permute(2,0,1)[None]
    tensor=F.interpolate(tensor,(256,256),mode='bilinear',align_corners=False)
    tensor=tensor/127.5-1.0
    return tensor,(h,w)


def patch_tokens(conv,pos,image:torch.Tensor)->torch.Tensor:
    with torch.inference_mode():
        value=conv(image).flatten(2).transpose(1,2)+pos
    return value[0]


def sample_features(tokens:torch.Tensor,yx:torch.Tensor)->torch.Tensor:
    feature=tokens.transpose(0,1).reshape(1,tokens.shape[1],32,32)
    y=yx[:,0]/256.0*2.0-1.0;x=yx[:,1]/256.0*2.0-1.0
    grid=torch.stack([x,y],dim=-1)[None,:,None]
    return F.grid_sample(feature,grid,mode='bilinear',align_corners=False)[0,:,:,0].transpose(0,1)


def raw_baseline(query:torch.Tensor,target_tokens:torch.Tensor,target_yx:torch.Tensor,valid:torch.Tensor)->dict:
    centers_y=(torch.arange(32,dtype=torch.float32)+.5)*8;centers_x=(torch.arange(32,dtype=torch.float32)+.5)*8
    yy,xx=torch.meshgrid(centers_y,centers_x,indexing='ij');centers=torch.stack([yy,xx],dim=-1).reshape(-1,2)
    q=F.normalize(query.float(),dim=-1);errors=[];per_frame=[]
    for frame in range(target_tokens.shape[0]):
        ids=torch.nonzero(valid[frame], as_tuple=False).flatten()
        if ids.numel()==0:continue
        t=F.normalize(target_tokens[frame].float(),dim=-1)
        prediction=centers[(q[ids]@t.T).argmax(dim=-1)]
        error=torch.linalg.norm(prediction-target_yx[frame,ids].float(),dim=-1)
        errors.append(error);per_frame.append({'frame_offset':frame,'samples':int(error.numel()),'mean_error':float(error.mean()),'median_error':float(error.median()),'hit16':float((error<=16).float().mean())})
    all_error=torch.cat(errors)
    return {'samples':int(all_error.numel()),'mean_error_px':float(all_error.mean()),'median_error_px':float(all_error.median()),'hit4':float((all_error<=4).float().mean()),'hit8':float((all_error<=8).float().mean()),'hit16':float((all_error<=16).float().mean()),'hit32':float((all_error<=32).float().mean()),'per_frame':per_frame}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT);ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA);ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());scenes=manifest['train_scenes']+manifest['dev_scenes'];target_frames=manifest['frames']['target_frames'];limit=int(manifest['sampling']['points_per_scene']);margin=float(manifest['sampling']['image_margin_px']);seed=int(manifest['seed'])
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);state=payload['state_dict'];conv=torch.nn.Conv2d(3,768,8,8).to(args.device);conv.load_state_dict({'weight':state['tapnext.lin_proj.weight'],'bias':state['tapnext.lin_proj.bias']});conv.eval();pos=state['tapnext.image_pos_emb'].to(args.device);del payload,state
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for index,scene in enumerate(scenes):
        root=args.data_root/scene;query_image,size=load_image(root/'rgbs'/'rgb_00000.jpg');h,w=size
        with np.load(root/'anno.npz',allow_pickle=False) as data:
            coords=np.asarray(data['trajs_2d'][[0]+target_frames],dtype=np.float32);visible=np.asarray((data['visibs']&data['valids'])[[0]+target_frames],dtype=bool)
        finite=np.isfinite(coords).all(-1);inside=(coords[...,0]>=margin)&(coords[...,0]<w-margin)&(coords[...,1]>=margin)&(coords[...,1]<h-margin)
        valid=visible&finite&inside;eligible=valid[0]&valid[1:].any(0);points=select_points(scene,eligible,limit,seed)
        if points.size<limit:raise RuntimeError(f'{scene}: only {points.size} eligible points')
        scale=np.asarray([256.0/w,256.0/h],dtype=np.float32);scaled_xy=coords[:,points]*scale;scaled_yx=scaled_xy[...,::-1].copy()
        query_tokens=patch_tokens(conv,pos,query_image.to(args.device));query_yx=torch.from_numpy(scaled_yx[0]).to(args.device);query_features=sample_features(query_tokens,query_yx).half().cpu()
        target_token_rows=[]
        for frame in target_frames:
            image,_=load_image(root/'rgbs'/f'rgb_{frame:05d}.jpg');target_token_rows.append(patch_tokens(conv,pos,image.to(args.device)).half().cpu())
        target_tokens=torch.stack(target_token_rows);target_yx=torch.from_numpy(scaled_yx[1:]);target_valid=torch.from_numpy(valid[1:,points])
        baseline=raw_baseline(query_features,target_tokens,target_yx,target_valid)
        record={'schema_version':'requerytap_locator_scene_cache_v0','scene':scene,'split':'train' if scene in manifest['train_scenes'] else 'dev','point_ids':torch.from_numpy(points),'query_features':query_features,'target_tokens':target_tokens,'target_yx':target_yx,'target_valid':target_valid,'target_frames':torch.tensor(target_frames,dtype=torch.int64),'image_size_original':torch.tensor([h,w]),'raw_baseline':baseline}
        path=args.output/f'{scene}.pt';torch.save(record,path);records.append({'scene':scene,'split':record['split'],'path':str(path.resolve()),'sha256':sha256(path),'size_bytes':path.stat().st_size,'samples':baseline['samples'],'raw_baseline':baseline})
        print(json.dumps({'index':index,'scene':scene,'split':record['split'],'points':int(points.size),'samples':baseline['samples'],'raw_median':baseline['median_error_px'],'raw_hit16':baseline['hit16']}),flush=True)
    index={'schema_version':'requerytap_locator_cache_index_v0','manifest':str(args.manifest.resolve()),'manifest_sha256':sha256(args.manifest),'checkpoint':str(args.checkpoint.resolve()),'checkpoint_size_bytes':args.checkpoint.stat().st_size,'records':records,'locked_data_read':manifest['locked_data_read']}
    path=args.output/'cache_index.json';path.write_text(json.dumps(index,indent=2,sort_keys=True)+'\n');print(json.dumps({'index':str(path),'sha256':sha256(path),'records':len(records)},indent=2))

if __name__=='__main__':main()
