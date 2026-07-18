#!/usr/bin/env python3
"""Fit-dev information audit for multi-view exact-point identity memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.build_requerytap_locator_cache import load_image, patch_tokens, sample_features

DEFAULT_CKPT=Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_DATA=Path('/gemini/code/FSPT/datasets/pointodyssey/train')
DEFAULT_CACHE=Path('/gemini/code/FSPT_requerytap_stage/outputs/locator_cache_20260718')
DEFAULT_MANIFEST=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_MULTIVIEW_IDENTITY_AUDIT_MANIFEST_2026-07-18.json')


def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def aggregate_scores(memory:torch.Tensor,memory_valid:torch.Tensor,target_tokens:torch.Tensor)->dict[str,torch.Tensor]:
    """Return P,N score matrices for four frozen temporal aggregators."""
    if memory.ndim!=3 or target_tokens.ndim!=2 or memory_valid.shape!=memory.shape[:2]:raise ValueError('invalid multiview shapes')
    normalized=F.normalize(memory.float(),dim=-1);target=F.normalize(target_tokens.float(),dim=-1)
    pair=torch.einsum('pmd,nd->pmn',normalized,target)
    mask=memory_valid.bool()
    if not bool(mask.any(dim=1).all()):raise ValueError('every point needs one visible memory view')
    masked=pair.masked_fill(~mask[...,None],-1e4)
    count=mask.sum(dim=1,keepdim=True).clamp_min(1).to(normalized.dtype)
    mean_identity=(normalized*mask[...,None]).sum(dim=1)/count
    mean_identity=F.normalize(mean_identity,dim=-1)
    return {
      'frame0':pair[:,0],
      'mean8':mean_identity@target.T,
      'max8':masked.max(dim=1).values,
      'logsumexp8':torch.logsumexp(masked*10.0,dim=1)/10.0,
    }


def summarize(errors:list[float])->dict[str,float|int]:
    x=np.asarray(errors,dtype=np.float64)
    return {'samples':int(x.size),'mean_error_px':float(x.mean()),'median_error_px':float(np.median(x)),'hit4':float((x<=4).mean()),'hit8':float((x<=8).mean()),'hit16':float((x<=16).mean()),'hit32':float((x<=32).mean())}


def bootstrap_ci(values:list[float],seed:int=17018,samples:int=10000)->dict[str,float|int]:
    x=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed);means=x[rng.integers(0,x.size,size=(samples,x.size))].mean(1)
    return {'mean':float(x.mean()),'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),'samples':samples,'seed':seed}


def evaluate_gate(scene_results:list[dict[str,Any]],manifest:dict[str,Any])->dict[str,Any]:
    primary=manifest['primary_variant'];reductions=[]
    for row in scene_results:reductions.append(row['metrics']['frame0']['median_error_px']-row['metrics'][primary]['median_error_px'])
    frame_errors=np.concatenate([np.asarray(row['error_rows']['frame0'],dtype=np.float64) for row in scene_results]);primary_errors=np.concatenate([np.asarray(row['error_rows'][primary],dtype=np.float64) for row in scene_results])
    hit_gain=float((primary_errors<=16).mean()-(frame_errors<=16).mean());spec=manifest['gate_D'];ci=bootstrap_ci(reductions)
    checks={
      'all_5_scenes_complete':len(scene_results)==len(manifest['scenes']) and all(row['status']=='complete' for row in scene_results),
      'positive_reduction_5_of_5':sum(value>0 for value in reductions)>=int(spec['positive_median_error_reduction_scenes_min']),
      'median_scene_reduction_ge_10px':float(np.median(reductions))>=float(spec['median_scene_error_reduction_px_min']),
      'scene_bootstrap_reduction_ci_lower_positive':ci['lower']>float(spec['scene_bootstrap_reduction_ci_lower_gt']),
      'aggregate_hit16_gain_ge_0p10':hit_gain>=float(spec['aggregate_hit16_absolute_gain_min']),
      'worst_scene_reduction_ge_minus2':min(reductions)>=float(spec['worst_scene_error_reduction_min']),
    };passed=all(checks.values())
    return {'primary_variant':primary,'median_error_reduction_per_scene':{row['scene']:value for row,value in zip(scene_results,reductions)},'median_scene_reduction_px':float(np.median(reductions)),'scene_bootstrap_reduction_ci':ci,'aggregate_hit16_absolute_gain':hit_gain,'worst_scene_reduction_px':float(min(reductions)),'checks':checks,'pass':passed,'decision':manifest['decision_if_pass'] if passed else manifest['decision_if_fail'],'training_allowed':False,'model_validation_allowed':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT);ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA);ap.add_argument('--cache-root',type=Path,default=DEFAULT_CACHE);ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());device=torch.device(args.device)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);state=payload['state_dict'];conv=torch.nn.Conv2d(3,768,8,8).to(device);conv.load_state_dict({'weight':state['tapnext.lin_proj.weight'],'bias':state['tapnext.lin_proj.bias']});conv.eval();pos=state['tapnext.image_pos_emb'].to(device);del payload,state
    centers_y=(torch.arange(32,device=device,dtype=torch.float32)+.5)*8;centers_x=(torch.arange(32,device=device,dtype=torch.float32)+.5)*8;yy,xx=torch.meshgrid(centers_y,centers_x,indexing='ij');centers=torch.stack([yy,xx],dim=-1).reshape(-1,2)
    scene_results=[]
    for index,scene in enumerate(manifest['scenes']):
        record=torch.load(args.cache_root/f'{scene}.pt',map_location='cpu',weights_only=False);point_ids=record['point_ids'].numpy();root=args.data_root/scene
        frames=manifest['query_memory_frames']
        with np.load(root/'anno.npz',allow_pickle=False) as data:
            coords=np.asarray(data['trajs_2d'][frames][:,point_ids],dtype=np.float32);visible=np.asarray((data['visibs']&data['valids'])[frames][:,point_ids],dtype=bool)
        h,w=map(int,record['image_size_original'].tolist());finite=np.isfinite(coords).all(-1);inside=(coords[...,0]>=8)&(coords[...,0]<w-8)&(coords[...,1]>=8)&(coords[...,1]<h-8);memory_valid=torch.from_numpy(visible&finite&inside).T.to(device)
        memory_rows=[]
        for frame in frames:
            image,_=load_image(root/'rgbs'/f'rgb_{frame:05d}.jpg');tokens=patch_tokens(conv,pos,image.to(device));xy=coords[frame-frames[0]]*np.asarray([256.0/w,256.0/h],dtype=np.float32);yx=torch.from_numpy(xy[...,::-1].copy()).to(device);memory_rows.append(sample_features(tokens,yx))
        memory=torch.stack(memory_rows,dim=1)
        frame0_diff=float((memory[:,0].half().cpu()-record['query_features']).abs().max().item())
        error_rows={key:[] for key in manifest['variants']}
        for target_index in range(record['target_tokens'].shape[0]):
            ids=torch.nonzero(record['target_valid'][target_index],as_tuple=False).flatten()
            if ids.numel()==0:continue
            score=aggregate_scores(memory[ids],memory_valid[ids],record['target_tokens'][target_index].to(device));target=record['target_yx'][target_index,ids].to(device).float()
            for key,value in score.items():
                prediction=centers[value.argmax(dim=-1)];error=torch.linalg.norm(prediction-target,dim=-1);error_rows[key].extend(error.cpu().tolist())
        metrics={key:summarize(values) for key,values in error_rows.items()}
        row={'scene':scene,'status':'complete','frame0_feature_max_abs_vs_cache':frame0_diff,'visible_memory_count':{'minimum':int(memory_valid.sum(1).min()),'median':float(memory_valid.sum(1).float().median()),'maximum':int(memory_valid.sum(1).max())},'metrics':metrics,'error_rows':error_rows};scene_results.append(row);print(json.dumps({'index':index,'scene':scene,'frame0_diff':frame0_diff,'metrics':metrics}),flush=True)
    gate=evaluate_gate(scene_results,manifest);output={'schema_version':'requerytap_multiview_identity_audit_v0','audit_status':'fit-dev GT-visible memory information audit; not deployable','source':{'manifest':str(args.manifest.resolve()),'manifest_sha256':file_sha256(args.manifest),'checkpoint':str(args.checkpoint.resolve()),'checkpoint_size_bytes':args.checkpoint.stat().st_size,'locked_data_read':manifest['locked_data_read']},'scene_results':scene_results,'gate_D':gate}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(args.output),'gate_D':gate},indent=2))

if __name__=='__main__':main()
