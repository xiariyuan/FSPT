#!/usr/bin/env python3
"""Fit-only training and frozen C1 gate for the ReQueryTAP exact-point locator."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mmp_tracker.requerytap import ExactPointGlobalLocator

DEFAULT_INDEX=Path('/gemini/code/FSPT_requerytap_stage/outputs/locator_cache_20260718/cache_index.json')
DEFAULT_MANIFEST=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_LOCATOR_TRAINING_MANIFEST_2026-07-18.json')


def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()



def cache_contract_sha256(manifest:dict)->str:
    payload={key:manifest[key] for key in ["seed","train_scenes","dev_scenes","frames","sampling","feature_cache"]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def model_state_sha256(module:torch.nn.Module)->str:
    h=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(np.asarray(value.shape,dtype=np.int64).tobytes());h.update(value.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Pair:
    scene_index:int
    frame_index:int
    point_index:int
    target_frame:int
    point_id:int


def load_records(index_path:Path)->tuple[dict[str,Any],list[dict[str,Any]]]:
    index=json.loads(index_path.read_text());records=[]
    for meta in index['records']:
        path=Path(meta['path'])
        if file_sha256(path)!=meta['sha256']:raise RuntimeError(f'cache hash mismatch: {path}')
        record=torch.load(path,map_location='cpu',weights_only=False)
        record['_meta']=meta;records.append(record)
    return index,records


def build_pairs(records:list[dict[str,Any]],split:str)->list[Pair]:
    pairs=[]
    for scene_index,record in enumerate(records):
        if record['_meta']['split']!=split:continue
        valid=torch.nonzero(record['target_valid'],as_tuple=False)
        frames=record['target_frames']
        point_ids=record['point_ids']
        for frame_index,point_index in valid.tolist():
            pairs.append(Pair(scene_index,frame_index,point_index,int(frames[frame_index]),int(point_ids[point_index])))
    return pairs


def fixed_dev_subset(pairs:list[Pair],records:list[dict[str,Any]],limit:int,seed:int)->list[Pair]:
    ranked=[]
    for pair in pairs:
        scene=records[pair.scene_index]['_meta']['scene']
        key=hashlib.sha256(f'{seed}|requerytap-locator-dev|{scene}|{pair.target_frame}|{pair.point_id}'.encode()).hexdigest()
        ranked.append((key,pair))
    ranked.sort(key=lambda item:item[0])
    return [pair for _,pair in ranked[:limit]]


def collate(records:list[dict[str,Any]],pairs:list[Pair],device:torch.device):
    query=[];tokens=[];target=[];scenes=[]
    for pair in pairs:
        record=records[pair.scene_index]
        query.append(record['query_features'][pair.point_index])
        tokens.append(record['target_tokens'][pair.frame_index])
        target.append(record['target_yx'][pair.frame_index,pair.point_index])
        scenes.append(record['_meta']['scene'])
    return torch.stack(query).to(device=device,dtype=torch.float32),torch.stack(tokens).to(device=device,dtype=torch.float32),torch.stack(target).to(device=device,dtype=torch.float32),scenes


def predict(locator:ExactPointGlobalLocator,query:torch.Tensor,tokens:torch.Tensor):
    identity=locator.encode_identity(query[:,None])
    batch=query.shape[0]
    coordinate,patch_logits,_=locator(identity,tokens,native_coordinate_yx=torch.zeros(batch,1,2,device=query.device),native_visibility_logit=torch.zeros(batch,1,1,device=query.device))
    return coordinate[:,0],patch_logits[:,0]


def target_patch(target_yx:torch.Tensor)->torch.Tensor:
    y=torch.clamp(torch.floor(target_yx[:,0]/8.0),0,31).long();x=torch.clamp(torch.floor(target_yx[:,1]/8.0),0,31).long()
    return y*32+x


def evaluate(locator,records,pairs,batch_size,device)->dict[str,Any]:
    locator.eval();errors=[];scene_errors:dict[str,list[float]]={}
    with torch.inference_mode():
        for start in range(0,len(pairs),batch_size):
            batch=pairs[start:start+batch_size];query,tokens,target,scenes=collate(records,batch,device);coordinate,_=predict(locator,query,tokens);error=torch.linalg.norm(coordinate-target,dim=-1).cpu().tolist();errors.extend(error)
            for scene,value in zip(scenes,error):scene_errors.setdefault(scene,[]).append(float(value))
    vector=np.asarray(errors,dtype=np.float64)
    per_scene={}
    for scene,values in scene_errors.items():
        x=np.asarray(values,dtype=np.float64);per_scene[scene]={'samples':int(x.size),'mean_error_px':float(x.mean()),'median_error_px':float(np.median(x)),'hit4':float((x<=4).mean()),'hit8':float((x<=8).mean()),'hit16':float((x<=16).mean()),'hit32':float((x<=32).mean())}
    return {'samples':int(vector.size),'mean_error_px':float(vector.mean()),'median_error_px':float(np.median(vector)),'hit4':float((vector<=4).mean()),'hit8':float((vector<=8).mean()),'hit16':float((vector<=16).mean()),'hit32':float((vector<=32).mean()),'per_scene':per_scene}


def bootstrap_ci(values:list[float],seed:int=17018,samples:int=10000)->dict[str,Any]:
    x=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed);means=x[rng.integers(0,x.size,size=(samples,x.size))].mean(1)
    return {'mean':float(x.mean()),'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),'samples':samples,'seed':seed}


def evaluate_gate(final_metrics:dict[str,Any],records:list[dict[str,Any]],manifest:dict[str,Any])->dict[str,Any]:
    dev_meta={r['_meta']['scene']:r['_meta']['raw_baseline'] for r in records if r['_meta']['split']=='dev'}
    reductions={scene:float(dev_meta[scene]['median_error_px']-metric['median_error_px']) for scene,metric in final_metrics['per_scene'].items()}
    values=list(reductions.values());ci=bootstrap_ci(values)
    gate_spec=manifest['gate_C1'];checks={
      'all_5_dev_scenes_complete':set(final_metrics['per_scene'])==set(manifest['dev_scenes']),
      'learned_median_error_le_12px':final_metrics['median_error_px']<=float(gate_spec['learned_median_error_px_max']),
      'learned_hit16_ge_0p60':final_metrics['hit16']>=float(gate_spec['learned_median_hit16_min']),
      'positive_error_reduction_5_of_5':sum(value>0 for value in values)>=int(gate_spec['positive_error_reduction_scenes_min']),
      'scene_bootstrap_error_reduction_ci_lower_positive':ci['lower']>float(gate_spec['scene_bootstrap_error_reduction_ci_lower_gt']),
      'worst_scene_error_reduction_ge_minus2':min(values)>=float(gate_spec['worst_scene_error_reduction_min']),
    };passed=all(checks.values())
    return {'raw_minus_learned_median_error_per_scene':reductions,'scene_bootstrap_error_reduction_ci':ci,'worst_scene_error_reduction':float(min(values)),'checks':checks,'pass':passed,'decision':'ALLOW_REQUERYTAP_PREDICTED_RESPAWN_ROLLOUT_GATE_C2' if passed else 'STOP_OR_REDESIGN_REQUERYTAP_LOCATOR','model_validation_allowed':False,'locked_data_allowed':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,default=DEFAULT_INDEX);ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());index,records=load_records(args.index)
    if index.get('manifest_cache_contract_sha256')!=cache_contract_sha256(manifest):raise RuntimeError('feature-cache contract differs from training manifest')
    if any(index['locked_data_read'].values()):raise RuntimeError('cache reports locked-data exposure')
    seed=int(manifest['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_float32_matmul_precision('high')
    device=torch.device(args.device);train_pairs=build_pairs(records,'train');dev_pairs=build_pairs(records,'dev');fixed_dev=fixed_dev_subset(dev_pairs,records,int(manifest['training']['early_stop_dev_samples']),seed)
    locator=ExactPointGlobalLocator(768,identity_dim=int(manifest['model']['identity_dim']),patch_size=8,image_size=(256,256),temperature=float(manifest['model']['temperature'])).to(device)
    for parameter in locator.parameters():parameter.requires_grad_(False)
    trainable=[]
    for module in [locator.identity_projection,locator.image_projection,locator.offset_head]:
        for parameter in module.parameters():parameter.requires_grad_(True);trainable.append(parameter)
    cfg=manifest['training'];optimizer=torch.optim.AdamW(trainable,lr=float(cfg['learning_rate']),weight_decay=float(cfg['weight_decay']))
    args.output.mkdir(parents=True,exist_ok=True);history=[];best_median=float('inf');best_epoch=-1;stale=0;checkpoint=args.output/'best_locator.pt'
    for epoch in range(int(cfg['epochs'])):
        locator.train();generator=torch.Generator().manual_seed(seed+epoch);indices=torch.randint(len(train_pairs),(int(cfg['samples_per_epoch']),),generator=generator).tolist();sampled=[train_pairs[i] for i in indices];loss_rows=[];ce_rows=[];coord_rows=[]
        for start in range(0,len(sampled),int(cfg['batch_size'])):
            batch=sampled[start:start+int(cfg['batch_size'])];query,tokens,target,_=collate(records,batch,device);optimizer.zero_grad(set_to_none=True);coordinate,logits=predict(locator,query,tokens);ce=F.cross_entropy(logits.float(),target_patch(target));coord=F.smooth_l1_loss(coordinate/256.0,target/256.0);loss=ce+10.0*coord
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(trainable,float(cfg['gradient_clip']));optimizer.step();loss_rows.append(float(loss.detach()));ce_rows.append(float(ce.detach()));coord_rows.append(float(coord.detach()))
        dev=evaluate(locator,records,fixed_dev,int(cfg['batch_size']),device);improved=dev['median_error_px']<best_median-1e-12
        if improved:
            best_median=dev['median_error_px'];best_epoch=epoch;stale=0;torch.save({'schema_version':'requerytap_locator_checkpoint_v0','epoch':epoch,'locator_state':locator.state_dict(),'manifest_sha256':file_sha256(args.manifest),'cache_index_sha256':file_sha256(args.index),'model_state_sha256':model_state_sha256(locator)},checkpoint)
        else:stale+=1
        row={'epoch':epoch,'train_loss':float(np.mean(loss_rows)),'train_patch_ce':float(np.mean(ce_rows)),'train_coordinate_loss':float(np.mean(coord_rows)),'fixed_dev':dev,'improved':improved,'stale_epochs':stale};history.append(row);print(json.dumps(row),flush=True)
        if stale>=int(cfg['early_stop_patience']):break
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);locator.load_state_dict(saved['locator_state']);locator.to(device);final=evaluate(locator,records,dev_pairs,int(cfg['batch_size']),device);gate=evaluate_gate(final,records,manifest)
    metrics={'schema_version':'requerytap_locator_training_result_v0','source':{'manifest':str(args.manifest.resolve()),'manifest_sha256':file_sha256(args.manifest),'cache_index':str(args.index.resolve()),'cache_index_sha256':file_sha256(args.index),'locked_data_read':index['locked_data_read']},'configuration':cfg,'train_pairs':len(train_pairs),'dev_pairs':len(dev_pairs),'fixed_dev_pairs':len(fixed_dev),'trainable_parameters':sum(p.numel() for p in trainable),'best_epoch':best_epoch,'history':history,'final_dev':final,'gate_C1':gate,'model_state_sha256':model_state_sha256(locator),'checkpoint_path':str(checkpoint.resolve()),'checkpoint_sha256':file_sha256(checkpoint)}
    path=args.output/'metrics.json';path.write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(path),'best_epoch':best_epoch,'final_dev':final,'gate_C1':gate,'model_state_sha256':metrics['model_state_sha256']},indent=2))

if __name__=='__main__':main()
