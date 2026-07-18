#!/usr/bin/env python3
"""Fit-dev audit of position-free TAPNext++ exact-point identity features."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from scripts.build_requerytap_locator_cache import load_image, sample_features
from scripts.audit_requerytap_multiview_identity import aggregate_scores, bootstrap_ci, summarize

DEFAULT_CKPT=Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_DATA=Path('/gemini/code/FSPT/datasets/pointodyssey/train')
DEFAULT_CACHE=Path('/gemini/code/FSPT_requerytap_stage/outputs/locator_cache_20260718')
DEFAULT_GATE_D=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_MULTIVIEW_IDENTITY_AUDIT_RESULT_2026-07-19.json')
DEFAULT_MANIFEST=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_POSITION_DECONFOUNDING_MANIFEST_2026-07-19.json')

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as s:
        for c in iter(lambda:s.read(8<<20),b''): h.update(c)
    return h.hexdigest()

def conv_tokens(conv:torch.nn.Conv2d,image:torch.Tensor)->torch.Tensor:
    with torch.inference_mode(): value=conv(image).flatten(2).transpose(1,2)
    return value[0]

def candidate_gate(rows:list[dict[str,Any]],variant:str,manifest:dict[str,Any])->dict[str,Any]:
    reductions=[]
    for row in rows:
        reductions.append(row['encoded_frame0']['median_error_px']-row['metrics'][variant]['median_error_px'])
    baseline=np.concatenate([np.asarray(r['encoded_frame0_error_rows'],float) for r in rows])
    value=np.concatenate([np.asarray(r['error_rows'][variant],float) for r in rows])
    x=np.asarray(reductions,float); spec=manifest['candidate_gate']; ci=bootstrap_ci(reductions,seed=int(manifest['seed']))
    checks={
      'all_5_scenes_complete':len(rows)==len(manifest['scenes']) and all(r['status']=='complete' for r in rows),
      'positive_reduction_5_of_5':int((x>0).sum())>=int(spec['positive_median_error_reduction_scenes_min']),
      'median_scene_reduction_ge_10px':float(np.median(x))>=float(spec['median_scene_error_reduction_px_min']),
      'scene_bootstrap_reduction_ci_lower_positive':ci['lower']>float(spec['scene_bootstrap_reduction_ci_lower_gt']),
      'aggregate_hit16_gain_ge_0p10':float((value<=16).mean()-(baseline<=16).mean())>=float(spec['aggregate_hit16_absolute_gain_min']),
      'worst_scene_reduction_ge_minus2':float(x.min())>=float(spec['worst_scene_error_reduction_min']),
    }
    return {'variant':variant,'per_scene_reduction_px':{r['scene']:float(v) for r,v in zip(rows,x)},'positive_scenes':int((x>0).sum()),'median_scene_reduction_px':float(np.median(x)),'scene_bootstrap_reduction_ci':ci,'aggregate_hit16_absolute_gain':float((value<=16).mean()-(baseline<=16).mean()),'worst_scene_reduction_px':float(x.min()),'checks':checks,'pass':all(checks.values())}

def incremental_gate(rows:list[dict[str,Any]],manifest:dict[str,Any])->dict[str,Any]:
    reductions=np.asarray([r['metrics']['conv_frame0']['median_error_px']-r['metrics']['conv_max8']['median_error_px'] for r in rows],float)
    spec=manifest['multiview_incremental_gate']; checks={
      'positive_reduction_4_of_5':int((reductions>0).sum())>=int(spec['positive_median_error_reduction_scenes_min']),
      'median_scene_reduction_ge_5px':float(np.median(reductions))>=float(spec['median_scene_error_reduction_px_min']),
      'worst_scene_reduction_ge_minus2':float(reductions.min())>=float(spec['worst_scene_error_reduction_min'])}
    return {'per_scene_reduction_px':{r['scene']:float(v) for r,v in zip(rows,reductions)},'positive_scenes':int((reductions>0).sum()),'median_scene_reduction_px':float(np.median(reductions)),'worst_scene_reduction_px':float(reductions.min()),'checks':checks,'pass':all(checks.values())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT); ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA); ap.add_argument('--cache-root',type=Path,default=DEFAULT_CACHE); ap.add_argument('--gate-d-result',type=Path,default=DEFAULT_GATE_D); ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text()); gate_d=json.loads(args.gate_d_result.read_text())
    if sha(args.gate_d_result)!=manifest['gate_D_result_sha256']: raise ValueError('Gate D result hash mismatch')
    gate_d_rows={r['scene']:r for r in gate_d['scene_results']}; device=torch.device(args.device)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False); state=payload['state_dict']; conv=torch.nn.Conv2d(3,768,8,8).to(device); conv.load_state_dict({'weight':state['tapnext.lin_proj.weight'],'bias':state['tapnext.lin_proj.bias']}); conv.eval(); del payload,state
    cy=(torch.arange(32,device=device,dtype=torch.float32)+.5)*8; cx=(torch.arange(32,device=device,dtype=torch.float32)+.5)*8; yy,xx=torch.meshgrid(cy,cx,indexing='ij'); centers=torch.stack([yy,xx],-1).reshape(-1,2)
    rows=[]
    for index,scene in enumerate(manifest['scenes']):
        record=torch.load(args.cache_root/f'{scene}.pt',map_location='cpu',weights_only=False); point_ids=record['point_ids'].numpy(); root=args.data_root/scene; h,w=map(int,record['image_size_original'].tolist())
        with np.load(root/'anno.npz',allow_pickle=False) as data:
            frames=manifest['query_memory_frames']; coords=np.asarray(data['trajs_2d'][frames][:,point_ids],np.float32); visible=np.asarray((data['visibs']&data['valids'])[frames][:,point_ids],bool)
        finite=np.isfinite(coords).all(-1); inside=(coords[...,0]>=8)&(coords[...,0]<w-8)&(coords[...,1]>=8)&(coords[...,1]<h-8); memory_valid=torch.from_numpy(visible&finite&inside).T.to(device)
        memory=[]
        for fi,frame in enumerate(frames):
            image,_=load_image(root/'rgbs'/f'rgb_{frame:05d}.jpg'); tokens=conv_tokens(conv,image.to(device)); xy=coords[fi]*np.asarray([256.0/w,256.0/h],np.float32); yx=torch.from_numpy(xy[...,::-1].copy()).to(device); memory.append(sample_features(tokens,yx))
        memory=torch.stack(memory,1); error_rows={k:[] for k in ('conv_frame0','conv_mean8','conv_max8','conv_logsumexp8')}
        for ti,frame in enumerate(manifest['target_frames']):
            ids=torch.nonzero(record['target_valid'][ti],as_tuple=False).flatten()
            if ids.numel()==0: continue
            image,_=load_image(root/'rgbs'/f'rgb_{frame:05d}.jpg'); target_tokens=conv_tokens(conv,image.to(device)); scores=aggregate_scores(memory[ids],memory_valid[ids],target_tokens); target=record['target_yx'][ti,ids].to(device).float()
            mapping={'conv_frame0':'frame0','conv_mean8':'mean8','conv_max8':'max8','conv_logsumexp8':'logsumexp8'}
            for out_key,score_key in mapping.items():
                pred=centers[scores[score_key].argmax(-1)]; error=torch.linalg.norm(pred-target,dim=-1); error_rows[out_key].extend(error.cpu().tolist())
        metrics={k:summarize(v) for k,v in error_rows.items()}; encoded=gate_d_rows[scene]
        row={'scene':scene,'status':'complete','visible_memory_count':{'minimum':int(memory_valid.sum(1).min()),'median':float(memory_valid.sum(1).float().median()),'maximum':int(memory_valid.sum(1).max())},'encoded_frame0':encoded['metrics']['frame0'],'encoded_frame0_error_rows':encoded['error_rows']['frame0'],'metrics':metrics,'error_rows':error_rows}; rows.append(row); print(json.dumps({'index':index,'scene':scene,'metrics':metrics}),flush=True)
    candidates={v:candidate_gate(rows,v,manifest) for v in manifest['candidate_variants']}; inc=incremental_gate(rows,manifest)
    if candidates['conv_max8']['pass'] and inc['pass']: decision='ALLOW_POSITION_FREE_MULTIVIEW_LOCATOR'
    elif candidates['conv_frame0']['pass']: decision='ALLOW_POSITION_FREE_SINGLE_VIEW_LOCATOR'
    else: decision='CLOSE_TAPNEXT_PATCH_IDENTITY_ROUTE'
    gate={'candidates':candidates,'multiview_incremental':inc,'pass':decision!='CLOSE_TAPNEXT_PATCH_IDENTITY_ROUTE','decision':decision,'training_allowed':False,'model_validation_allowed':False}
    output={'schema_version':'requerytap_position_deconfounding_audit_v0','audit_status':'fit-dev representation audit; no deployable claim','source':{'manifest':str(args.manifest.resolve()),'manifest_sha256':sha(args.manifest),'gate_D_result':str(args.gate_d_result.resolve()),'gate_D_result_sha256':sha(args.gate_d_result),'locked_data_read':manifest['locked_data_read']},'scene_results':rows,'gate_E':gate}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n'); print(json.dumps({'output':str(args.output),'gate_E':gate},indent=2))
if __name__=='__main__': main()
