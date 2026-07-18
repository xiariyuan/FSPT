#!/usr/bin/env python3
"""Causal fresh-query oracle for track-token re-instantiation.

At the first visible frame after a deterministic synthetic gap, create a new
TAPNext++ state from native, clean-teacher, or GT coordinates.  No recurrent
state is copied or injected.  The query frame is excluded from the primary
future-rollout metric.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_tapnextpp_query_state_bridge_static import (
    clone_tracking_state,
    file_sha256,
    load_model,
    model_step,
    prepare_frame,
    record_prediction,
    scale_query,
    summarize,
)
from audit_tapnextpp_query_state_bridge_dynamic import (
    choose_dynamic_query,
    load_rgb,
    target_yx,
)

DEFAULT_REPO=Path('/gemini/code/FSPT/external/tapnextpp/repo')
DEFAULT_CKPT=Path('/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt')
DEFAULT_DATA_ROOT=Path('/gemini/code/FSPT/datasets/pointodyssey/train')
DEFAULT_MANIFEST=Path('/gemini/code/FSPT_requerytap_stage/docs/generated/REQUERYTAP_FRESH_QUERY_ORACLE_MANIFEST_2026-07-18.json')


def atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)


def bootstrap_mean_ci(values:list[float],seed:int=17018,samples:int=10000)->dict[str,float|int|None]:
    v=np.asarray(values,dtype=np.float64)
    if v.size==0:return {'mean':None,'lower':None,'upper':None,'samples':0,'seed':seed}
    rng=np.random.default_rng(seed)
    means=v[rng.integers(0,v.size,size=(samples,v.size))].mean(1)
    return {'mean':float(v.mean()),'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),'samples':samples,'seed':seed}


def make_query_yx(yx:torch.Tensor)->torch.Tensor:
    value=yx.detach().float().reshape(2)
    return torch.cat([torch.zeros(1,device=value.device),value])[None,None]


def evaluate_gate(rows:list[dict[str,Any]],expected:int)->dict[str,Any]:
    complete=[r for r in rows if r.get('status')=='complete']
    gains=[float(r['delta']['fresh_GT_gain_vs_native']) for r in complete]
    frames=[float(r['delta']['fresh_GT_improved_frame_fraction']) for r in complete]
    all_complete=len(complete)==expected
    positive=sum(x>0 for x in gains)
    median=float(np.median(gains)) if gains else None
    median_frames=float(np.median(frames)) if frames else None
    worst=float(min(gains)) if gains else None
    ci=bootstrap_mean_ci(gains)
    checks={
      'all_scenes_complete':all_complete,
      'fresh_GT_positive_fraction_ge_0p80':all_complete and positive/expected>=.80,
      'fresh_GT_median_gain_ge_2px':all_complete and median is not None and median>=2.0,
      'fresh_GT_scene_bootstrap_ci_lower_positive':all_complete and ci['lower'] is not None and float(ci['lower'])>0,
      'fresh_GT_median_improved_frame_fraction_ge_0p75':all_complete and median_frames is not None and median_frames>=.75,
      'fresh_GT_no_scene_regression_worse_than_1px':all_complete and worst is not None and worst>=-1.0,
    }
    passed=all(checks.values())
    return {'expected_scenes':expected,'complete_scenes':len(complete),'positive_scenes':positive,'median_gain_px_256':median,'mean_gain_scene_bootstrap_ci':ci,'median_improved_frame_fraction':median_frames,'worst_scene_gain_px_256':worst,'checks':checks,'pass':passed,'decision':'ALLOW_REQUERYTAP_ARCHITECTURE_IMPLEMENTATION' if passed else 'CLOSE_REQUERYTAP_RESPAWN_ROUTE','training_allowed':False,'model_validation_allowed':False}


def run_scene(model,scene:Path,protocol:dict[str,Any],device:str)->dict[str,Any]:
    pre=int(protocol['pre_frames']); gap=int(protocol['gap_frames']); post=int(protocol['post_frames_including_query'])
    total=pre+gap+post; bridge=pre+gap
    point,coords,selection=choose_dynamic_query(scene,start_frame=int(protocol['start_frame']),pre_frames=pre,gap_frames=gap,post_frames=post,margin_px=float(protocol['margin_px']),motion_quantile=float(protocol['motion_quantile']))
    if int(selection['eligible_points'])<int(protocol['minimum_eligible_points']):raise RuntimeError('scene violates frozen eligibility')
    images=[load_rgb(scene,i)[0] for i in range(total)]
    shape=images[0].shape[:2]
    frames=[prepare_frame(image,device=device) for image in images]
    fixed_mean=np.broadcast_to(images[0].reshape(-1,3).mean(0).round().astype(np.uint8),images[0].shape).copy()
    blackout=prepare_frame(fixed_mean,device=device)
    initial_query=scale_query(coords[0],shape).to(device)
    teacher_state=None; student_state=None
    for rel in range(bridge):
        _,_,teacher_state=model_step(model,frames[rel],query=initial_query if teacher_state is None else None,state=teacher_state)
        inp=frames[rel] if rel<pre else blackout
        _,_,student_state=model_step(model,inp,query=initial_query if student_state is None else None,state=student_state)
    if teacher_state is None or student_state is None:raise RuntimeError('failed bridge states')
    # First post-gap frame supplies causal coordinates used to instantiate new states.
    native_first,native_vis,native_after=model_step(model,frames[bridge],state=student_state)
    teacher_first,_,_=model_step(model,frames[bridge],state=teacher_state)
    gt_yx=target_yx(coords[bridge],shape,device)
    queries={
      'fresh_native_coordinate_query':make_query_yx(native_first[0,-1,0]),
      'fresh_teacher_coordinate_query':make_query_yx(teacher_first[0,-1,0]),
      'fresh_GT_coordinate_query':make_query_yx(gt_yx),
    }
    branches={'native_student':native_after}
    query_frame_rows={'native_student':record_prediction(native_first,native_vis,gt_yx)}
    for name,query in queries.items():
        tracks,vis,state=model_step(model,frames[bridge],query=query,state=None)
        branches[name]=state
        query_frame_rows[name]=record_prediction(tracks,vis,gt_yx)
    rows={name:[] for name in branches}
    # Exclude the query frame; score only future frames.
    for rel in range(bridge+1,total):
        target=target_yx(coords[rel],shape,device)
        for name in tuple(branches):
            tracks,vis,branches[name]=model_step(model,frames[rel],state=branches[name])
            rows[name].append(record_prediction(tracks,vis,target))
    summary={name:summarize(values) for name,values in rows.items()}
    native=summary['native_student']['mean_error_px_256']
    deltas={}
    for name in queries:
        gain=native-summary[name]['mean_error_px_256']
        deltas[name+'_gain_vs_native']=gain
        deltas[name+'_improved_frame_fraction']=float(np.mean([c['error_px_256']<n['error_px_256'] for c,n in zip(rows[name],rows['native_student'])]))
    return {'scene':scene.name,'status':'complete','point_index':point,'selection':selection,'query_frame_rows':query_frame_rows,'future_summary':summary,'delta':{'fresh_native_gain_vs_native':deltas['fresh_native_coordinate_query_gain_vs_native'],'fresh_native_improved_frame_fraction':deltas['fresh_native_coordinate_query_improved_frame_fraction'],'fresh_teacher_gain_vs_native':deltas['fresh_teacher_coordinate_query_gain_vs_native'],'fresh_teacher_improved_frame_fraction':deltas['fresh_teacher_coordinate_query_improved_frame_fraction'],'fresh_GT_gain_vs_native':deltas['fresh_GT_coordinate_query_gain_vs_native'],'fresh_GT_improved_frame_fraction':deltas['fresh_GT_coordinate_query_improved_frame_fraction']},'future_frame_rows':rows}


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,default=DEFAULT_REPO); ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT); ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT); ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST); ap.add_argument('--device',default='cuda'); ap.add_argument('--skip-checkpoint-sha256',action='store_true'); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text()); scenes=list(manifest['scenes']); protocol=manifest['protocol']
    if len(scenes)!=17 or len(set(scenes))!=17:raise ValueError('manifest must contain 17 unique scenes')
    source_file=args.repo/'tapnet/tapnext/tapnext_torch.py'
    output={'schema_version':'requerytap_fresh_query_oracle_v0','audit_status':'fit-only causal query-respawn oracle; not learned or deployable','source':{'official_repository':'https://github.com/google-deepmind/tapnet','official_source_commit':'989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc','mirror_repo_commit':'4f3c01d15a5d3ed14639971d79bffef7748fc96e','implementation_source_sha256':file_sha256(source_file),'checkpoint':str(args.checkpoint.resolve()),'checkpoint_size_bytes':args.checkpoint.stat().st_size,'checkpoint_sha256':None if args.skip_checkpoint_sha256 else file_sha256(args.checkpoint),'manifest':str(args.manifest.resolve()),'manifest_sha256':file_sha256(args.manifest),'locked_data_read':manifest['locked_data_read']},'protocol':protocol,'scenes':scenes,'scene_results':[],'gate':None}
    atomic_json(args.output,output); model=load_model(args.repo,args.checkpoint,device=args.device)
    for index,name in enumerate(scenes):
        print(json.dumps({'scene':name,'index':index,'status':'start'}),flush=True)
        try: result=run_scene(model,args.data_root/name,protocol,args.device)
        except Exception as error: result={'scene':name,'status':'error','error_type':type(error).__name__,'error':str(error)}
        output['scene_results'].append(result); output['gate']=evaluate_gate(output['scene_results'],len(scenes)); atomic_json(args.output,output)
        print(json.dumps({'scene':name,'status':result['status'],'delta':result.get('delta'),'partial_gate':output['gate']}),flush=True); torch.cuda.empty_cache()
    print(json.dumps({'output':str(args.output),'gate':output['gate']},indent=2))

if __name__=='__main__':main()
