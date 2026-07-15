#!/usr/bin/env python3
"""Open-loop trajectory evaluation for a frozen Route-D risk selector.

The selector changes reported trajectory points but does not feed those points
back into MMP state. Results are development diagnostics, not closed-loop claims.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader
import yaml
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from datasets import compute_tapvid_metrics
from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import build_hypothesis_features
from projects.mmp_tracker.mmp_tracker.routeD_selector import RouteDRiskSelector
from projects.mmp_tracker.train_mmp import config_from_dict, resolve_dataset, resolve_eval_query_mode, resolve_eval_resolution

def mean(xs):
 xs=[float(x) for x in xs if math.isfinite(float(x))]
 return sum(xs)/len(xs) if xs else float('nan')

def norm_metrics(m):
 return {'AJ':float(m.get('AJ',m.get('average_jaccard',0.0))), 'OA':float(m.get('OA',m.get('occlusion_accuracy',0.0))), 'delta_avg':float(m.get('<avg',m.get('average_pts_within_thresh',0.0)))}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--selector-bundle',required=True); p.add_argument('--threshold',type=float,required=True); p.add_argument('--fusion-strength',type=float,default=None); p.add_argument('--max-switch-distance-px',type=float,default=None); p.add_argument('--profile-p1-tolerance',type=float,default=None); p.add_argument('--profile-min-coarse-gain',type=float,default=0.0); p.add_argument('--profile-min-total-gain',type=float,default=0.0); p.add_argument('--closed-loop',action='store_true'); p.add_argument('--split',default='train'); p.add_argument('--dataset-root',required=True); p.add_argument('--annotation-file',required=True); p.add_argument('--dataset-split',default='validation'); p.add_argument('--dataset',default=None); p.add_argument('--limit',type=int,default=32); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=17); p.add_argument('--output',required=True); args=p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
 cfg=yaml.safe_load(Path(args.config).read_text()); cfg.setdefault('model',{}).setdefault('tracking',{})['enable_multi_hypothesis_diagnostics']=True
 dc=cfg.setdefault('data',{}).setdefault(args.split,{});
 if args.dataset: dc['dataset']=args.dataset; dc['root']=args.dataset_root; dc['annotation_file']=args.annotation_file; dc['split']=args.dataset_split; dc['subset']=args.limit
 model=MMPTracker(config_from_dict(cfg)); ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(ck['model'],strict=True)
 device=torch.device(args.device if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); selector=RouteDRiskSelector(args.selector_bundle,device=device,threshold=args.threshold)
 if args.closed_loop:
  model.attach_routeD_selector(args.selector_bundle, device=device, p1_tolerance=args.profile_p1_tolerance or 0.0, min_coarse_gain=args.profile_min_coarse_gain, min_total_gain=args.profile_min_total_gain, threshold=args.threshold, update_memory=True)
 dataset,_=resolve_dataset(cfg,args.split,False); loader=DataLoader(dataset,batch_size=1,shuffle=False)
 acc={k:{'AJ':[],'OA':[],'delta_avg':[]} for k in ['baseline','local','routeD']}; rows=[]
 with torch.no_grad():
  for i,b in enumerate(loader):
   if i>=args.limit: break
   video=b['video'].to(device); q=b['query_points'].to(device); gt=b['target_points'].to(device); occ=b['occluded'].to(device).bool()
   baseline,pv,info=model(video,q,return_info=True)
   cp=info['hypothesis_candidate_points']; cq=info['hypothesis_candidate_quality']; ce=info['hypothesis_candidate_entropy']; pp=info['hypothesis_previous_points']; pc=info['hypothesis_previous_confidence']
   feat=build_hypothesis_features(cp,cq,cp[...,0,:],pp,pc,ce); valid=torch.isfinite(cp).all(-1)
   scale=torch.tensor([video.shape[-2]-1,video.shape[-1]-1],device=device,dtype=baseline.dtype); out=selector.select(feat,cp,valid,fallback_points=baseline,max_switch_distance_px=args.max_switch_distance_px,pixel_scale=scale if args.max_switch_distance_px is not None else None,fusion_strength=args.fusion_strength,profile_p1_tolerance=args.profile_p1_tolerance,profile_min_coarse_gain=args.profile_min_coarse_gain,profile_min_total_gain=args.profile_min_total_gain); routed=out['points']; local=cp[...,0,:]; frame_index=torch.arange(baseline.shape[-2],device=device).view(1,1,-1); active=frame_index>q[...,0].long().unsqueeze(-1); routed=torch.where(active.unsqueeze(-1),routed,baseline); local=torch.where(active.unsqueeze(-1),local,baseline)
   mode=resolve_eval_query_mode(b,dataset); res=resolve_eval_resolution(b,video,0)
   active_count=active.sum().clamp_min(1); row={'sample':i,'global_selection_rate':float(((out['index']>0)&active).sum()/active_count),'mean_fusion_alpha':float((out['fusion_alpha']*active).sum()/active_count)}
   for name,tr in [('baseline',baseline),('local',local),('routeD',routed)]:
    m=norm_metrics(compute_tapvid_metrics(tr[0],gt[0],pv[0],~occ[0],q[0],resolution=res,exclude_query_frame=True,query_mode=mode)); row[name]=m
    for k,v in m.items(): acc[name][k].append(v)
   rows.append(row); print(i,row)
 agg={n:{k:mean(v) for k,v in d.items()} for n,d in acc.items()}; delta={k:agg['routeD'][k]-agg['baseline'][k] for k in agg['baseline']}
 result={'evidence_tier':'development_diagnostic_only','paper_claim_eligible':False,'evaluation_mode':'closed_loop_routeD' if args.closed_loop else 'open_loop_posthoc_selector','closed_loop_state_updated':bool(args.closed_loop),'threshold':args.threshold,'seed':args.seed,'fusion_strength':args.fusion_strength,'max_switch_distance_px':args.max_switch_distance_px,'profile_p1_tolerance':args.profile_p1_tolerance,'profile_min_coarse_gain':args.profile_min_coarse_gain,'profile_min_total_gain':args.profile_min_total_gain,'samples':len(rows),'aggregate':agg,'delta_routeD_vs_baseline':delta,'per_sample':rows}
 Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='per_sample'},indent=2))
if __name__=='__main__': main()
