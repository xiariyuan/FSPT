#!/usr/bin/env python3
"""Calibrate a protocol-safe soft Route-D fusion on train-internal videos."""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
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

def normalize_metrics(m):
    return {
        'AJ':float(m.get('AJ',m.get('average_jaccard',0.0))),
        'OA':float(m.get('OA',m.get('occlusion_accuracy',0.0))),
        'delta_avg':float(m.get('<avg',m.get('average_pts_within_thresh',0.0))),
    }

def evaluate(tracks, record):
    return normalize_metrics(compute_tapvid_metrics(
        tracks[0], record['gt'][0], record['visibility'][0], ~record['occluded'][0],
        record['query'][0], resolution=record['resolution'], exclude_query_frame=True,
        query_mode=record['query_mode'],
    ))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--selector-bundle',required=True); p.add_argument('--dataset-root',required=True)
    p.add_argument('--annotation-file',required=True); p.add_argument('--sample-ids',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--delta-tolerance',type=float,default=0.0)
    p.add_argument('--seed',type=int,default=17); p.add_argument('--output',required=True); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); sample_ids={int(x) for x in args.sample_ids.split(',') if x.strip()}
    thresholds=[0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75]
    distances=[8.0,12.0,16.0,24.0,32.0,48.0,64.0,1.0e9]
    strengths=[0.0,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.75,1.0]
    cfg=yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault('model',{}).setdefault('tracking',{})['enable_multi_hypothesis_diagnostics']=True
    dc=cfg.setdefault('data',{}).setdefault('train',{}); dc['root']=args.dataset_root
    dc['annotation_file']=args.annotation_file; dc['split']='train'; dc['subset']=max(sample_ids)+1
    model=MMPTracker(config_from_dict(cfg)); checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
    selector=RouteDRiskSelector(args.selector_bundle,device=device,threshold=0.0)
    dataset,_=resolve_dataset(cfg,'train',False); loader=DataLoader(dataset,batch_size=1,shuffle=False)
    records=[]
    with torch.no_grad():
        for sample_index,batch in enumerate(loader):
            if sample_index>max(sample_ids): break
            if sample_index not in sample_ids: continue
            video=batch['video'].to(device); query=batch['query_points'].to(device)
            gt=batch['target_points'].to(device); occluded=batch['occluded'].to(device).bool()
            baseline,visibility,info=model(video,query,return_info=True)
            candidates=info['hypothesis_candidate_points']
            features=build_hypothesis_features(
                candidates,info['hypothesis_candidate_quality'],candidates[...,0,:],
                info['hypothesis_previous_points'],info['hypothesis_previous_confidence'],
                info['hypothesis_candidate_entropy'],
            )
            out=selector.select(features,candidates,torch.isfinite(candidates).all(-1))
            global_index=out['global_index']
            global_points=candidates.gather(
                -2,global_index.unsqueeze(-1).unsqueeze(-1).expand(*global_index.shape,1,2)
            ).squeeze(-2)
            scale=torch.tensor([video.shape[-2]-1,video.shape[-1]-1],device=device,dtype=global_points.dtype)
            distance=torch.norm((global_points-baseline)*scale,dim=-1)
            records.append({
                'sample_id':sample_index,'baseline':baseline,'visibility':visibility,'gt':gt,
                'occluded':occluded,'query':query,'resolution':resolve_eval_resolution(batch,video,0),
                'query_mode':resolve_eval_query_mode(batch,dataset),'probability':out['gate_probability'],
                'global_points':global_points,'distance':distance,
            })
    baseline_per_video=[evaluate(record['baseline'],record) for record in records]
    baseline={key:mean([row[key] for row in baseline_per_video]) for key in ['AJ','OA','delta_avg']}
    results=[]
    for threshold in thresholds:
        probability_denominator=max(1.0-threshold,1.0e-6)
        for max_distance in distances:
            for strength in strengths:
                values={key:[] for key in ['AJ','OA','delta_avg']}; rates=[]; alphas=[]
                for record in records:
                    eligible=(record['probability']>=threshold)&(record['distance']<=max_distance)
                    calibrated=((record['probability']-threshold)/probability_denominator).clamp(0.0,1.0)
                    alpha=float(strength)*calibrated*eligible.to(calibrated.dtype)
                    tracks=record['baseline']+alpha.unsqueeze(-1)*(record['global_points']-record['baseline'])
                    metrics=evaluate(tracks,record)
                    for key in values: values[key].append(metrics[key])
                    rates.append(float((alpha>0).float().mean())); alphas.append(float(alpha.mean()))
                aggregate={key:mean(value) for key,value in values.items()}
                results.append({
                    'threshold':threshold,'max_switch_distance_px':max_distance,
                    'fusion_strength':strength,'aggregate':aggregate,
                    'delta_vs_baseline':{key:aggregate[key]-baseline[key] for key in aggregate},
                    'fusion_rate':mean(rates),'mean_fusion_alpha':mean(alphas),
                })
    feasible=[row for row in results if row['aggregate']['delta_avg']>=baseline['delta_avg']-args.delta_tolerance]
    nontrivial=[row for row in feasible if row['fusion_strength']>0 and row['fusion_rate']>0]
    candidate_pool=nontrivial if nontrivial else feasible
    best=max(candidate_pool,key=lambda row:(row['aggregate']['AJ'],row['aggregate']['delta_avg']))
    result={
        'evidence_tier':'development_diagnostic_only','paper_claim_eligible':False,
        'calibration_scope':'train_internal_heldout_only','sample_ids':sorted(sample_ids),
        'baseline':baseline,'delta_tolerance':args.delta_tolerance,'seed':args.seed,
        'feasible_count':len(feasible),'nontrivial_feasible_count':len(nontrivial),
        'fallback_to_baseline':not bool(nontrivial),'best':best,'results':results,
    }
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='results'},indent=2))
if __name__=='__main__': main()
