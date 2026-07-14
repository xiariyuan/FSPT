#!/usr/bin/env python3
"""Calibrate a no-harm Route-D gate on train-internal held-out videos."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
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

def nm(m):
 return {'AJ':float(m.get('AJ',m.get('average_jaccard',0.0))), 'OA':float(m.get('OA',m.get('occlusion_accuracy',0.0))), 'delta_avg':float(m.get('<avg',m.get('average_pts_within_thresh',0.0)))}

def metric(tr,gt,pv,occ,q,res,mode):
 return nm(compute_tapvid_metrics(tr[0],gt[0],pv[0],~occ[0],q[0],resolution=res,exclude_query_frame=True,query_mode=mode))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--selector-bundle',required=True); p.add_argument('--dataset-root',required=True); p.add_argument('--annotation-file',required=True); p.add_argument('--sample-ids',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--delta-tolerance',type=float,default=0.0); p.add_argument('--output',required=True); args=p.parse_args()
 ids={int(x) for x in args.sample_ids.split(',') if x.strip()}; thresholds=[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]; distances=[4.0,8.0,12.0,16.0,24.0,32.0,48.0,64.0,1.0e9]
 cfg=yaml.safe_load(Path(args.config).read_text()); cfg.setdefault('model',{}).setdefault('tracking',{})['enable_multi_hypothesis_diagnostics']=True; dc=cfg.setdefault('data',{}).setdefault('train',{}); dc['root']=args.dataset_root; dc['annotation_file']=args.annotation_file; dc['split']='train'; dc['subset']=max(ids)+1
 model=MMPTracker(config_from_dict(cfg)); ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(ck['model'],strict=True); device=torch.device(args.device if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); selector=RouteDRiskSelector(args.selector_bundle,device=device,threshold=0.0)
 ds,_=resolve_dataset(cfg,'train',False); loader=DataLoader(ds,batch_size=1,shuffle=False); records=[]
 with torch.no_grad():
  for i,b in enumerate(loader):
   if i>max(ids): break
   if i not in ids: continue
   video=b['video'].to(device); q=b['query_points'].to(device); gt=b['target_points'].to(device); occ=b['occluded'].to(device).bool(); baseline,pv,info=model(video,q,return_info=True)
   cp=info['hypothesis_candidate_points']; feat=build_hypothesis_features(cp,info['hypothesis_candidate_quality'],cp[...,0,:],info['hypothesis_previous_points'],info['hypothesis_previous_confidence'],info['hypothesis_candidate_entropy']); valid=torch.isfinite(cp).all(-1); out=selector.select(feat,cp,valid)
   gi=out['global_index']; gp=cp.gather(-2,gi.unsqueeze(-1).unsqueeze(-1).expand(*gi.shape,1,2)).squeeze(-2); scale=torch.tensor([video.shape[-2]-1,video.shape[-1]-1],device=device,dtype=gp.dtype); dist=torch.norm((gp-baseline)*scale,dim=-1)
   records.append({'id':i,'baseline':baseline,'pv':pv,'gt':gt,'occ':occ,'q':q,'res':resolve_eval_resolution(b,video,0),'mode':resolve_eval_query_mode(b,ds),'prob':out['gate_probability'],'gp':gp,'dist':dist})
 baseline_metrics={k:mean([metric(r['baseline'],r['gt'],r['pv'],r['occ'],r['q'],r['res'],r['mode'])[k] for r in records]) for k in ['AJ','OA','delta_avg']}
 results=[]
 for th in thresholds:
  for md in distances:
   vals={'AJ':[],'OA':[],'delta_avg':[]}; rates=[]
   for r in records:
    use=(r['prob']>=th)&(r['dist']<=md); tr=torch.where(use.unsqueeze(-1),r['gp'],r['baseline']); m=metric(tr,r['gt'],r['pv'],r['occ'],r['q'],r['res'],r['mode']);
    for k in vals: vals[k].append(m[k])
    rates.append(float(use.float().mean()))
   agg={k:mean(v) for k,v in vals.items()}; results.append({'threshold':th,'max_switch_distance_px':md,'aggregate':agg,'delta_vs_baseline':{k:agg[k]-baseline_metrics[k] for k in agg},'global_selection_rate':mean(rates)})
 feasible=[r for r in results if r['aggregate']['delta_avg']>=baseline_metrics['delta_avg']-args.delta_tolerance]
 best=max(feasible,key=lambda r:(r['aggregate']['AJ'],r['aggregate']['delta_avg'])) if feasible else max(results,key=lambda r:(r['aggregate']['AJ']+r['aggregate']['delta_avg']))
 out={'evidence_tier':'development_diagnostic_only','paper_claim_eligible':False,'calibration_scope':'train_internal_heldout_only','sample_ids':sorted(ids),'baseline':baseline_metrics,'delta_tolerance':args.delta_tolerance,'best':best,'results':results}
 Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='results'},indent=2))
if __name__=='__main__': main()
