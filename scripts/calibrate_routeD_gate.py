#!/usr/bin/env python3
"""Calibrate risk-gate threshold on a held-out cache only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import torch
from torch.utils.data import DataLoader
from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import HypothesisScorer
from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import validate_routeD_candidate_cache
from projects.mmp_tracker.mmp_tracker.routeD_scorer_training import normalize_hypothesis_features, select_routeD_candidate

def main():
 p=argparse.ArgumentParser(); p.add_argument('--bundle',required=True); p.add_argument('--cache',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cpu'); args=p.parse_args()
 b=torch.load(args.bundle,map_location='cpu',weights_only=False); c=torch.load(args.cache,map_location='cpu',weights_only=False).get('cache',torch.load(args.cache,map_location='cpu',weights_only=False)); validate_routeD_candidate_cache(c)
 hidden_dim=int(b.get('config',{}).get('hidden_dim',0)) or int(b['model_state']['network.0.weight'].shape[0]); m=HypothesisScorer(feature_dim=int(b['feature_dim']),hidden_dim=hidden_dim); m.load_state_dict(b['model_state']); m.to(args.device).eval()
 mean=b.get('feature_mean'); std=b.get('feature_std')
 best=None
 thresholds=[i/20 for i in range(0,21)]
 f=c['features'].to(args.device).float();
 if mean is not None: f=normalize_hypothesis_features(f,mean,std)
 valid=c['candidate_valid_mask'].to(args.device)
 err=c['candidate_error_px'].to(args.device).float()
 with torch.no_grad(): logits=m(f); 
 results=[]
 for t in thresholds:
  pred,prob,_=select_routeD_candidate(logits,valid,selection_mode='risk_gate',gate_threshold=t)
  selected=err.gather(-1,pred.unsqueeze(-1)).squeeze(-1)
  results.append({'threshold':t,'error':float(selected.mean()),'selection_rate':float((pred>0).float().mean())})
  if best is None or results[-1]['error']<best['error']: best=results[-1]
 out={'best':best,'all':results,'evidence_tier':'development_diagnostic_only','paper_claim_eligible':False}
 Path(args.output).write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
