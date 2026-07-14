#!/usr/bin/env python3
"""Summarize Route-D candidate targets, oracle gain, and feature scales."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    payload=torch.load(args.cache,map_location='cpu',weights_only=False)
    c=payload.get('cache',payload)
    y=c['oracle_index'].long(); gain=c['oracle_gain_px'].float(); vis=c['visible'].bool()
    f=c['features'].float(); k=f.shape[1]
    result={
      'evidence_tier':'development_diagnostic_only','paper_claim_eligible':False,
      'rows':len(y),'candidate_count':k,
      'class_counts':torch.bincount(y,minlength=k).tolist(),
      'class_rates':(torch.bincount(y,minlength=k).float()/max(len(y),1)).tolist(),
      'global_target_rate':float((y>0).float().mean()),
      'global_improves_margin_rate':float(c['global_improves_local'].float().mean()),
      'feature_mean':f.mean((0,1)).tolist(),'feature_std':f.std((0,1),unbiased=False).tolist(),
      'subsets':{}
    }
    for name,mask in [('all',torch.ones_like(vis)),('visible',vis),('occluded',~vis)]:
      g=gain[mask]
      result['subsets'][name]={
        'rows':int(mask.sum()),'global_target_rate':float((y[mask]>0).float().mean()),
        'gain_mean_px':float(g.mean()),'gain_median_px':float(g.median()),
        'gain_p90_px':float(torch.quantile(g,0.9)),'gain_p99_px':float(torch.quantile(g,0.99)),
        'local_error_mean_px':float(c['local_error_px'][mask].float().mean()),
        'oracle_error_mean_px':float(c['oracle_error_px'][mask].float().mean()),
      }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
