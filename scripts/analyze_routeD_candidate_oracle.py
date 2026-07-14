#!/usr/bin/env python3
"""Route-D candidate oracle analysis.

Measures whether existing local/global proposals already contain useful
multi-modal alternatives. This is diagnostic only and does not change the
tracker or training objective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.train_mmp import config_from_dict, resolve_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',default='val')
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text())
    model=MMPTracker(config_from_dict(cfg))
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dataset,max_samples=resolve_dataset(cfg,args.split,False)
    loader=DataLoader(dataset,batch_size=1,shuffle=False)
    total=0; oracle=[]; local_errors=[]; global_top=[]; multi=0; entropy=[]; eff=[]
    with torch.no_grad():
      for idx,batch in enumerate(loader):
        if idx>=args.limit: break
        video=batch['video'].to(device)
        q=batch['query_points'].to(device)
        gt=batch['target_points'].to(device)
        tracks,vis,info=model(video,q,return_info=True)
        
        local=info['local_points']
        glob=info['global_candidate_points']
        cand=torch.cat([local.unsqueeze(-2), glob], dim=-2) # B,N,T,K,2
        if cand.shape[-2] <= 1: continue
        # final frame-wise oracle distance
        err=torch.norm(cand-gt.unsqueeze(-2),dim=-1)
        current=torch.norm(tracks-gt,dim=-1)
        current_err=current
        local_err=err[...,0]
        oracle_err=err.min(dim=-1).values
        total += err.numel()
        oracle.append(float(oracle_err.mean()))
        local_errors.append(float(local_err.mean()))
        gap=(current_err-oracle_err).mean()
        global_top.append(float(gap))
        if 'belief_entropy' in info:
          entropy.append(float(info['belief_entropy'].mean()))
          eff.append(float(info['belief_effective_hypotheses'].mean()))
        multi += int((err[...,1:].min(dim=-1).values < local_err).sum())
    result={'samples':args.limit,'mean_oracle_error_px':sum(oracle)/max(len(oracle),1),'mean_local_error_px':sum(local_errors)/max(len(local_errors),1),'mean_current_minus_oracle_px':sum(global_top)/max(len(global_top),1),'candidate_better_count':multi,'mean_belief_entropy':sum(entropy)/max(len(entropy),1),'mean_effective_hypotheses':sum(eff)/max(len(eff),1)}
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
