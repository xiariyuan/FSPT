#!/usr/bin/env python3
"""Evaluate safety-aware Route-D gate policies from OOF calibrator outputs.

This script does not retrain models. It audits decision rules using frozen
out-of-fold probabilities and safety signals.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import torch

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from scripts.train_routeD_domain_action_calibrator import (
    build_action_examples, evaluate_selection, infer_threshold_logits,
    load_cache, load_frozen_scorer
)

def policy_predict(ex, prob, cfg):
    use=(prob>=cfg['prob_threshold'])
    if cfg.get('require_p1',False):
        use &= ex['predicted_p1_margin'] >= cfg['min_p1_margin']
    if cfg.get('require_coarse',False):
        use &= ex['predicted_coarse_gain'] >= cfg['min_coarse_gain']
    if cfg.get('require_total',False):
        use &= ex['predicted_total_gain'] >= cfg['min_total_gain']
    return torch.where(use, ex['best_global_index'], torch.zeros_like(ex['best_global_index']))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scorer-bundle',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--probability',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    bundle=torch.load(args.scorer_bundle,map_location='cpu',weights_only=False)
    cache,_=load_cache(args.cache)
    scorer,thresholds=load_frozen_scorer(bundle,torch.device(args.device if torch.cuda.is_available() else 'cpu'))
    logits=infer_threshold_logits(scorer,cache,bundle,torch.device(args.device if torch.cuda.is_available() else 'cpu'),4096)
    ex=build_action_examples(cache,logits,thresholds)
    prob=torch.load(args.probability,map_location='cpu',weights_only=False) if args.probability.endswith('.pt') else torch.tensor(json.load(open(args.probability)))
    if isinstance(prob,dict): prob=prob['probabilities']
    policies=[]
    for th in [0.3,0.4,0.5,0.6,0.7,0.8]:
        for p1 in [False,True]:
            for coarse in [False,True]:
                cfg={'prob_threshold':th,'require_p1':p1,'require_coarse':coarse,
                     'min_p1_margin':0.0,'min_coarse_gain':0.0}
                pred=policy_predict(ex,prob,cfg)
                row=evaluate_selection(ex,pred,thresholds,'risk_policy')
                row.update(cfg)
                policies.append(row)
    feasible=[r for r in policies if r['gain_delta_1']>=0 and r['mean_gain_over_local_threshold_utility']>0]
    best=max(feasible,key=lambda r:(r['mean_gain_over_local_threshold_utility'],r['mean_gain_over_local_px'])) if feasible else None
    out={'evidence_tier':'development_diagnostic_only','best':best,'all':policies}
    Path(args.output).write_text(json.dumps(out,indent=2))
    print(json.dumps({'best':best},indent=2))
if __name__=='__main__': main()
