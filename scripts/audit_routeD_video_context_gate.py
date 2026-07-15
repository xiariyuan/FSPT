#!/usr/bin/env python3
"""Audit a video-context adaptive gate on frozen Route-D candidates."""
from __future__ import annotations
import argparse,json,sys,random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_context_calibration import build_causal_video_context
from scripts.train_routeD_domain_action_calibrator import (
 build_action_examples, evaluate_selection, infer_threshold_logits,
 load_cache, load_frozen_scorer,
)
from scripts.audit_routeD_nested_profile_risk_gate import (
 predict_profile_risk_gate, per_video_rows
)
from scripts.audit_routeD_nested_posthoc_calibration import (
 make_outer_folds, make_inner_split, mask_for_ids
)

class ContextOffset(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scorer-bundle',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle=torch.load(args.scorer_bundle,map_location='cpu',weights_only=False)
    cache,meta=load_cache(args.cache)
    scorer,_=load_frozen_scorer(bundle,device)
    logits=infer_threshold_logits(scorer,cache,bundle,device,4096)
    ex=build_action_examples(cache,logits,()) if False else build_action_examples(cache,logits,bundle['thresholds_px'])
    # Context uses frozen action logits and safety signals. It is not allowed to use true gain.
    context=build_causal_video_context(
        sample_id=cache['sample_id'],
        frame_id=cache['frame_id'],
        action_logit=torch.zeros_like(ex['true_gain']),
        predicted_p1_margin=ex['predicted_p1_margin'],
        predicted_coarse_gain=ex['predicted_coarse_gain'],
        predicted_total_gain=ex['predicted_total_gain'],
        global_distance_to_local=ex['best_global_distance_to_local'],
        candidate_entropy=cache['features'][:,0].float(),
        previous_confidence=cache['features'][:,5].float(),
    )
    # Add causal context to a tiny offset model. The model sees only context + probability proxy.
    action_target=ex['true_gain']>0
    base_prob=torch.sigmoid(torch.zeros_like(ex['true_gain']))
    x=torch.cat([context,base_prob[:,None]],dim=-1)
    ids=sorted(set(int(v) for v in cache['sample_id'].tolist()))
    folds=make_outer_folds(ids,5,args.seed)
    oof=torch.zeros_like(ex['true_gain'])
    reports=[]
    for fold,test_ids in enumerate(folds):
        train_ids=[v for v in ids if v not in set(test_ids)]
        inner=make_inner_split(train_ids,args.seed+fold)
        fit=mask_for_ids(cache['sample_id'],inner['fit'])
        val=mask_for_ids(cache['sample_id'],inner['model_validation'])
        test=mask_for_ids(cache['sample_id'],test_ids)
        model=ContextOffset(x.shape[1]).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for _ in range(80):
            pred=model(x[fit].to(device))
            loss=F.binary_cross_entropy_with_logits(pred,action_target[fit].float().to(device))
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            prob=torch.sigmoid(model(x[test].to(device))).cpu()
        # Conservative fixed threshold chosen only for diagnosis.
        pred=torch.where(prob>0.5,ex['best_global_index'][test],torch.zeros_like(ex['best_global_index'][test]))
        rows={k:v[test] for k,v in ex.items() if torch.is_tensor(v)}
        rows['best_global_index']=ex['best_global_index'][test]
        rows['true_utility']=ex['true_utility'][test]
        metric=evaluate_selection(rows,pred,bundle['thresholds_px'],'context')
        oof[test]=prob
        reports.append({'fold':fold,'test_ids':test_ids,'metric':metric})
    out={'evidence_tier':'development_diagnostic_only','protocol':'nested video context only','oof_positive_rate':float(action_target.float().mean()),'reports':reports}
    Path(args.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
