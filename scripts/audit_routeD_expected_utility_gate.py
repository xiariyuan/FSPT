#!/usr/bin/env python3
"""Nested CV audit for expected-utility Route-D gate."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import torch

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from scripts.train_routeD_domain_action_calibrator import (
 build_action_examples, evaluate_selection, infer_threshold_logits,
 load_cache, load_frozen_scorer,
)
from scripts.audit_routeD_nested_posthoc_calibration import (
 make_outer_folds,make_inner_split,mask_for_ids,subset_examples
)
from projects.mmp_tracker.mmp_tracker.routeD_utility_regression import (
 UtilityRegressionConfig,train_utility_regressor,infer_utility_regressor
)

def predict_from_utility(ex,prediction,utility_threshold,p1_threshold):
    selected=(prediction[:,0]>=utility_threshold)&(prediction[:,1]>=p1_threshold)
    return torch.where(selected,ex['best_global_index'],torch.zeros_like(ex['best_global_index']))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scorer-bundle',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle=torch.load(args.scorer_bundle,map_location='cpu',weights_only=False)
    cache,_=load_cache(args.cache)
    scorer,_=load_frozen_scorer(bundle,device)
    logits=infer_threshold_logits(scorer,cache,bundle,device,4096)
    ex=build_action_examples(cache,logits,bundle['thresholds_px'])
    sample_id=cache['sample_id'].long()
    ids=sorted(set(int(v) for v in sample_id.tolist()))
    folds=make_outer_folds(ids,5,args.seed)
    oof=torch.zeros(len(sample_id),2)
    reports=[]
    for fold,test_ids in enumerate(folds):
        train_ids=[v for v in ids if v not in set(test_ids)]
        inner=make_inner_split(train_ids,args.seed+fold)
        masks={k:mask_for_ids(sample_id,v) for k,v in inner.items()}
        test=mask_for_ids(sample_id,test_ids)
        train=subset_examples(ex,masks['fit'])
        val=subset_examples(ex,masks['model_validation'])
        policy=subset_examples(ex,masks['policy_calibration'])
        test_ex=subset_examples(ex,test)
        features_train=train['features']; features_val=val['features']
        # Targets: true utility gain and exact 1px utility delta.
        train_p1=(train['true_utility']*0.0) # placeholder overwritten below
        # Use threshold-utility gain as the regression target.
        train_utility=train['true_utility'].gather(1,train['best_global_index'][:,None]).squeeze(1)-train['true_utility'][:,0]
        val_utility=val['true_utility'].gather(1,val['best_global_index'][:,None]).squeeze(1)-val['true_utility'][:,0]
        test_utility=test_ex['true_utility'].gather(1,test_ex['best_global_index'][:,None]).squeeze(1)-test_ex['true_utility'][:,0]
        train_p1=torch.zeros_like(train_utility)
        val_p1=torch.zeros_like(val_utility)
        test_p1=torch.zeros_like(test_utility)
        result=train_utility_regressor(
            features_train,train_utility,train_p1,
            features_val,val_utility,val_p1,
            config=UtilityRegressionConfig(seed=args.seed+fold),
            device=device,
        )
        pred=infer_utility_regressor(
            result['model'],test_ex['features'],result['feature_mean'],result['feature_std'],device=device
        )
        oof[test]=pred
        best=None
        # policy calibration uses utility thresholds only
        for ut in [-0.02,0.0,0.01,0.02,0.05]:
            for p1 in [-0.1,-0.05,0.0]:
                pp=infer_utility_regressor(result['model'],policy['features'],result['feature_mean'],result['feature_std'],device=device)
                sel=predict_from_utility(policy,pp,ut,p1)
                m=evaluate_selection(policy,sel,bundle['thresholds_px'],'utility')
                if m['mean_gain_over_local_threshold_utility']>0 and m['gain_delta_1']>=0:
                    if best is None or m['mean_gain_over_local_threshold_utility']>best[0]: best=(m['mean_gain_over_local_threshold_utility'],ut,p1,m)
        test_pred=predict_from_utility(test_ex,pred, best[1],best[2]) if best else torch.zeros_like(test_ex['best_global_index'])
        reports.append({'fold':fold,'test_ids':test_ids,'policy':best,'test':evaluate_selection(test_ex,test_pred,bundle['thresholds_px'],'utility')})
    out={'evidence_tier':'development_diagnostic_only','reports':reports}
    Path(args.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
