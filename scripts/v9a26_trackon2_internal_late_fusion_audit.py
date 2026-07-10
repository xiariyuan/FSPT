#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
V9A2 = BASE / 'v9a2_anchor_uncertainty_reacquisition'
OUTDIR = BASE / 'v9a26_trackon2_internal_proxy'
JOINT = V9A2 / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
INTERNAL = OUTDIR / 'v9a26_trackon2_internal_proxy_features.npz'
FROZEN = V9A2 / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a26_trackon2_internal_late_fusion_audit.json'
OUT_DOC = ROOT / 'docs/v9a26_trackon2_internal_late_fusion_audit_result_2026-07-10.md'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import CANDIDATE, NATIVE, align_candidate, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import apply_touched_mask, per_video_delta, summarize_pv
from scripts.v9a1_controller_calibration_aware_prototype import choose_threshold_by_metric
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint
from scripts.v9a25_eval_dinov3_identity_features import binary_metrics, paired_summary


def metric_delta(metric: dict, base: dict) -> dict:
    keys=['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']
    return {k:None if metric.get(k) is None or base.get(k) is None else float(metric[k]-base[k]) for k in keys}


def ext_stats(data: dict, accept: np.ndarray) -> dict:
    ext=data['is_w16_extension'].astype(bool)
    return {
        'accepted_extension':int(np.sum(accept&ext)),
        'accepted_ext_good':int(np.sum(accept&ext&data['y_candidate_good'].astype(bool))),
        'accepted_ext_bad':int(np.sum(accept&ext&data['y_candidate_bad'].astype(bool))),
        'accepted_ext_false_visible':int(np.sum(accept&ext&data['y_false_visible'].astype(bool))),
        'accepted_ext_worse':int(np.sum(accept&ext&data['y_candidate_worse_px'].astype(bool))),
    }


def evaluate(name,data,native,cand_by,accept,native_metric,native_pv):
    recs,stats=apply_touched_mask(native,cand_by,data['metas'],accept.astype(bool))
    metric,pv=standard_and_ajrd(recs); pvr=per_video_delta(native_pv,pv)
    return {'variant':name,'metric':metric,'delta_vs_native':metric_delta(metric,native_metric),'apply_stats':stats,'extension_stats':ext_stats(data,accept),'per_video_summary':summarize_pv(pvr),'per_video_rows':pvr}


def internal_logreg_oof(X,y,groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof=np.full(len(y),np.nan,float)
    for tr,va in GroupKFold(n_splits=min(5,len(np.unique(groups)))).split(X,y,groups):
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
        model.fit(X[tr],y[tr]); oof[va]=model.predict_proba(X[va])[:,1]
    if not np.all(np.isfinite(oof)): raise RuntimeError('non-finite internal oof')
    return oof


def main():
    data=load_joint(JOINT)
    iz=np.load(INTERNAL,allow_pickle=True)
    ext_idx=np.asarray(iz['joint_indices'],dtype=np.int64)
    ext=data['is_w16_extension'].astype(bool)
    if not np.array_equal(ext_idx,np.where(ext)[0]): raise RuntimeError('alignment mismatch')
    X=iz['X_internal'].astype(np.float32)
    X=X[:,np.std(X,axis=0)>1e-10]
    y=data['y_candidate_good'][ext_idx].astype(int); groups=data['groups'][ext_idx]
    internal_ext=internal_logreg_oof(X,y,groups)

    report=json.loads(FROZEN.read_text())
    frozen_full=None
    for block in report['training']:
        for row in block['overall']:
            if row['feature_set']=='all' and row['model']=='logreg':
                frozen_full=np.asarray(block['full_scores']['all_logreg'],dtype=float)
    if frozen_full is None: raise RuntimeError('missing frozen score')
    frozen_ext=frozen_full[ext_idx]

    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    cand=torch.load(CANDIDATE,map_location='cpu',weights_only=False)
    ok,info,cand_by=align_candidate(native,cand)
    if not ok: raise RuntimeError(info)
    native_metric,native_pv=standard_and_ajrd(clone_records(native['records']))

    rows=[]
    frozen_accept=event_accept_mask(data,frozen_full,0.05,'event_max')
    frozen_row=evaluate('frozen_v9a2_fixed0.05',data,native,cand_by,frozen_accept,native_metric,native_pv); rows.append(frozen_row)
    internal_full=np.zeros(len(data['metas']),float); internal_full[ext_idx]=internal_ext
    internal_thr=float(choose_threshold_by_metric(y,internal_ext,metric='f1'))
    rows.append(evaluate('internal_only_oof_f1',data,native,cand_by,event_accept_mask(data,internal_full,internal_thr,'event_max'),native_metric,native_pv))

    corr={'pearson':float(np.corrcoef(frozen_ext,internal_ext)[0,1]),'spearman':None}
    try:
        from scipy.stats import spearmanr
        corr['spearman']=float(spearmanr(frozen_ext,internal_ext).statistic)
    except Exception: pass
    fusions=[]
    for alpha in [0.25,0.50,0.75]:
        fused=alpha*frozen_ext+(1-alpha)*internal_ext
        thr=float(choose_threshold_by_metric(y,fused,metric='f1'))
        full=np.zeros(len(data['metas']),float); full[ext_idx]=fused
        name=f'late_fusion_frozen{alpha:.2f}_internal{1-alpha:.2f}_event_max_oof_f1'
        row=evaluate(name,data,native,cand_by,event_accept_mask(data,full,thr,'event_max'),native_metric,native_pv)
        rows.append(row)
        fusions.append({'variant':name,'alpha_frozen':alpha,'alpha_internal':1-alpha,'threshold_f1':thr,'classifier':binary_metrics(y,fused),'result':row})
    paired={x['variant']:paired_summary(x['result'],frozen_row) for x in fusions}
    out={'protocol':'predeclared weights [0.25,0.50,0.75]; video-heldout OOF; event_max; OOF-F1; no trajectory threshold sweep','score_correlations':corr,'internal_classifier':{**binary_metrics(y,internal_ext),'threshold_f1':internal_thr},'rows':rows,'fusions':fusions,'paired_vs_frozen':paired}
    OUT_JSON.write_text(json.dumps(out,indent=2,ensure_ascii=False))

    lines=['# V9-A2.6 TrackOn2 Internal Proxy Late-Fusion Audit','',out['protocol'],'',f"Pearson correlation: {corr['pearson']:.4f}",f"Spearman correlation: {corr['spearman'] if corr['spearman'] is not None else 'n/a'}",'','| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |','|---|---:|---:|---:|---:|---|']
    for row in rows:
        d=row['delta_vs_native'];e=row['extension_stats'];pv=row['per_video_summary'];es=f"{e['accepted_extension']}/{e['accepted_ext_good']}/{e['accepted_ext_bad']}/{e['accepted_ext_false_visible']}/{e['accepted_ext_worse']}"
        lines.append(f"| {row['variant']} | {d['AJ']:+.4f} | {d['OA']:+.4f} | {d['AJ_RD_256']:+.4f} | {es} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    lines+=['','## Paired against frozen V9-A2 fixed0.05','','| Fusion | Mean | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |','|---|---:|---|---|---:|']
    for name,s in paired.items():
        ci=s['bootstrap_95_ci_mean'];lines.append(f"| {name} | {s['mean_diff']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {s['better']}/{s['worse']}/{s['equal']} | {s['exact_sign_flip_p'] if s['exact_sign_flip_p'] is not None else ''} |")
    OUT_DOC.write_text('\n'.join(lines))
    best=max(rows,key=lambda r:r['delta_vs_native']['AJ_RD_256'])
    print(json.dumps({'ok':True,'json':str(OUT_JSON),'doc':str(OUT_DOC),'best':best['variant'],'best_AJ_RD_256_delta':best['delta_vs_native']['AJ_RD_256'],'correlations':corr}))

if __name__=='__main__': main()
