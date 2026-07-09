#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_NPZ=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5c_verifier/cotracker3_v5c_local96_top20_candidate_dataset.npz'
DEFAULT_NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle/native_ref.pt'
DEFAULT_OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5c_verifier'

def npy(x:Any,dtype=None)->np.ndarray:
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def load_npz(path:Path):
    z=np.load(path,allow_pickle=True)
    X=np.asarray(z['X'],np.float32)
    y=np.asarray(z['y_good'],np.float32).astype(bool)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    names=[str(n) for n in z['feature_names'].tolist()]
    groups=np.asarray([m['video_id'] for m in metas],object)
    event_keys=np.asarray([f"{m['video_id']}::{m['query_idx']}::{m['frame_t']}" for m in metas],object)
    return X,y,metas,names,groups,event_keys

def make_models(seed:int=0):
    return {
        'logreg_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=seed)),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=seed),
        'random_forest_balanced': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, class_weight='balanced', random_state=seed),
    }

def prob_model(model,X):
    if hasattr(model,'predict_proba'):
        return model.predict_proba(X)[:,1]
    return model.decision_function(X)

def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32)); n+=int(q.shape[0])
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}

def run_ajrd(cache,out_json):
    subprocess.run([sys.executable,str(ROOT/'scripts/eval_aj_rd_from_cache.py'),'--cache-path',str(cache),'--output-json',str(out_json)],cwd=str(ROOT),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.loads(Path(out_json).read_text())

def build_records(native_payload, metas, probs, threshold, out_name):
    records=[]
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(native_payload['records'])}
    for r in native_payload['records']:
        rr=dict(r)
        rr['pred_tracks']=npy(r['pred_tracks'],np.float32).copy()
        rr['pred_visibility']=npy(r['pred_visibility'],bool).copy()
        records.append(rr)
    # event-level: one candidate per event, choose max prob then threshold.
    event_to_indices={}
    for i,m in enumerate(metas):
        key=(m['video_id'],int(m['query_idx']),int(m['frame_t']))
        event_to_indices.setdefault(key,[]).append(i)
    accepted=[]
    for key,idxs in event_to_indices.items():
        best=max(idxs,key=lambda i: probs[i])
        if probs[best] < threshold:
            continue
        m=metas[best]; vi=vid_to_idx[m['video_id']]; q=int(m['query_idx']); t=int(m['frame_t'])
        y,x=m['candidate_yx_px']
        records[vi]['pred_tracks'][q,t]=np.asarray([float(y)/255.0,float(x)/255.0],np.float32)
        records[vi]['pred_visibility'][q,t]=True
        accepted.append(best)
    return records,accepted

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',default=str(DEFAULT_NPZ))
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir',default=str(DEFAULT_OUTDIR))
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    X,y,metas,names,groups,event_keys=load_npz(Path(args.npz))
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    # Defensive leakage audit.
    forbidden=['err','gt','better','safe','good','native_err','candidate_err']
    leakage=[n for n in names if any(f in n.lower() for f in forbidden)]
    if leakage:
        raise RuntimeError(f'Potential GT leakage in feature names: {leakage}')
    models=make_models(args.seed)
    logo=LeaveOneGroupOut()
    model_results={}
    for model_name,model in models.items():
        oof=np.zeros(len(y),np.float32)
        fold_rows=[]
        for tr,te in logo.split(X,y,groups):
            if len(np.unique(y[tr]))<2:
                oof[te]=0.0
                continue
            m=model
            m.fit(X[tr],y[tr])
            p=prob_model(m,X[te]).astype(np.float32)
            oof[te]=p
            yhat=p>=0.5
            pr,rc,f1,_=precision_recall_fscore_support(y[te],yhat,average='binary',zero_division=0)
            fold_rows.append({'heldout_video':str(groups[te][0]),'n_test':int(len(te)),'n_pos':int(y[te].sum()),'p_mean':float(p.mean()),'precision@0.5':float(pr),'recall@0.5':float(rc),'f1@0.5':float(f1)})
        ap_score=float(average_precision_score(y,oof)) if y.sum()>0 else 0.0
        auc=float(roc_auc_score(y,oof)) if len(np.unique(y))==2 else None
        # threshold sweep candidate-level and event-level metrics
        thresholds=np.unique(np.concatenate([np.linspace(0.05,0.95,19), np.quantile(oof, np.linspace(0.5,0.99,15))])).astype(float)
        sweeps=[]
        native_std=eval_records(native['records'])
        native_cache=Path(args.native_cache)
        native_aj=run_ajrd(native_cache,outdir/'v5c_native_ref_ajrd.json')
        native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
        for thr in thresholds:
            pred=oof>=thr
            n_acc=int(pred.sum())
            if n_acc>0:
                prec=float((pred & y).sum()/n_acc)
                rec=float((pred & y).sum()/max(int(y.sum()),1))
            else:
                prec=0.0; rec=0.0
            records,accepted=build_records(native,metas,oof,float(thr),f'{model_name}_thr{thr:.3f}')
            # Only evaluate a manageable subset of thresholds: all candidate stats, but metric for notable thresholds.
            metric_row=None
            if len(accepted)>0 and (prec>=0.5 or len(sweeps)%4==0):
                payload=dict(native); payload['records']=records; payload['model_name']=f'{model_name}_thr{thr:.3f}'
                cache=outdir/f'{model_name}_thr{thr:.3f}.pt'; torch.save(payload,cache)
                std=eval_records(records); aj=run_ajrd(cache,outdir/f'{model_name}_thr{thr:.3f}_ajrd.json')
                metric_row={**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'delta':{k:(std.get(k,aj.get(k))-native_row[k] if k in native_row and k in std else None) for k in []}}
                metric_row['delta_vs_native']={k:(metric_row[k]-native_row[k]) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
                metric_row['cache']=str(cache)
            sweeps.append({'threshold':float(thr),'candidate_accept_count':n_acc,'candidate_precision':prec,'candidate_recall':rec,'event_accept_count':len(accepted),'metric':metric_row})
        # choose best by AJ_RD if metric available, with nonnegative AJ guard secondary
        metric_sweeps=[s for s in sweeps if s['metric']]
        best=sorted(metric_sweeps,key=lambda s:(s['metric']['delta_vs_native']['AJ_RD'],s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ']),reverse=True)[:10]
        model_results[model_name]={'ap':ap_score,'auc':auc,'folds':fold_rows,'oof_path':str(outdir/f'{model_name}_oof.npy'),'sweeps':sweeps,'best_metric_sweeps':best}
        np.save(outdir/f'{model_name}_oof.npy',oof)
    summary={
        'script':'scripts/train_cotracker3_online_v5c_lightweight_verifier.py',
        'npz':str(args.npz),
        'native_cache':str(args.native_cache),
        'n_samples':int(len(y)),
        'n_positive_good':int(y.sum()),
        'positive_rate':float(y.mean()),
        'feature_names':names,
        'leakage_audit_passed':True,
        'videos':sorted(set(map(str,groups.tolist()))),
        'models':model_results,
    }
    out=outdir/'v5c_lightweight_verifier_loov_summary.json'
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'n_samples':summary['n_samples'],'n_positive_good':summary['n_positive_good'],'positive_rate':summary['positive_rate'],'model_brief':{k:{'ap':v['ap'],'auc':v['auc'],'best':v['best_metric_sweeps'][:3]} for k,v in model_results.items()}},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
