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

DEFAULT_NPZ=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5d_visibility_verifier/cotracker3_v5d_event_visibility_dataset.npz'
DEFAULT_NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5d_visibility_verifier'

def npy(x:Any,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def load_dataset(path:Path,label_name:str):
    z=np.load(path,allow_pickle=True)
    X=np.asarray(z['X'],np.float32); names=[str(n) for n in z['feature_names'].tolist()]
    y=np.asarray(z[label_name],np.float32).astype(bool)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    groups=np.asarray([m['video_id'] for m in metas],object)
    return X,y,metas,names,groups

def models(seed):
    return {
        'logreg_balanced':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=seed)),
        'extra_trees_balanced':ExtraTreesClassifier(n_estimators=400,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'random_forest_balanced':RandomForestClassifier(n_estimators=400,max_depth=6,min_samples_leaf=3,class_weight='balanced',random_state=seed),
    }

def prob(model,X):
    if hasattr(model,'predict_proba'): return model.predict_proba(X)[:,1].astype(np.float32)
    return model.decision_function(X).astype(np.float32)

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

def build_open_records(native_payload,metas,probs,threshold):
    recs=[]
    for r in native_payload['records']:
        rr=dict(r); rr['pred_tracks']=npy(r['pred_tracks'],np.float32).copy(); rr['pred_visibility']=npy(r['pred_visibility'],bool).copy(); recs.append(rr)
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(native_payload['records'])}
    opened=[]
    # There can be duplicate event rows? Here each event unique; but guard by max prob.
    event_to_best={}
    for i,m in enumerate(metas):
        key=(m['video_id'],int(m['query_idx']),int(m['frame_t']))
        if key not in event_to_best or probs[i]>probs[event_to_best[key]]:
            event_to_best[key]=i
    for key,i in event_to_best.items():
        if probs[i] < threshold: continue
        m=metas[i]; vi=vid_to_idx[m['video_id']]; q=int(m['query_idx']); t=int(m['frame_t'])
        recs[vi]['pred_visibility'][q,t]=True
        opened.append(i)
    return recs,opened

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',default=str(DEFAULT_NPZ))
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir',default=str(DEFAULT_OUTDIR))
    ap.add_argument('--label',choices=['y_gt_visible','y_safe16','y_safe8','y_utility'],default='y_safe16')
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    X,y,metas,names,groups=load_dataset(Path(args.npz),args.label)
    # leakage audit: only metadata may contain gt, features must not.
    forbidden=['gt','err','safe','utility','label']
    leakage=[n for n in names if any(f in n.lower() for f in forbidden)]
    if leakage: raise RuntimeError(f'Potential leakage in feature names: {leakage}')
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    native_cache=Path(args.native_cache)
    native_std=eval_records(native['records']); native_aj=run_ajrd(native_cache,outdir/f'v5d_native_ref_{args.label}_ajrd.json')
    native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
    logo=LeaveOneGroupOut()
    all_results={}
    for name,model in models(args.seed).items():
        oof=np.zeros(len(y),np.float32); fold_rows=[]
        for tr,te in logo.split(X,y,groups):
            if len(np.unique(y[tr]))<2:
                oof[te]=0.0
                continue
            model.fit(X[tr],y[tr]); p=prob(model,X[te]); oof[te]=p
            for thr in [0.3,0.5,0.7]:
                pred=p>=thr
                pr,rc,f1,_=precision_recall_fscore_support(y[te],pred,average='binary',zero_division=0)
                fold_rows.append({'heldout_video':str(groups[te][0]),'threshold':thr,'n_test':int(len(te)),'n_pos':int(y[te].sum()),'precision':float(pr),'recall':float(rc),'f1':float(f1),'p_mean':float(p.mean())})
        ap_score=float(average_precision_score(y,oof)) if y.sum()>0 else 0.0
        auc=float(roc_auc_score(y,oof)) if len(np.unique(y))==2 else None
        thresholds=np.unique(np.concatenate([np.linspace(0.05,0.95,19),np.quantile(oof,np.linspace(0.5,0.99,20))])).astype(float)
        sweeps=[]
        for thr in thresholds:
            pred=oof>=thr; acc=int(pred.sum()); prec=float((pred & y).sum()/max(acc,1)); rec=float((pred & y).sum()/max(int(y.sum()),1))
            records,opened=build_open_records(native,metas,oof,float(thr))
            metric=None
            # Evaluate all thresholds with at least one opened event and either decent precision or regular sampling.
            if opened and (prec>=0.3 or len(sweeps)%3==0):
                payload=dict(native); payload['records']=records; payload['model_name']=f'v5d_{name}_{args.label}_thr{thr:.3f}'
                cache=outdir/f'v5d_{name}_{args.label}_thr{thr:.3f}.pt'; torch.save(payload,cache)
                std=eval_records(records); aj=run_ajrd(cache,outdir/f'v5d_{name}_{args.label}_thr{thr:.3f}_ajrd.json')
                metric={**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'cache':str(cache)}
                metric['delta_vs_native']={k:(metric[k]-native_row[k]) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            sweeps.append({'threshold':float(thr),'accept_count':acc,'precision':prec,'recall':rec,'opened_events':len(opened),'metric':metric})
        metric_sweeps=[s for s in sweeps if s['metric']]
        best=sorted(metric_sweeps,key=lambda s:(s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ_RD'],s['metric']['delta_vs_native']['AJ']),reverse=True)[:10]
        np.save(outdir/f'v5d_{name}_{args.label}_oof.npy',oof)
        all_results[name]={'ap':ap_score,'auc':auc,'folds':fold_rows,'sweeps':sweeps,'best_metric_sweeps':best,'oof_path':str(outdir/f'v5d_{name}_{args.label}_oof.npy')}
    summary={'script':'scripts/train_cotracker3_online_v5d_visibility_verifier.py','npz':str(args.npz),'label':args.label,'native_cache':str(args.native_cache),'n_samples':int(len(y)),'n_pos':int(y.sum()),'pos_rate':float(y.mean()),'feature_names':names,'leakage_audit_passed':True,'videos':sorted(set(map(str,groups.tolist()))),'native_row':native_row,'models':all_results}
    out=outdir/f'v5d_visibility_verifier_{args.label}_loov_summary.json'; out.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'n_samples':summary['n_samples'],'n_pos':summary['n_pos'],'pos_rate':summary['pos_rate'],'native_row':native_row,'model_brief':{k:{'ap':v['ap'],'auc':v['auc'],'best':v['best_metric_sweeps'][:3]} for k,v in all_results.items()}},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
