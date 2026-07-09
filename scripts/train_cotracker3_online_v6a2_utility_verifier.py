#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np, torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from datasets.metrics import compute_tapvid_metrics
DEFAULT_LABELS=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a2_utility_verifier/cotracker3_v6a2_utility_labels_from_v6a1.npz'
DEFAULT_NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a2_utility_verifier'

def npy(x:Any,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def load(path:Path,label_key:str,offset_key:str):
    z=np.load(path,allow_pickle=True)
    X=np.asarray(z['X'],np.float32); y=np.asarray(z[label_key],np.float32).astype(bool); offsets=np.asarray(z[offset_key],np.int64)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    names=[str(f) for f in z['feature_names'].tolist()]
    groups=np.asarray([m['video_id'] for m in metas],object)
    return X,y,offsets,metas,names,groups

def models(seed):
    return {
        'logreg_balanced':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',solver='liblinear',random_state=seed)),
        'extra_trees_balanced':ExtraTreesClassifier(n_estimators=500,max_depth=7,min_samples_leaf=3,class_weight='balanced',random_state=seed),
        'random_forest_balanced':RandomForestClassifier(n_estimators=500,max_depth=7,min_samples_leaf=3,class_weight='balanced',random_state=seed),
    }

def prob(m,X):
    if hasattr(m,'predict_proba'): return m.predict_proba(X)[:,1].astype(np.float32)
    return m.decision_function(X).astype(np.float32)

def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32)); pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32)); n+=q.shape[0]
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']}|{'n_records':len(vals),'n_queries':int(n)}

def run_ajrd(cache,out_json):
    subprocess.run([sys.executable,str(ROOT/'scripts/eval_aj_rd_from_cache.py'),'--cache-path',str(cache),'--output-json',str(out_json)],cwd=str(ROOT),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.loads(Path(out_json).read_text())

def build(native,metas,p,thr,offsets):
    records=[]
    for r in native['records']:
        rr=dict(r); rr['pred_tracks']=npy(r['pred_tracks'],np.float32).copy(); rr['pred_visibility']=npy(r['pred_visibility'],bool).copy(); records.append(rr)
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(native['records'])}
    opened=[]
    for i,m in enumerate(metas):
        if p[i]<thr: continue
        vi=vid_to_idx[m['video_id']]; q=int(m['query_idx']); t=int(m['frame_t'])+int(offsets[i])
        if 0<=t<records[vi]['pred_visibility'].shape[1]:
            records[vi]['pred_visibility'][q,t]=True; opened.append(i)
    return records,opened

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',default=str(DEFAULT_LABELS))
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir',default=str(DEFAULT_OUTDIR))
    ap.add_argument('--strategy',choices=['open_t','first_recovery','scoremax'],default='open_t')
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    label_key=f'y_useful_{args.strategy}'
    offset_key=f'offset_{args.strategy}'
    X,y,offsets,metas,names,groups=load(Path(args.labels),label_key,offset_key)
    # leakage audit
    forbidden=['gt','err','safe','label','useful','harm']
    leak=[n for n in names if any(s in n.lower() for s in forbidden)]
    if leak: raise RuntimeError(f'leak features {leak}')
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    native_std=eval_records(native['records']); native_aj=run_ajrd(Path(args.native_cache),outdir/f'v6a2_native_ref_{args.strategy}_ajrd.json')
    native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
    logo=LeaveOneGroupOut(); results={}
    for name,m in models(args.seed).items():
        oof=np.zeros(len(y),np.float32); folds=[]
        for tr,te in logo.split(X,y,groups):
            if len(np.unique(y[tr]))<2:
                continue
            m.fit(X[tr],y[tr]); pp=prob(m,X[te]); oof[te]=pp
            for thr in [0.5,0.6,0.7,0.8,0.9]:
                pred=pp>=thr; pr,rc,f1,_=precision_recall_fscore_support(y[te],pred,average='binary',zero_division=0)
                folds.append({'heldout_video':str(groups[te][0]),'threshold':thr,'n_test':int(len(te)),'n_pos':int(y[te].sum()),'precision':float(pr),'recall':float(rc),'f1':float(f1)})
        ap_score=float(average_precision_score(y,oof)); auc=float(roc_auc_score(y,oof)) if len(np.unique(y))==2 else None
        np.save(outdir/f'v6a2_{name}_{args.strategy}_oof.npy',oof)
        thresholds=np.unique(np.concatenate([np.linspace(0.1,0.95,18),np.quantile(oof,np.linspace(0.5,0.98,12))])).astype(float)
        sweeps=[]
        for thr in thresholds:
            pred=oof>=thr; acc=int(pred.sum()); prec=float((pred & y).sum()/max(acc,1)); rec=float((pred & y).sum()/max(int(y.sum()),1))
            metric=None
            if acc>0:
                records,opened=build(native,metas,oof,float(thr),offsets)
                payload=dict(native); payload['records']=records; payload['model_name']=f'v6a2_{name}_{args.strategy}_thr{thr:.3f}'
                cache=outdir/f'v6a2_{name}_{args.strategy}_thr{thr:.3f}.pt'; torch.save(payload,cache)
                std=eval_records(records); aj=run_ajrd(cache,outdir/f'v6a2_{name}_{args.strategy}_thr{thr:.3f}_ajrd.json')
                metric={**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'cache':str(cache),'opened':len(opened)}
                metric['delta_vs_native']={k:(metric[k]-native_row[k]) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            sweeps.append({'threshold':float(thr),'accept_count':acc,'precision':prec,'recall':rec,'metric':metric})
        metric_s=[s for s in sweeps if s['metric']]
        constrained=[s for s in metric_s if s['metric']['delta_vs_native']['AJ']>=-0.10 and s['metric']['delta_vs_native']['OA']>=-0.10]
        best_constrained=sorted(constrained,key=lambda s:(s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ_RD'],s['metric']['delta_vs_native']['AJ']),reverse=True)[:10]
        best_raw=sorted(metric_s,key=lambda s:(s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ_RD']),reverse=True)[:10]
        results[name]={'ap':ap_score,'auc':auc,'folds':folds,'oof_path':str(outdir/f'v6a2_{name}_{args.strategy}_oof.npy'),'sweeps':sweeps,'best_constrained':best_constrained,'best_raw':best_raw}
    summary={'script':'scripts/train_cotracker3_online_v6a2_utility_verifier.py','labels':str(args.labels),'strategy':args.strategy,'label_key':label_key,'n_samples':int(len(y)),'n_pos':int(y.sum()),'pos_rate':float(y.mean()),'native_cache':str(args.native_cache),'native_row':native_row,'feature_names':names,'leakage_audit_passed':True,'models':results}
    out=outdir/f'v6a2_utility_verifier_{args.strategy}_loov_summary.json'; out.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'strategy':args.strategy,'n_samples':summary['n_samples'],'n_pos':summary['n_pos'],'pos_rate':summary['pos_rate'],'model_brief':{k:{'ap':v['ap'],'auc':v['auc'],'best_constrained':v['best_constrained'][:3],'best_raw':v['best_raw'][:3]} for k,v in results.items()}},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
