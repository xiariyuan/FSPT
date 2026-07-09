#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_FEATURES = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b0_raw_visconf_verifier'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_features(path: Path, target: str, family: str):
    z = np.load(path, allow_pickle=True)
    X_raw = np.asarray(z['X'], np.float32)
    X_aug = np.asarray(z['X_aug'], np.float32)
    names_raw = [str(x) for x in z['feature_names'].tolist()]
    names_aug = [str(x) for x in z['feature_names_aug'].tolist()]
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    y = np.asarray(z[target], np.float32).astype(bool)
    groups = np.asarray([m['video_id'] for m in metas], object)
    def idxs_for(names):
        product_terms = ('score','native_visible')
        component_terms = ('raw_vis','raw_conf','vis_prob','conf_prob','min_prob','max_prob','prob_gap','raw_gap','vis_margin','conf_margin','limiting','prob_log_ratio','product_vs_min','component')
        if family == 'product_raw':
            return [i for i,n in enumerate(names) if n.startswith('raw::') and any(t in n for t in product_terms) and not any(t in n for t in component_terms if t not in ['product_vs_min'])]
        if family == 'component_raw':
            return [i for i,n in enumerate(names) if n.startswith('raw::') and any(t in n for t in component_terms)]
        if family == 'all_components':
            return [i for i,n in enumerate(names) if any(t in n for t in component_terms)]
        if family == 'all':
            return list(range(len(names)))
        if family == 'all_norm':
            return [i for i,n in enumerate(names) if not n.startswith('raw::')]
        raise ValueError(f'unknown family {family}')
    if family in {'product_raw','component_raw'}:
        # The family definition assumes augmented names with raw:: prefix; use augmented matrix.
        X = X_aug; names = names_aug; idx = idxs_for(names_aug)
    else:
        X = X_aug; names = names_aug; idx = idxs_for(names_aug)
    if len(idx) == 0:
        raise RuntimeError(f'empty feature family {family}')
    leak = [names[i] for i in idx if any(s in names[i].lower() for s in ['gt','label','reentry','useful'])]
    if leak:
        raise RuntimeError(f'potential leakage features: {leak[:20]}')
    return X[:, idx], y, groups, metas, [names[i] for i in idx]


def fit_oof_logreg(X, y, groups):
    oof = np.zeros(len(y), np.float32)
    folds = []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear'))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1].astype(np.float32)
        oof[te] = p
        for thr in [0.5,0.7,0.85,0.95]:
            pred = p >= thr
            pr, rc, f1, _ = precision_recall_fscore_support(y[te], pred, average='binary', zero_division=0)
            folds.append({'video': str(groups[te][0]), 'threshold': thr, 'n_test': int(len(te)), 'n_pos': int(y[te].sum()), 'precision': float(pr), 'recall': float(rc), 'f1': float(f1), 'p_mean': float(p.mean())})
    return oof, folds


def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt=torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'], bool))
        gv=torch.from_numpy(npy(r['gt_visibility'], bool))
        q=torch.from_numpy(npy(r['query_points'], np.float32))
        n += int(q.shape[0])
        m=compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}


def run_ajrd(cache: Path, out_json: Path):
    subprocess.run([sys.executable, str(ROOT/'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(cache), '--output-json', str(out_json)], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(out_json.read_text())


def build_open_records(native_payload, metas, probs, threshold):
    records=[]
    for r in native_payload['records']:
        rr=dict(r)
        rr['pred_visibility']=npy(r['pred_visibility'], bool).copy()
        records.append(rr)
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(records)}
    opened=[]
    for i,m in enumerate(metas):
        if probs[i] < threshold:
            continue
        vi=vid_to_idx.get(m['video_id'])
        if vi is None:
            continue
        q=int(m['query_idx']); t=int(m['frame_t'])
        if 0 <= t < records[vi]['pred_visibility'].shape[1]:
            records[vi]['pred_visibility'][q,t]=True
            opened.append(i)
    return records, opened


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', default=str(DEFAULT_FEATURES))
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--target', choices=['y_reentry_early4','y_reentry_early8','y_useful_open_t'], default='y_reentry_early4')
    ap.add_argument('--family', choices=['product_raw','component_raw','all_components','all_norm','all'], default='all_components')
    ap.add_argument('--max-thresholds', type=int, default=24)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    X,y,groups,metas,feature_names=load_features(Path(args.features), args.target, args.family)
    oof, folds=fit_oof_logreg(X,y,groups)
    ap_score=float(average_precision_score(y,oof)); auc_score=float(roc_auc_score(y,oof))
    np.save(outdir/f'v7b0_{args.target}_{args.family}_oof.npy', oof)
    native=torch.load(args.native_cache, map_location='cpu', weights_only=False)
    native_std=eval_records(native['records']); native_aj=run_ajrd(Path(args.native_cache), outdir/f'v7b0_native_ref_{args.target}_{args.family}_ajrd.json')
    native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
    thresholds=np.unique(np.concatenate([np.linspace(0.1,0.95,18), np.quantile(oof, np.linspace(0.5,0.99,14))])).astype(float)
    if len(thresholds)>args.max_thresholds:
        ids=np.linspace(0,len(thresholds)-1,args.max_thresholds).round().astype(int)
        thresholds=thresholds[ids]
    sweeps=[]
    for thr in thresholds:
        pred=oof>=thr
        acc=int(pred.sum()); prec=float((pred & y).sum()/max(acc,1)); rec=float((pred & y).sum()/max(int(y.sum()),1))
        metric=None
        if acc>0:
            records, opened=build_open_records(native, metas, oof, float(thr))
            payload=dict(native); payload['records']=records; payload['model_name']=f'v7b0_{args.target}_{args.family}_thr{thr:.3f}'
            cache=outdir/f'v7b0_{args.target}_{args.family}_thr{thr:.3f}.pt'
            torch.save(payload, cache)
            std=eval_records(records); aj=run_ajrd(cache, outdir/f'v7b0_{args.target}_{args.family}_thr{thr:.3f}_ajrd.json')
            metric={**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'opened':len(opened),'cache':str(cache)}
            metric['delta_vs_native']={k:(metric[k]-native_row[k]) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
        sweeps.append({'threshold':float(thr),'accept_count':acc,'precision':prec,'recall':rec,'metric':metric})
    metric_s=[s for s in sweeps if s['metric']]
    constrained=[s for s in metric_s if s['metric']['delta_vs_native']['AJ']>=-0.10 and s['metric']['delta_vs_native']['OA']>=-0.10]
    best_constrained=sorted(constrained, key=lambda s:(s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ_RD'],s['metric']['delta_vs_native']['AJ']), reverse=True)[:10]
    best_raw=sorted(metric_s, key=lambda s:(s['metric']['delta_vs_native']['AJ_RD_256'],s['metric']['delta_vs_native']['AJ_RD']), reverse=True)[:10]
    summary={'script':'scripts/train_cotracker3_online_v7b0_raw_visconf_verifier.py','features':str(args.features),'native_cache':str(args.native_cache),'target':args.target,'family':args.family,'n_samples':int(len(y)),'n_pos':int(y.sum()),'pos_rate':float(y.mean()),'feature_dim':int(X.shape[1]),'ap':ap_score,'auc':auc_score,'folds':folds,'native_row':native_row,'sweeps':sweeps,'best_constrained':best_constrained,'best_raw':best_raw,'feature_names':feature_names}
    out=outdir/f'v7b0_{args.target}_{args.family}_summary.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({'out':str(out),'target':args.target,'family':args.family,'n_pos':summary['n_pos'],'pos_rate':summary['pos_rate'],'ap':summary['ap'],'auc':summary['auc'],'best_constrained':best_constrained[:5],'best_raw':best_raw[:3]}, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
