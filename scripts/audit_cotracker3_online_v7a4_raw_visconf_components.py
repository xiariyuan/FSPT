#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path('/gemini/code/FSPT')
DEFAULT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_first10.pt'
DEFAULT_LABELS = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/cotracker3_v6a3_reentry_utility_labels.npz'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_first10_event_features.npz'


def npy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def add_stats(feats, names, prefix, vals):
    a = np.asarray(vals, np.float32)
    for k, v in [
        ('mean', a.mean()), ('max', a.max()), ('min', a.min()), ('std', a.std()),
        ('slope', a[-1] - a[0]), ('recovery', a.max() - a[0]), ('drop', a[0] - a.min())
    ]:
        feats.append(float(v)); names.append(f'{prefix}_{k}')


def extract_event(rec, meta, pre=2, post=4):
    q = int(meta['query_idx']); t = int(meta['frame_t'])
    raw_v = npy(rec['pred_raw_vis_logit'], np.float32)
    raw_c = npy(rec['pred_raw_conf_logit'], np.float32)
    vp = npy(rec['pred_vis_prob_component'], np.float32)
    cp = npy(rec['pred_conf_prob_component'], np.float32)
    score = npy(rec['pred_vis_score'], np.float32)
    pred_vis = npy(rec['pred_visibility'], bool)
    T = raw_v.shape[1]
    feats=[]; names=[]
    seqs = {k: [] for k in ['raw_vis','raw_conf','vis_prob','conf_prob','score','min_prob','max_prob','prob_gap','raw_gap','score_margin','vis_margin','conf_margin','limiting_is_vis','native_visible']}
    for dt in range(-pre, post+1):
        tt = min(max(t+dt, 0), T-1)
        rv=float(raw_v[q,tt]); rc=float(raw_c[q,tt]); v=float(vp[q,tt]); c=float(cp[q,tt]); s=float(score[q,tt])
        minp=min(v,c); maxp=max(v,c); prob_gap=v-c; raw_gap=rv-rc
        vals={
            'raw_vis':rv, 'raw_conf':rc, 'vis_prob':v, 'conf_prob':c, 'score':s,
            'min_prob':minp, 'max_prob':maxp, 'prob_gap':prob_gap, 'raw_gap':raw_gap,
            'score_margin':s-0.6, 'vis_margin':v-0.6, 'conf_margin':c-0.6,
            'limiting_is_vis':1.0 if v < c else 0.0,
            'native_visible':1.0 if bool(pred_vis[q,tt]) else 0.0,
            'available':1.0 if 0 <= t+dt < T else 0.0,
        }
        # ratio/log-ratio with clipping.
        vals['prob_log_ratio_vis_conf'] = float(np.log((v+1e-6)/(c+1e-6)))
        vals['product_vs_min'] = float(s - minp)
        for k,val in vals.items():
            feats.append(float(val)); names.append(f'{k}_dt{dt}')
            if k in seqs: seqs[k].append(float(val))
    for k,seq in seqs.items():
        add_stats(feats,names,k,seq)
    # Short-latency focused summaries t..t+4.
    post_vals = {k: seq[pre:] for k,seq in seqs.items()}
    for k,seq in post_vals.items():
        add_stats(feats,names,f'post_{k}',seq)
    # component disagreement summaries.
    gap = np.asarray(seqs['prob_gap'], np.float32)
    feats.append(float(np.max(np.abs(gap)))); names.append('prob_gap_abs_max')
    feats.append(float(np.mean(np.abs(gap)))); names.append('prob_gap_abs_mean')
    feats.append(float(np.max(np.minimum(np.asarray(seqs['vis_prob']), np.asarray(seqs['conf_prob']))))); names.append('min_component_max')
    feats.append(float(np.min(np.maximum(np.asarray(seqs['vis_prob']), np.asarray(seqs['conf_prob']))))); names.append('max_component_min')
    return feats, names


def percentile_features(X, names, groups):
    blocks=[X.astype(np.float32)]; block_names=[f'raw::{n}' for n in names]
    for scope, keys in [('video', groups)]:
        Z=np.zeros_like(X, dtype=np.float32); P=np.zeros_like(X, dtype=np.float32)
        for k in sorted(set(keys.tolist())):
            idx=np.where(keys==k)[0]
            sub=X[idx]
            Z[idx]=(sub-sub.mean(axis=0))/(sub.std(axis=0)+1e-6)
            for j in range(X.shape[1]):
                order=np.argsort(sub[:,j], kind='mergesort')
                ranks=np.empty(len(idx), dtype=np.float32)
                ranks[order]=np.arange(len(idx), dtype=np.float32)/float(max(len(idx)-1,1))
                P[idx,j]=ranks
        blocks += [Z,P]
        block_names += [f'{scope}_z::{n}' for n in names]
        block_names += [f'{scope}_pct::{n}' for n in names]
    return np.concatenate(blocks, axis=1), block_names


def feature_groups(names):
    product_terms = ('score','native_visible')
    component_terms = ('raw_vis','raw_conf','vis_prob','conf_prob','min_prob','max_prob','prob_gap','raw_gap','vis_margin','conf_margin','limiting','prob_log_ratio','product_vs_min','component')
    return {
        'product_raw':[i for i,n in enumerate(names) if n.startswith('raw::') and any(t in n for t in product_terms) and not any(t in n for t in component_terms if t not in ['product_vs_min'])],
        'component_raw':[i for i,n in enumerate(names) if n.startswith('raw::') and any(t in n for t in component_terms)],
        'component_video_z':[i for i,n in enumerate(names) if n.startswith('video_z::') and any(t in n for t in component_terms)],
        'component_video_pct':[i for i,n in enumerate(names) if n.startswith('video_pct::') and any(t in n for t in component_terms)],
        'all_components':[i for i,n in enumerate(names) if any(t in n for t in component_terms)],
        'all_raw':[i for i,n in enumerate(names) if n.startswith('raw::')],
        'all_norm':[i for i,n in enumerate(names) if not n.startswith('raw::')],
        'all':list(range(len(names))),
    }


def loov(X,y,groups,idxs):
    if len(idxs)==0 or y.sum()==0 or y.sum()==len(y) or len(set(groups))<2:
        return None
    oof=np.zeros(len(y), np.float32)
    for tr,te in LeaveOneGroupOut().split(X,y,groups):
        if len(np.unique(y[tr]))<2:
            return None
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'))
        clf.fit(X[tr][:,idxs], y[tr])
        oof[te]=clf.predict_proba(X[te][:,idxs])[:,1]
    return {'ap':float(average_precision_score(y,oof)), 'auc':float(roc_auc_score(y,oof))}


def top_features(X,y,names,idxs,k=10):
    rows=[]
    if len(idxs)==0 or y.sum()==0 or y.sum()==len(y): return rows
    for j in idxs:
        v=X[:,j]
        if len(np.unique(v))<2: continue
        ap=average_precision_score(y,v); auc=roc_auc_score(y,v)
        apn=average_precision_score(y,-v); aucn=roc_auc_score(y,-v)
        if apn>ap: rows.append((apn,max(auc,1-auc),'-',names[j]))
        else: rows.append((ap,max(auc,1-auc),'+',names[j]))
    return [{'feature':n,'direction':d,'ap':float(ap),'auc':float(auc)} for ap,auc,d,n in sorted(rows, reverse=True)[:k]]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(DEFAULT_CACHE))
    ap.add_argument('--labels', default=str(DEFAULT_LABELS))
    ap.add_argument('--out-npz', default=str(DEFAULT_OUT))
    args=ap.parse_args()
    cache=Path(args.cache); labels_path=Path(args.labels); out=Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload=torch.load(cache,map_location='cpu',weights_only=False)
    records=payload['records']; rec_by_vid={str(r['video_id']):r for r in records}
    labels=np.load(labels_path,allow_pickle=True)
    metas_all=[json.loads(str(m)) for m in labels['meta_json'].tolist()]
    y4_all=np.asarray(labels['y_reentry_early4'],np.float32).astype(bool)
    y8_all=np.asarray(labels['y_reentry_early8'],np.float32).astype(bool)
    yu_all=np.asarray(labels['y_useful_open_t'],np.float32).astype(bool)
    X=[]; names=None; used_meta=[]; used_idx=[]
    for i,m in enumerate(metas_all):
        vid=m['video_id']
        if vid not in rec_by_vid: continue
        feats,fn=extract_event(rec_by_vid[vid],m)
        if names is None: names=fn
        elif names!=fn: raise RuntimeError('feature names mismatch')
        X.append(feats); used_idx.append(i); used_meta.append(json.dumps({'source_index':i,**m},ensure_ascii=False))
    X=np.asarray(X,np.float32); used_idx=np.asarray(used_idx,np.int64)
    groups=np.asarray([json.loads(m)['video_id'] for m in used_meta],object)
    y4=y4_all[used_idx]; y8=y8_all[used_idx]; yu=yu_all[used_idx]
    Xn, nn=percentile_features(X,names,groups)
    fg=feature_groups(nn)
    audits={}
    for lname,y in [('early4',y4),('early8',y8),('useful_open_t',yu)]:
        audits[lname]={'n_pos':int(y.sum()),'base_rate':float(y.mean()) if len(y) else 0.0,'families':{}}
        for fam,idxs in fg.items():
            audits[lname]['families'][fam]={'n_features':len(idxs),'loov_logreg':loov(Xn,y,groups,idxs),'top_features':top_features(Xn,y,nn,idxs,10)}
    np.savez_compressed(out,X=X,feature_names=np.asarray(names,dtype=object),X_aug=Xn,feature_names_aug=np.asarray(nn,dtype=object),meta_json=np.asarray(used_meta,dtype=object),source_indices=used_idx,y_reentry_early4=y4.astype(np.float32),y_reentry_early8=y8.astype(np.float32),y_useful_open_t=yu.astype(np.float32),groups=groups,cache=str(cache),labels=str(labels_path))
    report={'script':'scripts/audit_cotracker3_online_v7a4_raw_visconf_components.py','cache':str(cache),'labels':str(labels_path),'out_npz':str(out),'n_events':int(X.shape[0]),'feature_dim':int(X.shape[1]),'feature_dim_aug':int(Xn.shape[1]),'videos':dict(Counter(groups)),'audits':audits}
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
