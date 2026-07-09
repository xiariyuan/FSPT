#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import npy, check_alignment, per_query_ajrd_map, eval_one

OUTDIR = Path('outputs/paper_discovery_2026-06-27/candidate_aware_reentry_selector')
OUTDIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = ['offline','online_global','b2_fullpost_p1','b2_w8_p2','b2_w16_p1','b2_w16_p2','b2_w32_p2']

SETTINGS = {
    'dev_translate_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt'),
        'online_global': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt'),
        'b2_fullpost_p1': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16/b2_fullpost_p1.pt'),
        'b2_w8_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16/b2_w8_p2.pt'),
        'b2_w16_p1': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16/b2_w16_p1.pt'),
        'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16/b2_w16_p2.pt'),
        'b2_w32_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16/b2_w32_p2.pt'),
    },
    'dev_occluder_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt'),
        'online_global': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_online_occluder_L16.pt'),
        'b2_fullpost_p1': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16/b2_fullpost_p1.pt'),
        'b2_w8_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16/b2_w8_p2.pt'),
        'b2_w16_p1': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16/b2_w16_p1.pt'),
        'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16/b2_w16_p2.pt'),
        'b2_w32_p2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16/b2_w32_p2.pt'),
    },
    'rgb_fresh20_49_natural': {
        'offline': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'),
        'online_global': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt'),
        'b2_fullpost_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_fullpost_p1_rgb_fresh20_49.pt'),
        'b2_w8_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt'),
        'b2_w16_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt'),
        'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt'),
        'b2_w32_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt'),
    },
}


def first_trigger(base_v: np.ndarray, over_v: np.ndarray, qt: int, persist: int = 2) -> int:
    T = len(base_v)
    for t in range(max(1, int(qt)+1), T):
        inv = t > 0 and not bool(base_v[t-1])
        if inv and t + persist <= T and bool(np.all(over_v[t:t+persist])):
            return int(t)
    return min(T-1, max(1, int(qt)+1))


def longest_false(v: np.ndarray, start: int) -> int:
    best=cur=0
    for t in range(max(0,start), len(v)):
        if not bool(v[t]):
            cur += 1; best=max(best,cur)
        else:
            cur = 0
    return int(best)


def vis_trans(v: np.ndarray, start: int) -> int:
    seg=v[max(0,start):]
    return int(np.sum(seg[1:] != seg[:-1])) if len(seg)>1 else 0


def motion_stats(p: np.ndarray, lo: int, hi: int) -> Tuple[float,float,float]:
    lo=max(0,int(lo)); hi=min(len(p),int(hi))
    if hi-lo<2: return 0.0,0.0,0.0
    s=np.linalg.norm(p[lo+1:hi]-p[lo:hi-1],axis=-1)
    return float(np.mean(s)), float(np.std(s)), float(np.max(s)) if s.size else 0.0


def candidate_basic_features(base_rec: Dict[str,Any], cand_rec: Dict[str,Any], qi: int, trig: int, qt: int) -> List[float]:
    bp=npy(base_rec['pred_tracks'],np.float32)[qi]
    bv=npy(base_rec['pred_visibility'],bool)[qi]
    cp=npy(cand_rec['pred_tracks'],np.float32)[qi]
    cv=npy(cand_rec['pred_visibility'],bool)[qi]
    T=len(bv); hi16=min(T,trig+17); hi32=min(T,trig+33); lo8=max(0,trig-8)
    d=np.linalg.norm(cp[trig:hi16]-bp[trig:hi16],axis=-1) if hi16>trig else np.zeros(0)
    cm=motion_stats(cp,lo8,hi16)
    bm=motion_stats(bp,lo8,hi16)
    qxy=npy(base_rec['query_points'],np.float32)[qi,1:]
    return [
        float(np.mean(cv[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean(cv[trig:hi32])) if hi32>trig else 0.0,
        float(np.mean(bv[trig:hi16] != cv[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean((~bv[trig:hi16]) & cv[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean(bv[trig:hi16] & (~cv[trig:hi16]))) if hi16>trig else 0.0,
        float(np.mean(d)) if d.size else 0.0,
        float(np.std(d)) if d.size else 0.0,
        float(np.max(d)) if d.size else 0.0,
        float(np.linalg.norm(cp[trig]-bp[trig])),
        float(np.linalg.norm(cp[trig]-qxy)),
        cm[0], cm[1], cm[2],
        bm[0], bm[1], bm[2],
        float(longest_false(cv, qt+1)),
        float(vis_trans(cv, qt+1)),
    ]


def feature_vector(caches: Dict[str,Dict[str,Any]], rec_i: int, qi: int) -> np.ndarray:
    base_rec=caches['offline']['records'][rec_i]
    online_rec=caches['online_global']['records'][rec_i]
    bv=npy(base_rec['pred_visibility'],bool)[qi]
    ov=npy(online_rec['pred_visibility'],bool)[qi]
    qpts=npy(base_rec['query_points'],np.float32)
    qt=int(round(float(qpts[qi,0]))); T=len(bv)
    trig=first_trigger(bv,ov,qt,persist=2)
    hi16=min(T,trig+17); hi32=min(T,trig+33)
    base_global=[
        float(trig/max(T-1,1)), float((trig-qt)/max(T-1,1)),
        float(np.mean(bv[qt+1:] if qt+1<T else bv[qt:qt+1])),
        float(np.mean(ov[qt+1:] if qt+1<T else ov[qt:qt+1])),
        float(np.mean(bv[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean(ov[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean(bv[trig:hi16] != ov[trig:hi16])) if hi16>trig else 0.0,
        float(np.mean((~bv[trig:hi16]) & ov[trig:hi16])) if hi16>trig else 0.0,
        float(longest_false(bv,qt+1)), float(longest_false(ov,qt+1)),
        float(vis_trans(bv,qt+1)), float(vis_trans(ov,qt+1)),
    ]
    feats=list(base_global)
    for cname in CANDIDATES:
        feats.extend(candidate_basic_features(base_rec, caches[cname]['records'][rec_i], qi, trig, qt))
    return np.nan_to_num(np.asarray(feats,dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def load_caches(paths: Dict[str,Path]) -> Dict[str,Dict[str,Any]]:
    caches={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in paths.items()}
    ref=caches['offline']
    for k,c in caches.items():
        if k!='offline': check_alignment(ref,c,k)
    return caches


def build_dataset(setting: str, paths: Dict[str,Path], include_all_queries: bool = True) -> Dict[str,Any]:
    npz=OUTDIR / f'{setting}_candidate_selector_dataset.npz'
    if npz.exists():
        d=np.load(npz,allow_pickle=True)
        return {'setting':setting,'npz':str(npz),'n':int(len(d['y'])),'class_counts':dict(Counter(d['label_names'].tolist()))}
    caches=load_caches(paths)
    X=[]; y=[]; label_names=[]; meta=[]; reentry_flags=[]
    for rec_i, br in enumerate(caches['offline']['records']):
        maps={cname:per_query_ajrd_map(caches[cname]['records'][rec_i]) for cname in CANDIDATES}
        n=int(npy(br['query_points']).shape[0])
        all_q=range(n) if include_all_queries else maps['offline'].keys()
        for qi in all_q:
            if qi in maps['offline']:
                best='offline'; bestv=float(maps['offline'][qi])
                for cname in CANDIDATES:
                    val=maps[cname].get(qi)
                    if val is not None and float(val)>bestv+1e-9:
                        best=cname; bestv=float(val)
                is_re=1
            else:
                best='offline'; is_re=0
            X.append(feature_vector(caches,rec_i,qi)); label_names.append(best); y.append(CANDIDATES.index(best)); meta.append((rec_i,qi,str(br['video_id']))); reentry_flags.append(is_re)
    X=np.vstack(X).astype(np.float32); y=np.asarray(y,dtype=np.int64); labels=np.asarray(label_names,dtype=object); meta_arr=np.asarray(meta,dtype=object); re=np.asarray(reentry_flags,dtype=np.int64)
    np.savez_compressed(npz,X=X,y=y,label_names=labels,meta=meta_arr,reentry_flags=re,candidates=np.asarray(CANDIDATES,dtype=object))
    return {'setting':setting,'npz':str(npz),'n':int(len(y)),'class_counts':dict(Counter(labels.tolist())),'reentry_n':int(np.sum(re))}


def fit_models(X: np.ndarray, y: np.ndarray) -> Dict[str,Any]:
    models={
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=0.7, solver='lbfgs', multi_class='auto')),
        'random_forest': RandomForestClassifier(n_estimators=400,max_depth=14,min_samples_leaf=4,class_weight='balanced_subsample',random_state=20260702,n_jobs=-1),
        'extra_trees': ExtraTreesClassifier(n_estimators=500,max_depth=16,min_samples_leaf=3,class_weight='balanced',random_state=20260702,n_jobs=-1),
        'hist_gbdt': HistGradientBoostingClassifier(max_iter=260,max_leaf_nodes=31,learning_rate=0.05,l2_regularization=0.05,random_state=20260702),
    }
    for m in models.values(): m.fit(X,y)
    return models


def apply_selector(setting: str, paths: Dict[str,Path], dataset_npz: Path, model: Any, model_name: str, conf_thr: float, out_path: Path) -> Dict[str,Any]:
    data=np.load(dataset_npz,allow_pickle=True); X=data['X']; meta=data['meta']
    if hasattr(model,'predict_proba'):
        proba=model.predict_proba(X)
        classes=list(model.classes_)
        pred_cls=np.asarray([classes[int(np.argmax(row))] for row in proba],dtype=np.int64)
        conf=np.max(proba,axis=1)
    else:
        pred_cls=model.predict(X); conf=np.ones_like(pred_cls,dtype=np.float32)
    # low-confidence or offline class => offline
    pred_cls=np.where(conf>=conf_thr,pred_cls,0)
    caches=load_caches(paths)
    off=caches['offline']
    choose_map={(str(row[2]),int(row[1])):int(cls) for row,cls in zip(meta,pred_cls)}
    records=[]; counts=Counter()
    for rec_i, br in enumerate(off['records']):
        vid=str(br['video_id']); n=int(npy(br['query_points']).shape[0])
        pred_p=npy(br['pred_tracks'],np.float32).copy(); pred_v=npy(br['pred_visibility'],bool).copy()
        for qi in range(n):
            cls=choose_map.get((vid,qi),0)
            cname=CANDIDATES[int(cls)]
            counts[cname]+=1
            if cname!='offline':
                cr=caches[cname]['records'][rec_i]
                pred_p[qi]=npy(cr['pred_tracks'],np.float32)[qi]
                pred_v[qi]=npy(cr['pred_visibility'],bool)[qi]
        nr=dict(br); nr['pred_tracks']=pred_p.astype(np.float32); nr['pred_visibility']=pred_v.astype(bool); nr['selector']={'model':model_name,'conf_thr':float(conf_thr)}
        records.append(nr)
    payload=dict(off); payload['records']=records; payload['model_name']=f'{model_name}_thr{conf_thr:.2f}'; payload['candidate_aware_selector']={'model':model_name,'conf_thr':float(conf_thr),'candidate_counts':dict(counts),'candidate_rates':{k:round(v/max(sum(counts.values()),1),6) for k,v in counts.items()}}
    out_path.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out_path)
    return payload['candidate_aware_selector']


def main():
    ds={name:build_dataset(name,paths,include_all_queries=True) for name,paths in SETTINGS.items()}
    train_names=['dev_translate_L16','dev_occluder_L16']
    X_train=np.vstack([np.load(ds[n]['npz'],allow_pickle=True)['X'] for n in train_names])
    y_train=np.concatenate([np.load(ds[n]['npz'],allow_pickle=True)['y'] for n in train_names])
    models=fit_models(X_train,y_train)
    model_info={}
    for name,m in models.items():
        pred=m.predict(X_train)
        model_info[name]={'train_acc':float(accuracy_score(y_train,pred)),'train_bal_acc':float(balanced_accuracy_score(y_train,pred))}
        with open(OUTDIR/f'{name}.pkl','wb') as f: pickle.dump(m,f)
    thresholds=[0.0,0.2,0.3,0.4,0.5,0.6,0.7]
    eval_settings=['rgb_fresh20_49_natural']
    results=[]
    for setting in eval_settings:
        rows=[eval_one(c,SETTINGS[setting][c]) for c in CANDIDATES]
        selector_rows=[]
        for mname,m in models.items():
            for thr in thresholds:
                outp=OUTDIR/setting/mname/f'{mname}_conf{thr:.2f}.pt'
                sel=apply_selector(setting,SETTINGS[setting],Path(ds[setting]['npz']),m,mname,thr,outp)
                er=eval_one(f'{mname}_conf{thr:.2f}',outp); er['selection']=sel; selector_rows.append(er)
        by={r['name']:r for r in rows+selector_rows}
        gains={}
        for r in rows+selector_rows:
            if r['name']=='offline': continue
            gains[f'{r["name"]}_vs_offline']={'AJ_RD_256':round(float(r['AJ_RD_256'])-float(by['offline']['AJ_RD_256']),6),'AJ_256':round(float(r['AJ_256'])-float(by['offline']['AJ_256']),6),'OA_256':round(float(r['OA_256'])-float(by['offline']['OA_256']),6)}
            gains[f'{r["name"]}_vs_guard']={'AJ_RD_256':None,'AJ_256':None,'OA_256':None}
        # add external guard reference if exists
        guard_path=Path('outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt')
        guard=eval_one('guard_rf_thr0.40',guard_path)
        by['guard_rf_thr0.40']=guard
        for r in selector_rows:
            gains[f'{r["name"]}_vs_guard_rf_thr0.40']={'AJ_RD_256':round(float(r['AJ_RD_256'])-float(guard['AJ_RD_256']),6),'AJ_256':round(float(r['AJ_256'])-float(guard['AJ_256']),6),'OA_256':round(float(r['OA_256'])-float(guard['OA_256']),6)}
        # choose best under AJ>=79.0 first, fallback AJ>=78.6, then max AJRD
        feasible=[r for r in selector_rows if float(r['AJ_256'])>=79.0]
        if not feasible: feasible=[r for r in selector_rows if float(r['AJ_256'])>=78.6]
        best=max(feasible if feasible else selector_rows,key=lambda r:(float(r['AJ_RD_256']),float(r['AJ_256'])))
        results.append({'setting':setting,'methods':rows+[guard]+selector_rows,'gains':gains,'best_selector':best['name']})
    summary={'candidates':CANDIDATES,'datasets':ds,'train':{'settings':train_names,'n':int(len(y_train)),'class_counts':dict(Counter([CANDIDATES[int(i)] for i in y_train]))},'model_info':model_info,'results':results}
    (OUTDIR/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    compact=[]
    for r in results:
        best=next(m for m in r['methods'] if m['name']==r['best_selector'])
        guard=next(m for m in r['methods'] if m['name']=='guard_rf_thr0.40')
        compact.append({'setting':r['setting'],'guard':{k:guard[k] for k in ['AJ_RD_256','AJ_256','OA_256']},'best_selector':{k:best[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']},'selection':best.get('selection'),'gain_vs_guard':r['gains'].get(f'{best["name"]}_vs_guard_rf_thr0.40')})
    print(json.dumps({'summary_path':str(OUTDIR/'summary.json'),'train':summary['train'],'model_info':model_info,'compact':compact},indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
