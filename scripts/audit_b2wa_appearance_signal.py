#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

IN = Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/appearance_signal_summary.json')
TRAJ_FEATURES = ['trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16','override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16','base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16','vis_agreement_frac_next16','override_visibility_transitions_next16','base_visibility_transitions_next16','base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16','base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16']


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def fnum(v):
    if v is None: return np.nan
    try: x=float(v)
    except Exception: return np.nan
    return x if math.isfinite(x) else np.nan


def feature_names(rows, kind):
    if kind == 'traj': return TRAJ_FEATURES
    app = sorted(rows[0]['appearance_features'].keys())
    if kind == 'app': return app
    return TRAJ_FEATURES + ['app::'+k for k in app]


def matrix(rows, kind):
    names = feature_names(rows, kind)
    arr=[]
    for r in rows:
        vals=[]
        for n in names:
            if n.startswith('app::'):
                vals.append(fnum(r['appearance_features'].get(n[5:])))
            elif n in r.get('appearance_features', {}) and kind == 'app':
                vals.append(fnum(r['appearance_features'].get(n)))
            else:
                vals.append(fnum(r['traj_features'].get(n)))
        arr.append(vals)
    return np.asarray(arr, dtype=np.float32), names


def label(rows, target):
    if target == 'window_has_reentry': return np.asarray([bool(r['window_has_reentry']) for r in rows], bool)
    if target == 'harmful_w16': return np.asarray([bool(r['w16_harmful_full']) for r in rows], bool)
    if target == 'helpful_w16': return np.asarray([bool(r['w16_helpful_target']) for r in rows], bool)
    if target == 'best_reject': return np.asarray([r['best_action']=='reject' for r in rows], bool)
    if target == 'best_long': return np.asarray([r['best_action'] in ('W8','W16') for r in rows], bool)
    raise ValueError(target)


def safe_auc(y,s):
    if len(np.unique(y)) < 2: return None
    return float(roc_auc_score(y.astype(int), s))


def single_feature_auc(X, names, y):
    out=[]
    for j,n in enumerate(names):
        x=X[:,j]
        m=np.isfinite(x)
        if m.sum() < 10 or len(np.unique(y[m])) < 2: continue
        # replace nonfinite by median for ranking consistency
        xx=x.copy()
        med=float(np.nanmedian(xx)) if np.any(np.isfinite(xx)) else 0.0
        xx[~np.isfinite(xx)] = med
        a=safe_auc(y, xx)
        if a is None: continue
        out.append({'feature':n,'auc_positive_high':round(a,6),'best_auc':round(max(a,1-a),6),'direction':'high' if a>=0.5 else 'low'})
    out.sort(key=lambda z:z['best_auc'], reverse=True)
    return out


def model_auc_cv(rows, kind, target):
    X,names=matrix(rows, kind)
    y=label(rows,target)
    groups=np.asarray([str(r['video_id']) for r in rows])
    n_splits=min(5, len(np.unique(groups)))
    if len(np.unique(y)) < 2 or n_splits < 2: return {'auc':None}
    pred=np.zeros(len(rows), dtype=np.float64)
    folds=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=n_splits).split(X,y,groups),1):
        clf=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.06,max_leaf_nodes=31,l2_regularization=0.01,random_state=100+fold)
        clf.fit(X[tr], y[tr].astype(int))
        p=clf.predict_proba(X[te])[:,1]
        pred[te]=p
        folds.append({'fold':fold,'n_test':int(len(te)),'auc':round(safe_auc(y[te],p),6) if safe_auc(y[te],p) is not None else None})
    return {'auc':round(safe_auc(y,pred),6),'folds':folds,'positive':int(y.sum()),'negative':int((~y).sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--in-jsonl', default=str(IN))
    ap.add_argument('--out-json', default=str(OUT))
    args=ap.parse_args()
    rows=load_rows(Path(args.in_jsonl))
    targets=['window_has_reentry','harmful_w16','helpful_w16','best_reject','best_long']
    result={'n_rows':len(rows),'targets':{},'feature_sets':['traj','app','combined']}
    for t in targets:
        tres={'labels':{'positive':int(label(rows,t).sum()),'negative':int((~label(rows,t)).sum())},'model_auc':{},'top_single_features':{}}
        for kind in ['traj','app','combined']:
            aucres=model_auc_cv(rows, kind, t)
            tres['model_auc'][kind]=aucres
            X,names=matrix(rows,kind)
            tres['top_single_features'][kind]=single_feature_auc(X,names,label(rows,t))[:12]
        result['targets'][t]=tres
    out=Path(args.out_json); out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False), flush=True)

if __name__=='__main__': main()
