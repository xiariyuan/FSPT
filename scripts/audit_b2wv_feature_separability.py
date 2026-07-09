#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROWS = Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/window_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/feature_separability_summary.json')
LAM = 'lambda_1'
FEATURES = [
    'trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16',
    'override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16',
    'base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16','vis_agreement_frac_next16',
    'override_visibility_transitions_next16','base_visibility_transitions_next16',
    'base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16',
    'base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16',
]


def load_rows() -> List[Dict[str, Any]]:
    rows=[]
    with ROWS.open() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def val(r, f):
    v=r.get('features',{}).get(f)
    if v is None: return None
    try:
        x=float(v)
    except Exception:
        return None
    if not math.isfinite(x): return None
    return x


def ranks_average(x: np.ndarray) -> np.ndarray:
    order=np.argsort(x)
    ranks=np.empty_like(x,dtype=float)
    i=0; n=len(x)
    while i<n:
        j=i+1
        while j<n and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2.0
        i=j
    return ranks


def auc(scores, labels):
    pairs=[(s,l) for s,l in zip(scores,labels) if s is not None]
    if not pairs: return None
    x=np.array([p[0] for p in pairs],dtype=float)
    y=np.array([bool(p[1]) for p in pairs],dtype=bool)
    npos=int(y.sum()); nneg=int((~y).sum())
    if npos==0 or nneg==0: return None
    r=ranks_average(x)
    sp=float(r[y].sum())
    return float((sp - npos*(npos+1)/2.0)/(npos*nneg))


def feature_table(rows, label_fn):
    labels=[label_fn(r) for r in rows]
    out=[]
    for f in FEATURES:
        scores=[val(r,f) for r in rows]
        a=auc(scores,labels)
        if a is None: continue
        out.append({'feature':f,'auc_positive_high':round(a,6),'best_auc':round(max(a,1-a),6),'direction':'high' if a>=0.5 else 'low'})
    out.sort(key=lambda z:z['best_auc'], reverse=True)
    return out


def label_counts(rows, label_fn):
    ys=[bool(label_fn(r)) for r in rows]
    return {'positive':int(sum(ys)), 'negative':int(len(ys)-sum(ys)), 'positive_rate':round(sum(ys)/max(len(ys),1),6)}


def summarize_subset(name, rows):
    return {
        'n':len(rows),
        'labels':{
            'best_accept': label_counts(rows, lambda r: r['best_action'][LAM] != 'reject'),
            'best_W16': label_counts(rows, lambda r: r['best_action'][LAM] == 'W16'),
            'best_reject': label_counts(rows, lambda r: r['best_action'][LAM] == 'reject'),
            'window_has_reentry': label_counts(rows, lambda r: r['window_has_reentry']),
        },
        'top_auc':{
            'best_accept': feature_table(rows, lambda r: r['best_action'][LAM] != 'reject')[:12],
            'best_W16': feature_table(rows, lambda r: r['best_action'][LAM] == 'W16')[:12],
            'best_reject': feature_table(rows, lambda r: r['best_action'][LAM] == 'reject')[:12],
            'window_has_reentry': feature_table(rows, lambda r: r['window_has_reentry'])[:12],
        }
    }


def main():
    rows=load_rows()
    out={'lambda':LAM,'all':summarize_subset('all', rows), 'by_dataset':{}}
    for ds in sorted(set(r['dataset'] for r in rows)):
        dr=[r for r in rows if r['dataset']==ds]
        out['by_dataset'][ds]=summarize_subset(ds, dr)
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
