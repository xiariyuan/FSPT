#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
IN=Path('outputs/paper_discovery_2026-06-27/b2wa_clip_pilot/rgb_dev10_clip_patch_rows.jsonl')
OUT=Path('outputs/paper_discovery_2026-06-27/b2wa_clip_pilot/clip_signal_summary.json')
TRAJ=['trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16','override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16','base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16','vis_agreement_frac_next16','override_visibility_transitions_next16','base_visibility_transitions_next16','base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16','base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16']
def rows():
 o=[]
 for line in IN.open():
  if line.strip(): o.append(json.loads(line))
 return o
def fnum(v):
 if v is None: return np.nan
 try:x=float(v)
 except Exception:return np.nan
 return x if math.isfinite(x) else np.nan
def names(rs,kind):
 clip=sorted(rs[0]['clip_features'].keys())
 if kind=='traj': return TRAJ
 if kind=='clip': return ['clip::'+k for k in clip]
 return TRAJ+['clip::'+k for k in clip]
def mat(rs,kind):
 ns=names(rs,kind); X=[]
 for r in rs:
  vals=[]
  for n in ns:
   vals.append(fnum(r['clip_features'].get(n[6:])) if n.startswith('clip::') else fnum(r['traj_features'].get(n)))
  X.append(vals)
 return np.asarray(X,np.float32),ns
def lab(rs,t):
 if t=='window_has_reentry': return np.asarray([bool(r['window_has_reentry']) for r in rs],bool)
 if t=='harmful_w16': return np.asarray([bool(r['w16_harmful_full']) for r in rs],bool)
 if t=='helpful_w16': return np.asarray([bool(r['w16_helpful_target']) for r in rs],bool)
 if t=='best_reject': return np.asarray([r['best_action']=='reject' for r in rs],bool)
 if t=='best_long': return np.asarray([r['best_action'] in ('W8','W16') for r in rs],bool)
 raise ValueError(t)
def auc(y,s): return None if len(np.unique(y))<2 else float(roc_auc_score(y.astype(int),s))
def single(X,ns,y):
 out=[]
 for j,n in enumerate(ns):
  x=X[:,j].copy(); m=np.isfinite(x)
  if m.sum()<10 or len(np.unique(y[m]))<2: continue
  med=float(np.nanmedian(x)) if np.any(np.isfinite(x)) else 0.0; x[~m]=med
  a=auc(y,x)
  if a is None: continue
  out.append({'feature':n,'auc_positive_high':round(a,6),'best_auc':round(max(a,1-a),6),'direction':'high' if a>=0.5 else 'low'})
 out.sort(key=lambda z:z['best_auc'],reverse=True); return out[:12]
def model(rs,kind,t):
 X,ns=mat(rs,kind); y=lab(rs,t); g=np.asarray([r['video_id'] for r in rs])
 if len(np.unique(y))<2: return {'auc':None}
 pred=np.zeros(len(rs)); folds=[]
 for i,(tr,te) in enumerate(GroupKFold(n_splits=min(5,len(np.unique(g)))).split(X,y,g),1):
  clf=HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=31,l2_regularization=.01,random_state=300+i)
  clf.fit(X[tr],y[tr].astype(int)); p=clf.predict_proba(X[te])[:,1]; pred[te]=p
  folds.append({'fold':i,'n_test':int(len(te)),'auc':round(auc(y[te],p),6) if auc(y[te],p) is not None else None})
 return {'auc':round(auc(y,pred),6),'folds':folds,'positive':int(y.sum()),'negative':int((~y).sum())}
def main():
 rs=rows(); targets=['window_has_reentry','harmful_w16','helpful_w16','best_reject','best_long']
 res={'n_rows':len(rs),'targets':{}}
 for t in targets:
  tr={'labels':{'positive':int(lab(rs,t).sum()),'negative':int((~lab(rs,t)).sum())},'model_auc':{},'top_single_features':{}}
  for kind in ['traj','clip','combined']:
   tr['model_auc'][kind]=model(rs,kind,t); X,ns=mat(rs,kind); tr['top_single_features'][kind]=single(X,ns,lab(rs,t))
  res['targets'][t]=tr
 OUT.write_text(json.dumps(res,indent=2,ensure_ascii=False)); print(json.dumps(res,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
