#!/usr/bin/env python3
from __future__ import annotations
import json, math, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.sweep_b2_w16_false_cost_guards_dev import standard_metrics
ROWS=Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/window_rows.jsonl')
OUT=Path('outputs/paper_discovery_2026-06-27/b2wv_p2safe_cv')
DATASETS={
'davis':{'base':Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),'override':Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),'p2':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt')},
'rgb_dev10':{'base':Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),'override':Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),'p2':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_persist2.pt')},
}
FEATS=['trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16','override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16','base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16','vis_agreement_frac_next16','override_visibility_transitions_next16','base_visibility_transitions_next16','base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16','base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16']
POLICIES=['p2_short_re99','p2_short_re97','p2_short_re95','p2_veto_re99','p2_veto_re97']
ACTION_W={'W16':16,'W4':4,'reject':None}
def load_rows():
 rows=[]
 for line in ROWS.open():
  if not line.strip(): continue
  r=json.loads(line)
  if float(r['features'].get('override_persist_len_cap16') or 0) >= 2: rows.append(r)
 return rows
def fnum(v:Any):
 if v is None: return np.nan
 try: x=float(v)
 except Exception: return np.nan
 return x if math.isfinite(x) else np.nan
def X(rows): return np.asarray([[fnum(r.get('features',{}).get(k)) for k in FEATS] for r in rows],dtype=np.float32)
def clf(seed): return HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=31,l2_regularization=.01,random_state=seed)
def thresh(scores,labels,retain):
 pos=scores[labels.astype(bool)]
 if len(pos)==0: return float(np.nanmin(scores)-1e-6)
 return float(np.quantile(pos,1-retain))
def auc(y,s): return None if len(np.unique(y))<2 else round(float(roc_auc_score(y.astype(int),s)),6)
def choose(pol,p,th):
 if pol.endswith('99'): retain=.99
 elif pol.endswith('97'): retain=.97
 else: retain=.95
 acts=[]
 for pi in p:
  if pi>=th[retain]: acts.append('W16')
  else: acts.append('W4' if 'short' in pol else 'reject')
 return acts
def rowmet(rows,acts):
 c={'W16':0,'W4':0,'reject':0}; re=[]
 for r,a in zip(rows,acts): c[a]+=1; re.append(bool(r['window_has_reentry']))
 ar=np.asarray(acts); re=np.asarray(re,bool); n=len(rows)
 return {'n':n,'counts':c,'rates':{k:round(v/max(n,1),6) for k,v in c.items()},'accept_rate':round(float(np.mean(ar!='reject')),6),'w16_rate':round(float(np.mean(ar=='W16')),6),'reentry_w16_rate':round(float(np.mean(ar[re]=='W16')),6) if np.any(re) else None,'nonreentry_modified_rate':round(float(np.mean(ar[~re]!='W16')),6) if np.any(~re) else None}
def oof(rows):
 xx=X(rows); y=np.asarray([bool(r['window_has_reentry']) for r in rows],bool); groups=np.asarray([f"{r['dataset']}::{r['video_id']}" for r in rows])
 gkf=GroupKFold(n_splits=min(5,len(np.unique(groups)))); chosen={p:[None]*len(rows) for p in POLICIES}; folds=[]; dump=[]
 for fi,(tr,te) in enumerate(gkf.split(xx,y,groups),1):
  m=clf(100+fi); m.fit(xx[tr],y[tr].astype(int)); ptr=m.predict_proba(xx[tr])[:,1]; pte=m.predict_proba(xx[te])[:,1]
  th={r:thresh(ptr,y[tr],r) for r in [.95,.97,.99]}; test=[rows[i] for i in te]
  f={'fold':fi,'test_n':len(te),'auc_reentry':auc(y[te],pte),'thresholds':{str(k):round(v,6) for k,v in th.items()},'policies':{}}
  for pol in POLICIES:
   acts=choose(pol,pte,th); f['policies'][pol]=rowmet(test,acts)
   for idx,a in zip(te,acts): chosen[pol][int(idx)]=a
  for loc,idx in enumerate(te):
   rd={'idx':int(idx),'dataset':rows[idx]['dataset'],'video_id':rows[idx]['video_id'],'query_idx':int(rows[idx]['query_idx']),'p_reentry':round(float(pte[loc]),6)}
   for pol in POLICIES: rd[f'action_{pol}']=chosen[pol][idx]
   dump.append(rd)
  folds.append(f)
 return {'chosen':chosen,'folds':folds,'dump':dump}
def npy(x,d=None):
 if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
 a=np.asarray(x); return a.astype(d) if d is not None else a
def lcache(p): return torch.load(p,map_location='cpu',weights_only=False)
def eval_ajrd(cp,out):
 subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cp),'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 return json.load(open(out))
def ev(cp,payload,od,name):
 aj=eval_ajrd(cp,od/f'{name}_ajrd.json'); st=standard_metrics(payload['records'])
 return {'AJ_RD_256':aj.get('true_AJ_RD_256'),'AJ_256':st['AJ_256_pct'],'OA_256':st['OA_256_pct'],'delta_avg_256':st['delta_avg_256_pct'],'cache':str(cp)}
def build(ds,pol,rows,acts,cfg):
 base=lcache(cfg['base']); over=lcache(cfg['override']); by={}
 for r,a in zip(rows,acts):
  if r['dataset']!=ds: continue
  by.setdefault((str(r['video_id']),int(r['query_idx'])),[]).append((r,a))
 for v in by.values(): v.sort(key=lambda z:int(z[0]['trigger_t']))
 cnt={'W16':0,'W4':0,'reject':0}; recs=[]
 for b,o in zip(base['records'],over['records']):
  vid=str(b['video_id']); pt=npy(b['pred_tracks'],np.float32).copy(); pv=npy(b['pred_visibility'],bool).copy(); ot=npy(o['pred_tracks'],np.float32); ov=npy(o['pred_visibility'],bool); n,T=pv.shape
  for qi in range(n):
   for r,a in by.get((vid,int(qi)),[]):
    cnt[a]+=1; W=ACTION_W[a]
    if W is None: continue
    t=int(r['trigger_t']); lo=max(0,t-1); hi=min(T,t+W+1); pt[qi,lo:hi]=ot[qi,lo:hi]; pv[qi,lo:hi]=ov[qi,lo:hi]
  rr=dict(b); rr['pred_tracks']=pt.astype(np.float32); rr['pred_visibility']=pv.astype(bool); rr['model_name']=f'b2wv_p2safe_{pol}_{ds}'; recs.append(rr)
 payload=dict(base); payload['model_name']=f'b2wv_p2safe_{pol}_{ds}'; payload['records']=recs
 return payload,{'action_counts':cnt,'total_windows':sum(cnt.values()),'w16_rate':round(cnt['W16']/max(sum(cnt.values()),1),6)}
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=load_rows(); fit=oof(rows)
 with (OUT/'oof_rows.jsonl').open('w') as f:
  for r in fit['dump']: f.write(json.dumps(r,ensure_ascii=False)+'\n')
 summ={'protocol':'P2-safe window shortening/veto, dev only, p2 candidates only','n_rows':len(rows),'folds':fit['folds'],'row_oof':{},'datasets':{}}
 for pol in POLICIES: summ['row_oof'][pol]=rowmet(rows,fit['chosen'][pol])
 for ds,cfg in DATASETS.items():
  od=OUT/ds; od.mkdir(parents=True,exist_ok=True); d={'baselines':{},'policies':{}}
  for nm,key in [('base','base'),('b2_p2','p2')]: d['baselines'][nm]=ev(cfg[key],lcache(cfg[key]),od,f'{nm}_{ds}')
  for pol in POLICIES:
   payload,stat=build(ds,pol,rows,fit['chosen'][pol],cfg); cp=od/f'b2wv_p2safe_{pol}_{ds}.pt'; torch.save(payload,cp); met=ev(cp,payload,od,f'b2wv_p2safe_{pol}_{ds}'); p2=d['baselines']['b2_p2']; base=d['baselines']['base']; met['stats']=stat; met['delta_vs_p2']={'AJ_RD_256':round(float(met['AJ_RD_256'])-float(p2['AJ_RD_256']),6),'AJ_256':round(float(met['AJ_256'])-float(p2['AJ_256']),6)}; met['delta_vs_base']={'AJ_RD_256':round(float(met['AJ_RD_256'])-float(base['AJ_RD_256']),6),'AJ_256':round(float(met['AJ_256'])-float(base['AJ_256']),6)}; d['policies'][pol]=met
  summ['datasets'][ds]=d
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False)); print(json.dumps(summ,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
