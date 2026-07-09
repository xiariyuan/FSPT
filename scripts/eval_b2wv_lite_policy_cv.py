#!/usr/bin/env python3
from __future__ import annotations
import json, math, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.sweep_b2_w16_false_cost_guards_dev import standard_metrics
ROWS=Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/window_rows.jsonl')
OUT=Path('outputs/paper_discovery_2026-06-27/b2wv_lite_policy_cv')
DATASETS={
 'davis':{'base':Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),'override':Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),'b2_w16':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_base.pt'),'b2_p2':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt')},
 'rgb_dev10':{'base':Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),'override':Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),'b2_w16':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_base.pt'),'b2_p2':Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_persist2.pt')},
}
ACTIONS=['reject','W4','W8','W16']; ACTION_W={'reject':None,'W4':4,'W8':8,'W16':16}; LAM='lambda_1'
FEATURES=['trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16','override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16','base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16','vis_agreement_frac_next16','override_visibility_transitions_next16','base_visibility_transitions_next16','base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16','base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16']
POLICIES=['argmax','preserve95_dyn','strict95_dyn']
def load_rows():
 rows=[]
 for line in ROWS.open():
  if line.strip(): rows.append(json.loads(line))
 return rows
def fnum(v:Any):
 if v is None: return np.nan
 try: x=float(v)
 except Exception: return np.nan
 return x if math.isfinite(x) else np.nan
def Xmat(rows): return np.asarray([[fnum(r.get('features',{}).get(k)) for k in FEATURES] for r in rows],dtype=np.float32)
def yutil(rows,a): return np.asarray([float(r['actions'][a]['utility'][LAM]) for r in rows],dtype=np.float64)
def mkclf(seed): return HistGradientBoostingClassifier(max_iter=120,learning_rate=0.06,max_leaf_nodes=31,l2_regularization=0.01,random_state=seed)
def mkreg(seed): return HistGradientBoostingRegressor(max_iter=120,learning_rate=0.06,max_leaf_nodes=31,l2_regularization=0.01,random_state=seed)
def auc(y,s):
 return None if len(np.unique(y))<2 else round(float(roc_auc_score(y.astype(int),s)),6)
def th_ret(scores,labels,ret=.95):
 pos=scores[labels.astype(bool)]
 if len(pos)==0: return float(np.nanmin(scores)-1e-6)
 return float(np.quantile(pos,1-ret))
def choose(policy,p,us,thr):
 out=[]
 for i in range(len(p)):
  vals={a:float(us[a][i]) for a in ACTIONS}; ba=max(ACTIONS,key=lambda a:vals[a]); bn=max(['W4','W8','W16'],key=lambda a:vals[a])
  if policy=='argmax': a=ba
  elif policy=='preserve95_dyn': a=bn if p[i]>=thr else ba
  elif policy=='strict95_dyn': a=bn if p[i]>=thr else 'reject'
  else: raise ValueError(policy)
  out.append(a)
 return out
def row_metrics(rows,acts):
 c={a:0 for a in ACTIONS}; tg=[]; fd=[]; re=[]
 for r,a in zip(rows,acts):
  c[a]+=1; tg.append(float(r['actions'][a]['target_gain'])); fd.append(float(r['actions'][a]['delta_full_AJ'])); re.append(bool(r['window_has_reentry']))
 ar=np.asarray(acts); re=np.asarray(re,bool); n=len(rows)
 return {'n':n,'action_counts':c,'action_rates':{a:round(c[a]/max(n,1),6) for a in ACTIONS},'mean_target_gain':round(float(np.mean(tg)),6),'mean_delta_full_AJ':round(float(np.mean(fd)),6),'accept_rate':round(float(np.mean(ar!='reject')),6),'accept_rate_reentry_windows':round(float(np.mean(ar[re]!='reject')),6) if np.any(re) else None,'accept_rate_nonreentry_windows':round(float(np.mean(ar[~re]!='reject')),6) if np.any(~re) else None}
def npy(x,dtype=None):
 if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
 a=np.asarray(x); return a.astype(dtype) if dtype is not None else a
def lcache(p): return torch.load(p,map_location='cpu',weights_only=False)
def eval_ajrd(cp,out):
 subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cp),'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 return json.load(open(out))
def eval_payload(cp,payload,outdir,name):
 aj=eval_ajrd(cp,outdir/f'{name}_ajrd.json'); st=standard_metrics(payload['records'])
 return {'cache':str(cp),'AJ_RD_256':aj.get('true_AJ_RD_256'),'AJ_RD':aj.get('true_AJ_RD'),'first_reentry_frame_proxy':aj.get('first_reentry_frame_proxy'),'AJ_256':st['AJ_256_pct'],'OA_256':st['OA_256_pct'],'delta_avg_256':st['delta_avg_256_pct']}
def fit_oof(rows):
 X=Xmat(rows); y=np.asarray([bool(r['window_has_reentry']) for r in rows],bool); groups=np.asarray([f"{r['dataset']}::{r['video_id']}" for r in rows])
 gkf=GroupKFold(n_splits=min(5,len(np.unique(groups))))
 chosen={p:[None]*len(rows) for p in POLICIES}; folds=[]; oof=[]
 for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
  clf=mkclf(10+fold); clf.fit(X[tr],y[tr].astype(int)); ptr=clf.predict_proba(X[tr])[:,1]; pte=clf.predict_proba(X[te])[:,1]; thr=th_ret(ptr,y[tr],.95)
  us_te={};
  for ai,a in enumerate(ACTIONS):
   reg=mkreg(100+fold*10+ai); reg.fit(X[tr],yutil([rows[i] for i in tr],a)); us_te[a]=reg.predict(X[te])
  test_rows=[rows[i] for i in te]; fi={'fold':fold,'test_n':len(te),'threshold95':round(thr,6),'auc_reentry':auc(y[te],pte),'policies':{}}
  for pol in POLICIES:
   acts=choose(pol,pte,us_te,thr); fi['policies'][pol]=row_metrics(test_rows,acts)
   for idx,a in zip(te,acts): chosen[pol][int(idx)]=a
  for loc,idx in enumerate(te):
   rd={'row_index':int(idx),'dataset':rows[idx]['dataset'],'video_id':rows[idx]['video_id'],'query_idx':int(rows[idx]['query_idx']),'trigger_t':int(rows[idx]['trigger_t']),'p_reentry':round(float(pte[loc]),6)}
   for pol in POLICIES: rd[f'action_{pol}']=chosen[pol][idx]
   oof.append(rd)
  folds.append(fi)
 for pol in POLICIES: assert all(a is not None for a in chosen[pol])
 return {'chosen':chosen,'folds':folds,'oof_rows':oof}
def build_cache(ds,pol,rows,acts,cfg):
 base=lcache(cfg['base']); over=lcache(cfg['override']); by={}
 for r,a in zip(rows,acts):
  if r['dataset']!=ds: continue
  by.setdefault((str(r['video_id']),int(r['query_idx'])),[]).append((r,a))
 for v in by.values(): v.sort(key=lambda z:int(z[0]['trigger_t']))
 cnt={a:0 for a in ACTIONS}; tw=ta=0; recs=[]
 for b,o in zip(base['records'],over['records']):
  vid=str(b['video_id']); pt=npy(b['pred_tracks'],np.float32).copy(); pv=npy(b['pred_visibility'],bool).copy(); ot=npy(o['pred_tracks'],np.float32); ov=npy(o['pred_visibility'],bool); n,T=pv.shape
  for qi in range(n):
   cs=by.get((vid,int(qi)),[])
   if not cs: continue
   tw+=1; anyacc=False
   for r,a in cs:
    cnt[a]+=1; W=ACTION_W[a]
    if W is None: continue
    t=int(r['trigger_t']); lo=max(0,t-1); hi=min(T,t+int(W)+1); pt[qi,lo:hi]=ot[qi,lo:hi]; pv[qi,lo:hi]=ov[qi,lo:hi]; anyacc=True
   if anyacc: ta+=1
  rr=dict(b); rr['pred_tracks']=pt.astype(np.float32); rr['pred_visibility']=pv.astype(bool); rr['model_name']=f'b2wv_lite_{pol}_{ds}'; recs.append(rr)
 payload=dict(base); payload['model_name']=f'b2wv_lite_{pol}_{ds}'; payload['records']=recs
 stat={'action_counts':cnt,'total_candidate_windows':int(sum(cnt.values())),'tracks_with_candidate':tw,'tracks_with_accept':ta,'accept_rate_windows':round((cnt['W4']+cnt['W8']+cnt['W16'])/max(sum(cnt.values()),1),6)}
 return payload,stat
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=load_rows(); fit=fit_oof(rows)
 with (OUT/'oof_window_policy_rows.jsonl').open('w') as f:
  for r in fit['oof_rows']: f.write(json.dumps(r,ensure_ascii=False)+'\n')
 summ={'protocol':'B2-WV-Lite OOF window policies on DAVIS + RGB dev0-9 only','policies':POLICIES,'folds':fit['folds'],'row_level_oof':{},'datasets':{}}
 for pol in POLICIES: summ['row_level_oof'][pol]=row_metrics(rows,fit['chosen'][pol])
 for ds,cfg in DATASETS.items():
  od=OUT/ds; od.mkdir(parents=True,exist_ok=True); d={'baselines':{},'policies':{}}
  for nm,key in [('base','base'),('b2_w16','b2_w16'),('b2_p2','b2_p2')]:
   payload=lcache(cfg[key]); d['baselines'][nm]=eval_payload(cfg[key],payload,od,f'{nm}_{ds}')
  for pol in POLICIES:
   payload,stat=build_cache(ds,pol,rows,fit['chosen'][pol],cfg); cp=od/f'b2wv_lite_{pol}_{ds}.pt'; torch.save(payload,cp); met=eval_payload(cp,payload,od,f'b2wv_lite_{pol}_{ds}')
   p2=d['baselines']['b2_p2']; base=d['baselines']['base']; met['stats']=stat; met['delta_vs_p2']={'AJ_RD_256':round(float(met['AJ_RD_256'])-float(p2['AJ_RD_256']),6),'AJ_256':round(float(met['AJ_256'])-float(p2['AJ_256']),6)}; met['delta_vs_base']={'AJ_RD_256':round(float(met['AJ_RD_256'])-float(base['AJ_RD_256']),6),'AJ_256':round(float(met['AJ_256'])-float(base['AJ_256']),6)}; d['policies'][pol]=met
  summ['datasets'][ds]=d
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False)); print(json.dumps(summ,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
