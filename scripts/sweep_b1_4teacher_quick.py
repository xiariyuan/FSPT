#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

OUT = Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_quick')
OUT.mkdir(parents=True, exist_ok=True)
TEACHERS = ['online','offline','trackon2','tapnext']
PATHS = {
  'online':'outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt',
  'offline':'outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt',
  'trackon2':'caches/trackon2_strided_original.pt',
  'tapnext':'outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt',
}
BASE=0.6189

def npy(x, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def finite(p): return np.all(np.isfinite(p), axis=-1)

def masked_mean(p, mask):
    cnt=mask.sum(axis=0).astype(np.float32)[...,None]
    sm=np.where(mask[...,None], p, 0.0).sum(axis=0,dtype=np.float32)
    out=np.full(p.shape[1:], np.nan, dtype=np.float32)
    np.divide(sm,cnt,out=out,where=cnt>0)
    return out.astype(np.float32)

def masked_median(p, mask):
    with np.errstate(all='ignore'):
        return np.nanmedian(np.where(mask[...,None], p, np.nan), axis=0).astype(np.float32)

def fill(a,b):
    bad=~np.all(np.isfinite(a),axis=-1)
    if np.any(bad):
        a=a.copy(); a[bad]=b[bad]
    return a.astype(np.float32)

def px_scale(osz):
    h,w=float(osz[0]),float(osz[1])
    return np.array([max(h-1,1),max(w-1,1)],dtype=np.float32)

def gmean_pairwise(p, osz):
    pts=p*px_scale(osz)
    ds=[]
    for i in range(pts.shape[0]):
        for j in range(i+1,pts.shape[0]):
            ds.append(np.linalg.norm(pts[i]-pts[j],axis=-1))
    return np.mean(np.stack(ds,axis=0),axis=0).astype(np.float32)

def trule(p, v, osz, rule):
    fm=finite(p)
    if rule=='all_median4': return masked_median(p,fm)
    if rule=='visible_median4': return fill(masked_median(p,fm&v), masked_median(p,fm))
    if rule=='all_mean4': return masked_mean(p,fm)
    if rule=='all_median3': return masked_median(p[:3],fm[:3])
    if rule=='visible_median3': return fill(masked_median(p[:3],fm[:3]&v[:3]), masked_median(p[:3],fm[:3]))
    if rule=='tapnext': return p[3].astype(np.float32)
    raise ValueError(rule)

def vrule(v, p, osz, rule):
    if rule=='union4': return v.sum(axis=0)>=1
    if rule=='half4': return v.sum(axis=0)>=2
    if rule=='strict4': return v.sum(axis=0)>=3
    if rule.startswith('gated4_'):
        tau=float(rule.split('_',1)[1])
        agree=gmean_pairwise(p,osz)<=tau
        c=v.sum(axis=0)
        return np.where(agree,c>=1,c>=2).astype(bool)
    if rule=='union3': return v[:3].sum(axis=0)>=1
    if rule=='majority3': return v[:3].sum(axis=0)>=2
    if rule.startswith('gated3_'):
        tau=float(rule.split('_',1)[1])
        agree=gmean_pairwise(p[:3],osz)<=tau
        c=v[:3].sum(axis=0)
        return np.where(agree,c>=1,c>=2).astype(bool)
    if rule=='tapnext': return v[3].astype(bool)
    raise ValueError(rule)

def eval_cache(cp,jp):
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cp),'--output-json',str(jp)],check=True)
    return json.load(open(jp))

caches={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in PATHS.items()}
ref=caches['online']
for t in TEACHERS[1:]:
    assert len(caches[t]['records'])==len(ref['records']), t
    for i,(a,b) in enumerate(zip(ref['records'],caches[t]['records'])):
        assert a['video_id']==b['video_id'], (t,i,a['video_id'],b['video_id'])
        assert np.asarray(a['pred_tracks']).shape==np.asarray(b['pred_tracks']).shape, (t,i)
print({'loaded_teachers':TEACHERS,'records':len(ref['records'])}, flush=True)

configs=[
 ('old3_all_median_gated144','all_median3','gated3_144'),
 ('old3_all_median_union','all_median3','union3'),
 ('old3_visible_median_majority','visible_median3','majority3'),
 ('tapnext_single','tapnext','tapnext'),
 ('b1_all_median4_union','all_median4','union4'),
 ('b1_all_median4_half','all_median4','half4'),
 ('b1_all_median4_strict','all_median4','strict4'),
 ('b1_all_median4_gated096','all_median4','gated4_96'),
 ('b1_all_median4_gated128','all_median4','gated4_128'),
 ('b1_all_median4_gated144','all_median4','gated4_144'),
 ('b1_all_median4_gated160','all_median4','gated4_160'),
 ('b1_all_median4_gated192','all_median4','gated4_192'),
 ('b1_visible_median4_union','visible_median4','union4'),
 ('b1_visible_median4_half','visible_median4','half4'),
 ('b1_visible_median4_gated144','visible_median4','gated4_144'),
 ('b1_all_mean4_gated144','all_mean4','gated4_144'),
]
rows=[]
for name,tr,vr in configs:
    print('===',name,'===', flush=True)
    records=[]
    for idx, r0 in enumerate(ref['records']):
        p=np.stack([npy(caches[t]['records'][idx]['pred_tracks'],np.float32) for t in TEACHERS],axis=0)
        v=np.stack([npy(caches[t]['records'][idx]['pred_visibility'],bool) for t in TEACHERS],axis=0)
        osz=npy(r0['original_size'],np.float32)
        r=dict(r0)
        r['pred_tracks']=trule(p,v,osz,tr)
        r['pred_visibility']=vrule(v,p,osz,vr)
        r['model_name']='b1_quick_'+name
        r['route_tracks_rule']=tr
        r['route_visibility_rule']=vr
        records.append(r)
    out=dict(ref); out['model_name']='b1_quick_'+name; out['records']=records
    out['route_tracks_rule']=tr; out['route_visibility_rule']=vr
    cp=OUT/f'{name}.pt'; jp=OUT/f'{name}_ajrd.json'
    torch.save(out,cp)
    m=eval_cache(cp,jp)
    bd=m.get('aj_rd_by_dmin_256') or {}
    row={
      'name':name,'tracks':tr,'visibility':vr,
      'true_AJ_RD':m.get('true_AJ_RD'),
      'true_AJ_RD_256':m.get('true_AJ_RD_256'),
      'proxy':m.get('first_reentry_frame_proxy'),
      'dmin1_256':bd.get('1'),'dmin4_256':bd.get('4'),'dmin16_256':bd.get('16'),
      'long20_lt4px':(m.get('long_occ_ge20') or {}).get('lt4px'),
      'long20_lt8px':(m.get('long_occ_ge20') or {}).get('lt8px'),
      'delta_vs_06189':None if m.get('true_AJ_RD_256') is None else round(float(m.get('true_AJ_RD_256'))-BASE,4),
    }
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False), flush=True)
rows=sorted(rows,key=lambda x:x.get('true_AJ_RD_256') if x.get('true_AJ_RD_256') is not None else -1, reverse=True)
summary={'baseline_06189':BASE,'best':rows[0],'rows':rows}
json.dump(summary,open(OUT/'summary.json','w'),indent=2,ensure_ascii=False)
print('=== BEST ===')
print(json.dumps(rows[:10],indent=2,ensure_ascii=False), flush=True)
