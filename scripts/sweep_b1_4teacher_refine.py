#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

OUT = Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine')
OUT.mkdir(parents=True, exist_ok=True)
TEACHERS = ['online','offline','trackon2','tapnext']
PATHS = {
  'online':'outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt',
  'offline':'outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt',
  'trackon2':'caches/trackon2_strided_original.pt',
  'tapnext':'outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt',
}
BASE=0.6264
OLD=0.6189

def npy(x,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def finite(p): return np.all(np.isfinite(p),axis=-1)

def masked_mean(p,m):
    cnt=m.sum(axis=0).astype(np.float32)[...,None]
    sm=np.where(m[...,None],p,0).sum(axis=0,dtype=np.float32)
    out=np.full(p.shape[1:],np.nan,dtype=np.float32)
    np.divide(sm,cnt,out=out,where=cnt>0)
    return out.astype(np.float32)

def masked_median(p,m):
    with np.errstate(all='ignore'):
        return np.nanmedian(np.where(m[...,None],p,np.nan),axis=0).astype(np.float32)

def fill(a,b):
    bad=~np.all(np.isfinite(a),axis=-1)
    if np.any(bad):
        a=a.copy(); a[bad]=b[bad]
    return a.astype(np.float32)

def px_scale(osz):
    h,w=float(osz[0]),float(osz[1])
    return np.array([max(h-1,1),max(w-1,1)],dtype=np.float32)

def gmean_pairwise(p,osz):
    pts=p*px_scale(osz); ds=[]
    for i in range(pts.shape[0]):
        for j in range(i+1,pts.shape[0]):
            ds.append(np.linalg.norm(pts[i]-pts[j],axis=-1))
    return np.mean(np.stack(ds,axis=0),axis=0).astype(np.float32)

def pairwise_to_refmean(p,osz,ref_ids):
    pts=p*px_scale(osz)
    ref=np.mean(pts[ref_ids],axis=0)
    return np.linalg.norm(pts-ref[None],axis=-1).astype(np.float32)

def track_build(p,v,osz,rule):
    fm=finite(p)
    if rule.startswith('median_'):
        ids=[int(x) for x in rule.split('_')[1:]]
        return masked_median(p[ids],fm[ids])
    if rule.startswith('visiblemedian_'):
        ids=[int(x) for x in rule.split('_')[1:]]
        return fill(masked_median(p[ids],fm[ids]&v[ids]), masked_median(p[ids],fm[ids]))
    if rule.startswith('mean_'):
        ids=[int(x) for x in rule.split('_')[1:]]
        return masked_mean(p[ids],fm[ids])
    if rule.startswith('hybrid4old3_tau'):
        # use all4 median when TAPNext agrees with old3 mean, else old3 median
        tau=float(rule.replace('hybrid4old3_tau',''))
        old=masked_median(p[:3],fm[:3])
        all4=masked_median(p,fm)
        oldmean=masked_mean(p[:3],fm[:3])
        tapdist=np.linalg.norm((p[3]-oldmean)*px_scale(osz),axis=-1)
        return np.where((tapdist<=tau)[...,None], all4, old).astype(np.float32)
    if rule.startswith('trim4_tau'):
        # choose median4 when all-teacher disagreement under tau, else old3 median
        tau=float(rule.replace('trim4_tau',''))
        old=masked_median(p[:3],fm[:3]); all4=masked_median(p,fm)
        agree=gmean_pairwise(p,osz)<=tau
        return np.where(agree[...,None], all4, old).astype(np.float32)
    raise ValueError(rule)

def vis_build(v,p,osz,rule):
    if rule.startswith('union_'):
        ids=[int(x) for x in rule.split('_')[1:]]; return v[ids].sum(axis=0)>=1
    if rule.startswith('half_'):
        ids=[int(x) for x in rule.split('_')[1:]]; return v[ids].sum(axis=0)>=2
    if rule.startswith('maj_'):
        ids=[int(x) for x in rule.split('_')[1:]]; need=len(ids)//2+1; return v[ids].sum(axis=0)>=need
    if rule.startswith('gated_'):
        # gated_<tau>_<ids...>: union on agree, half-vote on disagree
        parts=rule.split('_'); tau=float(parts[1]); ids=[int(x) for x in parts[2:]]
        subp=p[ids]; subv=v[ids]
        agree=gmean_pairwise(subp,osz)<=tau
        c=subv.sum(axis=0)
        return np.where(agree,c>=1,c>=max(2, len(ids)//2+1 if len(ids)>=5 else 2)).astype(bool)
    if rule.startswith('gatedstrict_'):
        parts=rule.split('_'); tau=float(parts[1]); ids=[int(x) for x in parts[2:]]
        subp=p[ids]; subv=v[ids]
        agree=gmean_pairwise(subp,osz)<=tau
        c=subv.sum(axis=0)
        return np.where(agree,c>=1,c>=min(len(ids),3)).astype(bool)
    raise ValueError(rule)

def eval_cache(cp,jp):
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cp),'--output-json',str(jp)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(jp))

caches={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in PATHS.items()}
ref=caches['online']
print({'records':len(ref['records']),'teachers':TEACHERS},flush=True)

configs=[]
ids4='0_1_2_3'; ids3='0_1_2'
# Fine tau around previous best and above.
for tau in [144,160,176,192,208,224,240,256,288,320,384,512,768]:
    configs.append((f'all4_gated{tau}',f'median_{ids4}',f'gated_{tau}_{ids4}'))
    configs.append((f'vis4_gated{tau}',f'visiblemedian_{ids4}',f'gated_{tau}_{ids4}'))
# TAPNext visibility only; tracks old3.
for tau in [96,128,144,160,192,224,256,320,512]:
    configs.append((f'old3tracks_vis4_gated{tau}',f'median_{ids3}',f'gated_{tau}_{ids4}'))
configs += [
    ('old3tracks_vis4_union',f'median_{ids3}',f'union_{ids4}'),
    ('old3tracks_vis4_half',f'median_{ids3}',f'half_{ids4}'),
]
# Teacher subsets including TAPNext.
for subset in [(0,1,3),(0,2,3),(1,2,3),(0,1,2),(0,1,2,3)]:
    s='_'.join(map(str,subset))
    for tau in [144,192,256]:
        configs.append((f'median{s}_gated{tau}',f'median_{s}',f'gated_{tau}_{s}'))
        configs.append((f'vismedian{s}_gated{tau}',f'visiblemedian_{s}',f'gated_{tau}_{s}'))
    configs.append((f'median{s}_union',f'median_{s}',f'union_{s}'))
# Hybrid tracks: all4 only if TAPNext agrees with old3.
for ttau in [16,24,32,48,64,96,128,192,256]:
    for vtau in [144,192,256]:
        configs.append((f'hybridTrack{ttau}_visG{vtau}',f'hybrid4old3_tau{ttau}',f'gated_{vtau}_{ids4}'))
        configs.append((f'trimTrack{ttau}_visG{vtau}',f'trim4_tau{ttau}',f'gated_{vtau}_{ids4}'))

# Dedupe while preserving order
seen=set(); unique=[]
for c in configs:
    if c[0] not in seen:
        seen.add(c[0]); unique.append(c)
configs=unique
print({'n_configs':len(configs)},flush=True)
rows=[]
for name,tr,vr in configs:
    records=[]
    for idx,r0 in enumerate(ref['records']):
        p=np.stack([npy(caches[t]['records'][idx]['pred_tracks'],np.float32) for t in TEACHERS],axis=0)
        v=np.stack([npy(caches[t]['records'][idx]['pred_visibility'],bool) for t in TEACHERS],axis=0)
        osz=npy(r0['original_size'],np.float32)
        r=dict(r0)
        r['pred_tracks']=track_build(p,v,osz,tr)
        r['pred_visibility']=vis_build(v,p,osz,vr)
        r['model_name']='b1_refine_'+name
        r['route_tracks_rule']=tr; r['route_visibility_rule']=vr
        records.append(r)
    out=dict(ref); out['model_name']='b1_refine_'+name; out['records']=records
    out['route_tracks_rule']=tr; out['route_visibility_rule']=vr
    cp=OUT/f'{name}.pt'; jp=OUT/f'{name}_ajrd.json'
    torch.save(out,cp)
    m=eval_cache(cp,jp)
    bd=m.get('aj_rd_by_dmin_256') or {}
    row={'name':name,'tracks':tr,'visibility':vr,'true_AJ_RD':m.get('true_AJ_RD'),'true_AJ_RD_256':m.get('true_AJ_RD_256'),'proxy':m.get('first_reentry_frame_proxy'),'dmin1_256':bd.get('1'),'dmin4_256':bd.get('4'),'dmin16_256':bd.get('16'),'long20_lt4px':(m.get('long_occ_ge20') or {}).get('lt4px'),'long20_lt8px':(m.get('long_occ_ge20') or {}).get('lt8px')}
    row['delta_vs_quick_best']=None if row['true_AJ_RD_256'] is None else round(float(row['true_AJ_RD_256'])-BASE,4)
    row['delta_vs_old3']=None if row['true_AJ_RD_256'] is None else round(float(row['true_AJ_RD_256'])-OLD,4)
    rows.append(row)
    if len(rows)%10==0 or (row['true_AJ_RD_256'] or 0)>BASE:
        print('ROW',json.dumps(row,ensure_ascii=False),flush=True)
rows=sorted(rows,key=lambda x:x.get('true_AJ_RD_256') if x.get('true_AJ_RD_256') is not None else -1,reverse=True)
summary={'quick_best_06264':BASE,'old3_06189':OLD,'best':rows[0],'rows':rows}
json.dump(summary,open(OUT/'summary.json','w'),indent=2,ensure_ascii=False)
print('=== BEST REFINE ===')
print(json.dumps(rows[:20],indent=2,ensure_ascii=False),flush=True)
