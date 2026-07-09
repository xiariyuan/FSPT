#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT=Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_learned_gate')
OUT.mkdir(parents=True,exist_ok=True)
TEACHERS=['online','offline','trackon2','tapnext']
PATHS={
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

def masked_median(p):
    with np.errstate(all='ignore'):
        return np.nanmedian(p,axis=0).astype(np.float32)

def pairwise_stats(p256):
    ds=[]
    for i in range(p256.shape[0]):
        for j in range(i+1,p256.shape[0]):
            ds.append(np.linalg.norm(p256[i]-p256[j],axis=-1))
    d=np.stack(ds,axis=0).astype(np.float32)
    return d.mean(axis=0), d.max(axis=0), d.min(axis=0)

def build_video_features(caches, vidx):
    ref=caches['online']['records'][vidx]
    p=np.stack([npy(caches[t]['records'][vidx]['pred_tracks'],np.float32) for t in TEACHERS],axis=0) # M,N,T,2 norm yx
    v=np.stack([npy(caches[t]['records'][vidx]['pred_visibility'],bool) for t in TEACHERS],axis=0) # M,N,T
    gt=npy(ref['gt_tracks'],np.float32)
    gvis=npy(ref['gt_visibility'],bool)
    q=npy(ref['query_points'],np.float32)
    allmed=masked_median(p)
    oldmed=masked_median(p[:3])
    p256=p*255.0; allmed256=allmed*255.0; oldmed256=oldmed*255.0; gt256=gt*255.0
    err256=np.linalg.norm(allmed256-gt256,axis=-1).astype(np.float32)
    meanD,maxD,minD=pairwise_stats(p256)
    oldMeanD,oldMaxD,oldMinD=pairwise_stats(p256[:3])
    tap_old_dist=np.linalg.norm(p256[3]-oldmed256,axis=-1).astype(np.float32)
    dist_to_med=np.stack([np.linalg.norm(p256[i]-allmed256,axis=-1) for i in range(4)],axis=-1).astype(np.float32)
    N,T=gt.shape[:2]
    tt=np.broadcast_to(np.arange(T,dtype=np.float32)[None,:],(N,T))
    qt=np.broadcast_to(q[:,0:1].astype(np.float32),(N,T))
    age=tt-qt
    # Feature tensor N,T,F. Use log1p for distance features.
    feats=[]
    feats.append(v.transpose(1,2,0).astype(np.float32))
    feats.append(v.sum(axis=0)[...,None].astype(np.float32))
    feats.append(v[:3].sum(axis=0)[...,None].astype(np.float32))
    feats.append(np.log1p(meanD)[...,None]); feats.append(np.log1p(maxD)[...,None]); feats.append(np.log1p(minD)[...,None])
    feats.append(np.log1p(oldMeanD)[...,None]); feats.append(np.log1p(oldMaxD)[...,None])
    feats.append(np.log1p(tap_old_dist)[...,None])
    feats.append(np.log1p(dist_to_med))
    feats.append((tt/max(T-1,1))[...,None])
    feats.append((age/max(T-1,1))[...,None])
    feats.append((age>=0)[...,None].astype(np.float32))
    feats.append(np.broadcast_to(q[:,1:2],(N,T))[...,None])
    feats.append(np.broadcast_to(q[:,2:3],(N,T))[...,None])
    X=np.concatenate(feats,axis=-1).reshape(N*T,-1).astype(np.float32)
    flat={
      'X':X,
      'gt_vis':gvis.reshape(-1),
      'err256':err256.reshape(-1),
      'allmed':allmed,
      'shape':(N,T),
      'video_id':ref['video_id'],
    }
    return flat

def eval_cache(cp,jp):
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cp),'--output-json',str(jp)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(jp))

print('loading caches',flush=True)
caches={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in PATHS.items()}
ref=caches['online']
V=len(ref['records'])
print({'videos':V},flush=True)
vids=[build_video_features(caches,i) for i in range(V)]
X=np.concatenate([d['X'] for d in vids],axis=0)
groups=np.concatenate([np.full(d['X'].shape[0],i,dtype=np.int32) for i,d in enumerate(vids)],axis=0)
gtvis=np.concatenate([d['gt_vis'] for d in vids])
err=np.concatenate([d['err256'] for d in vids])
print({'samples':int(X.shape[0]),'features':int(X.shape[1]),'gt_vis_rate':round(float(gtvis.mean()),4)},flush=True)

configs=[]
for label_thr in [4,8,16]:
  for C in [0.1,0.3,1.0]:
    configs.append((label_thr,C))
prob_thresholds=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70]
rows=[]
for label_thr,C in configs:
    y=(gtvis & (err < float(label_thr))).astype(np.int8)
    print({'label_thr':label_thr,'C':C,'positive_rate':round(float(y.mean()),4)},flush=True)
    prob=np.zeros(X.shape[0],dtype=np.float32)
    gkf=GroupKFold(n_splits=5)
    for fold,(tr,te) in enumerate(gkf.split(X,y,groups)):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=C,class_weight='balanced',solver='lbfgs',n_jobs=1))
        clf.fit(X[tr],y[tr])
        prob[te]=clf.predict_proba(X[te])[:,1].astype(np.float32)
        print({'label_thr':label_thr,'C':C,'fold':fold,'test_samples':int(len(te)),'prob_mean':round(float(prob[te].mean()),4)},flush=True)
    # split probabilities back by video
    pos=0
    per_vid_probs=[]
    for d in vids:
        n=d['X'].shape[0]
        N,T=d['shape']
        per_vid_probs.append(prob[pos:pos+n].reshape(N,T))
        pos+=n
    for pthr in prob_thresholds:
        records=[]
        for i,r0 in enumerate(ref['records']):
            r=dict(r0)
            r['pred_tracks']=vids[i]['allmed'].astype(np.float32)
            r['pred_visibility']=(per_vid_probs[i] >= pthr).astype(bool)
            r['model_name']=f'b1_learned_gate_l{label_thr}_C{C}_p{pthr}'
            r['route_tracks_rule']='all_median4'
            r['route_visibility_rule']=f'learned_logreg_group5_label{label_thr}_C{C}_prob{pthr}'
            records.append(r)
        out=dict(ref); out['model_name']=f'b1_learned_gate_l{label_thr}_C{C}_p{pthr}'; out['records']=records
        cp=OUT/f'learned_l{label_thr}_C{str(C).replace(".","p")}_p{str(pthr).replace(".","p")}.pt'
        jp=OUT/f'learned_l{label_thr}_C{str(C).replace(".","p")}_p{str(pthr).replace(".","p")}_ajrd.json'
        torch.save(out,cp)
        m=eval_cache(cp,jp)
        bd=m.get('aj_rd_by_dmin_256') or {}
        row={'name':cp.stem,'label_thr':label_thr,'C':C,'prob_thr':pthr,'true_AJ_RD':m.get('true_AJ_RD'),'true_AJ_RD_256':m.get('true_AJ_RD_256'),'proxy':m.get('first_reentry_frame_proxy'),'dmin1_256':bd.get('1'),'dmin4_256':bd.get('4'),'dmin16_256':bd.get('16'),'long20_lt4px':(m.get('long_occ_ge20') or {}).get('lt4px'),'long20_lt8px':(m.get('long_occ_ge20') or {}).get('lt8px')}
        row['delta_vs_06264']=None if row['true_AJ_RD_256'] is None else round(float(row['true_AJ_RD_256'])-BASE,4)
        row['delta_vs_old3']=None if row['true_AJ_RD_256'] is None else round(float(row['true_AJ_RD_256'])-OLD,4)
        rows.append(row)
        if (row['true_AJ_RD_256'] or 0)>=BASE:
            print('BEAT_OR_TIE',json.dumps(row,ensure_ascii=False),flush=True)
rows=sorted(rows,key=lambda x:x.get('true_AJ_RD_256') if x.get('true_AJ_RD_256') is not None else -1,reverse=True)
summary={'baseline_heuristic_06264':BASE,'old3_06189':OLD,'best':rows[0],'rows':rows}
json.dump(summary,open(OUT/'summary.json','w'),indent=2,ensure_ascii=False)
print('=== BEST LEARNED GATE ===')
print(json.dumps(rows[:20],indent=2,ensure_ascii=False),flush=True)
