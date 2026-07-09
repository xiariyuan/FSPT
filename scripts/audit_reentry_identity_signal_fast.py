#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def npy(x, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def first_trigger(bv, ov, qt, persist=2):
    T=len(bv)
    for t in range(max(1,qt+1),T-persist+1):
        if (not bool(bv[t-1])) and bool(np.all(ov[t:t+persist])):
            return t
    return None


def patch_desc(video,t,xy,r=5):
    T,H,W,C=video.shape
    if t is None or t<0 or t>=T: return None
    x,y=float(xy[0]),float(xy[1])
    if not np.isfinite(x+y): return None
    xi,yi=int(round(x)),int(round(y))
    if xi<0 or yi<0 or xi>=W or yi>=H: return None
    x0,x1=max(0,xi-r),min(W,xi+r+1); y0,y1=max(0,yi-r),min(H,yi+r+1)
    p=video[t,y0:y1,x0:x1].astype(np.float32)/255.0
    if p.size==0: return None
    # concise descriptor: RGB mean/std + 3x3 spatial means
    vals=[]; vals.extend(p.reshape(-1,3).mean(0)); vals.extend(p.reshape(-1,3).std(0))
    h,w=p.shape[:2]
    for gy in range(3):
      for gx in range(3):
        yy0=int(gy*h/3); yy1=max(yy0+1,int((gy+1)*h/3)); xx0=int(gx*w/3); xx1=max(xx0+1,int((gx+1)*w/3))
        vals.extend(p[yy0:yy1,xx0:xx1].reshape(-1,3).mean(0))
    f=np.asarray(vals,dtype=np.float32); f=f-f.mean(); n=np.linalg.norm(f)
    return f/(n+1e-8)


def cos(a,b):
    if a is None or b is None: return np.nan
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


def auc_score(y,s):
    y=np.asarray(y); s=np.asarray(s); m=np.isfinite(s); y=y[m]; s=s[m]
    if len(np.unique(y))<2: return None
    # rank AUC
    order=np.argsort(s)
    ranks=np.empty_like(order,dtype=np.float64); ranks[order]=np.arange(1,len(s)+1)
    npos=np.sum(y==1); nneg=np.sum(y==0)
    return float((np.sum(ranks[y==1])-npos*(npos+1)/2)/(npos*nneg))


def run(setting, stress, offline, online, b2, labels, out_json, per_class=400, radius=5):
    ds=torch.load(stress,map_location='cpu',weights_only=False)
    off=torch.load(offline,map_location='cpu',weights_only=False)
    on=torch.load(online,map_location='cpu',weights_only=False)
    b2c=torch.load(b2,map_location='cpu',weights_only=False)
    lab=np.load(labels,allow_pickle=True); y=lab['y'].astype(np.int64); meta=lab['meta']
    rng=np.random.default_rng(20260702)
    idx_pos=np.where(y==1)[0]; idx_neg=np.where(y==0)[0]
    idx=[]
    if len(idx_pos): idx.extend(rng.choice(idx_pos,size=min(per_class,len(idx_pos)),replace=False).tolist())
    if len(idx_neg): idx.extend(rng.choice(idx_neg,size=min(per_class,len(idx_neg)),replace=False).tolist())
    rng.shuffle(idx)
    vid_to_ds={str(r['video_id']):r for r in ds['records']}
    vid_to_i={str(r['video_id']):i for i,r in enumerate(off['records'])}
    rows=[]
    for j in idx:
        rec_i,qi,vid=meta[j]; qi=int(qi); vid=str(vid)
        if vid not in vid_to_i or vid not in vid_to_ds: continue
        i=vid_to_i[vid]
        video=npy(vid_to_ds[vid]['video'],np.uint8)
        br=off['records'][i]; orr=on['records'][i]; b2r=b2c['records'][i]
        bv=npy(br['pred_visibility'],bool)[qi]; ov=npy(orr['pred_visibility'],bool)[qi]
        bp=npy(br['pred_tracks'],np.float32)[qi]; op=npy(orr['pred_tracks'],np.float32)[qi]; b2p=npy(b2r['pred_tracks'],np.float32)[qi]
        qt=int(round(float(npy(br['query_points'],np.float32)[qi,0])))
        trig=first_trigger(bv,ov,qt)
        if trig is None: continue
        prev=np.where(bv[max(0,qt):trig])[0]
        if len(prev)==0: continue
        last_t=max(0,qt)+int(prev[-1])
        fl=patch_desc(video,last_t,bp[last_t],radius)
        sim_off=cos(fl,patch_desc(video,trig,bp[trig],radius))
        sim_on=cos(fl,patch_desc(video,trig,op[trig],radius))
        sim_b2=cos(fl,patch_desc(video,trig,b2p[trig],radius))
        rows.append({'label':int(y[j]),'sim_off':sim_off,'sim_on':sim_on,'sim_b2':sim_b2,'sim_b2_minus_off':sim_b2-sim_off if np.isfinite(sim_b2) and np.isfinite(sim_off) else np.nan,'sim_on_minus_off':sim_on-sim_off if np.isfinite(sim_on) and np.isfinite(sim_off) else np.nan})
    yy=[r['label'] for r in rows]
    def stat(k):
      vals=np.asarray([r[k] for r in rows],dtype=np.float32); out={}
      for labv in [0,1]:
        a=vals[np.asarray(yy)==labv]; a=a[np.isfinite(a)]
        out[str(labv)]={'n':int(a.size),'mean':round(float(a.mean()),6) if a.size else None,'median':round(float(np.median(a)),6) if a.size else None}
      return out
    summary={'setting':setting,'sample_rows':len(rows),'pos':int(sum(yy)),'neg':int(len(yy)-sum(yy)),'radius':radius,'auc':{k:auc_score(yy,[r[k] for r in rows]) for k in ['sim_off','sim_on','sim_b2','sim_b2_minus_off','sim_on_minus_off']},'stats':{k:stat(k) for k in ['sim_off','sim_on','sim_b2','sim_b2_minus_off','sim_on_minus_off']}}
    Path(out_json).parent.mkdir(parents=True,exist_ok=True); Path(out_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--setting',required=True); ap.add_argument('--stress',required=True); ap.add_argument('--offline',required=True); ap.add_argument('--online',required=True); ap.add_argument('--b2',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--out-json',required=True); ap.add_argument('--per-class',type=int,default=400); ap.add_argument('--radius',type=int,default=5)
    a=ap.parse_args(); run(a.setting,Path(a.stress),Path(a.offline),Path(a.online),Path(a.b2),Path(a.labels),Path(a.out_json),a.per_class,a.radius)
