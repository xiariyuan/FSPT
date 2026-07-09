#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT=Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import scripts.eval_reentry_guard_v1 as v1


def npy(x: Any, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def patch_feat(video: np.ndarray, t: int, xy: np.ndarray, radius: int=5) -> Optional[np.ndarray]:
    T,H,W,C=video.shape
    if t<0 or t>=T: return None
    x,y=float(xy[0]),float(xy[1])
    if not np.isfinite(x) or not np.isfinite(y): return None
    xi,yi=int(round(x)),int(round(y))
    if xi<0 or xi>=W or yi<0 or yi>=H: return None
    x0,x1=max(0,xi-radius),min(W,xi+radius+1)
    y0,y1=max(0,yi-radius),min(H,yi+radius+1)
    p=video[t,y0:y1,x0:x1].astype(np.float32)/255.0
    if p.size==0: return None
    mean=p.reshape(-1,3).mean(0); std=p.reshape(-1,3).std(0)
    # More discriminative than mean only: gradient-like spatial bins.
    bins=[]
    for gy in range(3):
        for gx in range(3):
            yy0=int(gy*p.shape[0]/3); yy1=max(yy0+1,int((gy+1)*p.shape[0]/3))
            xx0=int(gx*p.shape[1]/3); xx1=max(xx0+1,int((gx+1)*p.shape[1]/3))
            bins.extend(p[yy0:yy1,xx0:xx1].reshape(-1,3).mean(0).tolist())
    f=np.asarray(list(mean)+list(std)+bins,dtype=np.float32)
    f=f-f.mean(); n=np.linalg.norm(f)
    return f/(n+1e-8)


def cos(a,b):
    if a is None or b is None: return np.nan
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


def audit(setting: str, stress: Path, offline: Path, online: Path, b2: Path, labels_npz: Path, out_json: Path, radius: int=5):
    ds=torch.load(stress,map_location='cpu',weights_only=False)
    off=torch.load(offline,map_location='cpu',weights_only=False)
    on=torch.load(online,map_location='cpu',weights_only=False)
    b2c=torch.load(b2,map_location='cpu',weights_only=False)
    labels=np.load(labels_npz,allow_pickle=True)
    y=labels['y'].astype(np.int64); meta=labels['meta']
    vid_to_ds={str(r['video_id']):r for r in ds['records']}
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(off['records'])}
    rows=[]
    for row,label in zip(meta,y):
        vi,qi,vid=row; vid=str(vid); qi=int(qi)
        if vid not in vid_to_idx or vid not in vid_to_ds: continue
        i=vid_to_idx[vid]
        video=npy(vid_to_ds[vid]['video'],np.uint8)
        br=off['records'][i]; orr=on['records'][i]; b2r=b2c['records'][i]
        bv=npy(br['pred_visibility'],bool)[qi]; ov=npy(orr['pred_visibility'],bool)[qi]
        bp=npy(br['pred_tracks'],np.float32)[qi]
        op=npy(orr['pred_tracks'],np.float32)[qi]
        b2p=npy(b2r['pred_tracks'],np.float32)[qi]
        qpts=npy(br['query_points'],np.float32); qt=int(round(float(qpts[qi,0])))
        trig=v1.first_b2_trigger(bv,ov,qt,persist=2)
        if trig is None: continue
        prev=np.where(bv[max(0,qt):trig])[0]
        if len(prev)==0: continue
        last_t=max(0,qt)+int(prev[-1])
        f_last=patch_feat(video,last_t,bp[last_t],radius)
        f_b2=patch_feat(video,trig,b2p[trig],radius)
        f_on=patch_feat(video,trig,op[trig],radius)
        f_off=patch_feat(video,trig,bp[trig],radius)
        sim_b2=cos(f_last,f_b2); sim_on=cos(f_last,f_on); sim_off=cos(f_last,f_off)
        if not np.isfinite(sim_b2): continue
        rows.append({
            'label':int(label),'sim_b2':sim_b2,'sim_on':sim_on,'sim_off':sim_off,
            'sim_b2_minus_off':sim_b2-sim_off if np.isfinite(sim_off) else np.nan,
            'sim_on_minus_off':sim_on-sim_off if np.isfinite(sim_on) and np.isfinite(sim_off) else np.nan,
            'trig':int(trig),'last_t':int(last_t),'vid':vid,'qi':qi,
        })
    def vec(k): return np.asarray([r[k] for r in rows if np.isfinite(r[k])],dtype=np.float32)
    labels_arr=np.asarray([r['label'] for r in rows],dtype=np.int64)
    def auc(k):
        vals=np.asarray([r[k] for r in rows],dtype=np.float32)
        mask=np.isfinite(vals)
        if len(np.unique(labels_arr[mask]))<2: return None
        return float(roc_auc_score(labels_arr[mask],vals[mask]))
    def stats_by_label(k):
        vals=np.asarray([r[k] for r in rows],dtype=np.float32); mask=np.isfinite(vals)
        out={}
        for lab in [0,1]:
            a=vals[mask & (labels_arr==lab)]
            out[str(lab)]={'n':int(a.size),'mean':round(float(a.mean()),6) if a.size else None,'median':round(float(np.median(a)),6) if a.size else None,'p10':round(float(np.percentile(a,10)),6) if a.size else None,'p90':round(float(np.percentile(a,90)),6) if a.size else None}
        return out
    summary={
        'setting':setting,'n_rows':len(rows),'pos':int(labels_arr.sum()),'neg':int(len(labels_arr)-labels_arr.sum()),'radius':radius,
        'auc':{k:auc(k) for k in ['sim_b2','sim_on','sim_off','sim_b2_minus_off','sim_on_minus_off']},
        'stats_by_label':{k:stats_by_label(k) for k in ['sim_b2','sim_on','sim_off','sim_b2_minus_off','sim_on_minus_off']},
        'rows_sample':rows[:50]
    }
    out_json.parent.mkdir(parents=True,exist_ok=True); out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--setting',required=True); ap.add_argument('--stress',required=True); ap.add_argument('--offline',required=True); ap.add_argument('--online',required=True); ap.add_argument('--b2',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--out-json',required=True); ap.add_argument('--radius',type=int,default=5)
    a=ap.parse_args(); s=audit(a.setting,Path(a.stress),Path(a.offline),Path(a.online),Path(a.b2),Path(a.labels),Path(a.out_json),a.radius)
    print(json.dumps({k:s[k] for k in ['setting','n_rows','pos','neg','radius','auc','stats_by_label']},indent=2,ensure_ascii=False)[:6000])
if __name__=='__main__': main()
