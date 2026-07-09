#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def patch_feat(video: np.ndarray, t: int, xy: np.ndarray, radius: int = 5) -> Optional[np.ndarray]:
    T,H,W,C=video.shape
    if t < 0 or t >= T: return None
    x=float(xy[0]); y=float(xy[1])
    if not np.isfinite(x) or not np.isfinite(y): return None
    xi=int(round(x)); yi=int(round(y))
    if xi < 0 or xi >= W or yi < 0 or yi >= H: return None
    x0=max(0,xi-radius); x1=min(W,xi+radius+1)
    y0=max(0,yi-radius); y1=min(H,yi+radius+1)
    p=video[t,y0:y1,x0:x1].astype(np.float32)/255.0
    if p.size == 0: return None
    # simple identity descriptor: RGB mean/std + center patch flattened low-res histogram-ish stats
    mean=p.reshape(-1,3).mean(axis=0)
    std=p.reshape(-1,3).std(axis=0)
    # 4x4 spatial RGB average grid for local texture/layout
    grid=[]
    for gy in range(4):
        for gx in range(4):
            yy0=int(round(gy*p.shape[0]/4)); yy1=int(round((gy+1)*p.shape[0]/4))
            xx0=int(round(gx*p.shape[1]/4)); xx1=int(round((gx+1)*p.shape[1]/4))
            cell=p[yy0:max(yy1,yy0+1), xx0:max(xx1,xx0+1)]
            grid.extend(cell.reshape(-1,3).mean(axis=0).tolist())
    f=np.asarray(list(mean)+list(std)+grid,dtype=np.float32)
    f=f - f.mean()
    n=np.linalg.norm(f)
    if n < 1e-8: return f
    return f/n


def cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None: return None
    return float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-8)))


def eligible_events(gt_vis: np.ndarray, qt: int) -> List[Tuple[int,int,int]]:
    # return (occ_start, reentry_t, occ_len) for invisible run followed by visible after query
    out=[]; T=len(gt_vis); t=max(0,qt+1)
    while t<T:
        if bool(gt_vis[t]):
            t+=1; continue
        s=t
        while t<T and not bool(gt_vis[t]): t+=1
        e=t
        if e<T and bool(gt_vis[e]):
            out.append((s,e,e-s))
        t+=1
    return out


def sample_negative(video: np.ndarray, t: int, gt_xy: np.ndarray, rng: np.random.Generator, min_dist: float = 32.0, tries: int = 20) -> Optional[np.ndarray]:
    H,W=video.shape[1],video.shape[2]
    for _ in range(tries):
        xy=np.asarray([rng.uniform(0,W-1), rng.uniform(0,H-1)], dtype=np.float32)
        if np.linalg.norm(xy-gt_xy) >= min_dist:
            return xy
    return None


def analyze(stress_dataset: Path, offline_cache: Path, online_cache: Path, b2_cache: Path, out_json: Path, max_queries_per_video: int = 0, radius: int = 5) -> Dict[str,Any]:
    ds=torch.load(stress_dataset,map_location='cpu',weights_only=False)
    off=torch.load(offline_cache,map_location='cpu',weights_only=False)
    on=torch.load(online_cache,map_location='cpu',weights_only=False)
    b2=torch.load(b2_cache,map_location='cpu',weights_only=False)
    rng=np.random.default_rng(20260702)
    rows=[]
    sim_pos=[]; sim_neg=[]
    sim_last_to_off=[]; sim_last_to_on=[]; sim_last_to_b2=[]
    correct_off=[]; correct_on=[]; correct_b2=[]
    # Evaluate at first reentry event only per eligible query for this audit.
    for rec_i,(sr,or_,onr,b2r) in enumerate(zip(ds['records'],off['records'],on['records'],b2['records'])):
        video=npy(sr['video'],np.uint8)
        gt=npy(sr['gt_tracks'],np.float32); gv=npy(sr['gt_visibility'],bool); q=npy(sr['query_points'],np.float32)
        op=npy(or_['pred_tracks'],np.float32); onp=npy(onr['pred_tracks'],np.float32); b2p=npy(b2r['pred_tracks'],np.float32)
        n=q.shape[0]
        q_indices=range(n)
        if max_queries_per_video and n>max_queries_per_video:
            q_indices=rng.choice(n,size=max_queries_per_video,replace=False)
        for qi in q_indices:
            qt=int(round(float(q[qi,0])))
            evs=eligible_events(gv[qi],qt)
            if not evs: continue
            s,rt,occ_len=evs[0]
            # last visible before occlusion start, after query
            prev=np.where(gv[qi,max(0,qt):s])[0]
            if len(prev)==0: continue
            last_t=max(0,qt)+int(prev[-1])
            f_last=patch_feat(video,last_t,gt[qi,last_t],radius)
            f_gt=patch_feat(video,rt,gt[qi,rt],radius)
            neg_xy=sample_negative(video,rt,gt[qi,rt],rng)
            f_neg=patch_feat(video,rt,neg_xy,radius) if neg_xy is not None else None
            sp=cos(f_last,f_gt); sn=cos(f_last,f_neg)
            if sp is None or sn is None: continue
            sim_pos.append(sp); sim_neg.append(sn)
            f_off=patch_feat(video,rt,op[qi,rt],radius)
            f_on=patch_feat(video,rt,onp[qi,rt],radius)
            f_b2=patch_feat(video,rt,b2p[qi,rt],radius)
            so=cos(f_last,f_off); son=cos(f_last,f_on); sb=cos(f_last,f_b2)
            if so is not None: sim_last_to_off.append(so)
            if son is not None: sim_last_to_on.append(son)
            if sb is not None: sim_last_to_b2.append(sb)
            # coordinate correctness at 8px threshold as proxy
            eoff=float(np.linalg.norm(op[qi,rt]-gt[qi,rt])); eon=float(np.linalg.norm(onp[qi,rt]-gt[qi,rt])); eb2=float(np.linalg.norm(b2p[qi,rt]-gt[qi,rt]))
            correct_off.append(eoff<8); correct_on.append(eon<8); correct_b2.append(eb2<8)
            rows.append({'video_id':str(sr['video_id']),'qi':int(qi),'qt':qt,'last_t':int(last_t),'reentry_t':int(rt),'occ_len':int(occ_len),'sim_pos':sp,'sim_neg':sn,'sim_off':so,'sim_on':son,'sim_b2':sb,'err_off':eoff,'err_on':eon,'err_b2':eb2})
    y=np.asarray([1]*len(sim_pos)+[0]*len(sim_neg),dtype=np.int64)
    scores=np.asarray(sim_pos+sim_neg,dtype=np.float32)
    auc=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 and len(scores)>0 else None
    def stat(x):
        a=np.asarray(x,dtype=np.float32)
        if a.size==0: return {'n':0}
        return {'n':int(a.size),'mean':round(float(a.mean()),6),'std':round(float(a.std()),6),'median':round(float(np.median(a)),6),'p10':round(float(np.percentile(a,10)),6),'p90':round(float(np.percentile(a,90)),6)}
    summary={
        'stress_dataset':str(stress_dataset),
        'n_events':len(rows),
        'patch_radius':radius,
        'identity_positive_vs_random_negative_auc':auc,
        'sim_last_visible_to_gt_reentry':stat(sim_pos),
        'sim_last_visible_to_random_negative':stat(sim_neg),
        'sim_last_visible_to_offline_pred_at_reentry':stat(sim_last_to_off),
        'sim_last_visible_to_online_pred_at_reentry':stat(sim_last_to_on),
        'sim_last_visible_to_b2_pred_at_reentry':stat(sim_last_to_b2),
        'coord_correct_at_8px':{
            'offline':round(float(np.mean(correct_off)),6) if correct_off else None,
            'online':round(float(np.mean(correct_on)),6) if correct_on else None,
            'b2':round(float(np.mean(correct_b2)),6) if correct_b2 else None,
        },
        'rows_sample':rows[:50],
    }
    out_json.parent.mkdir(parents=True,exist_ok=True)
    out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stress',required=True)
    ap.add_argument('--offline',required=True)
    ap.add_argument('--online',required=True)
    ap.add_argument('--b2',required=True)
    ap.add_argument('--out-json',required=True)
    ap.add_argument('--max-queries-per-video',type=int,default=0)
    ap.add_argument('--radius',type=int,default=5)
    args=ap.parse_args()
    s=analyze(Path(args.stress),Path(args.offline),Path(args.online),Path(args.b2),Path(args.out_json),args.max_queries_per_video,args.radius)
    print(json.dumps(s,indent=2,ensure_ascii=False)[:5000])

if __name__=='__main__': main()
