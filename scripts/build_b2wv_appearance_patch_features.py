#!/usr/bin/env python3
"""Build lightweight RGB patch appearance features for B2-WV.

This script is intentionally opt-in and not run by default in the pipeline,
because TAPVid DAVIS/RGB pkl files are ~2.3GB each and loading them is costly.
It extracts small color/gradient descriptors around query, last-visible base,
and override candidate points, then writes a merged JSONL for window-level
verification experiments.
"""
from __future__ import annotations
import argparse, json, pickle, math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

ACTIONS = ['reject', 'W4', 'W8', 'W16']

def load_rows(path: Path) -> List[Dict[str, Any]]:
    out=[]
    with path.open() as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def yx_norm_to_pixel(yx, h, w):
    y=float(yx[0])*(h-1); x=float(yx[1])*(w-1)
    return y,x

def patch_desc(video: np.ndarray, t: int, yx_norm, radius: int=5) -> Dict[str, Any]:
    t=max(0,min(video.shape[0]-1,int(t)))
    frame=video[t]
    if frame.dtype != np.float32 and frame.dtype != np.float64:
        fr=frame.astype(np.float32)/255.0
    else:
        fr=frame.astype(np.float32)
        if fr.max()>1.5: fr=fr/255.0
    h,w=fr.shape[:2]
    y,x=yx_norm_to_pixel(yx_norm,h,w)
    cy=int(round(y)); cx=int(round(x)); r=int(radius)
    y0=max(0,cy-r); y1=min(h,cy+r+1); x0=max(0,cx-r); x1=min(w,cx+r+1)
    p=fr[y0:y1,x0:x1]
    if p.size==0:
        return {'ok':False}
    mean=p.reshape(-1,3).mean(axis=0)
    std=p.reshape(-1,3).std(axis=0)
    # simple gradient energy
    gray=p.mean(axis=2)
    gy=np.diff(gray,axis=0) if gray.shape[0]>1 else np.zeros((1,1),np.float32)
    gx=np.diff(gray,axis=1) if gray.shape[1]>1 else np.zeros((1,1),np.float32)
    ge=float(np.mean(np.abs(gx)))+float(np.mean(np.abs(gy)))
    return {'ok':True,'mean_rgb':[float(v) for v in mean],'std_rgb':[float(v) for v in std],'grad_energy':ge,'patch_h':int(p.shape[0]),'patch_w':int(p.shape[1])}

def desc_vec(d):
    if not d.get('ok'): return None
    return np.asarray(d['mean_rgb']+d['std_rgb']+[d['grad_energy']],dtype=np.float32)

def l2(a,b):
    if a is None or b is None: return None
    return float(np.linalg.norm(a-b))

def cosine(a,b):
    if a is None or b is None: return None
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<1e-9 or nb<1e-9: return None
    return float(np.dot(a,b)/(na*nb))

def last_visible_before(v: np.ndarray, t: int):
    for j in range(int(t)-1,-1,-1):
        if bool(v[j]): return j
    return None

def index_video_entries(pkl_path: Path):
    with pkl_path.open('rb') as f:
        data=pickle.load(f)
    return data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows', default='outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/rgb_dev10_window_rows.jsonl')
    ap.add_argument('--cache', default='outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
    ap.add_argument('--override-cache', default='outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
    ap.add_argument('--rgb-pkl', default='/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl')
    ap.add_argument('--out', default='outputs/paper_discovery_2026-06-27/b2wv_appearance/rgb_dev10_patch_rows.jsonl')
    ap.add_argument('--max-rows', type=int, default=2000)
    ap.add_argument('--radius', type=int, default=5)
    args=ap.parse_args()
    import torch
    rows=load_rows(Path(args.rows))[:max(0,args.max_rows)] if args.max_rows>0 else load_rows(Path(args.rows))
    base=torch.load(args.cache,map_location='cpu',weights_only=False)
    over=torch.load(args.override_cache,map_location='cpu',weights_only=False)
    print('loading large RGB pkl; this may take minutes...')
    data=index_video_entries(Path(args.rgb_pkl))
    by_name={f'rgb_stacking_{i:06d}': entry for i,entry in enumerate(data)}
    base_by_vid={str(r['video_id']):r for r in base['records']}
    over_by_vid={str(r['video_id']):r for r in over['records']}
    outp=Path(args.out); outp.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with outp.open('w') as f:
        for row in rows:
            vid=row['video_id']
            if vid not in by_name: continue
            entry=by_name[vid]; video=np.asarray(entry['video'])
            b=base_by_vid[vid]; o=over_by_vid[vid]
            qi=int(row['query_idx']); t=int(row['trigger_t']); qt=int(row['query_t'])
            base_tr=np.asarray(b['pred_tracks'])[qi]; base_v=np.asarray(b['pred_visibility'])[qi]
            over_tr=np.asarray(o['pred_tracks'])[qi]
            lv=last_visible_before(base_v,t)
            q_desc=patch_desc(video,qt,base_tr[qt],args.radius)
            lv_desc=patch_desc(video,lv,base_tr[lv],args.radius) if lv is not None else {'ok':False}
            ov_desc=patch_desc(video,t,over_tr[t],args.radius)
            qv,lvv,ovv=desc_vec(q_desc),desc_vec(lv_desc),desc_vec(ov_desc)
            app={
                'query_to_override_l2': l2(qv,ovv),
                'lastvis_to_override_l2': l2(lvv,ovv),
                'query_to_override_cos': cosine(qv,ovv),
                'lastvis_to_override_cos': cosine(lvv,ovv),
                'query_patch_ok': bool(q_desc.get('ok')),
                'lastvis_patch_ok': bool(lv_desc.get('ok')),
                'override_patch_ok': bool(ov_desc.get('ok')),
                'last_visible_t': lv,
            }
            row2=dict(row); row2['appearance_features']=app
            f.write(json.dumps(row2,ensure_ascii=False)+'\n'); n+=1
    print('wrote',outp,'rows',n)
if __name__=='__main__': main()
