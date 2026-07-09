#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c=0; j=int(t)-1
    while j>=0 and not bool(v[j]): c+=1; j-=1
    return c


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, persist: int=2, k: int=1) -> bool:
    if invisible_run_before(base_v,t) < k: return False
    if t + persist > len(over_v): return False
    return bool(np.all(over_v[t:t+persist]))


def check_alignment(base: Dict[str, Any], over: Dict[str, Any]) -> None:
    if len(base['records']) != len(over['records']): raise ValueError('record count mismatch')
    for i,(b,o) in enumerate(zip(base['records'], over['records'])):
        if str(b['video_id']) != str(o['video_id']): raise ValueError(f'video_id mismatch {i}: {b["video_id"]} vs {o["video_id"]}')
        for k in ['query_points','gt_tracks','gt_visibility','original_size']:
            if not np.allclose(npy(b[k]), npy(o[k]), atol=1e-6, rtol=1e-6):
                raise ValueError(f'alignment mismatch {i} {k}')


def build_b2_p2(base: Dict[str, Any], over: Dict[str, Any], name: str, pre: int=1, post: int=16, persist: int=2) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out_records=[]
    total_events=tracks_with_trigger=true_t=false_t=missed_re=gt_re=0
    per_video=[]
    for b,o in zip(base['records'], over['records']):
        pred_p=npy(b['pred_tracks'],np.float32).copy()
        pred_v=npy(b['pred_visibility'],bool).copy()
        base_v=pred_v.copy()
        over_p=npy(o['pred_tracks'],np.float32)
        over_v=npy(o['pred_visibility'],bool)
        gt_v=npy(b['gt_visibility'],bool)
        qpts=npy(b['query_points'],np.float32)
        n,T=pred_v.shape
        vs={'video_id':str(b['video_id']),'tracks':int(n),'events':0,'tracks_with_trigger':0,'true_tracks':0,'false_tracks':0,'missed_reentry':0,'gt_reentry':0}
        for qi in range(n):
            qt=int(round(float(qpts[qi,0]))); qt=max(0,min(T-1,qt))
            has_re=bool(find_reentry_events(gt_v[qi], qt))
            if has_re: gt_re+=1; vs['gt_reentry']+=1
            mask=np.zeros(T,dtype=bool); triggers=[]
            t=max(1,qt+1)
            while t<T:
                if trigger_ok(base_v[qi], over_v[qi], t, persist=persist, k=1):
                    lo=max(0,t-pre); hi=min(T,t+post+1)
                    mask[lo:hi]=True
                    triggers.append(int(t))
                    t=hi
                else:
                    t+=1
            if triggers:
                total_events += len(triggers); tracks_with_trigger+=1
                vs['events'] += len(triggers); vs['tracks_with_trigger'] += 1
                if has_re: true_t+=1; vs['true_tracks']+=1
                else: false_t+=1; vs['false_tracks']+=1
                pred_p[qi,mask]=over_p[qi,mask]
                pred_v[qi,mask]=over_v[qi,mask]
            elif has_re:
                missed_re+=1; vs['missed_reentry']+=1
        r=dict(b); r['pred_tracks']=pred_p.astype(np.float32); r['pred_visibility']=pred_v.astype(bool); r['model_name']=name
        r['b2w_config']={'k':1,'pre':pre,'post':post,'persist':persist,'trigger':'base invisible run >=1 and override visible for persist frames'}
        out_records.append(r); per_video.append(vs)
    payload=dict(base); payload['model_name']=name; payload['b2w_config']={'pre':pre,'post':post,'persist':persist}; payload['records']=out_records
    stats={'total_trigger_events':int(total_events),'tracks_with_trigger':int(tracks_with_trigger),'gt_reentry_tracks':int(gt_re),'triggered_reentry_tracks':int(true_t),'triggered_nonreentry_tracks':int(false_t),'missed_reentry_tracks':int(missed_re),'trigger_precision_track':round(true_t/max(tracks_with_trigger,1),6),'trigger_recall_track':round(true_t/max(gt_re,1),6),'per_video_trigger_stats':per_video}
    return payload,stats


def eval_ajrd(cache: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cache),'--output-json',str(out_json)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(out_json))


def standard_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    aj=[]; oa=[]; da=[]
    for r in payload['records']:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32))
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='strided')
        aj.append(float(m.get('AJ',0.0))); oa.append(float(m.get('OA',0.0))); da.append(float(m.get('average_pts_within_thresh',0.0)))
    return {'AJ_256':round(float(np.mean(aj))*100,4),'OA_256':round(float(np.mean(oa))*100,4),'delta_avg_256':round(float(np.mean(da))*100,4)}


def eval_payload(label: str, cache_path: Path, out_dir: Path) -> Dict[str, Any]:
    payload=torch.load(cache_path,map_location='cpu',weights_only=False)
    ajrd=eval_ajrd(cache_path,out_dir/f'{label}_ajrd.json')
    std=standard_metrics(payload)
    return {'label':label,'cache':str(cache_path),'AJ_RD_256':ajrd.get('true_AJ_RD_256'),'AJ_RD':ajrd.get('true_AJ_RD'),'first_reentry_frame_proxy':ajrd.get('first_reentry_frame_proxy'),**std}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-cache',required=True)
    ap.add_argument('--override-cache',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--name',default='b2_w16_p2_reentry_stress')
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    base=torch.load(args.base_cache,map_location='cpu',weights_only=False)
    over=torch.load(args.override_cache,map_location='cpu',weights_only=False)
    check_alignment(base,over)
    b2,stats=build_b2_p2(base,over,args.name,pre=1,post=16,persist=2)
    b2_cache=out_dir/f'{args.name}.pt'
    torch.save(b2,b2_cache)
    rows=[]
    rows.append(eval_payload('offline',Path(args.base_cache),out_dir))
    rows.append(eval_payload('online',Path(args.override_cache),out_dir))
    rows.append(eval_payload('b2_w16_p2',b2_cache,out_dir))
    # deltas
    off=rows[0]; on=rows[1]; b=rows[2]
    for r in rows:
        r['delta_AJ_RD_vs_offline']=round(float(r['AJ_RD_256'])-float(off['AJ_RD_256']),6)
        r['delta_AJ_vs_offline']=round(float(r['AJ_256'])-float(off['AJ_256']),6)
        r['delta_AJ_RD_vs_online']=round(float(r['AJ_RD_256'])-float(on['AJ_RD_256']),6)
        r['delta_AJ_vs_online']=round(float(r['AJ_256'])-float(on['AJ_256']),6)
    summary={'protocol':'ReEntry stress translate L16 smoke: offline vs online vs B2-W16-P2','base_cache':str(args.base_cache),'override_cache':str(args.override_cache),'b2_cache':str(b2_cache),'rows':rows,'b2_trigger_stats':stats}
    (out_dir/'stress_eval_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
