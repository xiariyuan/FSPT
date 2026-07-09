#!/usr/bin/env python3
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

PRED_DIR=Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions')
OUT_DIR=Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/eval')
BASE=PRED_DIR/'cotracker3_offline_translate_L16.pt'
OVER=PRED_DIR/'cotracker3_online_translate_L16.pt'


def npy(x:Any,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def invisible_run_before(v,t):
    c=0; j=int(t)-1
    while j>=0 and not bool(v[j]): c+=1; j-=1
    return c

def trigger_ok(base_v, over_v, t, persist=2, k=1):
    if invisible_run_before(base_v,t)<k: return False
    if t+persist>len(over_v): return False
    return bool(np.all(over_v[t:t+persist]))

def build_b2_p2(base,over,post=16,pre=1,persist=2,k=1):
    records=[]
    stats={'total_trigger_events':0,'tracks_with_trigger':0,'gt_reentry_tracks':0,'triggered_reentry_tracks':0,'triggered_nonreentry_tracks':0,'missed_reentry_tracks':0}
    for b,o in zip(base['records'],over['records']):
        pred_p=npy(b['pred_tracks'],np.float32).copy(); pred_v=npy(b['pred_visibility'],bool).copy()
        base_v=pred_v.copy(); over_p=npy(o['pred_tracks'],np.float32); over_v=npy(o['pred_visibility'],bool)
        gt_v=npy(b['gt_visibility'],bool); q=npy(b['query_points'],np.float32)
        n,T=pred_v.shape
        for qi in range(n):
            qt=int(round(float(q[qi,0])))
            has_re=bool(find_reentry_events(gt_v[qi],qt))
            if has_re: stats['gt_reentry_tracks']+=1
            mask=np.zeros(T,dtype=bool); triggers=[]
            t=max(1,qt+1)
            while t<T:
                if trigger_ok(base_v[qi],over_v[qi],t,persist=persist,k=k):
                    lo=max(0,t-pre); hi=min(T,t+post+1)
                    mask[lo:hi]=True; triggers.append(t); stats['total_trigger_events']+=1
                    t=hi
                else:
                    t+=1
            if triggers:
                stats['tracks_with_trigger']+=1
                if has_re: stats['triggered_reentry_tracks']+=1
                else: stats['triggered_nonreentry_tracks']+=1
                pred_p[qi,mask]=over_p[qi,mask]; pred_v[qi,mask]=over_v[qi,mask]
            elif has_re:
                stats['missed_reentry_tracks']+=1
        r=dict(b); r['pred_tracks']=pred_p.astype(np.float32); r['pred_visibility']=pred_v.astype(bool); r['model_name']='b2_w16_p2_translate_L16'; records.append(r)
    stats['trigger_precision_track']=round(stats['triggered_reentry_tracks']/max(stats['tracks_with_trigger'],1),6)
    stats['trigger_recall_track']=round(stats['triggered_reentry_tracks']/max(stats['gt_reentry_tracks'],1),6)
    payload=dict(base); payload['model_name']='b2_w16_p2_translate_L16'; payload['b2_method']='B2-W16-P2'; payload['records']=records
    return payload,stats

def standard_metrics(records):
    aj=[]; oa=[]; da=[]
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32))
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='strided')
        aj.append(float(m.get('AJ',0))); oa.append(float(m.get('OA',0))); da.append(float(m.get('average_pts_within_thresh',0)))
    return {'AJ_256':round(float(np.mean(aj))*100,4),'OA_256':round(float(np.mean(oa))*100,4),'delta_avg_256':round(float(np.mean(da))*100,4)}

def eval_ajrd(cache,out):
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cache),'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(out))

def recovery_rates(records):
    # simple RR@0/4/8 at 8px, visible+within threshold at reentry+timestep.
    counts={0:[0,0],4:[0,0],8:[0,0]}
    for r in records:
        pred=npy(r['pred_tracks'],np.float32); pv=npy(r['pred_visibility'],bool); gt=npy(r['gt_tracks'],np.float32); gv=npy(r['gt_visibility'],bool); q=npy(r['query_points'],np.float32)
        H,W=map(int,npy(r['original_size'],np.int32))
        for qi in range(gt.shape[0]):
            qt=int(round(float(q[qi,0]))); evs=find_reentry_events(gv[qi],qt)
            if not evs: continue
            rt=int(evs[0]['reentry_frame'])
            for tau in counts:
                tt=min(gt.shape[1]-1,rt+tau)
                counts[tau][1]+=1
                err=np.linalg.norm((pred[qi,tt]-gt[qi,tt])*np.array([H-1,W-1],dtype=np.float32))
                if bool(pv[qi,tt]) and bool(gv[qi,tt]) and err<=8:
                    counts[tau][0]+=1
    return {f'RR@{tau}_8px':round(ok/max(total,1),6) for tau,(ok,total) in counts.items()}

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    base=torch.load(BASE,map_location='cpu',weights_only=False); over=torch.load(OVER,map_location='cpu',weights_only=False)
    b2,stats=build_b2_p2(base,over)
    b2_cache=PRED_DIR/'b2_w16_p2_translate_L16.pt'; torch.save(b2,b2_cache)
    caches={'offline':BASE,'online':OVER,'b2_w16_p2':b2_cache}
    rows={}
    for name,cache in caches.items():
        payload=torch.load(cache,map_location='cpu',weights_only=False)
        ajrd=eval_ajrd(cache,OUT_DIR/f'{name}_ajrd.json')
        std=standard_metrics(payload['records'])
        rr=recovery_rates(payload['records'])
        rows[name]={
            'cache':str(cache),
            'AJ_RD_256':ajrd.get('true_AJ_RD_256'),
            'AJ_RD':ajrd.get('true_AJ_RD'),
            'first_reentry_frame_proxy':ajrd.get('first_reentry_frame_proxy'),
            **std,
            **rr,
        }
    rows['b2_w16_p2']['trigger_stats']=stats
    # deltas
    off=rows['offline']; on=rows['online']; b2r=rows['b2_w16_p2']
    deltas={
        'b2_vs_offline':{'AJ_RD_256':round(b2r['AJ_RD_256']-off['AJ_RD_256'],6),'AJ_256':round(b2r['AJ_256']-off['AJ_256'],6)},
        'online_vs_offline':{'AJ_RD_256':round(on['AJ_RD_256']-off['AJ_RD_256'],6),'AJ_256':round(on['AJ_256']-off['AJ_256'],6)},
        'b2_vs_online':{'AJ_RD_256':round(b2r['AJ_RD_256']-on['AJ_RD_256'],6),'AJ_256':round(b2r['AJ_256']-on['AJ_256'],6)},
    }
    sanity=json.load(open('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/sanity_summary.json'))
    summary={'stress_sanity':{k:sanity[k] for k in ['num_videos','num_kept_queries','reentry_query_rate','num_reentry_events','occ_length_mean','occ_length_median','nan_count','visible_oob_count','sanity_pass']},'methods':rows,'deltas':deltas}
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
