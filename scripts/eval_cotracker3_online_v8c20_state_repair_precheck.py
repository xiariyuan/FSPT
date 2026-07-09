#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, math, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'baselines/cotracker'))
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from scripts.export_cotracker3_online_v7a4_raw_visconf_components import load_video_for_record, make_record_from_state
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, CANDIDATE, align_candidate, build_events, clone_records, standard_and_ajrd, apply_events, err_px, npy
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native

OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c20_state_repair_precheck'
CKPT=ROOT/'baselines/cotracker/checkpoints/scaled_online.pth'
DAVIS=ROOT/'datasets/tapvid_davis/tapvid_davis.pkl'

def event_touch_set(events, native_by, cand_by, W=8, dist_thr=64.0):
    touched=defaultdict(dict); stats=Counter()
    for e in events:
        nr=native_by[e.video_id]; cr=cand_by[e.video_id]
        ntr=npy(nr['pred_tracks'],np.float32); ctr=npy(cr['pred_tracks'],np.float32); cvis=npy(cr['pred_visibility'],bool)
        q=e.query_idx; t0=e.t
        if err_px(ntr[q,t0], ctr[q,t0])>dist_thr: continue
        stats['selected_events']+=1
        T=ctr.shape[1]
        hit=False
        for tau in range(t0,min(T,t0+W+1)):
            if not bool(cvis[q,tau]): continue
            touched[e.video_id][(q,tau)]=ctr[q,tau].copy()
            stats['unique_touched_frames_total']+=1
            hit=True
        if hit: stats['events_with_touch']+=1
    # unique_touched_frames_total above overcounts if duplicate keys; recompute.
    stats['unique_touched_frames_total']=sum(len(v) for v in touched.values())
    return touched, dict(stats)

def write_state(model, rec, touch_map, mode, applied):
    if model.model.online_coords_predicted is None: return 0
    state=model.model.online_coords_predicted
    T=min(int(npy(rec['gt_tracks']).shape[1]), state.shape[1])
    H,W=model.interp_shape
    count=0
    logit95=math.log(0.95/0.05)
    for (q,tau), yx in touch_map.items():
        if tau>=T: continue
        key=(q,tau,mode)
        # Re-apply after each chunk is fine for state propagation, but count once.
        if mode!='repeat' and key in applied: pass
        y=float(yx[0]); x=float(yx[1])
        state[0,tau,q,0]=x*(W-1)
        state[0,tau,q,1]=y*(H-1)
        if mode=='coords_visconf_high':
            model.model.online_vis_predicted[0,tau,q]=logit95
            model.model.online_conf_predicted[0,tau,q]=logit95
        if key not in applied:
            applied.add(key); count+=1
    return count

def run_stream_with_repair(base_rec, ds, cand_by, touch_map, device, mode):
    model=CoTrackerOnlinePredictor(checkpoint=str(CKPT)).to(device).eval()
    video, queries = load_video_for_record(base_rec, ds, device)
    B,T,C,H,W=video.shape
    model(video_chunk=video, is_first_step=True, queries=queries, add_support_grid=False, grid_size=0)
    applied=set(); calls=0; t0=time.time()
    autocast_enabled=device.startswith('cuda')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_enabled):
        for ind in range(0, T-model.step, model.step):
            chunk=video[:, ind:ind+model.step*2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
            calls+=1
            write_state(model, base_rec, touch_map, mode, applied)
    rec=make_record_from_state(base_rec, model, 'v8c20_state_'+mode)
    return rec, {'calls':calls,'applied_unique':len(applied),'sec':round(time.time()-t0,3)}

def subset_records(records, names):
    ns=set(names)
    return [r for r in records if str(r['video_id']) in ns]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--videos',default='bike-packing,parkour,camel,shooting')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    videos=None if args.videos.strip().lower()=='all' else [x.strip() for x in args.videos.split(',') if x.strip()]
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    cand=torch.load(CANDIDATE,map_location='cpu',weights_only=False)
    ok,info,cand_by=align_candidate(native,cand)
    if not ok: raise RuntimeError(info)
    nby={str(r['video_id']):r for r in native['records']}
    metas=rebuild_events_from_native(native)
    events=[e for e in build_events(native,cand_by,metas) if e.video_id in set(videos)]
    touched, touch_stats=event_touch_set(events,nby,cand_by,W=8,dist_thr=64.0)
    base_subset=list(native['records']) if videos is None else subset_records(native['records'],videos)
    if videos is None:
        videos=[str(r['video_id']) for r in base_subset]
    nm,npv=standard_and_ajrd(clone_records(base_subset))
    # output-level baseline on same subset.
    sel=np.ones(len(events),dtype=bool)
    out_recs,out_stats=apply_events(base_subset,cand_by,events,sel,window=8,apply_mode='candidate_visible')
    out_metric,out_pv=standard_and_ajrd(out_recs)
    ds=TapVidDataset(str(DAVIS),dataset_type='davis',resize_to=[256,256],queried_first=True)
    state_rows=[]
    for mode in ['coords_only','coords_visconf_high']:
        recs=[]; per=[]
        for r in base_subset:
            vid=str(r['video_id'])
            rec,st=run_stream_with_repair(r,ds,cand_by,touched.get(vid,{}),args.device,mode)
            recs.append(rec); per.append({'video_id':vid,**st,'touches_available':len(touched.get(vid,{}))})
        m,pv=standard_and_ajrd(recs)
        state_rows.append({'mode':mode,'metric':m,'delta_vs_native':{k:(m[k]-nm[k] if m.get(k) is not None and nm.get(k) is not None else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},'per_run':per,'per_video':pv})
    report={'script':'scripts/eval_cotracker3_online_v8c20_state_repair_precheck.py','videos':videos,'device':args.device,'alignment':info,'touch_stats':touch_stats,'native_subset_metric':nm,'output_level':{'metric':out_metric,'delta_vs_native':{k:(out_metric[k]-nm[k] if out_metric.get(k) is not None and nm.get(k) is not None else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},'stats':out_stats},'state_rows':state_rows}
    out=OUT/'v8c20_state_repair_precheck_report.json'; out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'videos':videos,'touch_stats':touch_stats,'native':{k:round(v,4) if isinstance(v,float) else v for k,v in nm.items() if k in ['AJ','OA','AJ_RD','AJ_RD_256']},'output_delta':{k:round(v,4) if isinstance(v,float) else v for k,v in report['output_level']['delta_vs_native'].items()},'state_rows':[{'mode':r['mode'],'delta':{k:round(v,4) if isinstance(v,float) else v for k,v in r['delta_vs_native'].items()},'per_run':r['per_run']} for r in state_rows]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
