#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT')
DEFAULT_NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5d_visibility_verifier/cotracker3_v5d_event_visibility_dataset.npz'

def npy(x:Any,dtype=None)->np.ndarray:
    if isinstance(x,torch.Tensor):
        x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def run_len_backward(mask:np.ndarray,t:int)->int:
    c=0; i=t
    while i>=0 and bool(mask[i]):
        c+=1; i-=1
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--out-npz',default=str(DEFAULT_OUT))
    ap.add_argument('--support-score',type=float,default=0.80)
    ap.add_argument('--low-thr',type=float,default=0.60)
    ap.add_argument('--min-score',type=float,default=0.05)
    ap.add_argument('--trend-min',type=float,default=-0.05)
    ap.add_argument('--min-low-run',type=int,default=1)
    ap.add_argument('--subsample-stride',type=int,default=1)
    args=ap.parse_args()
    payload=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    X=[]; meta=[]; y_gt=[]; y_safe16=[]; y_safe8=[]; y_utility=[]
    feature_names=[
        'score_t','score_prev1','score_prev2','score_trend1','score_trend2',
        'score_mean_prev3','score_max_prev3','score_min_prev3','native_visible',
        'support_age_norm','last_visible_age_norm','low_run_norm','invis_run_norm',
        'coord_y_norm','coord_x_norm','support_coord_dist_norm','last_visible_coord_dist_norm',
        'motion_prev1_norm','motion_prev2_norm','query_age_norm','frame_t_norm','query_idx_norm','video_idx_norm',
        'near_border_norm','score_margin_to_thr'
    ]
    n_records=len(payload['records'])
    for vi,r in enumerate(payload['records']):
        vid=str(r['video_id'])
        pred_vis=npy(r['pred_visibility'],bool); gt_vis=npy(r['gt_visibility'],bool)
        score=npy(r.get('pred_vis_score',pred_vis.astype(np.float32)),np.float32)
        pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32)
        qpts=npy(r['query_points'],np.float32)
        N,T=pred_vis.shape
        low_mask=(score<=args.low_thr) | (~pred_vis)
        support_mask=pred_vis & (score>=args.support_score)
        for q in range(N):
            q_t=int(round(float(qpts[q,0])))
            support_candidates=np.where(support_mask[q])[0]
            for t in range(max(q_t+1,1),T):
                if args.subsample_stride>1 and ((t-q_t)%args.subsample_stride)!=0:
                    continue
                if not bool(low_mask[q,t]):
                    continue
                if float(score[q,t])<args.min_score:
                    continue
                if t>0 and float(score[q,t]-score[q,t-1])<args.trend_min:
                    continue
                support_before=support_candidates[support_candidates<t]
                if support_before.size==0:
                    continue
                sup_t=int(support_before[-1])
                low_run=run_len_backward(low_mask[q],t)
                invis_run=run_len_backward(~pred_vis[q],t)
                if low_run<args.min_low_run and invis_run<args.min_low_run:
                    continue
                last_vis=np.where(pred_vis[q,:t])[0]
                last_t=int(last_vis[-1]) if last_vis.size else sup_t
                s0=float(score[q,t]); sp1=float(score[q,t-1]) if t-1>=0 else s0; sp2=float(score[q,t-2]) if t-2>=0 else sp1
                prev=score[q,max(0,t-3):t]
                coord=pred[q,t]; sup=pred[q,sup_t]; last=pred[q,last_t]
                def dist(a,b): return float(np.linalg.norm((a-b)*255.0))/255.0
                motion1=dist(pred[q,t],pred[q,t-1]) if t-1>=0 else 0.0
                motion2=dist(pred[q,t-1],pred[q,t-2]) if t-2>=0 else 0.0
                y=float(coord[0]); x=float(coord[1]); border=min(y,x,1-y,1-x)
                feat=[
                    s0,sp1,sp2,s0-sp1,s0-sp2,
                    float(np.mean(prev)) if prev.size else s0,float(np.max(prev)) if prev.size else s0,float(np.min(prev)) if prev.size else s0,1.0 if bool(pred_vis[q,t]) else 0.0,
                    float(t-sup_t)/255.0,float(t-last_t)/255.0,float(low_run)/255.0,float(invis_run)/255.0,
                    y,x,dist(coord,sup),dist(coord,last),motion1,motion2,float(t-q_t)/255.0,float(t)/255.0,float(q)/max(N-1,1),float(vi)/max(n_records-1,1),float(border),float(args.low_thr-s0)
                ]
                err=float(np.linalg.norm((pred[q,t]-gt[q,t])*255.0))
                gt_v=bool(gt_vis[q,t]); safe16=bool(gt_v and err<=16.0); safe8=bool(gt_v and err<=8.0)
                X.append(feat); y_gt.append(1.0 if gt_v else 0.0); y_safe16.append(1.0 if safe16 else 0.0); y_safe8.append(1.0 if safe8 else 0.0); y_utility.append(1.0 if safe16 else 0.0)
                meta.append(json.dumps({'video_id':vid,'video_index':vi,'query_idx':int(q),'frame_t':int(t),'query_t':q_t,'support_t':sup_t,'last_visible_t':last_t,'native_visible':bool(pred_vis[q,t]),'score_t':s0,'gt_visible':gt_v,'native_err_px':err,'safe16':safe16,'safe8':safe8},ensure_ascii=False))
    X=np.asarray(X,np.float32); y_gt=np.asarray(y_gt,np.float32); y_safe16=np.asarray(y_safe16,np.float32); y_safe8=np.asarray(y_safe8,np.float32); y_utility=np.asarray(y_utility,np.float32); meta=np.asarray(meta,dtype=object)
    out=Path(args.out_npz); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,feature_names=np.asarray(feature_names,dtype=object),y_gt_visible=y_gt,y_safe16=y_safe16,y_safe8=y_safe8,y_utility=y_utility,meta_json=meta,native_cache=str(args.native_cache),params=json.dumps(vars(args)))
    from collections import Counter
    vids=Counter(json.loads(str(m))['video_id'] for m in meta.tolist())
    summary={'out_npz':str(out),'n_samples':int(X.shape[0]),'y_gt_visible':int(y_gt.sum()),'gt_rate':float(y_gt.mean()) if len(y_gt) else 0.0,'y_safe16':int(y_safe16.sum()),'safe16_rate':float(y_safe16.mean()) if len(y_safe16) else 0.0,'y_safe8':int(y_safe8.sum()),'safe8_rate':float(y_safe8.mean()) if len(y_safe8) else 0.0,'n_videos':len(vids),'videos':dict(vids),'feature_names':feature_names}
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
