#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def arr(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)


def pct(a, q):
    a=np.asarray(a,float)
    if a.size==0: return None
    return float(np.percentile(a,q))


def audit(path: Path):
    c=torch.load(path,map_location='cpu',weights_only=False)
    out=[]
    all_anchor=[]; all_valid=[]; all_vis=[]; all_pred_min=[]; all_pred_max=[]; all_gt_err_vis=[]; all_gt_err_all=[]
    for r in c['records']:
        q=arr(r['query_points']).astype(np.float32) # N, t,y,x normalized
        pred=arr(r['pred_tracks']).astype(np.float32) # N,T,y,x normalized
        pv=arr(r['pred_visibility']).astype(bool)
        gt=arr(r['gt_tracks']).astype(np.float32)
        gv=arr(r['gt_visibility']).astype(bool)
        osz=arr(r.get('original_size',[256,256])).astype(float)
        h,w=float(osz[0]),float(osz[1])
        N,T=q.shape[0],pred.shape[1]
        anchor=[]; inrange=[]; gt_err_all=[]; gt_err_vis=[]
        for i in range(N):
            t=int(round(float(q[i,0])))
            if 0<=t<T:
                qyx=np.array([q[i,1],q[i,2]],dtype=np.float32)
                p=pred[i,t]
                e=np.linalg.norm((p-qyx)*np.array([h-1,w-1],dtype=np.float32))
                anchor.append(e)
        ppx=pred.copy()
        gtpx=gt.copy()
        ppx[...,0]*=(h-1); ppx[...,1]*=(w-1)
        gtpx[...,0]*=(h-1); gtpx[...,1]*=(w-1)
        err=np.linalg.norm(ppx-gtpx,axis=-1)
        gt_err_all=err.reshape(-1)
        gt_err_vis=err[gv]
        inrange=((pred[...,0]>=-0.05)&(pred[...,0]<=1.05)&(pred[...,1]>=-0.05)&(pred[...,1]<=1.05)).reshape(-1)
        out.append({
            'video_id':r.get('video_id'),
            'n_queries':int(N),'frames':int(T),
            'pred_vis_rate':float(pv.mean()),
            'gt_vis_rate':float(gv.mean()),
            'pred_y_minmax':[float(np.nanmin(pred[...,0])),float(np.nanmax(pred[...,0]))],
            'pred_x_minmax':[float(np.nanmin(pred[...,1])),float(np.nanmax(pred[...,1]))],
            'pred_in_range_rate_pm005':float(inrange.mean()),
            'anchor_err_px_mean':float(np.mean(anchor)) if anchor else None,
            'anchor_err_px_median':float(np.median(anchor)) if anchor else None,
            'anchor_err_px_p95':pct(anchor,95),
            'err_px_all_median':pct(gt_err_all,50),
            'err_px_all_p75':pct(gt_err_all,75),
            'err_px_all_p95':pct(gt_err_all,95),
            'err_px_gt_visible_median':pct(gt_err_vis,50),
            'err_px_gt_visible_p75':pct(gt_err_vis,75),
            'err_px_gt_visible_p95':pct(gt_err_vis,95),
        })
        all_anchor.extend(anchor); all_valid.extend(inrange.tolist()); all_vis.append(float(pv.mean()))
        all_pred_min.append([float(np.nanmin(pred[...,0])),float(np.nanmin(pred[...,1]))])
        all_pred_max.append([float(np.nanmax(pred[...,0])),float(np.nanmax(pred[...,1]))])
        all_gt_err_all.extend(gt_err_all.tolist()); all_gt_err_vis.extend(gt_err_vis.tolist())
    return {
        'cache':str(path),
        'records':out,
        'aggregate':{
            'n_records':len(out),
            'anchor_err_px_mean':float(np.mean(all_anchor)) if all_anchor else None,
            'anchor_err_px_median':float(np.median(all_anchor)) if all_anchor else None,
            'anchor_err_px_p95':pct(all_anchor,95),
            'pred_in_range_rate_pm005':float(np.mean(all_valid)) if all_valid else None,
            'pred_vis_rate_mean':float(np.mean(all_vis)) if all_vis else None,
            'err_px_all_median':pct(all_gt_err_all,50),
            'err_px_all_p75':pct(all_gt_err_all,75),
            'err_px_all_p95':pct(all_gt_err_all,95),
            'err_px_gt_visible_median':pct(all_gt_err_vis,50),
            'err_px_gt_visible_p75':pct(all_gt_err_vis,75),
            'err_px_gt_visible_p95':pct(all_gt_err_vis,95),
        }
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--caches',nargs='+',required=True)
    ap.add_argument('--out-json',required=True)
    args=ap.parse_args()
    res={Path(p).stem:audit(Path(p)) for p in args.caches}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res,indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':args.out_json,'summary':{k:v['aggregate'] for k,v in res.items()}},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
