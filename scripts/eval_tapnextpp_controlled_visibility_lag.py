#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT')
REPO=ROOT/'external/tapnextpp/repo'
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(ROOT))
from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics
PIX=255.0

def arr(x,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def lag_reentry_visibility(vis: np.ndarray, lag: int) -> tuple[np.ndarray,int]:
    out=vis.copy(); flips=0
    if lag<=0: return out,0
    Q,T=vis.shape
    for q in range(Q):
        t=0
        while t<T:
            if not vis[q,t]:
                t+=1; continue
            start=t
            while t<T and vis[q,t]: t+=1
            end=t
            # only true re-entry: visible segment starts after an invisible frame
            if start>0 and not vis[q,start-1]:
                e=min(end,start+lag)
                flips += int(out[q,start:e].sum())
                out[q,start:e]=False
    return out,flips

def eval_payload(payload, vis_mode: str, lag: int=0):
    per={}; rows=[]; ajrds=[]; flips_total=0
    for r in payload['records']:
        vid=str(r['video_id'])
        pred_yx=arr(r['pred_tracks'],np.float32); gt_yx=arr(r['gt_tracks'],np.float32); q=arr(r['query_points'],np.float32)
        base_vis=arr(r['pred_visibility'],bool); gt_vis=arr(r['gt_visibility'],bool)
        if vis_mode=='original': pred_vis=base_vis.copy(); flips=0
        elif vis_mode=='lag': pred_vis,flips=lag_reentry_visibility(base_vis,lag)
        else: raise ValueError(vis_mode)
        flips_total+=flips
        pred_xy=pred_yx[...,::-1]*PIX; gt_xy=gt_yx[...,::-1]*PIX; qpx=q.copy(); qpx[:,1:]*=PIX
        m=compute_tapvid_metrics_official(query_points=qpx[None].astype(np.float32), gt_occluded=(~gt_vis)[None], gt_tracks=gt_xy[None].astype(np.float32), pred_occluded=(~pred_vis)[None], pred_tracks=pred_xy[None].astype(np.float32), query_mode='first', thresholds=(1,2,4,8,16))
        m={k:float(v.item() if hasattr(v,'item') else v) for k,v in m.items()}
        aj=compute_redetection_metrics(pred_tracks=torch.from_numpy(pred_xy.copy()).float().unsqueeze(0).permute(0,2,1,3), pred_visible=torch.from_numpy(pred_vis.copy()).bool().unsqueeze(0).permute(0,2,1), gt_tracks=torch.from_numpy(gt_xy.copy()).float().unsqueeze(0).permute(0,2,1,3), gt_visible=torch.from_numpy(gt_vis.copy()).bool().unsqueeze(0).permute(0,2,1))
        aj={k:float(v.item() if hasattr(v,'item') else v) for k,v in aj.items() if not k.startswith('raw_stats/')}
        row={
            'AJ':m['average_jaccard']*100,'OA':m['occlusion_accuracy']*100,'delta_avg':m['average_pts_within_thresh']*100,
            'd1':m['pts_within_1']*100,'d4':m['pts_within_4']*100,'d16':m['pts_within_16']*100,
            'J4':m['jaccard_4']*100,'J8':m['jaccard_8']*100,'J16':m['jaccard_16']*100,
            'AJ_RD':aj.get('AJ_RD'), 'AJ_RD_D4':aj.get('AJ_RD_D4_dmin1'), 'AJ_RD_D16':aj.get('AJ_RD_D16_dmin1'),
        }
        rows.append(row)
        if row['AJ_RD']==row['AJ_RD']: ajrds.append(row['AJ_RD'])
        per[vid]={**row,'flipped_visible_to_false':flips,'queries':int(q.shape[0]),'frames':int(pred_vis.shape[1])}
    def mean(k):
        vals=[x[k] for x in rows if x[k] is not None and x[k]==x[k]]
        return float(np.mean(vals)) if vals else None
    agg={k:mean(k) for k in ['AJ','OA','delta_avg','d1','d4','d16','J4','J8','J16','AJ_RD','AJ_RD_D4','AJ_RD_D16']}
    agg['flipped_visible_to_false_total']=flips_total
    agg['n_videos']=len(rows)
    return {'aggregate':agg,'per_video':per}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v4.pt')
    ap.add_argument('--out-json',default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_online_controlled_visibility_lag_audit.json')
    ap.add_argument('--lags',nargs='+',type=int,default=[1,2,4,8])
    args=ap.parse_args()
    payload=torch.load(ROOT/args.cache if not Path(args.cache).is_absolute() else args.cache,map_location='cpu',weights_only=False)
    original=eval_payload(payload,'original')
    results={'cache':args.cache,'note':'TAPNext++ online coordinates fixed; only binary visibility is lagged after predicted invisible->visible re-entry segments. Restoration is original online visibility.','original':original['aggregate'],'lags':{}}
    for L in args.lags:
        deg=eval_payload(payload,'lag',lag=L)
        delta={k:(original['aggregate'][k]-deg['aggregate'][k] if isinstance(original['aggregate'].get(k),float) and isinstance(deg['aggregate'].get(k),float) else None) for k in original['aggregate']}
        results['lags'][str(L)]={'degraded':deg['aggregate'],'restored_minus_degraded':delta}
    out=ROOT/args.out_json if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(json.dumps(results,indent=2,ensure_ascii=False))
    print('WROTE',out)
if __name__=='__main__': main()
