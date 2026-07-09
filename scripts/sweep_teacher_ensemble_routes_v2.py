#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
}
# fix typo defensively
CACHE_PATHS["cotracker3_offline"] = "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt"

def load_cache(path): return torch.load(path, map_location='cpu', weights_only=False)
def npy(x, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
def finite(pred_stack): return np.all(np.isfinite(pred_stack), axis=-1)
def masked_mean(pred_stack, mask):
    cnt=mask.sum(axis=0).astype(np.float32)[...,None]
    sm=np.where(mask[...,None], pred_stack, 0.0).sum(axis=0,dtype=np.float32)
    out=np.full(pred_stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(sm,cnt,out=out,where=cnt>0)
    return out.astype(np.float32)
def masked_median(pred_stack, mask):
    with np.errstate(all='ignore'):
        return np.nanmedian(np.where(mask[...,None], pred_stack, np.nan), axis=0).astype(np.float32)
def fill(a,b):
    bad=~np.all(np.isfinite(a),axis=-1)
    if np.any(bad):
        a=a.copy(); a[bad]=b[bad]
    return a.astype(np.float32)
def vis_rule(vis_stack, rule):
    c=vis_stack.sum(axis=0)
    if rule=='union': return (c>=1).astype(bool)
    if rule=='majority': return (c>=2).astype(bool)
    if rule=='intersection': return (c>=3).astype(bool)
    if rule=='online': return vis_stack[0].astype(bool)
    if rule=='offline': return vis_stack[1].astype(bool)
    if rule=='trackon2': return vis_stack[2].astype(bool)
    if rule=='union_online_offline': return (vis_stack[0]|vis_stack[1]).astype(bool)
    if rule=='union_online_trackon2': return (vis_stack[0]|vis_stack[2]).astype(bool)
    if rule=='union_offline_trackon2': return (vis_stack[1]|vis_stack[2]).astype(bool)
    raise ValueError(rule)
def px_scale(osz):
    h,w=float(osz[0]),float(osz[1]); return np.array([max(h-1,1),max(w-1,1)],dtype=np.float32)
def pairwise_mean_dist(pred_stack, osz):
    pts=pred_stack*px_scale(osz)
    M=pts.shape[0]; out=[]
    for i in range(M):
        ds=[]
        for j in range(M):
            if i!=j: ds.append(np.linalg.norm(pts[i]-pts[j],axis=-1))
        out.append(np.mean(ds,axis=0))
    return np.stack(out,axis=0).astype(np.float32)
def choose(pred_stack, idx):
    out=np.empty(pred_stack.shape[1:],dtype=np.float32)
    for i in range(pred_stack.shape[0]):
        m=idx==i; out[m]=pred_stack[i][m]
    return out
def choose_vis(vis_stack, idx):
    out=np.empty(vis_stack.shape[1:],dtype=bool)
    for i in range(vis_stack.shape[0]):
        m=idx==i; out[m]=vis_stack[i][m]
    return out

def tracks_rule(pred_stack, vis_stack, osz, rule):
    fm=finite(pred_stack)
    fallback_mean=masked_mean(pred_stack,fm)
    fallback_med=masked_median(pred_stack,fm)
    if rule=='visible_median': return fill(masked_median(pred_stack,fm&vis_stack), fallback_med)
    if rule=='visible_mean': return fill(masked_mean(pred_stack,fm&vis_stack), fallback_mean)
    if rule=='all_median': return fallback_med
    if rule=='all_mean': return fallback_mean
    if rule=='online': return pred_stack[0].astype(np.float32)
    if rule=='offline': return pred_stack[1].astype(np.float32)
    if rule=='trackon2': return pred_stack[2].astype(np.float32)
    if rule=='medoid':
        d=pairwise_mean_dist(pred_stack,osz)
        idx=np.argmin(d,axis=0)
        return choose(pred_stack,idx)
    if rule=='visible_medoid':
        d=pairwise_mean_dist(pred_stack,osz)
        idx=np.argmin(np.where(vis_stack,d,d+1e6),axis=0)
        idx[~np.any(vis_stack,axis=0)] = 1
        return choose(pred_stack,idx)
    if rule.startswith('pair_'):
        pair=rule.split('_',1)[1]
        ids={'online_offline':[0,1],'online_trackon2':[0,2],'offline_trackon2':[1,2]}[pair]
        subp=pred_stack[ids]; subv=vis_stack[ids]; subfm=finite(subp)
        return fill(masked_median(subp,subfm&subv), masked_median(subp,subfm))
    raise ValueError(rule)

def validate(caches):
    ref=caches[TEACHERS[0]]['records']
    for t in TEACHERS[1:]:
        assert len(caches[t]['records'])==len(ref)
        for i,(a,b) in enumerate(zip(ref,caches[t]['records'])):
            assert a['video_id']==b['video_id']
            for k in ['query_points','gt_tracks','gt_visibility','original_size','model_input_size']:
                assert np.allclose(npy(a[k]),npy(b[k]),atol=1e-6,rtol=1e-6), (t,i,k)

def build(name, trule, vrule, caches, out_cache):
    validate(caches)
    refp=caches[TEACHERS[0]]; records=[]
    for i,ref in enumerate(refp['records']):
        pred=np.stack([npy(caches[t]['records'][i]['pred_tracks'],np.float32) for t in TEACHERS],axis=0)
        vis=np.stack([npy(caches[t]['records'][i]['pred_visibility'],bool) for t in TEACHERS],axis=0)
        osz=npy(ref['original_size'],np.float32)
        r=dict(ref)
        r['pred_tracks']=tracks_rule(pred,vis,osz,trule)
        r['pred_visibility']=vis_rule(vis,vrule)
        r['model_name']='route_v2_'+name
        r['route_tracks_rule']=trule; r['route_visibility_rule']=vrule
        records.append(r)
    out=dict(refp); out['model_name']='route_v2_'+name; out['records']=records
    out['route_tracks_rule']=trule; out['route_visibility_rule']=vrule
    torch.save(out,out_cache)

def eval_cache(cache_path,json_path):
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(cache_path),'--output-json',str(json_path)],check=True)
    return json.load(open(json_path))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default='outputs/paper_discovery_2026-06-27/route_sweep_v2'); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    caches={t:load_cache(p) for t,p in CACHE_PATHS.items()}
    track_rules=['visible_median','visible_mean','all_median','all_mean','medoid','visible_medoid','online','offline','trackon2','pair_online_offline','pair_online_trackon2','pair_offline_trackon2']
    vis_rules=['union','majority','union_online_offline','union_online_trackon2','union_offline_trackon2','online','offline','trackon2']
    rows=[]
    for tr in track_rules:
        for vr in vis_rules:
            name=f'{tr}__vis_{vr}'
            cp=out/f'{name}.pt'; jp=out/f'{name}_ajrd.json'
            print('===',name,'===',flush=True)
            build(name,tr,vr,caches,cp)
            m=eval_cache(cp,jp)
            row={'name':name,'tracks':tr,'visibility':vr,'true_AJ_RD':m.get('true_AJ_RD'),'true_AJ_RD_256':m.get('true_AJ_RD_256'),'proxy':m.get('first_reentry_frame_proxy'),'median_px':(m.get('reentry_error') or {}).get('median_px'),'lt4px':(m.get('reentry_error') or {}).get('lt4px'),'long20_median_px':(m.get('long_occ_ge20') or {}).get('median_px'),'long20_lt4px':(m.get('long_occ_ge20') or {}).get('lt4px'),'long20_lt8px':(m.get('long_occ_ge20') or {}).get('lt8px')}
            rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    rows=sorted(rows,key=lambda r:r.get('true_AJ_RD_256') or -1,reverse=True)
    summary={'baseline_visible_median_majority':0.5871,'previous_best_visible_median_union':0.6137,'fixed_best':0.5546,'oracle':0.6509,'rows':rows}
    json.dump(summary,open(out/'summary.json','w'),indent=2,ensure_ascii=False)
    print('=== BEST V2 ===')
    print(json.dumps(rows[:20],indent=2,ensure_ascii=False))
if __name__=='__main__': main()
