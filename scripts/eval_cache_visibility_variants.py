#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from datasets.metrics import compute_tapvid_metrics

def npy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def clone_payload_with_vis(payload, variant):
    out=dict(payload); recs=[]
    for r0 in payload['records']:
        r=dict(r0)
        gt=npy(r['gt_visibility']).astype(bool)
        pred=npy(r['pred_visibility']).astype(bool)
        if variant=='original':
            r['pred_visibility']=pred
        elif variant=='gt_visibility':
            r['pred_visibility']=gt
        elif variant=='all_visible':
            r['pred_visibility']=np.ones_like(gt, dtype=bool)
        elif variant=='query_visible_fill':
            q=npy(r['query_points']).astype(np.float32)
            q_t=np.clip(np.rint(q[:,0]).astype(int),0,gt.shape[1]-1)
            pv=pred.copy(); pv[np.arange(gt.shape[0]), q_t]=True
            r['pred_visibility']=pv
        else:
            raise ValueError(variant)
        recs.append(r)
    out['records']=recs
    return out

def eval_std(payload):
    rows=[]; nq=0
    for r in payload['records']:
        pred=torch.from_numpy(npy(r['pred_tracks']).astype(np.float32)); gt=torch.from_numpy(npy(r['gt_tracks']).astype(np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility']).astype(bool)); gv=torch.from_numpy(npy(r['gt_visibility']).astype(bool)); q=torch.from_numpy(npy(r['query_points']).astype(np.float32))
        nq+=q.shape[0]
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='strided')
        rows.append(m)
    return {
        'AJ_256': round(float(np.mean([x['AJ'] for x in rows]))*100,4),
        'OA_256': round(float(np.mean([x['OA'] for x in rows]))*100,4),
        'delta_avg_256': round(float(np.mean([x['average_pts_within_thresh'] for x in rows]))*100,4),
        'n_queries': int(nq),
    }

def eval_ajrd(payload, tag):
    tmp=Path(tempfile.gettempdir())/f'{tag}.pt'; out=Path(tempfile.gettempdir())/f'{tag}.json'
    torch.save(payload,tmp)
    import subprocess
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(tmp),'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(out))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True); ap.add_argument('--out-json',required=True); args=ap.parse_args()
    payload=torch.load(args.cache,map_location='cpu',weights_only=False)
    variants=['original','gt_visibility','all_visible','query_visible_fill']
    rows=[]
    stem=Path(args.cache).stem
    for v in variants:
        p=clone_payload_with_vis(payload,v)
        std=eval_std(p); ajrd=eval_ajrd(p,f'visvar_{stem}_{v}')
        rows.append({'variant':v,'AJ_RD_256':ajrd.get('true_AJ_RD_256'),'first_reentry_frame_proxy':ajrd.get('first_reentry_frame_proxy'),**std})
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps({'cache':args.cache,'rows':rows},indent=2,ensure_ascii=False))
    print(json.dumps({'cache':args.cache,'rows':rows},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
