#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
PROJECT_ROOT=Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from datasets.metrics import compute_tapvid_metrics

def npy(x:Any,dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def slice_cache(path:Path, max_records:int)->Dict[str,Any]:
    c=torch.load(path,map_location='cpu',weights_only=False)
    if max_records>0: c=dict(c); c['records']=c['records'][:max_records]
    return c

def eval_std(c:Dict[str,Any])->Dict[str,float]:
    aj=[]; oa=[]; da=[]; n=0
    for r in c['records']:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32))
        n+=int(q.shape[0])
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='strided')
        aj.append(float(m.get('AJ',0.0))); oa.append(float(m.get('OA',0.0))); da.append(float(m.get('average_pts_within_thresh',0.0)))
    return {'AJ_256':round(float(np.mean(aj))*100,4),'OA_256':round(float(np.mean(oa))*100,4),'delta_avg_256':round(float(np.mean(da))*100,4),'n_records':len(c['records']),'n_queries':n}

def eval_ajrd(c:Dict[str,Any], tmpname:str)->Dict[str,Any]:
    tmp=Path(tempfile.gettempdir())/f'{tmpname}.pt'
    out=Path(tempfile.gettempdir())/f'{tmpname}_ajrd.json'
    torch.save(c,tmp)
    subprocess.run([sys.executable,'scripts/eval_aj_rd_from_cache.py','--cache-path',str(tmp),'--output-json',str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.load(open(out))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--items',nargs='+',required=True,help='name=path'); ap.add_argument('--max-records',type=int,default=1); ap.add_argument('--max-queries',type=int,default=0); ap.add_argument('--out-json',required=True); args=ap.parse_args()
    rows=[]
    for item in args.items:
        name,path=item.split('=',1)
        c=slice_cache(Path(path),args.max_records)
        if args.max_queries and args.max_queries > 0:
            for r in c['records']:
                mq = int(args.max_queries)
                for k in ['query_points','pred_tracks','pred_visibility','gt_tracks','target_points','gt_visibility','occluded','source_query_indices']:
                    if k in r and hasattr(r[k], '__getitem__'):
                        r[k] = r[k][:mq]
        std=eval_std(c); ajrd=eval_ajrd(c,'smoke_'+name.replace('/','_'))
        rows.append({'name':name,'cache':path,'AJ_RD_256':ajrd.get('true_AJ_RD_256'),'AJ_RD':ajrd.get('true_AJ_RD'),'first_reentry_frame_proxy':ajrd.get('first_reentry_frame_proxy'),**std})
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps({'rows':rows},indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':args.out_json,'rows':rows},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
