#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import torch


def eval_cache(cache_path: Path, tag: str):
    out = Path(tempfile.gettempdir()) / f'{tag}_ajrd.json'
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ajrd = json.load(open(out))
    # Use existing standard evaluator wrapper for one cache.
    out2 = Path(tempfile.gettempdir()) / f'{tag}_std.json'
    subprocess.run([sys.executable, 'scripts/eval_external_baseline_smoke_cache.py', '--max-records', '1', '--out-json', str(out2), '--items', f'{tag}={cache_path}'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    std = json.load(open(out2))['rows'][0]
    return {'AJ_RD_256': ajrd.get('true_AJ_RD_256'), 'AJ_256': std.get('AJ_256'), 'OA_256': std.get('OA_256'), 'delta_avg_256': std.get('delta_avg_256'), 'n_queries': std.get('n_queries')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-json', required=True)
    args=ap.parse_args()
    src=Path(args.cache)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base=torch.load(src,map_location='cpu',weights_only=False)
    variants={}
    for name in ['original','all_visible','gt_visibility','query_visible_fill']:
        c=torch.load(src,map_location='cpu',weights_only=False)
        for r in c['records']:
            if name=='all_visible':
                r['pred_visibility'] = torch.ones_like(torch.as_tensor(r['pred_visibility'])).numpy().astype(bool)
            elif name=='gt_visibility':
                r['pred_visibility'] = torch.as_tensor(r['gt_visibility']).numpy().astype(bool)
            elif name=='query_visible_fill':
                pv=torch.as_tensor(r['pred_visibility']).clone().bool()
                q=torch.as_tensor(r['query_points'])
                for i in range(q.shape[0]):
                    t=int(q[i,0].item())
                    if 0 <= t < pv.shape[1]: pv[i,t]=True
                r['pred_visibility']=pv.numpy().astype(bool)
        p=out_dir / f'{src.stem}_{name}.pt'
        torch.save(c,p)
        variants[name]={'cache':str(p), **eval_cache(p, f'{src.stem}_{name}')}
    Path(args.out_json).write_text(json.dumps({'source':str(src),'variants':variants},indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':args.out_json,'variants':variants},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
