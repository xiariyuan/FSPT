#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_candidate_oracles'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def find_visible_segments(vis: np.ndarray) -> list[tuple[int,int]]:
    out=[]; t=0; T=len(vis)
    while t<T:
        if not vis[t]:
            t+=1; continue
        s=t
        while t<T and vis[t]: t+=1
        out.append((s,t))
    return out


def collect_candidates(record: dict, *, pre_window:int, confirm_visible_len:int, tau_low:float, trend_min:float, max_jump_px:float, overlap_only:bool, stable_len:int|None=None) -> dict[str, np.ndarray]:
    base_vis=npy(record['pred_visibility'], bool)
    score=npy(record.get('pred_vis_score', base_vis.astype(np.float32)), np.float32)
    coords=npy(record['pred_tracks'], np.float32)
    gt=npy(record['gt_visibility'], bool)
    N,T=base_vis.shape
    raw=np.zeros_like(base_vis,bool)
    numeric=np.zeros_like(base_vis,bool)
    stable=np.zeros_like(base_vis,bool)
    for q in range(N):
        for s,e in find_visible_segments(base_vis[q]):
            if s<=0 or base_vis[q,s-1]:
                continue
            if s+confirm_visible_len>T or not bool(base_vis[q,s:s+confirm_visible_len].all()):
                continue
            lo=max(0,s-pre_window)
            for t in range(lo,s):
                if base_vis[q,t]:
                    continue
                if overlap_only and t<8:
                    continue
                raw[q,t]=True
                jump=float(np.linalg.norm((coords[q,t]-coords[q,s])*255.0))
                trend=float(score[q,t]-score[q,t-1]) if t>0 else 0.0
                ok=bool(score[q,t]>=tau_low and trend>=trend_min and jump<=max_jump_px)
                if ok:
                    numeric[q,t]=True
                    if stable_len is None:
                        stable[q,t]=True
                    else:
                        # require native visible segment length >= stable_len and candidate no earlier than pre_window.
                        stable[q,t]=bool((e-s)>=stable_len)
    return {'raw':raw, 'numeric':numeric, 'stable':stable}


def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool))
        q=torch.from_numpy(npy(r['query_points'],np.float32)); n += int(q.shape[0])
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}


def run_ajrd(cache, out_json):
    import subprocess
    subprocess.run([sys.executable, str(ROOT/'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(cache), '--output-json', str(out_json)], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(Path(out_json).read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--pre-window', type=int, default=2)
    ap.add_argument('--confirm-visible-len', type=int, default=4)
    ap.add_argument('--tau-low', type=float, default=0.55)
    ap.add_argument('--trend-min', type=float, default=-0.02)
    ap.add_argument('--max-jump-px', type=float, default=4.0)
    ap.add_argument('--overlap-only', action='store_true')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    variants={
        'raw_gt_visible_oracle': [],
        'numeric_gt_visible_oracle': [],
        'stable_gt_visible_oracle': [],
        'raw_all_open': [],
        'numeric_all_open': [],
    }
    audits={k:{'opened':0,'gt_visible':0,'gt_occluded':0} for k in variants}
    for r in native['records']:
        c=collect_candidates(r, pre_window=args.pre_window, confirm_visible_len=args.confirm_visible_len, tau_low=args.tau_low, trend_min=args.trend_min, max_jump_px=args.max_jump_px, overlap_only=args.overlap_only, stable_len=8)
        gt=npy(r['gt_visibility'],bool); base=npy(r['pred_visibility'],bool)
        masks={
            'raw_gt_visible_oracle': c['raw'] & gt,
            'numeric_gt_visible_oracle': c['numeric'] & gt,
            'stable_gt_visible_oracle': c['stable'] & gt,
            'raw_all_open': c['raw'],
            'numeric_all_open': c['numeric'],
        }
        for name,m in masks.items():
            rr=dict(r)
            vis=base.copy(); vis[m]=True
            rr['pred_visibility']=vis
            rr['model_name']=f'cotracker3_online_{name}'
            variants[name].append(rr)
            audits[name]['opened'] += int(m.sum())
            audits[name]['gt_visible'] += int((m&gt).sum())
            audits[name]['gt_occluded'] += int((m&~gt).sum())
    rows=[]
    # native too
    native_cache=outdir/'native_ref.pt'; torch.save(native,native_cache)
    std=eval_records(native['records']); aj=run_ajrd(native_cache,outdir/'native_ref_ajrd.json')
    native_row={'variant':'native','cache':str(native_cache),**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256']}
    rows.append(native_row)
    for name,records in variants.items():
        payload=dict(native); payload['records']=records; payload['model_name']=name
        cache=outdir/f'{name}.pt'; torch.save(payload,cache)
        std=eval_records(records); aj=run_ajrd(cache,outdir/f'{name}_ajrd.json')
        row={'variant':name,'cache':str(cache),**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'audit':audits[name]}
        row['audit']['precision']=row['audit']['gt_visible']/max(row['audit']['opened'],1)
        rows.append(row)
    base=rows[0]
    for r in rows:
        r['delta_vs_native']={k:(r[k]-base[k] if k in r and k in base else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
    report={'script':'scripts/eval_cotracker3_online_candidate_oracles.py','params':vars(args),'rows':rows}
    out=outdir/'candidate_oracle_report.json'; out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
