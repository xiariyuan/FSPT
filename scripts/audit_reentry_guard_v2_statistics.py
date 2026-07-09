#!/usr/bin/env python3
from __future__ import annotations

import json, math, sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn')
OUT = OUTDIR / 'statistics_summary.json'

SETTINGS = {
    'rgb_fresh20_49_natural': {
        'offline': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'),
        'b2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/b2_w16_p2_rgb_stacking_fresh20_49.pt'),
        'guard': OUTDIR / 'rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt',
    },
    'fresh20_49_translate_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt'),
        'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/b2_w16_p2_translate_L16_fresh20_49.pt'),
        'guard': OUTDIR / 'fresh20_49_translate_L16/random_forest/random_forest_thr0.40.pt',
    },
    'fresh20_49_occluder_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt'),
        'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/b2_w16_p2_occluder_L16_fresh20_49.pt'),
        'guard': OUTDIR / 'fresh20_49_occluder_L16/random_forest/random_forest_thr0.40.pt',
    },
}

def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def per_video_metrics(cache: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out={}
    for r in cache['records']:
        vid=str(r['video_id'])
        h,w=int(r['original_size'][0]), int(r['original_size'][1])
        pred=npy(r['pred_tracks'], np.float32); gt=npy(r['gt_tracks'], np.float32)
        pv=npy(r['pred_visibility'], bool); gv=npy(r['gt_visibility'], bool); q=npy(r['query_points'], np.float32)
        m=compute_tapvid_metrics(torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(pv), torch.from_numpy(gv), torch.from_numpy(q), resolution=256, query_mode='strided')
        rm=compute_reentry_metrics(pred, gt, pv, gv, q, h, w)
        out[vid]={'AJ_RD_256': float(rm['true_AJ_RD_256']) if rm.get('true_AJ_RD_256') is not None else np.nan, 'AJ_256': float(m.get('AJ',0.0))*100, 'OA_256': float(m.get('OA',0.0))*100, 'delta_avg_256': float(m.get('average_pts_within_thresh',0.0))*100}
    return out

def sign_p(pos:int, neg:int)->float:
    n=pos+neg
    if n<=0: return 1.0
    k=min(pos,neg)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/(2**n))

def paired(a,b,metric,seed=20260701):
    vids=sorted(set(a)&set(b)); vals=[]; used=[]
    for v in vids:
        x=a[v].get(metric,np.nan); y=b[v].get(metric,np.nan)
        if np.isfinite(x) and np.isfinite(y): vals.append(float(x-y)); used.append(v)
    arr=np.asarray(vals,dtype=float)
    if arr.size==0: return {'n':0}
    rng=np.random.default_rng(seed)
    boot=[float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)])) for _ in range(10000)]
    ci=np.percentile(boot,[2.5,97.5])
    pos=int(np.sum(arr>0)); neg=int(np.sum(arr<0)); zero=int(np.sum(arr==0))
    return {'n':int(arr.size),'mean_delta':round(float(np.mean(arr)),6),'median_delta':round(float(np.median(arr)),6),'ci95_bootstrap':[round(float(ci[0]),6),round(float(ci[1]),6)],'positive_videos':pos,'negative_videos':neg,'zero_videos':zero,'sign_test_p_two_sided':round(sign_p(pos,neg),8),'min_delta':round(float(np.min(arr)),6),'max_delta':round(float(np.max(arr)),6)}

def main():
    results=[]
    for name, paths in SETTINGS.items():
        caches={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
        perv={k:per_video_metrics(caches[k]) for k in caches}
        comps={}
        for a,b in [('guard','b2'),('guard','offline'),('b2','offline')]:
            comps[f'{a}_vs_{b}']={metric: paired(perv[a], perv[b], metric, seed=20260701+len(name)+len(metric)) for metric in ['AJ_RD_256','AJ_256','OA_256']}
        results.append({'setting':name,'paths':{k:str(v) for k,v in paths.items()},'comparisons':comps})
    OUT.write_text(json.dumps({'guard':'random_forest_thr0.40','results':results},indent=2,ensure_ascii=False))
    compact=[]
    for r in results:
        compact.append({'setting':r['setting'], 'guard_vs_b2_AJRD': r['comparisons']['guard_vs_b2']['AJ_RD_256'], 'guard_vs_b2_AJ': r['comparisons']['guard_vs_b2']['AJ_256'], 'guard_vs_offline_AJRD': r['comparisons']['guard_vs_offline']['AJ_RD_256']})
    print(json.dumps({'summary_path':str(OUT),'compact':compact},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
