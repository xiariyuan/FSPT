#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
DEFAULT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/online_migration_variants'
PIX=255.0


def arr(x: Any, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def find_visible_segments(vis):
    T=len(vis); segs=[]; t=0
    while t<T:
        if not vis[t]: t+=1; continue
        s=t
        while t<T and vis[t]: t+=1
        segs.append((s,t))
    return segs


def delayed_recovery(pred_yx, conf, base_vis, *, tau_low, pre_window, max_jump_px, min_len):
    out=base_vis.copy(); Q,T=base_vis.shape
    st={'candidate_frames':0,'recovered_frames':0,'rejected_jump':0,'rejected_short':0}
    for q in range(Q):
        for s,e in find_visible_segments(base_vis[q]):
            if s<=0 or base_vis[q,s-1]: continue
            lo=max(0,s-int(pre_window)); cand=[]
            for t in range(lo,s):
                if base_vis[q,t]: continue
                st['candidate_frames']+=1
                if conf[q,t] < tau_low: continue
                if math.isfinite(max_jump_px):
                    jump=float(np.linalg.norm((pred_yx[q,t]-pred_yx[q,s])*PIX))
                    if jump>max_jump_px:
                        st['rejected_jump']+=1; continue
                cand.append(t)
            if not cand: continue
            candset=set(cand); suffix=[]; t=s-1
            while t in candset:
                suffix.append(t); t-=1
            suffix=list(reversed(suffix))
            if len(suffix)<min_len:
                st['rejected_short'] += len(cand); continue
            out[q,suffix]=True; st['recovered_frames']+=len(suffix)
    return out, st


def causal_recovery(pred_yx, conf, base_vis, query_points, *, tau_low, min_prev_invis, max_step_px, conf_rise_min):
    """Zero-latency causal rule. It only uses frames <= t.

    Opens a frame if native says invisible, confidence is moderately high, the
    point has been native-invisible for min_prev_invis frames, confidence is not
    falling sharply, and coordinate step is locally smooth.
    """
    out=base_vis.copy(); Q,T=base_vis.shape
    st={'candidate_frames':0,'recovered_frames':0,'rejected_step':0,'rejected_rise':0}
    for q in range(Q):
        q_t=int(round(float(query_points[q,0]))) if query_points is not None else -1
        inv_run=0
        for t in range(T):
            if t <= q_t:
                inv_run=0 if base_vis[q,t] else inv_run+1
                continue
            if base_vis[q,t]:
                inv_run=0
                continue
            inv_run += 1
            st['candidate_frames'] += 1
            if inv_run < int(min_prev_invis):
                continue
            if conf[q,t] < tau_low:
                continue
            if t>0 and conf[q,t] - conf[q,t-1] < conf_rise_min:
                st['rejected_rise'] += 1
                continue
            if t>0 and math.isfinite(max_step_px):
                step=float(np.linalg.norm((pred_yx[q,t]-pred_yx[q,t-1])*PIX))
                if step>max_step_px:
                    st['rejected_step'] += 1
                    continue
            out[q,t]=True
            st['recovered_frames'] += 1
    return out, st


def save_variant(base_payload, out_path: Path, mode: str, params: dict):
    records=[]; stats=[]
    for r in base_payload['records']:
        nr=dict(r)
        pred_yx=arr(r['pred_tracks'], np.float32)
        conf=arr(r['pred_vis_conf'], np.float32)
        base_vis=arr(r['pred_visibility'], bool)
        if mode=='delayed':
            pred_vis, st=delayed_recovery(pred_yx, conf, base_vis, **params)
        elif mode=='causal':
            q=arr(r['query_points'], np.float32)
            pred_vis, st=causal_recovery(pred_yx, conf, base_vis, q, **params)
        else:
            raise ValueError(mode)
        nr['pred_visibility']=pred_vis.astype(bool)
        nr['visibility_variant']= {'mode': mode, **params}
        records.append(nr); stats.append(st)
    payload=dict(base_payload)
    payload['records']=records
    payload['model_name']=f"{base_payload.get('model_name','trackon2')}_{mode}_online_visibility_variant"
    payload['visibility_variant']= {'mode': mode, **params, 'note': 'Coordinates fixed; only pred_visibility is changed.'}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    total={k:int(sum(s.get(k,0) for s in stats)) for k in sorted({kk for s in stats for kk in s})}
    return total


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(DEFAULT_CACHE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    args=ap.parse_args()
    cache=Path(args.cache); outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    payload=torch.load(cache,map_location='cpu',weights_only=False)
    variants=[
        ('delayed_k4_tau04_jump24', 'delayed', dict(tau_low=0.4, pre_window=4, max_jump_px=24.0, min_len=1)),
        ('delayed_k4_tau05_jump24', 'delayed', dict(tau_low=0.5, pre_window=4, max_jump_px=24.0, min_len=1)),
        ('causal_tau075_prev2_step24_rise0', 'causal', dict(tau_low=0.75, min_prev_invis=2, max_step_px=24.0, conf_rise_min=0.0)),
        ('causal_tau070_prev2_step24_rise0', 'causal', dict(tau_low=0.70, min_prev_invis=2, max_step_px=24.0, conf_rise_min=0.0)),
        ('causal_tau065_prev2_step16_rise0', 'causal', dict(tau_low=0.65, min_prev_invis=2, max_step_px=16.0, conf_rise_min=0.0)),
    ]
    created=[]
    for name,mode,params in variants:
        out=outdir / f'{name}.pt'
        st=save_variant(payload,out,mode,params)
        created.append({'name':name,'mode':mode,'params':params,'cache':str(out),'stats':st})
    compare=outdir/'online_visibility_variants_metric_compare.json'
    items=[f'native={cache}']+[f"{c['name']}={c['cache']}" for c in created]
    cmd=[sys.executable, str(ROOT/'scripts/eval_external_baseline_smoke_cache.py'), '--items', *items, '--max-records','0','--out-json', str(compare)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    metrics=json.loads(compare.read_text())
    report={'source_cache':str(cache),'created':created,'metric_compare':metrics}
    report_path=outdir/'online_visibility_variants_report.json'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'report':str(report_path),'compare':str(compare),'created':created,'rows':metrics.get('rows')},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
