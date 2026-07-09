#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from scripts.train_cotracker3_online_v7b1_damage_aware_raw_visconf import eval_records, run_ajrd, build_open_records

DEFAULT_FEATURES = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
DEFAULT_BENEFIT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy'
DEFAULT_DAMAGE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/refine_grid'


def npy(x: Any, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def err_px(a,b):
    return float(np.linalg.norm((np.asarray(a,np.float32)-np.asarray(b,np.float32))*255.0))


def event_labels(native_payload, metas):
    rec_by_vid={str(r['video_id']):r for r in native_payload['records']}
    y_bad=[]; y_safe16=[]; y_safe8=[]; y_need=[]
    for m in metas:
        r=rec_by_vid[m['video_id']]
        q=int(m['query_idx']); t=int(m['frame_t'])
        pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32); gv=npy(r['gt_visibility'],bool); pv=npy(r['pred_visibility'],bool); score=npy(r['pred_vis_score'],np.float32)
        need=(not bool(pv[q,t])) and float(score[q,t])<0.6
        if not bool(gv[q,t]):
            bad=True; safe16=False; safe8=False
        else:
            e=err_px(pred[q,t],gt[q,t]); bad=e>16; safe16=e<=16; safe8=e<=8
        y_bad.append(bad); y_safe16.append(safe16); y_safe8.append(safe8); y_need.append(need)
    return np.asarray(y_bad,bool),np.asarray(y_safe16,bool),np.asarray(y_safe8,bool),np.asarray(y_need,bool)


def metric_for(native, metas, decision, thr, outdir, tag):
    records, opened = build_open_records(native, metas, decision, float(thr))
    payload=dict(native); payload['records']=records; payload['model_name']=tag
    cache=outdir/f'{tag}.pt'; ajrd=outdir/f'{tag}_ajrd.json'
    torch.save(payload, cache)
    std=eval_records(records); aj=run_ajrd(cache, ajrd)
    return {**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'opened':len(opened),'cache':str(cache)}, opened


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default=str(DEFAULT_FEATURES))
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--benefit',default=str(DEFAULT_BENEFIT))
    ap.add_argument('--damage',default=str(DEFAULT_DAMAGE))
    ap.add_argument('--outdir',default=str(DEFAULT_OUTDIR))
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    z=np.load(args.features,allow_pickle=True)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    y4=np.asarray(z['y_reentry_early4']).astype(bool)
    y8=np.asarray(z['y_reentry_early8']).astype(bool)
    yu=np.asarray(z['y_useful_open_t']).astype(bool)
    benefit=np.load(args.benefit); damage=np.load(args.damage)
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    native_std=eval_records(native['records']); native_aj=run_ajrd(Path(args.native_cache),outdir/'native_ref_ajrd.json')
    native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
    y_bad,y_safe16,y_safe8,y_need=event_labels(native,metas)
    lambdas=np.round(np.arange(1.15,1.61,0.05),3)
    thresholds=np.round(np.arange(-0.14,0.041,0.01),3)
    fixed=[(1.25,0.0),(1.25,-0.04),(1.5,-0.10),(1.5,0.0)]
    rows=[]
    seen=set()
    for lam in lambdas:
        decision=benefit-lam*damage
        for thr in thresholds:
            key=(float(lam),float(thr))
            if key in seen: continue
            seen.add(key)
            accept=decision>=thr
            if not accept.any(): continue
            tag=f'v7b1_refine_lam{lam:.2f}_thr{thr:.3f}'.replace('-','m')
            metric,opened=metric_for(native,metas,decision,float(thr),outdir,tag)
            delta={k:metric[k]-native_row[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            idx=np.where(accept)[0]
            rows.append({
                'lambda':float(lam),'threshold':float(thr),'accept_count':int(accept.sum()),
                'benefit_precision':float((accept & y4).sum()/max(int(accept.sum()),1)),'benefit_recall':float((accept & y4).sum()/max(int(y4.sum()),1)),
                'early8_rate':float(y8[accept].mean()),'useful_rate':float(yu[accept].mean()),
                'damage_rate':float(y_bad[accept].mean()),'safe16_rate':float(y_safe16[accept].mean()),'safe8_rate':float(y_safe8[accept].mean()),'need_open_rate':float(y_need[accept].mean()),
                'metric':metric,'delta_vs_native':delta,
            })
    constrained=[r for r in rows if r['delta_vs_native']['AJ']>=-0.10 and r['delta_vs_native']['OA']>=-0.10]
    success=[r for r in constrained if r['delta_vs_native']['AJ_RD']>=0.0015 and r['delta_vs_native']['AJ_RD_256']>=0.004]
    best_constrained=sorted(constrained,key=lambda r:(r['delta_vs_native']['AJ_RD_256'],r['delta_vs_native']['AJ_RD'],r['delta_vs_native']['AJ']),reverse=True)[:30]
    best_raw=sorted(rows,key=lambda r:(r['delta_vs_native']['AJ_RD_256'],r['delta_vs_native']['AJ_RD']),reverse=True)[:30]
    report={'script':'scripts/refine_cotracker3_online_v7b1_damage_aware_grid.py','native_row':native_row,'n_grid':len(rows),'success_count':len(success),'success_rows':success[:50],'best_constrained':best_constrained,'best_raw':best_raw,'grid':rows}
    out=outdir/'v7b1_refine_grid_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'n_grid':len(rows),'success_count':len(success),'best_constrained':[{k:r[k] for k in ['lambda','threshold','accept_count','benefit_precision','benefit_recall','damage_rate','safe16_rate','delta_vs_native']} for r in best_constrained[:12]]},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
