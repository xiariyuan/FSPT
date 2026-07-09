#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
DEFAULT_FEATURES = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
DEFAULT_BENEFIT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy'
DEFAULT_DAMAGE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8a_large_gain_oracles'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def err_px(pred_xy, gt_xy) -> float:
    return float(np.linalg.norm((np.asarray(pred_xy, np.float32) - np.asarray(gt_xy, np.float32)) * 255.0))


def clone_records(records: list[dict]) -> list[dict]:
    out=[]
    for r in records:
        rr=dict(r)
        rr['pred_tracks']=npy(r['pred_tracks'], np.float32).copy()
        rr['pred_visibility']=npy(r['pred_visibility'], bool).copy()
        out.append(rr)
    return out


def eval_records(records: list[dict]) -> dict:
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt=torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'], bool))
        gv=torch.from_numpy(npy(r['gt_visibility'], bool))
        q=torch.from_numpy(npy(r['query_points'], np.float32))
        n += int(q.shape[0])
        m=compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}


def run_ajrd(cache: Path, out_json: Path) -> dict:
    subprocess.run([sys.executable, str(ROOT/'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(cache), '--output-json', str(out_json)], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(out_json.read_text())


def metric_payload(payload: dict, outdir: Path, tag: str) -> dict:
    cache=outdir/f'{tag}.pt'
    ajrd=outdir/f'{tag}_ajrd.json'
    torch.save(payload, cache)
    std=eval_records(payload['records'])
    rd=run_ajrd(cache, ajrd)
    return {**std, 'AJ_RD':rd['true_AJ_RD'], 'AJ_RD_256':rd['true_AJ_RD_256'], 'cache':str(cache)}


def build_event_table(payload: dict, feature_npz: Any) -> tuple[list[dict], dict[str,int]]:
    metas=[json.loads(str(m)) for m in feature_npz['meta_json'].tolist()]
    vid_to_rec={str(r['video_id']): i for i,r in enumerate(payload['records'])}
    return metas, vid_to_rec


def segment_end(gt_vis: np.ndarray, q: int, t: int, max_len: int | None = None) -> int:
    T=gt_vis.shape[1]
    e=t
    steps=0
    while e<T and bool(gt_vis[q,e]) and (max_len is None or steps<max_len):
        e += 1; steps += 1
    return e


def apply_oracle(records: list[dict], metas: list[dict], vid_to_rec: dict[str,int], select: np.ndarray, *, mode: str, window: int | str, safe_only: bool = False) -> tuple[list[dict], dict]:
    """mode: vis or coord_vis. window: 0/4/8/16 or 'segment'.

    For vis mode, only visibility is modified. If safe_only=True, only frames where native coords are <=16px and GT visible are opened.
    For coord_vis, GT coordinates are copied and visibility is opened for GT-visible frames.
    """
    recs=clone_records(records)
    touched=set(); event_count=0; skipped=0; frames_considered=0
    for i,m in enumerate(metas):
        if not bool(select[i]):
            continue
        vi=vid_to_rec[m['video_id']]
        r=recs[vi]
        q=int(m['query_idx']); t0=int(m['frame_t'])
        gt=npy(r['gt_tracks'], np.float32); gv=npy(r['gt_visibility'], bool)
        pred=r['pred_tracks']; pv=r['pred_visibility']
        T=pv.shape[1]
        if t0 < 0 or t0 >= T:
            skipped += 1; continue
        if window == 'segment':
            if not bool(gv[q,t0]):
                skipped += 1; continue
            t1=segment_end(gv, q, t0, max_len=32)
        else:
            w=int(window)
            t1=min(T, t0+w+1)
        did=False
        for t in range(t0,t1):
            frames_considered += 1
            if not bool(gv[q,t]):
                continue
            if mode == 'vis':
                if safe_only:
                    e=err_px(pred[q,t], gt[q,t])
                    if e > 16.0:
                        continue
                pv[q,t]=True
                touched.add((vi,q,t)); did=True
            elif mode == 'coord_vis':
                pred[q,t]=gt[q,t]
                pv[q,t]=True
                touched.add((vi,q,t)); did=True
            else:
                raise ValueError(mode)
        if did:
            event_count += 1
    return recs, {'event_count':event_count, 'touched_frames':len(touched), 'skipped_events':skipped, 'frames_considered':frames_considered}


def apply_v7b2_default(records, metas, vid_to_rec, benefit, damage):
    decision=benefit - 1.35*damage
    select=(decision>=-0.01) & (damage<=0.7)
    recs=clone_records(records)
    touched=set(); ev=0
    for i,m in enumerate(metas):
        if not bool(select[i]): continue
        vi=vid_to_rec[m['video_id']]
        q=int(m['query_idx']); t=int(m['frame_t'])
        recs[vi]['pred_visibility'][q,t]=True
        touched.add((vi,q,t)); ev += 1
    return recs, {'event_count':ev, 'touched_frames':len(touched), 'policy':'decision>=-0.01 and damage<=0.7'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(DEFAULT_CACHE))
    ap.add_argument('--features', default=str(DEFAULT_FEATURES))
    ap.add_argument('--benefit', default=str(DEFAULT_BENEFIT))
    ap.add_argument('--damage', default=str(DEFAULT_DAMAGE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    payload=torch.load(args.cache, map_location='cpu', weights_only=False)
    z=np.load(args.features, allow_pickle=True)
    metas, vid_to_rec=build_event_table(payload, z)
    y4=np.asarray(z['y_reentry_early4']).astype(bool)
    y8=np.asarray(z['y_reentry_early8']).astype(bool)
    yu=np.asarray(z['y_useful_open_t']).astype(bool)
    labels={'early4':y4, 'early8':y8, 'useful_open_t':yu}
    rows=[]
    native_payload=dict(payload); native_payload['records']=clone_records(payload['records']); native_payload['model_name']='native_v8a_ref'
    native=metric_payload(native_payload,outdir,'native')
    rows.append({'variant':'native','action':'none','label':'none','window':'none','stats':{},'metric':native,'delta_vs_native':{k:0.0 for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}})
    if Path(args.benefit).exists() and Path(args.damage).exists():
        benefit=np.load(args.benefit); damage=np.load(args.damage)
        recs,stats=apply_v7b2_default(payload['records'],metas,vid_to_rec,benefit,damage)
        p=dict(payload); p['records']=recs; p['model_name']='v7b2_default_damage_ceiling_0p7'
        m=metric_payload(p,outdir,'v7b2_default_damage_ceiling_0p7')
        rows.append({'variant':'v7b2_default_damage_ceiling_0p7','action':'learned_policy_vis_t','label':'learned','window':0,'stats':stats,'metric':m,'delta_vs_native':{k:m[k]-native[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}})
    # Oracle grids.
    for lname,select in labels.items():
        for w in [0,4,8,16,'segment']:
            for safe_only in [True, False]:
                recs,stats=apply_oracle(payload['records'],metas,vid_to_rec,select,mode='vis',window=w,safe_only=safe_only)
                tag=f'oracle_vis_{lname}_w{w}_' + ('safe16' if safe_only else 'gtvis')
                p=dict(payload); p['records']=recs; p['model_name']=tag
                m=metric_payload(p,outdir,tag.replace('/','_'))
                rows.append({'variant':tag,'action':'visibility_only','label':lname,'window':w,'safe_only':safe_only,'stats':stats,'metric':m,'delta_vs_native':{k:m[k]-native[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}})
            # coord+vis oracle does not need safe_only; it copies GT coords at GT-visible frames.
            recs,stats=apply_oracle(payload['records'],metas,vid_to_rec,select,mode='coord_vis',window=w,safe_only=False)
            tag=f'oracle_coordvis_{lname}_w{w}'
            p=dict(payload); p['records']=recs; p['model_name']=tag
            m=metric_payload(p,outdir,tag.replace('/','_'))
            rows.append({'variant':tag,'action':'coord_plus_visibility_gt','label':lname,'window':w,'safe_only':None,'stats':stats,'metric':m,'delta_vs_native':{k:m[k]-native[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}})
    # Sort summaries.
    report={
        'script':'scripts/audit_cotracker3_online_v8a_large_gain_oracles.py',
        'cache':str(args.cache),
        'features':str(args.features),
        'native_metric':native,
        'label_counts':{k:int(v.sum()) for k,v in labels.items()},
        'rows':rows,
        'best_by_AJ_RD_256':sorted(rows,key=lambda r:r['delta_vs_native']['AJ_RD_256'],reverse=True)[:30],
        'best_safe_AJ':sorted([r for r in rows if r['delta_vs_native']['AJ']>=-0.10 and r['delta_vs_native']['OA']>=-0.10],key=lambda r:r['delta_vs_native']['AJ_RD_256'],reverse=True)[:30],
    }
    out=outdir/'v8a_large_gain_oracles_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({
        'out':str(out),
        'native':{k:round(native[k],4) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
        'label_counts':report['label_counts'],
        'best_by_AJ_RD_256':[
            {'variant':r['variant'],'stats':r['stats'],'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in report['best_by_AJ_RD_256'][:12]
        ],
        'best_safe_AJ':[
            {'variant':r['variant'],'stats':r['stats'],'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in report['best_safe_AJ'][:12]
        ],
    },indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
