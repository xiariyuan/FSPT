#!/usr/bin/env python3
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from datasets.metrics import compute_tapvid_metrics

NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
FEATURES=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8b_candidate_coord_sources'
OUTDIR.mkdir(parents=True,exist_ok=True)

CANDIDATES={
 'old_cotracker3_online_bridge': ROOT/'outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt',
 'old_cotracker3_offline_bridge': ROOT/'outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_offline_davis_first_input_bridge.pt',
 'trackon2_dinov3_davis': ROOT/'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis.pt',
 'trackon2_dinov3_bridge': ROOT/'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt',
}


def npy(x: Any, dtype=None):
    if isinstance(x, torch.Tensor):
        x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def clone_records(records):
    out=[]
    for r in records:
        rr=dict(r)
        rr['pred_tracks']=npy(r['pred_tracks'],np.float32).copy()
        rr['pred_visibility']=npy(r['pred_visibility'],bool).copy()
        out.append(rr)
    return out


def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32))
        gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool))
        gv=torch.from_numpy(npy(r['gt_visibility'],bool))
        q=torch.from_numpy(npy(r['query_points'],np.float32))
        n += int(q.shape[0])
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}


def run_ajrd(cache,out_json):
    subprocess.run([sys.executable,str(ROOT/'scripts/eval_aj_rd_from_cache.py'),'--cache-path',str(cache),'--output-json',str(out_json)],cwd=str(ROOT),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.loads(out_json.read_text())


def metric_payload(payload, tag):
    cache=OUTDIR/f'{tag}.pt'; aj=OUTDIR/f'{tag}_ajrd.json'
    torch.save(payload,cache)
    std=eval_records(payload['records']); rd=run_ajrd(cache,aj)
    return {**std,'AJ_RD':rd['true_AJ_RD'],'AJ_RD_256':rd['true_AJ_RD_256'],'cache':str(cache)}


def err_px(a,b):
    return float(np.linalg.norm((np.asarray(a,np.float32)-np.asarray(b,np.float32))*255.0))


def segment_end(gt_vis,q,t,max_len=32):
    T=gt_vis.shape[1]; e=t; c=0
    while e<T and bool(gt_vis[q,e]) and c<max_len:
        e+=1; c+=1
    return e


def align_candidate(native,cand):
    n_vids=[str(r['video_id']) for r in native['records']]
    c_by={str(r['video_id']):r for r in cand['records']}
    aligned=[]; problems=[]; qp_diffs=[]; gt_diffs=[]
    for nr in native['records']:
        vid=str(nr['video_id'])
        cr=c_by.get(vid)
        if cr is None:
            problems.append({'video_id':vid,'problem':'missing'}); continue
        nt=npy(nr['pred_tracks']); ct=npy(cr['pred_tracks'])
        if nt.shape!=ct.shape:
            problems.append({'video_id':vid,'problem':'shape_mismatch','native':nt.shape,'candidate':ct.shape}); continue
        qp_diffs.append(float(np.max(np.abs(npy(nr['query_points'],np.float32)-npy(cr['query_points'],np.float32)))))
        gt_diffs.append(float(np.max(np.abs(npy(nr['gt_tracks'],np.float32)-npy(cr['gt_tracks'],np.float32)))))
        aligned.append(cr)
    ok=len(problems)==0 and len(aligned)==len(native['records']) and max(qp_diffs or [0])<1e-4
    return ok, {'problems':problems[:10], 'n_aligned':len(aligned), 'max_query_diff':max(qp_diffs or [0]), 'max_gt_diff':max(gt_diffs or [0])}, {str(r['video_id']):r for r in aligned}


def apply_candidate_window(native_records, cand_by_vid, metas, select, *, window, mode='force_gt_visible'):
    """Replace native coords with candidate coords in selected event windows.

    mode:
      force_gt_visible: only GT-visible frames are touched, visibility forced true.
      candidate_visible: only candidate-visible frames are touched, visibility copied/true.
      force_all: all frames in window touched, visibility forced true. This is risky and diagnostic.
    """
    recs=clone_records(native_records)
    vid_to_i={str(r['video_id']):i for i,r in enumerate(recs)}
    stats=Counter(); safe_counts=Counter(); touched=set()
    for i,m in enumerate(metas):
        if not bool(select[i]): continue
        vid=m['video_id']; vi=vid_to_i[vid]; nr=recs[vi]; cr=cand_by_vid[vid]
        q=int(m['query_idx']); t0=int(m['frame_t'])
        pred=nr['pred_tracks']; pv=nr['pred_visibility']; gt=npy(nr['gt_tracks'],np.float32); gv=npy(nr['gt_visibility'],bool)
        ctracks=npy(cr['pred_tracks'],np.float32); cvis=npy(cr['pred_visibility'],bool)
        if window=='segment':
            if not bool(gv[q,t0]):
                stats['skipped_event']+=1; continue
            t1=segment_end(gv,q,t0,max_len=32)
        else:
            t1=min(pred.shape[1], t0+int(window)+1)
        event_touched=False
        for t in range(t0,t1):
            stats['frames_considered']+=1
            if mode=='force_gt_visible' and not bool(gv[q,t]):
                continue
            if mode=='candidate_visible' and not bool(cvis[q,t]):
                continue
            # force_all touches even GT-invisible frames, but visibility eval can penalize; keep for diagnostic if used.
            pred[q,t]=ctracks[q,t]
            pv[q,t]=True
            touched.add((vi,q,t)); event_touched=True
            if bool(gv[q,t]):
                e=err_px(ctracks[q,t],gt[q,t])
                safe_counts['gt_visible_touched']+=1
                safe_counts['safe16']+=int(e<=16)
                safe_counts['safe8']+=int(e<=8)
                safe_counts['safe4']+=int(e<=4)
            else:
                safe_counts['gt_invisible_touched']+=1
        if event_touched:
            stats['event_count']+=1
    stats=dict(stats); safe_counts=dict(safe_counts)
    stats['touched_frames']=len(touched)
    for k,v in safe_counts.items(): stats[k]=int(v)
    if safe_counts.get('gt_visible_touched',0)>0:
        den=safe_counts['gt_visible_touched']
        stats['candidate_safe16_rate_on_touched_gtvis']=safe_counts.get('safe16',0)/den
        stats['candidate_safe8_rate_on_touched_gtvis']=safe_counts.get('safe8',0)/den
        stats['candidate_safe4_rate_on_touched_gtvis']=safe_counts.get('safe4',0)/den
    return recs, stats


def candidate_event_frame_safety(cand_by_vid, native_records, metas, labels):
    n_by={str(r['video_id']):r for r in native_records}
    out={}
    for lname,select in labels.items():
        c=Counter()
        for i,m in enumerate(metas):
            if not bool(select[i]): continue
            vid=m['video_id']; q=int(m['query_idx']); t=int(m['frame_t'])
            nr=n_by[vid]; cr=cand_by_vid[vid]
            gt=npy(nr['gt_tracks'],np.float32); gv=npy(nr['gt_visibility'],bool); cp=npy(cr['pred_tracks'],np.float32); cv=npy(cr['pred_visibility'],bool)
            c['n']+=1; c['candidate_visible']+=int(bool(cv[q,t])); c['gt_visible']+=int(bool(gv[q,t]))
            if bool(gv[q,t]):
                e=err_px(cp[q,t],gt[q,t])
                c['safe16']+=int(e<=16); c['safe8']+=int(e<=8); c['safe4']+=int(e<=4)
        n=max(c['n'],1)
        out[lname]={k:int(v) for k,v in c.items()} | {k+'_rate':float(v/n) for k,v in c.items() if k!='n'}
    return out


def main():
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    z=np.load(FEATURES,allow_pickle=True)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    labels={
        'early4':np.asarray(z['y_reentry_early4']).astype(bool),
        'early8':np.asarray(z['y_reentry_early8']).astype(bool),
        'useful_open_t':np.asarray(z['y_useful_open_t']).astype(bool),
    }
    native_payload=dict(native); native_payload['records']=clone_records(native['records']); native_payload['model_name']='native_v8b_ref'
    native_metric=metric_payload(native_payload,'native')
    rows=[]; inspections={}
    rows.append({'source':'native','variant':'native','metric':native_metric,'delta_vs_native':{k:0.0 for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},'stats':{}})
    for name,path in CANDIDATES.items():
        if not path.exists():
            inspections[name]={'exists':False,'path':str(path)}; continue
        cand=torch.load(path,map_location='cpu',weights_only=False)
        ok,info,cand_by=align_candidate(native,cand)
        inspections[name]={'exists':True,'path':str(path),'align_ok':ok,**info}
        if not ok:
            continue
        # Standalone candidate metric.
        cand_payload=dict(cand); cand_payload['records']=clone_records(cand['records']); cand_payload['model_name']=f'{name}_standalone'
        m=metric_payload(cand_payload,f'{name}_standalone')
        rows.append({'source':name,'variant':f'{name}_standalone','action':'standalone_candidate','metric':m,'delta_vs_native':{k:m[k]-native_metric[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},'stats':{}})
        inspections[name]['event_frame_safety']=candidate_event_frame_safety(cand_by,native['records'],metas,labels)
        for lname,select in labels.items():
            for w in [0,4,8,16,'segment']:
                for mode in ['force_gt_visible','candidate_visible']:
                    recs,stats=apply_candidate_window(native['records'],cand_by,metas,select,window=w,mode=mode)
                    tag=f'{name}_{lname}_w{w}_{mode}'
                    p=dict(native); p['records']=recs; p['model_name']=tag
                    m=metric_payload(p,tag)
                    rows.append({'source':name,'variant':tag,'action':'candidate_coord_window','label':lname,'window':w,'mode':mode,'metric':m,'delta_vs_native':{k:m[k]-native_metric[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},'stats':stats})
    report={
        'script':'scripts/audit_cotracker3_online_v8b_candidate_coord_sources.py',
        'native_cache':str(NATIVE),'features':str(FEATURES),
        'native_metric':native_metric,
        'candidate_inspections':inspections,
        'label_counts':{k:int(v.sum()) for k,v in labels.items()},
        'rows':rows,
        'best_by_AJ_RD_256':sorted(rows,key=lambda r:r['delta_vs_native']['AJ_RD_256'],reverse=True)[:40],
        'best_safe_AJ':sorted([r for r in rows if r['delta_vs_native']['AJ']>=-0.10 and r['delta_vs_native']['OA']>=-0.10],key=lambda r:r['delta_vs_native']['AJ_RD_256'],reverse=True)[:40],
    }
    out=OUTDIR/'v8b_candidate_coord_sources_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({
        'out':str(out),
        'native':{k:round(native_metric[k],4) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
        'inspections':inspections,
        'best_by_AJ_RD_256':[{'variant':r['variant'],'stats':r.get('stats',{}),'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in report['best_by_AJ_RD_256'][:16]],
        'best_safe_AJ':[{'variant':r['variant'],'stats':r.get('stats',{}),'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in report['best_safe_AJ'][:16]],
    },indent=2,ensure_ascii=False))

if __name__=='__main__': main()
