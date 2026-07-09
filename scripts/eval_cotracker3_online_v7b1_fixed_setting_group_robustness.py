#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from scripts.train_cotracker3_online_v7b1_damage_aware_raw_visconf import eval_records, run_ajrd, build_open_records

FEATURES=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
BENEFIT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy'
DAMAGE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy'
OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/group_robustness'
OUTDIR.mkdir(parents=True,exist_ok=True)

def metric(payload, tag):
    cache=OUTDIR/f'{tag}.pt'; ajrd=OUTDIR/f'{tag}_ajrd.json'
    torch.save(payload,cache)
    std=eval_records(payload['records']); aj=run_ajrd(cache,ajrd)
    return {**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'cache':str(cache)}

def subset_payload(native, videos):
    vids=set(videos)
    p=dict(native)
    p['records']=[r for r in native['records'] if str(r['video_id']) in vids]
    return p

def greedy_folds(native, metas, k=5):
    counts=Counter(m['video_id'] for m in metas)
    vids=[str(r['video_id']) for r in native['records']]
    items=sorted(vids,key=lambda v:counts[v],reverse=True)
    folds=[[] for _ in range(k)]; loads=[0]*k
    for v in items:
        j=min(range(k),key=lambda i:(loads[i],len(folds[i])))
        folds[j].append(v); loads[j]+=counts[v]
    return folds,loads

def composition(metas, decision, thr, videos, y4, yu):
    vids=set(videos); idx=[i for i,m in enumerate(metas) if m['video_id'] in vids and decision[i]>=thr]
    if not idx: return {'accept':0}
    return {'accept':len(idx),'benefit_rate':float(y4[idx].mean()),'useful_rate':float(yu[idx].mean())}

def main():
    z=np.load(FEATURES,allow_pickle=True)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    y4=np.asarray(z['y_reentry_early4']).astype(bool); yu=np.asarray(z['y_useful_open_t']).astype(bool)
    benefit=np.load(BENEFIT); damage=np.load(DAMAGE)
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    folds,loads=greedy_folds(native,metas,5)
    candidates=[
        {'name':'best_refine_lam1.35_thr-0.01','lambda':1.35,'threshold':-0.01},
        {'name':'best_refine_lam1.30_thr0.01','lambda':1.30,'threshold':0.01},
        {'name':'best_refine_lam1.25_thr0.03','lambda':1.25,'threshold':0.03},
        {'name':'original_lam1.25_thr0.0','lambda':1.25,'threshold':0.0},
        {'name':'healthy_lam1.50_thr-0.10','lambda':1.50,'threshold':-0.10},
    ]
    rows=[]
    for fi,vids in enumerate(folds):
        native_sub=subset_payload(native,vids)
        nmet=metric(native_sub,f'native_fold{fi}')
        for c in candidates:
            dec=benefit-c['lambda']*damage
            # Build modified records on this fold only using full metas/decision; non-fold videos ignored by subset payload.
            recs,opened=build_open_records(native_sub,metas,dec,c['threshold'])
            p=dict(native_sub); p['records']=recs; p['model_name']=f"v7b1_{c['name']}_fold{fi}"
            mmet=metric(p,f"v7b1_{c['name']}_fold{fi}")
            delta={k:mmet[k]-nmet[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            rows.append({'fold':fi,'videos':vids,'event_load':loads[fi],'candidate':c,'opened':len(opened),'composition':composition(metas,dec,c['threshold'],vids,y4,yu),'native_metric':nmet,'metric':mmet,'delta_vs_fold_native':delta})
    # aggregate by candidate
    agg={}
    for c in candidates:
        cr=[r for r in rows if r['candidate']['name']==c['name']]
        agg[c['name']]={
            'candidate':c,
            'fold_count':len(cr),
            'mean_delta':{k:float(np.mean([r['delta_vs_fold_native'][k] for r in cr])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'min_delta':{k:float(np.min([r['delta_vs_fold_native'][k] for r in cr])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'max_delta':{k:float(np.max([r['delta_vs_fold_native'][k] for r in cr])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'success_folds':int(sum(1 for r in cr if r['delta_vs_fold_native']['AJ']>=-0.10 and r['delta_vs_fold_native']['OA']>=-0.10 and r['delta_vs_fold_native']['AJ_RD']>=0.0015 and r['delta_vs_fold_native']['AJ_RD_256']>=0.004)),
            'opened': [r['opened'] for r in cr],
        }
    report={'script':'scripts/eval_cotracker3_online_v7b1_fixed_setting_group_robustness.py','folds':folds,'loads':loads,'rows':rows,'aggregate':agg}
    out=OUTDIR/'v7b1_fixed_setting_group_robustness_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'fold_loads':loads,'aggregate':agg},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
