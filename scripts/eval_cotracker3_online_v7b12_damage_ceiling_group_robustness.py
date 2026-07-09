#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
import numpy as np
import torch

ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from scripts.train_cotracker3_online_v7b1_damage_aware_raw_visconf import eval_records, run_ajrd

FEATURES=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
BENEFIT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/benefit_oof_early4_all_components.npy'
DAMAGE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf/damage_oof_bad_topk.npy'
OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b12_damage_ceiling_group_robustness'
OUTDIR.mkdir(parents=True,exist_ok=True)

BASE_LAMBDA=1.35
BASE_THRESHOLD=-0.01


def npy(x: Any, dtype=None):
    if isinstance(x, torch.Tensor):
        x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def err_px(a,b):
    return float(np.linalg.norm((np.asarray(a,np.float32)-np.asarray(b,np.float32))*255.0))


def greedy_folds(native, metas, k=5):
    counts=Counter(m['video_id'] for m in metas)
    vids=[str(r['video_id']) for r in native['records']]
    items=sorted(vids,key=lambda v:counts[v],reverse=True)
    folds=[[] for _ in range(k)]; loads=[0]*k
    for v in items:
        j=min(range(k),key=lambda i:(loads[i],len(folds[i])))
        folds[j].append(v); loads[j]+=counts[v]
    return folds,loads


def subset_payload(native, videos):
    vids=set(videos)
    p=dict(native)
    p['records']=[r for r in native['records'] if str(r['video_id']) in vids]
    return p


def build_event_info(native, metas, y4, y8, yu, benefit, damage, decision):
    rec_by_vid={str(r['video_id']):r for r in native['records']}
    infos=[]
    for i,m in enumerate(metas):
        r=rec_by_vid[m['video_id']]
        q=int(m['query_idx']); t=int(m['frame_t'])
        pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32); gv=npy(r['gt_visibility'],bool); pv=npy(r['pred_visibility'],bool); score=npy(r['pred_vis_score'],np.float32)
        if not bool(gv[q,t]):
            bad=True; safe16=False; safe8=False
        else:
            e=err_px(pred[q,t],gt[q,t]); bad=e>16; safe16=e<=16; safe8=e<=8
        infos.append({
            'idx':i,'video_id':m['video_id'],'query_idx':q,'frame_t':t,
            'y4':bool(y4[i]),'y8':bool(y8[i]),'yu':bool(yu[i]),
            'bad':bad,'safe16':safe16,'safe8':safe8,
            'need_open':(not bool(pv[q,t])) and float(score[q,t])<0.6,
            'benefit':float(benefit[i]),'damage':float(damage[i]),'decision':float(decision[i]),
        })
    return infos


def select_indices(infos, videos=None, *, damage_ceiling=None, benefit_floor=None):
    vids=set(videos) if videos is not None else None
    idx=[]
    for info in infos:
        if vids is not None and info['video_id'] not in vids:
            continue
        if info['decision'] < BASE_THRESHOLD:
            continue
        if damage_ceiling is not None and info['damage'] > damage_ceiling:
            continue
        if benefit_floor is not None and info['benefit'] < benefit_floor:
            continue
        idx.append(info['idx'])
    return idx


def composition(infos, idx):
    c=Counter(); n=len(idx)
    per_video=defaultdict(int)
    for i in idx:
        info=infos[i]
        for k in ['y4','y8','yu','bad','safe16','safe8','need_open']:
            c[k]+=int(info[k])
        per_video[info['video_id']]+=1
    return {'n':n,**{k:int(v) for k,v in c.items()},**{k+'_rate':float(v/max(n,1)) for k,v in c.items()},'videos_with_accept':len(per_video),'max_accept_per_video':max(per_video.values()) if per_video else 0,'per_video_accept':dict(sorted(per_video.items()))}


def records_from_idx(native, idx, metas):
    records=[]
    for r in native['records']:
        rr=dict(r)
        rr['pred_visibility']=npy(r['pred_visibility'],bool).copy()
        records.append(rr)
    vid_to_i={str(r['video_id']):j for j,r in enumerate(records)}
    for i in idx:
        m=metas[i]
        if m['video_id'] not in vid_to_i:
            continue
        vi=vid_to_i[m['video_id']]
        q=int(m['query_idx']); t=int(m['frame_t'])
        records[vi]['pred_visibility'][q,t]=True
    return records


def metric_payload(payload, tag):
    cache=OUTDIR/f'{tag}.pt'; ajrd=OUTDIR/f'{tag}_ajrd.json'
    torch.save(payload,cache)
    std=eval_records(payload['records']); aj=run_ajrd(cache,ajrd)
    return {**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'cache':str(cache)}


def main():
    z=np.load(FEATURES,allow_pickle=True)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    y4=np.asarray(z['y_reentry_early4']).astype(bool); y8=np.asarray(z['y_reentry_early8']).astype(bool); yu=np.asarray(z['y_useful_open_t']).astype(bool)
    benefit=np.load(BENEFIT); damage=np.load(DAMAGE); decision=benefit-BASE_LAMBDA*damage
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    infos=build_event_info(native,metas,y4,y8,yu,benefit,damage,decision)
    folds,loads=greedy_folds(native,metas,5)
    policies=[
        {'name':'base_no_ceiling','damage_ceiling':None,'benefit_floor':None},
        {'name':'damage_ceiling0.7','damage_ceiling':0.7,'benefit_floor':None},
        {'name':'damage_ceiling0.6','damage_ceiling':0.6,'benefit_floor':None},
        {'name':'benefit0.2_damage0.7','damage_ceiling':0.7,'benefit_floor':0.2},
    ]
    # Full30 metrics for the fixed policies.
    native_full_metric=metric_payload(native,'native_full_ref')
    full_rows=[]
    for p in policies:
        idx=select_indices(infos,damage_ceiling=p['damage_ceiling'],benefit_floor=p['benefit_floor'])
        payload=dict(native); payload['records']=records_from_idx(native,idx,metas); payload['model_name']='v7b12_'+p['name']
        m=metric_payload(payload,'v7b12_full_'+p['name'])
        delta={k:m[k]-native_full_metric[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
        full_rows.append({'policy':p,'composition':composition(infos,idx),'metric':m,'delta_vs_native':delta})
    # Fold robustness metrics against fold-native.
    fold_rows=[]
    for fi,vids in enumerate(folds):
        native_sub=subset_payload(native,vids)
        native_metric=metric_payload(native_sub,f'native_fold{fi}')
        for p in policies:
            idx=select_indices(infos,videos=vids,damage_ceiling=p['damage_ceiling'],benefit_floor=p['benefit_floor'])
            payload=dict(native_sub); payload['records']=records_from_idx(native_sub,idx,metas); payload['model_name']=f"v7b12_{p['name']}_fold{fi}"
            m=metric_payload(payload,f"v7b12_{p['name']}_fold{fi}")
            delta={k:m[k]-native_metric[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            fold_rows.append({'fold':fi,'videos':vids,'load':loads[fi],'policy':p,'composition':composition(infos,idx),'metric':m,'native_metric':native_metric,'delta_vs_fold_native':delta})
    agg={}
    for p in policies:
        name=p['name']; rs=[r for r in fold_rows if r['policy']['name']==name]
        agg[name]={
            'policy':p,'fold_count':len(rs),
            'mean_delta':{k:float(np.mean([r['delta_vs_fold_native'][k] for r in rs])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'min_delta':{k:float(np.min([r['delta_vs_fold_native'][k] for r in rs])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'max_delta':{k:float(np.max([r['delta_vs_fold_native'][k] for r in rs])) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},
            'success_folds':int(sum(1 for r in rs if r['delta_vs_fold_native']['AJ']>=-0.10 and r['delta_vs_fold_native']['OA']>=-0.10 and r['delta_vs_fold_native']['AJ_RD']>=0.0015 and r['delta_vs_fold_native']['AJ_RD_256']>=0.004)),
            'mean_accept':float(np.mean([r['composition']['n'] for r in rs])),
            'mean_safe16_rate':float(np.mean([r['composition'].get('safe16_rate',0) for r in rs])),
            'mean_bad_rate':float(np.mean([r['composition'].get('bad_rate',0) for r in rs])),
        }
    report={'script':'scripts/eval_cotracker3_online_v7b12_damage_ceiling_group_robustness.py','base_setting':{'lambda':BASE_LAMBDA,'threshold':BASE_THRESHOLD},'folds':folds,'loads':loads,'native_full_metric':native_full_metric,'full_rows':full_rows,'fold_rows':fold_rows,'aggregate':agg}
    out=OUTDIR/'v7b12_damage_ceiling_group_robustness_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'full_rows':[{ 'name':r['policy']['name'],'n':r['composition']['n'],'safe16_rate':r['composition'].get('safe16_rate'),'bad_rate':r['composition'].get('bad_rate'),'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in full_rows], 'aggregate':agg},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
