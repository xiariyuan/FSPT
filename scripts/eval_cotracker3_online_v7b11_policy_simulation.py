#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict, Counter
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
OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b11_policy_simulation'
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

def select_indices(infos, *, topk=None, benefit_floor=None, damage_ceiling=None, threshold=BASE_THRESHOLD):
    idx=[info['idx'] for info in infos if info['decision']>=threshold]
    if benefit_floor is not None:
        idx=[i for i in idx if infos[i]['benefit']>=benefit_floor]
    if damage_ceiling is not None:
        idx=[i for i in idx if infos[i]['damage']<=damage_ceiling]
    if topk is not None:
        by=defaultdict(list)
        for i in idx:
            by[infos[i]['video_id']].append(i)
        kept=[]
        for _,lst in by.items():
            kept.extend(sorted(lst,key=lambda i:infos[i]['decision'],reverse=True)[:topk])
        idx=kept
    return sorted(set(idx))

def composition(infos, idx):
    c=Counter(); n=len(idx)
    for i in idx:
        for k in ['y4','y8','yu','bad','safe16','safe8','need_open']:
            c[k]+=int(infos[i][k])
    per_video=defaultdict(int)
    for i in idx:
        per_video[infos[i]['video_id']]+=1
    return {'n':n,**{k:int(v) for k,v in c.items()},**{k+'_rate':float(v/max(n,1)) for k,v in c.items()},'videos_with_accept':len(per_video),'max_accept_per_video':max(per_video.values()) if per_video else 0,'top_video_accepts':sorted(per_video.items(),key=lambda kv:kv[1],reverse=True)[:10]}

def records_from_idx(native, metas, idx):
    records=[]
    for r in native['records']:
        rr=dict(r)
        rr['pred_visibility']=npy(r['pred_visibility'],bool).copy()
        records.append(rr)
    vid_to_i={str(r['video_id']):j for j,r in enumerate(records)}
    for i in idx:
        m=metas[i]
        vi=vid_to_i[m['video_id']]
        q=int(m['query_idx']); t=int(m['frame_t'])
        records[vi]['pred_visibility'][q,t]=True
    return records

def metric_for(native, metas, idx, tag):
    payload=dict(native)
    payload['records']=records_from_idx(native,metas,idx)
    payload['model_name']=tag
    cache=OUTDIR/f'{tag}.pt'; ajrd=OUTDIR/f'{tag}_ajrd.json'
    torch.save(payload,cache)
    std=eval_records(payload['records']); aj=run_ajrd(cache,ajrd)
    return {**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256'],'opened':len(idx),'cache':str(cache)}

def main():
    z=np.load(FEATURES,allow_pickle=True)
    metas=[json.loads(str(m)) for m in z['meta_json'].tolist()]
    y4=np.asarray(z['y_reentry_early4']).astype(bool); y8=np.asarray(z['y_reentry_early8']).astype(bool); yu=np.asarray(z['y_useful_open_t']).astype(bool)
    benefit=np.load(BENEFIT); damage=np.load(DAMAGE); decision=benefit-BASE_LAMBDA*damage
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    native_std=eval_records(native['records']); native_aj=run_ajrd(NATIVE,OUTDIR/'native_ref_ajrd.json')
    native_row={**native_std,'AJ_RD':native_aj['true_AJ_RD'],'AJ_RD_256':native_aj['true_AJ_RD_256']}
    infos=build_event_info(native,metas,y4,y8,yu,benefit,damage,decision)
    policies=[]
    policies.append({'name':'base_lam1.35_thr-0.01','topk':None,'benefit_floor':None,'damage_ceiling':None})
    for k in [1,2,3,5,8,10,12,15]:
        policies.append({'name':f'topk{k}','topk':k,'benefit_floor':None,'damage_ceiling':None})
    for bf in [0.2,0.3,0.4,0.5,0.6]:
        policies.append({'name':f'benefit_floor{bf:.1f}','topk':None,'benefit_floor':bf,'damage_ceiling':None})
    for dc in [0.5,0.6,0.7,0.8,0.9]:
        policies.append({'name':f'damage_ceiling{dc:.1f}','topk':None,'benefit_floor':None,'damage_ceiling':dc})
    for k in [5,8,10]:
        for bf in [0.2,0.3]:
            policies.append({'name':f'topk{k}_benefit{bf:.1f}','topk':k,'benefit_floor':bf,'damage_ceiling':None})
    for k in [5,8,10]:
        for dc in [0.7,0.8,0.9]:
            policies.append({'name':f'topk{k}_damage{dc:.1f}','topk':k,'benefit_floor':None,'damage_ceiling':dc})
    for bf in [0.2,0.3]:
        for dc in [0.7,0.8,0.9]:
            policies.append({'name':f'benefit{bf:.1f}_damage{dc:.1f}','topk':None,'benefit_floor':bf,'damage_ceiling':dc})
    for k in [5,8]:
        for bf in [0.2,0.3]:
            for dc in [0.8,0.9]:
                policies.append({'name':f'topk{k}_benefit{bf:.1f}_damage{dc:.1f}','topk':k,'benefit_floor':bf,'damage_ceiling':dc})
    rows=[]
    seen=set()
    for p in policies:
        key=(p['topk'],p['benefit_floor'],p['damage_ceiling'])
        if key in seen: continue
        seen.add(key)
        idx=select_indices(infos,topk=p['topk'],benefit_floor=p['benefit_floor'],damage_ceiling=p['damage_ceiling'])
        if not idx: continue
        tag='v7b11_'+p['name'].replace('.','p').replace('-','m')
        metric=metric_for(native,metas,idx,tag)
        delta={k:metric[k]-native_row[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
        rows.append({'policy':p,'composition':composition(infos,idx),'metric':metric,'delta_vs_native':delta})
    constrained=[r for r in rows if r['delta_vs_native']['AJ']>=-0.10 and r['delta_vs_native']['OA']>=-0.10]
    success=[r for r in constrained if r['delta_vs_native']['AJ_RD']>=0.0015 and r['delta_vs_native']['AJ_RD_256']>=0.004]
    report={'script':'scripts/eval_cotracker3_online_v7b11_policy_simulation.py','base_setting':{'lambda':BASE_LAMBDA,'threshold':BASE_THRESHOLD},'native_row':native_row,'rows':rows,'success_count':len(success),'success_rows':success,'best_constrained':sorted(constrained,key=lambda r:(r['delta_vs_native']['AJ_RD_256'],r['delta_vs_native']['AJ_RD'],r['delta_vs_native']['AJ']),reverse=True)[:30],'best_raw':sorted(rows,key=lambda r:(r['delta_vs_native']['AJ_RD_256'],r['delta_vs_native']['AJ_RD']),reverse=True)[:30]}
    out=OUTDIR/'v7b11_policy_simulation_report.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'n_policies':len(rows),'success_count':len(success),'best_constrained':[{**r['policy'],'n':r['composition']['n'],'safe16_rate':r['composition'].get('safe16_rate'), 'bad_rate':r['composition'].get('bad_rate'), 'delta':{k:round(v,4) for k,v in r['delta_vs_native'].items()}} for r in report['best_constrained'][:15]]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
