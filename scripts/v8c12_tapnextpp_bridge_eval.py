#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
ROOT=Path('/gemini/code/FSPT'); sys.path.insert(0,str(ROOT))
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, align_candidate, build_events, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native, evaluate_variant
BASE=ROOT/'outputs/paper_discovery_2026-07-05/tapnextpp_smoke'
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c12_tapnextpp_bridge_eval'
CANDS={
 'tapnextpp_baseline':'tapnextpp_davis_baseline_cache.pt',
 'tapnextpp_baseline_w8':'tapnextpp_davis_baseline_w8_cache.pt',
 'tapnextpp_first_v4':'tapnextpp_davis_first_input_cache_v4.pt',
 'tapnextpp_first_v5_conf':'tapnextpp_davis_first_input_cache_v5_conf.pt',
 'tapnextpp_offline':'tapnextpp_davis_offline_cache.pt',
 'tapnextpp_offline_w8':'tapnextpp_davis_offline_w8_cache.pt',
}
def delta(m,n):
    return {k:(m[k]-n[k] if m.get(k) is not None and n.get(k) is not None else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
def bridge(native,cand):
    cby={str(r['video_id']):r for r in cand['records']}; recs=[]; problems=[]
    for nr in native['records']:
        vid=str(nr['video_id']); cr=cby.get(vid)
        if cr is None: problems.append({'video_id':vid,'problem':'missing'}); continue
        if tuple(cr['pred_tracks'].shape)!=tuple(nr['pred_tracks'].shape): problems.append({'video_id':vid,'problem':'shape','cand':tuple(cr['pred_tracks'].shape),'native':tuple(nr['pred_tracks'].shape)}); continue
        rr=dict(nr); rr['pred_tracks']=cr['pred_tracks']; rr['pred_visibility']=cr['pred_visibility']
        for k,v in cr.items():
            if k not in rr and k not in ['gt_tracks','gt_visibility','query_points']: rr[k]=v
        recs.append(rr)
    out=dict(cand); out['records']=recs; out['model_name']=str(cand.get('model_name','tapnextpp'))+'_native_metadata_bridge'
    return out,problems
def small(r):
    if r is None: return None
    return {'variant':r.get('variant'),'metric':{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.get('metric',{}).items() if k in ['AJ','OA','AJ_RD','AJ_RD_256']},'delta':{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.get('delta_vs_native',{}).items()},'pv':{k:r.get('per_video_summary',{}).get(k) for k in ['positive','negative','zero','mean']},'stats':{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.get('stats',{}).items() if k in ['selected_events','event_count','touched_frames','false_visible_rate_on_touched','damage16_rate','repair16_rate']}}
def key(r,k='AJ_RD_256'):
    v=r.get('delta_vs_native',{}).get(k); return -999 if v is None else float(v)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    nm,npv=standard_and_ajrd(clone_records(native['records']))
    metas=rebuild_events_from_native(native)
    rows=[]; info={}
    for name,fn in CANDS.items():
        path=BASE/fn; info[name]={'path':str(path),'exists':path.exists()}
        if not path.exists(): continue
        cand=torch.load(path,map_location='cpu',weights_only=False)
        raw_ok,raw_info,_=align_candidate(native,cand); info[name]['raw_align']=raw_info
        bc,problems=bridge(native,cand); info[name]['bridge_problems']=problems[:10]; info[name]['bridge_problem_count']=len(problems)
        if problems: continue
        ok,br_info,cb=align_candidate(native,bc); info[name]['bridge_align']=br_info
        ev=build_events(native,cb,metas)
        cm,_=standard_and_ajrd(clone_records(bc['records']))
        rows.append({'candidate':name,'candidate_type':'tapnextpp_metadata_bridge','variant':'standalone_bridge','metric':cm,'delta_vs_native':delta(cm,nm),'stats':{},'per_video_summary':{}})
        for w in [8,16]:
            r=evaluate_variant(native,cb,nm,npv,ev,'dist_nc_le64',w,'candidate_visible')
            r['candidate']=name; r['candidate_type']='tapnextpp_metadata_bridge'; rows.append(r)
    by={}
    for name in CANDS:
        cr=[r for r in rows if r.get('candidate')==name]
        if not cr: continue
        by[name]={'standalone':small(next((r for r in cr if r['variant']=='standalone_bridge'),None)),'w8':small(next((r for r in cr if r['variant']=='dist_nc_le64_w8_candidate_visible'),None)),'w16':small(next((r for r in cr if r['variant']=='dist_nc_le64_w16_candidate_visible'),None)),'best':small(sorted(cr,key=lambda r:key(r),reverse=True)[0])}
    rep={'script':'scripts/v8c12_tapnextpp_bridge_eval.py','protocol_note':'native-metadata bridge diagnostic: candidate pred_tracks/pred_visibility with native query/GT metadata','native_metric':nm,'event_count':len(metas),'candidate_info':info,'rows':rows,'by_candidate':by,'best_by_AJ_RD_256':sorted(rows,key=lambda r:key(r),reverse=True)[:30]}
    out=OUT/'v8c12_tapnextpp_bridge_eval_report.json'; out.write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'event_count':len(metas),'by_candidate':by,'best':[small(r) for r in rep['best_by_AJ_RD_256'][:10]]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
