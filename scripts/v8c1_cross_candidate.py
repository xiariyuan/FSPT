#!/usr/bin/env python3
import json, sys
from pathlib import Path
import torch
ROOT=Path('/gemini/code/FSPT'); sys.path.insert(0,str(ROOT))
from scripts.audit_cotracker3_online_v8b_candidate_coord_sources import CANDIDATES
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, align_candidate, build_events, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native, evaluate_variant
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c1_cross_candidate_generalization'

def delt(m,n):
    return {k:(m[k]-n[k] if m.get(k) is not None and n.get(k) is not None else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}

def small(r):
    if not r: return None
    return {'variant':r.get('variant'),'delta':{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.get('delta_vs_native',{}).items()},'pv':{k:r.get('per_video_summary',{}).get(k) for k in ['positive','negative','zero','mean']},'stats':{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.get('stats',{}).items() if k in ['selected_events','event_count','touched_frames','false_visible_rate_on_touched','damage16_rate']}}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    nm,npv=standard_and_ajrd(clone_records(native['records']))
    metas=rebuild_events_from_native(native)
    rows=[]; insp={}
    for name,path in CANDIDATES.items():
        cand=torch.load(path,map_location='cpu',weights_only=False)
        ok,info,cb=align_candidate(native,cand); insp[name]=info|{'path':str(path),'exists':path.exists()}
        if not ok: continue
        ev=build_events(native,cb,metas)
        cm,cpv=standard_and_ajrd(clone_records(cand['records']))
        rows.append({'candidate':name,'variant':'standalone','metric':cm,'delta_vs_native':delt(cm,nm),'stats':{},'per_video_summary':{}})
        for filt in ['all','dist_nc_le64']:
            for w in [8,16]:
                r=evaluate_variant(native,cb,nm,npv,ev,filt,w,'candidate_visible')
                r['candidate']=name; rows.append(r)
    def key(r,k='AJ_RD_256'):
        v=r.get('delta_vs_native',{}).get(k); return -999 if v is None else float(v)
    by={}
    for name in CANDIDATES:
        cr=[r for r in rows if r.get('candidate')==name]
        if not cr: continue
        by[name]={
          'standalone':small(next((r for r in cr if r['variant']=='standalone'),None)),
          'default':small(next((r for r in cr if r['variant']=='dist_nc_le64_w8_candidate_visible'),None)),
          'high_gain':small(next((r for r in cr if r['variant']=='dist_nc_le64_w16_candidate_visible'),None)),
          'best':small(sorted(cr,key=lambda r:key(r),reverse=True)[0])}
    rep={'script':'scripts/v8c1_cross_candidate.py','native_metric':nm,'event_count':len(metas),'candidate_inspections':insp,'rows':rows,'by_candidate':by,'best_by_AJ_RD_256':sorted(rows,key=lambda r:key(r),reverse=True)[:30]}
    out=OUT/'v8c1_cross_candidate_generalization_report.json'; out.write_text(json.dumps(rep,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'event_count':len(metas),'by_candidate':by,'best':[small(r) for r in rep['best_by_AJ_RD_256'][:10]]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
