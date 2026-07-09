#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'baselines/cotracker'))
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE,CANDIDATE,align_candidate,build_events,clone_records,standard_and_ajrd,apply_events,npy
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native
from cotracker.cotracker_eval_backup.core.eval_utils import compute_tapvid_metrics as ct_metric
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_eval_parity'

def backup_metric_for_record(r):
    pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32)
    pv=npy(r['pred_visibility'],bool); gv=npy(r['gt_visibility'],bool); q=npy(r['query_points'],np.float32).copy()
    # records store normalized y,x; backup CoTracker eval expects raster x,y relative to 256.
    pred_xy=pred[..., [1,0]]*255.0
    gt_xy=gt[..., [1,0]]*255.0
    q_px=q.copy(); q_px[:,1]*=255.0; q_px[:,2]*=255.0
    out=ct_metric(q_px[None], (~gv)[None], gt_xy[None], (~pv)[None], pred_xy[None], query_mode='first')
    return {
        'AJ': float(np.asarray(out['average_jaccard']).reshape(-1)[0])*100.0,
        'OA': float(np.asarray(out['occlusion_accuracy']).reshape(-1)[0])*100.0,
        'delta_avg': float(np.asarray(out['average_pts_within_thresh']).reshape(-1)[0])*100.0,
        'delta_4px': float(np.asarray(out['pts_within_4']).reshape(-1)[0])*100.0,
    }

def backup_standard(records):
    rows=[]
    for r in records:
        m=backup_metric_for_record(r); m['video_id']=str(r['video_id']); rows.append(m)
    agg={k:float(np.mean([x[k] for x in rows])) for k in ['AJ','OA','delta_avg','delta_4px']}
    return agg,rows

def diff(a,b):
    return {k:float(a[k]-b[k]) for k in ['AJ','OA','delta_avg','delta_4px']}

def maxabs(ds):
    return max(abs(v) for d in ds for v in d.values()) if ds else 0.0

def first_query_audit(records):
    total=0; bad_visible=0; bad_prior=0; examples=[]
    for r in records:
        gv=npy(r['gt_visibility'],bool); qpts=npy(r['query_points'],np.float32)
        for qi,q in enumerate(qpts):
            t=int(round(float(q[0]))); total+=1
            ok_vis=0<=t<gv.shape[1] and bool(gv[qi,t])
            ok_prior=True if t<=0 else not bool(gv[qi,:t].any())
            if not ok_vis: bad_visible+=1
            if not ok_prior: bad_prior+=1
            if (not ok_vis or not ok_prior) and len(examples)<10:
                examples.append({'video_id':str(r['video_id']),'query_idx':qi,'query_t':t,'visible_at_query':ok_vis,'prior_visible':not ok_prior})
    return {'n_queries':total,'bad_query_not_visible':bad_visible,'bad_prior_visible_before_query':bad_prior,'examples':examples}

def make_variants():
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    cand=torch.load(CANDIDATE,map_location='cpu',weights_only=False)
    ok,info,cb=align_candidate(native,cand)
    if not ok: raise RuntimeError(info)
    metas=rebuild_events_from_native(native)
    events=build_events(native,cb,metas)
    default_recs, default_stats=apply_events(clone_records(native['records']),cb,events,np.array([True]*len(events)),window=8,apply_mode='candidate_visible')
    # Above is all-events w8. For dist<=64 default, select by event distance.
    sel=np.array([e.dist_nc_px<=64.0 for e in events],dtype=bool)
    default_recs, default_stats=apply_events(clone_records(native['records']),cb,events,sel,window=8,apply_mode='candidate_visible')
    return native,cand,{'native':clone_records(native['records']),'trackon2':clone_records(cand['records']),'cvrm_dist64_w8':default_recs},{'alignment':info,'event_count':len(events),'default_stats':default_stats}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    native,cand,variants,meta=make_variants()
    variant_reports={}
    for name,recs in variants.items():
        project,_=standard_and_ajrd(clone_records(recs))
        backup,per=backup_standard(recs)
        d=diff(project,backup)
        variant_reports[name]={'project_standard':{k:project[k] for k in ['AJ','OA','delta_avg','delta_4px']},'backup_official_standard':backup,'diff_project_minus_backup':d,'max_abs_diff':max(abs(x) for x in d.values()),'per_video_backup':per}
    q_audit=first_query_audit(native['records'])
    protocol_gap={
        'dataset':'tapvid_davis full30 records',
        'query_mode':'first, audited from query_points and gt_visibility',
        'metric_parity':'project wrapper vs CoTracker backup official metric on identical records',
        'native_cache_protocol':native.get('protocol'),
        'native_schema':native.get('schema_version'),
        'candidate_protocol':cand.get('protocol'),
        'known_remaining_gap':'This audits metric/query compatibility on cached records. It does not rerun the full original paper evaluator with one-query/support-point variants beyond the cached true-streaming export.',
        'two_source_note':'CVRRM + TrackOn2 is a two-source output-level recovery system, not a single-model CoTracker3 result.'
    }
    report={'script':'scripts/audit_v8c31_eval_parity.py','sources':{'native':str(NATIVE),'candidate':str(CANDIDATE)},'protocol_gap':protocol_gap,'alignment_and_events':meta,'first_query_audit':q_audit,'variants':variant_reports,'summary':{'max_abs_diff_all_variants':max(v['max_abs_diff'] for v in variant_reports.values()),'query_audit_pass':q_audit['bad_query_not_visible']==0 and q_audit['bad_prior_visible_before_query']==0,'metric_parity_pass':max(v['max_abs_diff'] for v in variant_reports.values())<1e-5}}
    out=OUT/'v8c31_eval_parity_report.json'; out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out':str(out),'summary':report['summary'],'variant_diffs':{k:{kk:round(vv,8) for kk,vv in v['diff_project_minus_backup'].items()} for k,v in variant_reports.items()},'first_query_audit':q_audit,'native_project':variant_reports['native']['project_standard'],'native_backup':variant_reports['native']['backup_official_standard']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
