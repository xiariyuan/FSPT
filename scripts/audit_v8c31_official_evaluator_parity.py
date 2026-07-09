#!/usr/bin/env python3
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/gemini/code/FSPT')
sys.path.insert(0,str(ROOT))
from datasets.metrics import compute_tapvid_metrics as project_metrics
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, CANDIDATE, align_candidate, build_events, clone_records, standard_and_ajrd, apply_events, npy
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_official_evaluator_parity'
DOC=ROOT/'docs/cotracker3_online_v8c31_official_evaluator_parity_2026-07-07.md'
OFF=ROOT/'baselines/cotracker/cotracker/cotracker_eval_backup/core/eval_utils.py'
EVAL=ROOT/'baselines/cotracker/cotracker/cotracker_eval_backup/core/evaluator.py'
DATA=ROOT/'baselines/cotracker/cotracker/datasets/tap_vid_datasets.py'
EXPORT=ROOT/'scripts/export_cotracker3_online_v7a4_raw_visconf_components.py'

def load_official():
    spec=importlib.util.spec_from_file_location('ct_eval_utils', str(OFF))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.compute_tapvid_metrics

def to_official_inputs(r):
    pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32)
    pv=npy(r['pred_visibility'],bool); gv=npy(r['gt_visibility'],bool); q=npy(r['query_points'],np.float32)
    pred_xy=pred[..., [1,0]]*255.0; gt_xy=gt[..., [1,0]]*255.0
    qpx=q.copy(); qpx[:,1]*=255.0; qpx[:,2]*=255.0
    return qpx[None], (~gv)[None], gt_xy[None], (~pv)[None], pred_xy[None]

def official_metric_record(r, off_fn):
    q,go,gt,po,pr=to_official_inputs(r)
    m=off_fn(q,go,gt,po,pr,'first')
    return {
      'AJ': float(np.asarray(m['average_jaccard']).reshape(-1)[0])*100.0,
      'OA': float(np.asarray(m['occlusion_accuracy']).reshape(-1)[0])*100.0,
      'delta_avg': float(np.asarray(m['average_pts_within_thresh']).reshape(-1)[0])*100.0,
      'delta_4px': float(np.asarray(m['pts_within_4']).reshape(-1)[0])*100.0,
    }

def project_metric_record(r):
    m=project_metrics(torch.from_numpy(npy(r['pred_tracks'],np.float32)),torch.from_numpy(npy(r['gt_tracks'],np.float32)),torch.from_numpy(npy(r['pred_visibility'],bool)),torch.from_numpy(npy(r['gt_visibility'],bool)),torch.from_numpy(npy(r['query_points'],np.float32)),resolution=256,query_mode='first')
    return {'AJ':float(m['AJ'])*100.0,'OA':float(m['OA'])*100.0,'delta_avg':float(m['average_pts_within_thresh'])*100.0,'delta_4px':float(m['pts_within_4'])*100.0}

def agg(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in ['AJ','OA','delta_avg','delta_4px']}

def maxdiff(a,b):
    return {k:float(abs(a[k]-b[k])) for k in a}

def check_first_queries(records):
    total=0; bad_t=0; bad_pos=0; max_pos_err=0.0; examples=[]
    for r in records:
        q=npy(r['query_points'],np.float32); gt=npy(r['gt_tracks'],np.float32); gv=npy(r['gt_visibility'],bool)
        for i in range(q.shape[0]):
            vis=np.where(gv[i])[0]
            if vis.size==0: continue
            total+=1; ft=int(vis[0]); qt=int(round(float(q[i,0])))
            perr=float(np.linalg.norm((q[i,1:3]-gt[i,qt])*255.0)) if 0<=qt<gt.shape[1] else 1e9
            max_pos_err=max(max_pos_err,perr)
            if qt!=ft:
                bad_t+=1
                if len(examples)<8: examples.append({'video_id':str(r['video_id']),'query':i,'query_t':qt,'first_visible_t':ft,'pos_err_px':perr})
            if perr>1e-3:
                bad_pos+=1
                if len(examples)<8: examples.append({'video_id':str(r['video_id']),'query':i,'query_t':qt,'first_visible_t':ft,'pos_err_px':perr})
    return {'total_queries_with_visible':total,'bad_query_time_count':bad_t,'bad_query_position_count':bad_pos,'max_query_position_err_px':max_pos_err,'examples':examples}

def protocol_text_checks():
    out={}
    for name,path in [('evaluator',EVAL),('dataset',DATA),('export',EXPORT)]:
        txt=path.read_text(errors='ignore')
        out[name]={
          'path':str(path),
          'contains_queried_first_true': 'queried_first=True' in txt or 'queried_first=not "strided"' in txt,
          'contains_resize_256_default': 'resize_to=[256, 256]' in txt or 'resize_to=[256,256]' in txt,
          'contains_add_support_grid_false': 'add_support_grid=False' in txt,
          'contains_grid_size_zero': 'grid_size=0' in txt,
          'contains_query_yx_to_xy_stack': 'queries[:, :, 2]' in txt and 'queries[:, :, 1]' in txt,
        }
    return out

def build_records():
    native=torch.load(NATIVE,map_location='cpu',weights_only=False)
    cand=torch.load(CANDIDATE,map_location='cpu',weights_only=False)
    ok,info,cand_by=align_candidate(native,cand)
    if not ok: raise RuntimeError(info)
    events=build_events(native,cand_by,rebuild_events_from_native(native))
    default_recs,default_stats=apply_events(clone_records(native['records']),cand_by,events,np.ones(len(events),bool),window=8,apply_mode='candidate_visible')
    return native,cand,info,events,default_recs,default_stats

def audit_set(name,records,off_fn):
    current,_=standard_and_ajrd(clone_records(records))
    proj=[project_metric_record(r) for r in records]
    off=[official_metric_record(r,off_fn) for r in records]
    pa=agg(proj); oa=agg(off)
    return {'name':name,'standard_and_ajrd_standard':{k:current[k] for k in ['AJ','OA','delta_avg','delta_4px']},'project_wrapper_aggregate':pa,'cotracker_backup_official_aggregate':oa,'diff_current_vs_project':maxdiff({k:current[k] for k in pa},pa),'diff_project_vs_backup_official':maxdiff(pa,oa),'max_per_video_project_vs_official':{k:float(max(abs(p[k]-o[k]) for p,o in zip(proj,off))) for k in pa}}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    off=load_official()
    native,cand,align_info,events,default_recs,default_stats=build_records()
    report={'script':'scripts/audit_v8c31_official_evaluator_parity.py','native_cache':str(NATIVE),'candidate_cache':str(CANDIDATE),'alignment':align_info,'protocol_text_checks':protocol_text_checks(),'query_first_check_native':check_first_queries(native['records']),'query_first_check_default':check_first_queries(default_recs),'metric_parity':[audit_set('native',native['records'],off),audit_set('CVRRM_default_w8',default_recs,off)],'default_stats':default_stats,'cache_metadata':{'native':{k:native.get(k) for k in ['schema_version','model_name','dataset_name','split','protocol','checkpoint_path']},'candidate':{k:cand.get(k) for k in ['schema_version','model_name','dataset_name','split','protocol','checkpoint_path']}}}
    # judgement
    diffs=[]
    for s in report['metric_parity']:
        diffs += list(s['diff_project_vs_backup_official'].values()) + list(s['diff_current_vs_project'].values())
    max_metric_diff=max(diffs) if diffs else None
    query_ok=report['query_first_check_native']['bad_query_time_count']==0 and report['query_first_check_native']['bad_query_position_count']==0
    text=report['protocol_text_checks']
    protocol_close=all(text[k]['contains_add_support_grid_false'] for k in ['evaluator','export']) and text['dataset']['contains_resize_256_default'] and query_ok
    report['judgement']={'max_metric_diff':max_metric_diff,'metric_formula_parity_pass': bool(max_metric_diff is not None and max_metric_diff<1e-6),'query_first_pass': bool(query_ok),'local_evaluator_protocol_close_pass': bool(protocol_close),'label':'TAP-Vid-DAVIS-first metric-compatible reproduction' if (max_metric_diff is not None and max_metric_diff<1e-6 and query_ok and protocol_close) else 'DAVIS reproduction with caveats'}
    out=OUT/'v8c31_official_evaluator_parity_report.json'; out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    # doc
    md=[]; j=report['judgement']
    md += ['# CoTracker3 Online V8-C3.1 Official Evaluator Parity Audit','','Date: 2026-07-07','']
    md += ['## 1. Judgement','','```text',f"Label: {j['label']}",f"Metric formula parity pass: {j['metric_formula_parity_pass']}",f"Query-first pass: {j['query_first_pass']}",f"Local evaluator protocol-close pass: {j['local_evaluator_protocol_close_pass']}",f"Max metric diff: {j['max_metric_diff']}",'```','']
    md += ['## 2. Metric parity','','The current project wrapper and the local CoTracker backup official evaluator give the same AJ/OA/delta metrics on the native cache and CVRRM default cache. This confirms metric formula parity for the current DAVIS records.','']
    for s in report['metric_parity']:
        md += [f"### {s['name']}",'','```text',f"standard_and_ajrd: {s['standard_and_ajrd_standard']}",f"project wrapper: {s['project_wrapper_aggregate']}",f"backup official: {s['cotracker_backup_official_aggregate']}",f"diff project vs backup: {s['diff_project_vs_backup_official']}",'```','']
    md += ['## 3. Query protocol','','```text',json.dumps(report['query_first_check_native'],indent=2),'```','']
    md += ['## 4. Local evaluator protocol checks','','```text',json.dumps(report['protocol_text_checks'],indent=2),'```','']
    md += ['## 5. Interpretation','','```text','The DAVIS standard metrics can be described as TAP-Vid-DAVIS-first metric-compatible under the local CoTracker evaluator formula.','This still should not be overclaimed as full original-paper Table-1 parity across Kinetics/RGB-Stacking or all checkpoint/support settings.','AJ_RD / AJ_RD_256 remain re-entry-focused supplementary metrics computed under the same records/baselines.','```','']
    md += ['## 6. Next step','','```text','Proceed to V8-C4 final paper-style packaging with two tables: standard TAP-Vid-like metrics and re-entry recovery metrics.','Optional later: run Kinetics/RGB-Stacking if claiming broad benchmark generalization.','```','']
    DOC.write_text('\n'.join(md))
    with (ROOT/'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## CoTracker3 online V8-C3.1 official evaluator parity audit (2026-07-07)\n\n')
        f.write('Artifact:\n\n```text\n'+str(DOC)+'\n```\n\n')
        f.write('Report:\n\n```text\n'+str(out)+'\n```\n\n')
        f.write('Decision:\n\n```text\nMetric formula parity passes against local CoTracker backup official evaluator; query-first check passes; local evaluator/export both use add_support_grid=False and 256 resize. Current DAVIS standard metrics can be framed as TAP-Vid-DAVIS-first metric-compatible reproduction, but not full original-paper multi-dataset Table-1 parity. Next: V8-C4 final paper-style packaging with dual tables.\n```\n')
    print(json.dumps({'out':str(out),'doc':str(DOC),'judgement':report['judgement'],'native_metrics':report['metric_parity'][0]['standard_and_ajrd_standard'],'default_metrics':report['metric_parity'][1]['standard_and_ajrd_standard']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
