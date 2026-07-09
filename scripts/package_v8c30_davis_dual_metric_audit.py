#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path('/gemini/code/FSPT')
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c30_davis_dual_metric_audit'
DOC=ROOT/'docs/cotracker3_online_v8c30_davis_dual_metric_audit_2026-07-07.md'
P={
 'c01':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c01_recovery_risk_audit/v8c01_recovery_risk_audit_report.json',
 'c13':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c13_candidate_source_table/v8c13_candidate_source_table.json',
 'tap':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c12_tapnextpp_bridge_eval/v8c12_tapnextpp_bridge_eval_report.json',
 'ver':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c04_apply_fine_risk_verifier/v8c04_apply_fine_risk_verifier_report.json',
 'state':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c21_full_hook_eval/v8c21_full_report.json',
 'v7':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b12_damage_ceiling_group_robustness/v7b12_damage_ceiling_group_robustness_report.json',
}
def load(k): return json.loads(P[k].read_text())
def find_row(rep,variant,source=None):
    for r in rep['rows']:
        if r.get('variant')==variant and (source is None or r.get('event_source')==source): return r
    raise KeyError((variant,source))
def find_name(rows,name):
    for r in rows:
        if r.get('name')==name: return r
    raise KeyError(name)
def v7row(v7,name):
    for r in v7['full_rows']:
        if r['policy']['name']==name: return r
    raise KeyError(name)
def dlt(metric,native):
    return {k:(metric[k]-native[k] if metric.get(k) is not None and native.get(k) is not None else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
def std_entry(name,protocol,metric,delta,status,notes):
    return {'name':name,'protocol':protocol,'metric':{k:metric.get(k) for k in ['AJ','OA','delta_avg','delta_4px']},'delta_vs_native':{k:delta.get(k) for k in ['AJ','OA','delta_avg','delta_4px']},'status':status,'notes':notes}
def re_entry(name,protocol,metric,delta,status,notes,stats=None,rob=None):
    stats=stats or {}; rob=rob or {}
    return {'name':name,'protocol':protocol,'metric':{k:metric.get(k) for k in ['AJ_RD','AJ_RD_256']},'delta_vs_native':{k:delta.get(k) for k in ['AJ_RD','AJ_RD_256']},'repair_damage':{k:stats.get(k) for k in ['repair16_rate','damage16_rate','false_visible_rate_on_touched','touched_frames','event_count','selected_events'] if stats.get(k) is not None},'robustness':{k:rob.get(k) for k in ['positive','negative','zero','mean','median'] if rob.get(k) is not None},'status':status,'notes':notes}
def fmt(x,sign=False):
    if x is None: return ''
    return f'{float(x):+.4f}' if sign else f'{float(x):.4f}'
def md_std(rows):
    out=['| Method | Protocol tier | AJ | AJ Δ | OA | OA Δ | δavg | δavg Δ | δ4px | δ4px Δ | Status | Notes |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        m=r['metric']; d=r['delta_vs_native']
        out.append('| '+' | '.join([r['name'],r['protocol'],fmt(m.get('AJ')),fmt(d.get('AJ'),1),fmt(m.get('OA')),fmt(d.get('OA'),1),fmt(m.get('delta_avg')),fmt(d.get('delta_avg'),1),fmt(m.get('delta_4px')),fmt(d.get('delta_4px'),1),r['status'],r['notes']]).replace('|','/')+' |')
    return '\n'.join(out)
def md_re(rows):
    out=['| Method | Protocol tier | AJ_RD | AJ_RD Δ | AJ_RD_256 | AJ_RD_256 Δ | Repair16 | Damage16 | Robustness | Status | Notes |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for r in rows:
        m=r['metric']; d=r['delta_vs_native']; s=r['repair_damage']; rb=r['robustness']
        rob=''
        if 'positive' in rb: rob=f"{rb.get('positive')}/{rb.get('negative')}/{rb.get('zero')} pos/neg/zero"
        out.append('| '+' | '.join([r['name'],r['protocol'],fmt(m.get('AJ_RD')),fmt(d.get('AJ_RD'),1),fmt(m.get('AJ_RD_256')),fmt(d.get('AJ_RD_256'),1),fmt(s.get('repair16_rate')),fmt(s.get('damage16_rate')),rob,r['status'],r['notes']]).replace('|','/')+' |')
    return '\n'.join(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c01=load('c01'); c13=load('c13'); tap=load('tap'); ver=load('ver'); state=load('state'); v7=load('v7')
    native=c01['native_metric']; zero={k:0.0 for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
    default=find_row(c01,'dist_nc_le64_w8_candidate_visible','meta')
    high=find_row(c01,'dist_nc_le64_w16_candidate_visible','meta')
    all8=find_row(c01,'all_w8_candidate_visible','meta')
    verrow=next(r for r in ver['rows'] if r['variant']=='utility_et_ge_-0.3431')
    v7r=v7row(v7,'damage_ceiling0.7')
    tapw8=tap['by_candidate']['tapnextpp_first_v5_conf']['w8']
    tapw16=tap['by_candidate']['tapnextpp_first_v5_conf']['w16']
    c13rows=c13['rows']
    oldon=find_name(c13rows,'CVRRM + old CoTracker online w8')
    oldoff=find_name(c13rows,'CVRRM + old CoTracker offline w8')
    tapoff=find_name(c13rows,'CVRRM + TAPNext++ offline w8')
    state08=next(r for r in state['state_rows'] if abs(r['prob']-0.8)<1e-9)
    state09=next(r for r in state['state_rows'] if abs(r['prob']-0.9)<1e-9)
    std=[]
    std.append(std_entry('CoTracker3 online native','current DAVIS true-streaming reproduction',native,zero,'baseline','not claimed as original-paper table protocol yet'))
    std.append(std_entry('TrackOn2 standalone','strict aligned candidate cache',c01['trackon2_metric'],c01['trackon2_delta_vs_native'],'candidate baseline','two-source comparison baseline'))
    std.append(std_entry('V7-B2 visibility-only','strict output-level ablation',v7r['metric'],v7r['delta_vs_native'],'weak baseline','visibility-only helps little'))
    std.append(std_entry('CVRRM + TrackOn2 w8','strict output-level two-source',default['metric'],default['delta_vs_native'],'MAIN DEFAULT','best stability/quality tradeoff'))
    std.append(std_entry('CVRRM + TrackOn2 w16','strict output-level two-source',high['metric'],high['delta_vs_native'],'high-gain ablation','slightly higher re-entry, less robust'))
    std.append(std_entry('CVRRM + TAPNext++ online w8','metadata-bridge diagnostic',tapw8['metric'],tapw8['delta'],'supporting diagnostic','not strict raw-cache alignment'))
    std.append(std_entry('OOF verifier ablation','OOF diagnostic',verrow['metric'],verrow['delta_vs_native'],'diagnostic only','not default'))
    std.append(std_entry('State writeback p=0.90','state hook follow-up',state09['metric'],state09['delta_vs_native'],'future work only','does not beat output-level w16 and has lower AJ/OA'))
    re=[]
    re.append(re_entry('CoTracker3 online native','current DAVIS true-streaming reproduction',native,zero,'baseline','reference'))
    re.append(re_entry('TrackOn2 standalone','strict aligned candidate cache',c01['trackon2_metric'],c01['trackon2_delta_vs_native'],'candidate baseline','standalone improves standard AJ strongly but re-entry less than CVRRM'))
    re.append(re_entry('V7-B2 visibility-only','strict output-level ablation',v7r['metric'],v7r['delta_vs_native'],'weak baseline','visibility-only is not enough'))
    re.append(re_entry('CVRRM + TrackOn2 w8','strict output-level two-source',default['metric'],default['delta_vs_native'],'MAIN DEFAULT','main re-entry recovery result',default.get('stats'),default.get('per_video_summary')))
    re.append(re_entry('CVRRM + TrackOn2 w16','strict output-level two-source',high['metric'],high['delta_vs_native'],'high-gain ablation','highest output-level AJ_RD_256',high.get('stats'),high.get('per_video_summary')))
    re.append(re_entry('CVRRM + TAPNext++ online w8','metadata-bridge diagnostic',tapw8['metric'],tapw8['delta'],'supporting diagnostic','cross-candidate evidence but not strict raw-cache alignment',tapw8.get('stats'),tapw8.get('pv')))
    re.append(re_entry('CVRRM + old CoTracker online w8','strict aligned candidate',oldon['metric'],oldon['delta_vs_native'],'weak candidate evidence','small gain only',oldon.get('stats'),oldon.get('robustness')))
    re.append(re_entry('CVRRM + old CoTracker offline w8','strict aligned candidate',oldoff['metric'],oldoff['delta_vs_native'],'weak candidate evidence','small gain only',oldoff.get('stats'),oldoff.get('robustness')))
    re.append(re_entry('CVRRM + TAPNext++ offline w8','metadata-bridge diagnostic',tapoff['metric'],tapoff['delta_vs_native'],'negative evidence','harmful candidate',tapoff.get('stats'),tapoff.get('robustness')))
    re.append(re_entry('OOF verifier ablation','OOF diagnostic',verrow['metric'],verrow['delta_vs_native'],'diagnostic only','only +0.0001 over default',verrow.get('apply_stats'),verrow.get('per_video_summary')))
    re.append(re_entry('State writeback p=0.90','state hook follow-up',state09['metric'],state09['delta_vs_native'],'future work only','does not exceed output-level w16',None,state09.get('state_minus_output_summary')))
    report={'script':'scripts/package_v8c30_davis_dual_metric_audit.py','sources':{k:str(v) for k,v in P.items()},'protocol_position':{'standard_table':'current DAVIS true-streaming reproduction, not yet original paper one-query/support-point protocol','reentry_table':'same-cache, all-baseline re-entry-focused evaluation; valid as failure-mode analysis, not original main-table replacement'},'standard_tapvid_like_table':std,'reentry_recovery_table':re,'decision':{'main_method':'CVRRM + TrackOn2 w8 output-level','safe_claim':'DAVIS true-streaming reproduction plus re-entry-focused analysis','unsafe_claim':'original CoTracker3 paper main-table improvement','next':'run official-protocol comparability gap audit only if paper requires direct Table-1-style comparison'}}
    out=OUT/'v8c30_davis_dual_metric_audit_report.json'; out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    md=[]
    md+=['# CoTracker3 Online V8-C3.0 DAVIS Dual-Metric Protocol Audit','','Date: 2026-07-07','']
    md+=['## 1. Why this audit exists','','This audit avoids two mistakes: rejecting re-entry metrics just because they are not the original main-table metrics, and overclaiming AJ_RD_256 as an original-paper benchmark improvement.','']
    md+=['## 2. Standard TAP-Vid-like table on current DAVIS reproduction','',md_std(std),'']
    md+=['## 3. Re-entry recovery table on the same DAVIS records','',md_re(re),'']
    md+=['## 4. Protocol position','','```text','AJ_RD / AJ_RD_256 are valid as re-entry-focused failure-mode metrics because all baselines are evaluated with the same definition.','They are not replacements for the original CoTracker3 main-table AJ / δavg / OA.','Current DAVIS cache is a true-streaming first-input reproduction, not yet audited as the original paper one-query-at-a-time + support-points protocol.','CVRRM + TrackOn2 is a two-source recovery system, not a single-model CoTracker3 result.','```','']
    md+=['## 5. Decision','','```text','Use two tables:','1. Standard TAP-Vid-like table for AJ / OA / δavg / δ4px.','2. Re-entry recovery table for AJ_RD / AJ_RD_256 / repair-damage / robustness.','Main method remains CVRRM + TrackOn2 w8 output-level.','Do not claim original CoTracker3 Table-1 improvement until official-protocol comparability is separately audited.','```','']
    md+=['## 6. Next step','','```text','If the goal is a paper claim against original CoTracker3 tables: run official-protocol comparability gap audit.','If the goal is a re-entry paper: proceed to final writing with dual tables and clear protocol caveats.','Recommended: do a focused official-protocol gap audit on DAVIS before expanding to Kinetics/RGB-Stacking.','```','']
    DOC.write_text('\n'.join(md))
    with (ROOT/'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## CoTracker3 online V8-C3.0 DAVIS dual-metric protocol audit (2026-07-07)\n\n')
        f.write('Artifact:\n\n```text\n'+str(DOC)+'\n```\n\n')
        f.write('Report:\n\n```text\n'+str(out)+'\n```\n\n')
        f.write('Decision:\n\n```text\nUse dual tables: standard TAP-Vid-like metrics and re-entry recovery metrics. AJ_RD_256 is valid as same-baseline failure-mode analysis, but not a substitute for original CoTracker3 main-table protocol. Main method remains CVRRM + TrackOn2 w8. Next: official-protocol gap audit only if making direct original-paper Table-1-style claims.\n```\n')
    print(json.dumps({'out':str(out),'doc':str(DOC),'standard_rows':len(std),'reentry_rows':len(re),'main_delta_AJ_RD_256':default['delta_vs_native']['AJ_RD_256'],'main_delta_AJ':default['delta_vs_native']['AJ']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
