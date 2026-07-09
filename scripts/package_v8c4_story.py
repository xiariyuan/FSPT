#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path('/gemini/code/FSPT')
BASE=ROOT/'outputs/paper_discovery_2026-07-05'
OUT=BASE/'v8c4_story'
DOC=ROOT/'docs/v8c4_final_paper_story.md'

def load(rel): return json.loads((BASE/rel).read_text())
def fmt(x,sign=False):
    if x is None: return ''
    return f'{float(x):+.4f}' if sign else f'{float(x):.4f}'
def safe(d,*ks):
    for k in ks:
        d=d[k]
    return d
def table_std(rows):
    out=['| Method | Protocol | AJ Δ | OA Δ | δavg Δ | δ4px Δ | Status |','|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        d=r['delta_vs_native']
        out.append('| '+' | '.join([r['name'],r['protocol'],fmt(d.get('AJ'),1),fmt(d.get('OA'),1),fmt(d.get('delta_avg'),1),fmt(d.get('delta_4px'),1),r['status']])+' |')
    return '\n'.join(out)
def table_re(rows):
    out=['| Method | Protocol | AJ_RD Δ | AJ_RD_256 Δ | Repair16 | Damage16 | Robustness | Status |','|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        d=r['delta_vs_native']; s=r.get('repair_damage',{}); rb=r.get('robustness',{})
        rob=''
        if 'positive' in rb: rob=f"{rb.get('positive')}/{rb.get('negative')}/{rb.get('zero')}"
        out.append('| '+' | '.join([r['name'],r['protocol'],fmt(d.get('AJ_RD'),1),fmt(d.get('AJ_RD_256'),1),fmt(s.get('repair16_rate')),fmt(s.get('damage16_rate')),rob,r['status']])+' |')
    return '\n'.join(out)
def table_cand(rows):
    out=['| Candidate / Method | Protocol | AJ_RD_256 Δ | Status | Notes |','|---|---|---:|---|---|']
    for r in rows:
        out.append('| '+' | '.join([r['name'],r['protocol'],fmt(r['delta_vs_native'].get('AJ_RD_256'),1),r['status'],r['notes']])+' |')
    return '\n'.join(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dual=load('cotracker3_online_v8c30_davis_dual_metric_audit/v8c30_davis_dual_metric_audit_report.json')
    parity=load('cotracker3_online_v8c31_official_evaluator_parity/v8c31_official_evaluator_parity_report.json')
    cand=load('cotracker3_online_v8c13_candidate_source_table/v8c13_candidate_source_table.json')
    std=dual['standard_tapvid_like_table']; re=dual['reentry_recovery_table']; cand_rows=cand['rows']
    main_re=next(r for r in re if r['name']=='CVRRM + TrackOn2 w8')
    main_std=next(r for r in std if r['name']=='CVRRM + TrackOn2 w8')
    high_re=next(r for r in re if r['name']=='CVRRM + TrackOn2 w16')
    story={
      'script':'scripts/package_v8c4_story.py',
      'title':'CVRRM: Causal Candidate-Visible Re-entry Recovery for Online TAP',
      'main_method':'CVRRM + TrackOn2 bridge, dist<=64, W=8, output-level',
      'protocol_label':parity['judgement']['label'],
      'parity_judgement':parity['judgement'],
      'main_standard_delta':main_std['delta_vs_native'],
      'main_reentry_delta':main_re['delta_vs_native'],
      'high_gain_reentry_delta':high_re['delta_vs_native'],
      'standard_table':std,
      'reentry_table':re,
      'candidate_table':cand_rows,
      'claims':{
        'safe':[ 'TAP-Vid-DAVIS-first metric-compatible reproduction', 'standard metrics and re-entry metrics both improve', 'AJ_RD_256 is supplementary re-entry failure-mode metric', 'CVRRM is a two-source output-level recovery system using TrackOn2 as default candidate provider' ],
        'unsafe':[ 'full original CoTracker3 Table-1 benchmark improvement', 'single-model CoTracker3 improvement', 'candidate-agnostic universal recovery' ]
      },
      'next':{'paper_path':'write final method/results section with dual tables','extension_path':'Kinetics/RGB-Stacking after DAVIS story is frozen','improvement_path':'stronger strictly aligned candidate providers'}
    }
    out=OUT/'v8c4_final_paper_story.json'; out.write_text(json.dumps(story,indent=2,ensure_ascii=False))
    md=[]
    md+=['# V8-C4 Final Paper-Style Story','','## 1. Working title','','**CVRRM: Causal Candidate-Visible Re-entry Recovery for Online TAP**','']
    md+=['## 2. Core claim','','```text','On a TAP-Vid-DAVIS-first metric-compatible reproduction, CVRRM improves standard TAP-Vid-like metrics and re-entry-focused metrics.','It is a strict-causal output-level recovery mode, not a single-model CoTracker3 replacement.','```','']
    md+=['## 3. Method in one paragraph','','CVRRM detects low-confidence or invisible native CoTracker3 online events, opens a finite recovery window, and only replaces a frame when a strong candidate provider is currently visible and passes a native-candidate distance sanity check. The default uses TrackOn2 bridge as the candidate provider, `dist<=64`, and `W=8`.','']
    md+=['## 4. Standard TAP-Vid-like metrics','','Protocol label: `'+parity['judgement']['label']+'`','',table_std(std),'']
    md+=['## 5. Re-entry recovery metrics','','AJ_RD and AJ_RD_256 are supplementary failure-mode metrics, computed for all baselines with the same definition.','',table_re(re),'']
    md+=['## 6. Candidate-source evidence','',table_cand(cand_rows),'']
    md+=['## 7. Negative / neutral results','','```text','Learned verifier: not promoted. Best OOF verifier improves AJ_RD_256 by only +0.0001 over default.','State writeback: not promoted. Full30 p=0.90 nearly matches output-level w16 but lowers AJ/OA and is mixed per video.','Weak candidates: old CoTracker candidates produce only small gains; offline TAPNext++ is harmful.','```','']
    md+=['## 8. Claim boundaries','','Safe claims:','','```text','TAP-Vid-DAVIS-first metric-compatible reproduction.','Two-source output-level re-entry recovery system.','AJ_RD_256 is a supplementary re-entry-focused metric.','```','','Unsafe claims:','','```text','Full original CoTracker3 Table-1 improvement across all datasets.','Single-model CoTracker3 improvement.','Universal candidate-agnostic recovery.','```','']
    md+=['## 9. Recommended next step','','```text','Write final method/results section from this package.','Then optionally expand to Kinetics/RGB-Stacking, or search stronger strictly aligned candidate providers.','```','']
    DOC.write_text('\n'.join(md))
    with (ROOT/'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## V8-C4 final paper-style story packaging\n\n')
        f.write('Artifact:\n\n```text\n'+str(DOC)+'\n```\n\n')
        f.write('Report:\n\n```text\n'+str(out)+'\n```\n\n')
        f.write('Decision:\n\n```text\nMain story is now frozen for DAVIS: CVRRM + TrackOn2 w8 output-level, dual tables, TAP-Vid-DAVIS-first metric-compatible reproduction, AJ_RD_256 as supplementary re-entry metric. Next: write final paper sections or expand to more datasets.\n```\n')
    print(json.dumps({'out':str(out),'doc':str(DOC),'main_standard_delta':story['main_standard_delta'],'main_reentry_delta':story['main_reentry_delta'],'protocol':story['protocol_label']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
