#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/gemini/code/FSPT')
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c4_final_paper_story'
DOC=ROOT/'docs/cotracker3_online_v8c4_final_paper_story_2026-07-07.md'
P={
 'dual':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c30_davis_dual_metric_audit/v8c30_davis_dual_metric_audit_report.json',
 'parity':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_official_evaluator_parity/v8c31_official_evaluator_parity_report.json',
 'source':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c13_candidate_source_table/v8c13_candidate_source_table.json',
 'state':ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c21_full_hook_eval/v8c21_full_report.json',
}

def load(k): return json.loads(P[k].read_text())
def find(rows,name):
    for r in rows:
        if r.get('name')==name: return r
    raise KeyError(name)
def fmt(v,sign=False):
    if v is None: return ''
    return f'{float(v):+.4f}' if sign else f'{float(v):.4f}'
def table_std(rows):
    out=['| Method | AJ | ΔAJ | OA | ΔOA | δavg | Δδavg | δ4px | Δδ4px | Status |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        m=r['metric']; d=r['delta_vs_native']
        out.append('| '+' | '.join([r['name'],fmt(m.get('AJ')),fmt(d.get('AJ'),1),fmt(m.get('OA')),fmt(d.get('OA'),1),fmt(m.get('delta_avg')),fmt(d.get('delta_avg'),1),fmt(m.get('delta_4px')),fmt(d.get('delta_4px'),1),r.get('status','')])+' |')
    return '\n'.join(out)
def table_re(rows):
    out=['| Method | AJ_RD | ΔAJ_RD | AJ_RD_256 | ΔAJ_RD_256 | Repair16 | Damage16 | Robustness | Status |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        m=r['metric']; d=r['delta_vs_native']; s=r.get('repair_damage',{}); rb=r.get('robustness',{})
        rob=''
        if rb.get('positive') is not None: rob=f"{rb.get('positive')}/{rb.get('negative')}/{rb.get('zero')} pos/neg/zero"
        out.append('| '+' | '.join([r['name'],fmt(m.get('AJ_RD')),fmt(d.get('AJ_RD'),1),fmt(m.get('AJ_RD_256')),fmt(d.get('AJ_RD_256'),1),fmt(s.get('repair16_rate')),fmt(s.get('damage16_rate')),rob,r.get('status','')])+' |')
    return '\n'.join(out)
def table_source(rows):
    out=['| Row | Candidate | Protocol | ΔAJ_RD_256 | Status | Notes |', '| --- | --- | --- | ---: | --- | --- |']
    for r in rows:
        d=r.get('delta_vs_native',{})
        out.append('| '+' | '.join([r.get('name',''),r.get('candidate_source',''),r.get('protocol',''),fmt(d.get('AJ_RD_256'),1),r.get('status',''),r.get('notes','')]).replace('|','/')+' |')
    return '\n'.join(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dual=load('dual'); parity=load('parity'); source=load('source'); state=load('state')
    std=dual['standard_tapvid_like_table']; re=dual['reentry_recovery_table']; src=source['rows']
    native=find(std,'CoTracker3 online native')
    main_std=find(std,'CVRRM + TrackOn2 w8')
    main_re=find(re,'CVRRM + TrackOn2 w8')
    high_re=find(re,'CVRRM + TrackOn2 w16')
    tap_re=find(re,'CVRRM + TAPNext++ online w8')
    verifier=find(re,'OOF verifier ablation')
    st09=find(re,'State writeback p=0.90')
    story={
      'script':'scripts/package_v8c4_final_paper_story.py',
      'sources':{k:str(v) for k,v in P.items()},
      'method_name':'CVRRM: Causal Candidate-Visible Re-entry Recovery Mode',
      'protocol_label':parity['judgement']['label'],
      'safe_claim':'On TAP-Vid-DAVIS-first metric-compatible reproduction, CVRRM improves standard TAP-Vid-like metrics and re-entry-specific metrics.',
      'unsafe_claims':['Do not claim full original CoTracker3 multi-dataset Table-1 improvement yet.','Do not claim single-model CoTracker3 improvement; this is a two-source recovery system.','Do not treat AJ_RD_256 as a replacement for original TAP-Vid main metrics.'],
      'main_result':{'standard_delta':main_std['delta_vs_native'],'reentry_delta':main_re['delta_vs_native'],'standard_after':main_std['metric'],'reentry_after':main_re['metric']},
      'high_gain_result':{'reentry_delta':high_re['delta_vs_native'],'reentry_after':high_re['metric']},
      'supporting_cross_candidate':{'name':'CVRRM + TAPNext++ online w8','delta':tap_re['delta_vs_native'],'protocol':'metadata-bridge diagnostic'},
      'negative_results':{'learned_verifier':'Only +0.0001 AJ_RD_256 over default; diagnostic only.','state_writeback':'p=0.90 reaches +0.0285 AJ_RD_256 but does not beat output-level w16 and lowers AJ/OA; future work only.'},
      'standard_table':std,
      'reentry_table':re,
      'candidate_source_table':src,
      'next_research_options':['Kinetics/RGB-Stacking extension for broader benchmark generalization.','Strictly aligned stronger candidate providers such as TAPNext++/LocoTrack/TAPIR.','Paper writing with dual-table framing before new method changes.']
    }
    out=OUT/'v8c4_final_paper_story.json'; out.write_text(json.dumps(story,indent=2,ensure_ascii=False))
    md=[]
    md += ['# V8-C4 Final Paper-Style Story: CVRRM for Online TAP Re-entry Recovery','','Date: 2026-07-07','']
    md += ['## 1. One-sentence claim','','```text','CVRRM is a strict-causal, output-level, finite-horizon recovery mode for online TAP re-entry failures.','On TAP-Vid-DAVIS-first metric-compatible reproduction, it improves both standard TAP-Vid-like metrics and re-entry-focused metrics.','```','']
    md += ['## 2. Method definition','','```text','CVRRM = Causal Candidate-Visible Re-entry Recovery Mode','','Input: native CoTracker3 online outputs and an external candidate provider.','Trigger: native low-score / invisible event rebuilt from native history.','Sanity filter: native-candidate distance <= 64 px at event frame.','Recovery mode: W=8 by default.','Frame-level confirmation: replace a frame only if the candidate is visible at that frame.','Output: candidate coordinate + visible flag for accepted frames; native output otherwise.','```','']
    md += ['## 3. Protocol status','','```text',f"Protocol label: {parity['judgement']['label']}",f"Metric formula parity pass: {parity['judgement']['metric_formula_parity_pass']}",f"Query-first pass: {parity['judgement']['query_first_pass']}",f"Local evaluator protocol-close pass: {parity['judgement']['local_evaluator_protocol_close_pass']}",f"Max metric diff: {parity['judgement']['max_metric_diff']}",'```','','Safe claim: '+story['safe_claim'],'','Unsafe claims:']
    for x in story['unsafe_claims']: md.append('- '+x)
    md += ['','## 4. Standard TAP-Vid-like metrics on DAVIS','','These metrics answer whether overall tracking quality is preserved or improved.','',table_std(std),'']
    md += ['## 5. Re-entry recovery metrics on the same records','','These metrics answer whether the method specifically fixes re-entry failures.','',table_re(re),'']
    md += ['## 6. Candidate-source evidence','','This table shows CVRRM is candidate-aware: strong online candidate providers help, weak/offline candidates do not.','',table_source(src),'']
    md += ['## 7. Negative and neutral follow-ups','','### Learned verifier','','```text',f"OOF verifier AJ_RD_256 delta: {fmt(verifier['delta_vs_native']['AJ_RD_256'],1)}",f"Default AJ_RD_256 delta: {fmt(main_re['delta_vs_native']['AJ_RD_256'],1)}",'Decision: diagnostic only, not default.','```','','### State writeback','','```text',f"State writeback p=0.90 AJ_RD_256 delta: {fmt(st09['delta_vs_native']['AJ_RD_256'],1)}",f"Output-level high-gain w16 AJ_RD_256 delta: {fmt(high_re['delta_vs_native']['AJ_RD_256'],1)}",'Decision: future work only; it does not beat output-level high-gain and lowers AJ/OA.','```','']
    md += ['## 8. Main result summary','','```text',f"Base AJ: {fmt(native['metric']['AJ'])}",f"CVRRM w8 AJ: {fmt(main_std['metric']['AJ'])}",f"AJ delta: {fmt(main_std['delta_vs_native']['AJ'],1)}",'',f"Base OA: {fmt(native['metric']['OA'])}",f"CVRRM w8 OA: {fmt(main_std['metric']['OA'])}",f"OA delta: {fmt(main_std['delta_vs_native']['OA'],1)}",'',f"CVRRM w8 AJ_RD_256 delta: {fmt(main_re['delta_vs_native']['AJ_RD_256'],1)}",f"CVRRM w16 AJ_RD_256 delta: {fmt(high_re['delta_vs_native']['AJ_RD_256'],1)}",'```','']
    md += ['## 9. Limitations','','```text','1. This is a two-source recovery system: CoTracker3 online native + external candidate provider + CVRRM.','2. Current strongest claim is DAVIS-first metric-compatible reproduction, not full original multi-dataset benchmark parity.','3. AJ_RD / AJ_RD_256 are re-entry-focused supplementary metrics, not replacements for AJ/OA/δavg.','4. TrackOn2 bridge is the default candidate provider; TAPNext++ online-style is supporting diagnostic evidence.','```','']
    md += ['## 10. Recommended next steps','','```text','First: use this V8-C4 package as the paper/story baseline.','Then choose one expansion path:','A. Benchmark expansion: Kinetics subset -> RGB-Stacking smoke -> full Kinetics/RGB-Stacking.','B. Candidate expansion: strictly aligned TAPNext++ / LocoTrack / TAPIR candidate caches.','Avoid further learned-verifier/state-writeback work unless a new full30 signal clearly exceeds output-level CVRRM.','```','']
    DOC.write_text('\n'.join(md))
    with (ROOT/'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## CoTracker3 online V8-C4 final paper-style story package (2026-07-07)\n\n')
        f.write('Artifact:\n\n```text\n'+str(DOC)+'\n```\n\n')
        f.write('Report:\n\n```text\n'+str(out)+'\n```\n\n')
        f.write('Decision:\n\n```text\nCVRRM final story is ready for paper-style writing. Main method: output-level CVRRM + TrackOn2 bridge, dist<=64, W=8. It improves standard DAVIS-first metric-compatible metrics and re-entry metrics. Learned verifier and state writeback remain diagnostic/future work. Next choose benchmark expansion or candidate-provider expansion.\n```\n')
    print(json.dumps({'out':str(out),'doc':str(DOC),'protocol_label':story['protocol_label'],'main_standard_delta':story['main_result']['standard_delta'],'main_reentry_delta':story['main_result']['reentry_delta'],'next':story['next_research_options']},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
