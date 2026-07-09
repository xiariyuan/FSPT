#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path('/gemini/code/FSPT')
OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c13_candidate_source_table'
DOC=ROOT/'docs/cotracker3_online_v8c13_candidate_source_table_2026-07-07.md'
P_FINAL=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c05_final_recovery_method/v8c05_final_recovery_method_table.json'
P_CROSS=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c1_cross_candidate_generalization/v8c1_cross_candidate_generalization_report.json'
P_TAP=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c12_tapnextpp_bridge_eval/v8c12_tapnextpp_bridge_eval_report.json'

def load(p): return json.loads(Path(p).read_text())
def frow(rows,name):
    for r in rows:
        if r.get('name')==name: return r
    raise KeyError(name)
def crow(cross,cand,key): return cross['by_candidate'][cand][key]
def trow(tap,cand,key): return tap['by_candidate'][cand][key]
def rnd(v): return round(float(v),4) if isinstance(v,(int,float)) and not isinstance(v,bool) else v
def metric(row): return row.get('metric',{})
def delta(row):
    d=row.get('delta') or row.get('delta_vs_native') or {}
    return {k:d.get(k) for k in ['AJ','OA','AJ_RD','AJ_RD_256']}
def rob(row):
    r=row.get('robustness') or row.get('pv') or {}
    return {k:r.get(k) for k in ['positive','negative','zero','mean','success_folds'] if r.get(k) is not None}
def add(name,source,protocol,variant,row,status,notes):
    return {'name':name,'candidate_source':source,'protocol':protocol,'variant':variant,'status':status,'metric':{k:metric(row).get(k) for k in ['AJ','OA','AJ_RD','AJ_RD_256']},'delta_vs_native':delta(row),'robustness':rob(row),'notes':notes}
def fmt(v,sign=False):
    if v is None: return ''
    return f'{float(v):+0.4f}' if sign else f'{float(v):0.4f}'
def md_table(rows):
    out=['| Row | Candidate | Protocol | AJ Δ | OA Δ | AJ_RD Δ | AJ_RD_256 Δ | Robustness | Status | Notes |','| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for r in rows:
        d=r['delta_vs_native']; rb=r.get('robustness',{})
        robtxt=''
        if 'positive' in rb: robtxt=f"{rb.get('positive')}/{rb.get('negative')}/{rb.get('zero')} pos/neg/zero"
        elif 'success_folds' in rb: robtxt=f"{rb.get('success_folds')}/5 folds"
        vals=[r['name'],r['candidate_source'],r['protocol'],fmt(d.get('AJ'),1),fmt(d.get('OA'),1),fmt(d.get('AJ_RD'),1),fmt(d.get('AJ_RD_256'),1),robtxt,r['status'],r['notes']]
        out.append('| '+' | '.join(str(x).replace('|','/') for x in vals)+' |')
    return '\n'.join(out)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    final=load(P_FINAL); cross=load(P_CROSS); tap=load(P_TAP)
    fr=final['main_table']
    rows=[]
    rows.append(add('Native CoTracker3 online','none','strict native','native',frow(fr,'CoTracker3 online native'),'baseline','reference'))
    rows.append(add('V7-B2 visibility-only','none','strict output-level','damage_ceiling0.7',frow(fr,'V7-B2 visibility-only'),'baseline','visibility-only gain is small'))
    rows.append(add('TrackOn2 standalone','TrackOn2 bridge','strict aligned','standalone',frow(fr,'TrackOn2 standalone'),'candidate baseline','strong global candidate but less re-entry-focused than CVRRM'))
    rows.append(add('CVRRM + TrackOn2 w8 default','TrackOn2 bridge','strict aligned','dist<=64,w8',frow(fr,'CVRRM default: dist<=64, w8'),'MAIN DEFAULT','recommended main method'))
    rows.append(add('CVRRM + TrackOn2 w16 high-gain','TrackOn2 bridge','strict aligned','dist<=64,w16',frow(fr,'CVRRM high-gain: dist<=64, w16'),'high-gain ablation','slightly higher AJ_RD_256, less robust than w8'))
    rows.append(add('CVRRM + TAPNext++ online w8','TAPNext++ online-style','metadata-bridge diagnostic','dist<=64,w8',trow(tap,'tapnextpp_first_v5_conf','w8'),'supporting ablation','crosses +0.020 AJ_RD_256 but AJ is negative; diagnostic protocol'))
    rows.append(add('CVRRM + TAPNext++ online w16','TAPNext++ online-style','metadata-bridge diagnostic','dist<=64,w16',trow(tap,'tapnextpp_first_v5_conf','w16'),'supporting ablation','cross-candidate evidence; not default'))
    rows.append(add('CVRRM + old CoTracker online w8','old CoTracker3 online bridge','strict aligned','dist<=64,w8',crow(cross,'old_cotracker3_online_bridge','default'),'weak candidate evidence','small gain only'))
    rows.append(add('CVRRM + old CoTracker offline w8','old CoTracker3 offline bridge','strict aligned','dist<=64,w8',crow(cross,'old_cotracker3_offline_bridge','default'),'weak candidate evidence','small gain only'))
    rows.append(add('CVRRM + TAPNext++ offline w8','TAPNext++ offline','metadata-bridge diagnostic','dist<=64,w8',trow(tap,'tapnextpp_offline_w8','w8'),'negative evidence','harmful; high false-visible/damage'))
    rows.append(add('OOF verifier ablation','TrackOn2 bridge','OOF learned diagnostic','utility_et_ge_-0.3431',frow(fr,'OOF verifier ablation'),'diagnostic only','not default: only +0.0001 AJ_RD_256 over CVRRM default'))
    package={'script':'scripts/package_v8c13_candidate_source_table.py','sources':{'final':str(P_FINAL),'cross_candidate':str(P_CROSS),'tapnextpp_bridge':str(P_TAP)},'decision':{'default':'CVRRM + TrackOn2 bridge w8','high_gain':'CVRRM + TrackOn2 bridge w16','supporting_cross_candidate':'CVRRM + TAPNext++ online-style metadata bridge','not_default':['OOF verifier','old CoTracker candidates','TAPNext++ offline'],'next':'decide between final paper packaging and state-level repair; if state repair, use TrackOn2 bridge default only first'},'rows':rows}
    out=OUT/'v8c13_candidate_source_table.json'; out.write_text(json.dumps(package,indent=2,ensure_ascii=False))
    md=[]
    md+=['# CoTracker3 Online V8-C1.3 Candidate-Source Final Table','','Date: 2026-07-07','']
    md+=['## 1. Purpose','','Consolidate strict CVRRM results, cross-candidate results, TAPNext++ metadata-bridge diagnostics, and learned-verifier ablation into one candidate-source table.','']
    md+=['## 2. Candidate-source table','',md_table(rows),'']
    md+=['## 3. Final interpretation','','```text','CVRRM is not arbitrary candidate fusion. It needs a strong online re-entry candidate provider.','TrackOn2 bridge is the default provider.','TAPNext++ online-style metadata bridge is supporting cross-candidate evidence.','Old CoTracker candidates are too weak. Offline TAPNext++ is harmful.','OOF verifier remains diagnostic only.','```','']
    md+=['## 4. Recommended method statement','','```text','Main method: CVRRM + TrackOn2 bridge, dist<=64, W=8.','High-gain ablation: CVRRM + TrackOn2 bridge, dist<=64, W=16.','Cross-candidate diagnostic: CVRRM + TAPNext++ online-style metadata bridge.','```','']
    md+=['## 5. Next step','','```text','Option A: freeze output-level method and write final paper-style method section/results.','Option B: start V8-C2 state-level repair, but only with TrackOn2 bridge default first.','Recommended: do a short state-level repair feasibility precheck before committing.','```','']
    DOC.write_text('\n'.join(md))
    with (ROOT/'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## CoTracker3 online V8-C1.3 candidate-source final table (2026-07-07)\n\n')
        f.write('Artifact:\n\n```text\n'+str(DOC)+'\n```\n\n')
        f.write('Report:\n\n```text\n'+str(out)+'\n```\n\n')
        f.write('Decision:\n\n```text\nDefault remains CVRRM + TrackOn2 bridge w8. TAPNext++ online-style metadata bridge is added as supporting cross-candidate evidence. Old CoTracker candidates are weak; TAPNext++ offline is harmful; OOF verifier remains diagnostic only. Next: final paper packaging or V8-C2 state-level repair feasibility precheck.\n```\n')
    print(json.dumps({'out':str(out),'doc':str(DOC),'rows':[{ 'name':r['name'],'AJ_RD_256_delta':rnd(r['delta_vs_native'].get('AJ_RD_256')),'status':r['status']} for r in rows]},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
