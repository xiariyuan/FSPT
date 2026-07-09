#!/usr/bin/env python3
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SRC = Path('outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_supplement_summary.json')
DOC = Path('docs/reentry_tap_trackon2_first_input_supplement_2026-07-01.md')

METHODS = [
    'cotracker3_offline_first',
    'trackon2_first',
    'b2_w16_trackon_base',
    'b2_w16_p2_trackon_base',
    'b2_w16_cotracker_base',
    'b2_w16_p2_cotracker_base',
]
METRICS = ['AJ_RD_256','AJ_256','OA_256','delta_avg_256']


def sign_p(pos:int, neg:int)->float:
    n=pos+neg
    if n<=0: return 1.0
    k=min(pos,neg)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/(2**n))


def mean_nonnull(vals):
    xs=[float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def sum_queries(rows, method):
    return int(sum(int(r[method].get('n_reentry_queries') or 0) for r in rows))


def query_weighted_ajrd(rows, method):
    num=den=0.0
    for r in rows:
        n=int(r[method].get('n_reentry_queries') or 0)
        v=r[method].get('AJ_RD_256')
        if n>0 and v is not None and np.isfinite(float(v)):
            num += float(v)*n
            den += n
    return num/den if den>0 else None


def paired_stats(rows, a, b, metric):
    vals=[]; vids=[]
    for r in rows:
        va=r[a].get(metric); vb=r[b].get(metric)
        if va is not None and vb is not None and np.isfinite(float(va)) and np.isfinite(float(vb)):
            vals.append(float(va)-float(vb)); vids.append(r['video_id'])
    arr=np.asarray(vals,float)
    if arr.size==0: return {'n':0}
    rng=np.random.default_rng(20260701+len(metric)+len(a))
    boot=[float(np.mean(arr[rng.integers(0,arr.size,size=arr.size)])) for _ in range(10000)]
    ci=np.percentile(boot,[2.5,97.5])
    pos=int(np.sum(arr>0)); neg=int(np.sum(arr<0)); zero=int(np.sum(arr==0))
    return {
        'n': int(arr.size),
        'mean_delta': round(float(np.mean(arr)),6),
        'median_delta': round(float(np.median(arr)),6),
        'ci95_bootstrap': [round(float(ci[0]),6), round(float(ci[1]),6)],
        'positive_videos': pos,
        'negative_videos': neg,
        'zero_videos': zero,
        'sign_test_p_two_sided': round(sign_p(pos,neg),8),
        'min_delta': round(float(np.min(arr)),6),
        'max_delta': round(float(np.max(arr)),6),
    }


def md_table(headers, rows):
    lines=['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for row in rows: lines.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(lines)

def fmt(x,n=4):
    if x is None: return '—'
    return f'{float(x):.{n}f}'


def main():
    rows=[json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    agg={}
    for m in METHODS:
        agg[m]={}
        for metric in METRICS:
            if metric=='AJ_RD_256':
                agg[m][metric+'_query_weighted']=query_weighted_ajrd(rows,m)
                agg[m][metric+'_video_mean']=mean_nonnull([r[m].get(metric) for r in rows])
            else:
                agg[m][metric+'_video_mean']=mean_nonnull([r[m].get(metric) for r in rows])
        agg[m]['n_reentry_queries']=sum_queries(rows,m)
    comparisons={
        'b2_w16_p2_trackon_base_vs_trackon2_first': {metric: paired_stats(rows,'b2_w16_p2_trackon_base','trackon2_first',metric) for metric in METRICS},
        'b2_w16_trackon_base_vs_trackon2_first': {metric: paired_stats(rows,'b2_w16_trackon_base','trackon2_first',metric) for metric in METRICS},
        'trackon2_first_vs_cotracker3_offline_first': {metric: paired_stats(rows,'trackon2_first','cotracker3_offline_first',metric) for metric in METRICS},
        'b2_w16_p2_cotracker_base_vs_cotracker3_offline_first': {metric: paired_stats(rows,'b2_w16_p2_cotracker_base','cotracker3_offline_first',metric) for metric in METRICS},
    }
    summary={'source':str(SRC),'n_videos':len(rows),'methods':agg,'comparisons':comparisons}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(summary,indent=2,ensure_ascii=False))

    doc=[]
    doc.append('# TrackOn2 First-input Plug-in Supplement — 2026-07-01')
    doc.append('')
    doc.append('This supplement uses the parity-valid DAVIS first-query/input-resolution bridge. It is **not** mixed into the main ReEntry-TAP strided-original tables.')
    doc.append('')
    table=[]
    for m in ['cotracker3_offline_first','trackon2_first','b2_w16_trackon_base','b2_w16_p2_trackon_base','b2_w16_cotracker_base','b2_w16_p2_cotracker_base']:
        a=agg[m]
        table.append([m, fmt(a.get('AJ_RD_256_query_weighted')), fmt(a.get('AJ_RD_256_video_mean'),6), fmt(a.get('AJ_256_video_mean')), fmt(a.get('OA_256_video_mean')), fmt(a.get('delta_avg_256_video_mean')), a.get('n_reentry_queries')])
    doc.append('## Aggregate results')
    doc.append('')
    doc.append(md_table(['Method','query-weighted AJ_RD_256','video-mean AJ_RD_256','video-mean AJ_256','video-mean OA_256','video-mean delta_avg_256','re-entry queries'], table))
    doc.append('')
    doc.append('## Key paired comparisons')
    doc.append('')
    for cname in comparisons:
        doc.append(f'### {cname}')
        rows2=[]
        for metric in METRICS:
            s=comparisons[cname][metric]
            rows2.append([metric, fmt(s.get('mean_delta'),6), f"[{fmt(s.get('ci95_bootstrap',[None,None])[0],6)}, {fmt(s.get('ci95_bootstrap',[None,None])[1],6)}]", f"{s.get('positive_videos')}/{s.get('n')}", fmt(s.get('sign_test_p_two_sided'),8)])
        doc.append(md_table(['Metric','mean Δ','95% CI','positive videos','sign-test p'], rows2))
        doc.append('')
    doc.append('## Safe paper use')
    doc.append('')
    doc.append('```text')
    doc.append('Under a parity-valid first-query/input-resolution DAVIS protocol, B2-W16-P2 gives a small positive aggregate gain on top of a reproduced TrackOn2 baseline. This supports plug-in generality, but should be reported as supplementary evidence rather than a main ReEntry-TAP stress baseline.')
    doc.append('```')
    DOC.write_text('\n'.join(doc))
    print(json.dumps({'out_json':str(OUT),'doc':str(DOC),'trackon2_first':agg['trackon2_first'],'b2_p2_trackon_base':agg['b2_w16_p2_trackon_base'],'comparison':comparisons['b2_w16_p2_trackon_base_vs_trackon2_first']},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
