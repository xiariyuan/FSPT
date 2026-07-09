#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

DOC = Path('docs/reentry_tap_method_paper_tables_2026-07-01.md')
OUT = Path('docs/reentry_tap_method_paper_tables_plus_guard_external_2026-07-01.md')
ROOT = Path('outputs/paper_discovery_2026-06-27')


def load(p):
    with open(p, 'r') as f:
        return json.load(f)

def fmt(x, n=4):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    return f'{float(x):.{n}f}'

def md_table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    ])

def main():
    base = DOC.read_text() if DOC.exists() else '# ReEntry-TAP Method-Paper Tables — base missing\n'
    guard = load(ROOT / 'reentry_guard_v2_sklearn/summary.json')
    guard_stats = load(ROOT / 'reentry_guard_v2_sklearn/statistics_summary.json')
    # Prepare ReEntry-Guard main table
    rows=[]
    for res in guard['results']:
        setting=res['setting']
        b2=next(m for m in res['methods'] if m['name']=='b2_w16_p2')
        best=next(m for m in res['methods'] if m['name']==res['best_guard'])
        gain=res['gains'][f"{best['name']}_vs_b2"]
        rows.append([
            setting, 'B2-W16-P2', fmt(b2['AJ_RD_256']), fmt(b2['AJ_256']), fmt(b2['OA_256']), '—', '—'
        ])
        rows.append([
            setting, best['name'], fmt(best['AJ_RD_256']), fmt(best['AJ_256']), fmt(best['OA_256']), fmt(gain['AJ_RD_256']), fmt(gain['AJ_256'])
        ])
    guard_table = md_table(['Setting','Method','AJ_RD_256','AJ_256','OA_256','ΔAJ_RD vs B2','ΔAJ vs B2'], rows)

    rows=[]
    for res in guard_stats['results']:
        setting=res['setting']
        for metric in ['AJ_RD_256','AJ_256']:
            s=res['comparisons']['guard_vs_b2'][metric]
            rows.append([
                setting, metric, fmt(s['mean_delta'],6), f"[{fmt(s['ci95_bootstrap'][0],6)}, {fmt(s['ci95_bootstrap'][1],6)}]", f"{s['positive_videos']}/{s['n']}", fmt(s['sign_test_p_two_sided'],8)
            ])
    stat_table = md_table(['Setting','Metric','Mean Δ vs B2','95% CI','Positive videos','Sign-test p'], rows)

    # External supplement table from known doc/results.
    external_rows = [
        ['TAPNext stress smoke, dev0 translate_L16', 'TAPNext local checkpoint', '0.0311', '10.7467', '28.5831', 'not parity-established; do not use as main baseline'],
        ['TrackOn2 first-input DAVIS supplement', 'TrackOn2 first-input', '0.5444', '67.0406', '93.0615', 'parity-valid external baseline'],
        ['TrackOn2 first-input DAVIS supplement', 'B2-W16-P2 TrackOn2-base', '0.5509', '67.1260', '93.2224', 'supplemental plug-in gain'],
    ]
    external_table = md_table(['Protocol','Method','AJ_RD_256','AJ_256','OA_256','Use'], external_rows)

    add = f'''

---

# Addendum: ReEntry-Guard and External Baseline Supplement

## Table 8. ReEntry-Guard v2 over B2-W16-P2

{guard_table}

Key message:

```text
ReEntry-Guard v2 improves beyond the fixed B2-W16-P2 rule in all three fresh20-49 evaluations while preserving or slightly improving AJ.
```

## Table 9. ReEntry-Guard v2 video-level robustness

{stat_table}

Key message:

```text
The ReEntry-Guard v2 gain over B2 is modest but consistently positive, especially on natural RGB fresh20-49 and frozen occluder L16.
```

## Table 10. External baseline feasibility / supplement

{external_table}

Recommended use:

```text
Use TrackOn2 first-input as supplementary external plug-in evidence.
Do not use TAPNext stress smoke as a strong external baseline.
TrackOn2 ReEntry-TAP stress remains pending mmcv-enabled environment.
```

## Updated main-paper table recommendation

Main paper:

```text
Table 1: RGB fresh20-49 natural main result, include offline / online / B2 / ReEntry-Guard.
Table 2: Natural ablation, show B2-W16-P2 is a stable operating point.
Table 3: Frozen fresh20-49 ReEntry-TAP stress, include B2 and ReEntry-Guard if space allows.
Table 4: Oracle upper bound and gap closure.
Figure 1: Selective Local Re-entry Override framework.
```

Appendix:

```text
Dev severity curves.
Stress-induced-only analysis.
Per-video robustness.
TrackOn2 first-input supplement.
TAPNext/TrackOn2 stress feasibility notes.
```
'''
    OUT.write_text(base + add)
    print(json.dumps({'out': str(OUT), 'size': OUT.stat().st_size}, indent=2), flush=True)

if __name__ == '__main__':
    main()
