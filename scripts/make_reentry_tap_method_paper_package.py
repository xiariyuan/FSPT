#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path('outputs/paper_discovery_2026-06-27')
DOC = Path('docs/reentry_tap_method_paper_tables_2026-07-01.md')
FIGDIR = Path('docs/figures/reentry_tap_method')
FIGDIR.mkdir(parents=True, exist_ok=True)


def load(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def md_table(headers, rows):
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        out.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(out)


def fmt(x, n=4):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    return f'{float(x):.{n}f}'


def fig_bar(labels, values, title, ylabel, out):
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def fig_tradeoff(rows, title, out):
    plt.figure(figsize=(7, 5))
    for name, aj, ajrd in rows:
        plt.scatter([aj], [ajrd])
        plt.annotate(name, (aj, ajrd), textcoords='offset points', xytext=(5, 5), fontsize=8)
    plt.title(title)
    plt.xlabel('AJ_256')
    plt.ylabel('AJ_RD_256')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ab = load(str(ROOT / 'rgb_stacking_fresh20_49_natural_ablation/summary.json'))
    stats = load(str(ROOT / 'rgb_stacking_fresh20_49_natural_statistics/summary.json'))
    oracle = load(str(ROOT / 'b2_oracle_upper_bound/summary.json'))
    frozen = load(str(ROOT / 'reentry_stress_rgb_fresh20_49/fresh20_49_frozen_validation_summary.json'))
    provenance = load(str(ROOT / 'reentry_stress_rgb_fresh20_49/event_provenance_combined_summary.json'))
    frozen_stats = load(str(ROOT / 'reentry_stress_rgb_fresh20_49/fresh20_49_statistical_robustness_summary.json'))

    lines = []
    lines.append('# ReEntry-TAP Method-Paper Tables and Figures — 2026-07-01')
    lines.append('')
    lines.append('This document collects the main tables for the method-paper framing: **improving re-entry recovery in Tracking Any Point with Selective Local Override / B2-W16-P2**.')
    lines.append('')

    # Table 1 natural ablation
    lines.append('## Table 1. RGB fresh20-49 natural result and ablation')
    lines.append('')
    rows = []
    for m in ab['methods']:
        d = ab['gains'].get(f"{m['name']}_vs_offline", {})
        rows.append([
            m['name'], fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']),
            fmt(d.get('AJ_RD_256')), fmt(d.get('AJ_256')), fmt(d.get('OA_256')),
        ])
    lines.append(md_table(['Method','AJ_RD_256','AJ_256','OA_256','ΔAJ_RD','ΔAJ','ΔOA'], rows))
    lines.append('')
    lines.append('Key message: windowed local override is the core mechanism. Online/global improves AJ_RD only modestly but collapses AJ, while W8/W16 local override gives the strongest AJ_RD/AJ tradeoff.')
    lines.append('')

    trade_rows = [(m['name'], float(m['AJ_256']), float(m['AJ_RD_256'])) for m in ab['methods']]
    fig_tradeoff(trade_rows, 'RGB fresh20-49 natural tradeoff', FIGDIR / 'natural_tradeoff_aj_vs_ajrd.png')
    fig_bar([m['name'] for m in ab['methods']], [float(m['AJ_RD_256']) for m in ab['methods']], 'RGB fresh20-49 AJ_RD_256', 'AJ_RD_256', FIGDIR / 'natural_ablation_ajrd.png')

    # Table 2 natural statistics
    lines.append('## Table 2. RGB fresh20-49 natural video-level robustness')
    lines.append('')
    comp = stats['comparisons']['b2_w16_p2_vs_offline']
    rows = []
    for metric in ['AJ_RD_256','AJ_256','OA_256','delta_avg_256']:
        s = comp[metric]
        rows.append([
            metric,
            fmt(s['mean_delta'], 6),
            f"[{fmt(s['ci95_bootstrap'][0], 6)}, {fmt(s['ci95_bootstrap'][1], 6)}]",
            f"{s['positive_videos']}/{s['n']}",
            fmt(s['sign_test_p_two_sided'], 8),
            s.get('drop_gt_1', '—'),
            s.get('drop_gt_2', '—'),
        ])
    lines.append(md_table(['Metric','Mean Δ','95% CI','Positive videos','Sign-test p','Drop >1','Drop >2'], rows))
    lines.append('')
    lines.append('Key message: the main AJ_RD gain is video-consistent: +0.0613 mean video-level gain, 24/30 positive videos, and CI fully above zero.')
    lines.append('')

    # Table 3 frozen validation
    lines.append('## Table 3. Frozen RGB fresh20-49 L16 stress validation')
    lines.append('')
    rows = []
    for res in frozen['results']:
        fam = res['family']
        for m in res['methods']:
            rows.append([fam, m['name'], fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']), fmt(m['delta_avg_256'])])
    lines.append(md_table(['Stress','Method','AJ_RD_256','AJ_256','OA_256','delta_avg_256'], rows))
    lines.append('')

    lines.append('## Table 4. Frozen RGB fresh20-49 L16 stress gains')
    lines.append('')
    rows = []
    for res in frozen['results']:
        fam = res['family']
        for name, g in res['gains'].items():
            rows.append([fam, name, fmt(g['AJ_RD_256']), fmt(g['AJ_256']), fmt(g['OA_256'])])
    lines.append(md_table(['Stress','Comparison','ΔAJ_RD_256','ΔAJ_256','ΔOA_256'], rows))
    lines.append('')

    fig_bar([r['family'] for r in frozen['results']], [float(r['gains']['b2_vs_offline']['AJ_RD_256']) for r in frozen['results']], 'Frozen fresh20-49 B2 gains over offline', 'ΔAJ_RD_256', FIGDIR / 'frozen_fresh20_49_b2_gain_ajrd.png')

    # Table 5 stress-induced
    lines.append('## Table 5. Frozen RGB fresh20-49 stress-induced-only AJ_RD_256')
    lines.append('')
    rows = []
    for res in provenance['results']:
        stress = res['provenance']['stress_name']
        rates = res['provenance']['eligible_event_rate_by_source']
        off = res['ajrd_by_source']['offline']['AJ_RD_256_by_source']
        on = res['ajrd_by_source']['online']['AJ_RD_256_by_source']
        b2 = res['ajrd_by_source']['b2_w16_p2']['AJ_RD_256_by_source']
        rows.append([
            stress,
            fmt(rates['stress_induced'], 6),
            fmt(off['stress_induced']),
            fmt(on['stress_induced']),
            fmt(b2['stress_induced']),
            fmt(b2['stress_induced'] - off['stress_induced']),
            fmt(b2['stress_induced'] - on['stress_induced']),
        ])
    lines.append(md_table(['Stress','Stress-induced rate','offline','online','B2','B2-offline','B2-online'], rows))
    lines.append('')
    lines.append('Key message: on the 30-video frozen split, B2 remains positive on stress-induced-only re-entry events, not just mixed natural re-entry events.')
    lines.append('')

    # Table 6 frozen stats
    lines.append('## Table 6. Frozen RGB fresh20-49 video-level robustness')
    lines.append('')
    rows = []
    for res in frozen_stats['results']:
        fam = res['family']
        comp = res['comparisons']['b2_w16_p2_vs_offline']
        for metric in ['AJ_RD_256','AJ_256']:
            s = comp[metric]
            rows.append([
                fam, metric, fmt(s['mean_delta'], 6), f"[{fmt(s['ci95_bootstrap'][0], 6)}, {fmt(s['ci95_bootstrap'][1], 6)}]", f"{s['positive_videos']}/{s['n']}", fmt(s['sign_test_p_two_sided'], 8)
            ])
    lines.append(md_table(['Stress','Metric','Mean Δ','95% CI','Positive videos','Sign-test p'], rows))
    lines.append('')

    # Table 7 oracle
    lines.append('## Table 7. Oracle upper-bound analysis')
    lines.append('')
    rows = []
    for res in oracle['results']:
        setting = res['setting']
        by = {m['name']: m for m in res['methods']}
        for name in ['offline','online_global','b2_w16_p2','oracle_b2','oracle_online']:
            m = by[name]
            d_b2 = None
            if name.startswith('oracle'):
                d_b2 = float(m['AJ_RD_256']) - float(by['b2_w16_p2']['AJ_RD_256'])
            rows.append([setting, name, fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']), fmt(d_b2)])
    lines.append(md_table(['Setting','Method','AJ_RD_256','AJ_256','OA_256','ΔAJ_RD vs B2'], rows))
    lines.append('')
    lines.append('Key message: oracle_b2 consistently adds roughly +0.013 to +0.014 AJ_RD over B2-W16-P2, suggesting modest but real headroom for learned reliability gates.')
    lines.append('')

    # Figure manifest
    lines.append('## Generated figures')
    lines.append('')
    for p in sorted(FIGDIR.glob('*.png')):
        lines.append(f'- `{p}`')
    lines.append('')

    DOC.write_text('\n'.join(lines))
    print(json.dumps({'doc': str(DOC), 'figures': [str(p) for p in sorted(FIGDIR.glob('*.png'))]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
