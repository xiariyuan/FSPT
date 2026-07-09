#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('outputs/paper_discovery_2026-06-27')
DOC = Path('docs/reentry_tap_paper_tables_2026-07-01.md')
FIGDIR = Path('docs/figures/reentry_tap')
MANIFEST = Path('docs/reentry_tap_figures_manifest_2026-07-01.md')


def load(p: str | Path) -> Any:
    return json.load(open(p))


def fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return '—'
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f'{x:.{n}f}'
    return str(x)


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def method_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r['name']: r for r in rows}


def stress_rows(summary: Dict[str, Any], family_label: str) -> List[List[str]]:
    rows = []
    for s in summary['summaries']:
        L = s['L']
        for m in s['methods']:
            rows.append([family_label, L, m['name'], fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']), fmt(m['delta_avg_256'])])
    return rows


def gain_rows(summary: Dict[str, Any], family_label: str) -> List[List[str]]:
    rows = []
    for s in summary['summaries']:
        for comp in ['online_vs_offline', 'b2_vs_offline', 'b2_vs_online']:
            g = s['gains'][comp]
            rows.append([family_label, s['L'], comp, fmt(g['AJ_RD_256']), fmt(g['AJ_256']), fmt(g['OA_256'])])
    return rows


def stress_induced_rows(prov: Dict[str, Any]) -> List[List[str]]:
    rows = []
    for r in prov['results']:
        stress = r['provenance']['stress_name'].replace('translate_exit_reenter_', 'translate ').replace('moving_occluder_', 'occluder ')
        rates = r['provenance']['eligible_event_rate_by_source']
        off = r['ajrd_by_source']['offline']['AJ_RD_256_by_source']
        on = r['ajrd_by_source']['online']['AJ_RD_256_by_source']
        b2 = r['ajrd_by_source']['b2_w16_p2']['AJ_RD_256_by_source']
        rows.append([
            stress,
            fmt(rates.get('stress_induced')),
            fmt(off.get('stress_induced')),
            fmt(on.get('stress_induced')),
            fmt(b2.get('stress_induced')),
            fmt((b2.get('stress_induced') or 0) - (off.get('stress_induced') or 0)),
            fmt((b2.get('stress_induced') or 0) - (on.get('stress_induced') or 0)),
        ])
    return rows


def plot_metric(summary: Dict[str, Any], family_label: str, metric: str, ylabel: str, out_path: Path) -> None:
    Ls = [s['L'] for s in summary['summaries']]
    methods = ['cotracker3_offline', 'cotracker3_online', 'b2_w16_p2']
    labels = {'cotracker3_offline': 'Offline', 'cotracker3_online': 'Online', 'b2_w16_p2': 'B2-W16-P2'}
    plt.figure(figsize=(6, 4))
    for m in methods:
        vals = []
        for s in summary['summaries']:
            mm = method_map(s['methods'])[m]
            vals.append(mm[metric])
        plt.plot(Ls, vals, marker='o', label=labels[m])
    plt.xlabel('Stress length L')
    plt.ylabel(ylabel)
    plt.title(f'{family_label}: {ylabel} vs L')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_fresh(fresh: Dict[str, Any], metric: str, ylabel: str, out_path: Path) -> None:
    families = []
    offline = []; online = []; b2 = []
    for r in fresh['results']:
        families.append(r['family'])
        mm = {m['name']: m for m in r['methods']}
        offline.append(mm['offline'][metric])
        online.append(mm['online'][metric])
        b2.append(mm['b2_w16_p2'][metric])
    x = np.arange(len(families))
    w = 0.25
    plt.figure(figsize=(6, 4))
    plt.bar(x - w, offline, width=w, label='Offline')
    plt.bar(x, online, width=w, label='Online')
    plt.bar(x + w, b2, width=w, label='B2-W16-P2')
    plt.xticks(x, families)
    plt.ylabel(ylabel)
    plt.title(f'Fresh20-29 frozen validation: {ylabel}')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stress_induced(prov: Dict[str, Any], out_path: Path) -> None:
    names = []
    off = []; on = []; b2 = []
    for r in prov['results']:
        name = r['provenance']['stress_name'].replace('translate_exit_reenter_', 'T-').replace('moving_occluder_', 'O-')
        names.append(name)
        off.append(r['ajrd_by_source']['offline']['AJ_RD_256_by_source']['stress_induced'])
        on.append(r['ajrd_by_source']['online']['AJ_RD_256_by_source']['stress_induced'])
        b2.append(r['ajrd_by_source']['b2_w16_p2']['AJ_RD_256_by_source']['stress_induced'])
    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(9, 4))
    plt.bar(x - w, off, width=w, label='Offline')
    plt.bar(x, on, width=w, label='Online')
    plt.bar(x + w, b2, width=w, label='B2-W16-P2')
    plt.xticks(x, names, rotation=25)
    plt.ylabel('Stress-induced-only AJ_RD_256')
    plt.title('Stress-induced-only re-entry performance')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    rgb = load(ROOT / 'rgb_stacking_fresh20_49_aggregate/summary.json')
    translate = load(ROOT / 'reentry_stress_rgb_dev10/translate_severity_summary.json')
    occluder = load(ROOT / 'reentry_stress_rgb_dev10/occluder_severity_summary.json')
    prov = load(ROOT / 'reentry_stress_rgb_dev10/event_provenance_combined_summary.json')
    inter = load(ROOT / 'reentry_stress_rgb_dev10/intersection_query_severity_summary.json')
    ablate = load(ROOT / 'reentry_stress_rgb_dev10/stress_ablation_L16_summary.json')
    stats = load(ROOT / 'reentry_stress_rgb_dev10/stress_statistical_robustness_summary.json')
    fresh = load(ROOT / 'reentry_stress_rgb_fresh20_29/fresh20_29_frozen_validation_summary.json')
    fresh_stats = load(ROOT / 'reentry_stress_rgb_fresh20_29/fresh20_29_statistical_robustness_summary.json')
    fresh_prov = load(ROOT / 'reentry_stress_rgb_fresh20_29/event_provenance_combined_summary.json')

    lines = []
    lines.append('# ReEntry-TAP Paper Tables — 2026-07-01\n')
    lines.append('Generated from JSON summaries by `scripts/make_reentry_tap_paper_tables_and_figures.py`.\n')

    # Natural RGB table.
    qw = rgb['query_weighted']
    rows = []
    for key, label in [('offline','CoTracker3 offline'),('online','CoTracker3 online'),('b2_w16_p2','B2-W16-P2')]:
        m = qw[key]
        rows.append([label, fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']), fmt(m['delta_avg_256']), m['n_records'], m['n_queries']])
    lines.append('## Table 1. Natural RGB fresh20-49 validation\n')
    lines.append(md_table(['Method','AJ_RD_256','AJ_256','OA_256','delta_avg_256','#videos','#queries'], rows) + '\n')
    b2 = qw['b2_w16_p2']; off = qw['offline']; on = qw['online']
    lines.append(f"B2-W16-P2 vs offline: AJ_RD {fmt(b2['AJ_RD_256']-off['AJ_RD_256'])}, AJ {fmt(b2['AJ_256']-off['AJ_256'])}.  ")
    lines.append(f"B2-W16-P2 vs online: AJ_RD {fmt(b2['AJ_RD_256']-on['AJ_RD_256'])}, AJ {fmt(b2['AJ_256']-on['AJ_256'])}.\n")

    lines.append('## Table 2. Translate severity curve on RGB dev0-9\n')
    lines.append(md_table(['Family','L','Method','AJ_RD_256','AJ_256','OA_256','delta_avg_256'], stress_rows(translate, 'translate')) + '\n')
    lines.append('## Table 3. Moving-occluder severity curve on RGB dev0-9\n')
    lines.append(md_table(['Family','L','Method','AJ_RD_256','AJ_256','OA_256','delta_avg_256'], stress_rows(occluder, 'occluder')) + '\n')
    lines.append('## Table 4. Stress gains on RGB dev0-9\n')
    lines.append(md_table(['Family','L','Comparison','ΔAJ_RD_256','ΔAJ_256','ΔOA_256'], gain_rows(translate, 'translate') + gain_rows(occluder, 'occluder')) + '\n')

    lines.append('## Table 5. Stress-induced-only AJ_RD_256 on RGB dev0-9\n')
    lines.append(md_table(['Stress','stress-induced event rate','offline','online','B2-W16-P2','B2-offline','B2-online'], stress_induced_rows(prov)) + '\n')

    # Intersection compact.
    rows = []
    for fam in inter['results']:
        for s in fam['summaries']:
            rows.append([fam['family'], s['L'], s['sanity']['num_queries'], fmt(s['gains']['b2_vs_offline']['AJ_RD_256']), fmt(s['gains']['b2_vs_offline']['AJ_256']), fmt(s['gains']['online_vs_offline']['AJ_256'])])
    lines.append('## Table 6. Intersection-query strict audit\n')
    lines.append(md_table(['Family','L','#intersection queries','B2-offline ΔAJ_RD','B2-offline ΔAJ','online-offline ΔAJ'], rows) + '\n')

    # Ablation compact.
    rows = []
    for fam in ablate['results']:
        by_off = {g.replace('_vs_offline',''): v for g, v in fam['gains'].items()}
        for m in fam['methods']:
            d = by_off.get(m['name'])
            rows.append([fam['family'], m['name'], fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(d['AJ_RD_256']) if d else '—', fmt(d['AJ_256']) if d else '—'])
    lines.append('## Table 7. L16 stress ablation\n')
    lines.append(md_table(['Family','Method','AJ_RD_256','AJ_256','ΔAJ_RD vs offline','ΔAJ vs offline'], rows) + '\n')

    # Stats compact.
    rows = []
    for r in stats['results']:
        c = r['comparisons']['b2_w16_p2_vs_offline']
        rows.append([r['family'], r['L'], fmt(c['AJ_RD_256']['mean_delta']), f"[{fmt(c['AJ_RD_256']['ci95_bootstrap'][0])}, {fmt(c['AJ_RD_256']['ci95_bootstrap'][1])}]", f"{c['AJ_RD_256']['positive_videos']}/{c['AJ_RD_256']['n']}", fmt(c['AJ_RD_256']['sign_test_p_two_sided'], 6), fmt(c['AJ_256']['mean_delta']), f"[{fmt(c['AJ_256']['ci95_bootstrap'][0])}, {fmt(c['AJ_256']['ci95_bootstrap'][1])}]"])
    lines.append('## Table 8. Per-video statistical robustness on dev stress\n')
    lines.append(md_table(['Family','L','mean ΔAJ_RD','95% CI','positive videos','sign-test p','mean ΔAJ','95% CI'], rows) + '\n')

    # Fresh validation.
    rows = []
    for r in fresh['results']:
        for m in r['methods']:
            rows.append([r['family'], m['name'], fmt(m['AJ_RD_256']), fmt(m['AJ_256']), fmt(m['OA_256']), fmt(m['delta_avg_256'])])
    lines.append('## Table 9. Frozen RGB fresh20-29 validation\n')
    lines.append(md_table(['Stress','Method','AJ_RD_256','AJ_256','OA_256','delta_avg_256'], rows) + '\n')
    rows = []
    for r in fresh['results']:
        for comp in ['online_vs_offline','b2_vs_offline','b2_vs_online']:
            g = r['gains'][comp]
            rows.append([r['family'], comp, fmt(g['AJ_RD_256']), fmt(g['AJ_256']), fmt(g['OA_256'])])
    lines.append('## Table 10. Frozen RGB fresh20-29 gains\n')
    lines.append(md_table(['Stress','Comparison','ΔAJ_RD_256','ΔAJ_256','ΔOA_256'], rows) + '\n')

    rows = []
    for r in fresh_stats['results']:
        c = r['comparisons']['b2_w16_p2_vs_offline']
        rows.append([r['family'], fmt(c['AJ_RD_256']['mean_delta']), f"[{fmt(c['AJ_RD_256']['ci95_bootstrap'][0])}, {fmt(c['AJ_RD_256']['ci95_bootstrap'][1])}]", f"{c['AJ_RD_256']['positive_videos']}/{c['AJ_RD_256']['n']}", fmt(c['AJ_256']['mean_delta']), f"[{fmt(c['AJ_256']['ci95_bootstrap'][0])}, {fmt(c['AJ_256']['ci95_bootstrap'][1])}]"])
    lines.append('## Table 11. Frozen RGB fresh20-29 per-video robustness\n')
    lines.append(md_table(['Stress','mean ΔAJ_RD','95% CI','positive videos','mean ΔAJ','95% CI'], rows) + '\n')

    rows = stress_induced_rows(fresh_prov)
    lines.append('## Table 12. Frozen RGB fresh20-29 stress-induced-only AJ_RD_256\n')
    lines.append(md_table(['Stress','stress-induced event rate','offline','online','B2-W16-P2','B2-offline','B2-online'], rows) + '\n')

    DOC.write_text('\n'.join(lines))

    # Figures.
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_metric(translate, 'Translate stress', 'AJ_RD_256', 'AJ_RD_256', FIGDIR / 'translate_ajrd_vs_L.png')
    plot_metric(translate, 'Translate stress', 'AJ_256', 'AJ_256', FIGDIR / 'translate_aj_vs_L.png')
    plot_metric(occluder, 'Moving-occluder stress', 'AJ_RD_256', 'AJ_RD_256', FIGDIR / 'occluder_ajrd_vs_L.png')
    plot_metric(occluder, 'Moving-occluder stress', 'AJ_256', 'AJ_256', FIGDIR / 'occluder_aj_vs_L.png')
    plot_fresh(fresh, 'AJ_RD_256', 'AJ_RD_256', FIGDIR / 'fresh20_29_frozen_ajrd.png')
    plot_fresh(fresh, 'AJ_256', 'AJ_256', FIGDIR / 'fresh20_29_frozen_aj.png')
    plot_stress_induced(prov, FIGDIR / 'dev_stress_induced_only_ajrd.png')
    plot_stress_induced(fresh_prov, FIGDIR / 'fresh_stress_induced_only_ajrd.png')

    figs = sorted(FIGDIR.glob('*.png'))
    MANIFEST.write_text('# ReEntry-TAP Figures Manifest — 2026-07-01\n\n' + '\n'.join(f'- `{p}`' for p in figs) + '\n')
    print(json.dumps({'tables': str(DOC), 'figures_manifest': str(MANIFEST), 'figures': [str(p) for p in figs]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
