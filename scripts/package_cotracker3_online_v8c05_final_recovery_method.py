#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/gemini/code/FSPT')
OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c05_final_recovery_method'
DOC = ROOT / 'docs/cotracker3_online_v8c05_final_recovery_method_2026-07-07.md'
REPORTS = {
    'v7b12': ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b12_damage_ceiling_group_robustness/v7b12_damage_ceiling_group_robustness_report.json',
    'v8c01': ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c01_recovery_risk_audit/v8c01_recovery_risk_audit_report.json',
    'v8c04': ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c04_apply_fine_risk_verifier/v8c04_apply_fine_risk_verifier_report.json',
}
METRIC_KEYS = ['AJ', 'OA', 'AJ_RD', 'AJ_RD_256']


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def find_v7(v7: dict, name: str) -> dict:
    for r in v7['full_rows']:
        if r['policy']['name'] == name:
            return r
    raise KeyError(name)


def find_row(rep: dict, variant: str, source: str | None = None) -> dict:
    for r in rep['rows']:
        if r['variant'] == variant and (source is None or r.get('event_source') == source):
            return r
    raise KeyError((variant, source))


def compact_metrics(metric: dict) -> dict:
    return {k: metric.get(k) for k in METRIC_KEYS}


def table_row(name: str, kind: str, metric: dict, delta: dict, source: str, notes: str, stats=None, robustness=None, per_video=None) -> dict:
    return {
        'name': name,
        'kind': kind,
        'metric': compact_metrics(metric),
        'delta_vs_native': compact_metrics(delta),
        'source': source,
        'notes': notes,
        'stats': stats or {},
        'robustness': robustness or {},
        'per_video_delta': per_video or [],
    }


def fmt_delta(x):
    return '' if x is None else f'{float(x):+.4f}'


def fmt_abs(x):
    return '' if x is None else f'{float(x):.4f}'


def markdown_table(rows: list[dict]) -> str:
    lines = [
        '| Method | AJ Δ | OA Δ | AJ_RD Δ | AJ_RD_256 Δ | AJ_RD_256 abs | Robustness | Notes |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for r in rows:
        d = r['delta_vs_native']; m = r['metric']; rb = r.get('robustness') or {}
        rob = ''
        if 'positive' in rb:
            rob = f"{rb.get('positive')}/{rb.get('negative')}/{rb.get('zero')} pos/neg/zero"
        elif 'success_folds' in rb:
            rob = f"{rb.get('success_folds')}/5 folds pass"
        vals = [r['name'], fmt_delta(d.get('AJ')), fmt_delta(d.get('OA')), fmt_delta(d.get('AJ_RD')), fmt_delta(d.get('AJ_RD_256')), fmt_abs(m.get('AJ_RD_256')), rob, r.get('notes', '')]
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in vals) + ' |')
    return '\n'.join(lines)


def pv_table(row: dict, title: str) -> str:
    pvs = row.get('per_video_delta') or []
    if not pvs:
        return f'### {title}\n\nNo per-video rows.\n'
    lines = [f'### {title}', '', '| Video | ΔAJ_RD_256 | ΔAJ | ΔOA |', '| --- | ---: | ---: | ---: |']
    for r in sorted(pvs, key=lambda x: 999 if x.get('delta_AJ_RD_256') is None else x['delta_AJ_RD_256']):
        lines.append(f"| {r['video_id']} | {fmt_delta(r.get('delta_AJ_RD_256'))} | {fmt_delta(r.get('delta_AJ'))} | {fmt_delta(r.get('delta_OA'))} |")
    return '\n'.join(lines)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v7 = load(REPORTS['v7b12'])
    c01 = load(REPORTS['v8c01'])
    c04 = load(REPORTS['v8c04'])
    native = c01['native_metric']
    zero = {k: 0.0 for k in METRIC_KEYS}
    v7row = find_v7(v7, 'damage_ceiling0.7')
    all_w8 = find_row(c01, 'all_w8_candidate_visible', 'meta')
    default = find_row(c01, 'dist_nc_le64_w8_candidate_visible', 'meta')
    high = find_row(c01, 'dist_nc_le64_w16_candidate_visible', 'meta')
    verifier = find_row(c04, 'utility_et_ge_-0.3431')
    rows = [
        table_row('CoTracker3 online native', 'baseline', native, zero, str(REPORTS['v8c01']), 'reference'),
        table_row('V7-B2 visibility-only', 'visibility baseline', v7row['metric'], v7row['delta_vs_native'], str(REPORTS['v7b12']), 'visibility-only gain is small', robustness={'success_folds': v7['aggregate']['damage_ceiling0.7']['success_folds']}),
        table_row('TrackOn2 standalone', 'candidate baseline', c01['trackon2_metric'], c01['trackon2_delta_vs_native'], str(REPORTS['v8c01']), 'strong global tracker but lower AJ_RD_256 gain than CVRRM'),
        table_row('all-events w8 candidate-visible', 'ablation', all_w8['metric'], all_w8['delta_vs_native'], str(REPORTS['v8c01']), 'without distance sanity filter', all_w8.get('stats'), all_w8.get('per_video_summary'), all_w8.get('per_video_delta')),
        table_row('CVRRM default: dist<=64, w8', 'main method', default['metric'], default['delta_vs_native'], str(REPORTS['v8c01']), 'recommended default', default.get('stats'), default.get('per_video_summary'), default.get('per_video_delta')),
        table_row('CVRRM high-gain: dist<=64, w16', 'ablation', high['metric'], high['delta_vs_native'], str(REPORTS['v8c01']), 'slightly higher AJ_RD_256, more negative videos', high.get('stats'), high.get('per_video_summary'), high.get('per_video_delta')),
        table_row('OOF verifier ablation', 'learned ablation', verifier['metric'], verifier['delta_vs_native'], str(REPORTS['v8c04']), 'diagnostic only; not default', verifier.get('apply_stats'), verifier.get('per_video_summary'), verifier.get('per_video_delta')),
    ]
    package = {
        'script': 'scripts/package_cotracker3_online_v8c05_final_recovery_method.py',
        'reports': {k: str(v) for k, v in REPORTS.items()},
        'final_method_name': 'CVRRM: Causal Candidate-Visible Re-entry Recovery Mode',
        'default_variant': 'dist_nc_le64_w8_candidate_visible',
        'high_gain_variant': 'dist_nc_le64_w16_candidate_visible',
        'main_table': rows,
        'decision': {
            'default': 'dist_nc_le64_w8_candidate_visible',
            'high_gain_ablation': 'dist_nc_le64_w16_candidate_visible',
            'verifier_default': False,
            'next': 'V8-C1 cross-candidate / cross-native generalization audit, then state-level repair only if cross-source evidence holds.',
        },
    }
    out_json = OUTDIR / 'v8c05_final_recovery_method_table.json'
    out_json.write_text(json.dumps(package, indent=2, ensure_ascii=False))
    md = []
    md += ['# CoTracker3 Online V8-C0.5 Final Recovery Method Packaging', '', 'Date: 2026-07-07', '']
    md += ['## 1. Final method', '', '```text', 'CVRRM: Causal Candidate-Visible Re-entry Recovery Mode', '因果候选可见确认式重进入恢复模式', '```', '']
    md += ['## 2. Main result table', '', markdown_table(rows), '']
    md += ['## 3. Final decision', '', '```text', 'Default: dist_nc_le64_w8_candidate_visible', 'High-gain ablation: dist_nc_le64_w16_candidate_visible', 'Learned verifier: diagnostic ablation only, not default', '```', '']
    md += ['## 4. Default algorithm', '', '```text', 'For each query q and frame t:', '    low_mask = native_score[q,t] <= 0.60 or not native_visible[q,t]', '    if low_mask and trigger/cooldown conditions pass:', '        open recovery event e = (q,t)', '', 'For each recovery event e=(q,t0):', '    if distance(native_coord[q,t0], candidate_coord[q,t0]) > 64 px:', '        skip event', '    for tau in [t0, t0 + 8]:', '        when frame tau arrives:', '            if candidate_visible[q,tau]:', '                output candidate coord and visible', '            else:', '                keep native output', '```', '']
    md += ['Strict-causal note: the future-looking loop is a runtime recovery mode; each tau is processed only when that frame arrives.', '']
    md += ['## 5. Robustness tables', '', pv_table(default, 'Default: dist_nc_le64_w8_candidate_visible'), '', pv_table(high, 'High-gain: dist_nc_le64_w16_candidate_visible'), '']
    md += ['## 6. Learned verifier decision', '', 'The OOF verifier ablation improves AJ_RD_256 by only +0.0001 over the default. Stronger verifier filtering reduces negative videos but drops AJ_RD_256 below the +0.020 target. Therefore it is not promoted as the default.', '']
    md += ['## 7. Remaining limitations', '', '```text', '1. Output-level cached replay, not state-level repair yet.', '2. Cross-candidate-source generalization not yet verified.', '3. Some negative videos remain.', '```', '']
    md += ['## 8. Next step', '', '```text', 'V8-C1: cross-candidate / cross-native generalization audit.', 'Then test state-level repair only if cross-source evidence holds.', '```', '']
    DOC.write_text('\n'.join(md))
    mainline = ROOT / 'CURRENT_MAINLINE.md'
    with mainline.open('a') as f:
        f.write('\n\n## CoTracker3 online V8-C0.5 final recovery method packaging (2026-07-07)\n\n')
        f.write('Artifact:\n\n```text\n' + str(DOC.relative_to(ROOT)) + '\n```\n\n')
        f.write('Report:\n\n```text\n' + str(out_json.relative_to(ROOT)) + '\n```\n\n')
        f.write('Decision:\n\n```text\n')
        f.write('Promote CVRRM default: dist_nc_le64_w8_candidate_visible.\n')
        f.write('Default result: AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274, positive/negative/zero = 16/4/5.\n')
        f.write('High-gain ablation: dist_nc_le64_w16_candidate_visible, AJ_RD_256 +0.0286, positive/negative/zero = 15/5/5.\n')
        f.write('Do not promote learned fine-risk verifier as default.\n')
        f.write('Next: V8-C1 cross-candidate / cross-native generalization audit; state-level repair only if cross-source evidence holds.\n')
        f.write('```\n')
    print(json.dumps({'out_json': str(out_json), 'doc': str(DOC), 'rows': [{'name': r['name'], 'delta': r['delta_vs_native'], 'robustness': r['robustness']} for r in rows], 'decision': package['decision']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
