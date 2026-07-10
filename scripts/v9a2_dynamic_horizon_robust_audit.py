#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
REPORT = OUTDIR / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a2_dynamic_horizon_robust_threshold_audit.json'
OUT_DOC = ROOT / 'docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md'

from scripts.v9a2_dynamic_horizon_controller import (
    JOINT,
    event_accept_mask,
    evaluate_accept,
    load_joint,
)
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    CANDIDATE,
    NATIVE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)


def fmt(x):
    return '' if x is None else f'{float(x):+.4f}'


def main() -> None:
    report = json.loads(REPORT.read_text())
    data = load_joint(JOINT)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))

    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)
    y_good = data['y_candidate_good'].astype(bool)
    y_worse = data['y_candidate_worse_px'].astype(bool)

    rows = []
    baseline_specs = [
        ('w8_preserve_common_only', common),
        ('w16_accept_all_joint', common | ext),
        ('oracle_ext_candidate_good', common | (ext & y_good)),
        ('oracle_ext_good_not_worse', common | (ext & y_good & ~y_worse)),
    ]
    for name, accept in baseline_specs:
        row = evaluate_accept(name, data, native, cand_by, accept, native_metric, native_pv)
        row['kind'] = 'baseline'
        row['accepted_extension'] = int(np.sum(accept & ext))
        row['accepted_ext_good'] = int(np.sum(accept & ext & y_good))
        row['accepted_ext_bad'] = int(np.sum(accept & ext & data['y_candidate_bad'].astype(bool)))
        row['accepted_ext_false_visible'] = int(np.sum(accept & ext & data['y_false_visible'].astype(bool)))
        row['accepted_ext_worse'] = int(np.sum(accept & ext & y_worse))
        rows.append(row)

    # Limited, predeclared set: models from prior diagnostics, event_max only.
    desired_scores = ['all_logreg', 'anchor_logreg', 'base_extratrees']
    fixed_thresholds = [0.01, 0.02, 0.05, 0.10]
    training_by_score = {}
    for block in report['training']:
        for overall in block['overall']:
            name = f"{overall['feature_set']}_{overall['model']}"
            if name in block['full_scores']:
                training_by_score[name] = {
                    'scores': np.asarray(block['full_scores'][name], dtype=float),
                    'threshold_f1': float(overall['threshold_f1_oof']),
                    'ap': overall['ap'],
                    'auc': overall['auc'],
                    'brier': overall['brier'],
                }

    for score_name in desired_scores:
        if score_name not in training_by_score:
            continue
        info = training_by_score[score_name]
        thresholds = [('oof_f1', info['threshold_f1'])] + [(f'fixed_{t:.2f}', t) for t in fixed_thresholds]
        for tname, thr in thresholds:
            accept = event_accept_mask(data, info['scores'], float(thr), 'event_max')
            row = evaluate_accept(f'{score_name}_event_max_{tname}', data, native, cand_by, accept, native_metric, native_pv)
            row['kind'] = 'robust_threshold'
            row['score_name'] = score_name
            row['threshold_name'] = tname
            row['threshold'] = float(thr)
            row['accepted_extension'] = int(np.sum(accept & ext))
            row['accepted_ext_good'] = int(np.sum(accept & ext & y_good))
            row['accepted_ext_bad'] = int(np.sum(accept & ext & data['y_candidate_bad'].astype(bool)))
            row['accepted_ext_false_visible'] = int(np.sum(accept & ext & data['y_false_visible'].astype(bool)))
            row['accepted_ext_worse'] = int(np.sum(accept & ext & y_worse))
            row['classifier_oof'] = {k: info[k] for k in ['ap', 'auc', 'brier']}
            rows.append(row)

    best = sorted(rows, key=lambda r: r['delta_vs_native']['AJ_RD_256'] if r['delta_vs_native']['AJ_RD_256'] is not None else -999, reverse=True)
    out = {'rows': rows, 'best_by_AJ_RD_256': best, 'notes': {'protocol': 'predeclared compact robust threshold audit; event_max only; no trajectory-wide dense sweep'}}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ['# V9-A2.3c-R Robust Threshold and Stability Audit', '', 'Protocol: predeclared compact threshold audit; event_max only; no dense trajectory sweep.', '']
    lines += ['| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---:|---|']
    for row in best:
        d = row['delta_vs_native']; pv = row['per_video_summary']; stats = row['apply_stats']
        ext_stats = f"{row['accepted_extension']}/{row['accepted_ext_good']}/{row['accepted_ext_bad']}/{row['accepted_ext_false_visible']}/{row['accepted_ext_worse']}"
        lines.append(f"| {row['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {stats.get('accepted_frames', 0)} | {ext_stats} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    lines += ['', 'Interpretation notes:', '', '- W8 common rows are preserved by default for learned variants.', '- Only W16-extension rows are controlled.', '- This audit is stricter than the earlier dense trajectory threshold sweep.']
    OUT_DOC.write_text('\n'.join(lines))
    top = best[0]
    print(json.dumps({
        'ok': True,
        'json': str(OUT_JSON),
        'doc': str(OUT_DOC),
        'n_rows': len(rows),
        'best': top['variant'],
        'best_AJ_RD_256_delta': top['delta_vs_native']['AJ_RD_256'],
        'best_AJ_delta': top['delta_vs_native']['AJ'],
        'best_OA_delta': top['delta_vs_native']['OA'],
    }))


if __name__ == '__main__':
    main()
