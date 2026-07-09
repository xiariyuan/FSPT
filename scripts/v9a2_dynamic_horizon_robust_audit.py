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
OUT_DOC = ROOT / 'docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-08.md'

from scripts.v9a2_dynamic_horizon_controller import (
    JOINT,
    load_joint,
    event_accept_mask,
    evaluate_accept,
)
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    NATIVE,
    CANDIDATE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)


def fmt(x):
    if x is None:
        return ''
    return f'{float(x):+.4f}'


def ext_accept_stats(data: dict, accept: np.ndarray) -> dict:
    ext = data['is_w16_extension'].astype(bool)
    sel = ext & accept.astype(bool)
    total_good = float(np.sum(data['y_candidate_good'][ext]))
    out = {
        'accepted_ext': int(np.sum(sel)),
        'total_ext': int(np.sum(ext)),
        'accepted_ext_rate': float(np.mean(accept[ext])) if np.sum(ext) else None,
    }
    for key in ['y_candidate_good', 'y_candidate_bad', 'y_false_visible', 'y_candidate_worse_px', 'y_repair16', 'y_damage16', 'y_utility']:
        arr = data[key].astype(float)
        out[key + '_sum_ext_accepted'] = float(np.sum(arr[sel]))
        out[key + '_mean_ext_accepted'] = None if np.sum(sel) == 0 else float(np.mean(arr[sel]))
    out['candidate_good_recall_ext'] = None if total_good == 0 else float(np.sum(data['y_candidate_good'][sel]) / total_good)
    out['candidate_good_precision_ext'] = None if np.sum(sel) == 0 else float(np.mean(data['y_candidate_good'][sel]))
    return out


def find_training(report: dict, score_name: str) -> tuple[dict, dict]:
    for block in report['training']:
        if score_name in block['full_scores']:
            for row in block['overall']:
                if f"{row['feature_set']}_{row['model']}" == score_name:
                    return block, row
    raise KeyError(score_name)


def main() -> None:
    report = json.loads(REPORT.read_text())
    data = load_joint(JOINT)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(align_info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)

    rows = []
    def add_eval(name: str, accept: np.ndarray, extra: dict | None = None):
        row = evaluate_accept(name, data, native, cand_by, accept, native_metric, native_pv)
        row['extension_accept_stats'] = ext_accept_stats(data, accept)
        if extra:
            row.update(extra)
        rows.append(row)

    # Baselines and oracle anchors.
    add_eval('w8_preserve_common_only', common, {'category': 'baseline'})
    add_eval('w16_accept_all_joint', common | ext, {'category': 'baseline'})
    add_eval('oracle_ext_candidate_good', common | (ext & data['y_candidate_good'].astype(bool)), {'category': 'oracle'})
    add_eval('oracle_ext_good_not_worse', common | (ext & data['y_candidate_good'].astype(bool) & ~data['y_candidate_worse_px'].astype(bool)), {'category': 'oracle'})

    # Small robust set: no full trajectory sweep. Include OOF-F1 and a few fixed thresholds.
    planned = [
        ('all_logreg', 'event_max', [('oof_f1', None), ('fixed_0.02', 0.02), ('fixed_0.05', 0.05), ('dev_best_reference', 0.048428640060746736)]),
        ('all_logreg', 'event_mean', [('fixed_0.005', 0.005), ('fixed_0.05', 0.05)]),
        ('all_logreg', 'frame', [('fixed_0.05', 0.05)]),
        ('anchor_logreg', 'event_max', [('oof_f1', None), ('fixed_0.01', 0.01), ('fixed_0.02', 0.02)]),
        ('base_extratrees', 'event_max', [('oof_f1', None), ('fixed_0.30', 0.30), ('fixed_0.36', 0.36), ('fixed_0.50', 0.50)]),
        ('base_hgb', 'event_max', [('oof_f1', None), ('fixed_0.015', 0.015), ('fixed_0.05', 0.05)]),
        ('all_extratrees', 'event_max', [('oof_f1', None), ('fixed_0.32', 0.32), ('fixed_0.50', 0.50)]),
    ]
    for score_name, mode, thrs in planned:
        block, overall = find_training(report, score_name)
        scores = np.asarray(block['full_scores'][score_name], dtype=float)
        for label, thr in thrs:
            if thr is None:
                thr = float(overall['threshold_f1_oof'])
            accept = event_accept_mask(data, scores, float(thr), mode)
            add_eval(
                f'{score_name}_{mode}_{label}_thr_{float(thr):.6f}',
                accept,
                {
                    'category': 'robust_threshold',
                    'score_name': score_name,
                    'mode': mode,
                    'threshold_label': label,
                    'threshold': float(thr),
                    'oof_ap': overall['ap'],
                    'oof_auc': overall['auc'],
                    'oof_brier': overall['brier'],
                },
            )

    sorted_rows = sorted(rows, key=lambda r: r['delta_vs_native']['AJ_RD_256'] if r['delta_vs_native']['AJ_RD_256'] is not None else -999, reverse=True)
    out = {
        'script': 'scripts/v9a2_dynamic_horizon_robust_audit.py',
        'source_report': str(REPORT),
        'joint': str(JOINT),
        'n_rows': int(len(common)),
        'n_common': int(np.sum(common)),
        'n_extension': int(np.sum(ext)),
        'rows': rows,
        'best_by_AJ_RD_256': sorted_rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ['# V9-A2.3c-R Robust Threshold and Stability Audit Result', '']
    lines += ['## Summary', '']
    lines += [
        'This audit uses a small fixed threshold set instead of full trajectory sweep.',
        '',
        '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Ext accepted | Ext good precision | Ext good recall | Pos/Neg/Zero |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in sorted_rows[:24]:
        d = row['delta_vs_native']; pv = row['per_video_summary']; stats = row['apply_stats']; es = row['extension_accept_stats']
        prec = '' if es['candidate_good_precision_ext'] is None else f"{es['candidate_good_precision_ext']:.3f}"
        rec = '' if es['candidate_good_recall_ext'] is None else f"{es['candidate_good_recall_ext']:.3f}"
        lines.append(f"| {row['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {stats.get('accepted_frames', 0)} | {es['accepted_ext']} | {prec} | {rec} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    OUT_DOC.write_text('\n'.join(lines))
    print(json.dumps({
        'ok': True,
        'json': str(OUT_JSON),
        'doc': str(OUT_DOC),
        'n_eval': len(rows),
        'best': sorted_rows[0]['variant'],
        'best_AJ_RD_256_delta': sorted_rows[0]['delta_vs_native']['AJ_RD_256'],
        'best_AJ_delta': sorted_rows[0]['delta_vs_native']['AJ'],
        'best_OA_delta': sorted_rows[0]['delta_vs_native']['OA'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
