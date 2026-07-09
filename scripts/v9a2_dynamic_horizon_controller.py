#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
JOINT = OUTDIR / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    NATIVE,
    CANDIDATE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import (
    apply_touched_mask,
    label_stats,
    per_video_delta,
    summarize_pv,
)
from scripts.v9a1_controller_calibration_aware_prototype import (
    choose_threshold_by_metric,
    risk_coverage_curve,
    video_from_meta,
)


def load_joint(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    data['metas'] = [json.loads(str(x)) for x in data['meta_json'].tolist()]
    data['event_keys'] = [tuple(json.loads(str(x))) for x in data['event_key_json'].tolist()]
    data['row_keys'] = [tuple(json.loads(str(x))) for x in data['row_key_json'].tolist()]
    data['groups'] = np.asarray([video_from_meta(m) for m in data['metas']], dtype=object)
    return data


def select_X(data: dict, feature_set: str) -> tuple[np.ndarray, list[str]]:
    if feature_set == 'base':
        return data['X_base'].astype(np.float32), [str(x) for x in data['base_feature_names'].tolist()]
    if feature_set == 'anchor':
        return data['X_anchor'].astype(np.float32), [str(x) for x in data['anchor_feature_names'].tolist()]
    if feature_set == 'all':
        return data['X_all'].astype(np.float32), [str(x) for x in data['all_feature_names'].tolist()]
    raise ValueError(f'unknown feature_set {feature_set}')


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out = {
        'ap': float(average_precision_score(y, s)),
        'brier': float(brier_score_loss(y, np.clip(s, 0, 1))),
    }
    try:
        out['auc'] = float(roc_auc_score(y, s))
    except ValueError:
        out['auc'] = None
    out.update(risk_coverage_curve(s, 1.0 - y.astype(float), [0.25, 0.5, 0.75, 1.0]))
    return out


def groupheldout_extension_oof(data: dict, feature_set: str, target: str = 'y_candidate_good') -> dict:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, names = select_X(data, feature_set)
    ext = data['is_w16_extension'].astype(bool)
    X_ext = X[ext]
    y_ext = data[target][ext].astype(int)
    groups_ext = data['groups'][ext]
    unique_groups = np.unique(groups_ext)
    if len(unique_groups) < 2:
        raise ValueError('Need at least two videos with extension rows')
    n_splits = min(5, len(unique_groups))
    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        'hgb': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
        'extratrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, class_weight='balanced', random_state=0, n_jobs=-1),
    }
    oof_ext = {name: np.full(len(y_ext), np.nan, dtype=float) for name in models}
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X_ext, y_ext, groups_ext)):
        for name, model in models.items():
            clf = model.fit(X_ext[tr], y_ext[tr])
            if hasattr(clf, 'predict_proba'):
                score = clf.predict_proba(X_ext[va])[:, 1]
            else:
                raw = clf.decision_function(X_ext[va])
                score = 1 / (1 + np.exp(-raw))
            oof_ext[name][va] = score
            row = {'fold': int(fold), 'model': name, 'feature_set': feature_set, 'target': target, 'n_train': int(len(tr)), 'n_val': int(len(va)), 'val_videos': sorted(set(groups_ext[va].tolist())), 'positive_rate_val': float(np.mean(y_ext[va]))}
            row.update(binary_metrics(y_ext[va], score))
            row['threshold_f1'] = choose_threshold_by_metric(y_ext[va], score, metric='f1')
            fold_rows.append(row)
    overall = []
    for name, score in oof_ext.items():
        row = {'model': name, 'feature_set': feature_set, 'target': target, 'n_ext': int(len(y_ext)), 'positive_rate': float(np.mean(y_ext))}
        row.update(binary_metrics(y_ext, score))
        row['threshold_f1_oof'] = choose_threshold_by_metric(y_ext, score, metric='f1')
        overall.append(row)
    full_scores = {}
    ext_indices = np.where(ext)[0]
    for name, score_ext in oof_ext.items():
        full = np.zeros(len(ext), dtype=float)
        full[ext_indices] = score_ext
        full_scores[f'{feature_set}_{name}'] = full
    return {'feature_set': feature_set, 'target': target, 'feature_names': names, 'folds': fold_rows, 'overall': overall, 'full_scores': {k: v.tolist() for k, v in full_scores.items()}}


def event_accept_mask(data: dict, ext_scores: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)
    accept = common.copy()
    if mode == 'frame':
        accept[ext] = ext_scores[ext] >= threshold
        return accept
    # Event-level: if any/mean/max event score crosses threshold, accept all extension rows of that event.
    ev_to_idxs: dict[tuple, list[int]] = {}
    for i, ev in enumerate(data['event_keys']):
        if ext[i]:
            ev_to_idxs.setdefault(ev, []).append(i)
    for idxs in ev_to_idxs.values():
        vals = ext_scores[np.asarray(idxs, dtype=np.int64)]
        if mode == 'event_max':
            score = float(np.max(vals))
        elif mode == 'event_mean':
            score = float(np.mean(vals))
        else:
            raise ValueError(f'unknown mode {mode}')
        if score >= threshold:
            accept[np.asarray(idxs, dtype=np.int64)] = True
    return accept


def metric_delta(metric: dict, base: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: (None if metric.get(k) is None or base.get(k) is None else float(metric[k] - base[k])) for k in keys}


def evaluate_accept(name: str, data: dict, native: dict, cand_by: dict, accept: np.ndarray, native_metric: dict, native_pv: list[dict]) -> dict:
    labels = {k[2:]: data[k].astype(np.float32) for k in data if k.startswith('y_')}
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], accept.astype(bool))
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    return {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'label_stats_on_accepted': label_stats(labels, accept.astype(bool)),
        'per_video_summary': summarize_pv(pv_rows),
    }


def threshold_sweep_values(scores: np.ndarray) -> list[float]:
    ext_scores = scores[np.isfinite(scores) & (scores > 0)]
    if ext_scores.size == 0:
        return [1.0]
    vals = np.quantile(ext_scores, np.linspace(0.0, 1.0, 41)).tolist()
    vals.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return sorted(set(float(v) for v in vals))


def run_applyback(data: dict, score_blocks: list[dict]) -> dict:
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    rows = []
    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)
    rows.append(evaluate_accept('native_reject_all', data, native, cand_by, np.zeros(len(common), dtype=bool), native_metric, native_pv))
    rows.append(evaluate_accept('w8_preserve_common_only', data, native, cand_by, common, native_metric, native_pv))
    rows.append(evaluate_accept('w16_accept_all_joint', data, native, cand_by, common | ext, native_metric, native_pv))
    rows.append(evaluate_accept('oracle_w8_plus_ext_candidate_good', data, native, cand_by, common | (ext & data['y_candidate_good'].astype(bool)), native_metric, native_pv))
    rows.append(evaluate_accept('oracle_w8_plus_ext_good_not_worse', data, native, cand_by, common | (ext & data['y_candidate_good'].astype(bool) & ~data['y_candidate_worse_px'].astype(bool)), native_metric, native_pv))

    sweep_rows = []
    for block in score_blocks:
        for score_name, score_list in block['full_scores'].items():
            scores = np.asarray(score_list, dtype=float)
            for mode in ['frame', 'event_max', 'event_mean']:
                for thr in threshold_sweep_values(scores):
                    accept = event_accept_mask(data, scores, thr, mode)
                    row = evaluate_accept(f'learned_{score_name}_{mode}_thr_{thr:.4f}', data, native, cand_by, accept, native_metric, native_pv)
                    row['score_name'] = score_name
                    row['mode'] = mode
                    row['threshold'] = float(thr)
                    sweep_rows.append(row)
    best = sorted(sweep_rows, key=lambda r: -999 if r['delta_vs_native'].get('AJ_RD_256') is None else r['delta_vs_native']['AJ_RD_256'], reverse=True)
    # Keep top rows plus baselines in detailed report.
    return {'native_metric': native_metric, 'baselines': rows, 'sweep_top_by_AJ_RD_256': best[:60], 'sweep_rows_count': int(len(sweep_rows))}


def write_doc(report: dict, path: Path) -> None:
    def fmt(x):
        return '' if x is None else f'{float(x):+.4f}'
    lines = ['# V9-A2.3c Dynamic Horizon Controller Result', '']
    lines += ['## Extension classifier OOF summary', '', '| Feature set / Model | AP | AUC | Brier | Risk@50 |', '|---|---:|---:|---:|---:|']
    for block in report['training']:
        for row in block['overall']:
            auc = '' if row['auc'] is None else f"{row['auc']:.4f}"
            lines.append(f"| {row['feature_set']} / {row['model']} | {row['ap']:.4f} | {auc} | {row['brier']:.4f} | {row['coverage_0.50_mean_loss']:.4f} |")
    lines += ['', '## Apply-back baselines', '', '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---|']
    for row in report['applyback']['baselines']:
        d = row['delta_vs_native']; pv = row['per_video_summary']; stats = row['apply_stats']
        lines.append(f"| {row['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {stats.get('accepted_frames', 0)} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    lines += ['', '## Best learned dynamic-horizon rows', '', '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---|']
    for row in report['applyback']['sweep_top_by_AJ_RD_256'][:15]:
        d = row['delta_vs_native']; pv = row['per_video_summary']; stats = row['apply_stats']
        lines.append(f"| {row['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {stats.get('accepted_frames', 0)} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    path.write_text('\n'.join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', type=Path, default=JOINT)
    ap.add_argument('--feature-sets', nargs='+', default=['base', 'anchor', 'all'])
    ap.add_argument('--target', default='y_candidate_good')
    ap.add_argument('--outdir', type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_joint(args.joint)
    training = []
    for fs in args.feature_sets:
        training.append(groupheldout_extension_oof(data, fs, target=args.target))
    applyback = run_applyback(data, training)
    report = {'script': 'scripts/v9a2_dynamic_horizon_controller.py', 'joint': str(args.joint), 'target': args.target, 'training': training, 'applyback': applyback}
    # Drop full OOF arrays from compact report? Keep them for reproducibility; file remains manageable.
    out_json = args.outdir / 'v9a2_dynamic_horizon_controller_report.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out_doc = ROOT / 'docs/v9a2_dynamic_horizon_controller_result_2026-07-08.md'
    write_doc(report, out_doc)
    print(json.dumps({'ok': True, 'json': str(out_json), 'doc': str(out_doc), 'sweep_rows': applyback['sweep_rows_count'], 'best': applyback['sweep_top_by_AJ_RD_256'][0]['variant'], 'best_AJ_RD_256_delta': applyback['sweep_top_by_AJ_RD_256'][0]['delta_vs_native']['AJ_RD_256']}))


if __name__ == '__main__':
    main()
