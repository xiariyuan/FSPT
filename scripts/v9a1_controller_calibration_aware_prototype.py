#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a1_controller_calibration_aware_prototype'
DEFAULT_DATASET = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'


def video_from_meta(meta: dict) -> str:
    if meta.get('video_name') is not None:
        return str(meta['video_name'])
    if meta.get('video') is not None:
        return str(meta['video'])
    if meta.get('video_id') is not None:
        return str(meta['video_id'])
    if meta.get('video_idx') is not None:
        return f"video_{meta['video_idx']}"
    return 'unknown_video'


def build_video_groups(metas: list[dict]) -> np.ndarray:
    return np.array([video_from_meta(m) for m in metas], dtype=object)


def risk_coverage_curve(scores: np.ndarray, losses: np.ndarray, coverages: Iterable[float]) -> dict:
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if scores.shape[0] != losses.shape[0]:
        raise ValueError('scores and losses must have the same length')
    if scores.shape[0] == 0:
        raise ValueError('scores must be non-empty')
    order = np.argsort(-scores)
    losses_sorted = losses[order]
    n = int(losses_sorted.shape[0])
    out = {}
    for cov in coverages:
        cov = float(cov)
        if not (0 < cov <= 1):
            raise ValueError(f'coverage must be in (0, 1], got {cov}')
        k = max(1, int(np.ceil(cov * n)))
        out[f'coverage_{cov:.2f}_mean_loss'] = float(np.mean(losses_sorted[:k]))
    return out


def choose_threshold_by_metric(y_true: np.ndarray, scores: np.ndarray, metric: str = 'f1') -> float:
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError('y_true and scores must have the same length')
    if scores.shape[0] == 0:
        raise ValueError('scores must be non-empty')
    thresholds = np.unique(scores)
    best_thr = float(thresholds[0])
    best_val = -1.0
    for thr in thresholds:
        pred = scores >= thr
        tp = float(np.sum(pred & y_true))
        fp = float(np.sum(pred & ~y_true))
        fn = float(np.sum(~pred & y_true))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if metric == 'precision':
            val = precision
        elif metric == 'recall':
            val = recall
        elif metric == 'f1':
            val = 2 * precision * recall / max(precision + recall, 1e-12)
        else:
            raise ValueError(f'unknown metric: {metric}')
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    return best_thr


def load_v8c02_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    if 'meta_json' in data:
        data['metas'] = [json.loads(str(m)) for m in data['meta_json'].tolist()]
    else:
        n = int(np.asarray(next(iter(data.values()))).shape[0])
        data['metas'] = [{} for _ in range(n)]
    return data


def select_feature_matrix(data: dict) -> tuple[np.ndarray, list[str]]:
    if 'X' in data:
        X = np.asarray(data['X'], dtype=np.float32)
        if 'feature_names' in data:
            names = [str(x) for x in np.asarray(data['feature_names']).tolist()]
        else:
            names = [f'f{i}' for i in range(X.shape[1])]
        return X, names
    keys = sorted(k for k, v in data.items() if k.startswith('feature_') and np.asarray(v).ndim == 1)
    if not keys:
        raise KeyError('No X matrix or one-dimensional feature_* arrays found')
    X = np.stack([np.asarray(data[k], dtype=np.float32) for k in keys], axis=1)
    return X, keys


def label_array(data: dict, preferred: list[str]) -> tuple[np.ndarray, str]:
    for key in preferred:
        if key in data:
            return np.asarray(data[key]).astype(int), key
    raise KeyError(f'None of these label keys were found: {preferred}')


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        'ap': float(average_precision_score(y_true, scores)),
        'brier': float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
    }
    try:
        out['auc'] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out['auc'] = None
    return out


def train_groupheldout_models(data: dict, label_keys: list[str]) -> dict:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, feature_names = select_feature_matrix(data)
    y, label_name = label_array(data, label_keys)
    groups = build_video_groups(data['metas'])
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError('Need at least two video groups for video-heldout validation')

    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        'hgb': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
        'extratrees': ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, class_weight='balanced', random_state=0, n_jobs=-1),
    }
    folds = []
    oof_scores = {name: np.full(len(y), np.nan, dtype=float) for name in models}
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        for name, model in models.items():
            clf = model.fit(X[tr], y[tr])
            if hasattr(clf, 'predict_proba'):
                score = clf.predict_proba(X[va])[:, 1]
            else:
                score = clf.decision_function(X[va])
                score = 1 / (1 + np.exp(-score))
            oof_scores[name][va] = score
            row = {
                'fold': int(fold),
                'model': name,
                'label': label_name,
                'n_train': int(len(tr)),
                'n_val': int(len(va)),
                'val_videos': sorted(set(groups[va].tolist())),
                'positive_rate_val': float(np.mean(y[va])),
                'threshold_f1': choose_threshold_by_metric(y[va], score, metric='f1'),
            }
            row.update(_binary_metrics(y[va], score))
            row.update(risk_coverage_curve(score, 1.0 - y[va].astype(float), [0.25, 0.5, 0.75, 1.0]))
            folds.append(row)
    overall = []
    for name, score in oof_scores.items():
        row = {'model': name, 'label': label_name, 'n': int(len(y)), 'positive_rate': float(np.mean(y))}
        row.update(_binary_metrics(y, score))
        row.update(risk_coverage_curve(score, 1.0 - y.astype(float), [0.25, 0.5, 0.75, 1.0]))
        row['threshold_f1_oof'] = choose_threshold_by_metric(y, score, metric='f1')
        overall.append(row)
    return {
        'dataset_n': int(len(y)),
        'videos': sorted(set(groups.tolist())),
        'n_videos': int(len(unique_groups)),
        'feature_names': feature_names,
        'label': label_name,
        'folds': folds,
        'overall_oof': overall,
        'oof_scores': {k: v.tolist() for k, v in oof_scores.items()},
    }


def _metric_delta(metric: dict, native_metric: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: (None if metric.get(k) is None or native_metric.get(k) is None else float(metric[k] - native_metric[k])) for k in keys}


def evaluate_oof_acceptance(dataset_path: Path, tabular_report: dict) -> dict:
    import torch
    from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
        NATIVE, CANDIDATE, align_candidate, clone_records, standard_and_ajrd,
    )
    from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import (
        apply_touched_mask, label_stats, per_video_delta, summarize_pv, load_dataset as load_apply_dataset,
    )

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')
    ds = load_apply_dataset(dataset_path)
    metas = ds['metas']
    labels = ds['labels']
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))

    rows = []
    n = len(metas)
    baselines = {
        'native_reject_all': np.zeros(n, dtype=bool),
        'cvrrm_accept_all_touched': np.ones(n, dtype=bool),
    }
    for name, accept in baselines.items():
        recs, stats = apply_touched_mask(native, cand_by, metas, accept)
        metric, pv = standard_and_ajrd(recs)
        pv_rows = per_video_delta(native_pv, pv)
        rows.append({
            'variant': name,
            'threshold': None,
            'metric': metric,
            'delta_vs_native': _metric_delta(metric, native_metric),
            'apply_stats': stats,
            'label_stats_on_accepted': label_stats(labels, accept),
            'per_video_summary': summarize_pv(pv_rows),
        })

    thresholds = {r['model']: float(r['threshold_f1_oof']) for r in tabular_report['overall_oof']}
    for model, score_list in tabular_report['oof_scores'].items():
        scores = np.asarray(score_list, dtype=float)
        thr = thresholds[model]
        accept = scores >= thr
        recs, stats = apply_touched_mask(native, cand_by, metas, accept)
        metric, pv = standard_and_ajrd(recs)
        pv_rows = per_video_delta(native_pv, pv)
        rows.append({
            'variant': f'v9a1_{model}_oof_f1',
            'threshold': thr,
            'metric': metric,
            'delta_vs_native': _metric_delta(metric, native_metric),
            'apply_stats': stats,
            'label_stats_on_accepted': label_stats(labels, accept),
            'per_video_summary': summarize_pv(pv_rows),
        })

    return {
        'native_metric': native_metric,
        'rows': rows,
        'best_by_AJ_RD_256': sorted(rows, key=lambda r: -999 if r['delta_vs_native'].get('AJ_RD_256') is None else r['delta_vs_native']['AJ_RD_256'], reverse=True),
    }


def _fmt(x) -> str:
    if x is None:
        return ''
    return f'{float(x):+.4f}'


def write_markdown_report(report: dict, path: Path) -> None:
    lines = ['# V9-A1 Controller Calibration-Aware Prototype Result', '']
    lines += ['## Tabular group-heldout OOF summary', '']
    lines += ['| Model | AP | AUC | Brier | Risk@50 loss | Threshold |', '|---|---:|---:|---:|---:|---:|']
    for row in report['tabular']['overall_oof']:
        auc = '' if row['auc'] is None else f"{row['auc']:.4f}"
        lines.append(f"| {row['model']} | {row['ap']:.4f} | {auc} | {row['brier']:.4f} | {row['coverage_0.50_mean_loss']:.4f} | {row['threshold_f1_oof']:.4f} |")
    lines += ['', '## Trajectory apply-back summary', '']
    lines += ['| Variant | AJ Δ | OA Δ | δavg Δ | δ4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in report['trajectory']['rows']:
        d = row['delta_vs_native']; stats = row['apply_stats']; pv = row['per_video_summary']
        pnz = f"{pv['positive']}/{pv['negative']}/{pv['zero']}"
        lines.append(f"| {row['variant']} | {_fmt(d['AJ'])} | {_fmt(d['OA'])} | {_fmt(d['delta_avg'])} | {_fmt(d['delta_4px'])} | {_fmt(d['AJ_RD'])} | {_fmt(d['AJ_RD_256'])} | {stats.get('accepted_frames', 0)} | {pnz} |")
    lines += ['', '## Decision', '']
    best = report['trajectory']['best_by_AJ_RD_256'][0]
    default = next(r for r in report['trajectory']['rows'] if r['variant'] == 'cvrrm_accept_all_touched')
    best_gain = best['delta_vs_native']['AJ_RD_256']
    default_gain = default['delta_vs_native']['AJ_RD_256']
    if best['variant'] != 'cvrrm_accept_all_touched' and best_gain is not None and default_gain is not None and best_gain > default_gain:
        lines.append('V9-A1 has a positive apply-back signal over CVRRM accept-all on AJ_RD_256. Continue to stricter heldout thresholding and calibration-aware controller refinement.')
    else:
        lines.append('V9-A1 tabular models are learnable, but the current OOF-F1 apply-back does not beat the CVRRM accept-all baseline on AJ_RD_256. Do not scale to full ReEntryBeliefTrack from this result alone.')
    path.write_text('\n'.join(lines))


def strip_arrays_for_json(report: dict) -> dict:
    out = dict(report)
    tab = dict(out['tabular'])
    tab.pop('oof_scores', None)
    out['tabular'] = tab
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='V9-A1 controller-only calibration-aware prototype')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--tabular-only', action='store_true')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--label-keys', nargs='+', default=['y_candidate_good', 'candidate_good', 'good_16px'])
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        print(json.dumps({'ok': True, 'outdir': str(OUTDIR)}))
        return
    data = load_v8c02_dataset(args.dataset)
    tabular = train_groupheldout_models(data, args.label_keys)
    if args.tabular_only:
        path = OUTDIR / 'v9a1_tabular_cv_report.json'
        path.write_text(json.dumps(strip_arrays_for_json({'tabular': tabular})['tabular'], indent=2, ensure_ascii=False))
        print(json.dumps({'ok': True, 'report': str(path), 'label': tabular['label'], 'n': tabular['dataset_n']}))
        return
    trajectory = evaluate_oof_acceptance(args.dataset, tabular)
    report = {'script': 'scripts/v9a1_controller_calibration_aware_prototype.py', 'dataset': str(args.dataset), 'tabular': tabular, 'trajectory': trajectory}
    json_path = OUTDIR / 'v9a1_report.json'
    json_path.write_text(json.dumps(strip_arrays_for_json(report), indent=2, ensure_ascii=False))
    md_path = ROOT / 'docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md'
    write_markdown_report(report, md_path)
    with (ROOT / 'CURRENT_MAINLINE.md').open('a') as f:
        f.write('\n\n## V9-A1 controller calibration-aware prototype result\n\n')
        f.write('Artifacts:\n\n```text\noutputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_report.json\ndocs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md\n```\n\n')
        f.write('Decision:\n\n```text\nSee V9-A1 result doc for pass/fail judgement.\n```\n')
    print(json.dumps({'ok': True, 'json': str(json_path), 'doc': str(md_path)}))


if __name__ == '__main__':
    main()
