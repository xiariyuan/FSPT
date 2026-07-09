#!/usr/bin/env python3
"""Train/evaluate V8-C0.3 fine-risk verifier on V8-C0.2 touched-frame dataset.

Goal: check whether causal per-frame features can predict candidate-visible frame
risk/usefulness under video-level leave-one-out validation.

This script does not yet apply verifier to trajectories. It is a learnability
and threshold-feasibility audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/gemini/code/FSPT')
DEFAULT_DATASET = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c03_fine_risk_verifier'


def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y, score):
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def loov_classifier(X, y, groups, model_name: str):
    oof = np.zeros(len(y), np.float32)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(np.mean(y[tr])) if len(y[tr]) else 0.0
            continue
        if model_name == 'logreg':
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear'))
        elif model_name == 'extratrees':
            clf = ExtraTreesClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, class_weight='balanced', random_state=7, n_jobs=-1)
        elif model_name == 'rf':
            clf = RandomForestClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=3, class_weight='balanced', random_state=7, n_jobs=-1)
        else:
            raise ValueError(model_name)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def loov_regressor(X, y, groups, model_name: str):
    oof = np.zeros(len(y), np.float32)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if model_name == 'ridge':
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif model_name == 'extratrees':
            reg = ExtraTreesRegressor(n_estimators=400, max_features='sqrt', min_samples_leaf=3, random_state=7, n_jobs=-1)
        else:
            raise ValueError(model_name)
        reg.fit(X[tr], y[tr])
        oof[te] = reg.predict(X[te])
    return oof


def threshold_table(y_good, y_bad, score_good, score_bad=None):
    """Build threshold table for accepting frames.

    If score_bad is provided, decision score = score_good - score_bad.
    Otherwise decision score = score_good.
    """
    if score_bad is None:
        decision = score_good.copy()
    else:
        decision = score_good - score_bad
    qs = np.unique(np.quantile(decision, np.linspace(0, 1, 31)))
    rows = []
    for th in qs:
        keep = decision >= th
        n = int(keep.sum())
        if n == 0:
            continue
        rows.append({
            'threshold': float(th),
            'accept': n,
            'accept_rate': float(n / len(decision)),
            'good_rate': float(np.mean(y_good[keep])) if n else 0.0,
            'bad_rate': float(np.mean(y_bad[keep])) if n else 0.0,
            'good_count': int(np.sum(y_good[keep])),
            'bad_count': int(np.sum(y_bad[keep])),
            'net_good_minus_bad': int(np.sum(y_good[keep]) - np.sum(y_bad[keep])),
        })
    rows = sorted(rows, key=lambda r: (r['net_good_minus_bad'], r['good_rate'], -r['bad_rate']), reverse=True)
    return rows[:20]


def per_video_rates(y_good, y_bad, score, groups):
    out = {}
    for g in sorted(set(groups.tolist())):
        idx = groups == g
        out[str(g)] = {
            'n': int(idx.sum()),
            'good_rate': float(np.mean(y_good[idx])) if idx.any() else 0.0,
            'bad_rate': float(np.mean(y_bad[idx])) if idx.any() else 0.0,
            'score_mean': float(np.mean(score[idx])) if idx.any() else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(DEFAULT_DATASET))
    ap.add_argument('--outdir', default=str(OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.dataset, allow_pickle=True)
    X = z['X'].astype(np.float32)
    feature_names = [str(x) for x in z['feature_names'].tolist()]
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    groups = np.asarray([m['video_id'] for m in metas], dtype=object)
    y_good = z['y_candidate_good'].astype(np.float32)
    y_bad = z['y_candidate_bad'].astype(np.float32)
    y_false = z['y_false_visible'].astype(np.float32)
    y_worse = z['y_candidate_worse_px'].astype(np.float32)
    y_better = z['y_candidate_better_px'].astype(np.float32)
    y_utility = z['y_utility'].astype(np.float32)
    # Bad variants.
    y_bad_strict = ((y_false > 0.5) | (y_worse > 0.5)).astype(np.float32)
    labels = {
        'good': y_good,
        'bad': y_bad,
        'bad_strict_false_or_worse': y_bad_strict,
        'false_visible': y_false,
        'candidate_worse_px': y_worse,
        'candidate_better_px': y_better,
    }
    results = {
        'script': 'scripts/train_cotracker3_online_v8c03_fine_risk_verifier.py',
        'dataset': str(args.dataset),
        'n_samples': int(len(X)),
        'feature_dim': int(X.shape[1]),
        'n_videos': int(len(set(groups.tolist()))),
        'base_rates': {k: float(np.mean(v)) for k, v in labels.items()},
        'classifiers': {},
        'regressors': {},
        'feature_names': feature_names,
    }
    # Classifiers.
    for lname, y in labels.items():
        y_int = y.astype(int)
        results['classifiers'][lname] = {}
        if len(np.unique(y_int)) < 2:
            continue
        for mname in ['logreg', 'extratrees', 'rf']:
            oof = loov_classifier(X, y_int, groups, mname)
            results['classifiers'][lname][mname] = {
                'ap': safe_ap(y_int, oof),
                'auc': safe_auc(y_int, oof),
                'oof_path': str(outdir / f'oof_{lname}_{mname}.npy'),
                'per_video_score': per_video_rates(y_good, y_bad_strict, oof, groups),
            }
            np.save(outdir / f'oof_{lname}_{mname}.npy', oof)
    # Regressors for utility.
    for mname in ['ridge', 'extratrees']:
        oof = loov_regressor(X, y_utility, groups, mname)
        finite = np.isfinite(y_utility)
        results['regressors'][mname] = {
            'mae': float(mean_absolute_error(y_utility[finite], oof[finite])),
            'r2': float(r2_score(y_utility[finite], oof[finite])) if finite.sum() > 1 else None,
            'ap_for_good_using_pred': safe_ap(y_good.astype(int), oof),
            'auc_for_good_using_pred': safe_auc(y_good.astype(int), oof),
            'threshold_table_good': threshold_table(y_good.astype(int), y_bad_strict.astype(int), oof),
            'oof_path': str(outdir / f'oof_utility_{mname}.npy'),
        }
        np.save(outdir / f'oof_utility_{mname}.npy', oof)
    # Combined decision using good - bad_strict for best available models.
    good_et = np.load(outdir / 'oof_good_extratrees.npy') if (outdir / 'oof_good_extratrees.npy').exists() else None
    bad_et = np.load(outdir / 'oof_bad_strict_false_or_worse_extratrees.npy') if (outdir / 'oof_bad_strict_false_or_worse_extratrees.npy').exists() else None
    if good_et is not None and bad_et is not None:
        results['combined_good_minus_bad_extratrees'] = {
            'threshold_table': threshold_table(y_good.astype(int), y_bad_strict.astype(int), good_et, bad_et),
            'ap_for_good': safe_ap(y_good.astype(int), good_et - bad_et),
            'auc_for_good': safe_auc(y_good.astype(int), good_et - bad_et),
            'ap_for_not_bad': safe_ap((1 - y_bad_strict).astype(int), -(bad_et - good_et)),
            'auc_for_not_bad': safe_auc((1 - y_bad_strict).astype(int), -(bad_et - good_et)),
        }
    out = outdir / 'v8c03_fine_risk_verifier_report.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    compact = {
        'out': str(out),
        'dataset': str(args.dataset),
        'n_samples': len(X),
        'base_rates': results['base_rates'],
        'classifier_summary': {
            lname: {m: {kk: (round(vv,4) if isinstance(vv,float) else vv) for kk, vv in md.items() if kk in ['ap','auc']} for m, md in models.items()}
            for lname, models in results['classifiers'].items()
        },
        'regressor_summary': {
            m: {kk: (round(vv,4) if isinstance(vv,float) else vv) for kk, vv in md.items() if kk in ['mae','r2','ap_for_good_using_pred','auc_for_good_using_pred']}
            for m, md in results['regressors'].items()
        },
        'combined_top': results.get('combined_good_minus_bad_extratrees', {}).get('threshold_table', [])[:8],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
