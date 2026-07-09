#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IN_ROWS = Path('outputs/paper_discovery_2026-06-27/b2_rv_feature_audit_dev/trigger_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/b2_rv_dev_cv')
FEATURES = [
    'trigger_t_norm',
    'query_age_norm',
    'base_invis_run',
    'override_persist_len_cap16',
    'override_visible_frac_next16',
    'base_visible_frac_next16',
    'vis_agreement_frac_next16',
    'override_visibility_transitions_next16',
    'base_visibility_transitions_next16',
    'base_override_dist_t',
    'base_override_dist_mean_next4',
    'base_override_dist_mean_next16',
    'base_override_dist_max_next16',
    'override_speed_mean_next4',
    'override_speed_mean_next16',
    'base_speed_prev4',
    'n_trigger_windows',
]
POLICIES = [
    ('score_safe90', 'score', 0.90),
    ('score_safe85', 'score', 0.85),
    ('score_safe80', 'score', 0.80),
    ('harm_safe90', 'anti_harm', 0.90),
    ('harm_safe85', 'anti_harm', 0.85),
    ('harm_safe80', 'anti_harm', 0.80),
]


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    with IN_ROWS.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_float(v: Any) -> float:
    if v is None:
        return np.nan
    try:
        x = float(v)
    except Exception:
        return np.nan
    if not math.isfinite(x):
        return np.nan
    return x


def matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([[to_float(r.get(f)) for f in FEATURES] for r in rows], dtype=np.float32)


def labels(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    if key == 'true_trigger':
        return np.asarray([bool(r.get('has_gt_reentry')) for r in rows], dtype=bool)
    return np.asarray([bool(r.get(key)) for r in rows], dtype=bool)


def values(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def make_model() -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])


def safe_auc(y: np.ndarray, s: np.ndarray):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y.astype(int), s))


def eval_policy(rows: List[Dict[str, Any]], accept: np.ndarray) -> Dict[str, Any]:
    delta_full = values(rows, 'delta_full')
    delta_post = values(rows, 'delta_post')
    harmful = labels(rows, 'harmful_full')
    helpful = labels(rows, 'helpful_post')
    true_trig = labels(rows, 'true_trigger')
    accept = accept.astype(bool)
    n = len(rows)
    baseline_full_sum = float(np.sum(delta_full))
    baseline_post_sum = float(np.sum(delta_post))
    full_sum = float(np.sum(delta_full[accept]))
    post_sum = float(np.sum(delta_post[accept]))
    out = {
        'n': int(n),
        'accepted': int(np.sum(accept)),
        'accept_rate': round(float(np.mean(accept)), 6) if n else None,
        'full_sum': round(full_sum, 6),
        'post_sum': round(post_sum, 6),
        'full_mean_all_triggers': round(full_sum / max(n, 1), 6),
        'post_mean_all_triggers': round(post_sum / max(n, 1), 6),
        'baseline_full_sum': round(baseline_full_sum, 6),
        'baseline_post_sum': round(baseline_post_sum, 6),
        'full_delta_vs_accept_all': round(full_sum - baseline_full_sum, 6),
        'post_delta_vs_accept_all': round(post_sum - baseline_post_sum, 6),
        'post_retention': round(post_sum / baseline_post_sum, 6) if abs(baseline_post_sum) > 1e-9 else None,
        'full_retention': round(full_sum / baseline_full_sum, 6) if abs(baseline_full_sum) > 1e-9 else None,
        'harmful_total': int(np.sum(harmful)),
        'harmful_accepted': int(np.sum(harmful & accept)),
        'harmful_rejected': int(np.sum(harmful & (~accept))),
        'harmful_reject_rate': round(float(np.sum(harmful & (~accept)) / max(np.sum(harmful), 1)), 6),
        'helpful_total': int(np.sum(helpful)),
        'helpful_accepted': int(np.sum(helpful & accept)),
        'helpful_retention_rate': round(float(np.sum(helpful & accept) / max(np.sum(helpful), 1)), 6),
        'true_total': int(np.sum(true_trig)),
        'true_accepted': int(np.sum(true_trig & accept)),
        'true_retention_rate': round(float(np.sum(true_trig & accept) / max(np.sum(true_trig), 1)), 6),
    }
    return out


def select_threshold(train_rows: List[Dict[str, Any]], score: np.ndarray, post_retention_min: float) -> Tuple[float, Dict[str, Any]]:
    # Candidate thresholds include all quantiles and extremes.
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        return 0.0, eval_policy(train_rows, np.ones(len(train_rows), dtype=bool))
    qs = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))
    candidates = [float(np.min(finite) - 1e-6)] + [float(x) for x in qs] + [float(np.max(finite) + 1e-6)]
    baseline = eval_policy(train_rows, np.ones(len(train_rows), dtype=bool))
    best_thr = candidates[0]
    best = None
    for thr in candidates:
        accept = score >= thr
        met = eval_policy(train_rows, accept)
        post_ret = met['post_retention'] if met['post_retention'] is not None else -999
        if post_ret < post_retention_min:
            continue
        if best is None:
            best_thr, best = thr, met
            continue
        # Main objective: maximize full_sum, then harmful reject, then accept fewer.
        key = (met['full_sum'], met['harmful_reject_rate'], -met['accept_rate'])
        best_key = (best['full_sum'], best['harmful_reject_rate'], -best['accept_rate'])
        if key > best_key:
            best_thr, best = thr, met
    if best is None:
        # fallback: accept all
        return float(np.min(finite) - 1e-6), baseline
    return float(best_thr), best


def fold_group_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ds = {}
    for r in rows:
        ds.setdefault(r['dataset'], 0)
        ds[r['dataset']] += 1
    return ds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    X = matrix(rows)
    groups = np.asarray([f"{r['dataset']}::{r['video_id']}" for r in rows])
    y_harm = labels(rows, 'harmful_full')
    y_help = labels(rows, 'helpful_post')
    y_true = labels(rows, 'true_trigger')

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = {
        'p_harm': np.zeros(len(rows), dtype=np.float64),
        'p_help': np.zeros(len(rows), dtype=np.float64),
        'score': np.zeros(len(rows), dtype=np.float64),
        'anti_harm': np.zeros(len(rows), dtype=np.float64),
    }
    folds = []
    policy_accept = {name: np.zeros(len(rows), dtype=bool) for name, _, _ in POLICIES}

    for fold_id, (tr, te) in enumerate(gkf.split(X, y_harm, groups), start=1):
        train_rows = [rows[i] for i in tr]
        test_rows = [rows[i] for i in te]
        harm_model = make_model()
        help_model = make_model()
        harm_model.fit(X[tr], y_harm[tr].astype(int))
        help_model.fit(X[tr], y_help[tr].astype(int))
        p_harm_train = harm_model.predict_proba(X[tr])[:, 1]
        p_help_train = help_model.predict_proba(X[tr])[:, 1]
        score_train = p_help_train - p_harm_train
        anti_harm_train = -p_harm_train
        p_harm_test = harm_model.predict_proba(X[te])[:, 1]
        p_help_test = help_model.predict_proba(X[te])[:, 1]
        score_test = p_help_test - p_harm_test
        anti_harm_test = -p_harm_test
        oof['p_harm'][te] = p_harm_test
        oof['p_help'][te] = p_help_test
        oof['score'][te] = score_test
        oof['anti_harm'][te] = anti_harm_test
        fold_info = {
            'fold': fold_id,
            'train_n': int(len(tr)),
            'test_n': int(len(te)),
            'test_groups': sorted(set(groups[te].tolist())),
            'test_dataset_counts': fold_group_summary(test_rows),
            'auc_test_harmful_full': safe_auc(y_harm[te], p_harm_test),
            'auc_test_helpful_post': safe_auc(y_help[te], p_help_test),
            'auc_test_true_trigger_score': safe_auc(y_true[te], score_test),
            'baseline_accept_all_test': eval_policy(test_rows, np.ones(len(test_rows), dtype=bool)),
            'policies': {},
        }
        for pol_name, score_kind, post_ret in POLICIES:
            train_score = score_train if score_kind == 'score' else anti_harm_train
            test_score = score_test if score_kind == 'score' else anti_harm_test
            thr, train_metric = select_threshold(train_rows, train_score, post_ret)
            accept_test = test_score >= thr
            policy_accept[pol_name][te] = accept_test
            fold_info['policies'][pol_name] = {
                'score_kind': score_kind,
                'post_retention_min_train': post_ret,
                'threshold': round(float(thr), 6),
                'train_metric': train_metric,
                'test_metric': eval_policy(test_rows, accept_test),
            }
        folds.append(fold_info)

    # OOF aggregate policies
    all_accept = np.ones(len(rows), dtype=bool)
    aggregate = {
        'baseline_accept_all': eval_policy(rows, all_accept),
        'auc_oof_harmful_full': safe_auc(y_harm, oof['p_harm']),
        'auc_oof_helpful_post': safe_auc(y_help, oof['p_help']),
        'auc_oof_true_trigger_score': safe_auc(y_true, oof['score']),
        'policies': {name: eval_policy(rows, acc) for name, acc in policy_accept.items()},
    }
    # Dataset-specific policy metrics
    for pol_name, acc in policy_accept.items():
        aggregate['policies'][pol_name]['by_dataset'] = {}
        for ds in sorted(set(r['dataset'] for r in rows)):
            idx = np.asarray([i for i, r in enumerate(rows) if r['dataset'] == ds], dtype=int)
            aggregate['policies'][pol_name]['by_dataset'][ds] = eval_policy([rows[i] for i in idx], acc[idx])

    summary = {
        'protocol': 'Development-only B2-RV grouped cross-validation on DAVIS + RGB dev first-10. No RGB fresh20-49 is used.',
        'features': FEATURES,
        'rows_path': str(IN_ROWS),
        'n_rows': len(rows),
        'n_groups': int(len(np.unique(groups))),
        'folds': folds,
        'aggregate': aggregate,
        'oof_scores_path': str(OUT / 'oof_scores.jsonl'),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'oof_scores.jsonl').open('w') as f:
        for i, r in enumerate(rows):
            rr = {
                'dataset': r['dataset'],
                'video_id': r['video_id'],
                'query_idx': r['query_idx'],
                'class': r['class'],
                'delta_full': r['delta_full'],
                'delta_post': r['delta_post'],
                'harmful_full': r['harmful_full'],
                'helpful_post': r['helpful_post'],
                'has_gt_reentry': r['has_gt_reentry'],
                'p_harm': round(float(oof['p_harm'][i]), 6),
                'p_help': round(float(oof['p_help'][i]), 6),
                'score': round(float(oof['score'][i]), 6),
                'anti_harm': round(float(oof['anti_harm'][i]), 6),
            }
            for pol_name, acc in policy_accept.items():
                rr[f'accept_{pol_name}'] = bool(acc[i])
            f.write(json.dumps(rr, ensure_ascii=False) + '\n')
    print(json.dumps({
        'protocol': summary['protocol'],
        'n_rows': summary['n_rows'],
        'n_groups': summary['n_groups'],
        'aggregate': aggregate,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
