#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
V9A2 = BASE / 'v9a2_anchor_uncertainty_reacquisition'
OUTDIR = BASE / 'v9a26_trackon2_internal_proxy'
JOINT = V9A2 / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
INTERNAL = OUTDIR / 'v9a26_trackon2_internal_proxy_features.npz'
FAMILY_5FOLD = OUTDIR / 'v9a26_trackon2_internal_feature_family_audit.json'
FROZEN = V9A2 / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a26_trackon2_internal_logo_robustness_audit.json'
OUT_DOC = ROOT / 'docs/v9a26_trackon2_internal_logo_robustness_audit_result_2026-07-10.md'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (  # noqa: E402
    CANDIDATE,
    NATIVE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import (  # noqa: E402
    apply_touched_mask,
    per_video_delta,
    summarize_pv,
)
from scripts.v9a1_controller_calibration_aware_prototype import choose_threshold_by_metric  # noqa: E402
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint  # noqa: E402
from scripts.v9a25_eval_dinov3_identity_features import binary_metrics, paired_summary  # noqa: E402
from scripts.v9a26_trackon2_internal_feature_family_audit import feature_families  # noqa: E402

METRIC_KEYS = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']


def metric_delta(metric: dict, base: dict) -> dict:
    return {
        key: None if metric.get(key) is None or base.get(key) is None else float(metric[key] - base[key])
        for key in METRIC_KEYS
    }


def extension_stats(data: dict, accept: np.ndarray) -> dict[str, int]:
    ext = data['is_w16_extension'].astype(bool)
    accept = np.asarray(accept, dtype=bool) & ext
    return {
        'accepted_extension': int(np.sum(accept)),
        'accepted_ext_good': int(np.sum(accept & data['y_candidate_good'].astype(bool))),
        'accepted_ext_bad': int(np.sum(accept & data['y_candidate_bad'].astype(bool))),
        'accepted_ext_false_visible': int(np.sum(accept & data['y_false_visible'].astype(bool))),
        'accepted_ext_worse': int(np.sum(accept & data['y_candidate_worse_px'].astype(bool))),
    }


def evaluate(
    name: str,
    data: dict,
    native: dict,
    cand_by: dict,
    accept: np.ndarray,
    native_metric: dict,
    native_pv: list[dict],
) -> dict[str, Any]:
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], np.asarray(accept, dtype=bool))
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    return {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'extension_stats': extension_stats(data, accept),
        'per_video_summary': summarize_pv(pv_rows),
        'per_video_rows': pv_rows,
    }


def logo_logreg_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    score = np.full(len(y), np.nan, dtype=float)
    folds = []
    for fold, (tr, va) in enumerate(LeaveOneGroupOut().split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            raise RuntimeError(f'LOGO training fold {fold} has one class')
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced'),
        )
        model.fit(X[tr], y[tr])
        pred = model.predict_proba(X[va])[:, 1]
        score[va] = pred
        row = {
            'fold': int(fold),
            'heldout_video': str(groups[va][0]),
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'positive_rate_val': float(np.mean(y[va])),
        }
        try:
            row.update(binary_metrics(y[va], pred))
        except ValueError:
            row.update({'ap': None, 'auc': None, 'brier': None})
        folds.append(row)
    if not np.all(np.isfinite(score)):
        raise RuntimeError('non-finite LOGO scores')
    overall = binary_metrics(y, score)
    overall.update({
        'threshold_f1': float(choose_threshold_by_metric(y, score, metric='f1')),
        'positive_rate': float(np.mean(y)),
        'n_rows': int(len(y)),
        'n_features': int(X.shape[1]),
        'n_logo_folds': int(len(folds)),
    })
    return {'scores': score, 'overall': overall, 'folds': folds}


def fmt(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return '—'
    return f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'


def main() -> None:
    data = load_joint(JOINT)
    iz = np.load(INTERNAL, allow_pickle=True)
    ext_indices = np.asarray(iz['joint_indices'], dtype=np.int64)
    expected = np.where(data['is_w16_extension'].astype(bool))[0]
    if not np.array_equal(ext_indices, expected):
        raise RuntimeError('internal feature joint-index alignment mismatch')
    if not np.array_equal(iz['row_key_json'], data['row_key_json'][expected]):
        raise RuntimeError('internal feature row-key alignment mismatch')

    X_full = iz['X_internal'].astype(np.float32)
    names_full = [str(x) for x in iz['feature_names'].tolist()]
    variable = np.std(X_full, axis=0) > 1e-10
    X = X_full[:, variable]
    names = [name for name, keep in zip(names_full, variable) if keep]
    families = feature_families(names)
    y = data['y_candidate_good'][ext_indices].astype(int)
    groups = data['groups'][ext_indices]

    trained = {}
    for family, columns in families.items():
        fit = logo_logreg_oof(X[:, columns], y, groups)
        trained[family] = {
            'feature_names': [names[i] for i in columns],
            **fit,
        }

    frozen_report = json.loads(FROZEN.read_text())
    frozen_score = None
    for block in frozen_report['training']:
        for row in block['overall']:
            if row['feature_set'] == 'all' and row['model'] == 'logreg':
                frozen_score = np.asarray(block['full_scores']['all_logreg'], dtype=float)
                break
    if frozen_score is None:
        raise RuntimeError('frozen score missing')

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    candidate = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, candidate)
    if not ok:
        raise RuntimeError(info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))

    frozen = evaluate(
        'frozen_v9a2_fixed0.05',
        data,
        native,
        cand_by,
        event_accept_mask(data, frozen_score, 0.05, 'event_max'),
        native_metric,
        native_pv,
    )

    rows = []
    paired = {}
    for offset, (family, fit) in enumerate(trained.items()):
        full_score = np.zeros(len(data['metas']), dtype=float)
        full_score[ext_indices] = fit['scores']
        threshold = float(fit['overall']['threshold_f1'])
        name = f'v9a26_logo_{family}_logreg_event_max_oof_f1'
        result = evaluate(
            name,
            data,
            native,
            cand_by,
            event_accept_mask(data, full_score, threshold, 'event_max'),
            native_metric,
            native_pv,
        )
        rows.append({
            'family': family,
            'threshold': threshold,
            'classifier': fit['overall'],
            'result': result,
        })
        paired[family] = paired_summary(result, frozen, seed=20260730 + offset)

    fivefold = json.loads(FAMILY_5FOLD.read_text())
    fivefold_by_family = {item['family']: item for item in fivefold['trajectory_rows']}
    comparisons = {}
    for item in rows:
        old = fivefold_by_family[item['family']]
        comparisons[item['family']] = {
            'fivefold_ap': float(old['classifier']['ap']),
            'logo_ap': float(item['classifier']['ap']),
            'ap_difference': float(item['classifier']['ap'] - old['classifier']['ap']),
            'fivefold_AJ_RD_256_delta': float(old['result']['delta_vs_native']['AJ_RD_256']),
            'logo_AJ_RD_256_delta': float(item['result']['delta_vs_native']['AJ_RD_256']),
            'trajectory_difference': float(
                item['result']['delta_vs_native']['AJ_RD_256']
                - old['result']['delta_vs_native']['AJ_RD_256']
            ),
        }

    robust_winners = []
    for item in rows:
        stats = paired[item['family']]
        if (
            item['result']['metric']['AJ_RD_256'] > frozen['metric']['AJ_RD_256']
            and stats['bootstrap_95_ci_mean'][0] > 0.0
        ):
            robust_winners.append(item['family'])

    best_classifier = max(rows, key=lambda item: item['classifier']['ap'])
    best_trajectory = max(rows, key=lambda item: item['result']['delta_vs_native']['AJ_RD_256'])
    memory = next(item for item in rows if item['family'] == 'memory_consistency')
    memory_paired = paired['memory_consistency']
    if robust_winners:
        status = 'LOGO_CONFIRMS_ROBUST_INTERNAL_FAMILY_GAIN'
        interpretation = (
            f'LOGO robust winners: {robust_winners}. Freeze and independently validate before V9-A3.'
        )
    else:
        status = 'LOGO_CONFIRMS_NO_ROBUST_SELECTOR_GAIN'
        interpretation = (
            'Leave-one-video-out OOF does not produce any internal feature family whose paired mean 95% CI '
            'has a positive lower bound over frozen V9-A2. The promising 5-fold memory-consistency aggregate '
            'result is not sufficient to reopen selector tuning. Proceed to V9-A3 candidate-generation/reacquisition.'
        )

    report = {
        'date': '2026-07-10',
        'script': 'scripts/v9a26_trackon2_internal_logo_robustness_audit.py',
        'protocol': {
            'cv': 'LeaveOneGroupOut by video (20 folds)',
            'model': 'class-balanced logistic regression',
            'apply_back': 'preserve W8 common; event_max; global LOGO-OOF-F1 threshold',
            'families': 'all 13 predeclared TrackOn2 semantic feature families',
            'dense_threshold_search': False,
        },
        'fivefold_source': str(FAMILY_5FOLD),
        'family_models': {
            family: {
                'feature_names': fit['feature_names'],
                'classifier': fit['overall'],
                'folds': fit['folds'],
            }
            for family, fit in trained.items()
        },
        'frozen_baseline': frozen,
        'trajectory_rows': rows,
        'paired_vs_frozen': paired,
        'fivefold_vs_logo': comparisons,
        'decision': {
            'status': status,
            'robust_winners': robust_winners,
            'best_classifier_family': best_classifier['family'],
            'best_trajectory_family': best_trajectory['family'],
            'memory_consistency': {
                'AJ_RD_256_delta': memory['result']['delta_vs_native']['AJ_RD_256'],
                'aggregate_vs_frozen': float(
                    memory['result']['metric']['AJ_RD_256'] - frozen['metric']['AJ_RD_256']
                ),
                'paired_mean_vs_frozen': memory_paired['mean_diff'],
                'paired_ci_vs_frozen': memory_paired['bootstrap_95_ci_mean'],
                'better_worse_equal': [
                    memory_paired['better'], memory_paired['worse'], memory_paired['equal']
                ],
            },
            'interpretation': interpretation,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# V9-A2.6 TrackOn2 Internal LOGO Robustness Audit',
        '',
        'Date: 2026-07-10',
        '',
        '## Protocol',
        '',
        '```text',
        'Leave-one-video-out OOF over all 20 extension-bearing videos.',
        'All 13 predeclared internal feature families; class-balanced logistic regression.',
        'event_max + global LOGO-OOF-F1; W8 common preserved; no trajectory threshold sweep.',
        '```',
        '',
        '## LOGO results',
        '',
        '| Family | AP | AUC | Threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Aggregate vs frozen | Paired mean | 95% CI | Better/Worse/Equal |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for item in rows:
        family = item['family']
        classifier = item['classifier']
        result = item['result']
        delta = result['delta_vs_native']
        stats = paired[family]
        aggregate = result['metric']['AJ_RD_256'] - frozen['metric']['AJ_RD_256']
        ci = stats['bootstrap_95_ci_mean']
        lines.append(
            f"| {family} | {fmt(classifier['ap'])} | {fmt(classifier['auc'])} | "
            f"{item['threshold']:.6f} | {fmt(delta['AJ'], signed=True)} | "
            f"{fmt(delta['OA'], signed=True)} | {fmt(delta['AJ_RD_256'], signed=True)} | "
            f"{aggregate:+.6f} | {stats['mean_diff']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{stats['better']}/{stats['worse']}/{stats['equal']} |"
        )

    lines += [
        '',
        '## Five-fold versus LOGO stability',
        '',
        '| Family | 5-fold AP | LOGO AP | 5-fold AJ_RD_256 Δ | LOGO AJ_RD_256 Δ |',
        '|---|---:|---:|---:|---:|',
    ]
    for family, comparison in comparisons.items():
        lines.append(
            f"| {family} | {comparison['fivefold_ap']:.4f} | {comparison['logo_ap']:.4f} | "
            f"{comparison['fivefold_AJ_RD_256_delta']:+.4f} | "
            f"{comparison['logo_AJ_RD_256_delta']:+.4f} |"
        )

    lines += [
        '',
        '## Decision',
        '',
        f'**{status}**',
        '',
        interpretation,
    ]
    OUT_DOC.write_text('\n'.join(lines) + '\n')

    print(json.dumps({
        'ok': True,
        'json': str(OUT_JSON),
        'doc': str(OUT_DOC),
        'best_classifier_family': best_classifier['family'],
        'best_classifier_ap': best_classifier['classifier']['ap'],
        'best_trajectory_family': best_trajectory['family'],
        'best_trajectory_AJ_RD_256_delta': best_trajectory['result']['delta_vs_native']['AJ_RD_256'],
        'memory_consistency': report['decision']['memory_consistency'],
        'robust_winners': robust_winners,
        'decision': status,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
