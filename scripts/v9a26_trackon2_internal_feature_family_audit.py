#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
FROZEN = V9A2 / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a26_trackon2_internal_feature_family_audit.json'
OUT_DOC = ROOT / 'docs/v9a26_trackon2_internal_feature_family_audit_result_2026-07-10.md'

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

METRIC_KEYS = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


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


def logreg_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    score = np.full(len(y), np.nan, dtype=float)
    folds = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced'),
        )
        model.fit(X[tr], y[tr])
        pred = model.predict_proba(X[va])[:, 1]
        score[va] = pred
        row = {
            'fold': int(fold),
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'val_videos': sorted(set(groups[va].tolist())),
            'positive_rate_val': float(np.mean(y[va])),
        }
        row.update(binary_metrics(y[va], pred))
        folds.append(row)
    if not np.all(np.isfinite(score)):
        raise RuntimeError('non-finite OOF scores')
    overall = binary_metrics(y, score)
    overall.update({
        'threshold_f1': float(choose_threshold_by_metric(y, score, metric='f1')),
        'positive_rate': float(np.mean(y)),
        'n_rows': int(len(y)),
        'n_features': int(X.shape[1]),
    })
    return {'scores': score, 'overall': overall, 'folds': folds}


def feature_families(names: list[str]) -> dict[str, list[int]]:
    def select(predicate) -> list[int]:
        return [i for i, name in enumerate(names) if predicate(name)]

    correlation_global = select(
        lambda n: n.startswith(('c1_', 'c2_'))
        and not any(token in n for token in ['old_candidate', 'native'])
    )
    correlation_alignment = select(
        lambda n: n.startswith(('c1_', 'c2_'))
        and any(token in n for token in ['old_candidate', 'native'])
    )
    rerank_scores = select(lambda n: n.startswith(('rerank_u_', 'rerank_s_')))
    topk_geometry = select(lambda n: n.startswith('topk_'))
    visibility_uncertainty = select(
        lambda n: n.startswith(('visibility_', 'uncertainty_')) or n == 'old_candidate_visible'
    )
    offsets_proxy = select(lambda n: n.startswith(('offset_', 'proxy_')))
    query_update = select(lambda n: n.startswith('qinit_qnew_') or n == 'qnew_norm')
    memory = select(
        lambda n: n == 'memory_valid_count' or n.startswith(('qinit_mem_', 'qnew_mem_'))
    )

    def union(*parts: list[int]) -> list[int]:
        return sorted(set(index for part in parts for index in part))

    intrinsic_matching = union(correlation_global, rerank_scores, visibility_uncertainty)
    candidate_alignment = union(correlation_alignment, topk_geometry, offsets_proxy)
    state_only = union(visibility_uncertainty, offsets_proxy, query_update, memory)
    matching_all = union(
        correlation_global,
        correlation_alignment,
        rerank_scores,
        topk_geometry,
    )
    return {
        'correlation_global': correlation_global,
        'correlation_alignment': correlation_alignment,
        'rerank_scores': rerank_scores,
        'topk_geometry': topk_geometry,
        'visibility_uncertainty': visibility_uncertainty,
        'offsets_proxy_geometry': offsets_proxy,
        'query_update': query_update,
        'memory_consistency': memory,
        'intrinsic_matching': intrinsic_matching,
        'candidate_alignment': candidate_alignment,
        'state_only': state_only,
        'matching_all': matching_all,
        'all_internal': list(range(len(names))),
    }


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
    if X_full.shape != (557, 64) or len(names_full) != 64 or len(set(names_full)) != 64:
        raise RuntimeError(f'unexpected internal matrix: {X_full.shape}, names={len(names_full)}')
    if not np.all(np.isfinite(X_full)):
        raise RuntimeError('non-finite internal features')

    variable = np.std(X_full, axis=0) > 1e-10
    removed_constants = [name for name, keep in zip(names_full, variable) if not keep]
    X = X_full[:, variable]
    names = [name for name, keep in zip(names_full, variable) if keep]
    families = feature_families(names)
    if any(not columns for columns in families.values()):
        raise RuntimeError({name: len(columns) for name, columns in families.items()})

    y = data['y_candidate_good'][ext_indices].astype(int)
    groups = data['groups'][ext_indices]
    trained = {}
    for family, columns in families.items():
        fit = logreg_oof(X[:, columns], y, groups)
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
        raise RuntimeError('frozen all_logreg score missing')

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
        name = f'v9a26_family_{family}_logreg_event_max_oof_f1'
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
        paired[family] = paired_summary(result, frozen, seed=20260710 + offset)

    best_classifier = max(rows, key=lambda item: item['classifier']['ap'])
    best_trajectory = max(rows, key=lambda item: item['result']['delta_vs_native']['AJ_RD_256'])
    robust_winners = []
    for item in rows:
        stats = paired[item['family']]
        if (
            item['result']['metric']['AJ_RD_256'] > frozen['metric']['AJ_RD_256']
            and stats['bootstrap_95_ci_mean'][0] > 0.0
        ):
            robust_winners.append(item['family'])

    if robust_winners:
        status = 'INTERNAL_FEATURE_FAMILY_ROBUST_GAIN_FOUND'
        interpretation = (
            f"Robust winners: {robust_winners}. Do not move to V9-A3 before freezing and independently auditing this family."
        )
    else:
        status = 'INTERNAL_SIGNAL_REAL_SELECTOR_ROUTE_SATURATED'
        interpretation = (
            'TrackOn2 internal matching/state families contain heldout row-level signal, but no predeclared family '
            'produces a paired trajectory improvement with a positive lower confidence bound over frozen V9-A2. '
            'Stop selector/fusion/family tuning and move to V9-A3 multi-hypothesis candidate generation or '
            'trainable reacquisition.'
        )

    report = {
        'date': '2026-07-10',
        'script': 'scripts/v9a26_trackon2_internal_feature_family_audit.py',
        'protocol': {
            'target': 'candidate_good on exactly 557 frozen W16-extension rows',
            'cv': '5-fold video-group-heldout OOF',
            'model': 'class-balanced logistic regression',
            'apply_back': 'preserve W8 common; event_max; OOF-F1 only',
            'dense_threshold_search': False,
            'feature_families_predeclared_by_TrackOn2_module_semantics': True,
        },
        'integrity': {
            'internal_npz': str(INTERNAL),
            'internal_npz_sha256': sha256_file(INTERNAL),
            'n_rows': int(X_full.shape[0]),
            'original_feature_dim': int(X_full.shape[1]),
            'variable_feature_dim': int(X.shape[1]),
            'removed_constant_features': removed_constants,
            'finite_rate': float(np.mean(np.isfinite(X_full))),
            'row_alignment_pass': True,
        },
        'feature_families': {
            family: {
                'n_features': len(fit['feature_names']),
                'feature_names': fit['feature_names'],
                'classifier': fit['overall'],
                'folds': fit['folds'],
            }
            for family, fit in trained.items()
        },
        'frozen_baseline': frozen,
        'trajectory_rows': rows,
        'paired_vs_frozen': paired,
        'decision': {
            'status': status,
            'robust_winners': robust_winners,
            'best_classifier_family': best_classifier['family'],
            'best_trajectory_family': best_trajectory['family'],
            'interpretation': interpretation,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# V9-A2.6 TrackOn2 Internal Feature-Family Audit',
        '',
        'Date: 2026-07-10',
        '',
        '## Protocol',
        '',
        '```text',
        'Exactly 557 frozen W16-extension rows; 5-fold video-group-heldout OOF.',
        'Class-balanced logistic regression; event_max + OOF-F1; preserve W8 common.',
        'Feature families are defined by TrackOn2 module semantics; no trajectory threshold sweep.',
        '```',
        '',
        '## Family OOF and trajectory apply-back',
        '',
        '| Family | Dim | AP | AUC | Brier | Threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted | Pos/Neg/Zero |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for item in rows:
        classifier = item['classifier']
        result = item['result']
        delta = result['delta_vs_native']
        ext = result['extension_stats']
        pv = result['per_video_summary']
        lines.append(
            f"| {item['family']} | {classifier['n_features']} | {fmt(classifier['ap'])} | "
            f"{fmt(classifier['auc'])} | {fmt(classifier['brier'])} | {item['threshold']:.6f} | "
            f"{fmt(delta['AJ'], signed=True)} | {fmt(delta['OA'], signed=True)} | "
            f"{fmt(delta['AJ_RD_256'], signed=True)} | {ext['accepted_extension']} | "
            f"{pv['positive']}/{pv['negative']}/{pv['zero']} |"
        )

    lines += [
        '',
        '## Paired against frozen V9-A2 fixed0.05',
        '',
        '| Family | Aggregate Δ | Mean video Δ | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |',
        '|---|---:|---:|---|---|---:|',
    ]
    for family, stats in paired.items():
        aggregate = next(
            item['result']['metric']['AJ_RD_256'] - frozen['metric']['AJ_RD_256']
            for item in rows if item['family'] == family
        )
        ci = stats['bootstrap_95_ci_mean']
        lines.append(
            f"| {family} | {aggregate:+.6f} | {stats['mean_diff']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{stats['better']}/{stats['worse']}/{stats['equal']} | "
            f"{fmt(stats['exact_sign_flip_p'], digits=4)} |"
        )

    lines += [
        '',
        '## Decision',
        '',
        f"**{status}**",
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
        'robust_winners': robust_winners,
        'decision': status,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
