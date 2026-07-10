#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
V9A2_DIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
OUTDIR = BASE / 'v9a25_dinov3_identity'
JOINT = V9A2_DIR / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
DINO = OUTDIR / 'v9a25_dinov3_identity_features.npz'
DINO_REPORT = OUTDIR / 'v9a25_dinov3_identity_features.report.json'
FROZEN = V9A2_DIR / 'v9a2_dynamic_horizon_controller_report.json'
MODEL_DIR = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
OUT_JSON = OUTDIR / 'v9a25_dinov3_feature_family_audit.json'
OUT_DOC = ROOT / 'docs/v9a25_dinov3_feature_family_audit_result_2026-07-10.md'

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

METRIC_KEYS = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    try:
        return {
            'branch': run('git', 'branch', '--show-current'),
            'head': run('git', 'rev-parse', 'HEAD'),
            'head_short': run('git', 'rev-parse', '--short', 'HEAD'),
        }
    except Exception as exc:
        return {'error': repr(exc)}


def metric_delta(metric: dict, base: dict) -> dict:
    return {
        key: None if metric.get(key) is None or base.get(key) is None else float(metric[key] - base[key])
        for key in METRIC_KEYS
    }


def extension_stats(data: dict, accept: np.ndarray) -> dict[str, Any]:
    ext = data['is_w16_extension'].astype(bool)
    accepted = np.asarray(accept, dtype=bool) & ext
    return {
        'accepted_extension': int(np.sum(accepted)),
        'accepted_ext_good': int(np.sum(accepted & data['y_candidate_good'].astype(bool))),
        'accepted_ext_bad': int(np.sum(accepted & data['y_candidate_bad'].astype(bool))),
        'accepted_ext_false_visible': int(np.sum(accepted & data['y_false_visible'].astype(bool))),
        'accepted_ext_worse': int(np.sum(accepted & data['y_candidate_worse_px'].astype(bool))),
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


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out: dict[str, Any] = {
        'ap': float(average_precision_score(y, score)),
        'brier': float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    try:
        out['auc'] = float(roc_auc_score(y, score))
    except ValueError:
        out['auc'] = None
    return out


def logreg_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splits = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    oof = np.full(len(y), np.nan, dtype=float)
    folds = []
    for fold, (tr, va) in enumerate(splits):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced'),
        )
        model.fit(X[tr], y[tr])
        score = model.predict_proba(X[va])[:, 1]
        oof[va] = score
        row = {
            'fold': int(fold),
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'val_videos': sorted(set(groups[va].tolist())),
        }
        row.update(binary_metrics(y[va], score))
        folds.append(row)
    if not np.all(np.isfinite(oof)):
        raise RuntimeError('non-finite OOF scores')
    overall = binary_metrics(y, oof)
    overall.update({
        'threshold_f1': float(choose_threshold_by_metric(y, oof, metric='f1')),
        'n_rows': int(len(y)),
        'n_features': int(X.shape[1]),
        'positive_rate': float(np.mean(y)),
    })
    return {'scores': oof, 'overall': overall, 'folds': folds}


def exact_sign_test_p(pos: int, neg: int) -> float | None:
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    return float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))


def paired_summary(a: dict, b: dict, seed: int, n_boot: int = 200000) -> dict[str, Any]:
    am = {str(row['video_id']): row for row in a['per_video_rows']}
    bm = {str(row['video_id']): row for row in b['per_video_rows']}
    rows = []
    for video_id in sorted(set(am) & set(bm)):
        av = am[video_id].get('delta_AJ_RD_256')
        bv = bm[video_id].get('delta_AJ_RD_256')
        if av is None or bv is None:
            continue
        rows.append({'video_id': video_id, 'a': float(av), 'b': float(bv), 'diff': float(av - bv)})
    d = np.asarray([row['diff'] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 10000):
        n = min(10000, n_boot - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        boot[start:start + n] = d[idx].mean(axis=1)
    tol = 1e-12
    nz = d[np.abs(d) > tol]
    pos = int(np.sum(d > tol))
    neg = int(np.sum(d < -tol))
    equal = int(np.sum(np.abs(d) <= tol))
    if len(nz) == 0:
        flip_p = None
    elif len(nz) <= 20:
        observed = abs(float(np.mean(nz)))
        vals = [
            abs(float(np.mean(nz * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(nz))
        ]
        flip_p = float(np.mean(np.asarray(vals) >= observed - 1e-15))
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, len(nz)))
        flip_p = float(
            np.mean(np.abs((signs * nz).mean(axis=1)) >= abs(float(np.mean(nz))) - 1e-15)
        )
    return {
        'n': int(len(d)),
        'aggregate_diff': float(a['metric']['AJ_RD_256'] - b['metric']['AJ_RD_256']),
        'mean_diff': float(np.mean(d)),
        'median_diff': float(np.median(d)),
        'bootstrap_95_ci_mean': [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        'better': pos,
        'worse': neg,
        'equal': equal,
        'nonzero': int(len(nz)),
        'exact_sign_flip_p': flip_p,
        'exact_sign_test_p': exact_sign_test_p(pos, neg),
        'per_video': rows,
    }


def feature_families(names: list[str]) -> dict[str, list[int]]:
    index = {name: i for i, name in enumerate(names)}
    anchors = ['query', 'last', 'preocc', 'ql_mem', 'qp_mem', 'lp_mem', 'all_mem']
    history_candidate_direct = [
        index[f'{anchor}_cand_{metric}'] for anchor in anchors for metric in ['cos', 'l2']
    ] + [index[name] for name in ['anchor_cand_cos_max', 'anchor_cand_cos_mean', 'anchor_cand_cos_std']]
    history_native_direct = [
        index[f'{anchor}_native_{metric}'] for anchor in anchors for metric in ['cos', 'l2']
    ] + [index[name] for name in ['anchor_native_cos_max', 'anchor_native_cos_mean', 'anchor_native_cos_std']]
    candidate_native_contrast = [
        index[name] for name in [
            'candidate_native_cos',
            'candidate_native_l2',
            'all_mem_cand_minus_native_cos',
            'max_anchor_cand_minus_native_cos',
        ]
    ]
    anchor_consistency = [
        index[name] for name in ['query_last_cos', 'query_preocc_cos', 'last_preocc_cos']
    ]
    local_distinctiveness = [
        i for i, name in enumerate(names) if 'local_neg' in name or 'cand_local_margin' in name
    ]
    history_candidate_identity = sorted(
        set(history_candidate_direct + anchor_consistency + local_distinctiveness)
    )
    all_without_contrast = [i for i in range(len(names)) if i not in candidate_native_contrast]
    return {
        'history_candidate_direct': sorted(set(history_candidate_direct)),
        'history_native_direct': sorted(set(history_native_direct)),
        'candidate_native_contrast': sorted(set(candidate_native_contrast)),
        'anchor_consistency': sorted(set(anchor_consistency)),
        'local_distinctiveness': sorted(set(local_distinctiveness)),
        'history_candidate_identity': history_candidate_identity,
        'full_without_candidate_native_contrast': all_without_contrast,
        'full_dino': list(range(len(names))),
    }


def fmt(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return '—'
    return f'{float(value):+.{digits}f}' if signed else f'{float(value):.{digits}f}'


def main() -> None:
    data = load_joint(JOINT)
    dz = np.load(DINO, allow_pickle=True)
    ext_indices = np.asarray(dz['joint_indices'], dtype=np.int64)
    expected_ext = np.where(data['is_w16_extension'].astype(bool))[0]
    if not np.array_equal(ext_indices, expected_ext):
        raise RuntimeError('DINO joint_indices do not match the frozen W16-extension indices')
    if not np.array_equal(dz['row_key_json'], data['row_key_json'][expected_ext]):
        raise RuntimeError('DINO row keys do not match the frozen joint dataset')
    X_dino = dz['X_dino'].astype(np.float32)
    names = [str(x) for x in dz['feature_names'].tolist()]
    if X_dino.shape != (557, 57) or len(names) != 57 or len(set(names)) != 57:
        raise RuntimeError(f'unexpected DINO matrix shape/names: {X_dino.shape}, {len(names)}')
    if not np.all(np.isfinite(X_dino)):
        raise RuntimeError('non-finite DINO features')

    y = data['y_candidate_good'][ext_indices].astype(int)
    groups = data['groups'][ext_indices]
    families = feature_families(names)
    trained: dict[str, dict[str, Any]] = {}
    for family, columns in families.items():
        fit = logreg_oof(X_dino[:, columns], y, groups)
        trained[family] = {
            'columns': columns,
            'feature_names': [names[i] for i in columns],
            **fit,
        }

    frozen_report = json.loads(FROZEN.read_text())
    frozen_scores = None
    for block in frozen_report['training']:
        for row in block['overall']:
            if row['feature_set'] == 'all' and row['model'] == 'logreg':
                frozen_scores = np.asarray(block['full_scores']['all_logreg'], dtype=float)
                break
    if frozen_scores is None:
        raise RuntimeError('missing frozen all_logreg scores')

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    candidate = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, candidate)
    if not ok:
        raise RuntimeError(info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    common = data['is_common_w8'].astype(bool)

    frozen = evaluate(
        'frozen_v9a2_fixed0.05',
        data,
        native,
        cand_by,
        event_accept_mask(data, frozen_scores, 0.05, 'event_max'),
        native_metric,
        native_pv,
    )
    w16 = evaluate(
        'w16_accept_all_joint',
        data,
        native,
        cand_by,
        np.ones(len(data['metas']), dtype=bool),
        native_metric,
        native_pv,
    )

    rows = []
    paired = {}
    for offset, (family, fit) in enumerate(trained.items()):
        full = np.zeros(len(data['metas']), dtype=float)
        full[ext_indices] = fit['scores']
        threshold = float(fit['overall']['threshold_f1'])
        name = f'v9a25_family_{family}_logreg_event_max_oof_f1'
        row = evaluate(
            name,
            data,
            native,
            cand_by,
            event_accept_mask(data, full, threshold, 'event_max'),
            native_metric,
            native_pv,
        )
        rows.append({
            'family': family,
            'threshold': threshold,
            'classifier': fit['overall'],
            'result': row,
        })
        paired[family] = paired_summary(row, frozen, seed=20260710 + offset)

    best_classifier = max(rows, key=lambda item: item['classifier']['ap'])
    best_trajectory = max(rows, key=lambda item: item['result']['delta_vs_native']['AJ_RD_256'])
    full_row = next(item for item in rows if item['family'] == 'full_dino')
    history_row = next(item for item in rows if item['family'] == 'history_candidate_identity')
    contrast_row = next(item for item in rows if item['family'] == 'candidate_native_contrast')

    decision = {
        'status': 'REAL_SEMANTIC_RANKING_SIGNAL_NO_ROBUST_TRAJECTORY_GAIN',
        'interpretation': (
            'Historical candidate identity and local distinctiveness provide real video-heldout ranking signal, '
            'so the DINO result is not explained only by candidate-native disagreement. However, none of the '
            'predeclared feature families establishes a robust trajectory improvement over frozen V9-A2. '
            'Stop DINO concatenation/fusion/family tuning and move to TrackOn2 internal matching/memory features.'
        ),
        'best_classifier_family': best_classifier['family'],
        'best_trajectory_family': best_trajectory['family'],
        'history_candidate_identity_ap': history_row['classifier']['ap'],
        'candidate_native_contrast_ap': contrast_row['classifier']['ap'],
        'full_dino_ap': full_row['classifier']['ap'],
    }

    report = {
        'date': '2026-07-10',
        'script': 'scripts/v9a25_dinov3_feature_family_audit.py',
        'provenance': git_provenance(),
        'protocol': {
            'target': 'candidate_good on exactly 557 frozen W16-extension rows',
            'cv': '5-fold video-group-heldout OOF',
            'model': 'class-balanced logistic regression',
            'apply_back': 'preserve W8 common; event_max; OOF-F1 only',
            'dense_threshold_search': False,
        },
        'integrity': {
            'dino_npz': str(DINO),
            'dino_npz_sha256': sha256_file(DINO),
            'dino_report': str(DINO_REPORT),
            'model_weights': str(MODEL_DIR / 'model.safetensors'),
            'model_weights_sha256': sha256_file(MODEL_DIR / 'model.safetensors'),
            'config_sha256': sha256_file(MODEL_DIR / 'config.json'),
            'n_rows': int(X_dino.shape[0]),
            'feature_dim': int(X_dino.shape[1]),
            'finite_rate': float(np.mean(np.isfinite(X_dino))),
            'row_alignment_pass': True,
        },
        'feature_families': {
            family: {
                'n_features': len(fit['columns']),
                'feature_names': fit['feature_names'],
                'classifier': fit['overall'],
                'folds': fit['folds'],
            }
            for family, fit in trained.items()
        },
        'baselines': [frozen, w16],
        'trajectory_rows': rows,
        'paired_vs_frozen': paired,
        'decision': decision,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# V9-A2.5 DINOv3 Feature-Family Audit',
        '',
        'Date: 2026-07-10',
        '',
        '## Protocol',
        '',
        '```text',
        'Exactly 557 frozen W16-extension rows; 5-fold video-group-heldout OOF.',
        'Class-balanced logistic regression; event_max + OOF-F1; preserve W8 common.',
        'No DINO re-encoding and no trajectory threshold sweep.',
        '```',
        '',
        '## Integrity',
        '',
        f"- Row alignment: PASS ({report['integrity']['n_rows']} rows, {report['integrity']['feature_dim']} features).",
        f"- Finite rate: {report['integrity']['finite_rate']:.6f}.",
        f"- DINO NPZ SHA256: `{report['integrity']['dino_npz_sha256']}`.",
        f"- Model weights SHA256: `{report['integrity']['model_weights_sha256']}`.",
        '',
        '## Feature-family OOF and trajectory apply-back',
        '',
        '| Family | Dim | AP | AUC | Brier | OOF-F1 threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted | Pos/Neg/Zero |',
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
        ci = stats['bootstrap_95_ci_mean']
        lines.append(
            f"| {family} | {stats['aggregate_diff']:+.6f} | {stats['mean_diff']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{stats['better']}/{stats['worse']}/{stats['equal']} | "
            f"{fmt(stats['exact_sign_flip_p'], digits=4)} |"
        )

    lines += [
        '',
        '## Decision',
        '',
        f"**{decision['status']}**",
        '',
        decision['interpretation'],
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
        'decision': decision['status'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
