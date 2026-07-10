#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
JOINT = OUTDIR / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
CONTROLLER = OUTDIR / 'v9a2_dynamic_horizon_controller_report.json'
V9A1 = BASE / 'v9a1_controller_calibration_aware_prototype/v9a1_report.json'
OUT_JSON = OUTDIR / 'v9a2_paper_ready_ablation.json'
OUT_DOC = ROOT / 'docs/v9a2_paper_ready_ablation_result_2026-07-10.md'

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
from scripts.v9a2_anchor_uncertainty_reacquisition_prototype import (  # noqa: E402
    DAVIS_PKL,
    build_row_features,
    load_davis_pkl,
    record_maps,
)
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint  # noqa: E402

METRIC_KEYS = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
MAIN_SCORE = 'all_logreg'
MAIN_MODE = 'event_max'
MAIN_THRESHOLD = 0.05


def metric_delta(metric: dict, base: dict) -> dict:
    return {
        key: None if metric.get(key) is None or base.get(key) is None else float(metric[key] - base[key])
        for key in METRIC_KEYS
    }


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    try:
        return {
            'branch': run('git', 'branch', '--show-current'),
            'head': run('git', 'rev-parse', 'HEAD'),
            'head_short': run('git', 'rev-parse', '--short', 'HEAD'),
            'tracked_status': run('git', 'status', '--porcelain', '--untracked-files=no'),
        }
    except Exception as exc:  # provenance must not block the experiment
        return {'error': repr(exc)}


def zero_extension_stats() -> dict[str, Any]:
    return {
        'accepted_extension': 0,
        'total_extension': 0,
        'accepted_extension_fraction': None,
        'accepted_ext_good': 0,
        'accepted_ext_bad': 0,
        'accepted_ext_false_visible': 0,
        'accepted_ext_worse': 0,
        'total_extension_events': 0,
        'accepted_extension_events': 0,
        'event_activation_fraction': None,
    }


def extension_stats(data: dict, accept: np.ndarray) -> dict[str, Any]:
    accept = np.asarray(accept, dtype=bool)
    ext = data['is_w16_extension'].astype(bool)
    accepted_ext = accept & ext
    ext_indices = np.where(ext)[0]
    accepted_indices = np.where(accepted_ext)[0]
    ext_events = {data['event_keys'][i] for i in ext_indices.tolist()}
    accepted_events = {data['event_keys'][i] for i in accepted_indices.tolist()}
    n_ext = int(np.sum(ext))
    n_accepted = int(np.sum(accepted_ext))
    return {
        'accepted_extension': n_accepted,
        'total_extension': n_ext,
        'accepted_extension_fraction': float(n_accepted / n_ext) if n_ext else None,
        'accepted_ext_good': int(np.sum(accepted_ext & data['y_candidate_good'].astype(bool))),
        'accepted_ext_bad': int(np.sum(accepted_ext & data['y_candidate_bad'].astype(bool))),
        'accepted_ext_false_visible': int(np.sum(accepted_ext & data['y_false_visible'].astype(bool))),
        'accepted_ext_worse': int(np.sum(accepted_ext & data['y_candidate_worse_px'].astype(bool))),
        'total_extension_events': int(len(ext_events)),
        'accepted_extension_events': int(len(accepted_events)),
        'event_activation_fraction': float(len(accepted_events) / len(ext_events)) if ext_events else None,
    }


def eval_variant(
    name: str,
    data: dict,
    native: dict,
    cand_by: dict,
    accept: np.ndarray,
    native_metric: dict,
    native_pv: list[dict],
    **metadata: Any,
) -> dict[str, Any]:
    accept = np.asarray(accept, dtype=bool)
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], accept)
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    row = {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'extension_stats': extension_stats(data, accept),
        'per_video_summary': summarize_pv(pv_rows),
        'per_video_rows': pv_rows,
    }
    row.update(metadata)
    return row


def load_existing_scores(report: dict) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for block in report['training']:
        for row in block['overall']:
            name = f"{row['feature_set']}_{row['model']}"
            if name in block['full_scores']:
                out[name] = {
                    'scores': np.asarray(block['full_scores'][name], dtype=float),
                    'threshold_f1': float(row['threshold_f1_oof']),
                    'ap': float(row['ap']),
                    'auc': None if row['auc'] is None else float(row['auc']),
                    'brier': float(row['brier']),
                }
    return out


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    out: dict[str, Any] = {
        'ap': float(average_precision_score(y, score)),
        'brier': float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    try:
        out['auc'] = float(roc_auc_score(y, score))
    except ValueError:
        out['auc'] = None
    return out


def custom_logreg_oof(data: dict, columns: np.ndarray, name: str) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ext = data['is_w16_extension'].astype(bool)
    X = data['X_all'].astype(np.float32)[ext][:, columns]
    y = data['y_candidate_good'].astype(int)[ext]
    groups = data['groups'][ext]
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError(f'not enough video groups for {name}')
    oof = np.full(len(y), np.nan, dtype=float)
    folds = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced'),
        )
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        score = model.predict_proba(X[va])[:, 1]
        inference_seconds = time.perf_counter() - t0
        oof[va] = score
        fold_row = {
            'fold': int(fold),
            'n_train': int(len(tr)),
            'n_val': int(len(va)),
            'val_videos': sorted(set(groups[va].tolist())),
            'fit_seconds': float(fit_seconds),
            'inference_seconds': float(inference_seconds),
        }
        fold_row.update(binary_metrics(y[va], score))
        folds.append(fold_row)
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f'non-finite OOF score for {name}')
    threshold = float(choose_threshold_by_metric(y, oof, metric='f1'))
    full = np.zeros(len(ext), dtype=float)
    full[np.where(ext)[0]] = oof
    overall = binary_metrics(y, oof)
    overall.update({
        'threshold_f1': threshold,
        'n_features': int(len(columns)),
        'positive_rate': float(np.mean(y)),
    })
    return {
        'name': name,
        'scores': full,
        'overall': overall,
        'folds': folds,
        'columns': columns.tolist(),
    }


def anchor_group_columns(names: list[str]) -> dict[str, np.ndarray]:
    base: list[int] = []
    generic: list[int] = []
    query: list[int] = []
    last: list[int] = []
    preocc: list[int] = []
    all_anchor: list[int] = []
    global_names = {
        'dist_native_candidate_norm_yx',
        'event_age',
        'invis_age',
        'native_motion_prev_norm_yx',
        'sigma_proxy',
        'normalized_distance',
    }
    for i, name in enumerate(names):
        if name.startswith('base::'):
            base.append(i)
            continue
        if not name.startswith('anchor::'):
            continue
        all_anchor.append(i)
        raw = name.split('anchor::', 1)[1]
        if raw.startswith(('candidate_oob_', 'native_oob_')) or raw in global_names or re.fullmatch(r'r\d+_cand_valid', raw):
            generic.append(i)
        if raw.startswith('query_anchor_') or re.match(r'r\d+_query_', raw):
            query.append(i)
        if raw.startswith('last_anchor_') or raw == 'tau_minus_last_anchor' or re.match(r'r\d+_last_', raw):
            last.append(i)
        if raw.startswith('preocc_anchor_') or raw == 'event_minus_preocc_anchor' or re.match(r'r\d+_preocc_', raw):
            preocc.append(i)

    def arr(*parts: list[int]) -> np.ndarray:
        return np.asarray(sorted(set(itertools.chain.from_iterable(parts))), dtype=np.int64)

    return {
        'base_generic': arr(base, generic),
        'base_query_anchor': arr(base, generic, query),
        'base_last_anchor': arr(base, generic, last),
        'base_preocc_anchor': arr(base, generic, preocc),
        'base_all_anchors': arr(base, all_anchor),
    }


def pv_map(row: dict) -> dict[str, dict]:
    if row.get('per_video_rows') is None:
        raise ValueError(f"variant {row.get('variant')} has no per-video rows")
    return {str(r['video_id']): r for r in row['per_video_rows']}


def exact_sign_test_p(pos: int, neg: int) -> float | None:
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_uncertainty(
    a: dict,
    b: dict,
    metric_key: str = 'delta_AJ_RD_256',
    seed: int = 20260710,
    n_boot: int = 200000,
) -> dict[str, Any]:
    am = pv_map(a)
    bm = pv_map(b)
    values = []
    rows = []
    for vid in sorted(set(am) & set(bm)):
        av = am[vid].get(metric_key)
        bv = bm[vid].get(metric_key)
        if av is None or bv is None:
            continue
        diff = float(av - bv)
        values.append(diff)
        rows.append({'video_id': vid, 'a': float(av), 'b': float(bv), 'diff': diff})
    d = np.asarray(values, dtype=float)
    if len(d) == 0:
        raise RuntimeError(f'no paired values for {metric_key}')
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    boot_medians = np.empty(n_boot, dtype=np.float64)
    chunk = 10000
    for start in range(0, n_boot, chunk):
        n = min(chunk, n_boot - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        sample = d[idx]
        boot_means[start:start + n] = sample.mean(axis=1)
        boot_medians[start:start + n] = np.median(sample, axis=1)
    tol = 1e-12
    nz = d[np.abs(d) > tol]
    pos = int(np.sum(d > tol))
    neg = int(np.sum(d < -tol))
    equal = int(np.sum(np.abs(d) <= tol))
    if len(nz) == 0:
        sign_flip_p = None
    elif len(nz) <= 20:
        observed = abs(float(np.mean(nz)))
        perm = [
            abs(float(np.mean(nz * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(nz))
        ]
        sign_flip_p = float(np.mean(np.asarray(perm) >= observed - 1e-15))
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, len(nz)))
        sign_flip_p = float(
            np.mean(np.abs((signs * nz).mean(axis=1)) >= abs(float(np.mean(nz))) - 1e-15)
        )
    return {
        'metric_key': metric_key,
        'n': int(len(d)),
        'aggregate_diff': float(a['metric']['AJ_RD_256'] - b['metric']['AJ_RD_256']),
        'mean_diff': float(np.mean(d)),
        'median_diff': float(np.median(d)),
        'bootstrap_95_ci_mean': [float(x) for x in np.quantile(boot_means, [0.025, 0.975])],
        'bootstrap_95_ci_median': [float(x) for x in np.quantile(boot_medians, [0.025, 0.975])],
        'bootstrap_probability_mean_gt_zero': float(np.mean(boot_means > 0.0)),
        'better': pos,
        'worse': neg,
        'equal': equal,
        'nonzero': int(len(nz)),
        'exact_sign_flip_p': sign_flip_p,
        'exact_sign_test_p': exact_sign_test_p(pos, neg),
        'per_video': rows,
    }


def benchmark_efficiency(
    data: dict,
    main_scores: np.ndarray,
    main_accept: np.ndarray,
    inference_repeats: int,
    aggregation_repeats: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    ext = data['is_w16_extension'].astype(bool)
    X = data['X_all'].astype(np.float32)[ext]
    y = data['y_candidate_good'].astype(int)[ext]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced'),
    )
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        model.predict_proba(X)[:, 1]
        batch_ms = []
        for _ in range(inference_repeats):
            t0 = time.perf_counter()
            model.predict_proba(X)[:, 1]
            batch_ms.append((time.perf_counter() - t0) * 1000.0)
    event_accept_mask(data, main_scores, MAIN_THRESHOLD, MAIN_MODE)
    event_ms = []
    for _ in range(aggregation_repeats):
        t0 = time.perf_counter()
        event_accept_mask(data, main_scores, MAIN_THRESHOLD, MAIN_MODE)
        event_ms.append((time.perf_counter() - t0) * 1000.0)

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    native_by, cand_by = record_maps(native, cand)
    t0 = time.perf_counter()
    davis = load_davis_pkl(DAVIS_PKL)
    data_load_s = time.perf_counter() - t0
    source_metas = [json.loads(str(x)) for x in data['source_meta_json'].tolist()]
    ext_indices = np.where(ext)[0]
    feature_ms = []
    feature_stored_max_abs_diff = 0.0
    for i in ext_indices:
        m = source_metas[int(i)]
        vid = str(m['video_id'])
        t0 = time.perf_counter()
        feat, _ = build_row_features(m, native_by[vid], cand_by[vid], davis[vid]['video'], [5, 9, 17])
        elapsed = (time.perf_counter() - t0) * 1000.0
        if feat.shape[0] != 115 or not np.all(np.isfinite(feat)):
            raise RuntimeError(f'invalid benchmark feature at row {i}')
        feature_stored_max_abs_diff = max(
            feature_stored_max_abs_diff,
            float(np.max(np.abs(feat - data['X_anchor'][int(i)]))),
        )
        feature_ms.append(elapsed)

    ext_events = {data['event_keys'][i] for i in ext_indices.tolist()}
    accepted_ext_indices = np.where(main_accept & ext)[0]
    accepted_events = {data['event_keys'][i] for i in accepted_ext_indices.tolist()}
    return {
        'hardware_note': (
            'Single-thread CPU reference Python/Numpy/Scikit-learn implementation; cached video/candidate tensors; '
            'not optimized online kernel latency. Full-data fit is runtime-only; paper metrics use video-heldout OOF scores.'
        ),
        'n_extension_rows': int(len(ext_indices)),
        'n_extension_events': int(len(ext_events)),
        'accepted_extension_rows': int(len(accepted_ext_indices)),
        'accepted_extension_fraction': float(len(accepted_ext_indices) / max(1, len(ext_indices))),
        'accepted_events': int(len(accepted_events)),
        'event_activation_fraction': float(len(accepted_events) / max(1, len(ext_events))),
        'controller_fit_seconds_runtime_only': float(fit_s),
        'controller_inference_repeats': int(inference_repeats),
        'controller_batch_557_ms_p50': float(np.quantile(batch_ms, 0.50)),
        'controller_batch_557_ms_p95': float(np.quantile(batch_ms, 0.95)),
        'controller_per_row_us_from_batch_p50': float(
            np.quantile(batch_ms, 0.50) * 1000.0 / max(1, len(ext_indices))
        ),
        'event_aggregation_repeats': int(aggregation_repeats),
        'event_aggregation_ms_p50': float(np.quantile(event_ms, 0.50)),
        'event_aggregation_ms_p95': float(np.quantile(event_ms, 0.95)),
        'davis_pickle_load_seconds': float(data_load_s),
        'feature_extract_rows': int(len(feature_ms)),
        'feature_extract_total_seconds': float(np.sum(feature_ms) / 1000.0),
        'feature_extract_ms_mean': float(np.mean(feature_ms)),
        'feature_extract_ms_p50': float(np.quantile(feature_ms, 0.50)),
        'feature_extract_ms_p95': float(np.quantile(feature_ms, 0.95)),
        'feature_stored_max_abs_diff': float(feature_stored_max_abs_diff),
        'feature_parity_pass': bool(feature_stored_max_abs_diff <= 1e-6),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != 'per_video_rows'}


def fmt(x: float | None, signed: bool = False, digits: int = 4) -> str:
    if x is None:
        return '—'
    return f'{float(x):+.{digits}f}' if signed else f'{float(x):.{digits}f}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=Path, default=JOINT)
    parser.add_argument('--controller', type=Path, default=CONTROLLER)
    parser.add_argument('--v9a1', type=Path, default=V9A1)
    parser.add_argument('--out-json', type=Path, default=OUT_JSON)
    parser.add_argument('--out-doc', type=Path, default=OUT_DOC)
    parser.add_argument('--bootstrap-resamples', type=int, default=200000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260710)
    parser.add_argument('--inference-repeats', type=int, default=300)
    parser.add_argument('--aggregation-repeats', type=int, default=500)
    args = parser.parse_args()

    data = load_joint(args.joint)
    controller = json.loads(args.controller.read_text())
    existing = load_existing_scores(controller)
    names = [str(x) for x in data['all_feature_names'].tolist()]
    groups = anchor_group_columns(names)

    custom: dict[str, dict[str, Any]] = {}
    for name, cols in groups.items():
        if name == 'base_all_anchors':
            continue
        custom[name] = custom_logreg_oof(data, cols, name)

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(align_info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)
    y_good = data['y_candidate_good'].astype(bool)
    y_worse = data['y_candidate_worse_px'].astype(bool)

    variants: dict[str, dict[str, Any]] = {}

    def add(name: str, accept: np.ndarray, **metadata: Any) -> None:
        if name not in variants:
            variants[name] = eval_variant(
                name,
                data,
                native,
                cand_by,
                accept,
                native_metric,
                native_pv,
                **metadata,
            )

    add('w8_preserve_common_only', common, policy='CVRRM W8 accept-all')
    add('w16_accept_all_joint', common | ext, policy='CVRRM W16 accept-all')
    add('oracle_ext_candidate_good', common | (ext & y_good), policy='GT-derived upper bound')
    add(
        'oracle_ext_good_not_worse',
        common | (ext & y_good & ~y_worse),
        policy='GT-derived upper bound',
    )

    all_score = existing[MAIN_SCORE]['scores']
    all_f1 = existing[MAIN_SCORE]['threshold_f1']
    add(
        'v9a2_all_logreg_event_max_fixed_0.05',
        event_accept_mask(data, all_score, MAIN_THRESHOLD, MAIN_MODE),
        feature_set='all',
        mode=MAIN_MODE,
        threshold_protocol='fixed_0.05',
        threshold=MAIN_THRESHOLD,
    )
    add(
        'v9a2_all_logreg_event_max_oof_f1',
        event_accept_mask(data, all_score, all_f1, MAIN_MODE),
        feature_set='all',
        mode=MAIN_MODE,
        threshold_protocol='oof_f1',
        threshold=float(all_f1),
    )

    feature_ablation = []
    for score_name in ['base_logreg', 'anchor_logreg', 'all_logreg']:
        info = existing[score_name]
        feature_set = score_name.split('_', 1)[0]
        name = f'feature_{score_name}_event_max_fixed_0.05'
        add(
            name,
            event_accept_mask(data, info['scores'], MAIN_THRESHOLD, MAIN_MODE),
            feature_set=feature_set,
            mode=MAIN_MODE,
            threshold_protocol='fixed_0.05',
            threshold=MAIN_THRESHOLD,
        )
        feature_ablation.append({
            'variant': name,
            'feature_set': feature_set,
            'classifier': {key: info[key] for key in ['ap', 'auc', 'brier', 'threshold_f1']},
        })

    mode_ablation = []
    for mode in ['frame', 'event_max', 'event_mean']:
        name = f'mode_all_logreg_{mode}_fixed_0.05'
        add(
            name,
            event_accept_mask(data, all_score, MAIN_THRESHOLD, mode),
            feature_set='all',
            mode=mode,
            threshold_protocol='fixed_0.05',
            threshold=MAIN_THRESHOLD,
        )
        mode_ablation.append(name)

    threshold_ablation = [
        'w16_accept_all_joint',
        'v9a2_all_logreg_event_max_fixed_0.05',
        'v9a2_all_logreg_event_max_oof_f1',
    ]

    factorial_ablation = []
    for score_name in ['base_logreg', 'anchor_logreg', 'all_logreg']:
        info = existing[score_name]
        feature_set = score_name.split('_', 1)[0]
        for mode in ['frame', 'event_max', 'event_mean']:
            for protocol, threshold in [('fixed_0.05', MAIN_THRESHOLD), ('oof_f1', info['threshold_f1'])]:
                name = f'factorial_{feature_set}_{mode}_{protocol}'
                add(
                    name,
                    event_accept_mask(data, info['scores'], float(threshold), mode),
                    feature_set=feature_set,
                    mode=mode,
                    threshold_protocol=protocol,
                    threshold=float(threshold),
                )
                factorial_ablation.append({
                    'variant': name,
                    'feature_set': feature_set,
                    'mode': mode,
                    'protocol': protocol,
                    'threshold': float(threshold),
                    'classifier': {key: info[key] for key in ['ap', 'auc', 'brier', 'threshold_f1']},
                })

    anchor_ablation = []
    for group_name, result in custom.items():
        for protocol, threshold in [('fixed_0.05', MAIN_THRESHOLD), ('oof_f1', result['overall']['threshold_f1'])]:
            name = f'anchor_group_{group_name}_event_max_{protocol}'
            add(
                name,
                event_accept_mask(data, result['scores'], float(threshold), MAIN_MODE),
                anchor_group=group_name,
                feature_dim=int(result['overall']['n_features']),
                mode=MAIN_MODE,
                threshold_protocol=protocol,
                threshold=float(threshold),
            )
            anchor_ablation.append({
                'variant': name,
                'group': group_name,
                'protocol': protocol,
                'threshold': float(threshold),
                'classifier': result['overall'],
            })
    for protocol, threshold in [('fixed_0.05', MAIN_THRESHOLD), ('oof_f1', all_f1)]:
        name = f'anchor_group_base_all_anchors_event_max_{protocol}'
        add(
            name,
            event_accept_mask(data, all_score, float(threshold), MAIN_MODE),
            anchor_group='base_all_anchors',
            feature_dim=len(names),
            mode=MAIN_MODE,
            threshold_protocol=protocol,
            threshold=float(threshold),
        )
        anchor_ablation.append({
            'variant': name,
            'group': 'base_all_anchors',
            'protocol': protocol,
            'threshold': float(threshold),
            'classifier': {key: existing[MAIN_SCORE][key] for key in ['ap', 'auc', 'brier', 'threshold_f1']},
        })

    main_row = variants['v9a2_all_logreg_event_max_fixed_0.05']
    paired = {
        'fixed0.05_vs_w16': paired_uncertainty(
            main_row,
            variants['w16_accept_all_joint'],
            seed=args.bootstrap_seed,
            n_boot=args.bootstrap_resamples,
        ),
        'fixed0.05_vs_w8': paired_uncertainty(
            main_row,
            variants['w8_preserve_common_only'],
            seed=args.bootstrap_seed + 1,
            n_boot=args.bootstrap_resamples,
        ),
    }

    efficiency = benchmark_efficiency(
        data,
        all_score,
        event_accept_mask(data, all_score, MAIN_THRESHOLD, MAIN_MODE),
        inference_repeats=args.inference_repeats,
        aggregation_repeats=args.aggregation_repeats,
    )
    if not efficiency['feature_parity_pass']:
        raise RuntimeError(
            f"feature extraction parity failed: max_abs_diff={efficiency['feature_stored_max_abs_diff']}"
        )

    v9a1 = json.loads(args.v9a1.read_text())
    v9a1_best = next(
        row for row in v9a1['trajectory']['rows'] if row['variant'] == 'v9a1_extratrees_oof_f1'
    )
    native_row = {
        'variant': 'native',
        'metric': native_metric,
        'delta_vs_native': {key: 0.0 for key in METRIC_KEYS},
        'apply_stats': {'accepted_frames': 0},
        'extension_stats': zero_extension_stats(),
        'per_video_summary': {'n': 25, 'positive': 0, 'negative': 0, 'zero': 25},
    }
    v9a1_row = {
        'variant': 'v9a1_extratrees_oof_f1',
        **{key: v9a1_best[key] for key in ['metric', 'delta_vs_native', 'apply_stats', 'per_video_summary']},
        'extension_stats': zero_extension_stats(),
        'policy': 'predeclared V9-A1 ExtraTrees OOF-F1 diagnostic; W8-only filter',
    }
    paper_rows = [
        native_row,
        compact(variants['w8_preserve_common_only']),
        compact(variants['w16_accept_all_joint']),
        v9a1_row,
        compact(main_row),
        compact(variants['v9a2_all_logreg_event_max_oof_f1']),
        compact(variants['oracle_ext_candidate_good']),
    ]

    ci = paired['fixed0.05_vs_w16']['bootstrap_95_ci_mean']
    simultaneous_advantage = bool(
        main_row['metric']['AJ'] > variants['w16_accept_all_joint']['metric']['AJ']
        and main_row['metric']['OA'] > variants['w16_accept_all_joint']['metric']['OA']
        and main_row['metric']['AJ_RD_256'] > variants['w16_accept_all_joint']['metric']['AJ_RD_256']
        and main_row['per_video_summary']['positive'] >= variants['w16_accept_all_joint']['per_video_summary']['positive']
        and main_row['per_video_summary']['negative'] <= variants['w16_accept_all_joint']['per_video_summary']['negative']
    )
    strict_ci_nonnegative = bool(ci[0] >= 0.0)
    if strict_ci_nonnegative and simultaneous_advantage:
        decision = {
            'status': 'PASS_FREEZE_V9A2_PROTOTYPE',
            'strict_ci_nonnegative': True,
            'simultaneous_metric_advantage': True,
            'interpretation': (
                'The paired mean 95% CI versus W16 does not enter the negative region, while AJ/OA/AJ_RD_256 '
                'and the positive/negative video structure remain better. Freeze V9-A2 as the paper-ready '
                'dynamic-horizon prototype, with modest-effect caveats.'
            ),
        }
    else:
        decision = {
            'status': 'DIAGNOSTIC_PASS_STATISTICAL_FREEZE_GATE_NOT_MET',
            'strict_ci_nonnegative': strict_ci_nonnegative,
            'simultaneous_metric_advantage': simultaneous_advantage,
            'interpretation': (
                f"The aggregate fixed0.05 result is positive, but the paired mean 95% CI versus W16 is "
                f"[{ci[0]:+.6f}, {ci[1]:+.6f}] and crosses zero. The gain is concentrated. Retain V9-A2 "
                'as a paper-ready positive diagnostic/prototype, not a statistically established final method; '
                'proceed to V9-A2.5 semantic/internal identity features.'
            ),
        }

    result = {
        'script': 'scripts/v9a2_paper_ready_ablation.py',
        'date': '2026-07-10',
        'provenance': git_provenance(),
        'inputs': {
            'joint': str(args.joint),
            'controller': str(args.controller),
            'v9a1': str(args.v9a1),
            'native': str(NATIVE),
            'candidate': str(CANDIDATE),
        },
        'protocol': {
            'frozen_main_policy': (
                'all_logreg + event_max + fixed threshold 0.05; preserve W8 common; '
                'control W16 extension only'
            ),
            'no_dense_threshold_search': True,
            'paired_unit': 'DAVIS video with defined AJ_RD_256',
            'bootstrap_resamples': int(args.bootstrap_resamples),
            'bootstrap_seed': int(args.bootstrap_seed),
        },
        'paper_table': paper_rows,
        'feature_ablation': [
            {**item, 'result': compact(variants[item['variant']])} for item in feature_ablation
        ],
        'mode_ablation': [compact(variants[name]) for name in mode_ablation],
        'threshold_ablation': [compact(variants[name]) for name in threshold_ablation],
        'factorial_ablation': [
            {**item, 'result': compact(variants[item['variant']])} for item in factorial_ablation
        ],
        'anchor_group_definitions': {
            key: {'n_features': int(len(cols)), 'feature_names': [names[i] for i in cols]}
            for key, cols in groups.items()
        },
        'anchor_group_training': {
            key: {'overall': value['overall'], 'folds': value['folds']}
            for key, value in custom.items()
        },
        'anchor_group_ablation': [
            {**item, 'result': compact(variants[item['variant']])} for item in anchor_ablation
        ],
        'paired_uncertainty': paired,
        'efficiency': efficiency,
        'decision': decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# V9-A2.4 Paper-Ready Ablation Result',
        '',
        'Date: 2026-07-10',
        '',
        '## Frozen protocol',
        '',
        '```text',
        result['protocol']['frozen_main_policy'],
        '```',
        '',
        'No dense trajectory threshold search was used in V9-A2.4. Learned trajectory metrics use video-group-heldout OOF scores.',
        '',
        '## Unified absolute metrics',
        '',
        '| Variant | AJ | OA | delta_avg | delta_4px | AJ_RD | AJ_RD_256 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in paper_rows:
        m = row['metric']
        lines.append(
            f"| {row['variant']} | {fmt(m['AJ'])} | {fmt(m['OA'])} | {fmt(m['delta_avg'])} | "
            f"{fmt(m['delta_4px'])} | {fmt(m['AJ_RD'])} | {fmt(m['AJ_RD_256'])} |"
        )

    lines += [
        '',
        '## Unified deltas, action, and stability',
        '',
        '| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Accepted | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in paper_rows:
        d = row['delta_vs_native']
        pv = row['per_video_summary']
        accepted = row['apply_stats'].get('accepted_frames', 0)
        es = row['extension_stats']
        ext_text = (
            f"{es['accepted_extension']}/{es['accepted_ext_good']}/{es['accepted_ext_bad']}/"
            f"{es['accepted_ext_false_visible']}/{es['accepted_ext_worse']}"
        )
        lines.append(
            f"| {row['variant']} | {fmt(d['AJ'], True)} | {fmt(d['OA'], True)} | "
            f"{fmt(d['delta_avg'], True)} | {fmt(d['delta_4px'], True)} | "
            f"{fmt(d['AJ_RD'], True)} | {fmt(d['AJ_RD_256'], True)} | {accepted} | "
            f"{ext_text} | {pv['positive']}/{pv['negative']}/{pv['zero']} |"
        )

    def add_ablation_table(title: str, rows: list[dict]) -> None:
        lines.extend([
            '',
            f'## {title}',
            '',
            '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |',
            '|---|---:|---:|---:|---:|---|',
        ])
        for row in rows:
            d = row['delta_vs_native']
            es = row['extension_stats']
            pv = row['per_video_summary']
            ext_text = (
                f"{es['accepted_extension']}/{es['accepted_ext_good']}/{es['accepted_ext_bad']}/"
                f"{es['accepted_ext_false_visible']}/{es['accepted_ext_worse']}"
            )
            lines.append(
                f"| {row['variant']} | {fmt(d['AJ'], True)} | {fmt(d['OA'], True)} | "
                f"{fmt(d['AJ_RD_256'], True)} | {ext_text} | "
                f"{pv['positive']}/{pv['negative']}/{pv['zero']} |"
            )

    add_ablation_table('Feature-set ablation', [variants[item['variant']] for item in feature_ablation])
    add_ablation_table('Controller-mode ablation', [variants[name] for name in mode_ablation])
    add_ablation_table('Threshold-protocol ablation', [variants[name] for name in threshold_ablation])

    lines += [
        '',
        '## Full feature-set × controller-mode × threshold-protocol ablation',
        '',
        '| Feature | Mode | Protocol | Threshold | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted | Pos/Neg/Zero |',
        '|---|---|---|---:|---:|---:|---:|---:|---|',
    ]
    for item in factorial_ablation:
        row = variants[item['variant']]
        d = row['delta_vs_native']
        es = row['extension_stats']
        pv = row['per_video_summary']
        lines.append(
            f"| {item['feature_set']} | {item['mode']} | {item['protocol']} | {fmt(item['threshold'])} | "
            f"{fmt(d['AJ'], True)} | {fmt(d['OA'], True)} | {fmt(d['AJ_RD_256'], True)} | "
            f"{es['accepted_extension']} | {pv['positive']}/{pv['negative']}/{pv['zero']} |"
        )

    add_ablation_table('Anchor-group ablation', [variants[item['variant']] for item in anchor_ablation])

    lines += [
        '',
        '## Paired video-level uncertainty (AJ_RD_256)',
        '',
        '| Comparison | N | Aggregate Δ | Mean video Δ | Median video Δ | Bootstrap mean 95% CI | Better/Worse/Equal | Exact sign-flip p | Sign-test p |',
        '|---|---:|---:|---:|---:|---|---|---:|---:|',
    ]
    for name, stats in paired.items():
        interval = stats['bootstrap_95_ci_mean']
        lines.append(
            f"| {name} | {stats['n']} | {stats['aggregate_diff']:+.6f} | "
            f"{stats['mean_diff']:+.6f} | {stats['median_diff']:+.6f} | "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}] | "
            f"{stats['better']}/{stats['worse']}/{stats['equal']} | "
            f"{fmt(stats['exact_sign_flip_p'], False, 4)} | {fmt(stats['exact_sign_test_p'], False, 4)} |"
        )

    e = efficiency
    lines += [
        '',
        '## CPU reference efficiency',
        '',
        '| Quantity | Value |',
        '|---|---:|',
        f"| Extension rows / events | {e['n_extension_rows']} / {e['n_extension_events']} |",
        f"| Accepted extension rows | {e['accepted_extension_rows']} ({100 * e['accepted_extension_fraction']:.2f}%) |",
        f"| Activated events | {e['accepted_events']} ({100 * e['event_activation_fraction']:.2f}%) |",
        f"| Feature extraction total | {e['feature_extract_total_seconds']:.3f} s for {e['feature_extract_rows']} rows |",
        f"| Feature extraction p50 / p95 | {e['feature_extract_ms_p50']:.3f} / {e['feature_extract_ms_p95']:.3f} ms per row |",
        f"| Frozen feature parity max abs diff | {e['feature_stored_max_abs_diff']:.3e} (pass={e['feature_parity_pass']}) |",
        f"| Controller batch p50 / p95 | {e['controller_batch_557_ms_p50']:.3f} / {e['controller_batch_557_ms_p95']:.3f} ms for 557 rows |",
        f"| Event aggregation p50 / p95 | {e['event_aggregation_ms_p50']:.3f} / {e['event_aggregation_ms_p95']:.3f} ms |",
        '',
        e['hardware_note'],
        '',
        '## Decision',
        '',
        f"**{decision['status']}**",
        '',
        decision['interpretation'],
        '',
        '## Claim boundary',
        '',
        '```text',
        'The fixed0.05 dynamic-horizon policy is a positive predeclared prototype result.',
        'The gain over W16 is modest and concentrated, so paired uncertainty must accompany the aggregate metrics.',
        'RGB patch anchors do not establish a complete identity-aware reacquisition solution.',
        'Results are TAP-Vid-DAVIS-first metric-compatible reproduction, not full multi-dataset original Table-1 parity.',
        '```',
    ]
    args.out_doc.write_text('\n'.join(lines) + '\n')

    print(json.dumps({
        'ok': True,
        'json': str(args.out_json),
        'doc': str(args.out_doc),
        'main_AJ_RD_256_delta': main_row['delta_vs_native']['AJ_RD_256'],
        'vs_w16_aggregate': paired['fixed0.05_vs_w16']['aggregate_diff'],
        'vs_w16_mean': paired['fixed0.05_vs_w16']['mean_diff'],
        'vs_w16_ci': paired['fixed0.05_vs_w16']['bootstrap_95_ci_mean'],
        'decision': decision['status'],
        'feature_parity_pass': efficiency['feature_parity_pass'],
        'feature_parity_max_abs_diff': efficiency['feature_stored_max_abs_diff'],
        'factorial_rows': len(factorial_ablation),
        'anchor_rows': len(anchor_ablation),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
