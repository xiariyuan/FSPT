#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a25_dinov3_identity'
JOINT = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz'
DINO = OUTDIR / 'v9a25_dinov3_identity_features.npz'
FROZEN = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a25_dinov3_identity_eval.json'
OUT_DOC = ROOT / 'docs/v9a25_dinov3_identity_feature_pilot_result_2026-07-10.md'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    CANDIDATE,
    NATIVE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import (
    apply_touched_mask,
    per_video_delta,
    summarize_pv,
)
from scripts.v9a1_controller_calibration_aware_prototype import choose_threshold_by_metric
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint


def metric_delta(metric: dict, base: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: None if metric.get(k) is None or base.get(k) is None else float(metric[k] - base[k]) for k in keys}


def extension_stats(data: dict, accept: np.ndarray) -> dict:
    ext = data['is_w16_extension'].astype(bool)
    return {
        'accepted_extension': int(np.sum(accept & ext)),
        'accepted_ext_good': int(np.sum(accept & ext & data['y_candidate_good'].astype(bool))),
        'accepted_ext_bad': int(np.sum(accept & ext & data['y_candidate_bad'].astype(bool))),
        'accepted_ext_false_visible': int(np.sum(accept & ext & data['y_false_visible'].astype(bool))),
        'accepted_ext_worse': int(np.sum(accept & ext & data['y_candidate_worse_px'].astype(bool))),
    }


def eval_variant(name: str, data: dict, native: dict, cand_by: dict, accept: np.ndarray, native_metric: dict, native_pv: list[dict]) -> dict:
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], accept.astype(bool))
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    return {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'extension_stats': extension_stats(data, accept.astype(bool)),
        'per_video_summary': summarize_pv(pv_rows),
        'per_video_rows': pv_rows,
    }


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    out = {
        'ap': float(average_precision_score(y, score)),
        'brier': float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    try:
        out['auc'] = float(roc_auc_score(y, score))
    except ValueError:
        out['auc'] = None
    return out


def oof_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_set: str) -> list[dict]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')),
        'extratrees': ExtraTreesClassifier(n_estimators=700, min_samples_leaf=3, class_weight='balanced', random_state=0, n_jobs=-1),
    }
    outputs = []
    splits = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    for model_name, model in models.items():
        oof = np.full(len(y), np.nan, dtype=float)
        folds = []
        for fold, (tr, va) in enumerate(splits):
            clf = model.fit(X[tr], y[tr])
            score = clf.predict_proba(X[va])[:, 1]
            oof[va] = score
            row = {
                'fold': int(fold),
                'n_train': int(len(tr)),
                'n_val': int(len(va)),
                'val_videos': sorted(set(groups[va].tolist())),
                'positive_rate_val': float(np.mean(y[va])),
            }
            row.update(binary_metrics(y[va], score))
            folds.append(row)
        if not np.all(np.isfinite(oof)):
            raise RuntimeError(f'non-finite OOF for {feature_set}/{model_name}')
        overall = binary_metrics(y, oof)
        overall.update({
            'threshold_f1': float(choose_threshold_by_metric(y, oof, metric='f1')),
            'positive_rate': float(np.mean(y)),
            'n_rows': int(len(y)),
            'n_features': int(X.shape[1]),
        })
        outputs.append({'feature_set': feature_set, 'model': model_name, 'scores_ext': oof, 'overall': overall, 'folds': folds})
    return outputs


def single_feature_auc(X: np.ndarray, names: list[str], y: np.ndarray) -> list[dict]:
    from sklearn.metrics import roc_auc_score
    rows = []
    for i, name in enumerate(names):
        s = X[:, i]
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            continue
        rows.append({'feature': name, 'auc_positive_high': auc, 'best_auc': max(auc, 1.0 - auc), 'direction': 'high' if auc >= 0.5 else 'low'})
    return sorted(rows, key=lambda r: r['best_auc'], reverse=True)


def pv_map(row: dict) -> dict[str, dict]:
    return {str(r['video_id']): r for r in row['per_video_rows']}


def paired_summary(a: dict, b: dict, seed: int = 20260710, n_boot: int = 200000) -> dict:
    am = pv_map(a); bm = pv_map(b)
    rows = []
    for vid in sorted(set(am) & set(bm)):
        av = am[vid].get('delta_AJ_RD_256'); bv = bm[vid].get('delta_AJ_RD_256')
        if av is None or bv is None:
            continue
        rows.append({'video_id': vid, 'a': float(av), 'b': float(bv), 'diff': float(av - bv)})
    d = np.asarray([r['diff'] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 10000):
        n = min(10000, n_boot - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        boot[start:start+n] = d[idx].mean(axis=1)
    tol = 1e-12
    nz = d[np.abs(d) > tol]
    pos = int(np.sum(d > tol)); neg = int(np.sum(d < -tol)); equal = int(np.sum(np.abs(d) <= tol))
    if len(nz) == 0:
        flip_p = None
    elif len(nz) <= 20:
        observed = abs(float(np.mean(nz)))
        vals = [abs(float(np.mean(nz * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(nz))]
        flip_p = float(np.mean(np.asarray(vals) >= observed - 1e-15))
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, len(nz)))
        flip_p = float(np.mean(np.abs((signs * nz).mean(axis=1)) >= abs(float(np.mean(nz))) - 1e-15))
    n_nonzero = pos + neg
    if n_nonzero == 0:
        sign_p = None
    else:
        k = min(pos, neg)
        sign_p = float(min(1.0, 2.0 * sum(math.comb(n_nonzero, i) for i in range(k + 1)) / (2 ** n_nonzero)))
    return {
        'n': int(len(d)), 'mean_diff': float(np.mean(d)), 'median_diff': float(np.median(d)),
        'bootstrap_95_ci_mean': [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        'bootstrap_probability_mean_gt_zero': float(np.mean(boot > 0.0)),
        'better': pos, 'worse': neg, 'equal': equal, 'nonzero': int(len(nz)),
        'exact_sign_flip_p': flip_p, 'exact_sign_test_p': sign_p, 'per_video': rows,
    }


def fmt(x: float | None, signed: bool = False, digits: int = 4) -> str:
    if x is None:
        return ''
    return f'{float(x):+.{digits}f}' if signed else f'{float(x):.{digits}f}'


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_joint(JOINT)
    dz = np.load(DINO, allow_pickle=True)
    ext_indices = np.asarray(dz['joint_indices'], dtype=np.int64)
    expected_ext = np.where(data['is_w16_extension'].astype(bool))[0]
    if not np.array_equal(ext_indices, expected_ext):
        raise RuntimeError('DINO rows do not align with joint extension rows')
    X_dino = dz['X_dino'].astype(np.float32)
    dino_names = [str(x) for x in dz['feature_names'].tolist()]
    y = data['y_candidate_good'][ext_indices].astype(int)
    groups = data['groups'][ext_indices]
    X_base = data['X_base'][ext_indices].astype(np.float32)
    X_all = data['X_all'][ext_indices].astype(np.float32)
    sets = {
        'dino_only': X_dino,
        'base_plus_dino': np.concatenate([X_base, X_dino], axis=1),
        'all_plus_dino': np.concatenate([X_all, X_dino], axis=1),
    }
    model_rows = []
    for name, X in sets.items():
        model_rows.extend(oof_models(X, y, groups, name))

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    common = data['is_common_w8'].astype(bool); ext = data['is_w16_extension'].astype(bool)
    y_good = data['y_candidate_good'].astype(bool); y_worse = data['y_candidate_worse_px'].astype(bool)

    variants: dict[str, dict] = {}
    def add(name: str, accept: np.ndarray) -> None:
        if name not in variants:
            variants[name] = eval_variant(name, data, native, cand_by, accept, native_metric, native_pv)
    add('w8_preserve_common_only', common)
    add('w16_accept_all_joint', common | ext)
    add('oracle_ext_candidate_good', common | (ext & y_good))
    add('oracle_ext_good_not_worse', common | (ext & y_good & ~y_worse))

    frozen = json.loads(FROZEN.read_text())
    frozen_scores = None; frozen_thr = None
    for block in frozen['training']:
        for row in block['overall']:
            if row['feature_set'] == 'all' and row['model'] == 'logreg':
                frozen_scores = np.asarray(block['full_scores']['all_logreg'], dtype=float)
                frozen_thr = float(row['threshold_f1_oof'])
    if frozen_scores is None:
        raise RuntimeError('missing frozen all_logreg score')
    add('frozen_v9a2_fixed0.05', event_accept_mask(data, frozen_scores, 0.05, 'event_max'))
    add('frozen_v9a2_oof_f1', event_accept_mask(data, frozen_scores, float(frozen_thr), 'event_max'))

    eval_rows = []
    for m in model_rows:
        full = np.zeros(len(data['metas']), dtype=float)
        full[ext_indices] = m['scores_ext']
        for protocol, threshold in [('oof_f1', m['overall']['threshold_f1'])]:
            name = f"v9a25_{m['feature_set']}_{m['model']}_event_max_{protocol}"
            add(name, event_accept_mask(data, full, float(threshold), 'event_max'))
            eval_rows.append({'variant': name, 'feature_set': m['feature_set'], 'model': m['model'], 'protocol': protocol, 'threshold': float(threshold), 'classifier': m['overall']})
        if m['model'] == 'logreg':
            name = f"v9a25_{m['feature_set']}_{m['model']}_event_max_fixed0.05"
            add(name, event_accept_mask(data, full, 0.05, 'event_max'))
            eval_rows.append({'variant': name, 'feature_set': m['feature_set'], 'model': m['model'], 'protocol': 'fixed0.05', 'threshold': 0.05, 'classifier': m['overall']})

    primary_name = 'v9a25_all_plus_dino_logreg_event_max_oof_f1'
    primary = variants[primary_name]
    paired = {
        'primary_vs_frozen_v9a2_fixed0.05': paired_summary(primary, variants['frozen_v9a2_fixed0.05']),
        'primary_vs_w16': paired_summary(primary, variants['w16_accept_all_joint']),
    }
    top_single = single_feature_auc(X_dino, dino_names, y)[:20]

    result = {
        'protocol': {
            'primary': primary_name,
            'primary_threshold': 'video-heldout OOF-F1',
            'event_mode': 'event_max',
            'preserve_w8_common': True,
            'dense_trajectory_threshold_search': False,
        },
        'dino_feature_report': json.loads(DINO.with_suffix('.report.json').read_text()),
        'top_single_dino_features': top_single,
        'classifiers': [{k: v for k, v in m.items() if k != 'scores_ext'} for m in model_rows],
        'baselines': [variants[x] for x in ['w8_preserve_common_only','w16_accept_all_joint','frozen_v9a2_fixed0.05','frozen_v9a2_oof_f1','oracle_ext_candidate_good']],
        'semantic_variants': [{**x, 'result': variants[x['variant']]} for x in eval_rows],
        'paired_uncertainty': paired,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = ['# V9-A2.5 DINOv3 Identity Feature Pilot Result', '', '## Protocol', '', '```text', f"Primary: {primary_name}", 'Preserve W8 common; control W16 extension; event_max; video-heldout OOF-F1; no dense threshold sweep.', '```', '']
    lines += ['## Classifier OOF', '', '| Feature set / model | Features | AP | AUC | Brier | OOF-F1 threshold |', '|---|---:|---:|---:|---:|---:|']
    for m in model_rows:
        o=m['overall']
        lines.append(f"| {m['feature_set']} / {m['model']} | {o['n_features']} | {fmt(o['ap'])} | {fmt(o['auc'])} | {fmt(o['brier'])} | {o['threshold_f1']:.6f} |")
    lines += ['', '## Top single DINO features', '', '| Feature | Best AUC | Direction |', '|---|---:|---|']
    for row in top_single[:12]:
        lines.append(f"| {row['feature']} | {row['best_auc']:.4f} | {row['direction']} |")

    def add_table(title: str, rows: list[dict]) -> None:
        lines.extend(['', f'## {title}', '', '| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---:|---:|---:|---|'])
        for row in rows:
            d=row['delta_vs_native']; e=row['extension_stats']; pv=row['per_video_summary']
            es=f"{e['accepted_extension']}/{e['accepted_ext_good']}/{e['accepted_ext_bad']}/{e['accepted_ext_false_visible']}/{e['accepted_ext_worse']}"
            lines.append(f"| {row['variant']} | {fmt(d['AJ'],True)} | {fmt(d['OA'],True)} | {fmt(d['delta_avg'],True)} | {fmt(d['delta_4px'],True)} | {fmt(d['AJ_RD'],True)} | {fmt(d['AJ_RD_256'],True)} | {es} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    add_table('Baselines', result['baselines'])
    add_table('Semantic identity variants', [x['result'] for x in result['semantic_variants']])

    lines += ['', '## Paired video-level AJ_RD_256', '', '| Comparison | N | Mean | Median | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |', '|---|---:|---:|---:|---|---|---:|']
    for name, s in paired.items():
        ci=s['bootstrap_95_ci_mean']
        lines.append(f"| {name} | {s['n']} | {s['mean_diff']:+.6f} | {s['median_diff']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {s['better']}/{s['worse']}/{s['equal']} | {fmt(s['exact_sign_flip_p'])} |")

    frozen_delta = variants['frozen_v9a2_fixed0.05']['delta_vs_native']['AJ_RD_256']
    primary_delta = primary['delta_vs_native']['AJ_RD_256']
    lines += ['', '## Decision', '']
    if primary_delta > frozen_delta and paired['primary_vs_frozen_v9a2_fixed0.05']['bootstrap_95_ci_mean'][0] > 0:
        lines.append('Semantic identity passes strongly: it improves the frozen V9-A2 policy with positive paired evidence.')
    elif primary_delta > frozen_delta:
        lines.append('Semantic identity improves the aggregate frozen V9-A2 policy, but paired evidence is not yet conclusive.')
    else:
        lines.append('Semantic DINOv3 identity does not improve the frozen V9-A2 dynamic-horizon prototype under the predeclared OOF protocol.')
        lines.append('Do not scale this DINO feature route; move to TrackOn2 internal features or V9-A3 multi-hypothesis candidate generation.')
    OUT_DOC.write_text('\n'.join(lines))
    print(json.dumps({
        'ok': True,
        'json': str(OUT_JSON),
        'doc': str(OUT_DOC),
        'primary': primary_name,
        'primary_AJ_RD_256_delta': primary_delta,
        'frozen_AJ_RD_256_delta': frozen_delta,
        'primary_vs_frozen_mean': paired['primary_vs_frozen_v9a2_fixed0.05']['mean_diff'],
        'primary_vs_frozen_ci': paired['primary_vs_frozen_v9a2_fixed0.05']['bootstrap_95_ci_mean'],
    }))


if __name__ == '__main__':
    main()
