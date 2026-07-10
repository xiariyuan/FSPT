#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a26_trackon2_internal_proxy'
JOINT = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz'
INTERNAL = OUTDIR / 'v9a26_trackon2_internal_proxy_features.npz'
FROZEN = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a26_trackon2_internal_proxy_eval.json'
OUT_DOC = ROOT / 'docs/v9a26_trackon2_internal_proxy_result_2026-07-10.md'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import CANDIDATE, NATIVE, align_candidate, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import apply_touched_mask, per_video_delta, summarize_pv
from scripts.v9a1_controller_calibration_aware_prototype import choose_threshold_by_metric
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint
from scripts.v9a25_eval_dinov3_identity_features import binary_metrics, paired_summary, single_feature_auc


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


def evaluate(name: str, data: dict, native: dict, cand_by: dict, accept: np.ndarray, native_metric: dict, native_pv: list[dict]) -> dict:
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], accept.astype(bool))
    metric, pv = standard_and_ajrd(recs)
    pvr = per_video_delta(native_pv, pv)
    return {'variant': name, 'metric': metric, 'delta_vs_native': metric_delta(metric, native_metric), 'apply_stats': stats, 'extension_stats': extension_stats(data, accept), 'per_video_summary': summarize_pv(pvr), 'per_video_rows': pvr}


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
    splits = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    rows = []
    for model_name, model in models.items():
        score = np.full(len(y), np.nan, dtype=float)
        folds = []
        for fold, (tr, va) in enumerate(splits):
            clf = model.fit(X[tr], y[tr])
            pred = clf.predict_proba(X[va])[:, 1]
            score[va] = pred
            fr = {'fold': int(fold), 'n_train': int(len(tr)), 'n_val': int(len(va)), 'val_videos': sorted(set(groups[va].tolist())), 'positive_rate_val': float(np.mean(y[va]))}
            fr.update(binary_metrics(y[va], pred)); folds.append(fr)
        if not np.all(np.isfinite(score)):
            raise RuntimeError(f'non-finite OOF {feature_set}/{model_name}')
        overall = binary_metrics(y, score)
        overall.update({'threshold_f1': float(choose_threshold_by_metric(y, score, metric='f1')), 'positive_rate': float(np.mean(y)), 'n_rows': int(len(y)), 'n_features': int(X.shape[1])})
        rows.append({'feature_set': feature_set, 'model': model_name, 'scores_ext': score, 'overall': overall, 'folds': folds})
    return rows


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return ''
    return f'{float(x):+.4f}' if signed else f'{float(x):.4f}'


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_joint(JOINT)
    iz = np.load(INTERNAL, allow_pickle=True)
    ext_idx = np.asarray(iz['joint_indices'], dtype=np.int64)
    expected = np.where(data['is_w16_extension'].astype(bool))[0]
    if not np.array_equal(ext_idx, expected):
        raise RuntimeError('internal proxy alignment mismatch')
    X_internal_full = iz['X_internal'].astype(np.float32)
    internal_names_full = [str(x) for x in iz['feature_names'].tolist()]
    # Remove exact constants before standardization/model fitting.
    variable = np.std(X_internal_full, axis=0) > 1e-10
    X_internal = X_internal_full[:, variable]
    internal_names = [n for n, keep in zip(internal_names_full, variable) if keep]
    removed_constants = [n for n, keep in zip(internal_names_full, variable) if not keep]
    y = data['y_candidate_good'][ext_idx].astype(int)
    groups = data['groups'][ext_idx]
    X_base = data['X_base'][ext_idx].astype(np.float32)
    X_all = data['X_all'][ext_idx].astype(np.float32)
    feature_sets = {
        'internal_only': X_internal,
        'base_plus_internal': np.concatenate([X_base, X_internal], axis=1),
        'all_plus_internal': np.concatenate([X_all, X_internal], axis=1),
    }
    models = []
    for name, X in feature_sets.items():
        models.extend(oof_models(X, y, groups, name))

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
            variants[name] = evaluate(name, data, native, cand_by, accept, native_metric, native_pv)
    add('w8_preserve_common_only', common)
    add('w16_accept_all_joint', common | ext)
    add('oracle_ext_candidate_good', common | (ext & y_good))
    add('oracle_ext_good_not_worse', common | (ext & y_good & ~y_worse))

    frozen_report = json.loads(FROZEN.read_text())
    frozen_score = None; frozen_f1 = None
    for block in frozen_report['training']:
        for row in block['overall']:
            if row['feature_set'] == 'all' and row['model'] == 'logreg':
                frozen_score = np.asarray(block['full_scores']['all_logreg'], dtype=float)
                frozen_f1 = float(row['threshold_f1_oof'])
    if frozen_score is None:
        raise RuntimeError('frozen score missing')
    add('frozen_v9a2_fixed0.05', event_accept_mask(data, frozen_score, 0.05, 'event_max'))
    add('frozen_v9a2_oof_f1', event_accept_mask(data, frozen_score, float(frozen_f1), 'event_max'))

    semantic_rows = []
    for m in models:
        full = np.zeros(len(data['metas']), dtype=float); full[ext_idx] = m['scores_ext']
        name = f"v9a26_{m['feature_set']}_{m['model']}_event_max_oof_f1"
        add(name, event_accept_mask(data, full, float(m['overall']['threshold_f1']), 'event_max'))
        semantic_rows.append({'variant': name, 'feature_set': m['feature_set'], 'model': m['model'], 'protocol': 'oof_f1', 'threshold': float(m['overall']['threshold_f1']), 'classifier': m['overall']})
        if m['model'] == 'logreg':
            name2 = f"v9a26_{m['feature_set']}_{m['model']}_event_max_fixed0.05"
            add(name2, event_accept_mask(data, full, 0.05, 'event_max'))
            semantic_rows.append({'variant': name2, 'feature_set': m['feature_set'], 'model': m['model'], 'protocol': 'fixed0.05', 'threshold': 0.05, 'classifier': m['overall']})

    primary_name = 'v9a26_all_plus_internal_logreg_event_max_oof_f1'
    primary = variants[primary_name]
    paired = {
        'primary_vs_frozen_fixed0.05': paired_summary(primary, variants['frozen_v9a2_fixed0.05']),
        'primary_vs_w16': paired_summary(primary, variants['w16_accept_all_joint']),
    }
    result = {
        'protocol': {'primary': primary_name, 'threshold': 'video-heldout OOF-F1', 'event_mode': 'event_max', 'preserve_w8_common': True, 'dense_trajectory_threshold_search': False, 'provenance': 'TrackOn2 256-space M24 support-grid20 internal-state proxy; not exact old-cache latent state'},
        'feature_report': json.loads(INTERNAL.with_suffix('.report.json').read_text()),
        'removed_constant_features': removed_constants,
        'top_single_internal_features': single_feature_auc(X_internal, internal_names, y)[:25],
        'classifiers': [{k: v for k, v in m.items() if k != 'scores_ext'} for m in models],
        'baselines': [variants[x] for x in ['w8_preserve_common_only','w16_accept_all_joint','frozen_v9a2_fixed0.05','frozen_v9a2_oof_f1','oracle_ext_candidate_good']],
        'internal_variants': [{**x, 'result': variants[x['variant']]} for x in semantic_rows],
        'paired_uncertainty': paired,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = ['# V9-A2.6 TrackOn2 Internal-State Proxy Result', '', '## Provenance', '', '```text', result['protocol']['provenance'], '```', '', f"Primary: {primary_name}", '', 'No dense trajectory threshold search was used.', '']
    lines += ['## Classifier OOF', '', '| Feature set / model | Features | AP | AUC | Brier | OOF-F1 threshold |', '|---|---:|---:|---:|---:|---:|']
    for m in models:
        o=m['overall']; lines.append(f"| {m['feature_set']} / {m['model']} | {o['n_features']} | {fmt(o['ap'])} | {fmt(o['auc'])} | {fmt(o['brier'])} | {o['threshold_f1']:.6f} |")
    lines += ['', '## Top single internal features', '', '| Feature | Best AUC | Direction |', '|---|---:|---|']
    for r in result['top_single_internal_features'][:15]:
        lines.append(f"| {r['feature']} | {r['best_auc']:.4f} | {r['direction']} |")

    def table(title: str, rows: list[dict]) -> None:
        lines.extend(['', f'## {title}', '', '| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---:|---:|---:|---|'])
        for row in rows:
            d=row['delta_vs_native']; e=row['extension_stats']; pv=row['per_video_summary']; es=f"{e['accepted_extension']}/{e['accepted_ext_good']}/{e['accepted_ext_bad']}/{e['accepted_ext_false_visible']}/{e['accepted_ext_worse']}"
            lines.append(f"| {row['variant']} | {fmt(d['AJ'],True)} | {fmt(d['OA'],True)} | {fmt(d['delta_avg'],True)} | {fmt(d['delta_4px'],True)} | {fmt(d['AJ_RD'],True)} | {fmt(d['AJ_RD_256'],True)} | {es} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    table('Baselines', result['baselines'])
    table('Internal proxy variants', [x['result'] for x in result['internal_variants']])

    lines += ['', '## Paired video-level AJ_RD_256', '', '| Comparison | N | Mean | Median | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |', '|---|---:|---:|---:|---|---|---:|']
    for name, s in paired.items():
        ci=s['bootstrap_95_ci_mean']; lines.append(f"| {name} | {s['n']} | {s['mean_diff']:+.6f} | {s['median_diff']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {s['better']}/{s['worse']}/{s['equal']} | {fmt(s['exact_sign_flip_p'])} |")

    primary_delta = primary['delta_vs_native']['AJ_RD_256']; frozen_delta = variants['frozen_v9a2_fixed0.05']['delta_vs_native']['AJ_RD_256']
    lines += ['', '## Decision', '']
    if primary_delta > frozen_delta and paired['primary_vs_frozen_fixed0.05']['bootstrap_95_ci_mean'][0] > 0:
        lines.append('TrackOn2 internal proxy features pass strongly and improve the frozen V9-A2 policy with positive paired evidence.')
    elif primary_delta > frozen_delta:
        lines.append('TrackOn2 internal proxy features improve the aggregate frozen V9-A2 policy, but paired evidence remains inconclusive.')
    else:
        lines.append('TrackOn2 internal proxy features do not improve the frozen V9-A2 policy under the predeclared OOF protocol.')
        lines.append('The selector route is saturated; proceed to V9-A3 multi-hypothesis candidate generation / trainable reacquisition.')
    OUT_DOC.write_text('\n'.join(lines))
    print(json.dumps({'ok': True, 'json': str(OUT_JSON), 'doc': str(OUT_DOC), 'primary': primary_name, 'primary_AJ_RD_256_delta': primary_delta, 'frozen_AJ_RD_256_delta': frozen_delta, 'primary_vs_frozen_mean': paired['primary_vs_frozen_fixed0.05']['mean_diff'], 'primary_vs_frozen_ci': paired['primary_vs_frozen_fixed0.05']['bootstrap_95_ci_mean']}))


if __name__ == '__main__':
    main()
