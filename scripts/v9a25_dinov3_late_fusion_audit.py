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
V9A2_DIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
OUTDIR = BASE / 'v9a25_dinov3_identity'
JOINT = V9A2_DIR / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
DINO = OUTDIR / 'v9a25_dinov3_identity_features.npz'
FROZEN = V9A2_DIR / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a25_dinov3_late_fusion_audit.json'
OUT_DOC = ROOT / 'docs/v9a25_dinov3_late_fusion_audit_result_2026-07-10.md'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import CANDIDATE, NATIVE, align_candidate, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import apply_touched_mask, per_video_delta, summarize_pv
from scripts.v9a1_controller_calibration_aware_prototype import choose_threshold_by_metric
from scripts.v9a2_dynamic_horizon_controller import event_accept_mask, load_joint
from scripts.v9a25_eval_dinov3_identity_features import binary_metrics, paired_summary


def metric_delta(metric: dict, base: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: None if metric.get(k) is None or base.get(k) is None else float(metric[k] - base[k]) for k in keys}


def ext_stats(data: dict, accept: np.ndarray) -> dict:
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
    return {'variant': name, 'metric': metric, 'delta_vs_native': metric_delta(metric, native_metric), 'apply_stats': stats, 'extension_stats': ext_stats(data, accept), 'per_video_summary': summarize_pv(pvr), 'per_video_rows': pvr}


def dino_logreg_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, va in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError('non-finite DINO OOF')
    return oof


def main() -> None:
    data = load_joint(JOINT)
    dz = np.load(DINO, allow_pickle=True)
    ext_idx = np.asarray(dz['joint_indices'], dtype=np.int64)
    ext = data['is_w16_extension'].astype(bool)
    if not np.array_equal(ext_idx, np.where(ext)[0]):
        raise RuntimeError('alignment mismatch')
    y = data['y_candidate_good'][ext_idx].astype(int)
    groups = data['groups'][ext_idx]
    dino_ext = dino_logreg_oof(dz['X_dino'].astype(np.float32), y, groups)

    frozen_report = json.loads(FROZEN.read_text())
    frozen_full = None; frozen_f1 = None
    for block in frozen_report['training']:
        for row in block['overall']:
            if row['feature_set'] == 'all' and row['model'] == 'logreg':
                frozen_full = np.asarray(block['full_scores']['all_logreg'], dtype=float)
                frozen_f1 = float(row['threshold_f1_oof'])
    if frozen_full is None:
        raise RuntimeError('missing frozen scores')
    frozen_ext = frozen_full[ext_idx]

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    common = data['is_common_w8'].astype(bool)

    rows = []
    frozen_accept = event_accept_mask(data, frozen_full, 0.05, 'event_max')
    rows.append(evaluate('frozen_v9a2_fixed0.05', data, native, cand_by, frozen_accept, native_metric, native_pv))
    dino_full = np.zeros(len(data['metas']), dtype=float); dino_full[ext_idx] = dino_ext
    dino_thr = float(choose_threshold_by_metric(y, dino_ext, metric='f1'))
    rows.append(evaluate('dino_only_oof_f1', data, native, cand_by, event_accept_mask(data, dino_full, dino_thr, 'event_max'), native_metric, native_pv))

    fusions = []
    correlations = {
        'row_pearson': float(np.corrcoef(frozen_ext, dino_ext)[0, 1]),
        'row_spearman': None,
    }
    try:
        from scipy.stats import spearmanr
        correlations['row_spearman'] = float(spearmanr(frozen_ext, dino_ext).statistic)
    except Exception:
        pass
    for alpha in [0.25, 0.50, 0.75]:
        # alpha weights the frozen V9-A2 score; (1-alpha) weights DINO.
        fused_ext = alpha * frozen_ext + (1.0 - alpha) * dino_ext
        thr = float(choose_threshold_by_metric(y, fused_ext, metric='f1'))
        full = np.zeros(len(data['metas']), dtype=float); full[ext_idx] = fused_ext
        name = f'late_fusion_frozen{alpha:.2f}_dino{1-alpha:.2f}_event_max_oof_f1'
        row = evaluate(name, data, native, cand_by, event_accept_mask(data, full, thr, 'event_max'), native_metric, native_pv)
        rows.append(row)
        fusions.append({'variant': name, 'alpha_frozen': alpha, 'alpha_dino': 1-alpha, 'threshold_f1': thr, 'classifier': binary_metrics(y, fused_ext), 'result': row})

    by = {r['variant']: r for r in rows}
    paired = {x['variant']: paired_summary(x['result'], by['frozen_v9a2_fixed0.05']) for x in fusions}
    out = {
        'protocol': 'predeclared late fusion weights [0.25, 0.50, 0.75]; video-heldout OOF scores; event_max; OOF-F1; no trajectory threshold sweep',
        'score_correlations': correlations,
        'dino_classifier': {**binary_metrics(y, dino_ext), 'threshold_f1': dino_thr},
        'rows': rows,
        'fusions': fusions,
        'paired_vs_frozen': paired,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def fmt(x): return '' if x is None else f'{float(x):+.4f}'
    lines = ['# V9-A2.5 DINOv3 Late-Fusion Audit', '', out['protocol'], '', f"Row-score Pearson correlation: {correlations['row_pearson']:.4f}", f"Row-score Spearman correlation: {correlations['row_spearman'] if correlations['row_spearman'] is not None else 'n/a'}", '', '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---|']
    for row in rows:
        d=row['delta_vs_native']; e=row['extension_stats']; pv=row['per_video_summary']
        es=f"{e['accepted_extension']}/{e['accepted_ext_good']}/{e['accepted_ext_bad']}/{e['accepted_ext_false_visible']}/{e['accepted_ext_worse']}"
        lines.append(f"| {row['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {es} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    lines += ['', '## Paired against frozen V9-A2 fixed0.05', '', '| Fusion | Mean | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |', '|---|---:|---|---|---:|']
    for name, s in paired.items():
        ci=s['bootstrap_95_ci_mean']
        lines.append(f"| {name} | {s['mean_diff']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {s['better']}/{s['worse']}/{s['equal']} | {s['exact_sign_flip_p'] if s['exact_sign_flip_p'] is not None else ''} |")
    OUT_DOC.write_text('\n'.join(lines))
    best = max(rows, key=lambda r: r['delta_vs_native']['AJ_RD_256'])
    print(json.dumps({'ok': True, 'json': str(OUT_JSON), 'doc': str(OUT_DOC), 'best': best['variant'], 'best_AJ_RD_256_delta': best['delta_vs_native']['AJ_RD_256'], 'correlations': correlations}))


if __name__ == '__main__':
    main()
