#!/usr/bin/env python3
"""Apply V8-C0.3 OOF fine-risk verifier scores back to trajectories.

This script evaluates whether video-heldout verifier scores actually improve
trajectory metrics when used to filter candidate-visible touched frames.

Baseline to reproduce:
  dist_nc_le64_w8_candidate_visible from V8-C0.1.

Input dataset:
  V8-C0.2 touched-frame dataset built for dist_nc_le64_w8.
  Each row maps to a unique (video, query, frame_tau) touched frame.

Important:
  OOF scores are video-heldout predictions from V8-C0.3. Thresholds here are
  still selected on the same full report for exploration, so this is a policy
  development audit, not a final heldout claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    NATIVE,
    CANDIDATE,
    align_candidate,
    clone_records,
    npy,
    standard_and_ajrd,
    err_px,
)

DEFAULT_DATASET = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
DEFAULT_OOF_DIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c03_fine_risk_verifier'
OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c04_apply_fine_risk_verifier'


def load_dataset(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    out = {
        'metas': metas,
        'labels': {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith('y_')},
        'feature_names': [str(x) for x in z['feature_names'].tolist()],
        'X': z['X'].astype(np.float32),
    }
    return out


def apply_touched_mask(native: dict, cand_by: Dict[str, dict], metas: List[dict], accept: np.ndarray) -> tuple[list[dict], dict]:
    recs = clone_records(native['records'])
    vid_to_i = {str(r['video_id']): i for i, r in enumerate(recs)}
    n_orig_by_vid = {str(r['video_id']): r for r in native['records']}
    stats = {
        'accepted_frames': int(np.sum(accept)),
        'total_touched_candidates': int(len(accept)),
    }
    counters: dict[str, float] = {}
    def inc(k: str, v: float = 1.0):
        counters[k] = counters.get(k, 0.0) + float(v)
    for m, keep in zip(metas, accept):
        if not bool(keep):
            continue
        vid = str(m['video_id']); q = int(m['query_idx']); t = int(m['frame_tau'])
        ri = vid_to_i[vid]
        nr = recs[ri]
        cr = cand_by[vid]
        orig = n_orig_by_vid[vid]
        pred = nr['pred_tracks']; pv = nr['pred_visibility']
        ctracks = npy(cr['pred_tracks'], np.float32)
        cvis = npy(cr['pred_visibility'], bool)
        gt = npy(nr['gt_tracks'], np.float32)
        gv = npy(nr['gt_visibility'], bool)
        ntracks = npy(orig['pred_tracks'], np.float32)
        if not bool(cvis[q, t]):
            # Should not happen for this dataset, but keep guard.
            inc('skipped_not_candidate_visible')
            continue
        if bool(gv[q, t]):
            ne = err_px(ntracks[q, t], gt[q, t])
            ce = err_px(ctracks[q, t], gt[q, t])
            inc('gt_visible_accepted')
            inc('native_err_sum', ne)
            inc('candidate_err_sum', ce)
            for th in [4, 8, 16]:
                inc(f'repair{th}', float(ne > th and ce <= th))
                inc(f'damage{th}', float(ne <= th and ce > th))
                inc(f'candidate_safe{th}', float(ce <= th))
                inc(f'native_safe{th}', float(ne <= th))
            inc('candidate_better_px', float(ce + 1e-6 < ne))
            inc('candidate_worse_px', float(ce > ne + 1e-6))
        else:
            inc('gt_invisible_accepted')
            inc('false_visible_accepted')
        pred[q, t] = ctracks[q, t]
        pv[q, t] = True
    stats.update({k: int(v) if abs(v - round(v)) < 1e-9 else float(v) for k, v in counters.items()})
    if stats.get('accepted_frames', 0):
        stats['accept_rate'] = stats['accepted_frames'] / float(len(accept))
        stats['false_visible_rate'] = stats.get('false_visible_accepted', 0) / float(stats['accepted_frames'])
    if stats.get('gt_visible_accepted', 0):
        den = float(stats['gt_visible_accepted'])
        stats['mean_native_err_px'] = stats.get('native_err_sum', 0.0) / den
        stats['mean_candidate_err_px'] = stats.get('candidate_err_sum', 0.0) / den
        for th in [4, 8, 16]:
            stats[f'repair{th}_rate'] = stats.get(f'repair{th}', 0) / den
            stats[f'damage{th}_rate'] = stats.get(f'damage{th}', 0) / den
            stats[f'candidate_safe{th}_rate'] = stats.get(f'candidate_safe{th}', 0) / den
            stats[f'native_safe{th}_rate'] = stats.get(f'native_safe{th}', 0) / den
        stats['candidate_better_px_rate'] = stats.get('candidate_better_px', 0) / den
        stats['candidate_worse_px_rate'] = stats.get('candidate_worse_px', 0) / den
    return recs, stats


def per_video_delta(native_pv: list[dict], var_pv: list[dict]) -> list[dict]:
    base = {r['video_id']: r for r in native_pv}
    out = []
    for r in var_pv:
        b = base.get(r['video_id'])
        if not b:
            continue
        row = {'video_id': r['video_id']}
        for k in ['AJ', 'OA', 'AJ_RD', 'AJ_RD_256']:
            row[f'delta_{k}'] = None if r.get(k) is None or b.get(k) is None else float(r[k] - b[k])
        out.append(row)
    return out


def summarize_pv(rows: list[dict], key: str = 'delta_AJ_RD_256') -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {
        'n': int(len(vals)),
        'positive': int(sum(v > 1e-9 for v in vals)),
        'negative': int(sum(v < -1e-9 for v in vals)),
        'zero': int(sum(abs(v) <= 1e-9 for v in vals)),
        'mean': float(np.mean(vals)) if vals else None,
        'median': float(np.median(vals)) if vals else None,
        'worst': sorted(rows, key=lambda r: 999 if r.get(key) is None else r[key])[:8],
        'best': sorted(rows, key=lambda r: -999 if r.get(key) is None else r[key], reverse=True)[:8],
    }


def label_stats(labels: dict[str, np.ndarray], accept: np.ndarray) -> dict:
    out = {'accepted': int(np.sum(accept)), 'total': int(len(accept))}
    if out['accepted'] == 0:
        return out
    for k in ['candidate_good', 'candidate_bad', 'false_visible', 'candidate_worse_px', 'candidate_better_px', 'repair4', 'damage4', 'repair8', 'damage8', 'repair16', 'damage16']:
        if k in labels:
            arr = labels[k].astype(float)
            out[k + '_rate'] = float(np.nanmean(arr[accept]))
            out[k + '_count'] = int(np.nansum(arr[accept]))
    if 'utility' in labels:
        out['utility_mean'] = float(np.nanmean(labels['utility'][accept]))
        out['utility_median'] = float(np.nanmedian(labels['utility'][accept]))
    return out


def build_acceptance_rules(scores: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    good_rf = scores['good_rf']
    good_et = scores['good_extratrees']
    false_lr = scores['false_visible_logreg']
    bads_et = scores['bad_strict_extratrees']
    util_et = scores['utility_extratrees']
    rules: list[tuple[str, np.ndarray]] = []
    n = len(good_rf)
    rules.append(('accept_all_baseline', np.ones(n, dtype=bool)))
    # Fixed thresholds.
    for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        rules.append((f'good_rf_ge_{th:.2f}', good_rf >= th))
        rules.append((f'good_et_ge_{th:.2f}', good_et >= th))
    for th in [0.20, 0.30, 0.40, 0.50, 0.60]:
        rules.append((f'false_lr_le_{th:.2f}', false_lr <= th))
    for gth in [0.50, 0.60, 0.70]:
        for fth in [0.30, 0.40, 0.50]:
            rules.append((f'good_rf_ge_{gth:.2f}_false_lr_le_{fth:.2f}', (good_rf >= gth) & (false_lr <= fth)))
    combo = good_rf - false_lr
    for th in np.unique(np.quantile(combo, np.linspace(0.05, 0.95, 19))):
        rules.append((f'good_rf_minus_false_lr_ge_{th:.4f}', combo >= th))
    combo2 = good_rf - bads_et
    for th in np.unique(np.quantile(combo2, np.linspace(0.05, 0.95, 19))):
        rules.append((f'good_rf_minus_badstrict_et_ge_{th:.4f}', combo2 >= th))
    for th in np.unique(np.quantile(util_et, np.linspace(0.05, 0.95, 19))):
        rules.append((f'utility_et_ge_{th:.4f}', util_et >= th))
    # Deduplicate identical masks by bytes + name keep first occurrence.
    seen = set(); out = []
    for name, mask in rules:
        mask = np.asarray(mask, dtype=bool)
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key); out.append((name, mask))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', default=str(NATIVE))
    ap.add_argument('--candidate', default=str(CANDIDATE))
    ap.add_argument('--dataset', default=str(DEFAULT_DATASET))
    ap.add_argument('--oof-dir', default=str(DEFAULT_OOF_DIR))
    ap.add_argument('--outdir', default=str(OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    native = torch.load(args.native, map_location='cpu', weights_only=False)
    cand = torch.load(args.candidate, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed {align_info}')
    ds = load_dataset(Path(args.dataset))
    labels = ds['labels']; metas = ds['metas']
    oof_dir = Path(args.oof_dir)
    scores = {
        'good_rf': np.load(oof_dir / 'oof_good_rf.npy'),
        'good_extratrees': np.load(oof_dir / 'oof_good_extratrees.npy'),
        'false_visible_logreg': np.load(oof_dir / 'oof_false_visible_logreg.npy'),
        'bad_strict_extratrees': np.load(oof_dir / 'oof_bad_strict_false_or_worse_extratrees.npy'),
        'utility_extratrees': np.load(oof_dir / 'oof_utility_extratrees.npy'),
    }
    n = len(metas)
    for k, v in scores.items():
        if len(v) != n:
            raise RuntimeError(f'oof length mismatch {k}: {len(v)} vs {n}')
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))
    cand_metric, cand_pv = standard_and_ajrd(clone_records(cand['records']))
    rows = []
    for name, accept in build_acceptance_rules(scores, labels):
        recs, stats = apply_touched_mask(native, cand_by, metas, accept)
        metric, pv = standard_and_ajrd(recs)
        delta = {k: (metric[k] - native_metric[k]) if metric.get(k) is not None and native_metric.get(k) is not None else None for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']}
        pv_rows = per_video_delta(native_pv, pv)
        rows.append({
            'variant': name,
            'metric': metric,
            'delta_vs_native': delta,
            'apply_stats': stats,
            'label_stats_on_accepted': label_stats(labels, accept),
            'per_video_delta': pv_rows,
            'per_video_summary': summarize_pv(pv_rows),
        })
    def key(row: dict, metric='AJ_RD_256') -> float:
        v = row['delta_vs_native'].get(metric)
        return -999 if v is None else float(v)
    best_ajrd = sorted(rows, key=lambda r: key(r), reverse=True)[:30]
    best_safe = sorted([r for r in rows if key(r, 'AJ') >= -0.10 and key(r, 'OA') >= -0.10], key=lambda r: key(r), reverse=True)[:30]
    best_robust = sorted(rows, key=lambda r: (r['per_video_summary']['positive'] - r['per_video_summary']['negative'], key(r)), reverse=True)[:30]
    report = {
        'script': 'scripts/eval_cotracker3_online_v8c04_apply_fine_risk_verifier.py',
        'native': str(args.native),
        'candidate': str(args.candidate),
        'dataset': str(args.dataset),
        'oof_dir': str(args.oof_dir),
        'alignment': align_info,
        'native_metric': native_metric,
        'trackon2_metric': cand_metric,
        'trackon2_delta_vs_native': {k: (cand_metric[k] - native_metric[k]) if cand_metric.get(k) is not None and native_metric.get(k) is not None else None for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']},
        'score_files': {k: str(oof_dir / f) for k, f in {
            'good_rf': 'oof_good_rf.npy',
            'good_extratrees': 'oof_good_extratrees.npy',
            'false_visible_logreg': 'oof_false_visible_logreg.npy',
            'bad_strict_extratrees': 'oof_bad_strict_false_or_worse_extratrees.npy',
            'utility_extratrees': 'oof_utility_extratrees.npy',
        }.items()},
        'rows': rows,
        'best_by_AJ_RD_256': best_ajrd,
        'best_safe_AJ_OA': best_safe,
        'best_robust_pos_minus_neg': best_robust,
    }
    out = outdir / 'v8c04_apply_fine_risk_verifier_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    def compact(r: dict) -> dict:
        return {
            'variant': r['variant'],
            'delta': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r['delta_vs_native'].items()},
            'accepted': r['apply_stats'].get('accepted_frames'),
            'accept_rate': round(r['apply_stats'].get('accept_rate', 0), 4) if 'accept_rate' in r['apply_stats'] else None,
            'label_good_rate': round(r['label_stats_on_accepted'].get('candidate_good_rate', 0), 4) if r['label_stats_on_accepted'].get('accepted') else None,
            'label_bad_rate': round(r['label_stats_on_accepted'].get('candidate_bad_rate', 0), 4) if r['label_stats_on_accepted'].get('accepted') else None,
            'false_visible_rate': round(r['apply_stats'].get('false_visible_rate', 0), 4) if 'false_visible_rate' in r['apply_stats'] else None,
            'pv': {k: r['per_video_summary'].get(k) for k in ['positive','negative','zero','mean','median']},
        }
    print(json.dumps({
        'out': str(out),
        'trackon2_delta': {k: round(v,4) if isinstance(v,float) else v for k,v in report['trackon2_delta_vs_native'].items()},
        'baseline_accept_all': compact(next(r for r in rows if r['variant'] == 'accept_all_baseline')),
        'best_by_AJ_RD_256': [compact(r) for r in best_ajrd[:12]],
        'best_safe_AJ_OA': [compact(r) for r in best_safe[:12]],
        'best_robust': [compact(r) for r in best_robust[:12]],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
