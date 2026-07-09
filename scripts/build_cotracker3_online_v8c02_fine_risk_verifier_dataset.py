#!/usr/bin/env python3
"""Build V8-C0.2 fine-risk verifier dataset.

The dataset is built from frames touched by the current simple recovery mode:

  native causal event proposal + dist_nc<=64 event filter + candidate-visible
  recovery within a finite horizon.

A sample is a UNIQUE touched frame (video, query, frame), not an event-window
occurrence. This avoids overlap bias when multiple event windows touch the same
frame.

Feature rule:
  Only native/candidate outputs at frames <= touched frame tau are used.

Label rule:
  GT is used only to label whether candidate replacement is useful/damaging.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    NATIVE,
    FEATURES,
    CANDIDATE,
    Event,
    align_candidate,
    build_events,
    cosine,
    err_px,
    npy,
    velocity,
)
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native

OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c02_fine_risk_verifier_dataset'


def safe_arr(record: dict, key: str, shape_like: np.ndarray, dtype=np.float32) -> np.ndarray:
    if key in record:
        return npy(record[key], dtype)
    return np.zeros_like(shape_like, dtype=dtype)


def run_len_backward_1d(mask: np.ndarray, t: int) -> int:
    c = 0
    for i in range(t, -1, -1):
        if bool(mask[i]):
            c += 1
        else:
            break
    return c


def local_smoothness(track: np.ndarray, q: int, t: int, k: int = 3) -> tuple[float, float]:
    """Return mean speed and speed std over <=t causal local window in px."""
    speeds = []
    start = max(1, t - k + 1)
    for i in range(start, t + 1):
        speeds.append(float(np.linalg.norm((track[q, i] - track[q, i - 1]) * 255.0)))
    if not speeds:
        return 0.0, 0.0
    return float(np.mean(speeds)), float(np.std(speeds))


def build_event_objects(native: dict, cand_by_vid: Dict[str, dict]) -> list[Event]:
    rebuilt = rebuild_events_from_native(native)
    return build_events(native, cand_by_vid, rebuilt)


def collect_touched_frames(events: list[Event], native: dict, cand_by_vid: Dict[str, dict], *, window: int, event_filter: str) -> tuple[dict, dict]:
    """Collect unique candidate-visible touched frames from causal recovery mode.

    event_filter currently supports:
      all
      dist_nc_le64
      score_lt_0p50_or_longrun8
    """
    def keep_event(e: Event) -> bool:
        if event_filter == 'all':
            return True
        if event_filter == 'dist_nc_le64':
            return e.dist_nc_px <= 64.0
        if event_filter == 'score_lt_0p50_or_longrun8':
            return e.score_t < 0.50 or e.low_run >= 8
        raise ValueError(f'unknown event_filter={event_filter}')

    n_by = {str(r['video_id']): r for r in native['records']}
    touched: dict[tuple[str, int, int], dict] = {}
    stats = Counter()
    for e in events:
        if not keep_event(e):
            continue
        stats['selected_events'] += 1
        nr = n_by[e.video_id]
        cr = cand_by_vid[e.video_id]
        c_vis = npy(cr['pred_visibility'], bool)
        T = c_vis.shape[1]
        q = e.query_idx
        t0 = e.t
        t1 = min(T, t0 + int(window) + 1)
        event_touched = False
        for tau in range(t0, t1):
            if not bool(c_vis[q, tau]):
                continue
            key = (e.video_id, q, tau)
            if key not in touched:
                touched[key] = {
                    'video_id': e.video_id,
                    'query_idx': q,
                    'tau': tau,
                    'first_event_t': t0,
                    'min_event_age': tau - t0,
                    'max_event_age': tau - t0,
                    'touch_source_count': 1,
                    'event_filter': event_filter,
                    'window': int(window),
                }
            else:
                touched[key]['touch_source_count'] += 1
                touched[key]['min_event_age'] = min(touched[key]['min_event_age'], tau - t0)
                touched[key]['max_event_age'] = max(touched[key]['max_event_age'], tau - t0)
            event_touched = True
        if event_touched:
            stats['events_with_touch'] += 1
    stats['unique_touched_frames'] = len(touched)
    return touched, dict(stats)


def build_features_for_touched(native: dict, cand_by_vid: Dict[str, dict], touched: dict[tuple[str, int, int], dict]) -> tuple[np.ndarray, list[str], dict]:
    n_by = {str(r['video_id']): r for r in native['records']}
    rows: list[list[float]] = []
    labels: dict[str, list[Any]] = defaultdict(list)
    metas: list[dict] = []
    feature_names: list[str] | None = None

    for key, info in sorted(touched.items()):
        vid, q, tau = key
        nr = n_by[vid]
        cr = cand_by_vid[vid]
        n_pred = npy(nr['pred_tracks'], np.float32)
        c_pred = npy(cr['pred_tracks'], np.float32)
        n_vis = npy(nr['pred_visibility'], bool)
        c_vis = npy(cr['pred_visibility'], bool)
        gt = npy(nr['gt_tracks'], np.float32)
        gv = npy(nr['gt_visibility'], bool)
        score = safe_arr(nr, 'pred_vis_score', n_vis.astype(np.float32), np.float32)
        raw_v = safe_arr(nr, 'pred_raw_vis_logit', score, np.float32)
        raw_c = safe_arr(nr, 'pred_raw_conf_logit', score, np.float32)
        vp = safe_arr(nr, 'pred_vis_prob_component', score, np.float32)
        cp = safe_arr(nr, 'pred_conf_prob_component', score, np.float32)
        T = n_pred.shape[1]
        # Causal anchor: last native visible before tau.
        last_vis = np.where(n_vis[q, :tau])[0]
        last_t = int(last_vis[-1]) if len(last_vis) else max(0, int(round(float(nr['query_points'][q][0]))))
        gap = max(1, tau - last_t)
        vn = velocity(n_pred, q, tau)
        vc = velocity(c_pred, q, tau)
        pre_v = velocity(n_pred, q, last_t) if last_t > 0 else np.zeros(2, np.float32)
        c_from_anchor = (c_pred[q, tau] - n_pred[q, last_t]).astype(np.float32) * 255.0
        n_speed_mean, n_speed_std = local_smoothness(n_pred, q, tau, 4)
        c_speed_mean, c_speed_std = local_smoothness(c_pred, q, tau, 4)
        low_mask_q = (score[q] <= 0.60) | (~n_vis[q])
        # Features. All causal at <= tau.
        feats: list[float] = []
        names: list[str] = []
        def add(name: str, val: float | int | bool):
            names.append(name); feats.append(float(val))
        add('native_score_tau', score[q, tau])
        add('native_raw_vis_tau', raw_v[q, tau])
        add('native_raw_conf_tau', raw_c[q, tau])
        add('native_vis_prob_tau', vp[q, tau])
        add('native_conf_prob_tau', cp[q, tau])
        add('native_visible_tau', n_vis[q, tau])
        add('candidate_visible_tau', c_vis[q, tau])
        add('native_low_run_tau', run_len_backward_1d(low_mask_q, tau))
        add('native_invis_run_tau', run_len_backward_1d(~n_vis[q], tau))
        add('candidate_vis_run_tau', run_len_backward_1d(c_vis[q], tau))
        add('time_since_last_native_visible', gap)
        add('touch_source_count', info['touch_source_count'])
        add('min_event_age', info['min_event_age'])
        add('max_event_age', info['max_event_age'])
        add('dist_native_candidate_px', err_px(n_pred[q, tau], c_pred[q, tau]))
        add('dist_candidate_to_last_native_px', err_px(c_pred[q, tau], n_pred[q, last_t]))
        add('dist_native_to_last_native_px', err_px(n_pred[q, tau], n_pred[q, last_t]))
        add('native_speed_px', float(np.linalg.norm(vn)))
        add('candidate_speed_px', float(np.linalg.norm(vc)))
        add('velocity_disagree_px', float(np.linalg.norm(vc - vn)))
        add('anchor_cos', cosine(pre_v, c_from_anchor))
        add('candidate_step_from_anchor_per_gap', float(np.linalg.norm(c_from_anchor) / gap))
        add('native_speed_mean4_px', n_speed_mean)
        add('native_speed_std4_px', n_speed_std)
        add('candidate_speed_mean4_px', c_speed_mean)
        add('candidate_speed_std4_px', c_speed_std)
        add('candidate_minus_native_speed_mean4', c_speed_mean - n_speed_mean)
        add('candidate_minus_native_speed_std4', c_speed_std - n_speed_std)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError('feature name mismatch')
        rows.append(feats)

        gt_visible = bool(gv[q, tau])
        native_visible = bool(n_vis[q, tau])
        cand_visible = bool(c_vis[q, tau])
        if gt_visible:
            n_err = err_px(n_pred[q, tau], gt[q, tau])
            c_err = err_px(c_pred[q, tau], gt[q, tau])
        else:
            n_err = np.nan
            c_err = np.nan
        native_j = 0.0
        cand_j = 0.0
        if gt_visible:
            thresholds = [1, 2, 4, 8, 16]
            n_hits = sum(float(n_err <= th) for th in thresholds)
            c_hits = sum(float(c_err <= th) for th in thresholds)
            # Visibility-aware single-frame AJ-like proxy.
            native_j = n_hits / len(thresholds) if native_visible else 0.0
            cand_j = c_hits / len(thresholds) if cand_visible else 0.0
        elif (not gt_visible) and (not native_visible) and (not cand_visible):
            native_j = 1.0
            cand_j = 1.0
        elif (not gt_visible):
            native_j = 1.0 if not native_visible else 0.0
            cand_j = 1.0 if not cand_visible else 0.0
        utility = cand_j - native_j
        false_visible = bool(cand_visible and not gt_visible)
        candidate_better = bool(gt_visible and cand_visible and (c_err + 1e-6 < n_err))
        candidate_worse = bool(gt_visible and cand_visible and (c_err > n_err + 1e-6))
        candidate_good = bool(utility > 0.0)
        candidate_bad = bool(false_visible or utility < 0.0)
        labels['gt_visible'].append(gt_visible)
        labels['native_visible'].append(native_visible)
        labels['candidate_visible'].append(cand_visible)
        labels['native_err_px'].append(float(n_err) if gt_visible else np.nan)
        labels['candidate_err_px'].append(float(c_err) if gt_visible else np.nan)
        labels['err_delta_candidate_minus_native'].append(float(c_err - n_err) if gt_visible else np.nan)
        for th in [4, 8, 16]:
            labels[f'native_safe{th}'].append(bool(gt_visible and n_err <= th))
            labels[f'candidate_safe{th}'].append(bool(gt_visible and c_err <= th))
            labels[f'repair{th}'].append(bool(gt_visible and n_err > th and c_err <= th))
            labels[f'damage{th}'].append(bool(gt_visible and n_err <= th and c_err > th))
        labels['false_visible'].append(false_visible)
        labels['candidate_better_px'].append(candidate_better)
        labels['candidate_worse_px'].append(candidate_worse)
        labels['utility'].append(float(utility))
        labels['candidate_good'].append(candidate_good)
        labels['candidate_bad'].append(candidate_bad)
        metas.append({
            'video_id': vid,
            'query_idx': int(q),
            'frame_tau': int(tau),
            'last_native_visible_t': int(last_t),
            **info,
        })

    X = np.asarray(rows, np.float32)
    return X, feature_names or [], {'labels': labels, 'meta': metas}


def summarize(labels: dict[str, list[Any]], metas: list[dict]) -> dict:
    n = len(metas)
    out: dict[str, Any] = {'n_samples': int(n)}
    for k, vals in labels.items():
        arr = np.asarray(vals)
        if arr.dtype == bool:
            out[k + '_count'] = int(arr.sum())
            out[k + '_rate'] = float(arr.mean()) if n else 0.0
        else:
            arrf = arr.astype(float)
            finite = np.isfinite(arrf)
            if finite.any():
                out[k + '_mean'] = float(np.nanmean(arrf))
                out[k + '_median'] = float(np.nanmedian(arrf))
    by_video = defaultdict(list)
    for i, m in enumerate(metas):
        by_video[m['video_id']].append(i)
    video_rows = {}
    for vid, idxs in by_video.items():
        row = {'n': len(idxs)}
        for k, vals in labels.items():
            arr = np.asarray(vals)[idxs]
            if arr.dtype == bool:
                row[k + '_rate'] = float(arr.mean())
                row[k + '_count'] = int(arr.sum())
        video_rows[vid] = row
    out['by_video'] = video_rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', default=str(NATIVE))
    ap.add_argument('--candidate', default=str(CANDIDATE))
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--event-filter', choices=['all', 'dist_nc_le64', 'score_lt_0p50_or_longrun8'], default='dist_nc_le64')
    ap.add_argument('--outdir', default=str(OUTDIR))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    native = torch.load(args.native, map_location='cpu', weights_only=False)
    cand = torch.load(args.candidate, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')
    events = build_event_objects(native, cand_by)
    touched, touch_stats = collect_touched_frames(events, native, cand_by, window=args.window, event_filter=args.event_filter)
    X, feature_names, payload = build_features_for_touched(native, cand_by, touched)
    labels = payload['labels']
    metas = payload['meta']
    summary = summarize(labels, metas)
    tag = f'v8c02_fine_risk_{args.event_filter}_w{args.window}'
    out_npz = outdir / f'{tag}.npz'
    arrays = {
        'X': X,
        'feature_names': np.asarray(feature_names, dtype=object),
        'meta_json': np.asarray([json.dumps(m, ensure_ascii=False) for m in metas], dtype=object),
    }
    for k, vals in labels.items():
        arr = np.asarray(vals)
        if arr.dtype == bool:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        arrays['y_' + k] = arr
    np.savez_compressed(out_npz, **arrays)
    report = {
        'script': 'scripts/build_cotracker3_online_v8c02_fine_risk_verifier_dataset.py',
        'native': str(args.native),
        'candidate': str(args.candidate),
        'window': int(args.window),
        'event_filter': args.event_filter,
        'alignment': align_info,
        'event_count': int(len(events)),
        'touch_stats': touch_stats,
        'out_npz': str(out_npz),
        'feature_dim': int(X.shape[1]) if X.ndim == 2 else 0,
        'features': feature_names,
        'summary': summary,
    }
    out_report = outdir / f'{tag}_report.json'
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        'out_npz': str(out_npz),
        'out_report': str(out_report),
        'event_count': len(events),
        'touch_stats': touch_stats,
        'feature_dim': int(X.shape[1]) if X.ndim == 2 else 0,
        'summary': {k: v for k, v in summary.items() if k != 'by_video'},
        'negative_video_rates': {vid: summary['by_video'].get(vid) for vid in ['breakdance','camel','shooting','car-shadow','drift-straight']},
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
