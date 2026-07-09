#!/usr/bin/env python3
"""V8-C0.1 recovery risk audit.

This script audits the strict-causal V8-C0 result with three goals:

1. Rebuild native event proposals directly from the native cache and compare
   them with the meta event list used by V8-C0.
2. Re-evaluate the strongest simple recovery modes on both meta and rebuilt
   event lists.
3. Audit negative videos and simple risk filters around w8/w16 candidate-visible
   recovery.

No X/X_aug, V7 OOF scores, or oracle labels are used for policy selection.
GT is used only for evaluation and post-hoc damage analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

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
    apply_events,
    bool_get,
    build_events,
    clone_records,
    err_px,
    npy,
    run_len_backward,
    standard_and_ajrd,
)

OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c01_recovery_risk_audit'


def event_key(e: Event) -> tuple[str, int, int]:
    return (e.video_id, int(e.query_idx), int(e.t))


def meta_key(m: dict) -> tuple[str, int, int]:
    return (str(m['video_id']), int(m['query_idx']), int(m['frame_t']))


def rebuild_events_from_native(
    native: dict,
    *,
    support_score: float = 0.80,
    low_thr: float = 0.60,
    min_score: float = 0.05,
    trend_min: float = -0.10,
    min_low_run: int = 1,
    trigger_stride: int = 2,
    event_cooldown: int = 2,
) -> list[dict]:
    """Rebuild causal event proposals following the old event builder logic.

    This returns meta-like dicts without any future labels.
    """
    metas: list[dict] = []
    for vi, r in enumerate(native['records']):
        vid = str(r['video_id'])
        pred_vis = npy(r['pred_visibility'], bool)
        score = npy(r.get('pred_vis_score', pred_vis.astype(np.float32)), np.float32)
        qpts = npy(r['query_points'], np.float32)
        N, T = pred_vis.shape
        low_mask = (score <= low_thr) | (~pred_vis)
        support_mask = pred_vis & (score >= support_score)
        for q in range(N):
            q_t = int(round(float(qpts[q, 0])))
            support_candidates = np.where(support_mask[q])[0]
            next_allowed = max(q_t + 1, 1)
            for t in range(max(q_t + 1, 1), T):
                if t < next_allowed:
                    continue
                if trigger_stride > 1 and ((t - q_t) % trigger_stride) != 0:
                    continue
                if not bool(low_mask[q, t]):
                    continue
                if float(score[q, t]) < min_score:
                    continue
                if t > 0 and float(score[q, t] - score[q, t - 1]) < trend_min:
                    continue
                support_before = support_candidates[support_candidates < t]
                if support_before.size == 0:
                    continue
                sup_t = int(support_before[-1])
                low_run = run_len_backward(low_mask, q, t)
                invis_run = run_len_backward(~pred_vis, q, t)
                if low_run < min_low_run and invis_run < min_low_run:
                    continue
                last_vis = np.where(pred_vis[q, :t])[0]
                last_t = int(last_vis[-1]) if last_vis.size else sup_t
                metas.append({
                    'video_id': vid,
                    'video_index': vi,
                    'query_idx': int(q),
                    'frame_t': int(t),
                    'query_t': int(q_t),
                    'support_t': int(sup_t),
                    'last_visible_t': int(last_t),
                    'low_run': int(low_run),
                    'invis_run': int(invis_run),
                    'native_visible_t': bool(pred_vis[q, t]),
                    'score_t': float(score[q, t]),
                })
                next_allowed = t + max(event_cooldown, 1)
    return metas


def compare_event_lists(meta_metas: list[dict], rebuilt_metas: list[dict]) -> dict:
    old = {meta_key(m) for m in meta_metas}
    new = {meta_key(m) for m in rebuilt_metas}
    inter = old & new
    only_old = old - new
    only_new = new - old
    return {
        'n_meta': int(len(old)),
        'n_rebuilt': int(len(new)),
        'n_intersection': int(len(inter)),
        'n_only_meta': int(len(only_old)),
        'n_only_rebuilt': int(len(only_new)),
        'jaccard': float(len(inter) / max(len(old | new), 1)),
        'only_meta_examples': [list(x) for x in sorted(only_old)[:20]],
        'only_rebuilt_examples': [list(x) for x in sorted(only_new)[:20]],
    }


def per_video_delta(native_per_video: list[dict], variant_per_video: list[dict]) -> list[dict]:
    base = {x['video_id']: x for x in native_per_video}
    rows = []
    for pv in variant_per_video:
        vid = pv['video_id']
        b = base.get(vid)
        if not b:
            continue
        row = {'video_id': vid}
        for k in ['AJ', 'OA', 'AJ_RD', 'AJ_RD_256']:
            if pv.get(k) is None or b.get(k) is None:
                row[f'delta_{k}'] = None
            else:
                row[f'delta_{k}'] = float(pv[k] - b[k])
        row['variant_AJ_RD_256'] = pv.get('AJ_RD_256')
        row['native_AJ_RD_256'] = b.get('AJ_RD_256')
        rows.append(row)
    return rows


def summarize_per_video_deltas(rows: list[dict], key: str = 'delta_AJ_RD_256') -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {
        'n': int(len(vals)),
        'positive': int(sum(v > 1e-9 for v in vals)),
        'negative': int(sum(v < -1e-9 for v in vals)),
        'zero': int(sum(abs(v) <= 1e-9 for v in vals)),
        'mean': float(np.mean(vals)) if vals else None,
        'median': float(np.median(vals)) if vals else None,
        'worst': sorted(rows, key=lambda r: 999 if r.get(key) is None else r[key])[:10],
        'best': sorted(rows, key=lambda r: -999 if r.get(key) is None else r[key], reverse=True)[:10],
    }


def select_events(events: list[Event], name: str) -> np.ndarray:
    """Risk filters for audit. All are causal."""
    if name == 'all':
        return np.ones(len(events), dtype=bool)
    if name == 'score_lt_0p35':
        return np.asarray([e.score_t < 0.35 for e in events], bool)
    if name == 'score_lt_0p50':
        return np.asarray([e.score_t < 0.50 for e in events], bool)
    if name == 'score_lt_0p55':
        return np.asarray([e.score_t < 0.55 for e in events], bool)
    if name == 'score_lt_0p50_or_longrun8':
        return np.asarray([(e.score_t < 0.50) or (e.low_run >= 8) for e in events], bool)
    if name == 'cand_vis_t':
        return np.asarray([e.cand_visible_t for e in events], bool)
    if name == 'cand_visrun_ge2':
        return np.asarray([e.cand_vis_run >= 2 for e in events], bool)
    if name == 'cand_speed_le32':
        return np.asarray([e.cand_speed_px <= 32.0 for e in events], bool)
    if name == 'cand_speed_le64':
        return np.asarray([e.cand_speed_px <= 64.0 for e in events], bool)
    if name == 'dist_nc_le64':
        return np.asarray([e.dist_nc_px <= 64.0 for e in events], bool)
    if name == 'dist_nc_4_64':
        return np.asarray([4.0 <= e.dist_nc_px <= 64.0 for e in events], bool)
    if name == 'dist_nc_8_128':
        return np.asarray([8.0 <= e.dist_nc_px <= 128.0 for e in events], bool)
    if name == 'all_cand_speed64_dist128':
        return np.asarray([e.cand_speed_px <= 64.0 and e.dist_nc_px <= 128.0 for e in events], bool)
    if name == 'all_anchor_step64':
        return np.asarray([e.cand_step_from_last_per_gap <= 64.0 for e in events], bool)
    if name == 'all_anchor_cos_nonneg':
        return np.asarray([e.anchor_cos >= 0.0 for e in events], bool)
    if name == 'safe_motion_combo':
        return np.asarray([
            e.cand_speed_px <= 64.0 and e.dist_nc_px <= 128.0 and e.cand_step_from_last_per_gap <= 64.0
            for e in events
        ], bool)
    raise ValueError(f'unknown filter {name}')


def collect_event_damage_details(
    native_records: list[dict],
    cand_by_vid: dict[str, dict],
    events: list[Event],
    select: np.ndarray,
    *,
    window: int,
    apply_mode: str,
    video_filter: set[str] | None = None,
) -> list[dict]:
    """Detailed touched-frame stats by event/query for negative-video audit.

    Unique frame accounting is used within each event. It intentionally does not
    deduplicate across different events, because we want to see which event is
    risky.
    """
    n_by = {str(r['video_id']): r for r in native_records}
    rows: list[dict] = []
    for ev, keep in zip(events, select):
        if not bool(keep):
            continue
        if video_filter is not None and ev.video_id not in video_filter:
            continue
        nr = n_by[ev.video_id]
        cr = cand_by_vid[ev.video_id]
        n_pred = npy(nr['pred_tracks'], np.float32)
        n_vis = npy(nr['pred_visibility'], bool)
        c_pred = npy(cr['pred_tracks'], np.float32)
        c_vis = npy(cr['pred_visibility'], bool)
        gt = npy(nr['gt_tracks'], np.float32)
        gv = npy(nr['gt_visibility'], bool)
        q = ev.query_idx
        t0 = ev.t
        t1 = min(n_pred.shape[1], t0 + int(window) + 1)
        st = Counter()
        for t in range(t0, t1):
            if apply_mode == 'candidate_visible':
                do_touch = bool(c_vis[q, t]); out_visible = True
            elif apply_mode == 'copy_candidate_visibility':
                do_touch = True; out_visible = bool(c_vis[q, t])
            elif apply_mode == 'force_all':
                do_touch = True; out_visible = True
            else:
                do_touch = bool(c_vis[q, t]); out_visible = True
            if not do_touch:
                continue
            st['touched'] += 1
            st['cand_vis_touched'] += int(bool(c_vis[q, t]))
            if bool(gv[q, t]):
                st['gt_visible_touched'] += 1
                ne = err_px(n_pred[q, t], gt[q, t])
                ce = err_px(c_pred[q, t], gt[q, t])
                st['sum_native_err'] += ne
                st['sum_candidate_err'] += ce
                st['repair16'] += int(ne > 16.0 and ce <= 16.0)
                st['damage16'] += int(ne <= 16.0 and ce > 16.0)
                st['improve'] += int(ce + 1e-6 < ne)
                st['worse'] += int(ce > ne + 1e-6)
            else:
                st['gt_invisible_touched'] += 1
                if out_visible:
                    st['false_visible'] += 1
        if st['touched']:
            row = {
                'video_id': ev.video_id,
                'query_idx': int(ev.query_idx),
                'frame_t': int(ev.t),
                'last_visible_t': int(ev.last_visible_t),
                'score_t': float(ev.score_t),
                'cand_visible_t': bool(ev.cand_visible_t),
                'cand_vis_run': int(ev.cand_vis_run),
                'dist_nc_px': float(ev.dist_nc_px),
                'cand_speed_px': float(ev.cand_speed_px),
                'anchor_cos': float(ev.anchor_cos),
                **{k: int(v) if isinstance(v, (int, np.integer)) else float(v) for k, v in st.items()},
            }
            if st.get('gt_visible_touched', 0):
                den = float(st['gt_visible_touched'])
                row['mean_native_err'] = float(st['sum_native_err'] / den)
                row['mean_candidate_err'] = float(st['sum_candidate_err'] / den)
            rows.append(row)
    return rows


def evaluate_variant(
    native: dict,
    cand_by_vid: dict[str, dict],
    native_metric: dict,
    native_per_video: list[dict],
    events: list[Event],
    filter_name: str,
    window: int,
    apply_mode: str,
) -> dict:
    select = select_events(events, filter_name)
    recs, stats = apply_events(native['records'], cand_by_vid, events, select, window=window, apply_mode=apply_mode)
    metric, per_video = standard_and_ajrd(recs)
    delta = {
        k: (metric[k] - native_metric[k]) if metric.get(k) is not None and native_metric.get(k) is not None else None
        for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    }
    pv_delta = per_video_delta(native_per_video, per_video)
    return {
        'variant': f'{filter_name}_w{window}_{apply_mode}',
        'filter': filter_name,
        'window': int(window),
        'apply_mode': apply_mode,
        'metric': metric,
        'delta_vs_native': delta,
        'stats': stats,
        'per_video_delta': pv_delta,
        'per_video_summary': summarize_per_video_deltas(pv_delta),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', default=str(NATIVE))
    ap.add_argument('--features', default=str(FEATURES))
    ap.add_argument('--candidate', default=str(CANDIDATE))
    ap.add_argument('--outdir', default=str(OUTDIR))
    ap.add_argument('--windows', default='8,16')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in args.windows.split(',') if x.strip()]

    native = torch.load(args.native, map_location='cpu', weights_only=False)
    cand = torch.load(args.candidate, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')

    z = np.load(args.features, allow_pickle=True)
    meta_metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    rebuilt_metas = rebuild_events_from_native(native)
    event_compare = compare_event_lists(meta_metas, rebuilt_metas)

    meta_events = build_events(native, cand_by, meta_metas)
    rebuilt_events = build_events(native, cand_by, rebuilt_metas)

    native_metric, native_per_video = standard_and_ajrd(clone_records(native['records']))
    cand_metric, cand_per_video = standard_and_ajrd(clone_records(cand['records']))
    trackon2_delta = {
        k: (cand_metric[k] - native_metric[k]) if cand_metric.get(k) is not None and native_metric.get(k) is not None else None
        for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    }

    filters = [
        'all',
        'score_lt_0p35',
        'score_lt_0p50',
        'score_lt_0p55',
        'score_lt_0p50_or_longrun8',
        'cand_vis_t',
        'cand_visrun_ge2',
        'cand_speed_le32',
        'cand_speed_le64',
        'dist_nc_le64',
        'dist_nc_4_64',
        'dist_nc_8_128',
        'all_cand_speed64_dist128',
        'all_anchor_step64',
        'all_anchor_cos_nonneg',
        'safe_motion_combo',
    ]
    apply_modes = ['candidate_visible', 'copy_candidate_visibility']
    rows: list[dict] = []
    for source_name, events in [('meta', meta_events), ('rebuilt', rebuilt_events)]:
        for f in filters:
            for w in windows:
                for mode in apply_modes:
                    row = evaluate_variant(native, cand_by, native_metric, native_per_video, events, f, w, mode)
                    row['event_source'] = source_name
                    rows.append(row)

    def dkey(row: dict, key='AJ_RD_256') -> float:
        v = row.get('delta_vs_native', {}).get(key)
        return -999.0 if v is None else float(v)

    best = sorted(rows, key=lambda r: dkey(r), reverse=True)[:30]
    best_safe = sorted([
        r for r in rows
        if dkey(r, 'AJ') >= -0.10 and dkey(r, 'OA') >= -0.10
    ], key=lambda r: dkey(r), reverse=True)[:30]
    # Detailed negative-video event rows for main candidate.
    neg_videos = {'camel', 'shooting', 'breakdance', 'drift-straight', 'car-shadow'}
    main_select = select_events(meta_events, 'all')
    neg_details = collect_event_damage_details(
        native['records'], cand_by, meta_events, main_select, window=16, apply_mode='candidate_visible', video_filter=neg_videos
    )
    neg_by_video: dict[str, list[dict]] = defaultdict(list)
    for r in neg_details:
        neg_by_video[r['video_id']].append(r)
    neg_summary = {}
    for vid, items in neg_by_video.items():
        items_sorted = sorted(items, key=lambda x: (x.get('damage16', 0), x.get('false_visible', 0), x.get('worse', 0)), reverse=True)
        c = Counter()
        for it in items:
            for k in ['touched', 'gt_visible_touched', 'gt_invisible_touched', 'false_visible', 'repair16', 'damage16', 'improve', 'worse']:
                c[k] += int(it.get(k, 0))
        neg_summary[vid] = {
            'totals': dict(c),
            'top_risky_events': items_sorted[:20],
        }

    report = {
        'script': 'scripts/eval_cotracker3_online_v8c01_recovery_risk_audit.py',
        'native': str(args.native),
        'features': str(args.features),
        'candidate': str(args.candidate),
        'alignment': align_info,
        'event_compare_meta_vs_rebuilt': event_compare,
        'native_metric': native_metric,
        'trackon2_metric': cand_metric,
        'trackon2_delta_vs_native': trackon2_delta,
        'windows': windows,
        'filters': filters,
        'apply_modes': apply_modes,
        'rows': rows,
        'best_by_AJ_RD_256': best,
        'best_safe_AJ_OA': best_safe,
        'negative_video_audit_for_meta_all_w16_candidate_visible': neg_summary,
    }
    out = outdir / 'v8c01_recovery_risk_audit_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def compact(r: dict) -> dict:
        return {
            'variant': r['variant'],
            'source': r['event_source'],
            'delta': {k: (round(v, 4) if v is not None else None) for k, v in r['delta_vs_native'].items()},
            'pv': {
                'positive': r.get('per_video_summary', {}).get('positive'),
                'negative': r.get('per_video_summary', {}).get('negative'),
                'zero': r.get('per_video_summary', {}).get('zero'),
                'mean': round(r.get('per_video_summary', {}).get('mean'), 4) if r.get('per_video_summary', {}).get('mean') is not None else None,
            },
            'stats': {
                'selected_events': r.get('stats', {}).get('selected_events'),
                'event_count': r.get('stats', {}).get('event_count'),
                'touched_frames': r.get('stats', {}).get('touched_frames'),
                'false_visible_rate': r.get('stats', {}).get('false_visible_rate_on_touched'),
                'damage16_rate': r.get('stats', {}).get('damage16_rate'),
            }
        }
    print(json.dumps({
        'out': str(out),
        'event_compare': event_compare,
        'trackon2_delta': {k: round(v, 4) if v is not None else None for k, v in trackon2_delta.items()},
        'best_by_AJ_RD_256': [compact(r) for r in best[:12]],
        'best_safe_AJ_OA': [compact(r) for r in best_safe[:12]],
        'negative_video_totals': {k: v['totals'] for k, v in neg_summary.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
