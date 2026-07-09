#!/usr/bin/env python3
"""V8-C0 strict-causal recovery baseline ladder.

This script is intentionally conservative:
- It does NOT use X/X_aug from V7-A4 features as policy inputs.
- It does NOT use early4/early8/useful labels for policy selection.
- It does NOT use existing V7 benefit/damage OOF scores because those were
  trained with feature sets containing future/post-window information.

The script uses only causal fields at event_t:
- event meta fields such as low_run, invis_run, last_visible_t, score_t;
- native/candidate outputs at frames <= t;
- native-candidate relation/motion features computed from frames <= t.

It still uses GT after the fact to evaluate metrics and damage, which is valid
for evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
FEATURES = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
CANDIDATE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'
OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c0_causal_recovery_baselines'

EPS = 1e-9


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def clone_records(records: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in records:
        rr = dict(r)
        rr['pred_tracks'] = npy(r['pred_tracks'], np.float32).copy()
        rr['pred_visibility'] = npy(r['pred_visibility'], bool).copy()
        out.append(rr)
    return out


def err_px(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a, np.float32) - np.asarray(b, np.float32)) * 255.0))


def safe_get(arr: np.ndarray, q: int, t: int, default: float = 0.0) -> float:
    if arr is None:
        return float(default)
    if t < 0:
        t = 0
    if t >= arr.shape[1]:
        t = arr.shape[1] - 1
    return float(arr[q, t])


def bool_get(arr: np.ndarray, q: int, t: int, default: bool = False) -> bool:
    if arr is None:
        return bool(default)
    if t < 0:
        t = 0
    if t >= arr.shape[1]:
        t = arr.shape[1] - 1
    return bool(arr[q, t])


def run_len_backward(mask: np.ndarray, q: int, t: int) -> int:
    c = 0
    for i in range(t, -1, -1):
        if bool(mask[q, i]):
            c += 1
        else:
            break
    return c


def velocity(track: np.ndarray, q: int, t: int) -> np.ndarray:
    if t <= 0:
        return np.zeros(2, dtype=np.float32)
    return (track[q, t] - track[q, t - 1]).astype(np.float32) * 255.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb + EPS))


def align_candidate(native: dict, cand: dict) -> Tuple[bool, dict, Dict[str, dict]]:
    c_by = {str(r['video_id']): r for r in cand['records']}
    aligned: Dict[str, dict] = {}
    problems: List[dict] = []
    qp_diffs: List[float] = []
    gt_diffs: List[float] = []
    for nr in native['records']:
        vid = str(nr['video_id'])
        cr = c_by.get(vid)
        if cr is None:
            problems.append({'video_id': vid, 'problem': 'missing_candidate'})
            continue
        nt = npy(nr['pred_tracks']); ct = npy(cr['pred_tracks'])
        if nt.shape != ct.shape:
            problems.append({'video_id': vid, 'problem': 'shape_mismatch', 'native': nt.shape, 'candidate': ct.shape})
            continue
        qp_diffs.append(float(np.max(np.abs(npy(nr['query_points'], np.float32) - npy(cr['query_points'], np.float32)))))
        gt_diffs.append(float(np.max(np.abs(npy(nr['gt_tracks'], np.float32) - npy(cr['gt_tracks'], np.float32)))))
        aligned[vid] = cr
    ok = len(problems) == 0 and len(aligned) == len(native['records']) and max(qp_diffs or [0.0]) < 1e-4
    info = {
        'align_ok': bool(ok),
        'n_aligned': int(len(aligned)),
        'problems': problems[:10],
        'max_query_diff': float(max(qp_diffs or [0.0])),
        'max_gt_diff': float(max(gt_diffs or [0.0])),
    }
    return ok, info, aligned


def standard_and_ajrd(records: List[dict]) -> Tuple[dict, List[dict]]:
    std_vals: List[dict] = []
    per_video: List[dict] = []
    ajrd_vals: List[float] = []
    ajrd_vals_256: List[float] = []
    n_reentry_total = 0
    n_queries = 0
    for r in records:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        qpts = torch.from_numpy(npy(r['query_points'], np.float32))
        n_queries += int(qpts.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, qpts, resolution=256, query_mode='first')
        std = {
            'AJ': float(m['AJ']) * 100.0,
            'OA': float(m['OA']) * 100.0,
            'delta_avg': float(m['average_pts_within_thresh']) * 100.0,
            'delta_4px': float(m['pts_within_4']) * 100.0,
        }
        h, w = int(r['original_size'][0]), int(r['original_size'][1])
        rd = compute_reentry_metrics(
            pred_tracks=npy(r['pred_tracks'], np.float32),
            gt_tracks=npy(r['gt_tracks'], np.float32),
            pred_vis=npy(r['pred_visibility'], bool),
            gt_vis=npy(r['gt_visibility'], bool),
            query_points=npy(r['query_points'], np.float32),
            height=h,
            width=w,
        )
        q_rd = []
        q_rd256 = []
        for q in rd.get('per_query', []):
            if q.get('ajrd_summary', {}).get('aj_rd') is not None:
                q_rd.append(float(q['ajrd_summary']['aj_rd']))
                ajrd_vals.append(float(q['ajrd_summary']['aj_rd']))
            if q.get('ajrd_summary_256', {}).get('aj_rd') is not None:
                q_rd256.append(float(q['ajrd_summary_256']['aj_rd']))
                ajrd_vals_256.append(float(q['ajrd_summary_256']['aj_rd']))
        n_reentry_total += int(rd.get('n_reentry_queries', 0))
        pvrow = {
            'video_id': str(r['video_id']),
            **std,
            'AJ_RD': float(np.mean(q_rd)) if q_rd else None,
            'AJ_RD_256': float(np.mean(q_rd256)) if q_rd256 else None,
            'n_reentry_queries': int(rd.get('n_reentry_queries', 0)),
        }
        per_video.append(pvrow)
        std_vals.append(std)
    agg = {
        'AJ': float(np.mean([v['AJ'] for v in std_vals])) if std_vals else 0.0,
        'OA': float(np.mean([v['OA'] for v in std_vals])) if std_vals else 0.0,
        'delta_avg': float(np.mean([v['delta_avg'] for v in std_vals])) if std_vals else 0.0,
        'delta_4px': float(np.mean([v['delta_4px'] for v in std_vals])) if std_vals else 0.0,
        'AJ_RD': float(np.mean(ajrd_vals)) if ajrd_vals else None,
        'AJ_RD_256': float(np.mean(ajrd_vals_256)) if ajrd_vals_256 else None,
        'n_records': int(len(records)),
        'n_queries': int(n_queries),
        'n_reentry_queries_total': int(n_reentry_total),
    }
    return agg, per_video


@dataclass
class Event:
    idx: int
    video_id: str
    video_i: int
    query_idx: int
    t: int
    query_t: int
    last_visible_t: int
    low_run: int
    invis_run: int
    score_t: float
    native_visible_t: bool
    cand_visible_t: bool
    cand_vis_run: int
    native_score_cache_t: float
    raw_vis_t: float
    raw_conf_t: float
    vis_prob_t: float
    conf_prob_t: float
    dist_nc_px: float
    dist_c_last_px: float
    dist_n_last_px: float
    native_speed_px: float
    cand_speed_px: float
    vel_disagree_px: float
    anchor_cos: float
    cand_step_from_last_per_gap: float


def build_events(native: dict, cand_by_vid: Dict[str, dict], metas: List[dict]) -> List[Event]:
    n_by = {str(r['video_id']): r for r in native['records']}
    events: List[Event] = []
    for i, m in enumerate(metas):
        vid = str(m['video_id'])
        nr = n_by[vid]
        cr = cand_by_vid[vid]
        q = int(m['query_idx']); t = int(m['frame_t'])
        n_pred = npy(nr['pred_tracks'], np.float32)
        c_pred = npy(cr['pred_tracks'], np.float32)
        n_vis = npy(nr['pred_visibility'], bool)
        c_vis = npy(cr['pred_visibility'], bool)
        T = n_pred.shape[1]
        if t < 0 or t >= T:
            continue
        last_t = int(m.get('last_visible_t', max(0, t - 1)))
        last_t = max(0, min(last_t, T - 1))
        q_t = int(m.get('query_t', 0))
        low_run = int(m.get('low_run', 0))
        invis_run = int(m.get('invis_run', 0))
        score_meta = float(m.get('score_t', 0.0))
        score_arr = npy(nr.get('pred_vis_score', np.zeros_like(n_vis, dtype=np.float32)), np.float32)
        raw_vis_arr = npy(nr.get('pred_raw_vis_logit', np.zeros_like(score_arr)), np.float32)
        raw_conf_arr = npy(nr.get('pred_raw_conf_logit', np.zeros_like(score_arr)), np.float32)
        vis_prob_arr = npy(nr.get('pred_vis_prob_component', np.zeros_like(score_arr)), np.float32)
        conf_prob_arr = npy(nr.get('pred_conf_prob_component', np.zeros_like(score_arr)), np.float32)
        vn = velocity(n_pred, q, t)
        vc = velocity(c_pred, q, t)
        pre_v = velocity(n_pred, q, last_t) if last_t > 0 else np.zeros(2, dtype=np.float32)
        c_from_anchor = (c_pred[q, t] - n_pred[q, last_t]).astype(np.float32) * 255.0
        gap = max(1, t - last_t)
        events.append(Event(
            idx=i,
            video_id=vid,
            video_i=int(m.get('video_index', -1)),
            query_idx=q,
            t=t,
            query_t=q_t,
            last_visible_t=last_t,
            low_run=low_run,
            invis_run=invis_run,
            score_t=score_meta,
            native_visible_t=bool(n_vis[q, t]),
            cand_visible_t=bool(c_vis[q, t]),
            cand_vis_run=run_len_backward(c_vis, q, t),
            native_score_cache_t=safe_get(score_arr, q, t, score_meta),
            raw_vis_t=safe_get(raw_vis_arr, q, t, 0.0),
            raw_conf_t=safe_get(raw_conf_arr, q, t, 0.0),
            vis_prob_t=safe_get(vis_prob_arr, q, t, 0.0),
            conf_prob_t=safe_get(conf_prob_arr, q, t, 0.0),
            dist_nc_px=err_px(c_pred[q, t], n_pred[q, t]),
            dist_c_last_px=err_px(c_pred[q, t], n_pred[q, last_t]),
            dist_n_last_px=err_px(n_pred[q, t], n_pred[q, last_t]),
            native_speed_px=float(np.linalg.norm(vn)),
            cand_speed_px=float(np.linalg.norm(vc)),
            vel_disagree_px=float(np.linalg.norm(vc - vn)),
            anchor_cos=cosine(pre_v, c_from_anchor),
            cand_step_from_last_per_gap=float(np.linalg.norm(c_from_anchor) / gap),
        ))
    return events


def apply_events(
    native_records: List[dict],
    cand_by_vid: Dict[str, dict],
    events: List[Event],
    select: np.ndarray,
    *,
    window: int,
    apply_mode: str,
) -> Tuple[List[dict], dict]:
    """Apply selected causal recovery windows on output records.

    apply_mode:
      candidate_visible: at each frame tau in [t, t+window], replace only if candidate visibility at tau is true.
      force_all: replace every frame in [t, t+window] and force output visible.
      entry_force_then_candidate_visible: force event frame t, then candidate_visible for future frames.
      copy_candidate_visibility: replace coords every frame but copy candidate visibility.
    """
    recs = clone_records(native_records)
    vid_to_i = {str(r['video_id']): i for i, r in enumerate(recs)}
    stats = Counter()
    touched = set()
    selected_events = 0
    for ev, keep in zip(events, select):
        if not bool(keep):
            continue
        selected_events += 1
        ri = vid_to_i[ev.video_id]
        nr = recs[ri]
        cr = cand_by_vid[ev.video_id]
        pred = nr['pred_tracks']
        pv = nr['pred_visibility']
        native_orig_pred = npy(native_records[ri]['pred_tracks'], np.float32)
        native_orig_vis = npy(native_records[ri]['pred_visibility'], bool)
        gt = npy(nr['gt_tracks'], np.float32)
        gv = npy(nr['gt_visibility'], bool)
        ctracks = npy(cr['pred_tracks'], np.float32)
        cvis = npy(cr['pred_visibility'], bool)
        q = ev.query_idx
        t0 = ev.t
        t1 = min(pred.shape[1], t0 + int(window) + 1)
        event_touched = False
        for t in range(t0, t1):
            stats['frames_considered'] += 1
            do_touch = False
            out_visible = True
            if apply_mode == 'candidate_visible':
                do_touch = bool(cvis[q, t])
                out_visible = True
            elif apply_mode == 'force_all':
                do_touch = True
                out_visible = True
            elif apply_mode == 'entry_force_then_candidate_visible':
                do_touch = (t == t0) or bool(cvis[q, t])
                out_visible = True
            elif apply_mode == 'copy_candidate_visibility':
                do_touch = True
                out_visible = bool(cvis[q, t])
            else:
                raise ValueError(f'unknown apply_mode {apply_mode}')
            if not do_touch:
                continue
            touch_key = (ri, q, t)
            # Multiple event windows can overlap. The output is idempotent because the
            # same candidate frame is written, but damage/false-visible accounting must
            # be unique per output frame.
            if touch_key in touched:
                continue
            touched.add(touch_key)
            stats['touch_operations_unique'] += 1
            event_touched = True
            # Damage accounting before mutation.
            n_err = err_px(native_orig_pred[q, t], gt[q, t]) if bool(gv[q, t]) else None
            c_err = err_px(ctracks[q, t], gt[q, t]) if bool(gv[q, t]) else None
            if bool(gv[q, t]):
                stats['gt_visible_touched'] += 1
                if c_err is not None:
                    stats['candidate_safe16'] += int(c_err <= 16.0)
                    stats['candidate_safe8'] += int(c_err <= 8.0)
                    stats['candidate_safe4'] += int(c_err <= 4.0)
                if n_err is not None:
                    stats['native_safe16_before_touch'] += int(n_err <= 16.0)
                    stats['native_safe8_before_touch'] += int(n_err <= 8.0)
                if n_err is not None and c_err is not None:
                    stats['improve_px_count'] += int(c_err + 1e-6 < n_err)
                    stats['worse_px_count'] += int(c_err > n_err + 1e-6)
                    stats['repair16_count'] += int(n_err > 16.0 and c_err <= 16.0)
                    stats['damage16_count'] += int(n_err <= 16.0 and c_err > 16.0)
                    stats['repair8_count'] += int(n_err > 8.0 and c_err <= 8.0)
                    stats['damage8_count'] += int(n_err <= 8.0 and c_err > 8.0)
                    stats['sum_native_err_px'] += float(n_err)
                    stats['sum_candidate_err_px'] += float(c_err)
            else:
                stats['gt_invisible_touched'] += 1
                if out_visible:
                    stats['false_visible_touched'] += 1
            if bool(cvis[q, t]):
                stats['candidate_visible_touched'] += 1
            pred[q, t] = ctracks[q, t]
            pv[q, t] = bool(out_visible)
        if event_touched:
            stats['event_count'] += 1
    out = dict(stats)
    out['selected_events'] = int(selected_events)
    out['touched_frames'] = int(len(touched))
    if out.get('gt_visible_touched', 0):
        den = float(out['gt_visible_touched'])
        out['candidate_safe16_rate_on_touched_gtvis'] = out.get('candidate_safe16', 0) / den
        out['candidate_safe8_rate_on_touched_gtvis'] = out.get('candidate_safe8', 0) / den
        out['candidate_safe4_rate_on_touched_gtvis'] = out.get('candidate_safe4', 0) / den
        out['repair16_rate'] = out.get('repair16_count', 0) / den
        out['damage16_rate'] = out.get('damage16_count', 0) / den
        out['improve_px_rate'] = out.get('improve_px_count', 0) / den
        out['worse_px_rate'] = out.get('worse_px_count', 0) / den
        out['mean_native_err_px_on_touched_gtvis'] = out.get('sum_native_err_px', 0.0) / den
        out['mean_candidate_err_px_on_touched_gtvis'] = out.get('sum_candidate_err_px', 0.0) / den
    if out.get('touched_frames', 0):
        out['candidate_visible_rate_on_touched'] = out.get('candidate_visible_touched', 0) / float(out['touched_frames'])
        out['false_visible_rate_on_touched'] = out.get('false_visible_touched', 0) / float(out['touched_frames'])
    if selected_events:
        out['touch_rate_per_selected_event'] = out.get('event_count', 0) / float(selected_events)
    return recs, out


def policy_specs(events: List[Event], profile: str = "core") -> List[Tuple[str, Callable[[Event], bool], List[str]]]:
    specs: List[Tuple[str, Callable[[Event], bool], List[str]]] = []
    specs.append(('all_events', lambda e: True, ['candidate_visible', 'force_all', 'copy_candidate_visibility']))
    specs.append(('candidate_visible_t', lambda e: e.cand_visible_t, ['candidate_visible', 'force_all']))
    specs.append(('candidate_visrun_ge2', lambda e: e.cand_vis_run >= 2, ['candidate_visible', 'force_all']))
    score_grid = [0.35, 0.50] if profile == 'core' else [0.20, 0.35, 0.50, 0.60]
    run_grid = [4, 8] if profile == 'core' else [2, 4, 6, 8]
    dist_grid = [(16, 128), (16, 192)] if profile == 'core' else [(8, 64), (16, 128), (16, 192), (32, 256), (64, 320)]
    for thr in score_grid:
        specs.append((f'native_score_lt_{thr:.2f}_candvis', lambda e, thr=thr: e.score_t < thr and e.cand_visible_t, ['candidate_visible', 'force_all']))
    for l in run_grid:
        specs.append((f'lowrun_ge{l}_candvis', lambda e, l=l: e.low_run >= l and e.cand_visible_t, ['candidate_visible', 'force_all']))
        specs.append((f'invisrun_ge{l}_candvis', lambda e, l=l: e.invis_run >= l and e.cand_visible_t, ['candidate_visible', 'force_all']))
    for lo, hi in dist_grid:
        specs.append((f'dist_band_{lo}_{hi}_candvis', lambda e, lo=lo, hi=hi: e.cand_visible_t and lo <= e.dist_nc_px <= hi, ['candidate_visible', 'force_all']))
        specs.append((f'low_score_dist_band_{lo}_{hi}_candvis', lambda e, lo=lo, hi=hi: e.score_t < 0.50 and e.cand_visible_t and lo <= e.dist_nc_px <= hi, ['candidate_visible', 'force_all']))
    # Low-confidence mining: allow candidate even when TrackOn2 visibility is false, but require strong native failure and sane geometry.
    lowconf_grid = [0.35, 0.50] if profile == 'core' else [0.20, 0.35, 0.50]
    for thr in lowconf_grid:
        specs.append((
            f'lowconf_mine_score_lt_{thr:.2f}_band16_192',
            lambda e, thr=thr: (e.score_t < thr and 16.0 <= e.dist_nc_px <= 192.0 and e.cand_speed_px <= 96.0 and e.cand_step_from_last_per_gap <= 64.0),
            ['force_all', 'entry_force_then_candidate_visible', 'copy_candidate_visibility'],
        ))
    # Anchor/motion consistency variants; do not require candidate visibility in the lowconf version.
    specs.append((
        'anchor_consistency_low_score',
        lambda e: (e.score_t < 0.50 and e.anchor_cos > -0.25 and e.cand_step_from_last_per_gap <= 64.0 and e.dist_c_last_px <= 256.0),
        ['force_all', 'entry_force_then_candidate_visible', 'copy_candidate_visibility'],
    ))
    specs.append((
        'anchor_consistency_candvis',
        lambda e: (e.cand_visible_t and e.anchor_cos > -0.25 and e.cand_step_from_last_per_gap <= 64.0 and e.dist_c_last_px <= 256.0),
        ['candidate_visible', 'force_all'],
    ))
    # Conservative fusion is intentionally not in V8-C0 first implementation; it changes coords rather than selecting candidate.
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', default=str(NATIVE))
    ap.add_argument('--features', default=str(FEATURES))
    ap.add_argument('--candidate', default=str(CANDIDATE))
    ap.add_argument('--outdir', default=str(OUTDIR))
    ap.add_argument('--windows', default='0,4,8,16')
    ap.add_argument('--max-variants', type=int, default=0, help='debug: stop after this many variants; 0=all')
    ap.add_argument('--policy-profile', choices=['core','extended'], default='core')
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
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    # Explicit leakage audit for report only.
    feature_names = [str(x) for x in z['feature_names'].tolist()]
    leaky_feature_names = [n for n in feature_names if n.startswith('post_') or 'dt1' in n or 'dt2' in n or 'dt3' in n or 'dt4' in n]
    events = build_events(native, cand_by, metas)

    native_records = clone_records(native['records'])
    native_metric, native_per_video = standard_and_ajrd(native_records)
    cand_metric, cand_per_video = standard_and_ajrd(clone_records(cand['records']))

    rows: List[dict] = []
    rows.append({
        'variant': 'native',
        'policy': 'native',
        'window': None,
        'apply_mode': None,
        'metric': native_metric,
        'delta_vs_native': {k: 0.0 for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']},
        'stats': {},
    })
    rows.append({
        'variant': 'trackon2_standalone',
        'policy': 'trackon2_standalone',
        'window': None,
        'apply_mode': None,
        'metric': cand_metric,
        'delta_vs_native': {k: (cand_metric[k] - native_metric[k]) if cand_metric[k] is not None and native_metric[k] is not None else None for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']},
        'stats': {},
    })

    specs = policy_specs(events, profile=args.policy_profile)
    variant_count = 0
    for pname, fn, modes in specs:
        select = np.asarray([bool(fn(e)) for e in events], dtype=bool)
        base_sel_stats = {
            'selected_events': int(select.sum()),
            'candidate_visible_selected': int(sum(e.cand_visible_t for e, s in zip(events, select) if s)),
            'mean_score_selected': float(np.mean([e.score_t for e, s in zip(events, select) if s])) if select.any() else None,
            'mean_dist_nc_selected': float(np.mean([e.dist_nc_px for e, s in zip(events, select) if s])) if select.any() else None,
        }
        if base_sel_stats['selected_events']:
            base_sel_stats['candidate_visible_selected_rate'] = base_sel_stats['candidate_visible_selected'] / float(base_sel_stats['selected_events'])
        for w in windows:
            for mode in modes:
                variant_count += 1
                if args.max_variants and variant_count > args.max_variants:
                    break
                recs, stats = apply_events(native['records'], cand_by, events, select, window=w, apply_mode=mode)
                metric, per_video = standard_and_ajrd(recs)
                delta = {k: (metric[k] - native_metric[k]) if metric[k] is not None and native_metric[k] is not None else None for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']}
                rows.append({
                    'variant': f'{pname}_w{w}_{mode}',
                    'policy': pname,
                    'window': int(w),
                    'apply_mode': mode,
                    'metric': metric,
                    'delta_vs_native': delta,
                    'selection_stats': base_sel_stats,
                    'stats': stats,
                    'per_video': per_video,
                })
            if args.max_variants and variant_count > args.max_variants:
                break
        if args.max_variants and variant_count > args.max_variants:
            break

    def delta_key(row: dict, key: str) -> float:
        v = row.get('delta_vs_native', {}).get(key)
        return -999.0 if v is None else float(v)

    strict_rows = [r for r in rows if r['variant'] not in {'native', 'trackon2_standalone'}]
    best_by_ajrd256 = sorted(strict_rows, key=lambda r: delta_key(r, 'AJ_RD_256'), reverse=True)[:30]
    best_safe = sorted([
        r for r in strict_rows
        if delta_key(r, 'AJ') >= -0.10 and delta_key(r, 'OA') >= -0.10
    ], key=lambda r: delta_key(r, 'AJ_RD_256'), reverse=True)[:30]
    best_low_damage = sorted([
        r for r in strict_rows
        if r.get('stats', {}).get('false_visible_rate_on_touched', 1.0) <= 0.25
           and r.get('stats', {}).get('damage16_rate', 1.0) <= 0.25
    ], key=lambda r: delta_key(r, 'AJ_RD_256'), reverse=True)[:30]

    event_summary = {
        'n_events': int(len(events)),
        'native_visible_t_rate': float(np.mean([e.native_visible_t for e in events])) if events else 0.0,
        'candidate_visible_t_rate': float(np.mean([e.cand_visible_t for e in events])) if events else 0.0,
        'mean_score_t': float(np.mean([e.score_t for e in events])) if events else None,
        'mean_low_run': float(np.mean([e.low_run for e in events])) if events else None,
        'mean_invis_run': float(np.mean([e.invis_run for e in events])) if events else None,
        'mean_dist_nc_px': float(np.mean([e.dist_nc_px for e in events])) if events else None,
    }

    report = {
        'script': 'scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py',
        'causal_policy_note': 'Policies use only native/candidate cache fields at <= event_t and causal meta fields. X/X_aug and V7 OOF scores are intentionally not used.',
        'native': str(args.native),
        'features_meta_source': str(args.features),
        'candidate': str(args.candidate),
        'alignment': align_info,
        'leakage_audit': {
            'feature_package_has_leaky_future_features': bool(leaky_feature_names),
            'n_leaky_feature_names_detected': int(len(leaky_feature_names)),
            'examples': leaky_feature_names[:20],
            'policy_uses_X_or_X_aug': False,
            'policy_uses_v7_oof_scores': False,
            'policy_uses_oracle_labels': False,
        },
        'windows': windows,
        'policy_profile': args.policy_profile,
        'event_summary': event_summary,
        'native_metric': native_metric,
        'trackon2_standalone_metric': cand_metric,
        'trackon2_standalone_delta_vs_native': rows[1]['delta_vs_native'],
        'rows': rows,
        'best_by_AJ_RD_256': best_by_ajrd256,
        'best_safe_AJ_OA': best_safe,
        'best_low_damage': best_low_damage,
    }
    out = outdir / 'v8c0_causal_baseline_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # Compact stdout.
    def compact(row: dict) -> dict:
        return {
            'variant': row['variant'],
            'delta': {k: (round(v, 4) if v is not None else None) for k, v in row.get('delta_vs_native', {}).items()},
            'selected_events': row.get('selection_stats', {}).get('selected_events'),
            'touched_frames': row.get('stats', {}).get('touched_frames'),
            'false_visible_rate': row.get('stats', {}).get('false_visible_rate_on_touched'),
            'damage16_rate': row.get('stats', {}).get('damage16_rate'),
        }
    print(json.dumps({
        'out': str(out),
        'native_metric': {k: round(v, 4) if isinstance(v, float) else v for k, v in native_metric.items() if k in ['AJ','OA','AJ_RD','AJ_RD_256']},
        'trackon2_delta': {k: round(v, 4) if v is not None else None for k, v in rows[1]['delta_vs_native'].items()},
        'event_summary': event_summary,
        'best_by_AJ_RD_256': [compact(r) for r in best_by_ajrd256[:12]],
        'best_safe_AJ_OA': [compact(r) for r in best_safe[:12]],
        'best_low_damage': [compact(r) for r in best_low_damage[:12]],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
