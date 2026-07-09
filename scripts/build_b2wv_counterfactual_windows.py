#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, yx_norm_to_xy_256
from utils.reentry_metrics import compute_reappearance_segment_aj, eligible_reentry_events

OUT = Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev')
DATASETS = {
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
    },
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
    },
}
ACTIONS = {'reject': None, 'W4': 4, 'W8': 8, 'W16': 16}
LAMBDAS = [0.25, 0.5, 1.0, 2.0]
THRESHOLDS = (1, 2, 4, 8, 16)


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


def px(track_yx: np.ndarray) -> np.ndarray:
    return yx_norm_to_xy_256(np.asarray(track_yx, dtype=np.float32))


def px_dist(a_yx: np.ndarray, b_yx: np.ndarray) -> float:
    a = px(np.asarray(a_yx, dtype=np.float32)[None, :])[0]
    b = px(np.asarray(b_yx, dtype=np.float32)[None, :])[0]
    return float(np.linalg.norm(a - b))


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def persist_len(v: np.ndarray, t: int, cap: int = 16) -> int:
    c = 0
    for j in range(int(t), min(len(v), int(t) + int(cap))):
        if bool(v[j]):
            c += 1
        else:
            break
    return int(c)


def frac_true(v: np.ndarray, t: int, span: int) -> Optional[float]:
    hi = min(len(v), int(t) + int(span))
    if hi <= int(t):
        return None
    return float(np.mean(v[int(t):hi].astype(np.float32)))


def transitions(v: np.ndarray, t: int, span: int) -> int:
    hi = min(len(v), int(t) + int(span))
    if hi - int(t) < 2:
        return 0
    z = v[int(t):hi].astype(bool)
    return int(np.sum(z[1:] != z[:-1]))


def speed_mean(track_yx: np.ndarray, start: int, end: int) -> Optional[float]:
    lo = max(0, int(start))
    hi = min(len(track_yx), int(end))
    if hi - lo < 2:
        return None
    p = px(track_yx[lo:hi])
    d = np.linalg.norm(p[1:] - p[:-1], axis=-1)
    return float(np.mean(d)) if d.size else None


def dist_stats(base_tr: np.ndarray, over_tr: np.ndarray, t: int, span: int) -> Dict[str, Optional[float]]:
    vals = []
    for j in range(int(t), min(len(base_tr), int(t) + int(span))):
        vals.append(px_dist(base_tr[j], over_tr[j]))
    if not vals:
        return {'mean': None, 'max': None}
    return {'mean': float(np.mean(vals)), 'max': float(np.max(vals))}


def point_aj_mask(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return 0.0
    qt = max(0, min(mask.size - 1, int(query_t)))
    mask[qt] = False
    if not np.any(mask):
        return 0.0
    pred_px = px(pred_tracks)
    gt_px = px(gt_tracks)
    sq = np.sum((pred_px - gt_px) ** 2, axis=-1)
    vals = []
    gt_pos = gt_vis.astype(bool)
    pv = pred_vis.astype(bool)
    for thr in THRESHOLDS:
        within = sq < float(thr) ** 2
        tp = float(np.sum(mask & within & gt_pos & pv))
        gp = float(np.sum(mask & gt_pos))
        fp = float(np.sum(mask & pv & ((~gt_pos) | (~within))))
        vals.append(tp / (gp + fp) if (gp + fp) > 0 else 0.0)
    return float(np.mean(vals))


def point_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, start_t: int = 0) -> float:
    mask = np.ones(gt_vis.shape[0], dtype=bool)
    if start_t > 0:
        mask[:int(start_t)] = False
    return point_aj_mask(pred_tracks, gt_tracks, pred_vis, gt_vis, query_t, mask)


def apply_action(base_tr: np.ndarray, base_v: np.ndarray, over_tr: np.ndarray, over_v: np.ndarray, t: int, action: str, pre: int = 1) -> Tuple[np.ndarray, np.ndarray, int, int]:
    pred_tr = base_tr.copy()
    pred_v = base_v.copy()
    W = ACTIONS[action]
    if W is None:
        return pred_tr, pred_v, int(t), int(t)
    lo = max(0, int(t) - int(pre))
    hi = min(len(base_v), int(t) + int(W) + 1)
    pred_tr[lo:hi] = over_tr[lo:hi]
    pred_v[lo:hi] = over_v[lo:hi]
    return pred_tr, pred_v, lo, hi


def reentry_segment_mean(pred_tr: np.ndarray, gt_tr: np.ndarray, pred_v: np.ndarray, gt_v: np.ndarray, events: List[Dict[str, int]], h: int, w: int, lo: int, hi: int) -> Optional[float]:
    vals = []
    for ev in events:
        rt = int(ev['reentry_frame'])
        # Window-specific: include events whose re-entry frame falls inside or just after the local decision window.
        if rt < int(lo) or rt >= int(hi):
            continue
        ev2 = dict(ev)
        out = compute_reappearance_segment_aj(
            pred_tracks=pred_tr,
            gt_tracks=gt_tr,
            pred_visibility=pred_v,
            gt_visibility=gt_v,
            event=ev2,
            height=h,
            width=w,
            use_256_space=True,
        )
        if out is not None:
            vals.append(float(out['aj_segment']))
    if not vals:
        return None
    return float(np.mean(vals))


def candidate_times(base_v: np.ndarray, over_v: np.ndarray, query_t: int, skip: int = 16, max_windows: int = 8) -> List[int]:
    out = []
    t = max(1, int(query_t) + 1)
    T = len(base_v)
    while t < T:
        if invisible_run_before(base_v, t) >= 1 and bool(over_v[t]):
            out.append(int(t))
            if len(out) >= int(max_windows):
                break
            t = min(T, t + int(skip) + 1)
        else:
            t += 1
    return out


def features_for(base_tr: np.ndarray, over_tr: np.ndarray, base_v: np.ndarray, over_v: np.ndarray, q_t: int, t: int, cand_idx: int) -> Dict[str, Any]:
    T = len(base_v)
    d4 = dist_stats(base_tr, over_tr, t, 4)
    d8 = dist_stats(base_tr, over_tr, t, 8)
    d16 = dist_stats(base_tr, over_tr, t, 16)
    hi16 = min(T, t + 16)
    return {
        'trigger_t_norm': float(t / max(T - 1, 1)),
        'query_age_norm': float((t - q_t) / max(T - 1, 1)),
        'candidate_window_index': int(cand_idx),
        'base_invis_run': float(invisible_run_before(base_v, t)),
        'override_persist_len_cap16': float(persist_len(over_v, t, 16)),
        'override_visible_frac_next4': frac_true(over_v, t, 4),
        'override_visible_frac_next8': frac_true(over_v, t, 8),
        'override_visible_frac_next16': frac_true(over_v, t, 16),
        'base_visible_frac_next4': frac_true(base_v, t, 4),
        'base_visible_frac_next8': frac_true(base_v, t, 8),
        'base_visible_frac_next16': frac_true(base_v, t, 16),
        'vis_agreement_frac_next16': float(np.mean((base_v[t:hi16] == over_v[t:hi16]).astype(np.float32))) if hi16 > t else None,
        'override_visibility_transitions_next16': float(transitions(over_v, t, 16)),
        'base_visibility_transitions_next16': float(transitions(base_v, t, 16)),
        'base_override_dist_t': float(px_dist(base_tr[t], over_tr[t])),
        'base_override_dist_mean_next4': d4['mean'],
        'base_override_dist_mean_next8': d8['mean'],
        'base_override_dist_mean_next16': d16['mean'],
        'base_override_dist_max_next16': d16['max'],
        'base_speed_prev4': speed_mean(base_tr, max(q_t, t - 4), t + 1),
        'base_speed_prev8': speed_mean(base_tr, max(q_t, t - 8), t + 1),
        'override_speed_next4': speed_mean(over_tr, t, t + 4),
        'override_speed_next8': speed_mean(over_tr, t, t + 8),
        'override_speed_next16': speed_mean(over_tr, t, t + 16),
        'base_becomes_visible_next4': bool(np.any(base_v[t:min(T, t + 4)])),
        'override_loses_visible_next4': bool(np.any(~over_v[t:min(T, t + 4)])),
    }


def build_rows_for_dataset(name: str, base_path: Path, over_path: Path, max_candidates_per_track: int = 8) -> List[Dict[str, Any]]:
    base = load(base_path)
    over = load(over_path)
    rows: List[Dict[str, Any]] = []
    for rec_idx, (b, o) in enumerate(zip(base['records'], over['records'])):
        vid = str(b.get('video_id', rec_idx))
        h, w = int(b['original_size'][0]), int(b['original_size'][1])
        base_tr = npy(b['pred_tracks'], np.float32)
        over_tr = npy(o['pred_tracks'], np.float32)
        gt_tr = npy(b['gt_tracks'], np.float32)
        base_v = npy(b['pred_visibility'], bool)
        over_v = npy(o['pred_visibility'], bool)
        gt_v = npy(b['gt_visibility'], bool)
        qpts = npy(b['query_points'], np.float32)
        n, T = base_v.shape
        for qi in range(n):
            qt = max(0, min(T - 1, int(round(float(qpts[qi, 0])))))
            all_events = find_reentry_events(gt_v[qi], qt)
            elig_events = eligible_reentry_events(gt_v[qi], qt)
            cands = candidate_times(base_v[qi], over_v[qi], qt, skip=16, max_windows=max_candidates_per_track)
            if not cands:
                continue
            # Baseline metrics once per track.
            base_full = point_aj(base_tr[qi], gt_tr[qi], base_v[qi], gt_v[qi], qt, start_t=0)
            for ci, t in enumerate(cands):
                feats = features_for(base_tr[qi], over_tr[qi], base_v[qi], over_v[qi], qt, t, ci)
                lo_ref = max(0, int(t) - 1)
                hi_ref = min(T, int(t) + 16 + 1)
                window_has_reentry = any(lo_ref <= int(ev['reentry_frame']) < hi_ref for ev in all_events)
                eligible_window_has_reentry = any(lo_ref <= int(ev['reentry_frame']) < hi_ref for ev in elig_events)
                action_metrics: Dict[str, Any] = {}
                base_window_mask = np.zeros(T, dtype=bool)
                base_window_mask[lo_ref:hi_ref] = True
                base_post = point_aj(base_tr[qi], gt_tr[qi], base_v[qi], gt_v[qi], qt, start_t=lo_ref)
                base_window = point_aj_mask(base_tr[qi], gt_tr[qi], base_v[qi], gt_v[qi], qt, base_window_mask)
                base_reentry = reentry_segment_mean(base_tr[qi], gt_tr[qi], base_v[qi], gt_v[qi], elig_events, h, w, lo_ref, hi_ref)
                for action in ACTIONS:
                    pred_tr, pred_v, lo, hi = apply_action(base_tr[qi], base_v[qi], over_tr[qi], over_v[qi], t, action, pre=1)
                    mask = np.zeros(T, dtype=bool)
                    if action == 'reject':
                        mask[lo_ref:hi_ref] = True
                    else:
                        mask[lo:hi] = True
                    full = point_aj(pred_tr, gt_tr[qi], pred_v, gt_v[qi], qt, start_t=0)
                    post = point_aj(pred_tr, gt_tr[qi], pred_v, gt_v[qi], qt, start_t=lo_ref)
                    win = point_aj_mask(pred_tr, gt_tr[qi], pred_v, gt_v[qi], qt, mask)
                    reseg = reentry_segment_mean(pred_tr, gt_tr[qi], pred_v, gt_v[qi], elig_events, h, w, lo_ref, hi_ref)
                    delta_full = float(full - base_full)
                    delta_post = float(post - base_post)
                    delta_window = float(win - base_window)
                    # If no eligible re-entry event in this window, fall back to local window AJ as target gain.
                    if reseg is None or base_reentry is None:
                        target_gain = delta_window
                        delta_reentry = None
                    else:
                        delta_reentry = float(reseg - base_reentry)
                        target_gain = delta_reentry
                    utils = {}
                    for lam in LAMBDAS:
                        utils[f'lambda_{lam:g}'] = float(target_gain - float(lam) * max(0.0, -delta_full))
                    action_metrics[action] = {
                        'lo': int(lo),
                        'hi': int(hi),
                        'full_AJ': round(full, 6),
                        'post_AJ': round(post, 6),
                        'window_AJ': round(win, 6),
                        'reentry_segment_AJ': round(reseg, 6) if reseg is not None else None,
                        'delta_full_AJ': round(delta_full, 6),
                        'delta_post_AJ': round(delta_post, 6),
                        'delta_window_AJ': round(delta_window, 6),
                        'delta_reentry_segment_AJ': round(delta_reentry, 6) if delta_reentry is not None else None,
                        'target_gain': round(target_gain, 6),
                        'utility': {k: round(v, 6) for k, v in utils.items()},
                    }
                best = {}
                for lam in LAMBDAS:
                    k = f'lambda_{lam:g}'
                    best[k] = max(ACTIONS.keys(), key=lambda a: action_metrics[a]['utility'][k])
                row = {
                    'dataset': name,
                    'video_id': vid,
                    'record_index': int(rec_idx),
                    'query_idx': int(qi),
                    'query_t': int(qt),
                    'trigger_t': int(t),
                    'candidate_index': int(ci),
                    'track_has_reentry': bool(all_events),
                    'track_has_eligible_reentry': bool(elig_events),
                    'window_has_reentry': bool(window_has_reentry),
                    'eligible_window_has_reentry': bool(eligible_window_has_reentry),
                    'n_track_reentry_events': int(len(all_events)),
                    'n_eligible_reentry_events': int(len(elig_events)),
                    'first_reentry_t': int(all_events[0]['reentry_frame']) if all_events else None,
                    'first_eligible_reentry_t': int(elig_events[0]['reentry_frame']) if elig_events else None,
                    'features': feats,
                    'base_metrics': {
                        'full_AJ': round(base_full, 6),
                        'post_AJ_from_trigger': round(base_post, 6),
                        'window_AJ_W16_region': round(base_window, 6),
                        'reentry_segment_AJ_in_window': round(base_reentry, 6) if base_reentry is not None else None,
                    },
                    'actions': action_metrics,
                    'best_action': best,
                }
                rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'n_rows': int(len(rows)),
        'datasets': {},
        'best_action_distribution': {},
        'best_action_by_window_has_reentry': {},
        'fixed_action_means': {},
    }
    for ds in sorted(set(r['dataset'] for r in rows)):
        dr = [r for r in rows if r['dataset'] == ds]
        out['datasets'][ds] = {
            'n_rows': len(dr),
            'n_videos': len(set(r['video_id'] for r in dr)),
            'track_has_reentry_rate': round(float(np.mean([r['track_has_reentry'] for r in dr])), 6) if dr else None,
            'window_has_reentry_rate': round(float(np.mean([r['window_has_reentry'] for r in dr])), 6) if dr else None,
        }
    for lam in LAMBDAS:
        k = f'lambda_{lam:g}'
        vals = [r['best_action'][k] for r in rows]
        out['best_action_distribution'][k] = {a: int(vals.count(a)) for a in ACTIONS}
        out['best_action_distribution'][k]['total'] = len(vals)
        out['best_action_distribution'][k]['rates'] = {a: round(vals.count(a) / max(len(vals), 1), 6) for a in ACTIONS}
        out['best_action_by_window_has_reentry'][k] = {}
        for flag in [False, True]:
            sub = [r for r in rows if bool(r['window_has_reentry']) == flag]
            sv = [r['best_action'][k] for r in sub]
            out['best_action_by_window_has_reentry'][k][str(flag)] = {
                'n': len(sub),
                'counts': {a: int(sv.count(a)) for a in ACTIONS},
                'rates': {a: round(sv.count(a) / max(len(sv), 1), 6) for a in ACTIONS},
            }
    # Mean utility and deltas for fixed actions / oracle by lambda.
    for action in ACTIONS:
        out['fixed_action_means'][action] = {}
        for metric in ['delta_full_AJ', 'delta_post_AJ', 'delta_window_AJ', 'target_gain']:
            vals = [float(r['actions'][action][metric]) for r in rows]
            out['fixed_action_means'][action][metric] = round(float(np.mean(vals)), 6) if vals else None
    out['oracle_action_means'] = {}
    for lam in LAMBDAS:
        k = f'lambda_{lam:g}'
        chosen = [(r, r['best_action'][k]) for r in rows]
        out['oracle_action_means'][k] = {}
        for metric in ['delta_full_AJ', 'delta_post_AJ', 'delta_window_AJ', 'target_gain']:
            vals = [float(r['actions'][a][metric]) for r, a in chosen]
            out['oracle_action_means'][k][metric] = round(float(np.mean(vals)), 6) if vals else None
        vals = [float(r['actions'][a]['utility'][k]) for r, a in chosen]
        out['oracle_action_means'][k]['utility'] = round(float(np.mean(vals)), 6) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--max-candidates-per-track', type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for name, cfg in DATASETS.items():
        print(f'building {name}...', flush=True)
        rows = build_rows_for_dataset(name, cfg['base'], cfg['override'], max_candidates_per_track=args.max_candidates_per_track)
        all_rows.extend(rows)
        with (out_dir / f'{name}_window_rows.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'{name}: {len(rows)} windows', flush=True)
    with (out_dir / 'window_rows.jsonl').open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {
        'protocol': 'B2-WV counterfactual window dataset. Development only: DAVIS + RGB dev0-9. Candidate generator: base invisible run >=1 and override visible; non-overlap skip=16 after each candidate. Actions: reject, W4, W8, W16. Utilities use target_gain - lambda*standard_harm.',
        'datasets': {k: {kk: str(vv) for kk, vv in v.items()} for k, v in DATASETS.items()},
        'actions': ACTIONS,
        'lambdas': LAMBDAS,
        'summary': summarize(all_rows),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary['summary'], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
