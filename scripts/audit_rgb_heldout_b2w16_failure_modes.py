#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, yx_norm_to_xy_256

ROOT = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_heldout')
OUT = ROOT / 'b2_w16_heldout10_failure_modes'
CACHES = {
    'offline': ROOT / 'cotracker3_offline_rgb_stacking_heldout10.pt',
    'online': ROOT / 'cotracker3_online_rgb_stacking_heldout10.pt',
    'b2_w16': ROOT / 'b2_w16_heldout10' / 'b2_w16_rgb_heldout10.pt',
}
PER_VIDEO_AUDIT = ROOT / 'b2_w16_heldout10_per_video_audit' / 'summary.json'
THRESHOLDS = (1, 2, 4, 8, 16)
FAIL_VIDEOS = {'rgb_stacking_000010', 'rgb_stacking_000017', 'rgb_stacking_000015'}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def triggers_for_track(base_v: np.ndarray, over_v: np.ndarray, query_t: int, post: int = 16, pre: int = 1, k: int = 1) -> List[int]:
    T = len(base_v)
    ts: List[int] = []
    t = max(1, int(query_t) + 1)
    while t < T:
        if invisible_run_before(base_v, t) >= k and bool(over_v[t]):
            ts.append(int(t))
            hi = min(T, int(t) + int(post) + 1)
            t = hi
        else:
            t += 1
    return ts


def point_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, start_t: int = 0) -> float:
    T = gt_vis.shape[0]
    mask = np.ones(T, dtype=bool)
    qt = max(0, min(T - 1, int(query_t)))
    mask[qt] = False
    if start_t > 0:
        mask[: int(start_t)] = False
    if not np.any(mask):
        return 0.0
    pred_px = yx_norm_to_xy_256(np.asarray(pred_tracks, dtype=np.float32))
    gt_px = yx_norm_to_xy_256(np.asarray(gt_tracks, dtype=np.float32))
    sq = np.sum((pred_px - gt_px) ** 2, axis=-1)
    vals = []
    for thr in THRESHOLDS:
        within = sq < float(thr) ** 2
        gv = gt_vis.astype(bool)
        pv = pred_vis.astype(bool)
        tp = float(np.sum(mask & within & gv & pv))
        gp = float(np.sum(mask & gv))
        fp = float(np.sum(mask & pv & ((~gv) | (~within))))
        vals.append(tp / (gp + fp) if (gp + fp) > 0 else 0.0)
    return float(np.mean(vals))


def reentry_segment_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, reentry_t: int) -> float:
    return point_aj(pred_tracks, gt_tracks, pred_vis, gt_vis, query_t=reentry_t, start_t=reentry_t)


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return round(float(np.mean(xs)), 6)


def summarize_rows(rows: List[Dict[str, Any]], prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {
        prefix + 'n': len(rows),
        prefix + 'mean_delta_b2_offline_full': mean([r['delta_b2_offline_full'] for r in rows]),
        prefix + 'median_delta_b2_offline_full': round(float(np.median([r['delta_b2_offline_full'] for r in rows])), 6) if rows else None,
        prefix + 'severe_full_delta_lt_-0p10': int(sum(1 for r in rows if r['delta_b2_offline_full'] < -0.10)),
        prefix + 'harmful_full_delta_lt_-0p05': int(sum(1 for r in rows if r['delta_b2_offline_full'] < -0.05)),
        prefix + 'helpful_full_delta_gt_0p05': int(sum(1 for r in rows if r['delta_b2_offline_full'] > 0.05)),
    }
    re_rows = [r for r in rows if r.get('has_gt_reentry')]
    if re_rows:
        out[prefix + 'mean_delta_b2_offline_reentry'] = mean([r.get('delta_b2_offline_reentry', 0.0) for r in re_rows])
        out[prefix + 'median_delta_b2_offline_reentry'] = round(float(np.median([r.get('delta_b2_offline_reentry', 0.0) for r in re_rows])), 6)
    else:
        out[prefix + 'mean_delta_b2_offline_reentry'] = None
        out[prefix + 'median_delta_b2_offline_reentry'] = None
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in CACHES.items()}
    per_video = {r['video_id']: r for r in json.load(open(PER_VIDEO_AUDIT))['rows']}
    all_rows: List[Dict[str, Any]] = []
    video_summaries: List[Dict[str, Any]] = []
    for i, b_rec in enumerate(payloads['offline']['records']):
        vid = str(b_rec['video_id'])
        o_rec = payloads['online']['records'][i]
        w_rec = payloads['b2_w16']['records'][i]
        base_v = npy(b_rec['pred_visibility'], bool)
        over_v = npy(o_rec['pred_visibility'], bool)
        b2_v = npy(w_rec['pred_visibility'], bool)
        gt_v = npy(b_rec['gt_visibility'], bool)
        base_p = npy(b_rec['pred_tracks'], np.float32)
        over_p = npy(o_rec['pred_tracks'], np.float32)
        b2_p = npy(w_rec['pred_tracks'], np.float32)
        gt_p = npy(b_rec['gt_tracks'], np.float32)
        qpts = npy(b_rec['query_points'], np.float32)
        rows: List[Dict[str, Any]] = []
        for qi in range(gt_v.shape[0]):
            qt = int(round(float(qpts[qi, 0])))
            events = find_reentry_events(gt_v[qi], qt)
            has_re = bool(events)
            trig_ts = triggers_for_track(base_v[qi], over_v[qi], qt, post=16)
            cls = 'true_trigger' if (trig_ts and has_re) else ('false_trigger' if trig_ts else ('missed_reentry' if has_re else 'no_trigger'))
            off_aj = point_aj(base_p[qi], gt_p[qi], base_v[qi], gt_v[qi], qt)
            on_aj = point_aj(over_p[qi], gt_p[qi], over_v[qi], gt_v[qi], qt)
            b2_aj = point_aj(b2_p[qi], gt_p[qi], b2_v[qi], gt_v[qi], qt)
            row: Dict[str, Any] = {
                'video_id': vid,
                'query_idx': qi,
                'query_t': qt,
                'class': cls,
                'trigger_count': len(trig_ts),
                'first_trigger_t': trig_ts[0] if trig_ts else None,
                'has_gt_reentry': has_re,
                'first_reentry_t': int(events[0]['reentry_frame']) if events else None,
                'first_occ_length': int(events[0]['occ_length']) if events else None,
                'offline_AJ': round(off_aj, 6),
                'online_AJ': round(on_aj, 6),
                'b2_AJ': round(b2_aj, 6),
                'delta_online_offline_full': round(on_aj - off_aj, 6),
                'delta_b2_offline_full': round(b2_aj - off_aj, 6),
            }
            if has_re:
                rt = int(events[0]['reentry_frame'])
                off_re = reentry_segment_aj(base_p[qi], gt_p[qi], base_v[qi], gt_v[qi], rt)
                on_re = reentry_segment_aj(over_p[qi], gt_p[qi], over_v[qi], gt_v[qi], rt)
                b2_re = reentry_segment_aj(b2_p[qi], gt_p[qi], b2_v[qi], gt_v[qi], rt)
                row.update({
                    'offline_reentry_AJ': round(off_re, 6),
                    'online_reentry_AJ': round(on_re, 6),
                    'b2_reentry_AJ': round(b2_re, 6),
                    'delta_online_offline_reentry': round(on_re - off_re, 6),
                    'delta_b2_offline_reentry': round(b2_re - off_re, 6),
                })
            rows.append(row)
            all_rows.append(row)
        by_class = {c: [r for r in rows if r['class'] == c] for c in ['true_trigger', 'false_trigger', 'missed_reentry', 'no_trigger']}
        pv = per_video[vid]
        vs = {
            'video_id': vid,
            'is_standard_drop_gt2': vid in FAIL_VIDEOS,
            'per_video_metrics': pv,
            'n_tracks': len(rows),
            'n_gt_reentry_tracks': int(sum(1 for r in rows if r['has_gt_reentry'])),
            'n_triggered_tracks': int(sum(1 for r in rows if r['trigger_count'] > 0)),
            'n_trigger_events': int(sum(r['trigger_count'] for r in rows)),
            'n_true_trigger_tracks': len(by_class['true_trigger']),
            'n_false_trigger_tracks': len(by_class['false_trigger']),
            'n_missed_reentry_tracks': len(by_class['missed_reentry']),
            'n_no_trigger_tracks': len(by_class['no_trigger']),
            'trigger_precision_track': round(len(by_class['true_trigger']) / max(len(by_class['true_trigger']) + len(by_class['false_trigger']), 1), 6),
            'trigger_recall_track': round(len(by_class['true_trigger']) / max(len(by_class['true_trigger']) + len(by_class['missed_reentry']), 1), 6),
            'all': summarize_rows(rows),
            'true_trigger': summarize_rows(by_class['true_trigger']),
            'false_trigger': summarize_rows(by_class['false_trigger']),
            'missed_reentry': summarize_rows(by_class['missed_reentry']),
            'worst_full_delta_tracks': sorted(rows, key=lambda r: r['delta_b2_offline_full'])[:10],
            'worst_true_reentry_tracks': sorted(by_class['true_trigger'], key=lambda r: r.get('delta_b2_offline_reentry', 0.0))[:10],
            'best_true_reentry_tracks': sorted(by_class['true_trigger'], key=lambda r: r.get('delta_b2_offline_reentry', 0.0), reverse=True)[:10],
        }
        video_summaries.append(vs)
    failed = [v for v in video_summaries if v['is_standard_drop_gt2']]
    non_failed = [v for v in video_summaries if not v['is_standard_drop_gt2']]
    summary = {
        'protocol_note': 'Held-out analysis only. Do not tune B2-W16 on these videos.',
        'fail_videos_standard_drop_gt2': sorted(FAIL_VIDEOS),
        'n_videos': len(video_summaries),
        'failed_video_count': len(failed),
        'non_failed_video_count': len(non_failed),
        'failed_video_summaries': failed,
        'non_failed_video_summaries': non_failed,
        'all_video_summaries': video_summaries,
        'overall_by_class': {
            c: summarize_rows([r for r in all_rows if r['class'] == c])
            for c in ['true_trigger', 'false_trigger', 'missed_reentry', 'no_trigger']
        },
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'track_rows.jsonl').open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    compact = {
        'fail_videos_standard_drop_gt2': summary['fail_videos_standard_drop_gt2'],
        'overall_by_class': summary['overall_by_class'],
        'failed_compact': [
            {
                'video_id': v['video_id'],
                'b2_vs_offline_AJ_RD': v['per_video_metrics']['deltas']['b2_vs_offline_AJ_RD_256'],
                'b2_vs_offline_AJ': v['per_video_metrics']['deltas']['b2_vs_offline_AJ_256'],
                'trigger_precision': v['trigger_precision_track'],
                'trigger_recall': v['trigger_recall_track'],
                'n_true_trigger_tracks': v['n_true_trigger_tracks'],
                'n_false_trigger_tracks': v['n_false_trigger_tracks'],
                'false_trigger_mean_delta': v['false_trigger']['mean_delta_b2_offline_full'],
                'true_trigger_mean_delta': v['true_trigger']['mean_delta_b2_offline_full'],
                'true_trigger_mean_reentry_delta': v['true_trigger']['mean_delta_b2_offline_reentry'],
                'all_mean_delta': v['all']['mean_delta_b2_offline_full'],
            }
            for v in failed
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
