#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics
from utils.coords import find_reentry_events

ROOT = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke')
OUT = ROOT / 'rgb_b2_10video_per_video_audit'
CACHES = {
    'offline': ROOT / 'cotracker3_offline_rgb_stacking_10video.pt',
    'online': ROOT / 'cotracker3_online_rgb_stacking_10video.pt',
    'b2': ROOT / 'b2_offline_base_online_override_10video' / 'b2_rgb10_offline_base_online_override.pt',
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def standard_metrics(r: Dict[str, Any]) -> Dict[str, float]:
    pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
    gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
    pv = torch.from_numpy(npy(r['pred_visibility'], bool))
    gv = torch.from_numpy(npy(r['gt_visibility'], bool))
    q = torch.from_numpy(npy(r['query_points'], np.float32))
    m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
    return {
        'AJ_256': round(float(m.get('AJ', 0.0)) * 100.0, 4),
        'OA_256': round(float(m.get('OA', 0.0)) * 100.0, 4),
        'delta_avg_256': round(float(m.get('average_pts_within_thresh', 0.0)) * 100.0, 4),
    }


def ajrd_metrics(r: Dict[str, Any]) -> Dict[str, Any]:
    h, w = int(r['original_size'][0]), int(r['original_size'][1])
    m = compute_reentry_metrics(
        pred_tracks=npy(r['pred_tracks'], np.float32),
        gt_tracks=npy(r['gt_tracks'], np.float32),
        pred_vis=npy(r['pred_visibility'], bool),
        gt_vis=npy(r['gt_visibility'], bool),
        query_points=npy(r['query_points'], np.float32),
        height=h,
        width=w,
    )
    return {
        'AJ_RD_256': m.get('true_AJ_RD_256'),
        'AJ_RD': m.get('true_AJ_RD'),
        'proxy': m.get('first_reentry_frame_proxy'),
        'n_reentry_queries': int(m.get('n_reentry_queries', 0)),
        'dmin1_256': (m.get('aj_rd_by_dmin_256') or {}).get('1'),
        'dmin4_256': (m.get('aj_rd_by_dmin_256') or {}).get('4'),
        'dmin16_256': (m.get('aj_rd_by_dmin_256') or {}).get('16'),
    }


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def first_trigger(base_v: np.ndarray, over_v: np.ndarray, query_t: int, k: int = 1):
    for t in range(max(1, int(query_t) + 1), len(base_v)):
        if invisible_run_before(base_v, t) >= k and bool(over_v[t]):
            return int(t)
    return None


def trigger_stats(base_r: Dict[str, Any], over_r: Dict[str, Any]) -> Dict[str, Any]:
    base_v = npy(base_r['pred_visibility'], bool)
    over_v = npy(over_r['pred_visibility'], bool)
    gt_v = npy(base_r['gt_visibility'], bool)
    qpts = npy(base_r['query_points'], np.float32)
    total = int(gt_v.shape[0])
    gt_re = trig = true_t = false_t = miss_re = none = 0
    for qi in range(total):
        qt = int(round(float(qpts[qi, 0])))
        has_re = bool(find_reentry_events(gt_v[qi], qt))
        if has_re:
            gt_re += 1
        tr = first_trigger(base_v[qi], over_v[qi], qt)
        if tr is not None:
            trig += 1
            if has_re:
                true_t += 1
            else:
                false_t += 1
        else:
            if has_re:
                miss_re += 1
            else:
                none += 1
    return {
        'tracks': total,
        'gt_reentry_tracks': gt_re,
        'tracks_with_trigger': trig,
        'triggered_reentry_tracks': true_t,
        'triggered_nonreentry_tracks': false_t,
        'missed_reentry_tracks': miss_re,
        'trigger_precision': round(true_t / max(trig, 1), 6),
        'trigger_recall': round(true_t / max(gt_re, 1), 6),
        'false_trigger_rate': round(false_t / max(total, 1), 6),
    }


def delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in CACHES.items()}
    n = len(payloads['offline']['records'])
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        vid = str(payloads['offline']['records'][i]['video_id'])
        row: Dict[str, Any] = {'video_id': vid, 'idx': i}
        for name in ['offline', 'online', 'b2']:
            rec = payloads[name]['records'][i]
            mm = {}
            mm.update(standard_metrics(rec))
            mm.update(ajrd_metrics(rec))
            row[name] = mm
        row['trigger'] = trigger_stats(payloads['offline']['records'][i], payloads['online']['records'][i])
        row['deltas'] = {
            'b2_vs_offline_AJ_RD_256': delta(row['b2']['AJ_RD_256'], row['offline']['AJ_RD_256']),
            'b2_vs_offline_AJ_256': delta(row['b2']['AJ_256'], row['offline']['AJ_256']),
            'b2_vs_online_AJ_RD_256': delta(row['b2']['AJ_RD_256'], row['online']['AJ_RD_256']),
            'b2_vs_online_AJ_256': delta(row['b2']['AJ_256'], row['online']['AJ_256']),
        }
        rows.append(row)

    counts = {
        'n_videos': n,
        'b2_improves_AJRD_vs_offline': sum(1 for r in rows if (r['deltas']['b2_vs_offline_AJ_RD_256'] or -999) > 0),
        'b2_improves_AJRD_vs_offline_ge_0p01': sum(1 for r in rows if (r['deltas']['b2_vs_offline_AJ_RD_256'] or -999) >= 0.01),
        'b2_AJ_drop_vs_offline_le_1': sum(1 for r in rows if r['deltas']['b2_vs_offline_AJ_256'] is not None and r['deltas']['b2_vs_offline_AJ_256'] >= -1.0),
        'b2_AJ_drop_vs_offline_gt_2': sum(1 for r in rows if r['deltas']['b2_vs_offline_AJ_256'] is not None and r['deltas']['b2_vs_offline_AJ_256'] < -2.0),
        'b2_beats_online_AJRD': sum(1 for r in rows if (r['deltas']['b2_vs_online_AJ_RD_256'] or -999) > 0),
        'b2_beats_online_AJ_by_10': sum(1 for r in rows if r['deltas']['b2_vs_online_AJ_256'] is not None and r['deltas']['b2_vs_online_AJ_256'] >= 10.0),
    }

    def mean_for(name, key):
        vals = [r[name][key] for r in rows if r[name].get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    aggregate = {
        name: {
            'video_mean_AJ_RD_256': mean_for(name, 'AJ_RD_256'),
            'video_mean_AJ_256': mean_for(name, 'AJ_256'),
            'video_mean_OA_256': mean_for(name, 'OA_256'),
            'video_mean_delta_avg_256': mean_for(name, 'delta_avg_256'),
        }
        for name in ['offline', 'online', 'b2']
    }
    worst_aj = sorted(rows, key=lambda r: r['deltas']['b2_vs_offline_AJ_256'])[:5]
    best_ajrd = sorted(rows, key=lambda r: r['deltas']['b2_vs_offline_AJ_RD_256'], reverse=True)[:5]
    bad_ajrd = sorted(rows, key=lambda r: r['deltas']['b2_vs_offline_AJ_RD_256'])[:5]
    summary = {
        'caches': {k: str(v) for k, v in CACHES.items()},
        'aggregate_video_weighted': aggregate,
        'counts': counts,
        'best_b2_ajrd_gain_vs_offline': best_ajrd,
        'worst_b2_ajrd_delta_vs_offline': bad_ajrd,
        'worst_b2_aj_drop_vs_offline': worst_aj,
        'rows': rows,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'per_video_rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ['aggregate_video_weighted', 'counts', 'best_b2_ajrd_gain_vs_offline', 'worst_b2_ajrd_delta_vs_offline', 'worst_b2_aj_drop_vs_offline']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
