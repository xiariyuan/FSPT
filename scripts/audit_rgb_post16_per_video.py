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

ROOT = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke')
OUT = ROOT / 'rgb_post16_per_video_audit'
CACHES = {
    'offline': ROOT / 'cotracker3_offline_rgb_stacking_10video.pt',
    'online': ROOT / 'cotracker3_online_rgb_stacking_10video.pt',
    'full_post': ROOT / 'b2_offline_base_online_override_10video' / 'b2_rgb10_offline_base_online_override.pt',
    'post16': ROOT / 'rgb_b2_gate_ablation_10video' / 'post16.pt',
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
        for name, payload in payloads.items():
            rec = payload['records'][i]
            mm = {}
            mm.update(standard_metrics(rec))
            mm.update(ajrd_metrics(rec))
            row[name] = mm
        row['deltas'] = {
            'post16_vs_full_AJ_RD_256': delta(row['post16']['AJ_RD_256'], row['full_post']['AJ_RD_256']),
            'post16_vs_full_AJ_256': delta(row['post16']['AJ_256'], row['full_post']['AJ_256']),
            'post16_vs_offline_AJ_RD_256': delta(row['post16']['AJ_RD_256'], row['offline']['AJ_RD_256']),
            'post16_vs_offline_AJ_256': delta(row['post16']['AJ_256'], row['offline']['AJ_256']),
            'post16_vs_online_AJ_RD_256': delta(row['post16']['AJ_RD_256'], row['online']['AJ_RD_256']),
            'post16_vs_online_AJ_256': delta(row['post16']['AJ_256'], row['online']['AJ_256']),
            'full_vs_offline_AJ_RD_256': delta(row['full_post']['AJ_RD_256'], row['offline']['AJ_RD_256']),
            'full_vs_offline_AJ_256': delta(row['full_post']['AJ_256'], row['offline']['AJ_256']),
        }
        rows.append(row)

    def count(pred):
        return int(sum(1 for r in rows if pred(r)))

    counts = {
        'n_videos': n,
        'post16_improves_AJRD_vs_full': count(lambda r: (r['deltas']['post16_vs_full_AJ_RD_256'] or -999) > 0),
        'post16_improves_AJRD_vs_full_ge_0p01': count(lambda r: (r['deltas']['post16_vs_full_AJ_RD_256'] or -999) >= 0.01),
        'post16_improves_AJ_vs_full': count(lambda r: (r['deltas']['post16_vs_full_AJ_256'] or -999) > 0),
        'post16_improves_AJRD_vs_offline': count(lambda r: (r['deltas']['post16_vs_offline_AJ_RD_256'] or -999) > 0),
        'post16_improves_AJRD_vs_offline_ge_0p01': count(lambda r: (r['deltas']['post16_vs_offline_AJ_RD_256'] or -999) >= 0.01),
        'post16_AJ_drop_vs_offline_le_1': count(lambda r: r['deltas']['post16_vs_offline_AJ_256'] is not None and r['deltas']['post16_vs_offline_AJ_256'] >= -1.0),
        'post16_AJ_drop_vs_offline_gt_2': count(lambda r: r['deltas']['post16_vs_offline_AJ_256'] is not None and r['deltas']['post16_vs_offline_AJ_256'] < -2.0),
        'post16_beats_online_AJRD': count(lambda r: (r['deltas']['post16_vs_online_AJ_RD_256'] or -999) > 0),
        'post16_beats_online_AJ_by_10': count(lambda r: r['deltas']['post16_vs_online_AJ_256'] is not None and r['deltas']['post16_vs_online_AJ_256'] >= 10.0),
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
        for name in ['offline', 'online', 'full_post', 'post16']
    }

    summary = {
        'caches': {k: str(v) for k, v in CACHES.items()},
        'aggregate_video_weighted': aggregate,
        'counts': counts,
        'best_post16_gain_vs_full': sorted(rows, key=lambda r: r['deltas']['post16_vs_full_AJ_RD_256'], reverse=True)[:5],
        'worst_post16_delta_vs_full': sorted(rows, key=lambda r: r['deltas']['post16_vs_full_AJ_RD_256'])[:5],
        'worst_post16_AJ_drop_vs_offline': sorted(rows, key=lambda r: r['deltas']['post16_vs_offline_AJ_256'])[:5],
        'rows': rows,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'per_video_rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({k: summary[k] for k in ['aggregate_video_weighted', 'counts', 'best_post16_gain_vs_full', 'worst_post16_delta_vs_full', 'worst_post16_AJ_drop_vs_offline']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
