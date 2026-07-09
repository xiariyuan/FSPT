#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.tapvid_official_eval import sample_queries_strided, sample_queries_first, compute_tapvid_metrics_official


def pct(x):
    return round(float(x) * 100.0, 4)


def eval_sample(sample, query_mode='strided', query_stride=5):
    video = np.asarray(sample['video'])
    points = np.asarray(sample['points'], dtype=np.float32)  # assumed normalized [x,y]
    occ = np.asarray(sample['occluded'], dtype=bool)
    if query_mode == 'first':
        q = sample_queries_first(occ, points, video)
    else:
        q = sample_queries_strided(occ, points, video, query_stride=query_stride)
    query_points = q['query_points']
    target_points = q['target_points']
    target_occ = q['occluded']
    # GT as prediction. Official metric expects [x,y] tracks and occluded bools.
    metrics = compute_tapvid_metrics_official(
        query_points=query_points,
        gt_occluded=target_occ,
        gt_tracks=target_points * 255.0,
        pred_occluded=target_occ,
        pred_tracks=target_points * 255.0,
        query_mode=query_mode,
        thresholds=(1, 2, 4, 8, 16),
        get_trackwise_metrics=False,
    )
    flat = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in metrics.items()}
    return {
        'n_queries': int(query_points.shape[1]),
        'AJ_pct': pct(flat['average_jaccard']),
        'OA_pct': pct(flat['occlusion_accuracy']),
        'delta_avg_pct': pct(flat['average_pts_within_thresh']),
        'video_shape': list(video.shape),
        'points_min': float(np.nanmin(points)),
        'points_max': float(np.nanmax(points)),
        'occ_rate': float(np.mean(occ)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl')
    ap.add_argument('--out-json', default='outputs/paper_discovery_2026-06-27/rgb_stacking_smoke/rgb_stacking_gt_sanity.json')
    ap.add_argument('--max-videos', type=int, default=5)
    ap.add_argument('--query-mode', default='strided')
    ap.add_argument('--query-stride', type=int, default=5)
    args = ap.parse_args()
    data = pickle.load(open(args.pkl, 'rb'))
    n = min(len(data), args.max_videos) if args.max_videos > 0 else len(data)
    rows = []
    for i in range(n):
        r = eval_sample(data[i], query_mode=args.query_mode, query_stride=args.query_stride)
        r['index'] = i
        rows.append(r)
        print('ROW', r, flush=True)
    summary = {
        'dataset': 'tapvid_rgb_stacking',
        'n_total': len(data),
        'n_eval': n,
        'query_mode': args.query_mode,
        'query_stride': args.query_stride,
        'mean_AJ_pct': round(float(np.mean([r['AJ_pct'] for r in rows])), 4),
        'mean_OA_pct': round(float(np.mean([r['OA_pct'] for r in rows])), 4),
        'mean_delta_avg_pct': round(float(np.mean([r['delta_avg_pct'] for r in rows])), 4),
        'rows': rows,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
