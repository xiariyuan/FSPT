#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

ROOT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10')
OUT = ROOT / 'stress_statistical_robustness_summary.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def cache_path(family: str, L: int, method: str) -> Path:
    if family == 'translate':
        d = ROOT / f'translate_L{L}' / 'predictions'
        if method == 'offline':
            return d / f'cotracker3_offline_translate_L{L}.pt'
        if method == 'online':
            return d / f'cotracker3_online_translate_L{L}.pt'
        if method == 'b2_w16_p2':
            return d / f'b2_w16_p2_translate_L{L}.pt'
    if family == 'occluder':
        d = ROOT / f'occluder_L{L}' / 'predictions'
        if method == 'offline':
            return d / f'cotracker3_offline_occluder_L{L}.pt'
        if method == 'online':
            return d / f'cotracker3_online_occluder_L{L}.pt'
        if method == 'b2_w16_p2':
            return d / f'b2_w16_p2_occluder_L{L}.pt'
    raise ValueError((family, L, method))


def per_video_metrics(cache: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    rows = {}
    for r in cache['records']:
        vid = str(r['video_id'])
        h, w = int(r['original_size'][0]), int(r['original_size'][1])
        pred = npy(r['pred_tracks'], np.float32)
        gt = npy(r['gt_tracks'], np.float32)
        pvis = npy(r['pred_visibility'], bool)
        gvis = npy(r['gt_visibility'], bool)
        qpts = npy(r['query_points'], np.float32)
        # Standard TAP metrics.
        m = compute_tapvid_metrics(
            torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(pvis), torch.from_numpy(gvis), torch.from_numpy(qpts),
            resolution=256, query_mode='strided'
        )
        # Re-entry metrics.
        rm = compute_reentry_metrics(pred, gt, pvis, gvis, qpts, h, w)
        rows[vid] = {
            'AJ_RD_256': float(rm['true_AJ_RD_256']) if rm.get('true_AJ_RD_256') is not None else np.nan,
            'first_reentry_frame_proxy': float(rm.get('first_reentry_frame_proxy', np.nan)) if rm.get('first_reentry_frame_proxy') is not None else np.nan,
            'n_reentry_queries': int(rm.get('n_reentry_queries', 0)),
            'AJ_256': float(m.get('AJ', 0.0)) * 100.0,
            'OA_256': float(m.get('OA', 0.0)) * 100.0,
            'delta_avg_256': float(m.get('average_pts_within_thresh', 0.0)) * 100.0,
        }
    return rows


def sign_test_p(pos: int, neg: int) -> float:
    n = int(pos + neg)
    if n <= 0:
        return 1.0
    k = min(pos, neg)
    # two-sided exact binomial p under p=0.5
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * prob)


def paired_stats(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], metric: str, seed: int = 20260701, n_boot: int = 10000) -> Dict[str, Any]:
    vids = sorted(set(a.keys()) & set(b.keys()))
    diffs = []
    used = []
    for v in vids:
        x = a[v].get(metric, np.nan)
        y = b[v].get(metric, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            diffs.append(float(x - y))
            used.append(v)
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {'n': 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(np.mean(arr[idx])))
    ci = np.percentile(np.asarray(means), [2.5, 97.5])
    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    zero = int(np.sum(arr == 0))
    return {
        'n': int(arr.size),
        'mean_delta': round(float(np.mean(arr)), 6),
        'median_delta': round(float(np.median(arr)), 6),
        'ci95_bootstrap': [round(float(ci[0]), 6), round(float(ci[1]), 6)],
        'positive_videos': pos,
        'negative_videos': neg,
        'zero_videos': zero,
        'sign_test_p_two_sided': round(float(sign_test_p(pos, neg)), 6),
        'min_delta': round(float(np.min(arr)), 6),
        'max_delta': round(float(np.max(arr)), 6),
        'video_ids': used,
    }


def run_one(family: str, L: int) -> Dict[str, Any]:
    caches = {m: torch.load(cache_path(family, L, m), map_location='cpu', weights_only=False) for m in ['offline', 'online', 'b2_w16_p2']}
    perv = {m: per_video_metrics(caches[m]) for m in caches}
    comparisons = {}
    for a, b in [('b2_w16_p2', 'offline'), ('b2_w16_p2', 'online'), ('online', 'offline')]:
        comparisons[f'{a}_vs_{b}'] = {
            metric: paired_stats(perv[a], perv[b], metric, seed=20260701 + L + len(metric))
            for metric in ['AJ_RD_256', 'AJ_256', 'OA_256', 'delta_avg_256']
        }
    return {
        'family': family,
        'L': int(L),
        'n_videos': len(next(iter(perv.values()))),
        'comparisons': comparisons,
    }


def main() -> None:
    results = []
    for family in ['translate', 'occluder']:
        for L in [8, 16, 32]:
            print('stats', family, L, flush=True)
            results.append(run_one(family, L))
    OUT.write_text(json.dumps({'results': results}, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        c = r['comparisons']['b2_w16_p2_vs_offline']
        compact.append({
            'family': r['family'],
            'L': r['L'],
            'AJ_RD_delta': c['AJ_RD_256'],
            'AJ_delta': c['AJ_256'],
        })
    print(json.dumps({'summary_path': str(OUT), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
