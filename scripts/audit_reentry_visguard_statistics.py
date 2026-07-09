#!/usr/bin/env python3
from __future__ import annotations

import json, math, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_visguard_statistics')
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / 'statistics_summary.json'

ROOT = Path('outputs/paper_discovery_2026-06-27')

SETTINGS: Dict[str, Dict[str, Path]] = {
    'rgb_fresh20_49_natural': {
        'offline': ROOT/'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt',
        'b2': ROOT/'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt',
        'b2_vis': ROOT/'reentry_visibility_only_hybrids/rgb_fresh20_49_natural/offline_coord_b2_vis.pt',
        'guard': ROOT/'reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt',
        'guard_vis': ROOT/'reentry_visibility_only_hybrids/rgb_fresh20_49_natural/offline_coord_guard_vis.pt',
    },
    'fresh20_49_translate_L16': {
        'offline': ROOT/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt',
        'b2': ROOT/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/b2_w16_p2_translate_L16_fresh20_49.pt',
        'b2_vis': ROOT/'reentry_visibility_only_hybrids/fresh20_49_translate_L16/offline_coord_b2_vis.pt',
        'guard': ROOT/'reentry_guard_v2_sklearn/fresh20_49_translate_L16/random_forest/random_forest_thr0.40.pt',
        'guard_vis': ROOT/'reentry_visibility_only_hybrids/fresh20_49_translate_L16/offline_coord_guard_vis.pt',
    },
    'fresh20_49_occluder_L16': {
        'offline': ROOT/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt',
        'b2': ROOT/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/b2_w16_p2_occluder_L16_fresh20_49.pt',
        'b2_vis': ROOT/'reentry_visibility_only_hybrids/fresh20_49_occluder_L16/offline_coord_b2_vis.pt',
        'guard': ROOT/'reentry_guard_v2_sklearn/fresh20_49_occluder_L16/random_forest/random_forest_thr0.40.pt',
        'guard_vis': ROOT/'reentry_visibility_only_hybrids/fresh20_49_occluder_L16/offline_coord_guard_vis.pt',
    },
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def per_video_metrics(cache: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in cache['records']:
        vid = str(r['video_id'])
        h, w = int(r['original_size'][0]), int(r['original_size'][1])
        pred = npy(r['pred_tracks'], np.float32)
        gt = npy(r['gt_tracks'], np.float32)
        pv = npy(r['pred_visibility'], bool)
        gv = npy(r['gt_visibility'], bool)
        q = npy(r['query_points'], np.float32)
        m = compute_tapvid_metrics(torch.from_numpy(pred), torch.from_numpy(gt), torch.from_numpy(pv), torch.from_numpy(gv), torch.from_numpy(q), resolution=256, query_mode='strided')
        rm = compute_reentry_metrics(pred, gt, pv, gv, q, h, w)
        out[vid] = {
            'AJ_RD_256': float(rm['true_AJ_RD_256']) if rm.get('true_AJ_RD_256') is not None else np.nan,
            'n_reentry_queries': int(rm.get('n_reentry_queries', 0)),
            'AJ_256': float(m.get('AJ', 0.0)) * 100.0,
            'OA_256': float(m.get('OA', 0.0)) * 100.0,
            'delta_avg_256': float(m.get('average_pts_within_thresh', 0.0)) * 100.0,
        }
    return out


def sign_test_p(pos: int, neg: int) -> float:
    n = int(pos + neg)
    if n <= 0:
        return 1.0
    k = min(pos, neg)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * prob)


def paired_stats(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], metric: str, seed: int = 20260702, n_boot: int = 10000) -> Dict[str, Any]:
    vids = sorted(set(a.keys()) & set(b.keys()))
    vals = []
    used = []
    for v in vids:
        x = a[v].get(metric, np.nan)
        y = b[v].get(metric, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            vals.append(float(x - y))
            used.append(v)
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {'n': 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        boot[i] = float(np.mean(arr[idx]))
    ci = np.percentile(boot, [2.5, 97.5])
    pos = int(np.sum(arr > 0)); neg = int(np.sum(arr < 0)); zero = int(np.sum(arr == 0))
    order = np.argsort(arr)
    return {
        'n': int(arr.size),
        'mean_delta': round(float(np.mean(arr)), 6),
        'median_delta': round(float(np.median(arr)), 6),
        'ci95_bootstrap': [round(float(ci[0]), 6), round(float(ci[1]), 6)],
        'positive_videos': pos,
        'negative_videos': neg,
        'zero_videos': zero,
        'sign_test_p_two_sided': round(float(sign_test_p(pos, neg)), 8),
        'min_delta': round(float(np.min(arr)), 6),
        'max_delta': round(float(np.max(arr)), 6),
        'top_negative': [{'video_id': used[i], 'delta': round(float(arr[i]), 6)} for i in order[:5]],
        'top_positive': [{'video_id': used[i], 'delta': round(float(arr[i]), 6)} for i in order[-5:][::-1]],
    }


def aggregate(perv: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    metrics = ['AJ_RD_256', 'AJ_256', 'OA_256', 'delta_avg_256']
    out = {'n_videos': len(perv)}
    for m in metrics:
        vals = [v[m] for v in perv.values() if np.isfinite(v.get(m, np.nan))]
        out[m] = round(float(np.mean(vals)), 6) if vals else None
    out['n_reentry_queries'] = int(sum(v.get('n_reentry_queries', 0) for v in perv.values()))
    return out


def main() -> None:
    results: List[Dict[str, Any]] = []
    for setting, paths in SETTINGS.items():
        missing = {k: str(v) for k, v in paths.items() if not v.exists()}
        if missing:
            raise FileNotFoundError(f'{setting} missing: {missing}')
        caches = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in paths.items()}
        perv = {k: per_video_metrics(c) for k, c in caches.items()}
        aggregates = {k: aggregate(v) for k, v in perv.items()}
        pairs = [
            ('b2_vis', 'offline'),
            ('b2_vis', 'b2'),
            ('b2_vis', 'guard'),
            ('guard_vis', 'offline'),
            ('guard_vis', 'guard'),
            ('guard_vis', 'b2_vis'),
            ('b2', 'offline'),
            ('guard', 'b2'),
        ]
        comps: Dict[str, Dict[str, Any]] = {}
        for a, b in pairs:
            if a in perv and b in perv:
                comps[f'{a}_vs_{b}'] = {
                    metric: paired_stats(perv[a], perv[b], metric, seed=20260702 + len(setting) + len(metric) + len(a))
                    for metric in ['AJ_RD_256', 'AJ_256', 'OA_256', 'delta_avg_256']
                }
        results.append({
            'setting': setting,
            'paths': {k: str(v) for k, v in paths.items()},
            'aggregates': aggregates,
            'comparisons': comps,
        })
    summary = {
        'method_note': 'B2-Vis = offline coordinates + B2-W16-P2 visibility. Guard-Vis = offline coordinates + ReEntry-Guard visibility. Paired deltas are per-video method A minus method B.',
        'results': results,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        compact.append({
            'setting': r['setting'],
            'aggregates': {k: {m: r['aggregates'][k][m] for m in ['AJ_RD_256','AJ_256','OA_256']} for k in ['offline','b2','b2_vis','guard','guard_vis']},
            'b2_vis_vs_offline_AJRD': r['comparisons']['b2_vis_vs_offline']['AJ_RD_256'],
            'b2_vis_vs_b2_AJRD': r['comparisons']['b2_vis_vs_b2']['AJ_RD_256'],
            'guard_vis_vs_guard_AJRD': r['comparisons']['guard_vis_vs_guard']['AJ_RD_256'],
        })
    print(json.dumps({'summary_path': str(OUT), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
