#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

OUTDIR = Path('outputs/paper_discovery_2026-06-27/candidate_pool_oracle')


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def check_alignment(ref: Dict[str, Any], other: Dict[str, Any], name: str) -> None:
    if len(ref['records']) != len(other['records']):
        raise ValueError(f'{name}: record count mismatch')
    for i, (ra, rb) in enumerate(zip(ref['records'], other['records'])):
        if str(ra['video_id']) != str(rb['video_id']):
            raise ValueError(f'{name}: video mismatch {i}: {ra["video_id"]} vs {rb["video_id"]}')
        for k in ['query_points', 'gt_tracks', 'gt_visibility', 'original_size']:
            if k == 'gt_visibility':
                ok = np.array_equal(npy(ra[k], bool), npy(rb[k], bool))
            else:
                ok = np.allclose(npy(ra[k]), npy(rb[k]), atol=1e-6, rtol=1e-6)
            if not ok:
                raise ValueError(f'{name}: alignment mismatch record={i} key={k}')


def per_query_ajrd_map(record: Dict[str, Any]) -> Dict[int, float]:
    h, w = int(record['original_size'][0]), int(record['original_size'][1])
    m = compute_reentry_metrics(
        pred_tracks=npy(record['pred_tracks'], np.float32),
        gt_tracks=npy(record['gt_tracks'], np.float32),
        pred_vis=npy(record['pred_visibility'], bool),
        gt_vis=npy(record['gt_visibility'], bool),
        query_points=npy(record['query_points'], np.float32),
        height=h,
        width=w,
    )
    out: Dict[int, float] = {}
    for q in m.get('per_query', []):
        val = q.get('ajrd_summary_256', {}).get('aj_rd')
        if val is not None:
            out[int(q['query_idx'])] = float(val)
    return out


def eval_standard(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    q_total = 0
    for r in cache['records']:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        q_total += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
        aj.append(float(m.get('AJ', 0.0)))
        oa.append(float(m.get('OA', 0.0)))
        da.append(float(m.get('average_pts_within_thresh', 0.0)))
    return {
        'AJ_256': round(float(np.mean(aj)) * 100.0, 4),
        'OA_256': round(float(np.mean(oa)) * 100.0, 4),
        'delta_avg_256': round(float(np.mean(da)) * 100.0, 4),
        'n_records': len(cache['records']),
        'n_queries': int(q_total),
    }


def run_ajrd(cache_path: Path) -> Dict[str, Any]:
    out = cache_path.with_name(cache_path.stem + '_ajrd.json')
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out))


def eval_one(name: str, path: Path) -> Dict[str, Any]:
    cache = torch.load(path, map_location='cpu', weights_only=False)
    std = eval_standard(cache)
    ajrd = run_ajrd(path)
    return {
        'name': name,
        'cache': str(path),
        'AJ_RD_256': ajrd.get('true_AJ_RD_256'),
        'AJ_RD': ajrd.get('true_AJ_RD'),
        'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
        **std,
    }


def build_candidate_pool_oracle(name: str, candidate_paths: Dict[str, Path], out_path: Path) -> Dict[str, Any]:
    caches = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in candidate_paths.items()}
    base = caches['offline']
    for cname, c in caches.items():
        if cname != 'offline':
            check_alignment(base, c, cname)

    records = []
    selected_counts = Counter()
    selected_by_video = []
    improvement_bins = []
    total_re = 0

    for rec_i, br in enumerate(base['records']):
        maps = {cname: per_query_ajrd_map(c['records'][rec_i]) for cname, c in caches.items()}
        n = int(npy(br['query_points']).shape[0])
        chosen = np.array(['offline'] * n, dtype=object)
        chosen_val = np.full(n, np.nan, dtype=np.float32)
        offline_map = maps['offline']

        for qi, bval in offline_map.items():
            total_re += 1
            best_name = 'offline'
            best_val = float(bval)
            for cname, cmap in maps.items():
                val = cmap.get(qi)
                if val is not None and float(val) > best_val + 1e-9:
                    best_val = float(val)
                    best_name = cname
            chosen[qi] = best_name
            chosen_val[qi] = best_val
            selected_counts[best_name] += 1
            improvement_bins.append(float(best_val) - float(bval))

        nr = dict(br)
        pred_p = npy(br['pred_tracks'], np.float32).copy()
        pred_v = npy(br['pred_visibility'], bool).copy()
        for cname, c in caches.items():
            if cname == 'offline':
                continue
            use = chosen == cname
            if np.any(use):
                cr = c['records'][rec_i]
                pred_p[use] = npy(cr['pred_tracks'], np.float32)[use]
                pred_v[use] = npy(cr['pred_visibility'], bool)[use]
        nr['pred_tracks'] = pred_p.astype(np.float32)
        nr['pred_visibility'] = pred_v.astype(bool)
        nr['candidate_pool_oracle_selection'] = {
            'name': name,
            'candidate_counts': dict(Counter(chosen.tolist())),
            'eligible_reentry_queries': int(len(offline_map)),
        }
        records.append(nr)
        selected_by_video.append({
            'video_id': str(br['video_id']),
            'eligible_reentry_queries': int(len(offline_map)),
            'candidate_counts': dict(Counter(chosen.tolist())),
        })

    payload = dict(base)
    payload['records'] = records
    payload['model_name'] = name
    payload['candidate_pool_oracle_selection'] = {
        'name': name,
        'candidates': list(candidate_paths.keys()),
        'eligible_reentry_queries': int(total_re),
        'candidate_counts': dict(selected_counts),
        'candidate_rates': {k: round(v / max(total_re, 1), 6) for k, v in selected_counts.items()},
        'mean_per_query_oracle_improvement_over_offline': round(float(np.mean(improvement_bins)), 6) if improvement_bins else 0.0,
        'median_per_query_oracle_improvement_over_offline': round(float(np.median(improvement_bins)), 6) if improvement_bins else 0.0,
        'selected_by_video': selected_by_video,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload['candidate_pool_oracle_selection']


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidate_paths = {
        'offline': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'),
        'online_global': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt'),
        'b2_fullpost_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_fullpost_p1_rgb_fresh20_49.pt'),
        'b2_w8_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt'),
        'b2_w16_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt'),
        'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt'),
        'b2_w32_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt'),
        'guard_rf_thr0.40': Path('outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt'),
    }
    missing = [str(p) for p in candidate_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))

    oracle_path = OUTDIR / 'rgb_fresh20_49_natural_candidate_pool_oracle.pt'
    selection = build_candidate_pool_oracle('rgb_fresh20_49_natural_candidate_pool_oracle', candidate_paths, oracle_path)

    method_paths = {k: v for k, v in candidate_paths.items()}
    method_paths['candidate_pool_oracle'] = oracle_path
    method_paths['oracle_b2'] = Path('outputs/paper_discovery_2026-06-27/b2_oracle_upper_bound/rgb_fresh20_49_natural/rgb_fresh20_49_natural_oracle_choose_b2_per_reentry_query.pt')
    method_paths['oracle_online'] = Path('outputs/paper_discovery_2026-06-27/b2_oracle_upper_bound/rgb_fresh20_49_natural/rgb_fresh20_49_natural_oracle_choose_online_per_reentry_query.pt')

    rows = [eval_one(name, path) for name, path in method_paths.items() if path.exists()]
    by = {r['name']: r for r in rows}
    gains = {}
    for name in by:
        if name == 'offline':
            continue
        gains[f'{name}_vs_offline'] = {
            'AJ_RD_256': round(float(by[name]['AJ_RD_256']) - float(by['offline']['AJ_RD_256']), 6),
            'AJ_256': round(float(by[name]['AJ_256']) - float(by['offline']['AJ_256']), 6),
            'OA_256': round(float(by[name]['OA_256']) - float(by['offline']['OA_256']), 6),
        }
    for base_name in ['b2_w16_p2', 'guard_rf_thr0.40', 'oracle_b2']:
        if base_name in by:
            gains[f'candidate_pool_oracle_vs_{base_name}'] = {
                'AJ_RD_256': round(float(by['candidate_pool_oracle']['AJ_RD_256']) - float(by[base_name]['AJ_RD_256']), 6),
                'AJ_256': round(float(by['candidate_pool_oracle']['AJ_256']) - float(by[base_name]['AJ_256']), 6),
                'OA_256': round(float(by['candidate_pool_oracle']['OA_256']) - float(by[base_name]['OA_256']), 6),
            }

    summary = {
        'setting': 'rgb_fresh20_49_natural',
        'protocol': 'Per eligible re-entry query, choose the candidate with the highest per-query AJ_RD_256. Non-reentry queries stay offline. This is an upper bound, not deployable.',
        'candidate_paths': {k: str(v) for k, v in candidate_paths.items()},
        'oracle_cache': str(oracle_path),
        'methods': rows,
        'gains': gains,
        'selection': selection,
    }
    out_json = OUTDIR / 'rgb_fresh20_49_natural_candidate_pool_oracle_summary.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = {
        'out_json': str(out_json),
        'oracle_cache': str(oracle_path),
        'methods': [{k: r[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']} for r in rows],
        'gains': gains,
        'selection': {
            'eligible_reentry_queries': selection['eligible_reentry_queries'],
            'candidate_counts': selection['candidate_counts'],
            'candidate_rates': selection['candidate_rates'],
        }
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
