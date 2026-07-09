#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
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

OUTDIR = Path('outputs/paper_discovery_2026-06-27/b2_oracle_upper_bound')
OUT = OUTDIR / 'summary.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


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


def check_alignment(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    if len(a['records']) != len(b['records']):
        raise ValueError('record count mismatch')
    for i, (ra, rb) in enumerate(zip(a['records'], b['records'])):
        if str(ra['video_id']) != str(rb['video_id']):
            raise ValueError(f'video mismatch {i}: {ra["video_id"]} vs {rb["video_id"]}')
        for k in ['query_points', 'gt_tracks', 'gt_visibility', 'original_size']:
            if k == 'gt_visibility':
                ok = np.array_equal(npy(ra[k], bool), npy(rb[k], bool))
            else:
                ok = np.allclose(npy(ra[k]), npy(rb[k]), atol=1e-6, rtol=1e-6)
            if not ok:
                raise ValueError(f'alignment mismatch record={i} key={k}')


def build_oracle(base: Dict[str, Any], cand: Dict[str, Any], out_path: Path, name: str, eps: float = 1e-9) -> Dict[str, Any]:
    check_alignment(base, cand)
    records = []
    selected = total_re = 0
    selected_by_video = []
    for br, cr in zip(base['records'], cand['records']):
        base_map = per_query_ajrd_map(br)
        cand_map = per_query_ajrd_map(cr)
        n = int(npy(br['query_points']).shape[0])
        use = np.zeros(n, dtype=bool)
        for qi, bval in base_map.items():
            cval = cand_map.get(qi)
            if cval is not None:
                total_re += 1
                if cval > bval + eps:
                    use[qi] = True
                    selected += 1
        nr = dict(br)
        pred_p = npy(br['pred_tracks'], np.float32).copy()
        pred_v = npy(br['pred_visibility'], bool).copy()
        cp = npy(cr['pred_tracks'], np.float32)
        cv = npy(cr['pred_visibility'], bool)
        pred_p[use] = cp[use]
        pred_v[use] = cv[use]
        nr['pred_tracks'] = pred_p.astype(np.float32)
        nr['pred_visibility'] = pred_v.astype(bool)
        nr['oracle_selection'] = {
            'name': name,
            'selected_queries': int(np.sum(use)),
            'eligible_reentry_queries': int(len(base_map)),
        }
        records.append(nr)
        selected_by_video.append({'video_id': str(br['video_id']), 'selected_queries': int(np.sum(use)), 'eligible_reentry_queries': int(len(base_map))})
    payload = dict(base)
    payload['records'] = records
    payload['model_name'] = name
    payload['oracle_selection'] = {'name': name, 'selected_queries': int(selected), 'eligible_reentry_queries': int(total_re), 'selection_rate': round(selected / max(total_re, 1), 6), 'selected_by_video': selected_by_video}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload['oracle_selection']


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


def run_setting(setting: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    base = torch.load(paths['offline'], map_location='cpu', weights_only=False)
    b2 = torch.load(paths['b2'], map_location='cpu', weights_only=False)
    online = torch.load(paths['online'], map_location='cpu', weights_only=False)
    setting_dir = OUTDIR / setting
    b2_oracle_path = setting_dir / f'{setting}_oracle_choose_b2_per_reentry_query.pt'
    online_oracle_path = setting_dir / f'{setting}_oracle_choose_online_per_reentry_query.pt'
    b2_sel = build_oracle(base, b2, b2_oracle_path, f'{setting}_oracle_b2')
    online_sel = build_oracle(base, online, online_oracle_path, f'{setting}_oracle_online')
    method_paths = {
        'offline': paths['offline'],
        'online_global': paths['online'],
        'b2_w16_p2': paths['b2'],
        'oracle_b2': b2_oracle_path,
        'oracle_online': online_oracle_path,
    }
    rows = [eval_one(name, path) for name, path in method_paths.items()]
    by = {r['name']: r for r in rows}
    gains = {}
    for r in rows:
        if r['name'] == 'offline':
            continue
        gains[f'{r["name"]}_vs_offline'] = {
            'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['offline']['AJ_RD_256']), 6),
            'AJ_256': round(float(r['AJ_256']) - float(by['offline']['AJ_256']), 6),
            'OA_256': round(float(r['OA_256']) - float(by['offline']['OA_256']), 6),
        }
        if r['name'].startswith('oracle'):
            gains[f'{r["name"]}_vs_b2'] = {
                'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['b2_w16_p2']['AJ_RD_256']), 6),
                'AJ_256': round(float(r['AJ_256']) - float(by['b2_w16_p2']['AJ_256']), 6),
                'OA_256': round(float(r['OA_256']) - float(by['b2_w16_p2']['OA_256']), 6),
            }
    return {
        'setting': setting,
        'paths': {k: str(v) for k, v in paths.items()},
        'methods': rows,
        'gains': gains,
        'oracle_selection': {'oracle_b2': b2_sel, 'oracle_online': online_sel},
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    settings = {
        'rgb_fresh20_49_natural': {
            'offline': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'),
            'online': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt'),
            'b2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/b2_w16_p2_rgb_stacking_fresh20_49.pt'),
        },
        'fresh20_49_translate_L16': {
            'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt'),
            'online': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_online_translate_L16_fresh20_49.pt'),
            'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/b2_w16_p2_translate_L16_fresh20_49.pt'),
        },
        'fresh20_49_occluder_L16': {
            'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt'),
            'online': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_online_occluder_L16_fresh20_49.pt'),
            'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/b2_w16_p2_occluder_L16_fresh20_49.pt'),
        },
    }
    results = [run_setting(name, paths) for name, paths in settings.items()]
    summary = {'protocol': 'Per-query upper bound. For each eligible re-entry query, choose candidate branch over offline iff candidate per-query AJ_RD_256 is higher; non-reentry queries stay offline.', 'results': results}
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        compact.append({
            'setting': r['setting'],
            'methods': [{k: m[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']} for m in r['methods']],
            'gains': r['gains'],
            'oracle_selection': {k: {'selected_queries': v['selected_queries'], 'eligible_reentry_queries': v['eligible_reentry_queries'], 'selection_rate': v['selection_rate']} for k, v in r['oracle_selection'].items()},
        })
    print(json.dumps({'summary_path': str(OUT), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
