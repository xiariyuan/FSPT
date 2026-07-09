#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import eval_one, check_alignment, npy

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_visguard_sweep')
ROOT = Path('outputs/paper_discovery_2026-06-27')

VARIANTS = {
    'fullpost_p1': {'W': None, 'persist': 1, 'pre': 1},
    'w8_p2': {'W': 8, 'persist': 2, 'pre': 1},
    'w16_p1': {'W': 16, 'persist': 1, 'pre': 1},
    'w16_p2': {'W': 16, 'persist': 2, 'pre': 1},
    'w32_p2': {'W': 32, 'persist': 2, 'pre': 1},
}

SETTINGS: Dict[str, Dict[str, Path]] = {
    'dev_translate_L16': {
        'offline': ROOT/'reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt',
        'online': ROOT/'reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt',
        'variant_dir': ROOT/'reentry_stress_rgb_dev10/translate_L16/predictions/ablation_L16',
    },
    'dev_occluder_L16': {
        'offline': ROOT/'reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt',
        'online': ROOT/'reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_online_occluder_L16.pt',
        'variant_dir': ROOT/'reentry_stress_rgb_dev10/occluder_L16/predictions/ablation_L16',
    },
    'rgb_fresh20_49_natural': {
        'offline': ROOT/'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt',
        'online': ROOT/'rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt',
        'variant_dir': ROOT/'rgb_stacking_fresh20_49_natural_ablation',
    },
    'fresh20_49_translate_L16': {
        'offline': ROOT/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt',
        'online': ROOT/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_online_translate_L16_fresh20_49.pt',
        'variant_dir': OUTDIR/'fresh20_49_translate_L16/full_variants',
    },
    'fresh20_49_occluder_L16': {
        'offline': ROOT/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt',
        'online': ROOT/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_online_occluder_L16_fresh20_49.pt',
        'variant_dir': OUTDIR/'fresh20_49_occluder_L16/full_variants',
    },
}


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, persist: int = 1, k: int = 1) -> bool:
    if invisible_run_before(base_v, t) < int(k):
        return False
    if t + persist > len(over_v):
        return False
    return bool(np.all(over_v[t:t+persist]))


def find_reentry_events(gt_v: np.ndarray, qt: int):
    T = len(gt_v)
    t = max(0, int(qt) + 1)
    out = []
    while t < T:
        if bool(gt_v[t]):
            t += 1
            continue
        s = t
        while t < T and not bool(gt_v[t]):
            t += 1
        if t < T and bool(gt_v[t]):
            out.append((s, t, t-s))
        t += 1
    return out


def build_full_variant(base: Dict[str, Any], over: Dict[str, Any], out_path: Path, *, name: str, W: Optional[int], persist: int, pre: int = 1) -> Dict[str, Any]:
    check_alignment(base, over, name)
    records = []
    stats = {'total_trigger_events':0, 'tracks_with_trigger':0, 'gt_reentry_tracks':0, 'triggered_reentry_tracks':0, 'triggered_nonreentry_tracks':0, 'missed_reentry_tracks':0}
    for br, orr in zip(base['records'], over['records']):
        pred_p = npy(br['pred_tracks'], np.float32).copy()
        pred_v = npy(br['pred_visibility'], bool).copy()
        base_v = pred_v.copy()
        over_p = npy(orr['pred_tracks'], np.float32)
        over_v = npy(orr['pred_visibility'], bool)
        gt_v = npy(br['gt_visibility'], bool)
        qpts = npy(br['query_points'], np.float32)
        n, T = pred_v.shape
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            has_re = bool(find_reentry_events(gt_v[qi], qt))
            if has_re:
                stats['gt_reentry_tracks'] += 1
            mask = np.zeros(T, dtype=bool)
            triggers = []
            t = max(1, qt + 1)
            while t < T:
                if trigger_ok(base_v[qi], over_v[qi], t, persist=persist, k=1):
                    lo = max(0, t - int(pre))
                    hi = T if W is None else min(T, t + int(W) + 1)
                    mask[lo:hi] = True
                    triggers.append(int(t))
                    stats['total_trigger_events'] += 1
                    t = hi
                else:
                    t += 1
            if triggers:
                stats['tracks_with_trigger'] += 1
                if has_re:
                    stats['triggered_reentry_tracks'] += 1
                else:
                    stats['triggered_nonreentry_tracks'] += 1
                pred_p[qi, mask] = over_p[qi, mask]
                pred_v[qi, mask] = over_v[qi, mask]
            elif has_re:
                stats['missed_reentry_tracks'] += 1
        r = dict(br)
        r['pred_tracks'] = pred_p.astype(np.float32)
        r['pred_visibility'] = pred_v.astype(bool)
        r['model_name'] = name
        r['b2_ablation'] = {'name': name, 'W': W, 'persist': persist, 'pre': pre, 'runtime_trigger': 'predicted base/override visibility only'}
        records.append(r)
    payload = dict(base)
    payload['model_name'] = name
    payload['b2_ablation'] = {'name': name, 'W': W, 'persist': persist, 'pre': pre, 'runtime_trigger': 'predicted base/override visibility only'}
    payload['records'] = records
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    stats['trigger_precision_track'] = round(stats['triggered_reentry_tracks'] / max(stats['tracks_with_trigger'], 1), 6)
    stats['trigger_recall_track'] = round(stats['triggered_reentry_tracks'] / max(stats['gt_reentry_tracks'], 1), 6)
    return stats


def build_vis_only(name: str, coord_path: Path, vis_path: Path, out_path: Path) -> Dict[str, Any]:
    coord = torch.load(coord_path, map_location='cpu', weights_only=False)
    vis = torch.load(vis_path, map_location='cpu', weights_only=False)
    check_alignment(coord, vis, name)
    records = []
    for cr, vr in zip(coord['records'], vis['records']):
        nr = dict(cr)
        nr['pred_tracks'] = npy(cr['pred_tracks'], np.float32).copy()
        nr['pred_visibility'] = npy(vr['pred_visibility'], bool).copy()
        nr['reentry_visguard'] = {'name': name, 'coordinate_source': str(coord_path), 'visibility_source': str(vis_path)}
        records.append(nr)
    payload = dict(coord)
    payload['records'] = records
    payload['model_name'] = name
    payload['reentry_visguard'] = {'name': name, 'coordinate_source': str(coord_path), 'visibility_source': str(vis_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {'cache': str(out_path), 'coordinate_source': str(coord_path), 'visibility_source': str(vis_path)}


def variant_cache_path(setting: str, variant: str, paths: Dict[str, Path]) -> Path:
    d = paths['variant_dir']
    # existing natural/dev naming
    if setting == 'rgb_fresh20_49_natural':
        return d / f'b2_{variant}_rgb_fresh20_49.pt'
    if setting.startswith('dev_'):
        return d / f'b2_{variant}.pt'
    # generated fresh naming
    return d / f'b2_{variant}.pt'


def ensure_full_variant(setting: str, variant: str, cfg: Dict[str, Any], paths: Dict[str, Path], base_cache: Dict[str, Any], online_cache: Dict[str, Any]) -> Dict[str, Any]:
    p = variant_cache_path(setting, variant, paths)
    if p.exists():
        return {'cache': str(p), 'built': False}
    stats = build_full_variant(base_cache, online_cache, p, name=f'{setting}_b2_{variant}', **cfg)
    return {'cache': str(p), 'built': True, 'build_stats': stats}


def run_setting(setting: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    print('RUN_SETTING', setting, flush=True)
    offline_path = paths['offline']
    online_path = paths['online']
    missing = [str(p) for p in [offline_path, online_path] if not p.exists()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    base_cache = torch.load(offline_path, map_location='cpu', weights_only=False)
    online_cache = torch.load(online_path, map_location='cpu', weights_only=False)
    check_alignment(base_cache, online_cache, setting)
    sdir = OUTDIR / setting
    method_paths: Dict[str, Path] = {
        'offline': offline_path,
        'online_global': online_path,
    }
    full_info = {}
    vis_info = {}
    for variant, cfg in VARIANTS.items():
        info = ensure_full_variant(setting, variant, cfg, paths, base_cache, online_cache)
        full_info[variant] = info
        full_path = Path(info['cache'])
        method_paths[f'full_{variant}'] = full_path
        vis_path = sdir / f'visguard_{variant}.pt'
        vis_info[variant] = build_vis_only(f'{setting}_visguard_{variant}', offline_path, full_path, vis_path)
        method_paths[f'vis_{variant}'] = vis_path
    rows = []
    for name, p in method_paths.items():
        rows.append(eval_one(name, p))
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
        if 'full_w16_p2' in by:
            gains[f'{r["name"]}_vs_full_w16_p2'] = {
                'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['full_w16_p2']['AJ_RD_256']), 6),
                'AJ_256': round(float(r['AJ_256']) - float(by['full_w16_p2']['AJ_256']), 6),
                'OA_256': round(float(r['OA_256']) - float(by['full_w16_p2']['OA_256']), 6),
            }
    # Choose dev/final best under AJ >= offline - 1.0 and among visibility-only variants.
    vis_rows = [r for r in rows if r['name'].startswith('vis_')]
    feasible = [r for r in vis_rows if float(r['AJ_256']) >= float(by['offline']['AJ_256']) - 1.0]
    best_vis = max(feasible if feasible else vis_rows, key=lambda r: (float(r['AJ_RD_256']), float(r['AJ_256'])))
    result = {
        'setting': setting,
        'paths': {k: str(v) for k, v in paths.items()},
        'full_info': full_info,
        'vis_info': vis_info,
        'methods': rows,
        'gains': gains,
        'best_vis_under_AJ_loss_le_1': best_vis['name'],
    }
    (sdir / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('DONE_SETTING', setting, 'best', best_vis['name'], best_vis['AJ_RD_256'], best_vis['AJ_256'], flush=True)
    return result


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for setting, paths in SETTINGS.items():
        results.append(run_setting(setting, paths))
    summary = {'variants': VARIANTS, 'results': results}
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        best = next(m for m in r['methods'] if m['name'] == r['best_vis_under_AJ_loss_le_1'])
        key_names = ['offline', 'online_global', 'full_w16_p2', 'vis_w8_p2', 'vis_w16_p1', 'vis_w16_p2', 'vis_w32_p2', 'vis_fullpost_p1', best['name']]
        seen = set(); key_methods = []
        for n in key_names:
            if n in seen: continue
            seen.add(n)
            ms = [m for m in r['methods'] if m['name'] == n]
            if ms:
                key_methods.append({k: ms[0][k] for k in ['name','AJ_RD_256','AJ_256','OA_256']})
        compact.append({'setting': r['setting'], 'best': {k: best[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']}, 'key_methods': key_methods})
    print(json.dumps({'summary_path': str(OUTDIR/'summary.json'), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
