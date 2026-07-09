#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

ROOT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10')
OUT = ROOT / 'stress_ablation_L16_summary.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def paths(family: str, L: int = 16) -> Dict[str, Path]:
    if family == 'translate':
        d = ROOT / f'translate_L{L}' / 'predictions'
        return {
            'offline': d / f'cotracker3_offline_translate_L{L}.pt',
            'online': d / f'cotracker3_online_translate_L{L}.pt',
            'out_dir': d / 'ablation_L16',
        }
    if family == 'occluder':
        d = ROOT / f'occluder_L{L}' / 'predictions'
        return {
            'offline': d / f'cotracker3_offline_occluder_L{L}.pt',
            'online': d / f'cotracker3_online_occluder_L{L}.pt',
            'out_dir': d / 'ablation_L16',
        }
    raise ValueError(family)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, k: int = 1, persist: int = 1) -> bool:
    if invisible_run_before(base_v, t) < int(k):
        return False
    if t + persist > len(over_v):
        return False
    return bool(np.all(over_v[t:t + persist]))


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
                raise ValueError(f'alignment mismatch {i} key={k}')


def build_variant(base: Dict[str, Any], over: Dict[str, Any], out_path: Path, *, name: str, W: Optional[int], persist: int, pre: int = 1) -> Dict[str, Any]:
    records = []
    total_events = tracks_with_trigger = true_t = false_t = missed_re = gt_re = 0
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
                gt_re += 1
            mask = np.zeros(T, dtype=bool)
            triggers = []
            t = max(1, qt + 1)
            while t < T:
                if trigger_ok(base_v[qi], over_v[qi], t, k=1, persist=persist):
                    lo = max(0, t - int(pre))
                    hi = T if W is None else min(T, t + int(W) + 1)
                    mask[lo:hi] = True
                    triggers.append(int(t))
                    total_events += 1
                    t = hi
                else:
                    t += 1
            if triggers:
                tracks_with_trigger += 1
                if has_re:
                    true_t += 1
                else:
                    false_t += 1
                pred_p[qi, mask] = over_p[qi, mask]
                pred_v[qi, mask] = over_v[qi, mask]
            elif has_re:
                missed_re += 1
        r = dict(br)
        r['pred_tracks'] = pred_p.astype(np.float32)
        r['pred_visibility'] = pred_v.astype(bool)
        r['model_name'] = name
        r['b2_ablation'] = {'name': name, 'W': W, 'persist': persist, 'pre': pre}
        records.append(r)
    payload = dict(base)
    payload['model_name'] = name
    payload['b2_ablation'] = {'name': name, 'W': W, 'persist': persist, 'pre': pre}
    payload['records'] = records
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {
        'total_trigger_events': int(total_events),
        'tracks_with_trigger': int(tracks_with_trigger),
        'gt_reentry_tracks': int(gt_re),
        'triggered_reentry_tracks': int(true_t),
        'triggered_nonreentry_tracks': int(false_t),
        'missed_reentry_tracks': int(missed_re),
        'trigger_precision_track': round(true_t / max(tracks_with_trigger, 1), 6),
        'trigger_recall_track': round(true_t / max(gt_re, 1), 6),
    }


def eval_standard(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    for r in cache['records']:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
        aj.append(float(m.get('AJ', 0.0)))
        oa.append(float(m.get('OA', 0.0)))
        da.append(float(m.get('average_pts_within_thresh', 0.0)))
    return {
        'AJ_256': round(float(np.mean(aj)) * 100.0, 4),
        'OA_256': round(float(np.mean(oa)) * 100.0, 4),
        'delta_avg_256': round(float(np.mean(da)) * 100.0, 4),
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


def run_family(family: str) -> Dict[str, Any]:
    p = paths(family, 16)
    base = torch.load(p['offline'], map_location='cpu', weights_only=False)
    over = torch.load(p['online'], map_location='cpu', weights_only=False)
    check_alignment(base, over)
    out_dir = p['out_dir']
    specs = [
        ('offline', None, None, None),
        ('online_global', None, None, None),
        ('b2_fullpost_p1', None, None, {'W': None, 'persist': 1, 'pre': 1}),
        ('b2_w8_p2', None, None, {'W': 8, 'persist': 2, 'pre': 1}),
        ('b2_w16_p1', None, None, {'W': 16, 'persist': 1, 'pre': 1}),
        ('b2_w16_p2', None, None, {'W': 16, 'persist': 2, 'pre': 1}),
        ('b2_w32_p2', None, None, {'W': 32, 'persist': 2, 'pre': 1}),
    ]
    rows = []
    trigger_stats = {}
    for name, _, __, params in specs:
        if name == 'offline':
            cache_path = p['offline']
        elif name == 'online_global':
            cache_path = p['online']
        else:
            cache_path = out_dir / f'{name}.pt'
            trigger_stats[name] = build_variant(base, over, cache_path, name=f'{family}_{name}', **params)
        rows.append(eval_one(name, cache_path))
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
    summary = {'family': family, 'L': 16, 'methods': rows, 'gains': gains, 'trigger_stats': trigger_stats}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ablation_L16_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    results = [run_family('translate'), run_family('occluder')]
    OUT.write_text(json.dumps({'results': results}, indent=2, ensure_ascii=False))
    compact = []
    for fam in results:
        for r in fam['methods']:
            compact.append({
                'family': fam['family'],
                'method': r['name'],
                'AJ_RD_256': r['AJ_RD_256'],
                'AJ_256': r['AJ_256'],
                'OA_256': r['OA_256'],
                'delta_vs_offline': fam['gains'].get(f'{r["name"]}_vs_offline'),
            })
    print(json.dumps({'summary_path': str(OUT), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
