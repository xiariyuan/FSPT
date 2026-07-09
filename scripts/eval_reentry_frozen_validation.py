#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def cache_path(root: Path, fam: str, model: str, tag: str) -> Path:
    d = root / f'{fam}_L16' / 'predictions'
    if model == 'offline':
        return d / f'cotracker3_offline_{fam}_L16_{tag}.pt'
    if model == 'online':
        return d / f'cotracker3_online_{fam}_L16_{tag}.pt'
    if model == 'b2_w16_p2':
        return d / f'b2_w16_p2_{fam}_L16_{tag}.pt'
    raise ValueError(model)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, k: int = 1, persist: int = 2) -> bool:
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


def build_b2(base: Dict[str, Any], over: Dict[str, Any], out_path: Path, name: str) -> Dict[str, Any]:
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
                if trigger_ok(base_v[qi], over_v[qi], t, k=1, persist=2):
                    lo = max(0, t - 1)
                    hi = min(T, t + 16 + 1)
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
        r['b2_method'] = 'base_offline_override_online_persist2_w16_frozen_validation'
        records.append(r)
    payload = dict(base)
    payload['model_name'] = name
    payload['b2_method'] = 'base_offline_override_online_persist2_w16_frozen_validation'
    payload['records'] = records
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


def reentry_sanity(cache: Dict[str, Any]) -> Dict[str, Any]:
    n = re_q = events = 0
    lens = []
    for r in cache['records']:
        gv = npy(r['gt_visibility'], bool)
        q = npy(r['query_points'], np.float32)
        n += int(gv.shape[0])
        for i in range(gv.shape[0]):
            ev = find_reentry_events(gv[i], int(round(float(q[i, 0]))))
            if ev:
                re_q += 1
                events += len(ev)
                lens.extend([int(e['occ_length']) for e in ev])
    return {
        'num_queries': int(n),
        'num_reentry_queries': int(re_q),
        'reentry_query_rate': round(re_q / max(n, 1), 6),
        'num_reentry_events': int(events),
        'occ_length_mean': round(float(np.mean(lens)), 6) if lens else None,
        'occ_length_median': round(float(np.median(lens)), 6) if lens else None,
    }


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


def run_family(root: Path, fam: str, tag: str) -> Dict[str, Any]:
    off = torch.load(cache_path(root, fam, 'offline', tag), map_location='cpu', weights_only=False)
    on = torch.load(cache_path(root, fam, 'online', tag), map_location='cpu', weights_only=False)
    check_alignment(off, on)
    b2_path = cache_path(root, fam, 'b2_w16_p2', tag)
    trigger_stats = build_b2(off, on, b2_path, name=f'b2_w16_p2_{fam}_L16_{tag}')
    rows = [
        eval_one('offline', cache_path(root, fam, 'offline', tag)),
        eval_one('online', cache_path(root, fam, 'online', tag)),
        eval_one('b2_w16_p2', b2_path),
    ]
    by = {r['name']: r for r in rows}
    def delta(a: str, b: str) -> Dict[str, float]:
        return {
            'AJ_RD_256': round(float(by[a]['AJ_RD_256']) - float(by[b]['AJ_RD_256']), 6),
            'AJ_256': round(float(by[a]['AJ_256']) - float(by[b]['AJ_256']), 6),
            'OA_256': round(float(by[a]['OA_256']) - float(by[b]['OA_256']), 6),
        }
    summary = {
        'family': fam,
        'L': 16,
        'sanity': reentry_sanity(off),
        'methods': rows,
        'gains': {
            'online_vs_offline': delta('online', 'offline'),
            'b2_vs_offline': delta('b2_w16_p2', 'offline'),
            'b2_vs_online': delta('b2_w16_p2', 'online'),
        },
        'b2_trigger_stats': trigger_stats,
    }
    (root / f'{fam}_L16' / 'eval_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--families', default='translate,occluder')
    ap.add_argument('--output-json', default=None)
    args = ap.parse_args()
    root = Path(args.root)
    families = [x.strip() for x in args.families.split(',') if x.strip()]
    results = [run_family(root, fam, args.tag) for fam in families]
    out_path = Path(args.output_json) if args.output_json else root / f'{args.tag}_frozen_validation_summary.json'
    out_path.write_text(json.dumps({'split': args.tag, 'protocol': 'frozen_L16', 'results': results}, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        compact.append({
            'family': r['family'],
            'sanity': r['sanity'],
            'methods': [{k: m[k] for k in ['name','AJ_RD_256','AJ_256','OA_256','delta_avg_256']} for m in r['methods']],
            'gains': r['gains'],
            'trigger': r['b2_trigger_stats'],
        })
    print(json.dumps({'summary_path': str(out_path), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
