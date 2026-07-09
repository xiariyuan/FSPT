#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

ROOT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10')
OUT = ROOT / 'intersection_query_severity_summary.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def family_dir(family: str, L: int) -> Path:
    if family == 'translate':
        return ROOT / f'translate_L{L}'
    if family == 'occluder':
        return ROOT / f'occluder_L{L}'
    raise ValueError(family)


def method_cache_path(family: str, L: int, method: str) -> Path:
    pred = family_dir(family, L) / 'predictions'
    if family == 'translate':
        if method == 'offline':
            return pred / f'cotracker3_offline_translate_L{L}.pt'
        if method == 'online':
            return pred / f'cotracker3_online_translate_L{L}.pt'
        if method == 'b2_w16_p2':
            return pred / f'b2_w16_p2_translate_L{L}.pt'
    if family == 'occluder':
        if method == 'offline':
            return pred / f'cotracker3_offline_occluder_L{L}.pt'
        if method == 'online':
            return pred / f'cotracker3_online_occluder_L{L}.pt'
        if method == 'b2_w16_p2':
            return pred / f'b2_w16_p2_occluder_L{L}.pt'
    raise ValueError((family, L, method))


def filter_first_dim(v: Any, sel: np.ndarray, n_expected: int) -> Any:
    if isinstance(v, torch.Tensor):
        if v.ndim >= 1 and int(v.shape[0]) == n_expected:
            return v[torch.from_numpy(sel.astype(np.int64))]
        return v
    if isinstance(v, np.ndarray):
        if v.ndim >= 1 and int(v.shape[0]) == n_expected:
            return v[sel]
        return v
    return v


def build_intersection_indices(family: str, lengths: List[int]) -> Dict[str, set[int]]:
    per_L: Dict[int, Dict[str, set[int]]] = {}
    for L in lengths:
        stress = torch.load(family_dir(family, L) / 'stress_dataset.pt', map_location='cpu', weights_only=False)
        d: Dict[str, set[int]] = {}
        for r in stress['records']:
            vid = str(r.get('source_video_id', r['video_id']))
            d[vid] = set(int(x) for x in npy(r['source_query_indices'], np.int64))
        per_L[L] = d
    all_vids = sorted(set.intersection(*[set(x.keys()) for x in per_L.values()]))
    common: Dict[str, set[int]] = {}
    for vid in all_vids:
        common[vid] = set.intersection(*[per_L[L][vid] for L in lengths])
    return common


def save_filtered_cache(family: str, L: int, method: str, common: Dict[str, set[int]], out_dir: Path) -> Path:
    src_cache = torch.load(method_cache_path(family, L, method), map_location='cpu', weights_only=False)
    stress = torch.load(family_dir(family, L) / 'stress_dataset.pt', map_location='cpu', weights_only=False)
    new_records = []
    total = 0
    for cr, sr in zip(src_cache['records'], stress['records']):
        vid = str(sr.get('source_video_id', sr['video_id']))
        src_idx = npy(sr['source_query_indices'], np.int64)
        keep_set = common[vid]
        sel = np.array([i for i, x in enumerate(src_idx) if int(x) in keep_set], dtype=np.int64)
        # Keep ordering by the current cache; common set guarantees same source identities.
        n = int(src_idx.shape[0])
        nr = {}
        for k, v in cr.items():
            nr[k] = filter_first_dim(v, sel, n)
        nr['source_query_indices'] = src_idx[sel].astype(np.int32)
        nr['intersection_query_audit'] = {
            'family': family,
            'length': int(L),
            'n_common_for_source_video': int(len(keep_set)),
            'n_kept_in_record': int(len(sel)),
        }
        total += int(len(sel))
        new_records.append(nr)
    out = dict(src_cache)
    out['records'] = new_records
    out['protocol'] = str(out.get('protocol', '')) + '_intersection_query_L8_L16_L32'
    out['intersection_query_audit'] = {
        'family': family,
        'length': int(L),
        'method': method,
        'source_cache': str(method_cache_path(family, L, method)),
        'n_queries': int(total),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{family}_L{L}_{method}_intersection.pt'
    torch.save(out, out_path)
    return out_path


def eval_standard(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    for r in cache['records']:
        if int(r['query_points'].shape[0]) == 0:
            continue
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
        'AJ_256': round(float(np.mean(aj)) * 100.0, 4) if aj else None,
        'OA_256': round(float(np.mean(oa)) * 100.0, 4) if oa else None,
        'delta_avg_256': round(float(np.mean(da)) * 100.0, 4) if da else None,
    }


def run_ajrd(cache_path: Path) -> Dict[str, Any]:
    out = cache_path.with_suffix('').with_name(cache_path.stem + '_ajrd.json')
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


def gains(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {r['name']: r for r in rows}
    def delta(a: str, b: str) -> Dict[str, float]:
        return {
            'AJ_RD_256': round(float(by[a]['AJ_RD_256']) - float(by[b]['AJ_RD_256']), 6),
            'AJ_256': round(float(by[a]['AJ_256']) - float(by[b]['AJ_256']), 6),
            'OA_256': round(float(by[a]['OA_256']) - float(by[b]['OA_256']), 6),
        }
    return {
        'online_vs_offline': delta('online', 'offline'),
        'b2_vs_offline': delta('b2_w16_p2', 'offline'),
        'b2_vs_online': delta('b2_w16_p2', 'online'),
    }


def run_family(family: str, lengths: List[int]) -> Dict[str, Any]:
    common = build_intersection_indices(family, lengths)
    n_common_total = int(sum(len(v) for v in common.values()))
    out_dir = ROOT / f'{family}_intersection_query_severity'
    summaries = []
    for L in lengths:
        rows = []
        paths = {}
        for method in ['offline', 'online', 'b2_w16_p2']:
            p = save_filtered_cache(family, L, method, common, out_dir)
            paths[method] = p
            rows.append(eval_one(method, p))
        cache0 = torch.load(paths['offline'], map_location='cpu', weights_only=False)
        summaries.append({
            'L': int(L),
            'sanity': reentry_sanity(cache0),
            'methods': rows,
            'gains': gains(rows),
        })
    summary = {
        'family': family,
        'lengths': lengths,
        'n_common_source_queries_total': n_common_total,
        'n_common_by_source_video': {k: len(v) for k, v in common.items()},
        'summaries': summaries,
    }
    (out_dir / 'intersection_query_severity_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    lengths = [8, 16, 32]
    results = [run_family('translate', lengths), run_family('occluder', lengths)]
    OUT.write_text(json.dumps({'results': results}, indent=2, ensure_ascii=False))
    compact = []
    for fam in results:
        for s in fam['summaries']:
            compact.append({
                'family': fam['family'],
                'L': s['L'],
                'n_queries': s['sanity']['num_queries'],
                'reentry_rate': s['sanity']['reentry_query_rate'],
                'b2_vs_offline': s['gains']['b2_vs_offline'],
                'online_vs_offline': s['gains']['online_vs_offline'],
            })
    print(json.dumps({'summary_path': str(OUT), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
