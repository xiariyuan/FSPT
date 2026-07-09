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
from utils.coords import find_reentry_events

OUT_ROOT = Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev')
DATASETS = {
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
        'offline_AJ_RD_256': 0.3763,
        'offline_AJ_256': 80.0721,
    },
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
        'offline_AJ_RD_256': 0.5546,
        'offline_AJ_256': 70.0510,
    },
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, cfg: Dict[str, Any]) -> bool:
    if invisible_run_before(base_v, t) < int(cfg.get('k', 1)):
        return False
    persist = int(cfg.get('over_persist', 1))
    if t + persist > len(over_v):
        return False
    if not bool(np.all(over_v[t:t + persist])):
        return False
    mode = cfg.get('base_mode', 'any')
    if mode == 'invisible_at_t' and bool(base_v[t]):
        return False
    if mode == 'invisible_next2':
        hi = min(len(base_v), t + 2)
        if bool(np.any(base_v[t:hi])):
            return False
    if mode == 'invisible_next4':
        hi = min(len(base_v), t + 4)
        if bool(np.any(base_v[t:hi])):
            return False
    return True


def build_variant(base: Dict[str, Any], over: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    records = []
    total_events = tracks_with_trigger = true_t = false_t = missed_re = gt_re = 0
    for b, o in zip(base['records'], over['records']):
        pred_p = npy(b['pred_tracks'], np.float32).copy()
        pred_v = npy(b['pred_visibility'], bool).copy()
        base_v = pred_v.copy()
        over_p = npy(o['pred_tracks'], np.float32)
        over_v = npy(o['pred_visibility'], bool)
        gt_v = npy(b['gt_visibility'], bool)
        qpts = npy(b['query_points'], np.float32)
        n, T = pred_v.shape
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            has_re = bool(find_reentry_events(gt_v[qi], qt))
            if has_re:
                gt_re += 1
            mask = np.zeros(T, dtype=bool)
            triggers: List[int] = []
            t = max(1, qt + 1)
            max_windows = cfg.get('max_windows')
            while t < T:
                if max_windows is not None and len(triggers) >= int(max_windows):
                    break
                if trigger_ok(base_v[qi], over_v[qi], t, cfg):
                    lo = max(0, t - int(cfg.get('pre', 1)))
                    post = int(cfg.get('post', 16))
                    hi = T if post >= 9999 else min(T, t + post + 1)
                    mask[lo:hi] = True
                    triggers.append(int(t))
                    t = hi
                else:
                    t += 1
            if triggers:
                total_events += len(triggers)
                tracks_with_trigger += 1
                if has_re:
                    true_t += 1
                else:
                    false_t += 1
                pred_p[qi, mask] = over_p[qi, mask]
                pred_v[qi, mask] = over_v[qi, mask]
            elif has_re:
                missed_re += 1
        r = dict(b)
        r['pred_tracks'] = pred_p.astype(np.float32)
        r['pred_visibility'] = pred_v.astype(bool)
        r['model_name'] = cfg['name']
        r['b2_false_cost_guard_config'] = cfg
        records.append(r)
    payload = dict(base)
    payload['model_name'] = cfg['name']
    payload['b2_false_cost_guard_config'] = cfg
    payload['records'] = records
    stats = {
        'total_trigger_events': int(total_events),
        'tracks_with_trigger': int(tracks_with_trigger),
        'gt_reentry_tracks': int(gt_re),
        'triggered_reentry_tracks': int(true_t),
        'triggered_nonreentry_tracks': int(false_t),
        'missed_reentry_tracks': int(missed_re),
        'trigger_precision_track': round(true_t / max(tracks_with_trigger, 1), 6),
        'trigger_recall_track': round(true_t / max(gt_re, 1), 6),
    }
    return payload, stats


def eval_ajrd(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def standard_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    for r in records:
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
        'AJ_256_pct': round(float(np.mean(aj)) * 100.0, 4),
        'OA_256_pct': round(float(np.mean(oa)) * 100.0, 4),
        'delta_avg_256_pct': round(float(np.mean(da)) * 100.0, 4),
    }


def run_dataset(dataset_name: str, ds_cfg: Dict[str, Any], configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(ds_cfg['base'], map_location='cpu', weights_only=False)
    over = torch.load(ds_cfg['override'], map_location='cpu', weights_only=False)
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        print('===', dataset_name, cfg['name'], '===', flush=True)
        payload, trig = build_variant(base, over, cfg)
        cache = out_dir / f"{cfg['name']}.pt"
        ajrd_json = out_dir / f"{cfg['name']}_ajrd.json"
        torch.save(payload, cache)
        ajrd = eval_ajrd(cache, ajrd_json)
        std = standard_metrics(payload['records'])
        row = {
            'dataset': dataset_name,
            'name': cfg['name'],
            'config': cfg,
            'cache': str(cache),
            'true_AJ_RD_256': ajrd.get('true_AJ_RD_256'),
            'true_AJ_RD': ajrd.get('true_AJ_RD'),
            'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
            'AJ_256_pct': std['AJ_256_pct'],
            'OA_256_pct': std['OA_256_pct'],
            'delta_avg_256_pct': std['delta_avg_256_pct'],
            'AJ_RD_gain_vs_offline': round(float(ajrd.get('true_AJ_RD_256')) - float(ds_cfg['offline_AJ_RD_256']), 6),
            'AJ_delta_vs_offline': round(float(std['AJ_256_pct']) - float(ds_cfg['offline_AJ_256']), 6),
            **trig,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        dominated = False
        for q in rows:
            if q is r:
                continue
            if q['true_AJ_RD_256'] >= r['true_AJ_RD_256'] and q['AJ_256_pct'] >= r['AJ_256_pct'] and (q['true_AJ_RD_256'] > r['true_AJ_RD_256'] or q['AJ_256_pct'] > r['AJ_256_pct']):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda x: (x['true_AJ_RD_256'], x['AJ_256_pct']), reverse=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = [
        {'name': 'w16_base', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'w16_persist2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 2, 'base_mode': 'any'},
        {'name': 'w16_persist3', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 3, 'base_mode': 'any'},
        {'name': 'w16_k2', 'k': 2, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'w16_k4', 'k': 4, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'w16_base_invis_t', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'invisible_at_t'},
        {'name': 'w16_base_invis_t_persist2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 2, 'base_mode': 'invisible_at_t'},
        {'name': 'w16_base_invis_next2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'invisible_next2'},
        {'name': 'w16_max1', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any', 'max_windows': 1},
        {'name': 'w16_max2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any', 'max_windows': 2},
        {'name': 'w16_persist2_max1', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 2, 'base_mode': 'any', 'max_windows': 1},
        {'name': 'w16_base_invis_t_max1', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'invisible_at_t', 'max_windows': 1},
    ]
    all_rows = []
    by_dataset = {}
    for dname, dcfg in DATASETS.items():
        rows = run_dataset(dname, dcfg, configs)
        by_dataset[dname] = rows
        all_rows.extend(rows)
    summary = {
        'protocol': 'Development sweep only on DAVIS + RGB dev first-10. Do not use heldout-10 for tuning.',
        'datasets': {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k, v in DATASETS.items()},
        'configs': configs,
        'rows': all_rows,
        'pareto': {k: pareto(v) for k, v in by_dataset.items()},
        'best_by_dataset_AJ_RD': {k: sorted(v, key=lambda r: (r['true_AJ_RD_256'], r['AJ_256_pct']), reverse=True)[:5] for k, v in by_dataset.items()},
        'best_by_dataset_AJ': {k: sorted(v, key=lambda r: (r['AJ_256_pct'], r['true_AJ_RD_256']), reverse=True)[:5] for k, v in by_dataset.items()},
    }
    (OUT_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('=== PARETO ===')
    print(json.dumps(summary['pareto'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
