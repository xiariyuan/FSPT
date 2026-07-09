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

ROOT = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke')
OUT = ROOT / 'rgb_b2_gate_ablation_10video'
BASE_CACHE = ROOT / 'cotracker3_offline_rgb_stacking_10video.pt'
OVER_CACHE = ROOT / 'cotracker3_online_rgb_stacking_10video.pt'
VIDEO_FAIL = 'rgb_stacking_000008'


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


def px_dist(a_yx: np.ndarray, b_yx: np.ndarray, osz: np.ndarray) -> float:
    h, w = float(osz[0]), float(osz[1])
    scale = np.asarray([max(h - 1.0, 1.0), max(w - 1.0, 1.0)], dtype=np.float32)
    return float(np.linalg.norm((a_yx - b_yx) * scale))


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, base_p: np.ndarray, over_p: np.ndarray, osz: np.ndarray, t: int, qt: int, cfg: Dict[str, Any]) -> bool:
    if invisible_run_before(base_v, t) < int(cfg.get('k', 1)):
        return False
    persist = int(cfg.get('over_persist', 1))
    if t + persist > len(over_v):
        return False
    if not bool(np.all(over_v[t:t+persist])):
        return False
    base_mode = cfg.get('base_mode', 'any')
    if base_mode == 'invisible_at_t' and bool(base_v[t]):
        return False
    if base_mode == 'invisible_persist2':
        hi = min(len(base_v), t + 2)
        if bool(np.any(base_v[t:hi])):
            return False
    if base_mode == 'invisible_persist4':
        hi = min(len(base_v), t + 4)
        if bool(np.any(base_v[t:hi])):
            return False
    max_disagree = cfg.get('max_disagree_px')
    if max_disagree is not None and bool(base_v[t]) and bool(over_v[t]):
        if px_dist(base_p[t], over_p[t], osz) > float(max_disagree):
            return False
    min_disagree = cfg.get('min_disagree_px')
    if min_disagree is not None and bool(base_v[t]) and bool(over_v[t]):
        if px_dist(base_p[t], over_p[t], osz) < float(min_disagree):
            return False
    return True


def build_variant(base: Dict[str, Any], over: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    records = []
    total_triggers = tracks_with_trigger = true_t = false_t = missed_re = gt_re = 0
    per_video_trigger_stats = []
    for b, o in zip(base['records'], over['records']):
        pred_p = npy(b['pred_tracks'], np.float32).copy()
        pred_v = npy(b['pred_visibility'], bool).copy()
        base_p = pred_p.copy()
        base_v = pred_v.copy()
        over_p = npy(o['pred_tracks'], np.float32)
        over_v = npy(o['pred_visibility'], bool)
        gt_v = npy(b['gt_visibility'], bool)
        qpts = npy(b['query_points'], np.float32)
        osz = npy(b['original_size'], np.float32)
        n, T = pred_v.shape
        video_stats = {
            'video_id': str(b['video_id']),
            'tracks': int(n),
            'gt_reentry_tracks': 0,
            'triggers': 0,
            'tracks_with_trigger': 0,
            'triggered_reentry_tracks': 0,
            'triggered_nonreentry_tracks': 0,
            'missed_reentry_tracks': 0,
        }
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            has_re = bool(find_reentry_events(gt_v[qi], qt))
            if has_re:
                gt_re += 1
                video_stats['gt_reentry_tracks'] += 1
            mask = np.zeros(T, dtype=bool)
            triggers: List[int] = []
            t = max(1, qt + 1)
            while t < T:
                if trigger_ok(base_v[qi], over_v[qi], base_p[qi], over_p[qi], osz, t, qt, cfg):
                    lo = max(0, t - int(cfg.get('pre', 1)))
                    post = int(cfg.get('post', 9999))
                    hi = T if post >= 9999 else min(T, t + post + 1)
                    mask[lo:hi] = True
                    triggers.append(int(t))
                    t = hi
                else:
                    t += 1
            if triggers:
                total_triggers += len(triggers)
                tracks_with_trigger += 1
                video_stats['triggers'] += len(triggers)
                video_stats['tracks_with_trigger'] += 1
                if has_re:
                    true_t += 1
                    video_stats['triggered_reentry_tracks'] += 1
                else:
                    false_t += 1
                    video_stats['triggered_nonreentry_tracks'] += 1
                pred_p[qi, mask] = over_p[qi, mask]
                pred_v[qi, mask] = over_v[qi, mask]
            elif has_re:
                missed_re += 1
                video_stats['missed_reentry_tracks'] += 1
        r = dict(b)
        r['pred_tracks'] = pred_p.astype(np.float32)
        r['pred_visibility'] = pred_v.astype(bool)
        r['model_name'] = cfg['name']
        r['rgb_b2_gate_config'] = cfg
        records.append(r)
        per_video_trigger_stats.append(video_stats)
    payload = dict(base)
    payload['model_name'] = cfg['name']
    payload['rgb_b2_gate_config'] = cfg
    payload['records'] = records
    stats = {
        'total_triggers': int(total_triggers),
        'tracks_with_trigger': int(tracks_with_trigger),
        'gt_reentry_tracks': int(gt_re),
        'triggered_reentry_tracks': int(true_t),
        'triggered_nonreentry_tracks': int(false_t),
        'missed_reentry_tracks': int(missed_re),
        'trigger_precision_track': round(true_t / max(tracks_with_trigger, 1), 6),
        'trigger_recall_track': round(true_t / max(gt_re, 1), 6),
        'per_video_trigger_stats': per_video_trigger_stats,
    }
    return payload, stats


def eval_ajrd(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def standard_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    per = []
    for r in records:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
        row = {
            'video_id': str(r['video_id']),
            'AJ_256_pct': round(float(m.get('AJ', 0.0)) * 100.0, 4),
            'OA_256_pct': round(float(m.get('OA', 0.0)) * 100.0, 4),
            'delta_avg_256_pct': round(float(m.get('average_pts_within_thresh', 0.0)) * 100.0, 4),
        }
        per.append(row); aj.append(row['AJ_256_pct']); oa.append(row['OA_256_pct']); da.append(row['delta_avg_256_pct'])
    return {
        'AJ_256_pct': round(float(np.mean(aj)), 4),
        'OA_256_pct': round(float(np.mean(oa)), 4),
        'delta_avg_256_pct': round(float(np.mean(da)), 4),
        'per_video': per,
    }


def per_video_ajrd(records: List[Dict[str, Any]]) -> Dict[str, float]:
    out = {}
    for r in records:
        h, w = int(r['original_size'][0]), int(r['original_size'][1])
        m = compute_reentry_metrics(
            pred_tracks=npy(r['pred_tracks'], np.float32),
            gt_tracks=npy(r['gt_tracks'], np.float32),
            pred_vis=npy(r['pred_visibility'], bool),
            gt_vis=npy(r['gt_visibility'], bool),
            query_points=npy(r['query_points'], np.float32),
            height=h,
            width=w,
        )
        out[str(r['video_id'])] = m.get('true_AJ_RD_256')
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = torch.load(BASE_CACHE, map_location='cpu', weights_only=False)
    over = torch.load(OVER_CACHE, map_location='cpu', weights_only=False)
    configs = [
        {'name': 'baseline_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'persist2_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 2, 'base_mode': 'any'},
        {'name': 'persist3_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 3, 'base_mode': 'any'},
        {'name': 'persist4_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 4, 'base_mode': 'any'},
        {'name': 'k2_full', 'k': 2, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'k4_full', 'k': 4, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'base_invis_t_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'invisible_at_t'},
        {'name': 'base_invis_t_persist2_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 2, 'base_mode': 'invisible_at_t'},
        {'name': 'base_invis_next2_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'invisible_persist2'},
        {'name': 'post16', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'post32', 'k': 1, 'pre': 1, 'post': 32, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'persist2_post32', 'k': 1, 'pre': 1, 'post': 32, 'over_persist': 2, 'base_mode': 'any'},
        {'name': 'k2_post32', 'k': 2, 'pre': 1, 'post': 32, 'over_persist': 1, 'base_mode': 'any'},
        {'name': 'base_invis_t_post32', 'k': 1, 'pre': 1, 'post': 32, 'over_persist': 1, 'base_mode': 'invisible_at_t'},
        {'name': 'maxdisagree16_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any', 'max_disagree_px': 16},
        {'name': 'maxdisagree32_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any', 'max_disagree_px': 32},
        {'name': 'maxdisagree64_full', 'k': 1, 'pre': 1, 'post': 9999, 'over_persist': 1, 'base_mode': 'any', 'max_disagree_px': 64},
    ]
    rows = []
    for cfg in configs:
        print('===', cfg['name'], '===', flush=True)
        payload, trig = build_variant(base, over, cfg)
        cp = OUT / f"{cfg['name']}.pt"
        jp = OUT / f"{cfg['name']}_ajrd.json"
        torch.save(payload, cp)
        ajrd = eval_ajrd(cp, jp)
        std = standard_metrics(payload['records'])
        pv_ajrd = per_video_ajrd(payload['records'])
        fail = VIDEO_FAIL
        row = {
            'name': cfg['name'],
            'config': cfg,
            'cache': str(cp),
            'true_AJ_RD_256': ajrd.get('true_AJ_RD_256'),
            'true_AJ_RD': ajrd.get('true_AJ_RD'),
            'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
            'AJ_256_pct': std['AJ_256_pct'],
            'OA_256_pct': std['OA_256_pct'],
            'delta_avg_256_pct': std['delta_avg_256_pct'],
            'rgb000008_AJ_RD_256': pv_ajrd.get(fail),
            'rgb000008_AJ_256_pct': next((x['AJ_256_pct'] for x in std['per_video'] if x['video_id'] == fail), None),
            **{k:v for k,v in trig.items() if k != 'per_video_trigger_stats'},
            'rgb000008_trigger': next((x for x in trig['per_video_trigger_stats'] if x['video_id'] == fail), None),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows_sorted = sorted(rows, key=lambda r: (r['true_AJ_RD_256'], r['AJ_256_pct']), reverse=True)
    baseline = next(r for r in rows if r['name'] == 'baseline_full')
    summary = {
        'base_cache': str(BASE_CACHE),
        'override_cache': str(OVER_CACHE),
        'baseline': baseline,
        'best_by_AJ_RD': rows_sorted[:10],
        'best_by_rgb000008_AJ_RD': sorted(rows, key=lambda r: (r.get('rgb000008_AJ_RD_256') or -1), reverse=True)[:10],
        'rows': rows,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('=== BEST AJ_RD ===')
    print(json.dumps(rows_sorted[:8], indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
