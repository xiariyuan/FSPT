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
from utils.coords import find_reentry_events

OUT_ROOT = Path('outputs/paper_discovery_2026-06-27/b2wt_adaptive_dev')
DATASETS = {
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
        'offline_AJ_RD_256': 0.3763,
        'offline_AJ_256': 80.0721,
        'p2_AJ_RD_256': 0.4863,
        'p2_AJ_256': 79.9035,
    },
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
        'offline_AJ_RD_256': 0.5546,
        'offline_AJ_256': 70.0510,
        'p2_AJ_RD_256': 0.6251,
        'p2_AJ_256': 69.0119,
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


def trigger_ok(base_v: np.ndarray, over_v: np.ndarray, t: int, persist: int = 2, k: int = 1) -> bool:
    if invisible_run_before(base_v, t) < int(k):
        return False
    if t + persist > len(over_v):
        return False
    return bool(np.all(over_v[t:t + persist]))


def dist_px(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b) * 256.0))


def over_speed_px(over_p: np.ndarray, t: int) -> float:
    if t <= 0:
        return 0.0
    return dist_px(over_p[t], over_p[t - 1])


def base_visible_consecutive(base_v: np.ndarray, t: int, length: int) -> bool:
    if t + length > len(base_v):
        return False
    return bool(np.all(base_v[t:t + length]))


def choose_post(base_p: np.ndarray, over_p: np.ndarray, base_v: np.ndarray, over_v: np.ndarray, t: int, cfg: Dict[str, Any]) -> int:
    post = int(cfg.get('post', 16))
    # Trigger-time geometry shortening. Use only current trigger-frame distance.
    if 'risk_short_dist' in cfg:
        d = dist_px(base_p[t], over_p[t])
        if d >= float(cfg['risk_short_dist']):
            post = min(post, int(cfg.get('risk_short_post', 4)))
    if 'risk_mid_dist' in cfg:
        d = dist_px(base_p[t], over_p[t])
        if d >= float(cfg['risk_mid_dist']):
            post = min(post, int(cfg.get('risk_mid_post', 8)))
    return post


def compute_dynamic_hi(base_p: np.ndarray, over_p: np.ndarray, base_v: np.ndarray, over_v: np.ndarray, t: int, T: int, cfg: Dict[str, Any]) -> int:
    post = choose_post(base_p, over_p, base_v, over_v, t, cfg)
    max_hi = T if post >= 9999 else min(T, t + post + 1)
    min_keep = int(cfg.get('min_keep', 0))
    base_vis_stop_n = int(cfg.get('base_vis_stop_n', 0))
    close_tau = cfg.get('base_vis_close_tau')
    disagree_tau = cfg.get('base_vis_disagree_tau')
    speed_tau = cfg.get('over_speed_tau')
    dist_tau = cfg.get('dist_tau')
    alpha = cfg.get('dist_alpha')
    stop_over_lost = bool(cfg.get('stop_over_lost', False))
    d0 = max(dist_px(base_p[t], over_p[t]), 1e-6)
    hi = max_hi
    for u in range(t, max_hi):
        age = u - t + 1
        if age <= min_keep:
            continue
        stop = False
        if stop_over_lost and not bool(over_v[u]):
            stop = True
        if speed_tau is not None and u > t and over_speed_px(over_p, u) >= float(speed_tau):
            stop = True
        d = dist_px(base_p[u], over_p[u])
        if dist_tau is not None and d >= float(dist_tau):
            stop = True
        if alpha is not None and d >= float(alpha) * d0:
            stop = True
        if base_vis_stop_n > 0 and base_visible_consecutive(base_v, u, base_vis_stop_n):
            if close_tau is None and disagree_tau is None:
                stop = True
            elif close_tau is not None and d <= float(close_tau):
                stop = True
            elif disagree_tau is not None and bool(over_v[u]) and d >= float(disagree_tau):
                stop = True
        if stop:
            hi = max(t, u)  # stop before frame u; at least allow earlier frames
            break
    return int(max(t, hi))


def build_variant(base: Dict[str, Any], over: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    records = []
    total_events = tracks_with_trigger = true_t = false_t = missed_re = gt_re = 0
    early_stop_events = 0
    shortened_events = 0
    window_lengths: List[int] = []
    for b, o in zip(base['records'], over['records']):
        pred_p = npy(b['pred_tracks'], np.float32).copy()
        pred_v = npy(b['pred_visibility'], bool).copy()
        base_p = pred_p.copy()
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
            while t < T:
                if trigger_ok(base_v[qi], over_v[qi], t, persist=int(cfg.get('persist', 2)), k=int(cfg.get('k', 1))):
                    pre = int(cfg.get('pre', 1))
                    lo = max(0, t - pre)
                    post = choose_post(base_p[qi], over_p[qi], base_v[qi], over_v[qi], t, cfg)
                    max_hi = T if post >= 9999 else min(T, t + post + 1)
                    hi = compute_dynamic_hi(base_p[qi], over_p[qi], base_v[qi], over_v[qi], t, T, cfg)
                    hi = max(lo + 1, min(hi, max_hi))
                    mask[lo:hi] = True
                    triggers.append(int(t))
                    total_events += 1
                    wl = hi - lo
                    window_lengths.append(int(wl))
                    if hi < max_hi:
                        early_stop_events += 1
                    if post < int(cfg.get('post', 16)):
                        shortened_events += 1
                    t = max(hi, t + 1)
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
        r = dict(b)
        r['pred_tracks'] = pred_p.astype(np.float32)
        r['pred_visibility'] = pred_v.astype(bool)
        r['model_name'] = cfg['name']
        r['b2wt_config'] = cfg
        records.append(r)
    payload = dict(base)
    payload['model_name'] = cfg['name']
    payload['b2wt_config'] = cfg
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
        'early_stop_events': int(early_stop_events),
        'shortened_events': int(shortened_events),
        'mean_window_len': round(float(np.mean(window_lengths)), 4) if window_lengths else None,
        'median_window_len': round(float(np.median(window_lengths)), 4) if window_lengths else None,
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


def configs() -> List[Dict[str, Any]]:
    base = {'k': 1, 'pre': 1, 'post': 16, 'persist': 2}
    cs = []
    def add(name, **kw):
        c = dict(base); c.update(kw); c['name'] = name; cs.append(c)
    add('p2_w16')
    add('p2_w8', post=8)
    add('p2_w4', post=4)
    add('p2_stop_lost_min0', stop_over_lost=True, min_keep=0)
    add('p2_stop_lost_min4', stop_over_lost=True, min_keep=4)
    add('p2_stop_speed4_min2', over_speed_tau=4, min_keep=2)
    add('p2_stop_speed8_min2', over_speed_tau=8, min_keep=2)
    add('p2_stop_speed16_min2', over_speed_tau=16, min_keep=2)
    add('p2_basevis1', base_vis_stop_n=1, min_keep=2)
    add('p2_basevis2', base_vis_stop_n=2, min_keep=2)
    add('p2_basevis_disagree4', base_vis_stop_n=1, base_vis_disagree_tau=4, min_keep=2)
    add('p2_basevis_disagree8', base_vis_stop_n=1, base_vis_disagree_tau=8, min_keep=2)
    add('p2_riskdist4_post4', risk_short_dist=4, risk_short_post=4)
    add('p2_riskdist8_post4', risk_short_dist=8, risk_short_post=4)
    add('p2_riskdist8_post8', risk_short_dist=8, risk_short_post=8)
    add('p2_riskdist16_post8', risk_short_dist=16, risk_short_post=8)
    add('p2_disttau8_min4', dist_tau=8, min_keep=4)
    add('p2_disttau16_min4', dist_tau=16, min_keep=4)
    add('p2_alpha2_min4', dist_alpha=2.0, min_keep=4)
    add('p2_alpha3_min4', dist_alpha=3.0, min_keep=4)
    add('p2_combined_conservative', stop_over_lost=True, over_speed_tau=16, min_keep=4, base_vis_stop_n=1, base_vis_disagree_tau=8)
    add('p2_combined_risk_short', stop_over_lost=True, over_speed_tau=16, min_keep=4, risk_short_dist=8, risk_short_post=8)
    return cs


def run_dataset(name: str, dcfg: Dict[str, Any], cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(dcfg['base'], map_location='cpu', weights_only=False)
    over = torch.load(dcfg['override'], map_location='cpu', weights_only=False)
    rows = []
    for cfg in cfgs:
        print('===', name, cfg['name'], '===', flush=True)
        payload, trig = build_variant(base, over, cfg)
        cache = out_dir / f"{cfg['name']}.pt"
        ajrd_json = out_dir / f"{cfg['name']}_ajrd.json"
        torch.save(payload, cache)
        ajrd = eval_ajrd(cache, ajrd_json)
        std = standard_metrics(payload['records'])
        row = {
            'dataset': name,
            'name': cfg['name'],
            'config': cfg,
            'cache': str(cache),
            'true_AJ_RD_256': ajrd.get('true_AJ_RD_256'),
            'true_AJ_RD': ajrd.get('true_AJ_RD'),
            'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
            'AJ_256_pct': std['AJ_256_pct'],
            'OA_256_pct': std['OA_256_pct'],
            'delta_avg_256_pct': std['delta_avg_256_pct'],
            'AJ_RD_gain_vs_offline': round(float(ajrd.get('true_AJ_RD_256')) - float(dcfg['offline_AJ_RD_256']), 6),
            'AJ_delta_vs_offline': round(float(std['AJ_256_pct']) - float(dcfg['offline_AJ_256']), 6),
            'AJ_RD_delta_vs_p2_ref': round(float(ajrd.get('true_AJ_RD_256')) - float(dcfg['p2_AJ_RD_256']), 6),
            'AJ_delta_vs_p2_ref': round(float(std['AJ_256_pct']) - float(dcfg['p2_AJ_256']), 6),
            **trig,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ['dataset','name','true_AJ_RD_256','AJ_256_pct','AJ_RD_delta_vs_p2_ref','AJ_delta_vs_p2_ref','mean_window_len','early_stop_events','shortened_events']}, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfgs = configs()
    by_dataset = {}
    all_rows = []
    for dname, dcfg in DATASETS.items():
        rows = run_dataset(dname, dcfg, cfgs)
        by_dataset[dname] = rows
        all_rows.extend(rows)
    # aggregate deltas vs P2 equally across datasets
    agg = []
    for cfg in cfgs:
        rs = [r for r in all_rows if r['name'] == cfg['name']]
        agg.append({
            'name': cfg['name'],
            'mean_AJ_RD_delta_vs_p2_ref': round(float(np.mean([r['AJ_RD_delta_vs_p2_ref'] for r in rs])), 6),
            'mean_AJ_delta_vs_p2_ref': round(float(np.mean([r['AJ_delta_vs_p2_ref'] for r in rs])), 6),
            'rgb_AJ_RD_delta_vs_p2_ref': [r for r in rs if r['dataset']=='rgb_dev10'][0]['AJ_RD_delta_vs_p2_ref'],
            'rgb_AJ_delta_vs_p2_ref': [r for r in rs if r['dataset']=='rgb_dev10'][0]['AJ_delta_vs_p2_ref'],
            'davis_AJ_RD_delta_vs_p2_ref': [r for r in rs if r['dataset']=='davis'][0]['AJ_RD_delta_vs_p2_ref'],
            'davis_AJ_delta_vs_p2_ref': [r for r in rs if r['dataset']=='davis'][0]['AJ_delta_vs_p2_ref'],
        })
    # success-like candidates: keep AJ_RD within -0.003 both datasets, improve AJ in at least one / average
    candidates = [a for a in agg if a['rgb_AJ_RD_delta_vs_p2_ref'] >= -0.003 and a['davis_AJ_RD_delta_vs_p2_ref'] >= -0.003]
    candidates = sorted(candidates, key=lambda x: (x['mean_AJ_delta_vs_p2_ref'], x['mean_AJ_RD_delta_vs_p2_ref']), reverse=True)
    summary = {
        'protocol': 'Development-only B2-WT adaptive early-stop sweep on DAVIS + RGB dev0-9. No RGB fresh20-49 used.',
        'datasets': {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k, v in DATASETS.items()},
        'configs': cfgs,
        'rows': all_rows,
        'aggregate': sorted(agg, key=lambda x: (x['mean_AJ_delta_vs_p2_ref'], x['mean_AJ_RD_delta_vs_p2_ref']), reverse=True),
        'retention_candidates': candidates,
    }
    (OUT_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('wrote', OUT_ROOT / 'summary.json', flush=True)
    print('top retention candidates')
    for c in candidates[:10]:
        print(json.dumps(c, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
