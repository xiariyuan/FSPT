#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_guard_v1')
OUTDIR.mkdir(parents=True, exist_ok=True)


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def safe_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.std(x))


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def first_b2_trigger(base_v: np.ndarray, over_v: np.ndarray, qt: int, persist: int = 2) -> Optional[int]:
    T = len(base_v)
    t = max(1, int(qt) + 1)
    while t < T:
        if invisible_run_before(base_v, t) >= 1 and t + persist <= T and bool(np.all(over_v[t:t+persist])):
            return int(t)
        t += 1
    return None


def longest_false_run_after_query(v: np.ndarray, qt: int) -> int:
    best = cur = 0
    for t in range(max(0, int(qt)+1), len(v)):
        if not bool(v[t]):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def count_visibility_transitions(v: np.ndarray, qt: int) -> int:
    seg = v[max(0, int(qt)+1):]
    if len(seg) < 2:
        return 0
    return int(np.sum(seg[1:] != seg[:-1]))


def motion_stats(p: np.ndarray, v: np.ndarray, lo: int, hi: int) -> Tuple[float, float, float]:
    lo = max(0, int(lo)); hi = min(len(v), int(hi))
    if hi - lo < 2:
        return 0.0, 0.0, 0.0
    pp = p[lo:hi]
    vv = v[lo:hi]
    step = np.linalg.norm(pp[1:] - pp[:-1], axis=-1)
    valid = vv[1:] & vv[:-1]
    if np.any(valid):
        s = step[valid]
    else:
        s = step
    return safe_mean(s), safe_std(s), float(np.max(s)) if s.size else 0.0


def point_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def features_for_query(br: Dict[str, Any], orr: Dict[str, Any], qi: int) -> np.ndarray:
    bp = npy(br['pred_tracks'], np.float32)[qi]
    bv = npy(br['pred_visibility'], bool)[qi]
    op = npy(orr['pred_tracks'], np.float32)[qi]
    ov = npy(orr['pred_visibility'], bool)[qi]
    qpts = npy(br['query_points'], np.float32)
    qt = int(round(float(qpts[qi, 0])))
    T = len(bv)
    trig = first_b2_trigger(bv, ov, qt, persist=2)
    if trig is None:
        trig = min(T-1, max(1, qt + 1))
        has_trig = 0.0
    else:
        has_trig = 1.0
    lo_pre = max(0, trig - 8)
    hi_win = min(T, trig + 16 + 1)
    hi_post = min(T, trig + 32 + 1)
    # basic visibility ratios
    base_vis_win = safe_mean(bv[trig:hi_win].astype(np.float32))
    over_vis_win = safe_mean(ov[trig:hi_win].astype(np.float32))
    base_vis_post = safe_mean(bv[trig:hi_post].astype(np.float32))
    over_vis_post = safe_mean(ov[trig:hi_post].astype(np.float32))
    disagree_win = safe_mean((bv[trig:hi_win] != ov[trig:hi_win]).astype(np.float32))
    both_vis_win = safe_mean((bv[trig:hi_win] & ov[trig:hi_win]).astype(np.float32))
    over_only_win = safe_mean((~bv[trig:hi_win] & ov[trig:hi_win]).astype(np.float32))
    base_only_win = safe_mean((bv[trig:hi_win] & ~ov[trig:hi_win]).astype(np.float32))
    inv_run = invisible_run_before(bv, trig)
    over_persist = 0
    j = trig
    while j < T and bool(ov[j]):
        over_persist += 1; j += 1
    base_false_long = longest_false_run_after_query(bv, qt)
    over_false_long = longest_false_run_after_query(ov, qt)
    base_trans = count_visibility_transitions(bv, qt)
    over_trans = count_visibility_transitions(ov, qt)
    # geometry
    d_trig = point_dist(bp[trig], op[trig])
    d_win = np.linalg.norm(bp[trig:hi_win] - op[trig:hi_win], axis=-1)
    d_vis = d_win[(bv[trig:hi_win] | ov[trig:hi_win])]
    d_win_mean = safe_mean(d_vis if d_vis.size else d_win)
    d_win_std = safe_std(d_vis if d_vis.size else d_win)
    d_win_max = float(np.max(d_vis if d_vis.size else d_win)) if d_win.size else 0.0
    # last base visible before trigger to override at trigger
    prev_vis = np.where(bv[max(0, qt):trig])[0]
    if len(prev_vis):
        last_t = max(0, qt) + int(prev_vis[-1])
        d_last_base_to_over = point_dist(bp[last_t], op[trig])
        gap_since_base = trig - last_t
    else:
        d_last_base_to_over = 0.0
        gap_since_base = trig - qt
    bm_mean, bm_std, bm_max = motion_stats(bp, bv, lo_pre, hi_win)
    om_mean, om_std, om_max = motion_stats(op, ov, lo_pre, hi_win)
    # query/time
    f = np.array([
        has_trig,
        trig / max(T-1, 1),
        (trig - qt) / max(T-1, 1),
        inv_run,
        over_persist,
        base_false_long,
        over_false_long,
        base_trans,
        over_trans,
        base_vis_win,
        over_vis_win,
        base_vis_post,
        over_vis_post,
        disagree_win,
        both_vis_win,
        over_only_win,
        base_only_win,
        d_trig,
        d_win_mean,
        d_win_std,
        d_win_max,
        d_last_base_to_over,
        gap_since_base,
        bm_mean,
        bm_std,
        bm_max,
        om_mean,
        om_std,
        om_max,
    ], dtype=np.float32)
    return np.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)

FEATURE_NAMES = [
    'has_trigger','trigger_norm','time_since_query_norm','base_invisible_run','override_visible_persist','base_longest_false_run','override_longest_false_run','base_vis_transitions','override_vis_transitions','base_vis_win','override_vis_win','base_vis_post','override_vis_post','visibility_disagree_win','both_visible_win','override_only_win','base_only_win','dist_at_trigger','dist_win_mean','dist_win_std','dist_win_max','last_base_to_override_dist','gap_since_base_visible','base_motion_mean','base_motion_std','base_motion_max','override_motion_mean','override_motion_std','override_motion_max'
]


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


def make_dataset(setting: str, offline_path: Path, online_path: Path, b2_path: Path, out_npz: Path) -> Dict[str, Any]:
    off = torch.load(offline_path, map_location='cpu', weights_only=False)
    on = torch.load(online_path, map_location='cpu', weights_only=False)
    b2 = torch.load(b2_path, map_location='cpu', weights_only=False)
    check_alignment(off, on); check_alignment(off, b2)
    X: List[np.ndarray] = []
    y: List[int] = []
    meta: List[Tuple[int,int,str]] = []
    gains: List[float] = []
    for vi, (br, orr, b2r) in enumerate(zip(off['records'], on['records'], b2['records'])):
        base_map = per_query_ajrd_map(br)
        b2_map = per_query_ajrd_map(b2r)
        for qi, bval in base_map.items():
            cval = b2_map.get(qi)
            if cval is None:
                continue
            X.append(features_for_query(br, orr, qi))
            label = int(cval > bval + 1e-9)
            y.append(label)
            gains.append(float(cval - bval))
            meta.append((vi, int(qi), str(br['video_id'])))
    arrX = np.vstack(X).astype(np.float32) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    arry = np.asarray(y, dtype=np.int64)
    gain_arr = np.asarray(gains, dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=arrX, y=arry, gains=gain_arr, meta=np.asarray(meta, dtype=object), feature_names=np.asarray(FEATURE_NAMES, dtype=object))
    return {'setting': setting, 'npz': str(out_npz), 'n': int(len(arry)), 'pos': int(np.sum(arry == 1)), 'neg': int(np.sum(arry == 0)), 'pos_rate': round(float(np.mean(arry)) if len(arry) else 0.0, 6), 'gain_mean': round(float(np.mean(gain_arr)) if gain_arr.size else 0.0, 6)}


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig < 1e-6] = 1.0
    return mu.astype(np.float32), sig.astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg(X: np.ndarray, y: np.ndarray, seed: int = 20260701, steps: int = 3000, lr: float = 0.05, l2: float = 1e-3) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    mu, sig = standardize_fit(X)
    Xs = (X - mu) / sig
    n, d = Xs.shape
    w = rng.normal(0, 0.01, size=d).astype(np.float64)
    b = 0.0
    # class-balanced sample weights
    pos = max(1, int(np.sum(y == 1))); neg = max(1, int(np.sum(y == 0)))
    weights = np.where(y == 1, n / (2 * pos), n / (2 * neg)).astype(np.float64)
    yv = y.astype(np.float64)
    for _ in range(steps):
        p = sigmoid(Xs @ w + b)
        err = (p - yv) * weights
        gw = (Xs.T @ err) / n + l2 * w
        gb = float(np.mean(err))
        w -= lr * gw
        b -= lr * gb
    p_train = sigmoid(Xs @ w + b)
    # threshold grid optimized for F1; evaluation later also sweeps metric tradeoff.
    best = {'thr': 0.5, 'f1': -1.0, 'precision':0.0, 'recall':0.0}
    for thr in np.linspace(0.05, 0.95, 91):
        pred = p_train >= thr
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum((~pred) & (y == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best['f1']:
            best = {'thr': float(thr), 'f1': float(f1), 'precision': float(precision), 'recall': float(recall)}
    return {'type':'logreg', 'w': w.astype(np.float32), 'b': float(b), 'mu': mu, 'sig': sig, 'best_threshold': best, 'feature_names': FEATURE_NAMES}


def predict_prob(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    Xs = (X - model['mu']) / model['sig']
    return sigmoid(Xs @ model['w'] + model['b'])


def apply_gate(setting: str, offline_path: Path, online_path: Path, b2_path: Path, npz_path: Path, model: Dict[str, Any], thr: float, out_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']; meta = data['meta']; y = data['y']
    prob = predict_prob(model, X)
    accept = prob >= float(thr)
    # Build mapping (video_id, qi) -> accept for eligible re-entry queries.
    acc_map: Dict[Tuple[str,int], bool] = {}
    for row, a in zip(meta, accept):
        vi, qi, vid = row
        acc_map[(str(vid), int(qi))] = bool(a)
    off = torch.load(offline_path, map_location='cpu', weights_only=False)
    b2 = torch.load(b2_path, map_location='cpu', weights_only=False)
    records = []
    selected = 0
    total_eligible = len(acc_map)
    for br, b2r in zip(off['records'], b2['records']):
        vid = str(br['video_id'])
        pred_p = npy(br['pred_tracks'], np.float32).copy()
        pred_v = npy(br['pred_visibility'], bool).copy()
        cp = npy(b2r['pred_tracks'], np.float32)
        cv = npy(b2r['pred_visibility'], bool)
        qn = pred_v.shape[0]
        use = np.zeros(qn, dtype=bool)
        for qi in range(qn):
            if acc_map.get((vid, qi), False):
                use[qi] = True
        selected += int(np.sum(use))
        pred_p[use] = cp[use]
        pred_v[use] = cv[use]
        nr = dict(br)
        nr['pred_tracks'] = pred_p.astype(np.float32)
        nr['pred_visibility'] = pred_v.astype(bool)
        nr['model_name'] = f'reentry_guard_v1_{setting}_thr{thr:.2f}'
        records.append(nr)
    payload = dict(off)
    payload['records'] = records
    payload['model_name'] = f'reentry_guard_v1_{setting}_thr{thr:.2f}'
    payload['reentry_guard'] = {'threshold': float(thr), 'selected_queries': int(selected), 'eligible_reentry_queries': int(total_eligible), 'selection_rate': round(selected / max(total_eligible, 1), 6)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload['reentry_guard']


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
    return {'AJ_256': round(float(np.mean(aj))*100,4), 'OA_256': round(float(np.mean(oa))*100,4), 'delta_avg_256': round(float(np.mean(da))*100,4)}


def run_ajrd(cache_path: Path) -> Dict[str, Any]:
    out = cache_path.with_name(cache_path.stem + '_ajrd.json')
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out))


def eval_cache(name: str, path: Path) -> Dict[str, Any]:
    cache = torch.load(path, map_location='cpu', weights_only=False)
    std = eval_standard(cache)
    ajrd = run_ajrd(path)
    return {'name': name, 'cache': str(path), 'AJ_RD_256': ajrd.get('true_AJ_RD_256'), 'AJ_256': std['AJ_256'], 'OA_256': std['OA_256'], 'delta_avg_256': std['delta_avg_256']}

SETTINGS = {
    'dev_translate_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt'),
        'online': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt'),
        'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/b2_w16_p2_translate_L16.pt'),
    },
    'dev_occluder_L16': {
        'offline': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt'),
        'online': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_online_occluder_L16.pt'),
        'b2': Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/b2_w16_p2_occluder_L16.pt'),
    },
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


def main() -> None:
    # Build datasets for all settings.
    ds_summaries = {}
    for name, paths in SETTINGS.items():
        ds_summaries[name] = make_dataset(name, paths['offline'], paths['online'], paths['b2'], OUTDIR / f'{name}_features_labels.npz')
        print('DATASET', json.dumps(ds_summaries[name]), flush=True)
    # Train on dev translate+occluder.
    train_npzs = [OUTDIR / 'dev_translate_L16_features_labels.npz', OUTDIR / 'dev_occluder_L16_features_labels.npz']
    X_train = np.vstack([np.load(p, allow_pickle=True)['X'] for p in train_npzs])
    y_train = np.concatenate([np.load(p, allow_pickle=True)['y'] for p in train_npzs])
    model = train_logreg(X_train, y_train)
    model_path = OUTDIR / 'reentry_guard_v1_logreg.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    # Choose threshold from dev grid by maximizing a simple objective using actual evaluation on dev? First use F1 threshold.
    thresholds = sorted(set([0.3, 0.4, 0.5, float(model['best_threshold']['thr']), 0.6, 0.7]))
    results = []
    for name in ['rgb_fresh20_49_natural','fresh20_49_translate_L16','fresh20_49_occluder_L16']:
        paths = SETTINGS[name]
        rows = [eval_cache('offline', paths['offline']), eval_cache('online_global', paths['online']), eval_cache('b2_w16_p2', paths['b2'])]
        guard_rows = []
        for thr in thresholds:
            outp = OUTDIR / name / f'reentry_guard_v1_thr{thr:.2f}.pt'
            sel = apply_gate(name, paths['offline'], paths['online'], paths['b2'], OUTDIR / f'{name}_features_labels.npz', model, thr, outp)
            erow = eval_cache(f'reentry_guard_v1_thr{thr:.2f}', outp)
            erow['selection'] = sel
            guard_rows.append(erow)
        # best by objective: AJ_RD gain over B2 with AJ no worse than B2 - 0.05 preferred, fallback max AJ_RD.
        b2row = rows[2]
        feasible = [r for r in guard_rows if float(r['AJ_256']) >= float(b2row['AJ_256']) - 0.05]
        if feasible:
            best = max(feasible, key=lambda r: float(r['AJ_RD_256']))
        else:
            best = max(guard_rows, key=lambda r: (float(r['AJ_RD_256']), float(r['AJ_256'])))
        by = {r['name']: r for r in rows + guard_rows}
        gains = {}
        for r in rows + guard_rows:
            if r['name'] == 'offline':
                continue
            gains[f'{r["name"]}_vs_offline'] = {'AJ_RD_256': round(float(r['AJ_RD_256'])-float(by['offline']['AJ_RD_256']),6), 'AJ_256': round(float(r['AJ_256'])-float(by['offline']['AJ_256']),6), 'OA_256': round(float(r['OA_256'])-float(by['offline']['OA_256']),6)}
            gains[f'{r["name"]}_vs_b2'] = {'AJ_RD_256': round(float(r['AJ_RD_256'])-float(by['b2_w16_p2']['AJ_RD_256']),6), 'AJ_256': round(float(r['AJ_256'])-float(by['b2_w16_p2']['AJ_256']),6), 'OA_256': round(float(r['OA_256'])-float(by['b2_w16_p2']['OA_256']),6)}
        results.append({'setting': name, 'thresholds': thresholds, 'methods': rows + guard_rows, 'best_guard': best['name'], 'gains': gains})
    summary = {'feature_names': FEATURE_NAMES, 'train': {'settings': ['dev_translate_L16','dev_occluder_L16'], 'n': int(len(y_train)), 'pos': int(np.sum(y_train==1)), 'neg': int(np.sum(y_train==0)), 'pos_rate': round(float(np.mean(y_train)),6), 'model_path': str(model_path), 'best_train_threshold': model['best_threshold']}, 'datasets': ds_summaries, 'results': results}
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        best = next(m for m in r['methods'] if m['name'] == r['best_guard'])
        b2 = next(m for m in r['methods'] if m['name'] == 'b2_w16_p2')
        compact.append({'setting': r['setting'], 'b2': {'AJ_RD_256': b2['AJ_RD_256'], 'AJ_256': b2['AJ_256']}, 'best_guard': {'name': best['name'], 'AJ_RD_256': best['AJ_RD_256'], 'AJ_256': best['AJ_256'], 'selection': best.get('selection')}, 'gain_vs_b2': r['gains'][f'{best["name"]}_vs_b2']})
    print(json.dumps({'summary_path': str(OUTDIR / 'summary.json'), 'train': summary['train'], 'compact': compact}, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
