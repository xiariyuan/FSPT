#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_b2_false_trigger_taxonomy import point_aj
from scripts.sweep_b2_w16_false_cost_guards_dev import trigger_ok
from utils.coords import find_reentry_events, yx_norm_to_xy_256

OUT = Path('outputs/paper_discovery_2026-06-27/b2_rv_feature_audit_dev')
DATASETS = {
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
        'p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt'),
    },
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
        'p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_persist2.pt'),
    },
}
CFG = {'name': 'b2_w16_p2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 2, 'base_mode': 'any'}
FEATURES = [
    'trigger_t_norm',
    'query_age_norm',
    'base_invis_run',
    'override_persist_len_cap16',
    'override_visible_frac_next16',
    'base_visible_frac_next16',
    'vis_agreement_frac_next16',
    'override_visibility_transitions_next16',
    'base_visibility_transitions_next16',
    'base_override_dist_t',
    'base_override_dist_mean_next4',
    'base_override_dist_mean_next16',
    'base_override_dist_max_next16',
    'override_speed_mean_next4',
    'override_speed_mean_next16',
    'base_speed_prev4',
    'n_trigger_windows',
]


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


def px(points_yx: np.ndarray) -> np.ndarray:
    return yx_norm_to_xy_256(np.asarray(points_yx, dtype=np.float32))


def px_dist(a_yx: np.ndarray, b_yx: np.ndarray) -> float:
    a = px(np.asarray(a_yx, dtype=np.float32)[None, :])[0]
    b = px(np.asarray(b_yx, dtype=np.float32)[None, :])[0]
    return float(np.linalg.norm(a - b))


def speed_mean(track_yx: np.ndarray, start: int, end: int) -> Optional[float]:
    T = len(track_yx)
    lo = max(0, int(start))
    hi = min(T, int(end))
    if hi - lo < 2:
        return None
    p = px(track_yx[lo:hi])
    d = np.linalg.norm(p[1:] - p[:-1], axis=-1)
    return float(np.mean(d)) if d.size else None


def frac_true(v: np.ndarray, start: int, end: int) -> Optional[float]:
    lo = max(0, int(start))
    hi = min(len(v), int(end))
    if hi <= lo:
        return None
    return float(np.mean(v[lo:hi].astype(np.float32)))


def trans_count(v: np.ndarray, start: int, end: int) -> int:
    lo = max(0, int(start))
    hi = min(len(v), int(end))
    if hi - lo < 2:
        return 0
    a = v[lo:hi].astype(bool)
    return int(np.sum(a[1:] != a[:-1]))


def override_persist_len(v: np.ndarray, t: int, cap: int = 16) -> int:
    c = 0
    for j in range(int(t), min(len(v), int(t) + cap)):
        if bool(v[j]):
            c += 1
        else:
            break
    return int(c)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def trigger_times(base_v: np.ndarray, over_v: np.ndarray, query_t: int, cfg: Dict[str, Any]) -> List[int]:
    ts: List[int] = []
    T = len(base_v)
    t = max(1, int(query_t) + 1)
    while t < T:
        if trigger_ok(base_v, over_v, t, cfg):
            ts.append(int(t))
            post = int(cfg.get('post', 16))
            hi = T if post >= 9999 else min(T, t + post + 1)
            t = hi
        else:
            t += 1
    return ts


def base_override_dist_stats(base_tr: np.ndarray, over_tr: np.ndarray, t: int, span: int) -> Dict[str, Optional[float]]:
    vals: List[float] = []
    T = len(base_tr)
    for j in range(int(t), min(T, int(t) + int(span))):
        vals.append(px_dist(base_tr[j], over_tr[j]))
    if not vals:
        return {'mean': None, 'max': None}
    return {'mean': float(np.mean(vals)), 'max': float(np.max(vals))}


def row_for_trigger(dataset: str, vid: str, qi: int, qt: int, t: int, ts: List[int], bvis: np.ndarray, ovis: np.ndarray, pvis: np.ndarray, gv: np.ndarray, btracks: np.ndarray, otracks: np.ndarray, ptracks: np.ndarray, gttracks: np.ndarray) -> Dict[str, Any]:
    events = find_reentry_events(gv, qt)
    has_re = bool(events)
    T = len(gv)
    start = max(0, int(t) - 1)
    base_full = point_aj(btracks, gttracks, bvis, gv, qt, start_t=0)
    p2_full = point_aj(ptracks, gttracks, pvis, gv, qt, start_t=0)
    base_post = point_aj(btracks, gttracks, bvis, gv, qt, start_t=start)
    p2_post = point_aj(ptracks, gttracks, pvis, gv, qt, start_t=start)
    d4 = base_override_dist_stats(btracks, otracks, t, 4)
    d16 = base_override_dist_stats(btracks, otracks, t, 16)
    row: Dict[str, Any] = {
        'dataset': dataset,
        'video_id': vid,
        'query_idx': int(qi),
        'query_t': int(qt),
        'trigger_t': int(t),
        'has_gt_reentry': has_re,
        'n_gt_reentry_events': int(len(events)),
        'first_gt_reentry_t': int(events[0]['reentry_frame']) if events else None,
        'class': 'true_trigger' if has_re else 'false_trigger',
        'delta_full': round(float(p2_full - base_full), 6),
        'delta_post': round(float(p2_post - base_post), 6),
        'harmful_full': bool((p2_full - base_full) < -0.05),
        'severe_full': bool((p2_full - base_full) < -0.10),
        'low_cost_full': bool((p2_full - base_full) >= -0.01),
        'helpful_post': bool((p2_post - base_post) > 0.05),
        'bad_post': bool((p2_post - base_post) < -0.05),
        # runtime-visible features
        'trigger_t_norm': float(t / max(T - 1, 1)),
        'query_age_norm': float((t - qt) / max(T - 1, 1)),
        'base_invis_run': float(invisible_run_before(bvis, t)),
        'override_persist_len_cap16': float(override_persist_len(ovis, t, 16)),
        'override_visible_frac_next16': frac_true(ovis, t, t + 16),
        'base_visible_frac_next16': frac_true(bvis, t, t + 16),
        'vis_agreement_frac_next16': float(np.mean((bvis[t:min(T, t+16)] == ovis[t:min(T, t+16)]).astype(np.float32))) if min(T, t+16) > t else None,
        'override_visibility_transitions_next16': float(trans_count(ovis, t, t + 16)),
        'base_visibility_transitions_next16': float(trans_count(bvis, t, t + 16)),
        'base_override_dist_t': float(px_dist(btracks[t], otracks[t])),
        'base_override_dist_mean_next4': d4['mean'],
        'base_override_dist_mean_next16': d16['mean'],
        'base_override_dist_max_next16': d16['max'],
        'override_speed_mean_next4': speed_mean(otracks, t, t + 4),
        'override_speed_mean_next16': speed_mean(otracks, t, t + 16),
        'base_speed_prev4': speed_mean(btracks, max(qt, t - 4), t + 1),
        'n_trigger_windows': float(len(ts)),
    }
    return row


def collect_dataset(name: str, cfg: Dict[str, Path]) -> List[Dict[str, Any]]:
    base = load(cfg['base'])
    over = load(cfg['override'])
    p2 = load(cfg['p2'])
    rows: List[Dict[str, Any]] = []
    for br, orr, pr in zip(base['records'], over['records'], p2['records']):
        vid = str(br['video_id'])
        bvis = npy(br['pred_visibility'], bool)
        ovis = npy(orr['pred_visibility'], bool)
        pvis = npy(pr['pred_visibility'], bool)
        gv = npy(br['gt_visibility'], bool)
        qpts = npy(br['query_points'], np.float32)
        btracks = npy(br['pred_tracks'], np.float32)
        otracks = npy(orr['pred_tracks'], np.float32)
        ptracks = npy(pr['pred_tracks'], np.float32)
        gttracks = npy(br['gt_tracks'], np.float32)
        n, T = gv.shape
        for qi in range(n):
            qt = max(0, min(T - 1, int(round(float(qpts[qi, 0])))))
            ts = trigger_times(bvis[qi], ovis[qi], qt, CFG)
            if not ts:
                continue
            t = int(ts[0])
            rows.append(row_for_trigger(name, vid, qi, qt, t, ts, bvis[qi], ovis[qi], pvis[qi], gv[qi], btracks[qi], otracks[qi], ptracks[qi], gttracks[qi]))
    return rows


def summarize_vals(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    arr = np.asarray(vals, dtype=np.float32)
    return {'n': int(arr.size), 'mean': round(float(np.mean(arr)), 6), 'median': round(float(np.median(arr)), 6), 'p10': round(float(np.percentile(arr, 10)), 6), 'p90': round(float(np.percentile(arr, 90)), 6)}


def ranks_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auc_score(scores: List[float], labels: List[bool]) -> Optional[float]:
    pairs = [(float(s), bool(y)) for s, y in zip(scores, labels) if s is not None and math.isfinite(float(s))]
    if not pairs:
        return None
    s = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=bool)
    npos = int(np.sum(y))
    nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return None
    r = ranks_average(s)
    sum_pos = float(np.sum(r[y]))
    auc = (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc)


def feature_auc_table(rows: List[Dict[str, Any]], label_key: str) -> List[Dict[str, Any]]:
    labels = [bool(r[label_key]) for r in rows]
    out: List[Dict[str, Any]] = []
    for f in FEATURES:
        scores = [r.get(f) for r in rows]
        auc = auc_score(scores, labels)
        if auc is None:
            continue
        out.append({
            'feature': f,
            'auc_positive_high': round(float(auc), 6),
            'best_auc': round(float(max(auc, 1.0 - auc)), 6),
            'direction': 'high' if auc >= 0.5 else 'low',
        })
    out.sort(key=lambda x: x['best_auc'], reverse=True)
    return out


def group_feature_stats(rows: List[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    pos = [r for r in rows if bool(r[label_key])]
    neg = [r for r in rows if not bool(r[label_key])]
    return {
        'label': label_key,
        'positive_n': len(pos),
        'negative_n': len(neg),
        'features': {
            f: {
                'positive': summarize_vals([r.get(f) for r in pos]),
                'negative': summarize_vals([r.get(f) for r in neg]),
            }
            for f in FEATURES
        }
    }


def summarize_rows(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    true_rows = [r for r in rows if r['class'] == 'true_trigger']
    false_rows = [r for r in rows if r['class'] == 'false_trigger']
    return {
        'name': name,
        'n_triggered_tracks': len(rows),
        'true_trigger_tracks': len(true_rows),
        'false_trigger_tracks': len(false_rows),
        'true_trigger_rate': round(len(true_rows) / max(len(rows), 1), 6),
        'delta_full_all': summarize_vals([r['delta_full'] for r in rows]),
        'delta_post_all': summarize_vals([r['delta_post'] for r in rows]),
        'delta_full_true': summarize_vals([r['delta_full'] for r in true_rows]),
        'delta_post_true': summarize_vals([r['delta_post'] for r in true_rows]),
        'delta_full_false': summarize_vals([r['delta_full'] for r in false_rows]),
        'delta_post_false': summarize_vals([r['delta_post'] for r in false_rows]),
        'harmful_full_count': int(sum(1 for r in rows if r['harmful_full'])),
        'harmful_full_rate': round(sum(1 for r in rows if r['harmful_full']) / max(len(rows), 1), 6),
        'helpful_post_count': int(sum(1 for r in rows if r['helpful_post'])),
        'helpful_post_rate': round(sum(1 for r in rows if r['helpful_post']) / max(len(rows), 1), 6),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for name, cfg in DATASETS.items():
        rows = collect_dataset(name, cfg)
        by_dataset[name] = rows
        all_rows.extend(rows)

    summary = {
        'protocol': 'Development-only feature separability audit for future B2-RV. Uses DAVIS + RGB dev first-10 only. Does not use RGB fresh20-49.',
        'config': CFG,
        'datasets': {k: {kk: str(vv) for kk, vv in cfg.items()} for k, cfg in DATASETS.items()},
        'dataset_summaries': {k: summarize_rows(v, k) for k, v in by_dataset.items()},
        'combined_summary': summarize_rows(all_rows, 'combined_dev'),
        'feature_auc': {
            'harmful_full': feature_auc_table(all_rows, 'harmful_full'),
            'helpful_post': feature_auc_table(all_rows, 'helpful_post'),
            'true_trigger': feature_auc_table(all_rows, 'has_gt_reentry'),
            'bad_post': feature_auc_table(all_rows, 'bad_post'),
        },
        'feature_stats': {
            'harmful_full': group_feature_stats(all_rows, 'harmful_full'),
            'helpful_post': group_feature_stats(all_rows, 'helpful_post'),
            'true_trigger': group_feature_stats(all_rows, 'has_gt_reentry'),
            'bad_post': group_feature_stats(all_rows, 'bad_post'),
        },
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'trigger_rows.jsonl').open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for name, rows in by_dataset.items():
        with (OUT / f'{name}_trigger_rows.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({
        'protocol': summary['protocol'],
        'dataset_summaries': summary['dataset_summaries'],
        'combined_summary': summary['combined_summary'],
        'top_auc_harmful_full': summary['feature_auc']['harmful_full'][:10],
        'top_auc_helpful_post': summary['feature_auc']['helpful_post'][:10],
        'top_auc_true_trigger': summary['feature_auc']['true_trigger'][:10],
        'top_auc_bad_post': summary['feature_auc']['bad_post'][:10],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
