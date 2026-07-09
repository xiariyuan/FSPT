#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_rgb_heldout_b2w16 import standard_metrics, ajrd_metrics, delta
from scripts.audit_b2_occ_length_bins import event_rows_for_method, summarize as summarize_occ, BINS
from scripts.audit_b2_false_trigger_taxonomy import point_aj, summarize_vals
from scripts.sweep_b2_w16_false_cost_guards_dev import trigger_ok
from utils.coords import find_reentry_events, yx_norm_to_xy_256

OUT = Path('outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit')
METHODS = {
    'fixed_offline': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
    'global_b1_vis4': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
    'b2_fullpost': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt'),
    'b2_w16': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_base.pt'),
    'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt'),
    'b2_gt_window_oracle': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b2_localized_oracle/fixed_offline__override_vis4_gated288__pre0_post32.pt'),
}
BASE_CACHE = METHODS['fixed_offline']
OVERRIDE_CACHE = METHODS['global_b1_vis4']
CFG_P2 = {'name': 'b2_w16_p2', 'k': 1, 'pre': 1, 'post': 16, 'over_persist': 2, 'base_mode': 'any'}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


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


def px_dist(a_yx: np.ndarray, b_yx: np.ndarray) -> float:
    a = yx_norm_to_xy_256(np.asarray(a_yx, dtype=np.float32)[None, :])[0]
    b = yx_norm_to_xy_256(np.asarray(b_yx, dtype=np.float32)[None, :])[0]
    return float(np.linalg.norm(a - b))


def safe_delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 6)


def aggregate_per_video(method_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    n_videos = len(next(iter(method_payloads.values()))['records'])
    for i in range(n_videos):
        vid = str(method_payloads['fixed_offline']['records'][i]['video_id'])
        row: Dict[str, Any] = {'video_id': vid, 'idx': i}
        for name, payload in method_payloads.items():
            rec = payload['records'][i]
            mm: Dict[str, Any] = {}
            mm.update(standard_metrics(rec))
            mm.update(ajrd_metrics(rec))
            row[name] = mm
        row['deltas'] = {
            'p2_vs_fixed_AJ_RD_256': safe_delta(row['b2_w16_p2'].get('AJ_RD_256'), row['fixed_offline'].get('AJ_RD_256')),
            'p2_vs_fixed_AJ_256': safe_delta(row['b2_w16_p2'].get('AJ_256'), row['fixed_offline'].get('AJ_256')),
            'p2_vs_w16_AJ_RD_256': safe_delta(row['b2_w16_p2'].get('AJ_RD_256'), row['b2_w16'].get('AJ_RD_256')),
            'p2_vs_w16_AJ_256': safe_delta(row['b2_w16_p2'].get('AJ_256'), row['b2_w16'].get('AJ_256')),
            'p2_vs_full_AJ_RD_256': safe_delta(row['b2_w16_p2'].get('AJ_RD_256'), row['b2_fullpost'].get('AJ_RD_256')),
            'p2_vs_full_AJ_256': safe_delta(row['b2_w16_p2'].get('AJ_256'), row['b2_fullpost'].get('AJ_256')),
            'p2_vs_b1_AJ_RD_256': safe_delta(row['b2_w16_p2'].get('AJ_RD_256'), row['global_b1_vis4'].get('AJ_RD_256')),
            'p2_vs_b1_AJ_256': safe_delta(row['b2_w16_p2'].get('AJ_256'), row['global_b1_vis4'].get('AJ_256')),
            'w16_vs_fixed_AJ_RD_256': safe_delta(row['b2_w16'].get('AJ_RD_256'), row['fixed_offline'].get('AJ_RD_256')),
            'w16_vs_fixed_AJ_256': safe_delta(row['b2_w16'].get('AJ_256'), row['fixed_offline'].get('AJ_256')),
        }
        rows.append(row)

    def count(pred):
        return int(sum(1 for r in rows if pred(r)))

    def mean_for(method: str, key: str):
        vals = [r[method].get(key) for r in rows if r[method].get(key) is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    summary = {
        'video_weighted': {
            m: {
                'video_mean_AJ_RD_256': mean_for(m, 'AJ_RD_256'),
                'video_mean_AJ_256': mean_for(m, 'AJ_256'),
                'video_mean_OA_256': mean_for(m, 'OA_256'),
                'video_mean_delta_avg_256': mean_for(m, 'delta_avg_256'),
            }
            for m in method_payloads
        },
        'counts': {
            'n_videos': len(rows),
            'p2_improves_AJ_RD_vs_fixed': count(lambda r: (r['deltas']['p2_vs_fixed_AJ_RD_256'] or -999) > 0),
            'p2_improves_AJ_RD_vs_fixed_ge_0p01': count(lambda r: (r['deltas']['p2_vs_fixed_AJ_RD_256'] or -999) >= 0.01),
            'p2_AJ_drop_vs_fixed_le_1p5': count(lambda r: r['deltas']['p2_vs_fixed_AJ_256'] is not None and r['deltas']['p2_vs_fixed_AJ_256'] >= -1.5),
            'p2_AJ_drop_vs_fixed_gt_3': count(lambda r: r['deltas']['p2_vs_fixed_AJ_256'] is not None and r['deltas']['p2_vs_fixed_AJ_256'] < -3.0),
            'p2_AJ_RD_within_0p005_of_fullpost': count(lambda r: r['deltas']['p2_vs_full_AJ_RD_256'] is not None and r['deltas']['p2_vs_full_AJ_RD_256'] >= -0.005),
            'p2_AJ_RD_loses_to_fullpost_by_gt_0p01': count(lambda r: r['deltas']['p2_vs_full_AJ_RD_256'] is not None and r['deltas']['p2_vs_full_AJ_RD_256'] < -0.01),
            'p2_AJ_RD_within_0p005_of_b1': count(lambda r: r['deltas']['p2_vs_b1_AJ_RD_256'] is not None and r['deltas']['p2_vs_b1_AJ_RD_256'] >= -0.005),
            'p2_AJ_beats_b1_by_10': count(lambda r: r['deltas']['p2_vs_b1_AJ_256'] is not None and r['deltas']['p2_vs_b1_AJ_256'] >= 10.0),
            'p2_improves_AJRD_vs_w16': count(lambda r: (r['deltas']['p2_vs_w16_AJ_RD_256'] or -999) > 0),
            'p2_improves_AJ_vs_w16': count(lambda r: (r['deltas']['p2_vs_w16_AJ_256'] or -999) > 0),
        },
        'worst_p2_AJ_drop_vs_fixed': sorted(rows, key=lambda r: r['deltas']['p2_vs_fixed_AJ_256'] if r['deltas']['p2_vs_fixed_AJ_256'] is not None else 999)[:10],
        'best_p2_AJRD_gain_vs_fixed': sorted(rows, key=lambda r: r['deltas']['p2_vs_fixed_AJ_RD_256'] if r['deltas']['p2_vs_fixed_AJ_RD_256'] is not None else -999, reverse=True)[:10],
        'rows': rows,
    }
    return summary


def aggregate_occ(method_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for name, payload in method_payloads.items():
        rows = event_rows_for_method(payload)
        rows_by_method[name] = rows
        summaries[name] = summarize_occ(rows)
    comparison: Dict[str, Dict[str, Any]] = {}
    for bname, _, _ in BINS:
        comparison[bname] = {}
        for method in method_payloads:
            comparison[bname][method] = summaries[method][bname]['mean_AJ_segment_256']
        comparison[bname]['p2_vs_fixed'] = safe_delta(comparison[bname]['b2_w16_p2'], comparison[bname]['fixed_offline'])
        comparison[bname]['p2_vs_w16'] = safe_delta(comparison[bname]['b2_w16_p2'], comparison[bname]['b2_w16'])
        comparison[bname]['p2_vs_full'] = safe_delta(comparison[bname]['b2_w16_p2'], comparison[bname]['b2_fullpost'])
        comparison[bname]['p2_vs_b1'] = safe_delta(comparison[bname]['b2_w16_p2'], comparison[bname]['global_b1_vis4'])
        comparison[bname]['n_events'] = summaries['fixed_offline'][bname]['n_events']
    return {'summaries': summaries, 'comparison': comparison, 'rows_by_method': rows_by_method}


def aggregate_trigger_taxonomy(base: Dict[str, Any], over: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    base_records = base['records']
    over_records = over['records']
    p2_records = p2['records']
    rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    true_rows: List[Dict[str, Any]] = []
    counts = {
        'total_tracks': 0,
        'gt_reentry_tracks': 0,
        'triggered_tracks': 0,
        'true_trigger_tracks': 0,
        'false_trigger_tracks': 0,
        'no_trigger_reentry_tracks': 0,
        'nontrigger_nonreentry_tracks': 0,
    }
    video: Dict[str, Dict[str, Any]] = {}

    for br, orr, pr in zip(base_records, over_records, p2_records):
        vid = str(br['video_id'])
        vs = video.setdefault(vid, {
            'video_id': vid,
            'total_tracks': 0,
            'gt_reentry_tracks': 0,
            'triggered_tracks': 0,
            'true_trigger_tracks': 0,
            'false_trigger_tracks': 0,
            'no_trigger_reentry_tracks': 0,
            'false_gt_visible_at_trigger': 0,
            'false_gt_invisible_at_trigger': 0,
            'false_harmful_full_lt_minus_0p05': 0,
            'false_severe_full_lt_minus_0p10': 0,
            'false_low_cost_full_ge_minus_0p01': 0,
            'false_delta_full': [],
            'false_delta_post': [],
            'true_delta_full': [],
            'true_delta_post': [],
            'false_base_override_dist': [],
        })
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
            counts['total_tracks'] += 1
            vs['total_tracks'] += 1
            qt = max(0, min(T - 1, int(round(float(qpts[qi, 0])))))
            events = find_reentry_events(gv[qi], qt)
            has_re = bool(events)
            if has_re:
                counts['gt_reentry_tracks'] += 1
                vs['gt_reentry_tracks'] += 1
            ts = trigger_times(bvis[qi], ovis[qi], qt, CFG_P2)
            trig = ts[0] if ts else None
            triggered = trig is not None
            if triggered:
                counts['triggered_tracks'] += 1
                vs['triggered_tracks'] += 1
            if triggered and has_re:
                counts['true_trigger_tracks'] += 1
                vs['true_trigger_tracks'] += 1
                cls = 'true_trigger'
            elif triggered and not has_re:
                counts['false_trigger_tracks'] += 1
                vs['false_trigger_tracks'] += 1
                cls = 'false_trigger'
            elif (not triggered) and has_re:
                counts['no_trigger_reentry_tracks'] += 1
                vs['no_trigger_reentry_tracks'] += 1
                cls = 'no_trigger_reentry'
            else:
                counts['nontrigger_nonreentry_tracks'] += 1
                cls = 'none'

            base_aj = point_aj(btracks[qi], gttracks[qi], bvis[qi], gv[qi], qt, start_t=0)
            p2_aj = point_aj(ptracks[qi], gttracks[qi], pvis[qi], gv[qi], qt, start_t=0)
            delta_full = float(p2_aj - base_aj)
            if trig is not None:
                start = max(0, int(trig) - 1)
                base_post = point_aj(btracks[qi], gttracks[qi], bvis[qi], gv[qi], qt, start_t=start)
                p2_post = point_aj(ptracks[qi], gttracks[qi], pvis[qi], gv[qi], qt, start_t=start)
                delta_post = float(p2_post - base_post)
                dist_bo = px_dist(btracks[qi, trig], otracks[qi, trig])
                gt_visible_at_trigger = bool(gv[qi, trig])
            else:
                base_post = p2_post = delta_post = None
                dist_bo = None
                gt_visible_at_trigger = None

            row = {
                'video_id': vid,
                'query_idx': int(qi),
                'query_t': int(qt),
                'has_gt_reentry': has_re,
                'n_gt_reentry_events': int(len(events)),
                'first_gt_reentry_t': int(events[0]['reentry_frame']) if events else None,
                'triggered': triggered,
                'n_trigger_windows': len(ts),
                'first_trigger_t': trig,
                'class': cls,
                'gt_visible_at_trigger': gt_visible_at_trigger,
                'base_visible_at_trigger': bool(bvis[qi, trig]) if trig is not None else None,
                'override_visible_at_trigger': bool(ovis[qi, trig]) if trig is not None else None,
                'override_persist2_at_trigger': bool(np.all(ovis[qi, trig:trig+2])) if trig is not None and trig+2 <= T else None,
                'base_invisible_run_before_trigger': invisible_run_before(bvis[qi], trig) if trig is not None else None,
                'base_override_dist_256_at_trigger': round(dist_bo, 4) if dist_bo is not None else None,
                'fixed_query_AJ_256': round(base_aj, 6),
                'p2_query_AJ_256': round(p2_aj, 6),
                'delta_p2_minus_fixed_full_AJ': round(delta_full, 6),
                'fixed_post_AJ_256': round(base_post, 6) if base_post is not None else None,
                'p2_post_AJ_256': round(p2_post, 6) if p2_post is not None else None,
                'delta_p2_minus_fixed_post_AJ': round(delta_post, 6) if delta_post is not None else None,
            }
            rows.append(row)
            if cls == 'false_trigger':
                false_rows.append(row)
                if gt_visible_at_trigger:
                    vs['false_gt_visible_at_trigger'] += 1
                else:
                    vs['false_gt_invisible_at_trigger'] += 1
                vs['false_delta_full'].append(delta_full)
                if delta_post is not None:
                    vs['false_delta_post'].append(delta_post)
                if dist_bo is not None:
                    vs['false_base_override_dist'].append(dist_bo)
                if delta_full >= -0.01:
                    vs['false_low_cost_full_ge_minus_0p01'] += 1
                if delta_full < -0.05:
                    vs['false_harmful_full_lt_minus_0p05'] += 1
                if delta_full < -0.10:
                    vs['false_severe_full_lt_minus_0p10'] += 1
            elif cls == 'true_trigger':
                true_rows.append(row)
                vs['true_delta_full'].append(delta_full)
                if delta_post is not None:
                    vs['true_delta_post'].append(delta_post)

    false_delta_full = [float(r['delta_p2_minus_fixed_full_AJ']) for r in false_rows]
    false_delta_post = [float(r['delta_p2_minus_fixed_post_AJ']) for r in false_rows if r['delta_p2_minus_fixed_post_AJ'] is not None]
    true_delta_full = [float(r['delta_p2_minus_fixed_full_AJ']) for r in true_rows]
    true_delta_post = [float(r['delta_p2_minus_fixed_post_AJ']) for r in true_rows if r['delta_p2_minus_fixed_post_AJ'] is not None]

    false_gt_visible = sum(1 for r in false_rows if r['gt_visible_at_trigger'])
    false_gt_invisible = sum(1 for r in false_rows if r['gt_visible_at_trigger'] is False)
    false_low_cost = sum(1 for r in false_rows if float(r['delta_p2_minus_fixed_full_AJ']) >= -0.01)
    false_harmful = sum(1 for r in false_rows if float(r['delta_p2_minus_fixed_full_AJ']) < -0.05)
    false_severe = sum(1 for r in false_rows if float(r['delta_p2_minus_fixed_full_AJ']) < -0.10)

    video_rows = []
    for vs in video.values():
        fd = vs.pop('false_delta_full')
        fdp = vs.pop('false_delta_post')
        td = vs.pop('true_delta_full')
        tdp = vs.pop('true_delta_post')
        bod = vs.pop('false_base_override_dist')
        vs['false_delta_full_stats'] = summarize_vals(fd)
        vs['false_delta_post_stats'] = summarize_vals(fdp)
        vs['true_delta_full_stats'] = summarize_vals(td)
        vs['true_delta_post_stats'] = summarize_vals(tdp)
        vs['false_base_override_dist_stats'] = summarize_vals(bod)
        vs['trigger_precision_track'] = round(vs['true_trigger_tracks'] / max(vs['triggered_tracks'], 1), 6)
        vs['trigger_recall_track'] = round(vs['true_trigger_tracks'] / max(vs['gt_reentry_tracks'], 1), 6)
        vs['false_harmful_rate_among_false'] = round(vs['false_harmful_full_lt_minus_0p05'] / max(vs['false_trigger_tracks'], 1), 6)
        video_rows.append(vs)

    summary = {
        'config': CFG_P2,
        'counts': counts,
        'trigger_precision_track': round(counts['true_trigger_tracks'] / max(counts['triggered_tracks'], 1), 6),
        'trigger_recall_track': round(counts['true_trigger_tracks'] / max(counts['gt_reentry_tracks'], 1), 6),
        'false_trigger_summary': {
            'false_triggers': len(false_rows),
            'gt_visible_at_trigger': int(false_gt_visible),
            'gt_invisible_at_trigger': int(false_gt_invisible),
            'low_cost_full_delta_ge_-0.01': int(false_low_cost),
            'low_cost_rate': round(false_low_cost / max(len(false_rows), 1), 6),
            'harmful_full_delta_lt_-0.05': int(false_harmful),
            'harmful_rate': round(false_harmful / max(len(false_rows), 1), 6),
            'severe_full_delta_lt_-0.10': int(false_severe),
            'severe_rate': round(false_severe / max(len(false_rows), 1), 6),
            'delta_full_stats': summarize_vals(false_delta_full),
            'delta_post_stats': summarize_vals(false_delta_post),
        },
        'true_trigger_summary': {
            'true_triggers': len(true_rows),
            'delta_full_stats': summarize_vals(true_delta_full),
            'delta_post_stats': summarize_vals(true_delta_post),
        },
        'video_rows': sorted(video_rows, key=lambda x: (x['false_harmful_full_lt_minus_0p05'], x['false_trigger_tracks']), reverse=True),
        'worst_false_trigger_rows': sorted(false_rows, key=lambda r: float(r['delta_p2_minus_fixed_full_AJ']))[:50],
        'best_true_trigger_rows': sorted(true_rows, key=lambda r: float(r['delta_p2_minus_fixed_full_AJ']), reverse=True)[:50],
    }
    return {'summary': summary, 'rows': rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {name: load(path) for name, path in METHODS.items()}

    per_video = aggregate_per_video(payloads)
    occ = aggregate_occ(payloads)
    trig = aggregate_trigger_taxonomy(payloads['fixed_offline'], payloads['global_b1_vis4'], payloads['b2_w16_p2'])

    # write outputs
    (OUT / 'per_video_summary.json').write_text(json.dumps({k: v for k, v in per_video.items() if k != 'rows'}, indent=2, ensure_ascii=False))
    with (OUT / 'per_video_rows.jsonl').open('w') as f:
        for r in per_video['rows']:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    (OUT / 'occ_length_summary.json').write_text(json.dumps({k: v for k, v in occ.items() if k != 'rows_by_method'}, indent=2, ensure_ascii=False))
    for method, rows in occ['rows_by_method'].items():
        with (OUT / f'occ_{method}_event_rows.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    (OUT / 'trigger_taxonomy_summary.json').write_text(json.dumps(trig['summary'], indent=2, ensure_ascii=False))
    with (OUT / 'trigger_taxonomy_rows.jsonl').open('w') as f:
        for r in trig['rows']:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'methods': {k: str(v) for k, v in METHODS.items()},
        'p2_config': CFG_P2,
        'per_video': {k: v for k, v in per_video.items() if k != 'rows'},
        'occ_length': {k: v for k, v in occ.items() if k != 'rows_by_method'},
        'trigger_taxonomy': trig['summary'],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({
        'per_video_counts': per_video['counts'],
        'video_weighted': per_video['video_weighted'],
        'occ_comparison': occ['comparison'],
        'trigger_counts': trig['summary']['counts'],
        'trigger_precision_track': trig['summary']['trigger_precision_track'],
        'trigger_recall_track': trig['summary']['trigger_recall_track'],
        'false_trigger_summary': trig['summary']['false_trigger_summary'],
        'true_trigger_summary': trig['summary']['true_trigger_summary'],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
