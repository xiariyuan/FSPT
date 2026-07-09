#!/usr/bin/env python3
from __future__ import annotations
import json, sys, math, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'baselines/cotracker'))

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from scripts.export_cotracker3_online_v7a4_raw_visconf_components import load_video_for_record, make_record_from_state
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, CANDIDATE, align_candidate, build_events, clone_records, standard_and_ajrd, apply_events, err_px, npy
from scripts.eval_cotracker3_online_v8c01_recovery_risk_audit import rebuild_events_from_native

OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v8c21_full_hook_eval'
CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'


def build_touch(events, native_by, cand_by, W=8, dist_thr=64.0):
    by_vid = defaultdict(dict)
    stats = Counter()
    for e in events:
        nr = native_by[e.video_id]
        cr = cand_by[e.video_id]
        ntr = npy(nr['pred_tracks'], np.float32)
        ctr = npy(cr['pred_tracks'], np.float32)
        cvis = npy(cr['pred_visibility'], bool)
        q = int(e.query_idx)
        t0 = int(e.t)
        if err_px(ntr[q, t0], ctr[q, t0]) > dist_thr:
            continue
        stats['selected_events'] += 1
        hit = False
        for tau in range(t0, min(ctr.shape[1], t0 + W + 1)):
            if not bool(cvis[q, tau]):
                continue
            by_vid[e.video_id][(q, tau)] = ctr[q, tau].copy()
            hit = True
        if hit:
            stats['events_with_touch'] += 1
    stats['unique_touched_frames_total'] = sum(len(v) for v in by_vid.values())
    return by_vid, dict(stats)


def apply_hook(model, touch_map, prob):
    if model.model.online_coords_predicted is None:
        return 0
    coords_state = model.model.online_coords_predicted
    vis_state = model.model.online_vis_predicted
    conf_state = model.model.online_conf_predicted
    _, T, N, _ = coords_state.shape
    H, W = model.interp_shape
    p = float(prob)
    logit = math.log(p / max(1e-6, 1.0 - p))
    c = 0
    for (q, tau), yx in touch_map.items():
        if tau >= T or q >= N:
            continue
        y = float(yx[0]); x = float(yx[1])
        coords_state[0, tau, q, 0] = x * (W - 1)
        coords_state[0, tau, q, 1] = y * (H - 1)
        vis_state[0, tau, q] = logit
        conf_state[0, tau, q] = logit
        c += 1
    return c


def run_video(base_record, ds, touch_map, device, prob):
    model = CoTrackerOnlinePredictor(checkpoint=str(CKPT)).to(device).eval()
    video, queries = load_video_for_record(base_record, ds, device)
    _, T, _, _, _ = video.shape
    model(video_chunk=video, is_first_step=True, queries=queries, add_support_grid=False, grid_size=0)
    calls = 0
    applied_sum = 0
    t0 = time.time()
    auto = device.startswith('cuda')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=auto):
        for ind in range(0, T - model.step, model.step):
            chunk = video[:, ind: ind + model.step * 2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
            calls += 1
            applied_sum += apply_hook(model, touch_map, prob)
    rec = make_record_from_state(base_record, model, f'v8c21_prob_{prob:.2f}')
    return rec, {'video_id': str(base_record['video_id']), 'calls': calls, 'applied_sum': applied_sum, 'touches_available': len(touch_map), 'sec': round(time.time() - t0, 3)}


def dlt(metric, native):
    return {k: (metric[k] - native[k] if metric.get(k) is not None and native.get(k) is not None else None) for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']}


def pv_delta(native_pv, other_pv):
    nb = {r['video_id']: r for r in native_pv}
    out = []
    for r in other_pv:
        b = nb.get(r['video_id'])
        if not b:
            continue
        out.append({
            'video_id': r['video_id'],
            'delta_AJ': None if r.get('AJ') is None or b.get('AJ') is None else r['AJ'] - b['AJ'],
            'delta_OA': None if r.get('OA') is None or b.get('OA') is None else r['OA'] - b['OA'],
            'delta_AJ_RD': None if r.get('AJ_RD') is None or b.get('AJ_RD') is None else r['AJ_RD'] - b['AJ_RD'],
            'delta_AJ_RD_256': None if r.get('AJ_RD_256') is None or b.get('AJ_RD_256') is None else r['AJ_RD_256'] - b['AJ_RD_256'],
            'native_AJ_RD_256': b.get('AJ_RD_256'),
            'metric_AJ_RD_256': r.get('AJ_RD_256'),
        })
    return out


def summarize_state_minus_output(output_rows, state_rows):
    ob = {r['video_id']: r for r in output_rows}
    diffs = []
    for r in state_rows:
        o = ob.get(r['video_id'])
        if not o:
            continue
        a = r.get('delta_AJ_RD_256')
        b = o.get('delta_AJ_RD_256')
        if a is None or b is None:
            continue
        diffs.append(a - b)
    return {
        'n': len(diffs),
        'mean': float(np.mean(diffs)) if diffs else None,
        'median': float(np.median(diffs)) if diffs else None,
        'positive': int(sum(x > 1e-9 for x in diffs)),
        'negative': int(sum(x < -1e-9 for x in diffs)),
        'zero': int(sum(abs(x) <= 1e-9 for x in diffs)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(info)
    n_by = {str(r['video_id']): r for r in native['records']}
    metas = rebuild_events_from_native(native)
    events = build_events(native, cand_by, metas)
    touches, touch_stats = build_touch(events, n_by, cand_by, W=8, dist_thr=64.0)
    base_recs = list(native['records'])
    native_metric, native_pv = standard_and_ajrd(clone_records(base_recs))
    output_recs, output_stats = apply_events(clone_records(base_recs), cand_by, events, np.ones(len(events), dtype=bool), window=8, apply_mode='candidate_visible')
    output_metric, output_pv = standard_and_ajrd(output_recs)
    output_pv_delta = pv_delta(native_pv, output_pv)
    ds = TapVidDataset(str(DAVIS), dataset_type='davis', resize_to=[256, 256], queried_first=True)
    state_rows = []
    for prob in [0.80, 0.90]:
        recs = []
        runs = []
        for rec in base_recs:
            vid = str(rec['video_id'])
            out_rec, run = run_video(rec, ds, touches.get(vid, {}), device, prob)
            recs.append(out_rec)
            runs.append(run)
        metric, pv = standard_and_ajrd(recs)
        pvd = pv_delta(native_pv, pv)
        state_rows.append({
            'prob': prob,
            'metric': metric,
            'delta_vs_native': dlt(metric, native_metric),
            'per_video_delta': pvd,
            'state_minus_output_summary': summarize_state_minus_output(output_pv_delta, pvd),
            'runs': runs,
        })
    report = {
        'script': 'scripts/v8c21_full.py',
        'device': device,
        'alignment': info,
        'touch_stats': touch_stats,
        'native_metric': native_metric,
        'output_level': {
            'metric': output_metric,
            'delta_vs_native': dlt(output_metric, native_metric),
            'stats': output_stats,
            'per_video_delta': output_pv_delta,
        },
        'state_rows': state_rows,
    }
    out = OUT / 'v8c21_full_report.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    def short_delta(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    print(json.dumps({
        'out': str(out),
        'device': device,
        'touch_stats': touch_stats,
        'output_delta': short_delta(report['output_level']['delta_vs_native']),
        'state_rows': [
            {'prob': r['prob'], 'delta': short_delta(r['delta_vs_native']), 'state_minus_output': r['state_minus_output_summary']}
            for r in state_rows
        ],
    }, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
