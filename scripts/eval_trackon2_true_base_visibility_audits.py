#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
REPO = ROOT / 'external/tapnextpp/repo'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics

PIX = 255.0
DEFAULT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_true_base_visibility_audits.json'


def arr(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None and r.get(key) == r.get(key)]
    return float(np.mean(vals)) if vals else None


def eval_record(r: dict, pred_vis: np.ndarray) -> dict:
    pred_yx = arr(r['pred_tracks'], np.float32)
    gt_yx = arr(r['gt_tracks'], np.float32)
    q = arr(r['query_points'], np.float32)
    gt_vis = arr(r['gt_visibility'], bool)

    pred_xy = pred_yx[..., ::-1] * PIX
    gt_xy = gt_yx[..., ::-1] * PIX
    qpx = q.copy()
    qpx[:, 1:] *= PIX
    gt_occ = ~gt_vis
    pred_occ = ~pred_vis

    tap = compute_tapvid_metrics_official(
        query_points=qpx[None].astype(np.float32),
        gt_occluded=gt_occ[None].astype(bool),
        gt_tracks=gt_xy[None].astype(np.float32),
        pred_occluded=pred_occ[None].astype(bool),
        pred_tracks=pred_xy[None].astype(np.float32),
        query_mode='first', thresholds=(1, 2, 4, 8, 16),
    )
    tap = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in tap.items()}

    aj = compute_redetection_metrics(
        pred_tracks=torch.from_numpy(pred_xy.copy()).float().unsqueeze(0).permute(0, 2, 1, 3),
        pred_visible=torch.from_numpy(pred_vis.copy()).bool().unsqueeze(0).permute(0, 2, 1),
        gt_tracks=torch.from_numpy(gt_xy.copy()).float().unsqueeze(0).permute(0, 2, 1, 3),
        gt_visible=torch.from_numpy(gt_vis.copy()).bool().unsqueeze(0).permute(0, 2, 1),
    )
    aj = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in aj.items() if not k.startswith('raw_stats/')}

    pred_true = pred_vis.sum()
    gt_true = gt_vis.sum()
    gt_false = (~gt_vis).sum()
    false_visible = (pred_vis & ~gt_vis).sum()
    false_invisible = (~pred_vis & gt_vis).sum()
    true_visible = (pred_vis & gt_vis).sum()
    return {
        'AJ': tap['average_jaccard'] * 100.0,
        'OA': tap['occlusion_accuracy'] * 100.0,
        'delta_avg': tap['average_pts_within_thresh'] * 100.0,
        'd1': tap['pts_within_1'] * 100.0,
        'd2': tap['pts_within_2'] * 100.0,
        'd4': tap['pts_within_4'] * 100.0,
        'd8': tap['pts_within_8'] * 100.0,
        'd16': tap['pts_within_16'] * 100.0,
        'J1': tap['jaccard_1'] * 100.0,
        'J2': tap['jaccard_2'] * 100.0,
        'J4': tap['jaccard_4'] * 100.0,
        'J8': tap['jaccard_8'] * 100.0,
        'J16': tap['jaccard_16'] * 100.0,
        'AJ_RD': aj.get('AJ_RD'),
        'AJ_RD_D4': aj.get('AJ_RD_D4_dmin1'),
        'AJ_RD_D16': aj.get('AJ_RD_D16_dmin1'),
        'pred_visible_rate': float(pred_vis.mean()),
        'gt_visible_rate': float(gt_vis.mean()),
        'false_visible_rate_all': float(false_visible / max(pred_vis.size, 1)),
        'false_visible_rate_gt_occ': float(false_visible / max(gt_false, 1)),
        'false_invisible_rate_gt_vis': float(false_invisible / max(gt_true, 1)),
        'visible_precision': float(true_visible / max(pred_true, 1)),
        'visible_recall': float(true_visible / max(gt_true, 1)),
        'n_points': int(q.shape[0]),
        'n_frames': int(pred_vis.shape[1]),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [
        'AJ','OA','delta_avg','d1','d2','d4','d8','d16','J1','J2','J4','J8','J16',
        'AJ_RD','AJ_RD_D4','AJ_RD_D16','pred_visible_rate','gt_visible_rate',
        'false_visible_rate_all','false_visible_rate_gt_occ','false_invisible_rate_gt_vis',
        'visible_precision','visible_recall',
    ]
    out = {k: mean(rows, k) for k in keys}
    out['n_videos'] = len(rows)
    out['n_queries'] = int(sum(r['n_points'] for r in rows))
    return out


def eval_cache(records: list[dict], mode: str, tau: float | None = None, local_params: dict | None = None) -> tuple[dict, dict]:
    rows = []
    stats_total = {'candidate_frames': 0, 'recovered_frames': 0, 'rejected_jump': 0, 'rejected_short': 0, 'opened_false_visible': 0, 'opened_true_visible': 0}
    for r in records:
        if mode == 'native':
            pred_vis = arr(r['pred_visibility'], bool)
            st = None
        elif mode == 'global_threshold':
            conf = arr(r['pred_vis_conf'], np.float32)
            pred_vis = conf >= float(tau)
            st = None
        elif mode == 'local_recovery':
            pred_yx = arr(r['pred_tracks'], np.float32)
            conf = arr(r['pred_vis_conf'], np.float32)
            base_vis = arr(r['pred_visibility'], bool)
            gt_vis = arr(r['gt_visibility'], bool)
            pred_vis, st = recover_local(pred_yx, conf, base_vis, gt_vis=gt_vis, **local_params)
            for k in stats_total:
                stats_total[k] += st.get(k, 0)
        else:
            raise ValueError(mode)
        rows.append(eval_record(r, pred_vis))
    agg = aggregate(rows)
    if mode == 'local_recovery':
        agg.update(stats_total)
    return agg, {'per_video': rows}


def find_visible_segments(vis: np.ndarray):
    T = len(vis)
    segs = []
    t = 0
    while t < T:
        if not vis[t]:
            t += 1
            continue
        s = t
        while t < T and vis[t]:
            t += 1
        segs.append((s, t))
    return segs


def recover_local(
    pred_yx: np.ndarray,
    conf: np.ndarray,
    base_vis: np.ndarray,
    *,
    gt_vis: np.ndarray | None,
    tau_low: float,
    pre_window: int,
    max_jump_px: float,
    min_recovered_segment_len: int,
):
    """Open only an invisible suffix immediately before native re-entry.

    For online TrackOn2 migration, native visibility is the saved binary output,
    not conf>=0.5. The module only changes output visibility and does not feed
    changes back into TrackOn2 memory/state.
    """
    out = base_vis.copy()
    Q, T = base_vis.shape
    recovered_total = 0
    candidate_total = 0
    rejected_short = 0
    rejected_jump = 0
    opened_false = 0
    opened_true = 0
    for q in range(Q):
        for s, _e in find_visible_segments(base_vis[q]):
            if s <= 0 or base_vis[q, s - 1]:
                continue
            lo = max(0, s - int(pre_window))
            cand = []
            for t in range(lo, s):
                if base_vis[q, t]:
                    continue
                candidate_total += 1
                if conf[q, t] < tau_low:
                    continue
                if math.isfinite(max_jump_px):
                    jump = float(np.linalg.norm((pred_yx[q, t] - pred_yx[q, s]) * PIX))
                    if jump > max_jump_px:
                        rejected_jump += 1
                        continue
                cand.append(t)
            if not cand:
                continue
            cand_set = set(cand)
            suffix = []
            t = s - 1
            while t in cand_set:
                suffix.append(t)
                t -= 1
            suffix = list(reversed(suffix))
            if len(suffix) < int(min_recovered_segment_len):
                rejected_short += len(cand)
                continue
            out[q, suffix] = True
            recovered_total += len(suffix)
            if gt_vis is not None:
                opened_true += int(gt_vis[q, suffix].sum())
                opened_false += int((~gt_vis[q, suffix]).sum())
    return out, {
        'candidate_frames': int(candidate_total),
        'recovered_frames': int(recovered_total),
        'rejected_jump': int(rejected_jump),
        'rejected_short': int(rejected_short),
        'opened_true_visible': int(opened_true),
        'opened_false_visible': int(opened_false),
    }


def delta(agg: dict, native: dict) -> dict:
    keys = ['AJ','OA','delta_avg','J1','J2','J4','J8','J16','AJ_RD','AJ_RD_D4','AJ_RD_D16','pred_visible_rate','false_visible_rate_gt_occ','false_invisible_rate_gt_vis','visible_precision','visible_recall']
    return {k: (agg[k] - native[k] if isinstance(agg.get(k), float) and isinstance(native.get(k), float) else None) for k in keys}


def safe_success(d: dict) -> bool:
    return bool(d.get('AJ') is not None and d.get('OA') is not None and d.get('AJ_RD') is not None and d['AJ'] >= -0.10 and d['OA'] >= -0.10 and d['AJ_RD'] >= 0.01)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(DEFAULT_CACHE))
    ap.add_argument('--out-json', default=str(DEFAULT_OUT))
    ap.add_argument('--global-taus', nargs='+', type=float, default=[0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
    ap.add_argument('--tau-lows', nargs='+', type=float, default=[0.30,0.40,0.50,0.60,0.65,0.70,0.75])
    ap.add_argument('--pre-windows', nargs='+', type=int, default=[1,2,4,8])
    ap.add_argument('--max-jumps', nargs='+', default=['8','16','24','32','48','64','inf'])
    ap.add_argument('--min-lens', nargs='+', type=int, default=[1,2])
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = ROOT / cache
    payload = torch.load(cache, map_location='cpu', weights_only=False)
    records = payload['records']
    required = ['pred_tracks','pred_visibility','pred_vis_conf','gt_tracks','gt_visibility','query_points']
    for i, r in enumerate(records):
        miss = [k for k in required if k not in r]
        if miss:
            raise KeyError(f'record {i} {r.get("video_id")} missing {miss}')

    native, _ = eval_cache(records, 'native')

    global_rows = []
    for tau in args.global_taus:
        agg, _ = eval_cache(records, 'global_threshold', tau=tau)
        d = delta(agg, native)
        global_rows.append({'tau': float(tau), 'aggregate': agg, 'delta_vs_native': d, 'success': safe_success(d)})

    local_rows = []
    for tau_low in args.tau_lows:
        for pre_window in args.pre_windows:
            for mj in args.max_jumps:
                max_jump = float('inf') if str(mj).lower() == 'inf' else float(mj)
                for min_len in args.min_lens:
                    params = dict(tau_low=float(tau_low), pre_window=int(pre_window), max_jump_px=max_jump, min_recovered_segment_len=int(min_len))
                    agg, _ = eval_cache(records, 'local_recovery', local_params=params)
                    d = delta(agg, native)
                    local_rows.append({
                        'tau_low': float(tau_low),
                        'pre_window': int(pre_window),
                        'max_jump_px': max_jump if np.isfinite(max_jump) else 'inf',
                        'min_recovered_segment_len': int(min_len),
                        'aggregate': agg,
                        'delta_vs_native': d,
                        'success': safe_success(d),
                    })

    def score(row):
        d = row['delta_vs_native']
        aj = d.get('AJ') if d.get('AJ') is not None else -999
        oa = d.get('OA') if d.get('OA') is not None else -999
        ajrd = d.get('AJ_RD') if d.get('AJ_RD') is not None else -999
        penalty = max(0.0, -aj - 0.10) + max(0.0, -oa - 0.10)
        return ajrd - 0.05 * penalty

    global_sorted = sorted(global_rows, key=score, reverse=True)
    local_sorted = sorted(local_rows, key=score, reverse=True)
    successes = [r for r in local_rows if r['success']] + [r for r in global_rows if r['success']]

    out = {
        'cache': str(cache),
        'model_name': payload.get('model_name'),
        'schema_version': payload.get('schema_version'),
        'note': 'Online migration audit: TrackOn2 coordinates are fixed. Native base uses saved pred_visibility; threshold/local variants only modify output visibility, not tracker state.',
        'success_criterion': 'AJ>=native-0.10, OA>=native-0.10, AJ_RD>=native+0.01',
        'native': native,
        'global_threshold': {
            'top_20_by_safe_score': global_sorted[:20],
            'all_results': global_rows,
        },
        'local_recovery': {
            'top_20_by_safe_score': local_sorted[:20],
            'all_results': local_rows,
        },
        'success_count': len(successes),
        'successes': successes[:50],
    }
    outp = Path(args.out_json)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps({
        'out_json': str(outp),
        'native': native,
        'success_count': len(successes),
        'best_global': global_sorted[:5],
        'best_local': local_sorted[:10],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
