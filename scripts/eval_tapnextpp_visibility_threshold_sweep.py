#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
REPO = ROOT / 'external/tapnextpp/repo'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from tapnet.tapnextpp.metrics.aj_rd import compute_redetection_metrics

PIX = 255.0


def arr(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def eval_record(r: dict, tau: float) -> dict:
    pred_yx = arr(r['pred_tracks'], np.float32)
    gt_yx = arr(r['gt_tracks'], np.float32)
    q = arr(r['query_points'], np.float32)
    gt_vis = arr(r['gt_visibility'], bool)
    conf = arr(r['pred_vis_conf'], np.float32)
    pred_vis = conf >= float(tau)

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


def mean_metric(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None and r.get(key) == r.get(key)]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt')
    ap.add_argument('--out-json', default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_visibility_threshold_sweep.json')
    ap.add_argument('--taus', nargs='+', type=float, default=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    records = payload['records']

    required = ['pred_tracks', 'pred_vis_conf', 'gt_tracks', 'gt_visibility', 'query_points']
    for i, r in enumerate(records):
        missing = [k for k in required if k not in r]
        if missing:
            raise KeyError(f'record {i} {r.get("video_id")} missing {missing}')

    rows_by_tau = {}
    for tau in args.taus:
        per_video = {}
        rows = []
        for r in records:
            m = eval_record(r, tau)
            rows.append(m)
            per_video[str(r['video_id'])] = m
        keys = [
            'AJ', 'OA', 'delta_avg', 'd1', 'd2', 'd4', 'd8', 'd16',
            'J1', 'J2', 'J4', 'J8', 'J16',
            'AJ_RD', 'AJ_RD_D4', 'AJ_RD_D16',
            'pred_visible_rate', 'gt_visible_rate',
            'false_visible_rate_all', 'false_visible_rate_gt_occ', 'false_invisible_rate_gt_vis',
            'visible_precision', 'visible_recall',
        ]
        agg = {k: mean_metric(rows, k) for k in keys}
        agg['n_videos'] = len(rows)
        agg['n_queries'] = int(sum(r['n_points'] for r in rows))
        rows_by_tau[f'{tau:.2f}'] = {'tau': float(tau), 'aggregate': agg, 'per_video': per_video}

    # Use tau=0.50 as native reference if present, else nearest to 0.5.
    tau_keys = list(rows_by_tau.keys())
    ref_key = min(tau_keys, key=lambda k: abs(float(k) - 0.5))
    ref = rows_by_tau[ref_key]['aggregate']
    for k, block in rows_by_tau.items():
        agg = block['aggregate']
        block['delta_vs_tau_ref'] = {
            kk: (agg[kk] - ref[kk] if isinstance(agg.get(kk), float) and isinstance(ref.get(kk), float) else None)
            for kk in ['AJ', 'OA', 'delta_avg', 'J4', 'J8', 'J16', 'AJ_RD', 'AJ_RD_D4', 'AJ_RD_D16', 'pred_visible_rate', 'false_visible_rate_gt_occ', 'false_invisible_rate_gt_vis']
        }

    compact_rows = []
    for k in tau_keys:
        a = rows_by_tau[k]['aggregate']
        d = rows_by_tau[k]['delta_vs_tau_ref']
        compact_rows.append({
            'tau': rows_by_tau[k]['tau'],
            'AJ': a['AJ'], 'OA': a['OA'], 'delta_avg': a['delta_avg'],
            'J4': a['J4'], 'J8': a['J8'], 'J16': a['J16'],
            'AJ_RD': a['AJ_RD'], 'AJ_RD_D4': a['AJ_RD_D4'], 'AJ_RD_D16': a['AJ_RD_D16'],
            'pred_visible_rate': a['pred_visible_rate'],
            'false_visible_rate_gt_occ': a['false_visible_rate_gt_occ'],
            'false_invisible_rate_gt_vis': a['false_invisible_rate_gt_vis'],
            'dAJ_vs_ref': d['AJ'], 'dOA_vs_ref': d['OA'], 'dAJRD_vs_ref': d['AJ_RD'],
        })

    out = {
        'cache': str(cache_path),
        'model_name': payload.get('model_name'),
        'protocol': payload.get('protocol'),
        'reference_tau_key': ref_key,
        'note': 'Coordinates are fixed; only visibility threshold pred_vis_conf >= tau is varied.',
        'compact_rows': compact_rows,
        'results_by_tau': rows_by_tau,
    }
    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps({'out_json': str(out_path), 'reference_tau_key': ref_key, 'compact_rows': compact_rows}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
