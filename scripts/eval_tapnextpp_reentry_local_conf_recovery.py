#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, math
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
        e = t
        segs.append((s, e))
    return segs


def recover_local(
    pred_yx: np.ndarray,
    conf: np.ndarray,
    base_vis: np.ndarray,
    *,
    tau_low: float,
    pre_window: int,
    max_jump_px: float,
    min_recovered_segment_len: int,
):
    """Recover only frames immediately before native re-entry segments.

    base_vis is native conf>=0.5. For each visible segment starting at s>0 after
    an invisible frame, consider frames [s-pre_window, s). If conf>=tau_low and
    coordinate is close to the native re-entry coordinate at s, open it.
    """
    out = base_vis.copy()
    Q, T = base_vis.shape
    recovered_total = 0
    candidate_total = 0
    rejected_short = 0
    rejected_jump = 0
    for q in range(Q):
        segs = find_visible_segments(base_vis[q])
        for s, e in segs:
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
            # Keep only contiguous suffix ending at s-1. This prevents opening an
            # isolated earlier blip that does not directly attach to the re-entry.
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
    stats = {
        'candidate_frames': int(candidate_total),
        'recovered_frames': int(recovered_total),
        'rejected_jump': int(rejected_jump),
        'rejected_short': int(rejected_short),
    }
    return out, stats


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
        'J4': tap['jaccard_4'] * 100.0,
        'J8': tap['jaccard_8'] * 100.0,
        'J16': tap['jaccard_16'] * 100.0,
        'AJ_RD': aj.get('AJ_RD'),
        'AJ_RD_D4': aj.get('AJ_RD_D4_dmin1'),
        'AJ_RD_D16': aj.get('AJ_RD_D16_dmin1'),
        'pred_visible_rate': float(pred_vis.mean()),
        'false_visible_rate_gt_occ': float(false_visible / max(gt_false, 1)),
        'false_invisible_rate_gt_vis': float(false_invisible / max(gt_true, 1)),
        'visible_precision': float(true_visible / max(pred_true, 1)),
        'visible_recall': float(true_visible / max(gt_true, 1)),
        'n_points': int(q.shape[0]),
    }


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r.get(key) == r.get(key)]
    return float(np.mean(vals)) if vals else None


def eval_config(records, *, tau_low, pre_window, max_jump_px, min_recovered_segment_len):
    rows = []
    stats_total = {'candidate_frames': 0, 'recovered_frames': 0, 'rejected_jump': 0, 'rejected_short': 0}
    for r in records:
        pred_yx = arr(r['pred_tracks'], np.float32)
        conf = arr(r['pred_vis_conf'], np.float32)
        base_vis = conf >= 0.5
        pred_vis, st = recover_local(
            pred_yx, conf, base_vis,
            tau_low=tau_low,
            pre_window=pre_window,
            max_jump_px=max_jump_px,
            min_recovered_segment_len=min_recovered_segment_len,
        )
        for k in stats_total:
            stats_total[k] += st[k]
        rows.append(eval_record(r, pred_vis))
    keys = ['AJ','OA','delta_avg','J4','J8','J16','AJ_RD','AJ_RD_D4','AJ_RD_D16','pred_visible_rate','false_visible_rate_gt_occ','false_invisible_rate_gt_vis','visible_precision','visible_recall']
    agg = {k: mean(rows, k) for k in keys}
    agg.update(stats_total)
    agg['n_videos'] = len(rows)
    agg['n_queries'] = int(sum(r['n_points'] for r in rows))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v5_conf.pt')
    ap.add_argument('--out-json', default='outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_reentry_local_conf_recovery_sweep.json')
    ap.add_argument('--tau-lows', nargs='+', type=float, default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument('--pre-windows', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--max-jumps', nargs='+', default=['16', '32', '64', 'inf'])
    ap.add_argument('--min-lens', nargs='+', type=int, default=[1, 2])
    args = ap.parse_args()
    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = ROOT / cache
    payload = torch.load(cache, map_location='cpu', weights_only=False)
    records = payload['records']

    native = eval_config(records, tau_low=0.5, pre_window=0, max_jump_px=float('inf'), min_recovered_segment_len=1)
    results = []
    for tau_low in args.tau_lows:
        for pre_window in args.pre_windows:
            for mj in args.max_jumps:
                max_jump = float('inf') if str(mj).lower() == 'inf' else float(mj)
                for min_len in args.min_lens:
                    agg = eval_config(records, tau_low=tau_low, pre_window=pre_window, max_jump_px=max_jump, min_recovered_segment_len=min_len)
                    delta = {k: (agg[k] - native[k] if isinstance(agg.get(k), float) and isinstance(native.get(k), float) else None) for k in ['AJ','OA','delta_avg','J4','J8','J16','AJ_RD','AJ_RD_D4','AJ_RD_D16','pred_visible_rate','false_visible_rate_gt_occ','false_invisible_rate_gt_vis']}
                    results.append({
                        'tau_low': tau_low,
                        'pre_window': pre_window,
                        'max_jump_px': max_jump if np.isfinite(max_jump) else 'inf',
                        'min_recovered_segment_len': min_len,
                        'aggregate': agg,
                        'delta_vs_native': delta,
                    })
    # Sort by safe utility: prefer AJ_RD gain, but require small AJ/OA cost.
    def score(row):
        d = row['delta_vs_native']
        aj = d['AJ'] if d['AJ'] is not None else -999
        oa = d['OA'] if d['OA'] is not None else -999
        ajrd = d['AJ_RD'] if d['AJ_RD'] is not None else -999
        penalty = max(0, -aj - 0.10) + max(0, -oa - 0.10)
        return ajrd - 0.05 * penalty
    results_sorted = sorted(results, key=score, reverse=True)
    out = {
        'cache': str(cache),
        'note': 'Native visibility is conf>=0.5. Recovery lowers threshold only in invisible frames immediately preceding native re-entry segments, with coordinate jump and min-length gates.',
        'native': native,
        'success_criterion': 'AJ>=native-0.10, OA>=native-0.10, AJ_RD>=native+0.01',
        'top_20_by_safe_score': results_sorted[:20],
        'all_results': results,
    }
    outp = Path(args.out_json)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({'out_json': str(outp), 'native': native, 'top_10': results_sorted[:10]}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
