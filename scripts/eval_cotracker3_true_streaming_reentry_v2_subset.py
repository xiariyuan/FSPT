#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from datasets.metrics import compute_tapvid_metrics

DEFAULT_REF_CACHE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v2_subset_eval'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def find_visible_segments(vis_1d: np.ndarray):
    T = len(vis_1d)
    segs = []
    t = 0
    while t < T:
        if not vis_1d[t]:
            t += 1
            continue
        s = t
        while t < T and vis_1d[t]:
            t += 1
        segs.append((s, t))
    return segs


def safe_recovery_mask(
    score_nt: np.ndarray,
    coords_yx_nt: np.ndarray,
    base_vis_nt: np.ndarray,
    *,
    tau_low: float,
    pre_window: int,
    max_jump_px: float,
    min_len: int,
    confirm_visible_len: int = 2,
    trend_min: float = -0.02,
    allowed_frames: set[int] | None = None,
):
    """Conservative GT-free re-entry candidate mask.

    Design intent:
    - only repair the invisible suffix immediately before a stable native visible segment;
    - optionally restrict repairs to frames that will be copied to the next online window;
    - use score, local trend, short jump, and contiguous suffix checks.
    """
    N, T = base_vis_nt.shape
    open_mask = np.zeros_like(base_vis_nt, dtype=bool)
    stats = {
        'candidate_frames': 0,
        'opened_frames': 0,
        'rejected_score': 0,
        'rejected_trend': 0,
        'rejected_jump': 0,
        'rejected_unstable_reentry': 0,
        'rejected_short': 0,
        'rejected_not_allowed': 0,
    }
    for q in range(N):
        for s, e in find_visible_segments(base_vis_nt[q]):
            if s <= 0 or base_vis_nt[q, s - 1]:
                continue
            if s + int(confirm_visible_len) > T or not bool(base_vis_nt[q, s:s + int(confirm_visible_len)].all()):
                stats['rejected_unstable_reentry'] += 1
                continue
            lo = max(0, s - int(pre_window))
            cand = []
            for t in range(lo, s):
                if base_vis_nt[q, t]:
                    continue
                if allowed_frames is not None and t not in allowed_frames:
                    stats['rejected_not_allowed'] += 1
                    continue
                stats['candidate_frames'] += 1
                if score_nt[q, t] < float(tau_low):
                    stats['rejected_score'] += 1
                    continue
                if t > 0 and (float(score_nt[q, t] - score_nt[q, t - 1]) < float(trend_min)):
                    stats['rejected_trend'] += 1
                    continue
                if math.isfinite(max_jump_px):
                    jump = float(np.linalg.norm((coords_yx_nt[q, t] - coords_yx_nt[q, s]) * 255.0))
                    if jump > float(max_jump_px):
                        stats['rejected_jump'] += 1
                        continue
                cand.append(t)
            if not cand:
                continue
            candset = set(cand)
            suffix = []
            t = s - 1
            while t in candset:
                suffix.append(t)
                t -= 1
            suffix = list(reversed(suffix))
            if len(suffix) < int(min_len):
                stats['rejected_short'] += len(cand)
                continue
            open_mask[q, suffix] = True
            stats['opened_frames'] += len(suffix)
    return open_mask, stats

def load_video_for_record(record: dict, ds: TapVidDataset):
    vid = str(record['video_id'])
    idx = ds.video_names.index(vid)
    sample = ds[idx]
    rgbs = sample.video.unsqueeze(0).cuda().float()
    q_norm = npy(record['query_points'], np.float32)
    q = torch.zeros((1, q_norm.shape[0], 3), device='cuda', dtype=torch.float32)
    q[0, :, 0] = torch.from_numpy(q_norm[:, 0]).cuda()
    q[0, :, 1] = torch.from_numpy(q_norm[:, 2] * 255.0).cuda()  # x
    q[0, :, 2] = torch.from_numpy(q_norm[:, 1] * 255.0).cuda()  # y
    return rgbs, q


def state_to_arrays(model: CoTrackerOnlinePredictor, T: int, N: int):
    coords_xy = model.model.online_coords_predicted[0, :T, :N].detach().float().cpu().numpy()
    raw_v = model.model.online_vis_predicted[0, :T, :N].detach().float().cpu()
    raw_c = model.model.online_conf_predicted[0, :T, :N].detach().float().cpu()
    score_tn = (torch.sigmoid(raw_v) * torch.sigmoid(raw_c)).numpy()
    interp_h, interp_w = model.interp_shape
    coords_xy[..., 0] *= 255.0 / max(float(interp_w - 1), 1.0)
    coords_xy[..., 1] *= 255.0 / max(float(interp_h - 1), 1.0)
    coords_yx_nt = coords_xy.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32) / 255.0
    score_nt = score_tn.transpose(1, 0).astype(np.float32)
    pred_vis_nt = (score_nt > 0.6)
    return coords_yx_nt, score_nt, pred_vis_nt


def make_record(base_record: dict, pred_tracks_nt: np.ndarray, pred_vis_nt: np.ndarray, score_nt: np.ndarray, name: str):
    rec = dict(base_record)
    rec['pred_tracks'] = pred_tracks_nt.astype(np.float32)
    rec['pred_visibility'] = pred_vis_nt.astype(bool)
    rec['pred_vis_score'] = score_nt.astype(np.float32)
    rec['model_name'] = name
    rec['adapter_version'] = name
    rec['raw_coordinate_note'] = 'True streaming CoTrackerOnlinePredictor subset evaluation; coordinates fixed for output-only and state-writeback variants except state writeback may change later coordinates through online state.'
    return rec


def run_variant(model: CoTrackerOnlinePredictor, base_record: dict, video: torch.Tensor, queries_xy: torch.Tensor, variant: str, params: dict):
    T = int(video.shape[1])
    N = int(queries_xy.shape[1])
    model(video_chunk=video, is_first_step=True, queries=queries_xy, add_support_grid=False, grid_size=0)
    high_logit = float(logit(params.get('state_prob', 0.97)))
    state_stats_total = {'candidate_frames': 0, 'opened_frames': 0, 'rejected_jump': 0, 'rejected_short': 0}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for ind in range(0, T - model.step, model.step):
            chunk = video[:, ind: ind + model.step * 2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
            if variant in ('state_writeback', 'state_plus_output'):
                cur_T = min(int(model.model.online_vis_predicted.shape[1]), T)
                cur_N = min(int(model.model.online_vis_predicted.shape[2]), N)
                coords_yx_nt, score_nt, base_vis_nt = state_to_arrays(model, cur_T, cur_N)
                # Only write frames that will initialize the next online window.
                # For CoTracker3 window_len=16, step=8, after a chunk starting at ind,
                # the next call copies the previous overlap [ind+step, ind+2*step).
                allowed_frames = set(range(ind + model.step, min(ind + 2 * model.step, cur_T)))
                open_mask_nt, st = safe_recovery_mask(score_nt, coords_yx_nt, base_vis_nt, tau_low=params['tau_low'], pre_window=params['pre_window'], max_jump_px=params['max_jump_px'], min_len=params['min_len'], confirm_visible_len=params['confirm_visible_len'], trend_min=params['trend_min'], allowed_frames=allowed_frames)
                q_idx, t_idx = np.where(open_mask_nt)
                if len(q_idx) > 0:
                    tt = torch.from_numpy(t_idx).cuda()
                    qq = torch.from_numpy(q_idx).cuda()
                    old_v = model.model.online_vis_predicted[0, tt, qq]
                    old_c = model.model.online_conf_predicted[0, tt, qq]
                    model.model.online_vis_predicted[0, tt, qq] = torch.maximum(old_v, torch.full_like(old_v, high_logit))
                    model.model.online_conf_predicted[0, tt, qq] = torch.maximum(old_c, torch.full_like(old_c, high_logit))
                for k in state_stats_total:
                    state_stats_total[k] += int(st.get(k, 0))
    coords_yx_nt, score_nt, pred_vis_nt = state_to_arrays(model, T, N)
    output_stats = {'candidate_frames': 0, 'opened_frames': 0, 'rejected_jump': 0, 'rejected_short': 0}
    if variant in ('output_only', 'state_plus_output'):
        open_mask_nt, output_stats = safe_recovery_mask(score_nt, coords_yx_nt, pred_vis_nt, tau_low=params['tau_low'], pre_window=params['pre_window'], max_jump_px=params['max_jump_px'], min_len=params['min_len'], confirm_visible_len=params['confirm_visible_len'], trend_min=params['trend_min'], allowed_frames=None)
        pred_vis_nt = pred_vis_nt.copy()
        pred_vis_nt[open_mask_nt] = True
    rec = make_record(base_record, coords_yx_nt, pred_vis_nt, score_nt, f'cotracker3_true_streaming_{variant}')
    stats = {
        'state_stats': state_stats_total,
        'output_stats': output_stats,
        'visible_rate': float(pred_vis_nt.mean()),
    }
    return rec, stats


def eval_std(records: list[dict]) -> dict:
    rows = []
    n = 0
    for r in records:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        n += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
        rows.append({k: float(m.get(k, 0.0)) for k in ['AJ', 'OA', 'average_pts_within_thresh', 'pts_within_4']})
    return {
        'AJ': float(np.mean([r['AJ'] for r in rows])) * 100.0,
        'OA': float(np.mean([r['OA'] for r in rows])) * 100.0,
        'delta_avg': float(np.mean([r['average_pts_within_thresh'] for r in rows])) * 100.0,
        'delta_4px': float(np.mean([r['pts_within_4'] for r in rows])) * 100.0,
        'n_records': len(records),
        'n_queries': n,
    }


def eval_ajrd_cache(payload: dict, out_cache: Path, out_json: Path) -> dict:
    torch.save(payload, out_cache)
    subprocess.run([sys.executable, str(ROOT / 'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(out_cache), '--output-json', str(out_json)], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(out_json.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-cache', default=str(DEFAULT_REF_CACHE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--max-videos', type=int, default=3)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--tau-low', type=float, default=0.55)
    ap.add_argument('--pre-window', type=int, default=2)
    ap.add_argument('--max-jump-px', type=float, default=16.0)
    ap.add_argument('--min-len', type=int, default=2)
    ap.add_argument('--confirm-visible-len', type=int, default=2)
    ap.add_argument('--trend-min', type=float, default=-0.02)
    ap.add_argument('--state-prob', type=float, default=0.65)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = {'tau_low': args.tau_low, 'pre_window': args.pre_window, 'max_jump_px': args.max_jump_px, 'min_len': args.min_len, 'confirm_visible_len': args.confirm_visible_len, 'trend_min': args.trend_min, 'state_prob': args.state_prob, 'write_region': 'overlap_only_for_state'}
    ref_payload = torch.load(args.ref_cache, map_location='cpu', weights_only=False)
    ref_records = ref_payload['records']
    end = min(len(ref_records), args.start_index + args.max_videos)
    selected = list(range(args.start_index, end))
    print('loading DAVIS dataset once', flush=True)
    ds = TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256, 256], queried_first=True)
    print('loading CoTrackerOnlinePredictor once', flush=True)
    model = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).cuda().eval()
    variants = ['native', 'output_only', 'state_writeback', 'state_plus_output']
    records_by_variant = {v: [] for v in variants}
    stats_by_variant = {v: [] for v in variants}
    per_video = []
    t_all = time.time()
    for idx in selected:
        base_record = ref_records[idx]
        video, queries_xy = load_video_for_record(base_record, ds)
        item = {'index': int(idx), 'video_id': str(base_record['video_id']), 'frames': int(video.shape[1]), 'queries': int(queries_xy.shape[1]), 'variants': {}}
        for variant in variants:
            t0 = time.time()
            rec, st = run_variant(model, base_record, video, queries_xy, variant, params)
            tr_diff = np.abs(npy(rec['pred_tracks'], np.float32) - npy(base_record['pred_tracks'], np.float32))
            vis_diff = npy(rec['pred_visibility'], bool) != npy(base_record['pred_visibility'], bool)
            stats_by_variant[variant].append(st)
            records_by_variant[variant].append(rec)
            item['variants'][variant] = {
                'sec': round(float(time.time() - t0), 3),
                'visible_rate': st['visible_rate'],
                'opened_state': st['state_stats']['opened_frames'],
                'opened_output': st['output_stats']['opened_frames'],
                'diff_vs_existing_arch_vis_rate': float(vis_diff.mean()),
                'diff_vs_existing_arch_track_mean': float(tr_diff.mean()),
            }
            print(json.dumps({'video': item['video_id'], 'variant': variant, **item['variants'][variant]}, ensure_ascii=False), flush=True)
            torch.cuda.empty_cache()
        per_video.append(item)
        del video, queries_xy
    summaries = {}
    for variant in variants:
        payload = dict(ref_payload)
        payload['model_name'] = f'cotracker3_true_streaming_{variant}_subset'
        payload['schema_version'] = 'cotracker3_true_streaming_reentry_v2_subset_v1'
        payload['protocol'] = 'true_streaming_first_input_subset'
        payload['records'] = records_by_variant[variant]
        payload['subset_indices'] = selected
        payload['reentry_params'] = params
        cache_path = outdir / f'cotracker3_true_streaming_{variant}_subset.pt'
        ajrd_path = outdir / f'cotracker3_true_streaming_{variant}_subset_ajrd.json'
        ajrd = eval_ajrd_cache(payload, cache_path, ajrd_path)
        std = eval_std(records_by_variant[variant])
        total_stats = {}
        for key in ['candidate_frames', 'opened_frames', 'rejected_jump', 'rejected_short']:
            total_stats[f'state_{key}'] = int(sum(s['state_stats'].get(key, 0) for s in stats_by_variant[variant]))
            total_stats[f'output_{key}'] = int(sum(s['output_stats'].get(key, 0) for s in stats_by_variant[variant]))
        summaries[variant] = {
            'cache': str(cache_path),
            **std,
            'AJ_RD': ajrd.get('true_AJ_RD'),
            'AJ_RD_256': ajrd.get('true_AJ_RD_256'),
            'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
            'stats': total_stats,
        }
    native = summaries['native']
    for variant in ['output_only', 'state_writeback', 'state_plus_output']:
        summaries[variant]['delta_vs_native'] = {
            k: (summaries[variant][k] - native[k] if isinstance(summaries[variant].get(k), (int, float)) and isinstance(native.get(k), (int, float)) else None)
            for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256', 'first_reentry_frame_proxy']
        }
    report = {
        'script': 'scripts/eval_cotracker3_true_streaming_reentry_v2_subset.py',
        'params': params,
        'selected_indices': selected,
        'num_videos': len(selected),
        'total_sec': round(float(time.time() - t_all), 2),
        'per_video': per_video,
        'summaries': summaries,
        'success_criterion': 'For subset gate: AJ>=native-0.10, OA>=native-0.10, AJ_RD>=native+0.01 before considering full run.',
    }
    out_report = outdir / 'cotracker3_true_streaming_reentry_v2_subset_report.json'
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'out_report': str(out_report), 'summaries': summaries}, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
