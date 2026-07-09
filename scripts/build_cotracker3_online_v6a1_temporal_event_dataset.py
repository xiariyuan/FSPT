#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a1_temporal_verifier/cotracker3_v6a1_temporal_event_dataset.npz'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def run_len_backward(mask: np.ndarray, t: int) -> int:
    c = 0
    i = t
    while i >= 0 and bool(mask[i]):
        c += 1
        i -= 1
    return c


def max_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def pad_get(arr: np.ndarray, idx: int, default: float) -> float:
    if idx < 0 or idx >= len(arr):
        return float(default)
    return float(arr[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--out-npz', default=str(DEFAULT_OUT))
    ap.add_argument('--support-score', type=float, default=0.80)
    ap.add_argument('--low-thr', type=float, default=0.60)
    ap.add_argument('--min-score', type=float, default=0.05)
    ap.add_argument('--trend-min', type=float, default=-0.10)
    ap.add_argument('--min-low-run', type=int, default=1)
    ap.add_argument('--pre', type=int, default=2)
    ap.add_argument('--post', type=int, default=4)
    ap.add_argument('--trigger-stride', type=int, default=2)
    ap.add_argument('--event-cooldown', type=int, default=2, help='skip additional triggers this many frames after a trigger for same query')
    args = ap.parse_args()

    payload = torch.load(args.native_cache, map_location='cpu', weights_only=False)
    X = []
    meta = []
    y_future_safe16 = []
    y_future_safe8 = []
    y_future_gt_visible = []
    y_open_t_safe16 = []
    best_safe16_offset = []
    best_safe8_offset = []

    # Feature names are GT-free. Some features include post-window model predictions, allowed by short-latency K.
    feature_names = []
    for dt in range(-args.pre, args.post + 1):
        feature_names.append(f'score_dt{dt}')
    for dt in range(-args.pre, args.post + 1):
        feature_names.append(f'vis_dt{dt}')
    feature_names += [
        'score_mean_pre', 'score_mean_post', 'score_max_post', 'score_min_post', 'score_std_post',
        'score_slope_t_to_post', 'score_recovery_amount', 'first_score_ge_06_offset_norm',
        'visible_count_post_norm', 'max_visible_run_post_norm',
        'low_run_before_norm', 'invis_run_before_norm', 'support_age_norm', 'last_visible_age_norm',
        'coord_y_t', 'coord_x_t', 'near_border_t',
        'dist_t_to_support_norm', 'dist_t_to_last_visible_norm',
        'mean_motion_post_norm', 'max_motion_post_norm', 'std_motion_post_norm',
        'drift_t_to_post_norm', 'motion_prev1_norm', 'motion_prev2_norm',
        'query_age_norm', 'frame_t_norm', 'query_idx_norm', 'video_idx_norm',
        'score_margin_to_thr_t', 'post_window_available_norm',
    ]

    n_records = len(payload['records'])
    for vi, r in enumerate(payload['records']):
        vid = str(r['video_id'])
        pred_vis = npy(r['pred_visibility'], bool)
        gt_vis = npy(r['gt_visibility'], bool)
        score = npy(r.get('pred_vis_score', pred_vis.astype(np.float32)), np.float32)
        pred = npy(r['pred_tracks'], np.float32)
        gt = npy(r['gt_tracks'], np.float32)
        qpts = npy(r['query_points'], np.float32)
        N, T = pred_vis.shape
        low_mask = (score <= args.low_thr) | (~pred_vis)
        support_mask = pred_vis & (score >= args.support_score)
        for q in range(N):
            q_t = int(round(float(qpts[q, 0])))
            support_candidates = np.where(support_mask[q])[0]
            next_allowed = max(q_t + 1, 1)
            for t in range(max(q_t + 1, 1), T):
                if t < next_allowed:
                    continue
                if args.trigger_stride > 1 and ((t - q_t) % args.trigger_stride) != 0:
                    continue
                if not bool(low_mask[q, t]):
                    continue
                if float(score[q, t]) < args.min_score:
                    continue
                if t > 0 and float(score[q, t] - score[q, t - 1]) < args.trend_min:
                    continue
                support_before = support_candidates[support_candidates < t]
                if support_before.size == 0:
                    continue
                sup_t = int(support_before[-1])
                low_run = run_len_backward(low_mask[q], t)
                invis_run = run_len_backward(~pred_vis[q], t)
                if low_run < args.min_low_run and invis_run < args.min_low_run:
                    continue
                last_vis = np.where(pred_vis[q, :t])[0]
                last_t = int(last_vis[-1]) if last_vis.size else sup_t

                idxs = list(range(t - args.pre, t + args.post + 1))
                score_seq = np.asarray([pad_get(score[q], ii, float(score[q, t])) for ii in idxs], np.float32)
                vis_seq = np.asarray([1.0 if (0 <= ii < T and bool(pred_vis[q, ii])) else 0.0 for ii in idxs], np.float32)
                post_idxs = list(range(t, min(T, t + args.post + 1)))
                post_scores = score[q, post_idxs]
                post_vis = pred_vis[q, post_idxs]
                post_available = len(post_idxs) / float(args.post + 1)

                # score offset: first frame in future window crossing native threshold.
                first_ge = None
                for off, ii in enumerate(post_idxs):
                    if float(score[q, ii]) >= args.low_thr or bool(pred_vis[q, ii]):
                        first_ge = off
                        break
                first_ge_norm = 1.0 if first_ge is None else float(first_ge) / max(args.post, 1)

                def dist(a, b):
                    return float(np.linalg.norm((a - b) * 255.0)) / 255.0

                motions = []
                for ii in range(t, min(T - 1, t + args.post)):
                    motions.append(dist(pred[q, ii + 1], pred[q, ii]))
                motions = np.asarray(motions, np.float32) if motions else np.asarray([0.0], np.float32)
                coord_t = pred[q, t]
                y = float(coord_t[0]); x = float(coord_t[1])
                border = min(y, x, 1 - y, 1 - x)
                motion_prev1 = dist(pred[q, t], pred[q, t - 1]) if t - 1 >= 0 else 0.0
                motion_prev2 = dist(pred[q, t - 1], pred[q, t - 2]) if t - 2 >= 0 else 0.0
                drift_t_to_post = dist(pred[q, min(T - 1, t + args.post)], pred[q, t])

                feat = []
                feat.extend(score_seq.tolist())
                feat.extend(vis_seq.tolist())
                feat.extend([
                    float(np.mean(score_seq[:args.pre])) if args.pre > 0 else float(score[q, t]),
                    float(np.mean(post_scores)) if len(post_scores) else float(score[q, t]),
                    float(np.max(post_scores)) if len(post_scores) else float(score[q, t]),
                    float(np.min(post_scores)) if len(post_scores) else float(score[q, t]),
                    float(np.std(post_scores)) if len(post_scores) else 0.0,
                    float(post_scores[-1] - score[q, t]) if len(post_scores) else 0.0,
                    float(np.max(post_scores) - score[q, t]) if len(post_scores) else 0.0,
                    first_ge_norm,
                    float(np.sum(post_vis)) / float(args.post + 1),
                    float(max_true_run(post_vis)) / float(args.post + 1),
                    float(low_run) / 255.0,
                    float(invis_run) / 255.0,
                    float(t - sup_t) / 255.0,
                    float(t - last_t) / 255.0,
                    y, x, float(border),
                    dist(pred[q, t], pred[q, sup_t]),
                    dist(pred[q, t], pred[q, last_t]),
                    float(np.mean(motions)),
                    float(np.max(motions)),
                    float(np.std(motions)),
                    drift_t_to_post,
                    motion_prev1,
                    motion_prev2,
                    float(t - q_t) / 255.0,
                    float(t) / 255.0,
                    float(q) / max(N - 1, 1),
                    float(vi) / max(n_records - 1, 1),
                    float(args.low_thr - score[q, t]),
                    float(post_available),
                ])

                # Window labels. Find any safe frame in short-latency window.
                safe16_offsets = []
                safe8_offsets = []
                gt_offsets = []
                for off, ii in enumerate(post_idxs):
                    err = float(np.linalg.norm((pred[q, ii] - gt[q, ii]) * 255.0))
                    if bool(gt_vis[q, ii]):
                        gt_offsets.append(off)
                    if bool(gt_vis[q, ii]) and err <= 16.0:
                        safe16_offsets.append(off)
                    if bool(gt_vis[q, ii]) and err <= 8.0:
                        safe8_offsets.append(off)
                open_t_err = float(np.linalg.norm((pred[q, t] - gt[q, t]) * 255.0))
                open_t_safe16 = bool(gt_vis[q, t] and open_t_err <= 16.0)

                X.append(feat)
                y_future_safe16.append(1.0 if safe16_offsets else 0.0)
                y_future_safe8.append(1.0 if safe8_offsets else 0.0)
                y_future_gt_visible.append(1.0 if gt_offsets else 0.0)
                y_open_t_safe16.append(1.0 if open_t_safe16 else 0.0)
                best_safe16_offset.append(float(safe16_offsets[0]) if safe16_offsets else -1.0)
                best_safe8_offset.append(float(safe8_offsets[0]) if safe8_offsets else -1.0)
                meta.append(json.dumps({
                    'video_id': vid, 'video_index': vi, 'query_idx': int(q), 'frame_t': int(t), 'query_t': int(q_t),
                    'support_t': int(sup_t), 'last_visible_t': int(last_t), 'low_run': int(low_run), 'invis_run': int(invis_run),
                    'native_visible_t': bool(pred_vis[q, t]), 'score_t': float(score[q, t]),
                    'open_t_safe16': open_t_safe16, 'open_t_err_px': open_t_err,
                    'future_safe16': bool(safe16_offsets), 'best_safe16_offset': int(safe16_offsets[0]) if safe16_offsets else -1,
                    'future_safe8': bool(safe8_offsets), 'best_safe8_offset': int(safe8_offsets[0]) if safe8_offsets else -1,
                    'future_gt_visible': bool(gt_offsets), 'post_available': post_available,
                }, ensure_ascii=False))
                next_allowed = t + max(args.event_cooldown, 1)

    X = np.asarray(X, np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        X=X,
        feature_names=np.asarray(feature_names, dtype=object),
        y_future_safe16=np.asarray(y_future_safe16, np.float32),
        y_future_safe8=np.asarray(y_future_safe8, np.float32),
        y_future_gt_visible=np.asarray(y_future_gt_visible, np.float32),
        y_open_t_safe16=np.asarray(y_open_t_safe16, np.float32),
        best_safe16_offset=np.asarray(best_safe16_offset, np.float32),
        best_safe8_offset=np.asarray(best_safe8_offset, np.float32),
        meta_json=np.asarray(meta, dtype=object),
        native_cache=str(args.native_cache),
        params=json.dumps(vars(args)),
    )
    np.savez_compressed(out, **arrays)
    from collections import Counter
    vids = Counter(json.loads(str(m))['video_id'] for m in meta)
    summary = {
        'out_npz': str(out), 'n_samples': int(X.shape[0]),
        'future_safe16_pos': int(np.sum(arrays['y_future_safe16'])), 'future_safe16_rate': float(np.mean(arrays['y_future_safe16'])) if X.shape[0] else 0.0,
        'future_safe8_pos': int(np.sum(arrays['y_future_safe8'])), 'future_safe8_rate': float(np.mean(arrays['y_future_safe8'])) if X.shape[0] else 0.0,
        'future_gt_visible_pos': int(np.sum(arrays['y_future_gt_visible'])), 'future_gt_visible_rate': float(np.mean(arrays['y_future_gt_visible'])) if X.shape[0] else 0.0,
        'open_t_safe16_pos': int(np.sum(arrays['y_open_t_safe16'])), 'open_t_safe16_rate': float(np.mean(arrays['y_open_t_safe16'])) if X.shape[0] else 0.0,
        'n_videos': len(vids), 'videos': dict(vids), 'feature_dim': len(feature_names), 'feature_names': feature_names,
    }
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
