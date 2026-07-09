#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events

DEFAULT_VIDEO_CACHE = Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
DEFAULT_GT_CACHE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
DEFAULT_OUT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16')


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def shift_curve(T: int, t0: int, ramp: int, hold: int, amp_px: float) -> np.ndarray:
    dx = np.zeros(T, dtype=np.float32)
    ramp = max(1, int(ramp))
    t0 = int(t0)
    hold = int(hold)
    for t in range(T):
        if t < t0:
            v = 0.0
        elif t < t0 + ramp:
            v = amp_px * ((t - t0 + 1) / ramp)
        elif t < t0 + ramp + hold:
            v = amp_px
        elif t < t0 + 2 * ramp + hold:
            v = amp_px * (1.0 - ((t - (t0 + ramp + hold) + 1) / ramp))
        else:
            v = 0.0
        dx[t] = float(v)
    return dx


def translate_frame(frame: np.ndarray, dx: int, dy: int, fill_mode: str = 'frame_mean') -> np.ndarray:
    H, W = frame.shape[:2]
    if fill_mode == 'zero':
        fill = np.zeros((1, 1, 3), dtype=np.uint8)
    elif fill_mode == 'gray':
        fill = np.full((1, 1, 3), 127, dtype=np.uint8)
    else:
        fill = np.asarray(np.mean(frame.reshape(-1, 3), axis=0), dtype=np.uint8).reshape(1, 1, 3)
    out = np.tile(fill, (H, W, 1)).astype(np.uint8)
    dx = int(dx)
    dy = int(dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def make_stress_video(video: np.ndarray, dx: np.ndarray, dy: np.ndarray, fill_mode: str) -> np.ndarray:
    out = []
    for t in range(video.shape[0]):
        out.append(translate_frame(video[t], int(round(float(dx[t]))), int(round(float(dy[t]))), fill_mode=fill_mode))
    return np.stack(out, axis=0).astype(np.uint8)


def transform_tracks_yx(tracks: np.ndarray, vis: np.ndarray, dx: np.ndarray, dy: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    # normalized yx coordinates use size-1 denominators in this project.
    out = tracks.astype(np.float32).copy()
    out[..., 0] += dy.reshape(1, -1) / max(H - 1, 1)
    out[..., 1] += dx.reshape(1, -1) / max(W - 1, 1)
    inside = (out[..., 0] >= 0.0) & (out[..., 0] <= 1.0) & (out[..., 1] >= 0.0) & (out[..., 1] <= 1.0)
    new_vis = vis.astype(bool) & inside
    return out.astype(np.float32), new_vis.astype(bool)


def transform_queries(q: np.ndarray, dx: np.ndarray, dy: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    q2 = q.astype(np.float32).copy()
    qt = np.rint(q2[:, 0]).astype(int)
    qt = np.clip(qt, 0, len(dx) - 1)
    q2[:, 1] += dy[qt] / max(H - 1, 1)
    q2[:, 2] += dx[qt] / max(W - 1, 1)
    inside = (q2[:, 1] >= 0.0) & (q2[:, 1] <= 1.0) & (q2[:, 2] >= 0.0) & (q2[:, 2] <= 1.0)
    return q2.astype(np.float32), inside.astype(bool)


def summarize_events(gt_vis: np.ndarray, query_points: np.ndarray) -> Dict[str, Any]:
    n, T = gt_vis.shape
    re_q = 0
    total_events = 0
    occ_lens: List[int] = []
    for i in range(n):
        qt = int(round(float(query_points[i, 0])))
        ev = find_reentry_events(gt_vis[i], qt)
        if ev:
            re_q += 1
            total_events += len(ev)
            occ_lens.extend([int(e['occ_length']) for e in ev])
    hist = {}
    for L in occ_lens:
        if L <= 4:
            b = '1-4'
        elif L <= 8:
            b = '5-8'
        elif L <= 16:
            b = '9-16'
        elif L <= 32:
            b = '17-32'
        else:
            b = '33+'
        hist[b] = hist.get(b, 0) + 1
    return {
        'num_queries': int(n),
        'num_reentry_queries': int(re_q),
        'reentry_query_rate': round(re_q / max(n, 1), 6),
        'num_reentry_events': int(total_events),
        'mean_reentry_events_per_query': round(total_events / max(n, 1), 6),
        'occ_length_mean': round(float(np.mean(occ_lens)), 6) if occ_lens else None,
        'occ_length_median': round(float(np.median(occ_lens)), 6) if occ_lens else None,
        'occ_length_histogram': hist,
    }


def draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], radius: int = 3) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, fill=color)


def make_gif(record: Dict[str, Any], out_path: Path, query_idx: int, max_frames: int = 80) -> None:
    video = record['video']
    gt = record['gt_tracks'][query_idx]
    vis = record['gt_visibility'][query_idx]
    qt = int(round(float(record['query_points'][query_idx, 0])))
    events = find_reentry_events(vis, qt)
    H, W = int(record['original_size'][0]), int(record['original_size'][1])
    frames = []
    T = video.shape[0]
    if events:
        e0 = events[0]
        start = max(0, int(e0['occ_start_t']) - 12)
        end = min(T, int(e0['reentry_frame']) + 24)
    else:
        start, end = 0, min(T, max_frames)
    if end - start > max_frames:
        end = start + max_frames
    re_frames = {int(e['reentry_frame']) for e in events}
    for t in range(start, end):
        im = Image.fromarray(video[t].astype(np.uint8)).convert('RGB')
        dr = ImageDraw.Draw(im)
        if bool(vis[t]):
            y = float(gt[t, 0]) * (H - 1)
            x = float(gt[t, 1]) * (W - 1)
            if 0 <= x <= W - 1 and 0 <= y <= H - 1:
                draw_point(dr, x, y, (0, 255, 0), radius=3)
        else:
            # draw a red frame if invisible
            dr.rectangle((1, 1, W - 2, H - 2), outline=(255, 0, 0), width=2)
        if t in re_frames:
            dr.rectangle((4, 4, W - 5, H - 5), outline=(255, 255, 0), width=3)
        dr.text((5, 5), f"t={t} q={query_idx} vis={int(bool(vis[t]))}", fill=(255, 255, 255))
        frames.append(im)
    if frames:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=80, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-cache', default=str(DEFAULT_VIDEO_CACHE))
    ap.add_argument('--gt-cache', default=str(DEFAULT_GT_CACHE))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--length', type=int, default=16)
    ap.add_argument('--amplitude-frac', type=float, default=0.4)
    ap.add_argument('--t0', type=int, default=40)
    ap.add_argument('--ramp', type=int, default=8)
    ap.add_argument('--fill-mode', choices=['frame_mean','gray','zero'], default='frame_mean')
    ap.add_argument('--num-gifs', type=int, default=12)
    args = ap.parse_args()

    video_cache = Path(args.video_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'sample_visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.gt_cache, map_location='cpu', weights_only=False)
    records_out: List[Dict[str, Any]] = []
    per_video = []
    total_original_queries = 0
    total_kept_queries = 0
    total_nan = 0
    total_visible_oob = 0
    gif_made = 0

    for rec in payload['records']:
        vid = str(rec['video_id'])
        npz = np.load(video_cache / f'{vid}.npz')
        video = np.asarray(npz['video']).astype(np.uint8)
        T, H, W, _ = video.shape
        gt = npy(rec['gt_tracks'], np.float32)
        vis = npy(rec['gt_visibility'], bool)
        q = npy(rec['query_points'], np.float32)
        dx = shift_curve(T, args.t0, args.ramp, args.length, args.amplitude_frac * W)
        dy = np.zeros_like(dx)
        stress_video = make_stress_video(video, dx, dy, fill_mode=args.fill_mode)
        stress_gt, stress_vis = transform_tracks_yx(gt, vis, dx, dy, H, W)
        stress_q, q_inside = transform_queries(q, dx, dy, H, W)
        qt = np.rint(stress_q[:, 0]).astype(int)
        qt = np.clip(qt, 0, T - 1)
        q_visible = stress_vis[np.arange(stress_vis.shape[0]), qt]
        keep = q_inside & q_visible
        total_original_queries += int(len(q))
        total_kept_queries += int(np.sum(keep))
        stress_gt = stress_gt[keep]
        stress_vis = stress_vis[keep]
        stress_q = stress_q[keep]
        source_indices = np.nonzero(keep)[0].astype(np.int32)
        nan_count = int(np.isnan(stress_gt).sum() + np.isnan(stress_q).sum())
        total_nan += nan_count
        visible = stress_vis
        if visible.any():
            vis_coords = stress_gt[visible]
            visible_oob = int(np.sum((vis_coords < -1e-5) | (vis_coords > 1.0 + 1e-5)))
        else:
            visible_oob = 0
        total_visible_oob += visible_oob
        ev_summary = summarize_events(stress_vis, stress_q) if len(stress_q) else {
            'num_queries': 0, 'num_reentry_queries': 0, 'reentry_query_rate': 0, 'num_reentry_events': 0,
            'mean_reentry_events_per_query': 0, 'occ_length_mean': None, 'occ_length_median': None, 'occ_length_histogram': {}
        }
        out_rec = {
            'video_id': f'{vid}_translate_L{args.length}',
            'source_video_id': vid,
            'sequence_index': int(rec.get('sequence_index', -1)),
            'frame_count': int(T),
            'video': stress_video.astype(np.uint8),
            'query_points': stress_q.astype(np.float32),
            'gt_tracks': stress_gt.astype(np.float32),
            'target_points': stress_gt.astype(np.float32),
            'gt_visibility': stress_vis.astype(bool),
            'occluded': (~stress_vis).astype(bool),
            'original_size': np.asarray([H, W], dtype=np.int32),
            'model_input_size': np.asarray([H, W], dtype=np.int32),
            'source_query_indices': source_indices,
            'stress_shift_xy_px': np.stack([dx, dy], axis=1).astype(np.float32),
            'stress_type': 'translate_exit_reenter',
            'stress_params': {
                'length': int(args.length), 'amplitude_frac': float(args.amplitude_frac), 't0': int(args.t0),
                'ramp': int(args.ramp), 'fill_mode': args.fill_mode,
            },
        }
        records_out.append(out_rec)
        pv = {
            'video_id': out_rec['video_id'],
            'source_video_id': vid,
            'frame_count': int(T),
            'original_queries': int(len(q)),
            'kept_queries': int(len(stress_q)),
            'query_keep_rate': round(len(stress_q) / max(len(q), 1), 6),
            'query_frame_visible_rate': 1.0 if len(stress_q) else None,
            'nan_count': nan_count,
            'visible_oob_count': visible_oob,
            **ev_summary,
        }
        per_video.append(pv)
        # gif samples: prefer reentry queries
        if gif_made < args.num_gifs and len(stress_q):
            re_idxs = []
            for i in range(len(stress_q)):
                if find_reentry_events(stress_vis[i], int(round(float(stress_q[i, 0])))):
                    re_idxs.append(i)
            for qi in re_idxs[: max(0, args.num_gifs - gif_made)]:
                make_gif(out_rec, vis_dir / f'{out_rec["video_id"]}_q{qi:04d}.gif', qi)
                gif_made += 1
                if gif_made >= args.num_gifs:
                    break
        print(json.dumps(pv, ensure_ascii=False), flush=True)

    # aggregate
    total_reentry_q = int(sum(v['num_reentry_queries'] for v in per_video))
    total_reentry_events = int(sum(v['num_reentry_events'] for v in per_video))
    occ_lens: List[int] = []
    for r in records_out:
        for i in range(len(r['query_points'])):
            ev = find_reentry_events(r['gt_visibility'][i], int(round(float(r['query_points'][i, 0]))))
            occ_lens.extend([int(e['occ_length']) for e in ev])
    hist = {}
    for L in occ_lens:
        hist[str(L)] = hist.get(str(L), 0) + 1
    sanity = {
        'stress_name': f'translate_exit_reenter_L{args.length}',
        'stress_type': 'translate_exit_reenter',
        'stress_params': {
            'length': int(args.length), 'amplitude_frac': float(args.amplitude_frac), 't0': int(args.t0),
            'ramp': int(args.ramp), 'fill_mode': args.fill_mode,
        },
        'num_videos': int(len(records_out)),
        'num_original_queries': int(total_original_queries),
        'num_kept_queries': int(total_kept_queries),
        'query_keep_rate': round(total_kept_queries / max(total_original_queries, 1), 6),
        'query_frame_visible_rate': 1.0 if total_kept_queries else None,
        'num_reentry_queries': int(total_reentry_q),
        'reentry_query_rate': round(total_reentry_q / max(total_kept_queries, 1), 6),
        'num_reentry_events': int(total_reentry_events),
        'mean_reentry_events_per_query': round(total_reentry_events / max(total_kept_queries, 1), 6),
        'occ_length_mean': round(float(np.mean(occ_lens)), 6) if occ_lens else None,
        'occ_length_median': round(float(np.median(occ_lens)), 6) if occ_lens else None,
        'occ_length_histogram_exact': hist,
        'nan_count': int(total_nan),
        'visible_oob_count': int(total_visible_oob),
        'sample_gifs': int(gif_made),
        'per_video': per_video,
        'sanity_pass': bool(total_kept_queries > 0 and total_nan == 0 and total_visible_oob == 0 and (total_reentry_q / max(total_kept_queries, 1)) >= 0.2),
    }
    stress_payload = {
        'schema_version': 1,
        'dataset_name': 'reentry_stress_rgb_dev10',
        'stress_name': sanity['stress_name'],
        'stress_type': sanity['stress_type'],
        'stress_params': sanity['stress_params'],
        'source_cache': str(args.gt_cache),
        'source_video_cache': str(video_cache),
        'records': records_out,
    }
    torch.save(stress_payload, out_dir / 'stress_dataset.pt')
    (out_dir / 'sanity_summary.json').write_text(json.dumps(sanity, indent=2, ensure_ascii=False))
    manifest = {
        'stress_dataset': str(out_dir / 'stress_dataset.pt'),
        'sanity_summary': str(out_dir / 'sanity_summary.json'),
        'sample_visualizations': str(vis_dir),
        'sanity_pass': sanity['sanity_pass'],
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('STRESS_OK', json.dumps(sanity, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
