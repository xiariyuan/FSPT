#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset
from utils.coords import find_reentry_events


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def sample_video_uint8(sample: Dict[str, Any]) -> np.ndarray:
    v = sample['video']
    if isinstance(v, torch.Tensor):
        arr = v.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[1] == 3:
            arr = np.transpose(arr, (0, 2, 3, 1))
    else:
        arr = np.asarray(v)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


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
    dx = int(dx); dy = int(dy)
    src_x0 = max(0, -dx); src_x1 = min(W, W - dx)
    dst_x0 = max(0, dx); dst_x1 = min(W, W + dx)
    src_y0 = max(0, -dy); src_y1 = min(H, H - dy)
    dst_y0 = max(0, dy); dst_y1 = min(H, H + dy)
    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def make_translate_video(video: np.ndarray, dx: np.ndarray, dy: np.ndarray, fill_mode: str) -> np.ndarray:
    return np.stack([translate_frame(video[t], int(round(float(dx[t]))), int(round(float(dy[t]))), fill_mode) for t in range(video.shape[0])], axis=0).astype(np.uint8)


def transform_tracks_yx(tracks: np.ndarray, vis: np.ndarray, dx: np.ndarray, dy: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    out = tracks.astype(np.float32).copy()
    out[..., 0] += dy.reshape(1, -1) / max(H - 1, 1)
    out[..., 1] += dx.reshape(1, -1) / max(W - 1, 1)
    inside = (out[..., 0] >= 0.0) & (out[..., 0] <= 1.0) & (out[..., 1] >= 0.0) & (out[..., 1] <= 1.0)
    return out.astype(np.float32), (vis.astype(bool) & inside).astype(bool)


def transform_queries(q: np.ndarray, dx: np.ndarray, dy: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    q2 = q.astype(np.float32).copy()
    qt = np.rint(q2[:, 0]).astype(int)
    qt = np.clip(qt, 0, len(dx) - 1)
    q2[:, 1] += dy[qt] / max(H - 1, 1)
    q2[:, 2] += dx[qt] / max(W - 1, 1)
    inside = (q2[:, 1] >= 0.0) & (q2[:, 1] <= 1.0) & (q2[:, 2] >= 0.0) & (q2[:, 2] <= 1.0)
    return q2.astype(np.float32), inside.astype(bool)


def occluder_rects(T: int, H: int, W: int, length: int, t0: int, speed_px: float) -> np.ndarray:
    width = max(4, int(round(float(length) * float(speed_px))))
    rects = np.zeros((T, 4), dtype=np.float32)
    sweep_frames = int(np.ceil((W + width) / max(float(speed_px), 1e-6)))
    for t in range(T):
        u = t - int(t0)
        if u < 0 or u > sweep_frames:
            continue
        x0 = -width + float(speed_px) * u
        rects[t] = [x0, 0.0, x0 + width, float(H)]
    return rects


def overlay_occluder(video: np.ndarray, rects: np.ndarray, fill_mode: str) -> np.ndarray:
    out = video.copy().astype(np.uint8)
    T, H, W, _ = out.shape
    for t in range(T):
        x0, y0, x1, y1 = rects[t]
        ix0 = max(0, int(np.floor(x0))); ix1 = min(W, int(np.ceil(x1)))
        iy0 = max(0, int(np.floor(y0))); iy1 = min(H, int(np.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if fill_mode == 'gray':
            fill = np.full((1, 1, 3), 127, dtype=np.uint8)
        elif fill_mode == 'zero':
            fill = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            fill = np.asarray(np.mean(out[t].reshape(-1, 3), axis=0), dtype=np.uint8).reshape(1, 1, 3)
        out[t, iy0:iy1, ix0:ix1] = fill
    return out


def apply_occluder_visibility(tracks: np.ndarray, vis: np.ndarray, rects: np.ndarray, H: int, W: int) -> np.ndarray:
    y_px = tracks[..., 0] * max(H - 1, 1)
    x_px = tracks[..., 1] * max(W - 1, 1)
    new_vis = vis.astype(bool).copy()
    for t in range(tracks.shape[1]):
        x0, y0, x1, y1 = rects[t]
        if x1 <= x0:
            continue
        covered = (x_px[:, t] >= x0) & (x_px[:, t] <= x1) & (y_px[:, t] >= y0) & (y_px[:, t] <= y1)
        new_vis[:, t] &= ~covered
    return new_vis.astype(bool)


def summarize_events(gt_vis: np.ndarray, q: np.ndarray) -> Dict[str, Any]:
    n = int(gt_vis.shape[0])
    re_q = events = 0
    lens = []
    for i in range(n):
        ev = find_reentry_events(gt_vis[i], int(round(float(q[i, 0]))))
        if ev:
            re_q += 1
            events += len(ev)
            lens.extend(int(e['occ_length']) for e in ev)
    hist = {}
    for L in lens:
        if L <= 4: b = '1-4'
        elif L <= 8: b = '5-8'
        elif L <= 16: b = '9-16'
        elif L <= 32: b = '17-32'
        else: b = '33+'
        hist[b] = hist.get(b, 0) + 1
    return {
        'num_queries': n,
        'num_reentry_queries': int(re_q),
        'reentry_query_rate': round(re_q / max(n, 1), 6),
        'num_reentry_events': int(events),
        'mean_reentry_events_per_query': round(events / max(n, 1), 6),
        'occ_length_mean': round(float(np.mean(lens)), 6) if lens else None,
        'occ_length_median': round(float(np.median(lens)), 6) if lens else None,
        'occ_length_histogram': hist,
    }


def make_gif(record: Dict[str, Any], out_path: Path, qi: int, max_frames: int = 80) -> None:
    video = record['video']
    gt = record['gt_tracks'][qi]
    vis = record['gt_visibility'][qi]
    qt = int(round(float(record['query_points'][qi, 0])))
    evs = find_reentry_events(vis, qt)
    if not evs:
        return
    H, W = [int(x) for x in record['original_size']]
    e0 = evs[0]
    start = max(0, int(e0['occ_start_t']) - 12)
    end = min(video.shape[0], int(e0['reentry_frame']) + 24)
    if end - start > max_frames:
        end = start + max_frames
    re_frames = {int(e['reentry_frame']) for e in evs}
    frames = []
    for t in range(start, end):
        im = Image.fromarray(video[t].astype(np.uint8)).convert('RGB')
        dr = ImageDraw.Draw(im)
        if bool(vis[t]):
            y = float(gt[t, 0]) * (H - 1); x = float(gt[t, 1]) * (W - 1)
            if 0 <= x <= W - 1 and 0 <= y <= H - 1:
                dr.ellipse((x-3, y-3, x+3, y+3), outline=(0,255,0), fill=(0,255,0))
        else:
            dr.rectangle((1, 1, W - 2, H - 2), outline=(255,0,0), width=2)
        if t in re_frames:
            dr.rectangle((4,4,W-5,H-5), outline=(255,255,0), width=3)
        dr.text((5,5), f't={t} q={qi} vis={int(bool(vis[t]))}', fill=(255,255,255))
        frames.append(im)
    if frames:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=80, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-cache', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--stress-type', choices=['translate_exit_reenter','moving_occluder'], required=True)
    ap.add_argument('--data-root', default='/gemini/code/datasets/tapvid_rgb_stacking')
    ap.add_argument('--start-index', type=int, default=20)
    ap.add_argument('--num-videos', type=int, default=10)
    ap.add_argument('--length', type=int, default=16)
    ap.add_argument('--t0', type=int, default=40)
    ap.add_argument('--ramp', type=int, default=8)
    ap.add_argument('--amplitude-frac', type=float, default=0.4)
    ap.add_argument('--speed-px', type=float, default=4.0)
    ap.add_argument('--fill-mode', choices=['frame_mean','gray','zero'], default='frame_mean')
    ap.add_argument('--num-gifs', type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'sample_visualizations'; vis_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.source_cache, map_location='cpu', weights_only=False)
    ds = TAPVidRGBStackingDataset(root=args.data_root, query_mode='strided', query_stride=5, start_index=args.start_index, num_videos=args.num_videos)
    videos = {}
    for i in range(len(ds)):
        s = ds[i]
        videos[str(s['video_name'])] = sample_video_uint8(s)

    records_out: List[Dict[str, Any]] = []
    per_video = []
    total_original = total_kept = total_nan = total_visible_oob = gif_made = 0
    for rec in payload['records']:
        vid = str(rec['video_id'])
        if vid not in videos:
            raise KeyError(f'video {vid} not found in dataset subset')
        video = videos[vid]
        T, H, W, _ = video.shape
        gt = npy(rec['gt_tracks'], np.float32)
        vis = npy(rec['gt_visibility'], bool)
        q = npy(rec['query_points'], np.float32)
        if args.stress_type == 'translate_exit_reenter':
            dx = shift_curve(T, args.t0, args.ramp, args.length, args.amplitude_frac * W)
            dy = np.zeros_like(dx)
            stress_video = make_translate_video(video, dx, dy, args.fill_mode)
            stress_gt, stress_vis = transform_tracks_yx(gt, vis, dx, dy, H, W)
            stress_q, q_inside = transform_queries(q, dx, dy, H, W)
            extra = {'stress_shift_xy_px': np.stack([dx, dy], axis=1).astype(np.float32)}
            stress_name = f'translate_exit_reenter_L{args.length}'
        else:
            rects = occluder_rects(T, H, W, args.length, args.t0, args.speed_px)
            stress_video = overlay_occluder(video, rects, args.fill_mode)
            stress_gt = gt.astype(np.float32).copy()
            stress_vis = apply_occluder_visibility(stress_gt, vis, rects, H, W)
            stress_q = q.astype(np.float32).copy()
            q_inside = (stress_q[:,1] >= 0) & (stress_q[:,1] <= 1) & (stress_q[:,2] >= 0) & (stress_q[:,2] <= 1)
            extra = {'occluder_rect_xyxy_px': rects.astype(np.float32)}
            stress_name = f'moving_occluder_L{args.length}'
        qt = np.rint(stress_q[:,0]).astype(int)
        qt = np.clip(qt, 0, T - 1)
        q_visible = stress_vis[np.arange(stress_vis.shape[0]), qt]
        keep = q_inside & q_visible
        total_original += int(len(q))
        total_kept += int(np.sum(keep))
        stress_gt = stress_gt[keep]
        stress_vis = stress_vis[keep]
        stress_q = stress_q[keep]
        source_indices = np.nonzero(keep)[0].astype(np.int32)
        nan_count = int(np.isnan(stress_gt).sum() + np.isnan(stress_q).sum())
        total_nan += nan_count
        if stress_vis.any():
            vc = stress_gt[stress_vis]
            visible_oob = int(np.sum((vc < -1e-5) | (vc > 1.0 + 1e-5)))
        else:
            visible_oob = 0
        total_visible_oob += visible_oob
        ev_summary = summarize_events(stress_vis, stress_q) if len(stress_q) else {'num_queries':0,'num_reentry_queries':0,'reentry_query_rate':0,'num_reentry_events':0,'mean_reentry_events_per_query':0,'occ_length_mean':None,'occ_length_median':None,'occ_length_histogram':{}}
        suffix = 'translate' if args.stress_type == 'translate_exit_reenter' else 'occluder'
        out_rec = {
            'video_id': f'{vid}_{suffix}_L{args.length}',
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
            'stress_type': args.stress_type,
            'stress_params': vars(args),
            **extra,
        }
        records_out.append(out_rec)
        pv = {'video_id': out_rec['video_id'], 'source_video_id': vid, 'frame_count': int(T), 'original_queries': int(len(q)), 'kept_queries': int(len(stress_q)), 'query_keep_rate': round(len(stress_q)/max(len(q),1),6), 'query_frame_visible_rate': 1.0 if len(stress_q) else None, 'nan_count': nan_count, 'visible_oob_count': visible_oob, **ev_summary}
        per_video.append(pv)
        if gif_made < args.num_gifs and len(stress_q):
            re_idxs = [i for i in range(len(stress_q)) if find_reentry_events(stress_vis[i], int(round(float(stress_q[i,0]))))]
            for qi in re_idxs[:max(0, args.num_gifs - gif_made)]:
                make_gif(out_rec, vis_dir / f'{out_rec["video_id"]}_q{qi:04d}.gif', qi)
                gif_made += 1
                if gif_made >= args.num_gifs:
                    break
        print(json.dumps(pv, ensure_ascii=False), flush=True)
    occ_lens = []
    for r in records_out:
        for i in range(len(r['query_points'])):
            ev = find_reentry_events(r['gt_visibility'][i], int(round(float(r['query_points'][i,0]))))
            occ_lens.extend(int(e['occ_length']) for e in ev)
    total_re_q = int(sum(v['num_reentry_queries'] for v in per_video))
    total_events = int(sum(v['num_reentry_events'] for v in per_video))
    sanity = {
        'stress_name': stress_name,
        'stress_type': args.stress_type,
        'stress_params': vars(args),
        'num_videos': int(len(records_out)),
        'num_original_queries': int(total_original),
        'num_kept_queries': int(total_kept),
        'query_keep_rate': round(total_kept/max(total_original,1),6),
        'query_frame_visible_rate': 1.0 if total_kept else None,
        'num_reentry_queries': total_re_q,
        'reentry_query_rate': round(total_re_q/max(total_kept,1),6),
        'num_reentry_events': total_events,
        'mean_reentry_events_per_query': round(total_events/max(total_kept,1),6),
        'occ_length_mean': round(float(np.mean(occ_lens)),6) if occ_lens else None,
        'occ_length_median': round(float(np.median(occ_lens)),6) if occ_lens else None,
        'nan_count': int(total_nan),
        'visible_oob_count': int(total_visible_oob),
        'sample_gifs': int(gif_made),
        'per_video': per_video,
        'sanity_pass': bool(total_kept > 0 and total_nan == 0 and total_visible_oob == 0 and (total_re_q / max(total_kept,1)) >= 0.2),
    }
    stress_payload = {'schema_version':1, 'dataset_name':'reentry_stress_rgb_from_cache', 'stress_name':stress_name, 'stress_type':args.stress_type, 'stress_params':vars(args), 'source_cache':str(args.source_cache), 'records':records_out}
    torch.save(stress_payload, out_dir / 'stress_dataset.pt')
    (out_dir / 'sanity_summary.json').write_text(json.dumps(sanity, indent=2, ensure_ascii=False))
    (out_dir / 'manifest.json').write_text(json.dumps({'stress_dataset':str(out_dir/'stress_dataset.pt'), 'sanity_summary':str(out_dir/'sanity_summary.json'), 'sample_visualizations':str(vis_dir), 'sanity_pass':sanity['sanity_pass']}, indent=2, ensure_ascii=False))
    print('STRESS_OK', json.dumps(sanity, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
