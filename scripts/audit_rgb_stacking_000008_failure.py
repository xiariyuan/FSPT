#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, yx_norm_to_xy_256

ROOT = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke')
OUT = ROOT / 'rgb_000008_failure_audit'
DATASET = Path('/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl')
CACHES = {
    'offline': ROOT / 'cotracker3_offline_rgb_stacking_10video.pt',
    'online': ROOT / 'cotracker3_online_rgb_stacking_10video.pt',
    'b2': ROOT / 'b2_offline_base_online_override_10video' / 'b2_rgb10_offline_base_online_override.pt',
}
VID = 'rgb_stacking_000008'
VIDX = 8
THRESHOLDS = (1, 2, 4, 8, 16)
COLORS = {
    'GT': (0, 255, 0),
    'offline': (255, 220, 0),
    'online': (0, 180, 255),
    'B2': (255, 60, 255),
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = int(t) - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def first_trigger(base_v: np.ndarray, over_v: np.ndarray, query_t: int, k: int = 1) -> Optional[int]:
    for t in range(max(1, int(query_t) + 1), len(base_v)):
        if invisible_run_before(base_v, t) >= k and bool(over_v[t]):
            return int(t)
    return None


def point_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, start_t: int = 0) -> float:
    t_len = gt_vis.shape[0]
    mask = np.ones(t_len, dtype=bool)
    qt = max(0, min(t_len - 1, int(query_t)))
    mask[qt] = False
    if start_t > 0:
        mask[: int(start_t)] = False
    if not np.any(mask):
        return 0.0
    pred_px = yx_norm_to_xy_256(np.asarray(pred_tracks, dtype=np.float32))
    gt_px = yx_norm_to_xy_256(np.asarray(gt_tracks, dtype=np.float32))
    sq = np.sum((pred_px - gt_px) ** 2, axis=-1)
    vals = []
    for thr in THRESHOLDS:
        within = sq < float(thr) ** 2
        gt_pos = gt_vis.astype(bool)
        pv = pred_vis.astype(bool)
        tp = float(np.sum(mask & within & gt_pos & pv))
        gp = float(np.sum(mask & gt_pos))
        fp = float(np.sum(mask & pv & ((~gt_pos) | (~within))))
        vals.append(tp / (gp + fp) if (gp + fp) > 0 else 0.0)
    return float(np.mean(vals))


def reentry_segment_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, reentry_t: int) -> float:
    return point_aj(pred_tracks, gt_tracks, pred_vis, gt_vis, query_t=reentry_t, start_t=reentry_t)


def px_error_at(pred_tracks: np.ndarray, gt_tracks: np.ndarray, t: int) -> Optional[float]:
    if t < 0 or t >= pred_tracks.shape[0]:
        return None
    p = yx_norm_to_xy_256(pred_tracks[t][None, :])[0]
    g = yx_norm_to_xy_256(gt_tracks[t][None, :])[0]
    return float(np.linalg.norm(p - g))


def get_font(size: int = 15):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def yx_to_xy_img(pt_yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    y = float(pt_yx[0]) * max(h - 1, 1)
    x = float(pt_yx[1]) * max(w - 1, 1)
    return x, y


def draw_point(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], color: Tuple[int, int, int], label: str, r: int = 5) -> None:
    x, y = xy
    draw.ellipse((x-r, y-r, x+r, y+r), outline=(0,0,0), width=4)
    draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=3)
    draw.text((x+r+2, y-r-2), label, fill=color, stroke_width=2, stroke_fill=(0,0,0), font=get_font(12))


def draw_trail(draw: ImageDraw.ImageDraw, pts: np.ndarray, vis: np.ndarray, h: int, w: int, t: int, color: Tuple[int,int,int], width: int = 2) -> None:
    coords = []
    for tt in range(max(0, t-10), t+1):
        if bool(vis[tt]):
            coords.append(yx_to_xy_img(pts[tt], h, w))
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=width)


def render_case(case: Dict[str, Any], video: np.ndarray, recs: Dict[str, Dict[str, Any]]) -> str:
    qi = int(case['query_idx'])
    h, w = int(video.shape[1]), int(video.shape[2])
    T = int(video.shape[0])
    qt = int(case['query_t'])
    trigger = case.get('trigger_t')
    reentry = case.get('first_reentry_t')
    frames = [qt]
    if trigger is not None:
        frames += [max(0, trigger-1), trigger, min(T-1, trigger+8)]
    if reentry is not None:
        frames += [max(0, reentry-1), reentry, min(T-1, reentry+8)]
    frames.append(T-1)
    frames = sorted(set(int(max(0, min(T-1, f))) for f in frames))
    if len(frames) > 6:
        priority = [qt, trigger, reentry, (reentry+8 if reentry is not None else None), (trigger+8 if trigger is not None else None), T-1]
        frames = []
        for f in priority:
            if f is not None:
                ff = int(max(0, min(T-1, f)))
                if ff not in frames:
                    frames.append(ff)
        frames = sorted(frames[:6])
    panels = []
    font = get_font(14)
    small = get_font(12)
    track_sets = {
        'GT': (npy(recs['offline']['gt_tracks'], np.float32)[qi], npy(recs['offline']['gt_visibility'], bool)[qi]),
        'offline': (npy(recs['offline']['pred_tracks'], np.float32)[qi], npy(recs['offline']['pred_visibility'], bool)[qi]),
        'online': (npy(recs['online']['pred_tracks'], np.float32)[qi], npy(recs['online']['pred_visibility'], bool)[qi]),
        'B2': (npy(recs['b2']['pred_tracks'], np.float32)[qi], npy(recs['b2']['pred_visibility'], bool)[qi]),
    }
    for f in frames:
        img = Image.fromarray(video[f]).convert('RGB')
        draw = ImageDraw.Draw(img)
        flags = []
        if f == qt: flags.append('query')
        if trigger is not None and f == trigger: flags.append('B2 trigger')
        if reentry is not None and f == reentry: flags.append('GT re-entry')
        title = f"{VID} q={qi} t={f}" + (" | " + ", ".join(flags) if flags else "")
        draw.rectangle((0,0,w,26), fill=(0,0,0))
        draw.text((6,5), title, fill=(255,255,255), font=font)
        for name, (pts, vis) in track_sets.items():
            draw_trail(draw, pts, vis, h, w, f, COLORS[name])
        for name, (pts, vis) in track_sets.items():
            if bool(vis[f]):
                draw_point(draw, yx_to_xy_img(pts[f], h, w), COLORS[name], name)
        draw.rectangle((0,h-22,w,h), fill=(0,0,0))
        draw.text((6,h-18), 'GT green | offline yellow | online cyan | B2 magenta', fill=(255,255,255), font=small)
        panels.append(img)
    gap = 8
    cw = sum(p.width for p in panels) + gap * (len(panels)-1)
    ch = max(p.height for p in panels) + 120
    canvas = Image.new('RGB', (cw, ch), (20,20,20))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width + gap
    draw = ImageDraw.Draw(canvas)
    y = max(p.height for p in panels) + 10
    lines = [
        f"category={case['category']} query={qi} trigger={trigger} reentry={reentry}",
        f"offline_AJ={case['offline_AJ']:.3f} online_AJ={case['online_AJ']:.3f} B2_AJ={case['b2_AJ']:.3f} B2-offline={case['delta_b2_offline_full']:+.3f}",
        f"offline_reAJ={case.get('offline_reentry_AJ')} online_reAJ={case.get('online_reentry_AJ')} B2_reAJ={case.get('b2_reentry_AJ')} delta_re={case.get('delta_b2_offline_reentry')}",
        f"reason={case.get('reason','')}",
    ]
    for line in lines:
        draw.text((10,y), line, fill=(255,255,255), font=font)
        y += 24
    OUT.joinpath('images').mkdir(parents=True, exist_ok=True)
    fn = f"{case['category']}__{VID}__q{qi}.png"
    path = OUT / 'images' / fn
    canvas.save(path)
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in CACHES.items()}
    recs = {k: payloads[k]['records'][VIDX] for k in payloads}
    assert str(recs['offline']['video_id']) == VID
    with DATASET.open('rb') as f:
        data = pickle.load(f)
    video = np.asarray(data[VIDX]['video'], dtype=np.uint8)

    base_v = npy(recs['offline']['pred_visibility'], bool)
    over_v = npy(recs['online']['pred_visibility'], bool)
    gt_v = npy(recs['offline']['gt_visibility'], bool)
    qpts = npy(recs['offline']['query_points'], np.float32)
    rows: List[Dict[str, Any]] = []
    for qi in range(gt_v.shape[0]):
        qt = int(round(float(qpts[qi,0])))
        events = find_reentry_events(gt_v[qi], qt)
        trig = first_trigger(base_v[qi], over_v[qi], qt)
        has_re = bool(events)
        offline_aj = point_aj(npy(recs['offline']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], base_v[qi], gt_v[qi], qt)
        online_aj = point_aj(npy(recs['online']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], over_v[qi], gt_v[qi], qt)
        b2_aj = point_aj(npy(recs['b2']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], npy(recs['b2']['pred_visibility'],bool)[qi], gt_v[qi], qt)
        row: Dict[str, Any] = {
            'video_id': VID,
            'query_idx': qi,
            'query_t': qt,
            'has_gt_reentry': has_re,
            'first_reentry_t': int(events[0]['reentry_frame']) if events else None,
            'first_occ_length': int(events[0]['occ_length']) if events else None,
            'triggered': trig is not None,
            'trigger_t': trig,
            'class': 'true_trigger' if (trig is not None and has_re) else ('false_trigger' if trig is not None else ('missed_reentry' if has_re else 'none')),
            'gt_visible_at_trigger': bool(gt_v[qi,trig]) if trig is not None else None,
            'offline_visible_at_trigger': bool(base_v[qi,trig]) if trig is not None else None,
            'online_visible_at_trigger': bool(over_v[qi,trig]) if trig is not None else None,
            'offline_AJ': offline_aj,
            'online_AJ': online_aj,
            'b2_AJ': b2_aj,
            'delta_online_offline_full': online_aj - offline_aj,
            'delta_b2_offline_full': b2_aj - offline_aj,
        }
        if has_re:
            rt = int(events[0]['reentry_frame'])
            off_re = reentry_segment_aj(npy(recs['offline']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], base_v[qi], gt_v[qi], rt)
            on_re = reentry_segment_aj(npy(recs['online']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], over_v[qi], gt_v[qi], rt)
            b2_re = reentry_segment_aj(npy(recs['b2']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], npy(recs['b2']['pred_visibility'],bool)[qi], gt_v[qi], rt)
            row.update({
                'offline_reentry_AJ': round(off_re,6),
                'online_reentry_AJ': round(on_re,6),
                'b2_reentry_AJ': round(b2_re,6),
                'delta_online_offline_reentry': round(on_re - off_re,6),
                'delta_b2_offline_reentry': round(b2_re - off_re,6),
                'offline_err_at_reentry': px_error_at(npy(recs['offline']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], rt),
                'online_err_at_reentry': px_error_at(npy(recs['online']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], rt),
                'b2_err_at_reentry': px_error_at(npy(recs['b2']['pred_tracks'],np.float32)[qi], npy(recs['offline']['gt_tracks'],np.float32)[qi], rt),
            })
        rows.append(row)

    false_rows = [r for r in rows if r['class']=='false_trigger']
    true_rows = [r for r in rows if r['class']=='true_trigger']
    reentry_rows = [r for r in rows if r['has_gt_reentry']]
    summary = {
        'video_id': VID,
        'n_tracks': len(rows),
        'n_reentry': len(reentry_rows),
        'n_triggers': sum(1 for r in rows if r['triggered']),
        'n_true_triggers': len(true_rows),
        'n_false_triggers': len(false_rows),
        'n_missed_reentry': sum(1 for r in rows if r['class']=='missed_reentry'),
        'trigger_precision': round(len(true_rows)/max(sum(1 for r in rows if r['triggered']),1),6),
        'trigger_recall': round(len(true_rows)/max(len(reentry_rows),1),6),
        'false_trigger_rate': round(len(false_rows)/max(len(rows),1),6),
        'mean_delta_b2_offline_full_false': round(float(np.mean([r['delta_b2_offline_full'] for r in false_rows])),6) if false_rows else None,
        'mean_delta_b2_offline_full_true': round(float(np.mean([r['delta_b2_offline_full'] for r in true_rows])),6) if true_rows else None,
        'mean_delta_b2_offline_reentry_true': round(float(np.mean([r.get('delta_b2_offline_reentry',0.0) for r in true_rows])),6) if true_rows else None,
        'worst_false_triggers': sorted(false_rows, key=lambda r:r['delta_b2_offline_full'])[:20],
        'worst_true_reentry_deltas': sorted(true_rows, key=lambda r:r.get('delta_b2_offline_reentry',0.0))[:20],
        'best_true_reentry_gains': sorted(true_rows, key=lambda r:r.get('delta_b2_offline_reentry',0.0), reverse=True)[:20],
    }

    # Render representative cases.
    cases = []
    for r in summary['worst_true_reentry_deltas'][:3]:
        c = dict(r); c['category']='bad_true_reentry_override'; c['reason']='B2/online re-entry segment worse than offline'; cases.append(c)
    for r in summary['worst_false_triggers'][:3]:
        c = dict(r); c['category']='bad_false_trigger'; c['reason']='false trigger hurts full-track AJ'; cases.append(c)
    for r in summary['best_true_reentry_gains'][:3]:
        c = dict(r); c['category']='good_true_reentry_recovery'; c['reason']='B2/online improves re-entry segment'; cases.append(c)
    rendered = []
    for c in cases:
        try:
            c['image'] = render_case(c, video, recs)
        except Exception as e:
            c['render_error'] = repr(e)
        rendered.append(c)
    summary['rendered_cases'] = rendered

    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT/'track_rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['video_id','n_tracks','n_reentry','n_triggers','n_true_triggers','n_false_triggers','n_missed_reentry','trigger_precision','trigger_recall','false_trigger_rate','mean_delta_b2_offline_full_false','mean_delta_b2_offline_full_true','mean_delta_b2_offline_reentry_true']}, indent=2))
    print('rendered_images')
    for c in rendered:
        print(c.get('category'), c.get('query_idx'), c.get('image', c.get('render_error')))


if __name__ == '__main__':
    main()
