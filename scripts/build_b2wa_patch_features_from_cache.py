#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROWS = Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/rgb_dev10_window_rows.jsonl')
VIDEO_CACHE = Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
BASE_CACHE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
OVER_CACHE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
OUT_DIR = Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot')
LAM = 'lambda_1'
RADII = [5, 9, 17]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[Dict[str, Any]], max_rows: int, seed: int = 123) -> List[Dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        best = r['best_action'][LAM]
        re = 're' if r.get('window_has_reentry') else 'nore'
        harmful = 'harm' if float(r['actions']['W16']['delta_full_AJ']) < -0.05 else 'ok'
        key = f'{best}_{re}_{harmful}'
        buckets.setdefault(key, []).append(r)
    keys = sorted(buckets)
    per = max(1, max_rows // max(len(keys), 1))
    out = []
    for k in keys:
        b = buckets[k]
        idx = rng.permutation(len(b))[:min(len(b), per)]
        out.extend([b[i] for i in idx])
    if len(out) < max_rows:
        used = {(r['video_id'], r['query_idx'], r['trigger_t']) for r in out}
        rem = [r for r in rows if (r['video_id'], r['query_idx'], r['trigger_t']) not in used]
        idx = rng.permutation(len(rem))[:max_rows - len(out)]
        out.extend([rem[i] for i in idx])
    rng.shuffle(out)
    return out[:max_rows]


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def yx_to_pixel(yx, h: int, w: int) -> Tuple[int, int]:
    y = int(round(float(yx[0]) * (h - 1)))
    x = int(round(float(yx[1]) * (w - 1)))
    return max(0, min(h - 1, y)), max(0, min(w - 1, x))


def patch(frame: np.ndarray, yx, r: int) -> np.ndarray:
    fr = frame.astype(np.float32)
    if fr.max() > 1.5:
        fr = fr / 255.0
    h, w = fr.shape[:2]
    y, x = yx_to_pixel(yx, h, w)
    pad = int(r)
    padded = np.pad(fr, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    yp, xp = y + pad, x + pad
    return padded[yp - pad:yp + pad + 1, xp - pad:xp + pad + 1]


def patch_at_offset(frame: np.ndarray, yx, r: int, dy: int, dx: int) -> np.ndarray:
    h, w = frame.shape[:2]
    y, x = yx_to_pixel(yx, h, w)
    yy = max(0, min(h - 1, y + int(dy))) / max(h - 1, 1)
    xx = max(0, min(w - 1, x + int(dx))) / max(w - 1, 1)
    return patch(frame, [yy, xx], r)


def hist_desc(p: np.ndarray, bins: int = 8) -> np.ndarray:
    vals = []
    for c in range(3):
        h, _ = np.histogram(p[..., c], bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        h = h / max(float(h.sum()), 1e-6)
        vals.append(h)
    return np.concatenate(vals)


def desc(p: np.ndarray) -> np.ndarray:
    flat = p.reshape(-1, 3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    hist = hist_desc(p, 8)
    gray = p.mean(axis=2)
    gx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    grad = np.asarray([float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy)))], dtype=np.float32)
    return np.concatenate([mean, std, hist, grad]).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def ncc(pa: np.ndarray, pb: np.ndarray) -> float:
    a = pa.mean(axis=2).reshape(-1).astype(np.float32)
    b = pb.mean(axis=2).reshape(-1).astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)


def last_visible_before(v: np.ndarray, t: int):
    for j in range(int(t) - 1, -1, -1):
        if bool(v[j]):
            return int(j)
    return None


def features_for_radius(video: np.ndarray, base_tr: np.ndarray, base_v: np.ndarray, over_tr: np.ndarray, query_t: int, trigger_t: int, r: int) -> Dict[str, float]:
    t = int(trigger_t)
    q = int(query_t)
    lv = last_visible_before(base_v, t)
    frame_q = video[q]
    frame_t = video[t]
    pq = patch(frame_q, base_tr[q], r)
    pc = patch(frame_t, over_tr[t], r)
    p_last = patch(video[lv], base_tr[lv], r) if lv is not None else pq
    dq, dc, dl = desc(pq), desc(pc), desc(p_last)
    off = max(2 * r, 8)
    neg_patches = [patch_at_offset(frame_t, over_tr[t], r, dy, dx) for dy, dx in [(off, 0), (-off, 0), (0, off), (0, -off), (off, off), (-off, -off), (off, -off), (-off, off)]]
    neg_desc = [desc(p) for p in neg_patches]
    q_c = cos(dq, dc)
    l_c = cos(dl, dc)
    q_neg = max(cos(dq, nd) for nd in neg_desc)
    l_neg = max(cos(dl, nd) for nd in neg_desc)
    return {
        f'r{r}_query_candidate_cos': q_c,
        f'r{r}_last_candidate_cos': l_c,
        f'r{r}_query_candidate_l2': l2(dq, dc),
        f'r{r}_last_candidate_l2': l2(dl, dc),
        f'r{r}_query_candidate_ncc': ncc(pq, pc),
        f'r{r}_last_candidate_ncc': ncc(p_last, pc),
        f'r{r}_query_margin_cos': q_c - q_neg,
        f'r{r}_last_margin_cos': l_c - l_neg,
        f'r{r}_candidate_grad_mean': float(dc[-2:].mean()),
        f'r{r}_last_visible_age': float(t - lv) if lv is not None else float(t - q),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=str(ROWS))
    ap.add_argument('--video-cache', default=str(VIDEO_CACHE))
    ap.add_argument('--base-cache', default=str(BASE_CACHE))
    ap.add_argument('--override-cache', default=str(OVER_CACHE))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--max-rows', type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = stratified_sample(load_rows(Path(args.rows)), int(args.max_rows))
    base = torch.load(args.base_cache, map_location='cpu', weights_only=False)
    over = torch.load(args.override_cache, map_location='cpu', weights_only=False)
    base_by_vid = {str(r['video_id']): r for r in base['records']}
    over_by_vid = {str(r['video_id']): r for r in over['records']}
    video_cache = Path(args.video_cache)
    videos: Dict[str, np.ndarray] = {}
    out_rows = []
    for idx, row in enumerate(rows):
        vid = str(row['video_id'])
        if vid not in videos:
            npz = np.load(video_cache / f'{vid}.npz')
            videos[vid] = np.asarray(npz['video'])
        video = videos[vid]
        b = base_by_vid[vid]
        o = over_by_vid[vid]
        qi = int(row['query_idx'])
        base_tr = npy(b['pred_tracks'], np.float32)[qi]
        base_v = npy(b['pred_visibility'], bool)[qi]
        over_tr = npy(o['pred_tracks'], np.float32)[qi]
        app = {}
        for r in RADII:
            app.update(features_for_radius(video, base_tr, base_v, over_tr, int(row['query_t']), int(row['trigger_t']), r))
        row2 = {
            'dataset': row['dataset'],
            'video_id': row['video_id'],
            'record_index': int(row['record_index']),
            'query_idx': int(row['query_idx']),
            'query_t': int(row['query_t']),
            'trigger_t': int(row['trigger_t']),
            'window_has_reentry': bool(row['window_has_reentry']),
            'eligible_window_has_reentry': bool(row['eligible_window_has_reentry']),
            'best_action': row['best_action'][LAM],
            'w16_delta_full_AJ': float(row['actions']['W16']['delta_full_AJ']),
            'w16_target_gain': float(row['actions']['W16']['target_gain']),
            'w16_harmful_full': bool(float(row['actions']['W16']['delta_full_AJ']) < -0.05),
            'w16_helpful_target': bool(float(row['actions']['W16']['target_gain']) > 0.05),
            'traj_features': row['features'],
            'appearance_features': app,
        }
        out_rows.append(row2)
        if (idx + 1) % 250 == 0:
            print(f'processed {idx+1}/{len(rows)}', flush=True)
    out_jsonl = out_dir / 'rgb_dev10_patch_rows.jsonl'
    with out_jsonl.open('w') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {
        'n_rows': len(out_rows),
        'source_rows': str(args.rows),
        'video_cache': str(video_cache),
        'radii': RADII,
        'out_jsonl': str(out_jsonl),
        'label_counts': {
            'window_has_reentry': int(sum(r['window_has_reentry'] for r in out_rows)),
            'w16_harmful_full': int(sum(r['w16_harmful_full'] for r in out_rows)),
            'w16_helpful_target': int(sum(r['w16_helpful_target'] for r in out_rows)),
            'best_action': {a: int(sum(1 for r in out_rows if r['best_action'] == a)) for a in ['reject','W4','W8','W16']},
        },
    }
    (out_dir / 'patch_feature_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
