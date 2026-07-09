#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import yx_norm_to_xy_256

ROOT = Path('outputs/paper_discovery_2026-06-27')
OUT = ROOT / 'reentry_visibility_bottleneck_final'
OUT.mkdir(parents=True, exist_ok=True)

SETTINGS = {
    'rgb_fresh20_49_natural': {
        'offline': ROOT / 'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt',
        'full_w8_p2': ROOT / 'rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt',
        'vis_w8_p2': ROOT / 'reentry_visguard_sweep/rgb_fresh20_49_natural/visguard_w8_p2.pt',
        'full_w16_p2': ROOT / 'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt',
        'vis_w16_p2': ROOT / 'reentry_visguard_sweep/rgb_fresh20_49_natural/visguard_w16_p2.pt',
    },
    'fresh20_49_translate_L16': {
        'offline': ROOT / 'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt',
        'full_w8_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_translate_L16/full_variants/b2_w8_p2.pt',
        'vis_w8_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_translate_L16/visguard_w8_p2.pt',
        'full_w16_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_translate_L16/full_variants/b2_w16_p2.pt',
        'vis_w16_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_translate_L16/visguard_w16_p2.pt',
    },
    'fresh20_49_occluder_L16': {
        'offline': ROOT / 'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt',
        'full_w8_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_occluder_L16/full_variants/b2_w8_p2.pt',
        'vis_w8_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_occluder_L16/visguard_w8_p2.pt',
        'full_w16_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_occluder_L16/full_variants/b2_w16_p2.pt',
        'vis_w16_p2': ROOT / 'reentry_visguard_sweep/fresh20_49_occluder_L16/visguard_w16_p2.pt',
    },
}


def first_events(gt_visibility: np.ndarray, query_t: int):
    t_len = len(gt_visibility)
    t = max(0, int(query_t) + 1)
    out = []
    while t < t_len:
        if bool(gt_visibility[t]):
            t += 1
            continue
        occ_start = t
        while t < t_len and not bool(gt_visibility[t]):
            t += 1
        if t < t_len and bool(gt_visibility[t]):
            reentry_t = t
            visible_end = reentry_t
            while visible_end < t_len and bool(gt_visibility[visible_end]):
                visible_end += 1
            out.append((occ_start, reentry_t, visible_end, reentry_t - occ_start))
        t += 1
    return out


def norm(d):
    return {
        'events': d['events'],
        'frames': d['frames'],
        'coord1_rate': round(d['coord1'] / max(d['frames'], 1), 6),
        'coord2_rate': round(d['coord2'] / max(d['frames'], 1), 6),
        'coord4_rate': round(d['coord4'] / max(d['frames'], 1), 6),
        'coord8_rate': round(d['coord8'] / max(d['frames'], 1), 6),
        'coord16_rate': round(d['coord16'] / max(d['frames'], 1), 6),
        'predvis_recall': round(d['predvis'] / max(d['frames'], 1), 6),
        'joint8_rate': round(d['joint8'] / max(d['frames'], 1), 6),
        'joint16_rate': round(d['joint16'] / max(d['frames'], 1), 6),
        'first_coord1_rate': round(d['first_coord1'] / max(d['events'], 1), 6),
        'first_coord2_rate': round(d['first_coord2'] / max(d['events'], 1), 6),
        'first_coord4_rate': round(d['first_coord4'] / max(d['events'], 1), 6),
        'first_coord8_rate': round(d['first_coord8'] / max(d['events'], 1), 6),
        'first_coord16_rate': round(d['first_coord16'] / max(d['events'], 1), 6),
        'first_predvis_rate': round(d['first_predvis'] / max(d['events'], 1), 6),
        'first_joint8_rate': round(d['first_joint8'] / max(d['events'], 1), 6),
    }


def _empty_sum():
    return {
        'events': 0,
        'frames': 0,
        'coord1': 0,
        'coord2': 0,
        'coord4': 0,
        'coord8': 0,
        'coord16': 0,
        'predvis': 0,
        'joint8': 0,
        'joint16': 0,
        'first_coord1': 0,
        'first_coord2': 0,
        'first_coord4': 0,
        'first_coord8': 0,
        'first_coord16': 0,
        'first_predvis': 0,
        'first_joint8': 0,
    }


def audit_setting(setting, paths):
    caches = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in paths.items()}
    methods = list(caches.keys())
    sums = {k: _empty_sum() for k in methods}
    nrec = len(caches['offline']['records'])

    for ri in range(nrec):
        base = caches['offline']['records'][ri]
        gt = np.asarray(base['gt_tracks'], np.float32)
        gt_vis = np.asarray(base['gt_visibility'], bool)
        query_points = np.asarray(base['query_points'], np.float32)
        pred_tracks = {k: np.asarray(caches[k]['records'][ri]['pred_tracks'], np.float32) for k in methods}
        pred_vis = {k: np.asarray(caches[k]['records'][ri]['pred_visibility'], bool) for k in methods}

        for qi in range(query_points.shape[0]):
            qt = int(round(float(query_points[qi, 0])))
            for _occ_start, rt, visible_end, _occ_len in first_events(gt_vis[qi], qt):
                visible_len = visible_end - rt
                if visible_len <= 0:
                    continue
                seg = slice(rt, visible_end)
                gt_xy_256 = yx_norm_to_xy_256(gt[qi, seg])
                for method in methods:
                    pred_xy_256 = yx_norm_to_xy_256(pred_tracks[method][qi, seg])
                    dist = np.linalg.norm(pred_xy_256 - gt_xy_256, axis=-1)
                    c1 = dist < 1
                    c2 = dist < 2
                    c4 = dist < 4
                    c8 = dist < 8
                    c16 = dist < 16
                    vis = pred_vis[method][qi, seg]
                    d = sums[method]
                    d['events'] += 1
                    d['frames'] += visible_len
                    d['coord1'] += int(np.sum(c1))
                    d['coord2'] += int(np.sum(c2))
                    d['coord4'] += int(np.sum(c4))
                    d['coord8'] += int(np.sum(c8))
                    d['coord16'] += int(np.sum(c16))
                    d['predvis'] += int(np.sum(vis))
                    d['joint8'] += int(np.sum(vis & c8))
                    d['joint16'] += int(np.sum(vis & c16))

                    first_pred_xy_256 = yx_norm_to_xy_256(pred_tracks[method][qi, rt : rt + 1])[0]
                    first_gt_xy_256 = yx_norm_to_xy_256(gt[qi, rt : rt + 1])[0]
                    first_dist = float(np.linalg.norm(first_pred_xy_256 - first_gt_xy_256))
                    first_vis = bool(pred_vis[method][qi, rt])
                    d['first_coord1'] += int(first_dist < 1)
                    d['first_coord2'] += int(first_dist < 2)
                    d['first_coord4'] += int(first_dist < 4)
                    d['first_coord8'] += int(first_dist < 8)
                    d['first_coord16'] += int(first_dist < 16)
                    d['first_predvis'] += int(first_vis)
                    d['first_joint8'] += int(first_vis and first_dist < 8)

    return {
        'setting': setting,
        'coordinate_space': '256-space pixels converted from normalized yx coordinates',
        'bugfix_note': 'This script intentionally converts normalized [y,x] coordinates to xy_256 before applying pixel thresholds. Older outputs that compared normalized distances directly to 8/16 px are invalid.',
        'methods': {k: norm(v) for k, v in sums.items()},
    }


def main():
    results = []
    for setting, paths in SETTINGS.items():
        missing = {k: str(v) for k, v in paths.items() if not v.exists()}
        if missing:
            raise FileNotFoundError(f'{setting}: {missing}')
        res = audit_setting(setting, paths)
        (OUT / f'{setting}_coord_visibility.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))
        results.append(res)
        print('DONE', setting, flush=True)
    summary = {'results': results}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
