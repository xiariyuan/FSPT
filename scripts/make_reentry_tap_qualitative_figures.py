#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, pixel_l2_error

ROOT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10')
FIGDIR = Path('docs/figures/reentry_tap')
QUALDIR = FIGDIR / 'qualitative_examples'
MANIFEST = Path('docs/reentry_tap_qualitative_manifest_2026-07-01.md')


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def cache_paths(family: str, L: int = 16) -> Dict[str, Path]:
    d = ROOT / f'{family}_L{L}' / 'predictions'
    return {
        'offline': d / f'cotracker3_offline_{family}_L{L}.pt',
        'online': d / f'cotracker3_online_{family}_L{L}.pt',
        'b2': d / f'b2_w16_p2_{family}_L{L}.pt',
    }


def load_family(family: str, L: int = 16) -> Dict[str, Any]:
    stress = torch.load(ROOT / f'{family}_L{L}' / 'stress_dataset.pt', map_location='cpu', weights_only=False)
    paths = cache_paths(family, L)
    caches = {k: torch.load(v, map_location='cpu', weights_only=False) for k, v in paths.items()}
    return {'stress': stress, 'caches': caches, 'paths': paths}


def dist_px(pred: np.ndarray, gt: np.ndarray, h: int, w: int) -> np.ndarray:
    return pixel_l2_error(pred, gt, h, w, pred_fmt='yx_norm', gt_fmt='yx_norm')


def query_score(ro: Dict[str, Any], rn: Dict[str, Any], rb: Dict[str, Any], qi: int) -> Dict[str, Any] | None:
    gt_v = npy(ro['gt_visibility'], bool)[qi]
    q = npy(ro['query_points'], np.float32)[qi]
    qt = int(round(float(q[0])))
    evs = find_reentry_events(gt_v, qt)
    if not evs:
        return None
    ev = evs[0]
    re = int(ev['reentry_frame'])
    end = min(len(gt_v), re + 24)
    h, w = int(ro['original_size'][0]), int(ro['original_size'][1])
    gt = npy(ro['gt_tracks'], np.float32)[qi, re:end]
    vis = gt_v[re:end]
    if not np.any(vis):
        return None
    po = npy(ro['pred_tracks'], np.float32)[qi, re:end]
    pn = npy(rn['pred_tracks'], np.float32)[qi, re:end]
    pb = npy(rb['pred_tracks'], np.float32)[qi, re:end]
    vo = npy(ro['pred_visibility'], bool)[qi, re:end]
    vn = npy(rn['pred_visibility'], bool)[qi, re:end]
    vb = npy(rb['pred_visibility'], bool)[qi, re:end]
    eo = dist_px(po[vis], gt[vis], h, w) if np.any(vis) else np.array([])
    en = dist_px(pn[vis], gt[vis], h, w) if np.any(vis) else np.array([])
    eb = dist_px(pb[vis], gt[vis], h, w) if np.any(vis) else np.array([])
    # Visibility-aware simple score: low distance and visible near re-entry.
    def good_rate(err, pv):
        vv = pv[vis]
        return float(np.mean((err < 8.0) & vv)) if err.size else 0.0
    go = good_rate(eo, vo)
    gn = good_rate(en, vn)
    gb = good_rate(eb, vb)
    return {
        'query_idx': int(qi),
        'query_t': qt,
        'reentry_frame': re,
        'occ_start_t': int(ev['occ_start_t']),
        'occ_length': int(ev['occ_length']),
        'offline_good_rate': go,
        'online_good_rate': gn,
        'b2_good_rate': gb,
        'b2_minus_offline': gb - go,
        'b2_minus_online': gb - gn,
        'offline_mean_err': float(np.mean(eo)) if eo.size else None,
        'online_mean_err': float(np.mean(en)) if en.size else None,
        'b2_mean_err': float(np.mean(eb)) if eb.size else None,
    }


def select_cases(family: str, data: Dict[str, Any]) -> Dict[str, Any]:
    stress = data['stress']
    caches = data['caches']
    success_candidates = []
    failure_candidates = []
    online_collapse_candidates = []
    for ri, (sr, ro, rn, rb) in enumerate(zip(stress['records'], caches['offline']['records'], caches['online']['records'], caches['b2']['records'])):
        n = npy(ro['query_points']).shape[0]
        for qi in range(n):
            sc = query_score(ro, rn, rb, qi)
            if sc is None:
                continue
            sc['record_index'] = int(ri)
            sc['video_id'] = str(ro['video_id'])
            # Success: B2 improves offline and online, and is accurate after re-entry.
            if sc['b2_good_rate'] >= 0.6 and sc['b2_minus_offline'] >= 0.25 and sc['b2_minus_online'] >= -0.05:
                success_candidates.append(sc)
            # Online collapse: online bad, B2 good.
            if sc['b2_good_rate'] >= 0.6 and sc['online_good_rate'] <= 0.2:
                online_collapse_candidates.append(sc)
            # Failure: B2 worse than offline after re-entry.
            if sc['b2_minus_offline'] <= -0.25:
                failure_candidates.append(sc)
    success_candidates.sort(key=lambda x: (x['b2_minus_offline'], x['b2_good_rate']), reverse=True)
    online_collapse_candidates.sort(key=lambda x: (x['b2_good_rate'] - x['online_good_rate'], x['b2_good_rate']), reverse=True)
    failure_candidates.sort(key=lambda x: (x['b2_minus_offline'], x['offline_good_rate']))
    return {
        'success': success_candidates[0] if success_candidates else None,
        'online_collapse': online_collapse_candidates[0] if online_collapse_candidates else None,
        'failure': failure_candidates[0] if failure_candidates else None,
        'counts': {
            'success_candidates': len(success_candidates),
            'online_collapse_candidates': len(online_collapse_candidates),
            'failure_candidates': len(failure_candidates),
        },
    }


def yx_to_xy_px(yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    return float(yx[1]) * (w - 1), float(yx[0]) * (h - 1)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], r: int = 4, width: int = 2) -> None:
    draw.line((x-r, y-r, x+r, y+r), fill=color, width=width)
    draw.line((x-r, y+r, x+r, y-r), fill=color, width=width)


def draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], r: int = 4) -> None:
    draw.ellipse((x-r, y-r, x+r, y+r), outline=color, width=2)


def annotate_frame(frame: np.ndarray, ro: Dict[str, Any], rn: Dict[str, Any], rb: Dict[str, Any], qi: int, t: int, label: str) -> Image.Image:
    im = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(im)
    h, w = int(ro['original_size'][0]), int(ro['original_size'][1])
    gt = npy(ro['gt_tracks'], np.float32)[qi, t]
    gv = bool(npy(ro['gt_visibility'], bool)[qi, t])
    preds = {
        'offline': (ro, (255, 60, 60)),
        'online': (rn, (60, 120, 255)),
        'B2': (rb, (255, 220, 40)),
    }
    # Draw predictions first.
    for name, (rec, color) in preds.items():
        pv = bool(npy(rec['pred_visibility'], bool)[qi, t])
        yx = npy(rec['pred_tracks'], np.float32)[qi, t]
        x, y = yx_to_xy_px(yx, h, w)
        if 0 <= x < w and 0 <= y < h:
            if pv:
                draw_circle(draw, x, y, color, r=5)
            else:
                draw_cross(draw, x, y, color, r=5)
    # GT last, filled green.
    xg, yg = yx_to_xy_px(gt, h, w)
    if gv and 0 <= xg < w and 0 <= yg < h:
        draw.ellipse((xg-3, yg-3, xg+3, yg+3), fill=(0, 255, 0), outline=(0, 255, 0))
    else:
        draw.rectangle((1, 1, w-2, h-2), outline=(255, 0, 0), width=2)
    # Text background.
    draw.rectangle((0, 0, w, 24), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    return im


def make_contact_sheet(family: str, data: Dict[str, Any], case: Dict[str, Any], out_path: Path, title: str) -> Dict[str, Any]:
    ri = int(case['record_index'])
    qi = int(case['query_idx'])
    stress_rec = data['stress']['records'][ri]
    ro = data['caches']['offline']['records'][ri]
    rn = data['caches']['online']['records'][ri]
    rb = data['caches']['b2']['records'][ri]
    re = int(case['reentry_frame'])
    occ = int(case['occ_start_t'])
    T = stress_rec['video'].shape[0]
    frames = sorted(set([max(0, occ-2), occ, min(T-1, occ+2), re, min(T-1, re+4), min(T-1, re+12)]))
    imgs = []
    for t in frames:
        lab = f'{family} | t={t} | q={qi}'
        if t == re:
            lab += ' | re-entry'
        elif occ <= t < re:
            lab += ' | invisible'
        imgs.append(annotate_frame(stress_rec['video'][t], ro, rn, rb, qi, t, lab))
    # Legend panel.
    w, h = imgs[0].size
    legend = Image.new('RGB', (w, h), color=(245, 245, 245))
    d = ImageDraw.Draw(legend)
    d.text((10, 10), title, fill=(0, 0, 0))
    d.text((10, 40), f"video: {case['video_id']}", fill=(0, 0, 0))
    d.text((10, 62), f"query={qi}, occ_len={case['occ_length']}, re={case['reentry_frame']}", fill=(0, 0, 0))
    d.text((10, 90), f"good-rate after re-entry:", fill=(0, 0, 0))
    d.text((10, 112), f"offline={case['offline_good_rate']:.2f}", fill=(255, 60, 60))
    d.text((10, 134), f"online={case['online_good_rate']:.2f}", fill=(60, 120, 255))
    d.text((10, 156), f"B2={case['b2_good_rate']:.2f}", fill=(150, 130, 0))
    d.text((10, 190), 'Legend: GT=green dot,', fill=(0, 0, 0))
    d.text((10, 212), 'offline=red, online=blue, B2=yellow', fill=(0, 0, 0))
    imgs.append(legend)
    cols = 3
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new('RGB', (cols*w, rows*h), color=(255,255,255))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols)*w, (i // cols)*h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return {'out_path': str(out_path), 'case': case, 'frames': frames}


def make_construction_diagram(out_path: Path) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    def box(x, y, w, h, text, fc):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor='black', linewidth=1.5))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11, wrap=True)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='->', mutation_scale=18, linewidth=1.5))
    box(0.4, 4.2, 2.1, 1.0, 'Original RGB video\n+ TAP queries', '#e8f1ff')
    box(3.2, 4.2, 2.3, 1.0, 'translate_exit_reenter\nL=8/16/32', '#fff2cc')
    box(6.2, 4.2, 2.3, 1.0, 'Out-of-frame\nre-entry events', '#ffe6cc')
    box(9.2, 4.2, 2.3, 1.0, 'Evaluate\nAJ_RD + AJ', '#e2f0d9')
    arrow(2.5,4.7,3.2,4.7); arrow(5.5,4.7,6.2,4.7); arrow(8.5,4.7,9.2,4.7)
    box(0.4, 2.3, 2.1, 1.0, 'Original RGB video\n+ TAP queries', '#e8f1ff')
    box(3.2, 2.3, 2.3, 1.0, 'moving_occluder\nL=8/16/32', '#f4cccc')
    box(6.2, 2.3, 2.3, 1.0, 'Occlusion-induced\nre-entry events', '#eadcf8')
    box(9.2, 2.3, 2.3, 1.0, 'Event provenance\nnatural / stress / mixed', '#d9ead3')
    arrow(2.5,2.8,3.2,2.8); arrow(5.5,2.8,6.2,2.8); arrow(8.5,2.8,9.2,2.8)
    box(0.8, 0.4, 2.4, 1.0, 'Offline base\nstandard-strong', '#fce5cd')
    box(4.0, 0.4, 2.4, 1.0, 'Online override\nre-entry-strong', '#cfe2f3')
    box(7.2, 0.4, 3.3, 1.0, 'B2-W16-P2\nlocal override only near\npredicted re-entry', '#fff2cc')
    arrow(3.2,0.9,4.0,0.9); arrow(6.4,0.9,7.2,0.9)
    ax.text(6, 5.75, 'ReEntry-TAP: controlled re-entry stress test + local reliability intervention', ha='center', va='center', fontsize=15, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    QUALDIR.mkdir(parents=True, exist_ok=True)
    make_construction_diagram(FIGDIR / 'reentry_tap_construction_diagram.png')
    manifest = ['# ReEntry-TAP Qualitative Figures Manifest — 2026-07-01\n']
    manifest.append(f'- Construction diagram: `{FIGDIR / "reentry_tap_construction_diagram.png"}`\n')
    all_cases = {}
    for family in ['translate', 'occluder']:
        data = load_family(family, 16)
        cases = select_cases(family, data)
        all_cases[family] = cases
        manifest.append(f'\n## {family} L16\n')
        manifest.append(f'Candidate counts: `{cases["counts"]}`\n')
        for kind in ['success', 'online_collapse', 'failure']:
            case = cases.get(kind)
            if case is None:
                manifest.append(f'- {kind}: none found\n')
                continue
            out_path = QUALDIR / f'{family}_L16_{kind}.png'
            meta = make_contact_sheet(family, data, case, out_path, f'{family} L16: {kind}')
            manifest.append(f'- {kind}: `{out_path}`\n')
            manifest.append(f'  - video={case["video_id"]}, query={case["query_idx"]}, occ_len={case["occ_length"]}, re={case["reentry_frame"]}, good_rates=(off {case["offline_good_rate"]:.2f}, online {case["online_good_rate"]:.2f}, B2 {case["b2_good_rate"]:.2f})\n')
    MANIFEST.write_text('\n'.join(manifest))
    (QUALDIR / 'selected_cases.json').write_text(json.dumps(all_cases, indent=2, ensure_ascii=False))
    print(json.dumps({'manifest': str(MANIFEST), 'qual_dir': str(QUALDIR), 'construction': str(FIGDIR / 'reentry_tap_construction_diagram.png'), 'cases': all_cases}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
