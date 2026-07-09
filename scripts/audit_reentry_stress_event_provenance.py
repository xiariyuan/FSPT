#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events
from utils.reentry_metrics import (
    DEFAULT_AJRD_D_MINS,
    DEFAULT_AJ_THRESHOLDS,
    compute_reappearance_segment_aj,
    eligible_reentry_events,
    summarize_reappearance_ajrd,
)

DEFAULT_ROOT = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10')
DEFAULT_SOURCE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def source_map(source_cache: Path) -> Dict[str, Dict[str, Any]]:
    payload = torch.load(source_cache, map_location='cpu', weights_only=False)
    return {str(r['video_id']): r for r in payload['records']}


def classify_event(natural_occ: np.ndarray, stress_occ: np.ndarray, start: int, end: int) -> Dict[str, Any]:
    # interval [start, end) is invisible before re-entry.
    nat = int(np.sum(natural_occ[start:end]))
    st = int(np.sum(stress_occ[start:end]))
    if st > 0 and nat == 0:
        src = 'stress_induced'
    elif st == 0 and nat > 0:
        src = 'natural'
    elif st > 0 and nat > 0:
        src = 'mixed'
    else:
        src = 'other'
    return {
        'source': src,
        'num_natural_occ_frames': nat,
        'num_stress_occ_frames': st,
        'num_other_occ_frames': max(0, int(end - start) - nat - st),
    }


def stress_reason_masks(stress_record: Dict[str, Any], source_record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return natural_occ and stress_occ masks, shape N_kept,T.

    natural_occ: original source visibility is false.
    stress_occ: source visible but stress transform/occluder makes point invisible.
    """
    source_idx = npy(stress_record['source_query_indices'], np.int64)
    orig_vis = npy(source_record['gt_visibility'], bool)[source_idx]
    orig_gt = npy(source_record['gt_tracks'], np.float32)[source_idx]
    H, W = [int(x) for x in npy(stress_record['original_size'], np.int32)]
    stype = str(stress_record.get('stress_type', ''))
    natural_occ = ~orig_vis
    if stype == 'translate_exit_reenter':
        shift = npy(stress_record['stress_shift_xy_px'], np.float32)  # T,2 [dx,dy]
        dx = shift[:, 0].reshape(1, -1)
        dy = shift[:, 1].reshape(1, -1)
        y = orig_gt[..., 0] + dy / max(H - 1, 1)
        x = orig_gt[..., 1] + dx / max(W - 1, 1)
        inside = (y >= 0.0) & (y <= 1.0) & (x >= 0.0) & (x <= 1.0)
        stress_occ = orig_vis & (~inside)
    elif stype == 'moving_occluder':
        rects = npy(stress_record['occluder_rect_xyxy_px'], np.float32)
        x_px = orig_gt[..., 1] * max(W - 1, 1)
        y_px = orig_gt[..., 0] * max(H - 1, 1)
        covered = np.zeros_like(orig_vis, dtype=bool)
        for t in range(orig_vis.shape[1]):
            x0, y0, x1, y1 = rects[t]
            if x1 <= x0:
                continue
            covered[:, t] = (x_px[:, t] >= x0) & (x_px[:, t] <= x1) & (y_px[:, t] >= y0) & (y_px[:, t] <= y1)
        stress_occ = orig_vis & covered
    else:
        raise ValueError(f'Unknown stress_type={stype}')
    return natural_occ.astype(bool), stress_occ.astype(bool)


def build_event_rows(stress_dataset: Path, source_cache: Path, out_dir: Path) -> Dict[str, Any]:
    stress = torch.load(stress_dataset, map_location='cpu', weights_only=False)
    smap = source_map(source_cache)
    rows: List[Dict[str, Any]] = []
    summary_counts = {'natural': 0, 'stress_induced': 0, 'mixed': 0, 'other': 0}
    eligible_counts = {'natural': 0, 'stress_induced': 0, 'mixed': 0, 'other': 0}
    query_with = {'natural': set(), 'stress_induced': set(), 'mixed': set(), 'other': set()}
    occ_by_source = {'natural': [], 'stress_induced': [], 'mixed': [], 'other': []}
    for rec_i, sr in enumerate(stress['records']):
        src_vid = str(sr.get('source_video_id', '') or str(sr['video_id']).split('_translate_')[0].split('_occluder_')[0])
        if src_vid not in smap:
            raise KeyError(f'source video not found: {src_vid}')
        nat_occ, st_occ = stress_reason_masks(sr, smap[src_vid])
        gt_vis = npy(sr['gt_visibility'], bool)
        qpts = npy(sr['query_points'], np.float32)
        eligible_set = set()
        for qi in range(gt_vis.shape[0]):
            qt = int(round(float(qpts[qi, 0])))
            for ev in eligible_reentry_events(gt_vis[qi], qt):
                eligible_set.add((qi, int(ev['occ_start_t']), int(ev['reentry_frame'])))
        for qi in range(gt_vis.shape[0]):
            qt = int(round(float(qpts[qi, 0])))
            events = find_reentry_events(gt_vis[qi], qt)
            for ei, ev in enumerate(events):
                start = int(ev['occ_start_t'])
                re = int(ev['reentry_frame'])
                cls = classify_event(nat_occ[qi], st_occ[qi], start, re)
                source = cls['source']
                is_elig = (qi, start, re) in eligible_set
                summary_counts[source] += 1
                if is_elig:
                    eligible_counts[source] += 1
                    query_with[source].add((str(sr['video_id']), qi))
                occ_by_source[source].append(int(ev['occ_length']))
                rows.append({
                    'stress_name': stress.get('stress_name'),
                    'stress_type': stress.get('stress_type'),
                    'stress_params': stress.get('stress_params', {}),
                    'video_id': str(sr['video_id']),
                    'source_video_id': src_vid,
                    'record_index': int(rec_i),
                    'query_idx': int(qi),
                    'source_query_idx': int(npy(sr['source_query_indices'], np.int64)[qi]),
                    'event_idx': int(ei),
                    'query_t': int(qt),
                    'occ_start_t': start,
                    'reentry_frame': re,
                    'occ_length': int(ev['occ_length']),
                    'last_visible_t': int(ev['last_visible_t']),
                    'event_source': source,
                    'eligible': bool(is_elig),
                    **cls,
                })
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / 'event_rows.jsonl'
    with rows_path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    total = len(rows)
    total_elig = sum(eligible_counts.values())
    def stats(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {'n': 0, 'mean': None, 'median': None, 'hist': {}}
        hist = {}
        for v in vals:
            if v <= 4:
                b = '1-4'
            elif v <= 8:
                b = '5-8'
            elif v <= 16:
                b = '9-16'
            elif v <= 32:
                b = '17-32'
            else:
                b = '33+'
            hist[b] = hist.get(b, 0) + 1
        return {'n': len(vals), 'mean': round(float(np.mean(vals)), 6), 'median': round(float(np.median(vals)), 6), 'hist': hist}
    summary = {
        'stress_dataset': str(stress_dataset),
        'stress_name': stress.get('stress_name'),
        'stress_type': stress.get('stress_type'),
        'stress_params': stress.get('stress_params', {}),
        'total_events': int(total),
        'total_eligible_events': int(total_elig),
        'event_counts_by_source': summary_counts,
        'event_rate_by_source': {k: round(v / max(total, 1), 6) for k, v in summary_counts.items()},
        'eligible_event_counts_by_source': eligible_counts,
        'eligible_event_rate_by_source': {k: round(v / max(total_elig, 1), 6) for k, v in eligible_counts.items()},
        'eligible_query_counts_by_source': {k: len(v) for k, v in query_with.items()},
        'occ_length_by_source': {k: stats(v) for k, v in occ_by_source.items()},
        'event_rows_path': str(rows_path),
    }
    (out_dir / 'event_provenance_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load_event_rows(path: Path) -> Dict[Tuple[str, int, int, int], str]:
    mp: Dict[Tuple[str, int, int, int], str] = {}
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not bool(r.get('eligible')):
            continue
        key = (str(r['video_id']), int(r['query_idx']), int(r['occ_start_t']), int(r['reentry_frame']))
        mp[key] = str(r['event_source'])
    return mp


def ajrd_by_source(cache_path: Path, event_rows_path: Path) -> Dict[str, Any]:
    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    src_map = load_event_rows(event_rows_path)
    sources = ['stress_induced', 'natural', 'mixed', 'other']
    per_source_query_scores_256 = {s: [] for s in sources}
    per_source_query_scores = {s: [] for s in sources}
    event_counts = {s: 0 for s in sources}
    query_counts = {s: 0 for s in sources}
    for r in payload['records']:
        vid = str(r['video_id'])
        h, w = int(r['original_size'][0]), int(r['original_size'][1])
        pred = npy(r['pred_tracks'], np.float32)
        gt = npy(r['gt_tracks'], np.float32)
        pvis = npy(r['pred_visibility'], bool)
        gvis = npy(r['gt_visibility'], bool)
        qpts = npy(r['query_points'], np.float32)
        for qi in range(gt.shape[0]):
            qt = int(round(float(qpts[qi, 0])))
            evs_by_source = {s: [] for s in sources}
            for ev in eligible_reentry_events(gvis[qi], qt):
                key = (vid, int(qi), int(ev['occ_start_t']), int(ev['reentry_frame']))
                s = src_map.get(key, 'other')
                ev2 = dict(ev)
                ev2['query_t'] = qt
                evs_by_source[s].append(ev2)
            for s, evs in evs_by_source.items():
                if not evs:
                    continue
                query_counts[s] += 1
                event_counts[s] += len(evs)
                segs = []
                segs256 = []
                for ev in evs:
                    row = compute_reappearance_segment_aj(pred[qi], gt[qi], pvis[qi], gvis[qi], ev, h, w, thresholds=DEFAULT_AJ_THRESHOLDS, use_256_space=False)
                    row256 = compute_reappearance_segment_aj(pred[qi], gt[qi], pvis[qi], gvis[qi], ev, h, w, thresholds=DEFAULT_AJ_THRESHOLDS, use_256_space=True)
                    if row is not None:
                        segs.append(row)
                    if row256 is not None:
                        segs256.append(row256)
                sm = summarize_reappearance_ajrd(segs, d_mins=DEFAULT_AJRD_D_MINS)
                sm256 = summarize_reappearance_ajrd(segs256, d_mins=DEFAULT_AJRD_D_MINS)
                if sm.get('aj_rd') is not None:
                    per_source_query_scores[s].append(float(sm['aj_rd']))
                if sm256.get('aj_rd') is not None:
                    per_source_query_scores_256[s].append(float(sm256['aj_rd']))
    return {
        'cache': str(cache_path),
        'model_name': payload.get('model_name'),
        'event_counts_by_source': event_counts,
        'query_counts_by_source': query_counts,
        'AJ_RD_by_source': {s: round(float(np.mean(v)), 4) if v else None for s, v in per_source_query_scores.items()},
        'AJ_RD_256_by_source': {s: round(float(np.mean(v)), 4) if v else None for s, v in per_source_query_scores_256.items()},
    }


def run_one(stress_dir: Path, source_cache: Path) -> Dict[str, Any]:
    stress_dataset = stress_dir / 'stress_dataset.pt'
    out_dir = stress_dir / 'event_provenance'
    prov = build_event_rows(stress_dataset, source_cache, out_dir)
    preds = stress_dir / 'predictions'
    # Family-specific file names.
    name = stress_dir.name
    if name.startswith('translate_L'):
        L = name.replace('translate_L', '')
        files = {
            'offline': preds / f'cotracker3_offline_translate_L{L}.pt',
            'online': preds / f'cotracker3_online_translate_L{L}.pt',
            'b2_w16_p2': preds / f'b2_w16_p2_translate_L{L}.pt',
        }
    elif name.startswith('occluder_L'):
        L = name.replace('occluder_L', '')
        files = {
            'offline': preds / f'cotracker3_offline_occluder_L{L}.pt',
            'online': preds / f'cotracker3_online_occluder_L{L}.pt',
            'b2_w16_p2': preds / f'b2_w16_p2_occluder_L{L}.pt',
        }
    else:
        files = {}
    ajrd = {}
    for m, p in files.items():
        if p.exists():
            ajrd[m] = ajrd_by_source(p, out_dir / 'event_rows.jsonl')
    combined = {'stress_dir': str(stress_dir), 'provenance': prov, 'ajrd_by_source': ajrd}
    (out_dir / 'provenance_ajrd_summary.json').write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    ap.add_argument('--source-cache', default=str(DEFAULT_SOURCE))
    ap.add_argument('--families', default='translate,occluder')
    ap.add_argument('--lengths', default='8,16,32')
    args = ap.parse_args()
    root = Path(args.root)
    lengths = [int(x) for x in args.lengths.split(',') if x.strip()]
    families = [x.strip() for x in args.families.split(',') if x.strip()]
    results = []
    for fam in families:
        for L in lengths:
            if fam == 'translate':
                d = root / f'translate_L{L}'
            elif fam == 'occluder':
                d = root / f'occluder_L{L}'
            else:
                continue
            if not (d / 'stress_dataset.pt').exists():
                print('skip missing', d, flush=True)
                continue
            print('=== provenance', d, '===', flush=True)
            results.append(run_one(d, Path(args.source_cache)))
    out = root / 'event_provenance_combined_summary.json'
    out.write_text(json.dumps({'results': results}, indent=2, ensure_ascii=False))
    # Compact print.
    compact = []
    for r in results:
        p = r['provenance']
        row = {
            'stress_name': p['stress_name'],
            'total_events': p['total_events'],
            'eligible_event_counts_by_source': p['eligible_event_counts_by_source'],
            'eligible_event_rate_by_source': p['eligible_event_rate_by_source'],
        }
        if 'b2_w16_p2' in r['ajrd_by_source']:
            row['b2_AJ_RD_256_by_source'] = r['ajrd_by_source']['b2_w16_p2']['AJ_RD_256_by_source']
        if 'offline' in r['ajrd_by_source']:
            row['offline_AJ_RD_256_by_source'] = r['ajrd_by_source']['offline']['AJ_RD_256_by_source']
        compact.append(row)
    print(json.dumps({'combined_summary': str(out), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
