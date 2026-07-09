#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics
from utils.coords import find_reentry_events

OUT_DEV = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/trigger_taxonomy_stress_summary.json')
OUT_FRESH = Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/trigger_taxonomy_stress_summary.json')


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


def trigger_frames(base_v: np.ndarray, over_v: np.ndarray, qt: int, W: int = 16, persist: int = 2) -> List[int]:
    out = []
    T = len(base_v)
    t = max(1, int(qt) + 1)
    while t < T:
        ok = invisible_run_before(base_v, t) >= 1 and t + persist <= T and bool(np.all(over_v[t:t+persist]))
        if ok:
            out.append(int(t))
            t = min(T, t + W + 1)
        else:
            t += 1
    return out


def load_event_sources(stress_dir: Path) -> Dict[Tuple[str, int], set[str]]:
    rows_path = stress_dir / 'event_provenance' / 'event_rows.jsonl'
    mp: Dict[Tuple[str, int], set[str]] = {}
    if not rows_path.exists():
        return mp
    for line in rows_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if not bool(r.get('eligible')):
            continue
        k = (str(r['video_id']), int(r['query_idx']))
        mp.setdefault(k, set()).add(str(r['event_source']))
    return mp


def single_query_aj(record: Dict[str, Any], qi: int) -> float:
    pred = torch.from_numpy(npy(record['pred_tracks'][qi:qi+1], np.float32))
    gt = torch.from_numpy(npy(record['gt_tracks'][qi:qi+1], np.float32))
    pv = torch.from_numpy(npy(record['pred_visibility'][qi:qi+1], bool))
    gv = torch.from_numpy(npy(record['gt_visibility'][qi:qi+1], bool))
    q = torch.from_numpy(npy(record['query_points'][qi:qi+1], np.float32))
    m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
    return float(m.get('AJ', 0.0)) * 100.0


def cache_paths(root: Path, family: str, L: int) -> Dict[str, Path]:
    d = root / f'{family}_L{L}' / 'predictions'
    fresh = 'fresh20_29' in str(root)
    if fresh:
        return {
            'offline': d / f'cotracker3_offline_{family}_L{L}_fresh20_29.pt',
            'online': d / f'cotracker3_online_{family}_L{L}_fresh20_29.pt',
            'b2': d / f'b2_w16_p2_{family}_L{L}_fresh20_29.pt',
        }
    return {
        'offline': d / f'cotracker3_offline_{family}_L{L}.pt',
        'online': d / f'cotracker3_online_{family}_L{L}.pt',
        'b2': d / f'b2_w16_p2_{family}_L{L}.pt',
    }


def run_one(root: Path, family: str, L: int) -> Dict[str, Any]:
    stress_dir = root / f'{family}_L{L}'
    p = cache_paths(root, family, L)
    off = torch.load(p['offline'], map_location='cpu', weights_only=False)
    on = torch.load(p['online'], map_location='cpu', weights_only=False)
    b2 = torch.load(p['b2'], map_location='cpu', weights_only=False)
    src_by_q = load_event_sources(stress_dir)
    counts = {
        'total_tracks': 0,
        'gt_reentry_tracks': 0,
        'triggered_tracks': 0,
        'false_trigger_tracks': 0,
        'triggered_natural_tracks': 0,
        'triggered_stress_induced_tracks': 0,
        'triggered_mixed_tracks': 0,
        'missed_natural_tracks': 0,
        'missed_stress_induced_tracks': 0,
        'missed_mixed_tracks': 0,
        'total_natural_tracks': 0,
        'total_stress_induced_tracks': 0,
        'total_mixed_tracks': 0,
    }
    false_deltas = []
    false_rows = []
    for ri, (ro, rn, rb) in enumerate(zip(off['records'], on['records'], b2['records'])):
        vid = str(ro['video_id'])
        base_v = npy(ro['pred_visibility'], bool)
        over_v = npy(rn['pred_visibility'], bool)
        gt_v = npy(ro['gt_visibility'], bool)
        qpts = npy(ro['query_points'], np.float32)
        n = qpts.shape[0]
        counts['total_tracks'] += int(n)
        for qi in range(n):
            qt = int(round(float(qpts[qi, 0])))
            evs = find_reentry_events(gt_v[qi], qt)
            has_re = bool(evs)
            if has_re:
                counts['gt_reentry_tracks'] += 1
            sources = src_by_q.get((vid, qi), set())
            # Count only eligible-source classes; a track can be in multiple classes.
            if 'natural' in sources:
                counts['total_natural_tracks'] += 1
            if 'stress_induced' in sources:
                counts['total_stress_induced_tracks'] += 1
            if 'mixed' in sources:
                counts['total_mixed_tracks'] += 1
            tr = trigger_frames(base_v[qi], over_v[qi], qt)
            if tr:
                counts['triggered_tracks'] += 1
                if not has_re:
                    counts['false_trigger_tracks'] += 1
                    aj_b2 = single_query_aj(rb, qi)
                    aj_off = single_query_aj(ro, qi)
                    delta = aj_b2 - aj_off
                    false_deltas.append(delta)
                    false_rows.append({'video_id': vid, 'query_idx': int(qi), 'first_trigger': int(tr[0]), 'delta_AJ_256': round(delta, 6)})
                else:
                    if 'natural' in sources:
                        counts['triggered_natural_tracks'] += 1
                    if 'stress_induced' in sources:
                        counts['triggered_stress_induced_tracks'] += 1
                    if 'mixed' in sources:
                        counts['triggered_mixed_tracks'] += 1
            else:
                if 'natural' in sources:
                    counts['missed_natural_tracks'] += 1
                if 'stress_induced' in sources:
                    counts['missed_stress_induced_tracks'] += 1
                if 'mixed' in sources:
                    counts['missed_mixed_tracks'] += 1
    fd = np.asarray(false_deltas, dtype=np.float64)
    false_summary = {
        'n': int(fd.size),
        'mean_delta_AJ_256': round(float(np.mean(fd)), 6) if fd.size else None,
        'median_delta_AJ_256': round(float(np.median(fd)), 6) if fd.size else None,
        'low_cost_delta_ge_minus_0p1_rate': round(float(np.mean(fd >= -0.1)), 6) if fd.size else None,
        'harmful_delta_lt_minus_1_rate': round(float(np.mean(fd < -1.0)), 6) if fd.size else None,
        'severe_delta_lt_minus_5_rate': round(float(np.mean(fd < -5.0)), 6) if fd.size else None,
        'worst_10': sorted(false_rows, key=lambda x: x['delta_AJ_256'])[:10],
    }
    def rate(num, den):
        return round(float(num) / max(float(den), 1.0), 6)
    rates = {
        'trigger_precision_track': rate(counts['triggered_tracks'] - counts['false_trigger_tracks'], counts['triggered_tracks']),
        'trigger_recall_track': rate(counts['triggered_tracks'] - counts['false_trigger_tracks'], counts['gt_reentry_tracks']),
        'natural_trigger_recall': rate(counts['triggered_natural_tracks'], counts['total_natural_tracks']),
        'stress_induced_trigger_recall': rate(counts['triggered_stress_induced_tracks'], counts['total_stress_induced_tracks']),
        'mixed_trigger_recall': rate(counts['triggered_mixed_tracks'], counts['total_mixed_tracks']),
        'false_trigger_rate_among_triggered': rate(counts['false_trigger_tracks'], counts['triggered_tracks']),
    }
    return {
        'root': str(root),
        'family': family,
        'L': int(L),
        'counts': counts,
        'rates': rates,
        'false_trigger_cost': false_summary,
    }


def run_root(root: Path, families: List[str], lengths: List[int], out_path: Path) -> Dict[str, Any]:
    results = []
    for fam in families:
        for L in lengths:
            print('taxonomy', root, fam, L, flush=True)
            results.append(run_one(root, fam, L))
    summary = {'root': str(root), 'results': results}
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    dev = run_root(Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10'), ['translate','occluder'], [8,16,32], OUT_DEV)
    fresh = run_root(Path('outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29'), ['translate','occluder'], [16], OUT_FRESH)
    compact = []
    for group in [dev, fresh]:
        for r in group['results']:
            compact.append({
                'root': Path(group['root']).name,
                'family': r['family'],
                'L': r['L'],
                'precision': r['rates']['trigger_precision_track'],
                'recall': r['rates']['trigger_recall_track'],
                'natural_recall': r['rates']['natural_trigger_recall'],
                'stress_induced_recall': r['rates']['stress_induced_trigger_recall'],
                'false_trigger_rate': r['rates']['false_trigger_rate_among_triggered'],
                'false_mean_delta_AJ': r['false_trigger_cost']['mean_delta_AJ_256'],
            })
    print(json.dumps({'dev_summary': str(OUT_DEV), 'fresh_summary': str(OUT_FRESH), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
