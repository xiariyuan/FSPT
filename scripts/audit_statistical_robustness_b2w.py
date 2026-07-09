#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

OUT = Path('outputs/paper_discovery_2026-06-27/statistical_robustness_audit')
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    {
        'name': 'davis_p2_vs_fixed',
        'path': Path('outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/per_video_rows.jsonl'),
        'deltas': {
            'AJ_RD_256': 'p2_vs_fixed_AJ_RD_256',
            'AJ_256': 'p2_vs_fixed_AJ_256',
            'vs_global_B1_AJ_256': 'p2_vs_b1_AJ_256',
            'vs_fullpost_AJ_RD_256': 'p2_vs_full_AJ_RD_256',
        },
        'description': 'DAVIS B2-W16-P2 vs CoTracker3 offline fixed base',
    },
    {
        'name': 'rgb_fresh20_49_p2_vs_offline',
        'path': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/per_video_rows.jsonl'),
        'deltas': {
            'AJ_RD_256': 'p2_vs_offline_AJ_RD_256',
            'AJ_256': 'p2_vs_offline_AJ_256',
            'vs_online_AJ_RD_256': 'p2_vs_online_AJ_RD_256',
            'vs_online_AJ_256': 'p2_vs_online_AJ_256',
        },
        'description': 'RGB fresh20-49 B2-W16-P2 vs CoTracker3 offline base',
    },
    {
        'name': 'trackon_first_p2_vs_trackon',
        'path': Path('outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_rows.jsonl'),
        'deltas': {
            'AJ_RD_256': 'p2_trackon_base_vs_trackon_AJ_RD_256',
            'AJ_256': 'p2_trackon_base_vs_trackon_AJ_256',
        },
        'description': 'Supplementary TrackOn2 first-input B2-W16-P2 vs TrackOn2 baseline',
    },
    {
        'name': 'trackon_first_cotracker_base_p2_vs_cotracker',
        'path': Path('outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_rows.jsonl'),
        'deltas': {
            'AJ_RD_256': 'p2_cotracker_base_vs_cotracker_AJ_RD_256',
            'AJ_256': 'p2_cotracker_base_vs_cotracker_AJ_256',
        },
        'description': 'Supplementary first-input CoTracker base + TrackOn2 override vs CoTracker base',
    },
]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sign_test_two_sided(x: np.ndarray) -> Dict[str, Any]:
    # Exact two-sided binomial sign test ignoring zeros, p=0.5.
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    n = pos + neg
    if n == 0:
        return {'positive': pos, 'negative': neg, 'zero': int(np.sum(x == 0)), 'n_nonzero': n, 'p_two_sided': None}
    k = min(pos, neg)
    # sum_{i=0}^k C(n,i) / 2^n, doubled
    import math
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * prob)
    return {'positive': pos, 'negative': neg, 'zero': int(np.sum(x == 0)), 'n_nonzero': n, 'p_two_sided': round(float(p), 8)}


def bootstrap_mean_ci(x: np.ndarray, n_boot=20000, seed=123) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return {'mean': None}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return {
        'n': int(n),
        'mean': round(float(np.mean(x)), 6),
        'median': round(float(np.median(x)), 6),
        'std': round(float(np.std(x, ddof=1)), 6) if n > 1 else 0.0,
        'ci95_low': round(float(np.percentile(means, 2.5)), 6),
        'ci95_high': round(float(np.percentile(means, 97.5)), 6),
        'p_boot_mean_le_0': round(float(np.mean(means <= 0)), 8),
        'p_boot_mean_ge_0': round(float(np.mean(means >= 0)), 8),
    }


def summarize_delta(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = []
    videos = []
    for r in rows:
        v = r.get('deltas', {}).get(key)
        if v is not None:
            vals.append(float(v))
            videos.append(r.get('video_id'))
    x = np.asarray(vals, dtype=np.float64)
    summ = bootstrap_mean_ci(x)
    summ.update(sign_test_two_sided(x))
    summ['improve_rate'] = round(float(np.mean(x > 0)), 6) if len(x) else None
    summ['drop_gt_0p01'] = int(np.sum(x < -0.01))
    summ['drop_gt_0p05'] = int(np.sum(x < -0.05))
    summ['top_negative'] = [
        {'video_id': videos[i], 'delta': round(float(x[i]), 6)}
        for i in np.argsort(x)[:5]
    ] if len(x) else []
    summ['top_positive'] = [
        {'video_id': videos[i], 'delta': round(float(x[i]), 6)}
        for i in np.argsort(x)[-5:][::-1]
    ] if len(x) else []
    return summ


def main() -> None:
    result = {'protocol': 'Per-video paired bootstrap/sign-test robustness audit. Uses existing per-video metrics only; no model inference.', 'tasks': {}}
    for task in TASKS:
        rows = load_rows(task['path'])
        tres = {'description': task['description'], 'path': str(task['path']), 'n_videos': len(rows), 'metrics': {}}
        for metric, key in task['deltas'].items():
            tres['metrics'][metric] = summarize_delta(rows, key)
        result['tasks'][task['name']] = tres
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
