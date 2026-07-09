#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def npy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cache_path(root: Path, fam: str, method: str, tag: str) -> Path:
    d = root / f'{fam}_L16' / 'predictions'
    if method == 'offline':
        return d / f'cotracker3_offline_{fam}_L16_{tag}.pt'
    if method == 'online':
        return d / f'cotracker3_online_{fam}_L16_{tag}.pt'
    if method == 'b2_w16_p2':
        return d / f'b2_w16_p2_{fam}_L16_{tag}.pt'
    raise ValueError(method)


def out_cache_path(root: Path, fam: str, method: str, tag: str) -> Path:
    return cache_path(root, fam, method, tag)


def merge_caches(paths: List[Path], out_path: Path, tag: str) -> Dict[str, Any]:
    loaded = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    out = dict(loaded[0])
    out['records'] = []
    for x in loaded:
        out['records'].extend(x['records'])
    out['merged_from'] = [str(p) for p in paths]
    out['split_tag'] = tag
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {'out_cache': str(out_path), 'n_records': len(out['records']), 'n_queries': int(sum(int(npy(r['query_points']).shape[0]) for r in out['records'])), 'inputs': [str(p) for p in paths]}


def merge_stress(root_a: Path, tag_a: str, root_b: Path, tag_b: str, out_root: Path, out_tag: str, fam: str) -> Dict[str, Any]:
    stress_a = torch.load(root_a / f'{fam}_L16/stress_dataset.pt', map_location='cpu', weights_only=False)
    stress_b = torch.load(root_b / f'{fam}_L16/stress_dataset.pt', map_location='cpu', weights_only=False)
    out = dict(stress_a)
    out['records'] = list(stress_a['records']) + list(stress_b['records'])
    out['merged_from'] = [str(root_a / f'{fam}_L16/stress_dataset.pt'), str(root_b / f'{fam}_L16/stress_dataset.pt')]
    out['split_tag'] = out_tag
    out_dir = out_root / f'{fam}_L16'
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_dir / 'stress_dataset.pt')
    sanity_a = json.load(open(root_a / f'{fam}_L16/sanity_summary.json'))
    sanity_b = json.load(open(root_b / f'{fam}_L16/sanity_summary.json'))
    # Recompute a compact combined sanity from source summaries.
    per_video = list(sanity_a.get('per_video', [])) + list(sanity_b.get('per_video', []))
    n_orig = int(sanity_a['num_original_queries'] + sanity_b['num_original_queries'])
    n_kept = int(sanity_a['num_kept_queries'] + sanity_b['num_kept_queries'])
    n_re_q = int(sanity_a['num_reentry_queries'] + sanity_b['num_reentry_queries'])
    n_ev = int(sanity_a['num_reentry_events'] + sanity_b['num_reentry_events'])
    nan_count = int(sanity_a['nan_count'] + sanity_b['nan_count'])
    oob = int(sanity_a['visible_oob_count'] + sanity_b['visible_oob_count'])
    occ_lens = []
    for pv in per_video:
        # Use per-video means only for approximate weighted mean if raw lens absent.
        pass
    combined = {
        'stress_name': sanity_a.get('stress_name'),
        'stress_type': sanity_a.get('stress_type'),
        'stress_params': sanity_a.get('stress_params'),
        'merged_from': [str(root_a / f'{fam}_L16/sanity_summary.json'), str(root_b / f'{fam}_L16/sanity_summary.json')],
        'num_videos': int(sanity_a['num_videos'] + sanity_b['num_videos']),
        'num_original_queries': n_orig,
        'num_kept_queries': n_kept,
        'query_keep_rate': round(n_kept / max(n_orig, 1), 6),
        'query_frame_visible_rate': 1.0 if n_kept else None,
        'num_reentry_queries': n_re_q,
        'reentry_query_rate': round(n_re_q / max(n_kept, 1), 6),
        'num_reentry_events': n_ev,
        'mean_reentry_events_per_query': round(n_ev / max(n_kept, 1), 6),
        'occ_length_mean_note': 'See per_video for source-split weighted details; not recomputed from raw lengths in merge script.',
        'nan_count': nan_count,
        'visible_oob_count': oob,
        'sample_gifs': int(sanity_a.get('sample_gifs', 0) + sanity_b.get('sample_gifs', 0)),
        'per_video': per_video,
        'sanity_pass': bool(n_kept > 0 and nan_count == 0 and oob == 0),
    }
    (out_dir / 'sanity_summary.json').write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    (out_dir / 'manifest.json').write_text(json.dumps({'stress_dataset': str(out_dir / 'stress_dataset.pt'), 'sanity_summary': str(out_dir / 'sanity_summary.json'), 'merged_from': combined['merged_from'], 'sanity_pass': combined['sanity_pass']}, indent=2, ensure_ascii=False))
    return {'family': fam, 'stress_dataset': str(out_dir / 'stress_dataset.pt'), 'sanity': combined}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-a', required=True)
    ap.add_argument('--tag-a', required=True)
    ap.add_argument('--root-b', required=True)
    ap.add_argument('--tag-b', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--out-tag', required=True)
    ap.add_argument('--families', default='translate,occluder')
    args = ap.parse_args()
    root_a = Path(args.root_a); root_b = Path(args.root_b); out_root = Path(args.out_root)
    families = [x.strip() for x in args.families.split(',') if x.strip()]
    summary = {'out_root': str(out_root), 'out_tag': args.out_tag, 'families': []}
    for fam in families:
        fam_sum = merge_stress(root_a, args.tag_a, root_b, args.tag_b, out_root, args.out_tag, fam)
        fam_sum['prediction_caches'] = {}
        for method in ['offline', 'online']:
            inputs = [cache_path(root_a, fam, method, args.tag_a), cache_path(root_b, fam, method, args.tag_b)]
            outp = out_cache_path(out_root, fam, method, args.out_tag)
            fam_sum['prediction_caches'][method] = merge_caches(inputs, outp, args.out_tag)
        summary['families'].append(fam_sum)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f'{args.out_tag}_merge_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
