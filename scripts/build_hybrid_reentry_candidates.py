#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def check_alignment(coord_cache: Dict[str, Any], vis_cache: Dict[str, Any]) -> None:
    if len(coord_cache['records']) != len(vis_cache['records']):
        raise ValueError('record count mismatch')
    for i, (a, b) in enumerate(zip(coord_cache['records'], vis_cache['records'])):
        if str(a['video_id']) != str(b['video_id']):
            raise ValueError(f'video mismatch {i}: {a["video_id"]} vs {b["video_id"]}')
        for k in ['query_points', 'gt_tracks', 'gt_visibility', 'original_size']:
            if k == 'gt_visibility':
                ok = np.array_equal(npy(a[k], bool), npy(b[k], bool))
            else:
                ok = np.allclose(npy(a[k]), npy(b[k]), atol=1e-6, rtol=1e-6)
            if not ok:
                raise ValueError(f'alignment mismatch record={i} key={k}')


def build(coord_path: Path, vis_path: Path, out_path: Path, name: str) -> Dict[str, Any]:
    coord = torch.load(coord_path, map_location='cpu', weights_only=False)
    vis = torch.load(vis_path, map_location='cpu', weights_only=False)
    check_alignment(coord, vis)
    records = []
    for cr, vr in zip(coord['records'], vis['records']):
        nr = dict(cr)
        nr['pred_tracks'] = npy(cr['pred_tracks'], np.float32).copy()
        nr['pred_visibility'] = npy(vr['pred_visibility'], bool).copy()
        nr['hybrid_candidate'] = {
            'name': name,
            'coordinate_source': str(coord_path),
            'visibility_source': str(vis_path),
        }
        records.append(nr)
    payload = dict(coord)
    payload['records'] = records
    payload['model_name'] = name
    payload['hybrid_candidate'] = {
        'name': name,
        'coordinate_source': str(coord_path),
        'visibility_source': str(vis_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {
        'name': name,
        'out_path': str(out_path),
        'coordinate_source': str(coord_path),
        'visibility_source': str(vis_path),
        'n_records': len(records),
        'n_queries': int(sum(npy(r['query_points']).shape[0] for r in records)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    base = Path('outputs/paper_discovery_2026-06-27')
    offline = base / 'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'
    online = base / 'rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt'
    b2 = base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt'
    b2w8 = base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt'
    b2w16p1 = base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt'
    b2w32 = base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt'
    guard = base / 'reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt'

    specs = {
        'online_coord_offline_vis': (online, offline),
        'online_coord_b2_vis': (online, b2),
        'online_coord_guard_vis': (online, guard),
        'b2_w8_coord_offline_vis': (b2w8, offline),
        'b2_w8_coord_guard_vis': (b2w8, guard),
        'b2_w16p1_coord_offline_vis': (b2w16p1, offline),
        'b2_w16p2_coord_offline_vis': (b2, offline),
        'b2_w32_coord_offline_vis': (b2w32, offline),
    }
    rows = []
    for name, (coord_path, vis_path) in specs.items():
        out_path = out_dir / f'{name}.pt'
        rows.append(build(coord_path, vis_path, out_path, name))
    manifest = out_dir / 'hybrid_manifest.json'
    manifest.write_text(json.dumps({'hybrids': rows}, indent=2, ensure_ascii=False))
    print(json.dumps({'manifest': str(manifest), 'hybrids': rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
