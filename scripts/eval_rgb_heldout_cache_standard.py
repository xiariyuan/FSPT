#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from datasets.metrics import compute_tapvid_metrics


def npy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    payload = torch.load(args.cache, map_location='cpu', weights_only=False)
    rows = []
    per = []
    for r in payload['records']:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='strided')
        row = {k: float(v) for k, v in m.items()}
        rows.append(row)
        per.append({
            'video_id': str(r['video_id']),
            'AJ_256_pct': round(row.get('AJ', 0.0) * 100.0, 4),
            'OA_256_pct': round(row.get('OA', 0.0) * 100.0, 4),
            'delta_avg_256_pct': round(row.get('average_pts_within_thresh', 0.0) * 100.0, 4),
        })
    out = {
        'cache': args.cache,
        'n_records': len(payload['records']),
        'n_queries': int(sum(r['query_points'].shape[0] for r in payload['records'])),
        'AJ_256_pct': round(float(np.mean([x.get('AJ', 0.0) for x in rows])) * 100.0, 4),
        'OA_256_pct': round(float(np.mean([x.get('OA', 0.0) for x in rows])) * 100.0, 4),
        'delta_avg_256_pct': round(float(np.mean([x.get('average_pts_within_thresh', 0.0) for x in rows])) * 100.0, 4),
        'per_video': per,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_video'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
