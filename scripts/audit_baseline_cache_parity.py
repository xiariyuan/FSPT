#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

CACHES = {
    'cotracker3_offline_main': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
    'trackon2_strided_original_existing': Path('caches/trackon2_strided_original.pt'),
    'tapnext_strided_original_existing': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt'),
    'trackon2_first_input_bridge': Path('outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'),
}
OUT = Path('outputs/paper_discovery_2026-06-27/baseline_parity_audit')


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


def q_to_yx_norm(q: np.ndarray) -> np.ndarray:
    # query_points are [t, y_norm, x_norm]
    return np.asarray(q[:, 1:3], dtype=np.float32)


def norm_dist_256(a_yx: np.ndarray, b_yx: np.ndarray) -> np.ndarray:
    d = (a_yx - b_yx) * 256.0
    return np.linalg.norm(d, axis=-1)


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None, 'p99': None, 'max': None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        'n': int(arr.size),
        'mean': round(float(np.mean(arr)), 6),
        'median': round(float(np.median(arr)), 6),
        'p90': round(float(np.percentile(arr, 90)), 6),
        'p99': round(float(np.percentile(arr, 99)), 6),
        'max': round(float(np.max(arr)), 6),
    }


def audit_cache(name: str, path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'name': name, 'path': str(path), 'exists': False}
    payload = load(path)
    records = payload.get('records', [])
    all_anchor = []
    all_anchor_gt = []
    all_pred_minmax = []
    all_vis_rates = []
    all_gt_vis_rates = []
    n_queries = 0
    n_anchor_lt2 = 0
    n_anchor_gt_lt2 = 0
    examples_bad = []
    for ri, r in enumerate(records):
        pred = npy(r['pred_tracks'], np.float32)  # N,T,2 yx norm
        gt = npy(r['gt_tracks'], np.float32)
        pv = npy(r['pred_visibility'], bool)
        gv = npy(r['gt_visibility'], bool)
        q = npy(r['query_points'], np.float32)
        n, T = pred.shape[:2]
        qyxn = q_to_yx_norm(q)
        qts = np.clip(np.rint(q[:, 0]).astype(int), 0, T - 1)
        idx = np.arange(n)
        pred_anchor = pred[idx, qts]
        gt_anchor = gt[idx, qts]
        d_pred = norm_dist_256(pred_anchor, qyxn)
        d_gt = norm_dist_256(gt_anchor, qyxn)
        all_anchor.extend(d_pred.tolist())
        all_anchor_gt.extend(d_gt.tolist())
        n_queries += n
        n_anchor_lt2 += int(np.sum(d_pred < 2.0))
        n_anchor_gt_lt2 += int(np.sum(d_gt < 2.0))
        all_pred_minmax.append([float(np.nanmin(pred)), float(np.nanmax(pred))])
        all_vis_rates.append(float(np.mean(pv)))
        all_gt_vis_rates.append(float(np.mean(gv)))
        if np.max(d_pred) > 50 and len(examples_bad) < 10:
            qi = int(np.argmax(d_pred))
            examples_bad.append({
                'record_index': ri,
                'video_id': str(r.get('video_id')),
                'query_idx': qi,
                'query_t': int(qts[qi]),
                'query_yx_norm': qyxn[qi].tolist(),
                'pred_anchor_yx_norm': pred_anchor[qi].tolist(),
                'gt_anchor_yx_norm': gt_anchor[qi].tolist(),
                'anchor_error_256': round(float(d_pred[qi]), 4),
                'gt_anchor_error_256': round(float(d_gt[qi]), 4),
                'pred_visible_at_query': bool(pv[qi, qts[qi]]),
                'gt_visible_at_query': bool(gv[qi, qts[qi]]),
            })
    pred_min = min(x[0] for x in all_pred_minmax) if all_pred_minmax else None
    pred_max = max(x[1] for x in all_pred_minmax) if all_pred_minmax else None
    return {
        'name': name,
        'path': str(path),
        'exists': True,
        'model_name': payload.get('model_name'),
        'n_records': len(records),
        'n_queries': int(n_queries),
        'anchor_error_256_pred_vs_query': summarize(all_anchor),
        'anchor_error_256_gt_vs_query': summarize(all_anchor_gt),
        'anchor_pred_lt2_rate': round(n_anchor_lt2 / max(n_queries, 1), 6),
        'anchor_gt_lt2_rate': round(n_anchor_gt_lt2 / max(n_queries, 1), 6),
        'pred_coord_global_min': pred_min,
        'pred_coord_global_max': pred_max,
        'pred_visibility_rate_mean_video': summarize(all_vis_rates),
        'gt_visibility_rate_mean_video': summarize(all_gt_vis_rates),
        'bad_anchor_examples': examples_bad,
        'schema_keys': sorted(list(payload.keys())),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit_cache(k, v) for k, v in CACHES.items()]
    summary = {'caches': rows}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
