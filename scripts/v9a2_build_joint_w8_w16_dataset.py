#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path('/gemini/code/FSPT')
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
W8 = OUTDIR / 'v9a2_features_dist_nc_le64_w8_v3.npz'
W16 = OUTDIR / 'v9a2_features_dist_nc_le64_w16_v3.npz'
OUT = OUTDIR / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
REPORT = OUTDIR / 'v9a2_joint_w8_w16_dataset_report.json'


def load_keys(z):
    return [tuple(json.loads(str(x))) for x in z['row_key_json'].tolist()]


def concat_obj(a, b):
    return np.asarray(list(a.tolist()) + list(b.tolist()), dtype=object)


def main() -> None:
    w8 = np.load(W8, allow_pickle=True)
    w16 = np.load(W16, allow_pickle=True)
    keys8 = load_keys(w8)
    keyset8 = set(keys8)
    keys16 = load_keys(w16)
    ext_idx = np.asarray([i for i, k in enumerate(keys16) if k not in keyset8], dtype=np.int64)
    common_idx_w16 = np.asarray([i for i, k in enumerate(keys16) if k in keyset8], dtype=np.int64)
    assert len(keys8) == 1456, len(keys8)
    assert len(ext_idx) == 557, len(ext_idx)
    assert len(common_idx_w16) == 1456, len(common_idx_w16)
    arrays = {}
    for key in ['X_base', 'X_anchor', 'X_all']:
        arrays[key] = np.concatenate([w8[key], w16[key][ext_idx]], axis=0).astype(np.float32)
    for key in ['base_feature_names', 'anchor_feature_names', 'all_feature_names']:
        arrays[key] = w8[key]
    for key in ['meta_json', 'source_meta_json', 'row_key_json', 'event_key_json', 'row_type']:
        arrays[key] = concat_obj(w8[key], w16[key][ext_idx])
    arrays['is_common_w8'] = np.concatenate([np.ones(len(w8['row_key_json']), dtype=np.int8), np.zeros(len(ext_idx), dtype=np.int8)])
    arrays['is_w16_extension'] = np.concatenate([np.zeros(len(w8['row_key_json']), dtype=np.int8), np.ones(len(ext_idx), dtype=np.int8)])
    for key in w8.files:
        if key.startswith('y_'):
            arrays[key] = np.concatenate([w8[key], w16[key][ext_idx]], axis=0)
    arrays['source_w8_features'] = np.asarray(str(W8), dtype=object)
    arrays['source_w16_features'] = np.asarray(str(W16), dtype=object)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    row_types = [str(x) for x in arrays['row_type'].tolist()]
    labels = {}
    for key, arr in arrays.items():
        if key.startswith('y_'):
            v = np.asarray(arr).astype(float)
            labels[key] = {'mean': float(np.mean(v)), 'sum': float(np.sum(v))}
    report = {
        'out': str(OUT),
        'n_rows': int(arrays['X_all'].shape[0]),
        'row_type_counts': {t: int(row_types.count(t)) for t in sorted(set(row_types))},
        'feature_dim_base': int(arrays['X_base'].shape[1]),
        'feature_dim_anchor': int(arrays['X_anchor'].shape[1]),
        'feature_dim_all': int(arrays['X_all'].shape[1]),
        'finite_rate_X_all': float(np.mean(np.isfinite(arrays['X_all']))),
        'n_w8_common': int(len(w8['row_key_json'])),
        'n_w16_extension': int(len(ext_idx)),
        'n_common_skipped_from_w16': int(len(common_idx_w16)),
        'labels': labels,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'ok': True, 'out': str(OUT), 'report': str(REPORT), 'n_rows': report['n_rows'], 'row_type_counts': report['row_type_counts'], 'feature_dim_all': report['feature_dim_all'], 'finite_rate': report['finite_rate_X_all']}))


if __name__ == '__main__':
    main()
