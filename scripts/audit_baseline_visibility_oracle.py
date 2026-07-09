#!/usr/bin/env python3
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

OUT = Path('outputs/paper_discovery_2026-06-27/baseline_parity_audit')
CACHES = {
    'trackon2_strided_original_existing': Path('caches/trackon2_strided_original.pt'),
    'tapnext_strided_original_existing': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt'),
    'trackon2_first_input_bridge': Path('outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'),
}


def eval_ajrd(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def eval_standard(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_rgb_heldout_cache_standard.py', '--cache', str(cache_path), '--out-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def make_variant(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    out = dict(payload)
    recs = []
    for r in payload['records']:
        rr = dict(r)
        gtvis = np.asarray(r['gt_visibility']).astype(bool)
        if mode == 'gt_visibility':
            rr['pred_visibility'] = gtvis
        elif mode == 'all_visible':
            rr['pred_visibility'] = np.ones_like(gtvis, dtype=bool)
        elif mode == 'query_visible_fill':
            pv = np.asarray(r['pred_visibility']).astype(bool).copy()
            q = np.asarray(r['query_points']).astype(np.float32)
            T = pv.shape[1]
            qts = np.clip(np.rint(q[:,0]).astype(int), 0, T-1)
            pv[np.arange(pv.shape[0]), qts] = True
            rr['pred_visibility'] = pv
        else:
            raise ValueError(mode)
        rr['model_name'] = str(payload.get('model_name','')) + '_' + mode
        recs.append(rr)
    out['records'] = recs
    out['model_name'] = str(payload.get('model_name','')) + '_' + mode
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, path in CACHES.items():
        if not path.exists():
            summary[name] = {'exists': False, 'path': str(path)}
            continue
        payload = torch.load(path, map_location='cpu', weights_only=False)
        summary[name] = {'exists': True, 'path': str(path), 'variants': {}}
        for mode in ['original', 'gt_visibility', 'all_visible', 'query_visible_fill']:
            if mode == 'original':
                cp = path
            else:
                cp = OUT / f'{name}_{mode}.pt'
                torch.save(make_variant(payload, mode), cp)
            try:
                aj = eval_ajrd(cp, OUT / f'{name}_{mode}_ajrd.json')
                st = eval_standard(cp, OUT / f'{name}_{mode}_standard.json')
                summary[name]['variants'][mode] = {
                    'cache': str(cp),
                    'AJ_RD_256': aj.get('true_AJ_RD_256'),
                    'AJ_256': st.get('AJ_256_pct'),
                    'OA_256': st.get('OA_256_pct'),
                    'delta_avg_256': st.get('delta_avg_256_pct'),
                    'n_records': st.get('n_records'),
                    'n_queries': st.get('n_queries'),
                }
            except Exception as e:
                summary[name]['variants'][mode] = {'error': repr(e)}
    (OUT / 'visibility_oracle_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
