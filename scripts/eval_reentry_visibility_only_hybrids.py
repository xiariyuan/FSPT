#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import eval_one, check_alignment, npy

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_visibility_only_hybrids')


def build_hybrid(name: str, coord_path: Path, vis_path: Path, out_path: Path) -> Dict[str, Any]:
    coord = torch.load(coord_path, map_location='cpu', weights_only=False)
    vis = torch.load(vis_path, map_location='cpu', weights_only=False)
    check_alignment(coord, vis, name)
    records = []
    for cr, vr in zip(coord['records'], vis['records']):
        nr = dict(cr)
        nr['pred_tracks'] = npy(cr['pred_tracks'], np.float32).copy()
        nr['pred_visibility'] = npy(vr['pred_visibility'], bool).copy()
        nr['visibility_only_hybrid'] = {
            'name': name,
            'coordinate_source': str(coord_path),
            'visibility_source': str(vis_path),
        }
        records.append(nr)
    payload = dict(coord)
    payload['records'] = records
    payload['model_name'] = name
    payload['visibility_only_hybrid'] = {
        'name': name,
        'coordinate_source': str(coord_path),
        'visibility_source': str(vis_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {'name': name, 'cache': str(out_path), 'coord': str(coord_path), 'vis': str(vis_path)}


def setting_paths() -> Dict[str, Dict[str, Path]]:
    b = Path('outputs/paper_discovery_2026-06-27')
    return {
        'rgb_fresh20_49_natural': {
            'offline': b/'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt',
            'online': b/'rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt',
            'b2': b/'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt',
            'b2_w8': b/'rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt',
            'b2_w16p1': b/'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt',
            'b2_w32': b/'rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt',
            'fullpost': b/'rgb_stacking_fresh20_49_natural_ablation/b2_fullpost_p1_rgb_fresh20_49.pt',
            'guard': b/'reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt',
        },
        'fresh20_49_translate_L16': {
            'offline': b/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt',
            'online': b/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_online_translate_L16_fresh20_49.pt',
            'b2': b/'reentry_stress_rgb_fresh20_49/translate_L16/predictions/b2_w16_p2_translate_L16_fresh20_49.pt',
            'guard': b/'reentry_guard_v2_sklearn/fresh20_49_translate_L16/random_forest/random_forest_thr0.40.pt',
        },
        'fresh20_49_occluder_L16': {
            'offline': b/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt',
            'online': b/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_online_occluder_L16_fresh20_49.pt',
            'b2': b/'reentry_stress_rgb_fresh20_49/occluder_L16/predictions/b2_w16_p2_occluder_L16_fresh20_49.pt',
            'guard': b/'reentry_guard_v2_sklearn/fresh20_49_occluder_L16/random_forest/random_forest_thr0.40.pt',
        },
        'dev_translate_L16': {
            'offline': b/'reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt',
            'online': b/'reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt',
            'b2': b/'reentry_stress_rgb_dev10/translate_L16/predictions/b2_w16_p2_translate_L16.pt',
        },
        'dev_occluder_L16': {
            'offline': b/'reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt',
            'online': b/'reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_online_occluder_L16.pt',
            'b2': b/'reentry_stress_rgb_dev10/occluder_L16/predictions/b2_w16_p2_occluder_L16.pt',
        },
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    settings = setting_paths()
    all_results: List[Dict[str, Any]] = []
    for setting, p in settings.items():
        missing = {k: str(v) for k, v in p.items() if not v.exists()}
        if missing:
            print('SKIP missing', setting, missing, flush=True)
            continue
        sdir = OUTDIR / setting
        hybrid_specs = {
            'offline_coord_online_vis': p['online'],
            'offline_coord_b2_vis': p['b2'],
        }
        if 'guard' in p:
            hybrid_specs['offline_coord_guard_vis'] = p['guard']
        for extra in ['b2_w8', 'b2_w16p1', 'b2_w32', 'fullpost']:
            if extra in p:
                hybrid_specs[f'offline_coord_{extra}_vis'] = p[extra]
        built = []
        for name, vis_src in hybrid_specs.items():
            outp = sdir / f'{name}.pt'
            built.append(build_hybrid(f'{setting}_{name}', p['offline'], vis_src, outp))
        method_paths: Dict[str, Path] = {
            'offline': p['offline'],
            'online_global': p['online'],
            'b2_w16_p2': p['b2'],
        }
        if 'guard' in p:
            method_paths['reentry_guard_rf_thr0.40'] = p['guard']
        for binfo in built:
            # short display name without setting prefix
            display = binfo['name'].replace(setting + '_', '')
            method_paths[display] = Path(binfo['cache'])
        rows = [eval_one(name, path) for name, path in method_paths.items()]
        by = {r['name']: r for r in rows}
        gains = {}
        for r in rows:
            if r['name'] == 'offline':
                continue
            gains[f'{r["name"]}_vs_offline'] = {
                'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['offline']['AJ_RD_256']), 6),
                'AJ_256': round(float(r['AJ_256']) - float(by['offline']['AJ_256']), 6),
                'OA_256': round(float(r['OA_256']) - float(by['offline']['OA_256']), 6),
            }
            if 'b2_w16_p2' in by:
                gains[f'{r["name"]}_vs_b2'] = {
                    'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['b2_w16_p2']['AJ_RD_256']), 6),
                    'AJ_256': round(float(r['AJ_256']) - float(by['b2_w16_p2']['AJ_256']), 6),
                    'OA_256': round(float(r['OA_256']) - float(by['b2_w16_p2']['OA_256']), 6),
                }
        # choose best under AJ >= offline - 1.0, then max AJ_RD
        feasible = [r for r in rows if float(r['AJ_256']) >= float(by['offline']['AJ_256']) - 1.0]
        best = max(feasible, key=lambda r: (float(r['AJ_RD_256']), float(r['AJ_256'])))
        result = {'setting': setting, 'methods': rows, 'gains': gains, 'built_hybrids': built, 'best_under_AJ_loss_le_1': best['name']}
        (sdir / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        all_results.append(result)
        print('SETTING_DONE', setting, 'best', best['name'], best['AJ_RD_256'], best['AJ_256'], flush=True)
    summary = {'results': all_results}
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in all_results:
        best = next(m for m in r['methods'] if m['name'] == r['best_under_AJ_loss_le_1'])
        keep_names = ['offline', 'online_global', 'b2_w16_p2', 'reentry_guard_rf_thr0.40', best['name'], 'offline_coord_b2_vis', 'offline_coord_guard_vis', 'offline_coord_online_vis']
        compact.append({
            'setting': r['setting'],
            'key_methods': [{k: m[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']} for m in r['methods'] if m['name'] in keep_names],
            'best_under_AJ_loss_le_1': {k: best[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']},
            'gains_for_best': {k:v for k,v in r['gains'].items() if k.startswith(best['name'] + '_')},
        })
    print(json.dumps({'summary_path': str(OUTDIR/'summary.json'), 'compact': compact}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
