#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import build_candidate_pool_oracle, eval_one

OUTDIR = Path('outputs/paper_discovery_2026-06-27/candidate_pool_oracle_expanded')


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = Path('outputs/paper_discovery_2026-06-27')
    hybrid = base / 'candidate_pool_oracle/hybrid_candidates'
    candidate_paths = {
        # Original candidates
        'offline': base / 'rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt',
        'online_global': base / 'rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt',
        'b2_fullpost_p1': base / 'rgb_stacking_fresh20_49_natural_ablation/b2_fullpost_p1_rgb_fresh20_49.pt',
        'b2_w8_p2': base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt',
        'b2_w16_p1': base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt',
        'b2_w16_p2': base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt',
        'b2_w32_p2': base / 'rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt',
        'guard_rf_thr0.40': base / 'reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt',
        # Coordinate / visibility disentangled hybrid candidates
        'online_coord_offline_vis': hybrid / 'online_coord_offline_vis.pt',
        'online_coord_b2_vis': hybrid / 'online_coord_b2_vis.pt',
        'online_coord_guard_vis': hybrid / 'online_coord_guard_vis.pt',
        'b2_w8_coord_offline_vis': hybrid / 'b2_w8_coord_offline_vis.pt',
        'b2_w8_coord_guard_vis': hybrid / 'b2_w8_coord_guard_vis.pt',
        'b2_w16p1_coord_offline_vis': hybrid / 'b2_w16p1_coord_offline_vis.pt',
        'b2_w16p2_coord_offline_vis': hybrid / 'b2_w16p2_coord_offline_vis.pt',
        'b2_w32_coord_offline_vis': hybrid / 'b2_w32_coord_offline_vis.pt',
    }
    missing = [str(p) for p in candidate_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))

    oracle_path = OUTDIR / 'rgb_fresh20_49_natural_expanded_candidate_pool_oracle.pt'
    selection = build_candidate_pool_oracle(
        'rgb_fresh20_49_natural_expanded_candidate_pool_oracle',
        candidate_paths,
        oracle_path,
    )

    method_paths = dict(candidate_paths)
    method_paths['expanded_candidate_pool_oracle'] = oracle_path
    method_paths['previous_candidate_pool_oracle'] = base / 'candidate_pool_oracle/rgb_fresh20_49_natural_candidate_pool_oracle.pt'
    method_paths['oracle_b2'] = base / 'b2_oracle_upper_bound/rgb_fresh20_49_natural/rgb_fresh20_49_natural_oracle_choose_b2_per_reentry_query.pt'
    method_paths['oracle_online'] = base / 'b2_oracle_upper_bound/rgb_fresh20_49_natural/rgb_fresh20_49_natural_oracle_choose_online_per_reentry_query.pt'

    rows = [eval_one(name, path) for name, path in method_paths.items() if path.exists()]
    by = {r['name']: r for r in rows}
    gains = {}
    for name in by:
        if name == 'offline':
            continue
        gains[f'{name}_vs_offline'] = {
            'AJ_RD_256': round(float(by[name]['AJ_RD_256']) - float(by['offline']['AJ_RD_256']), 6),
            'AJ_256': round(float(by[name]['AJ_256']) - float(by['offline']['AJ_256']), 6),
            'OA_256': round(float(by[name]['OA_256']) - float(by['offline']['OA_256']), 6),
        }
    for base_name in ['b2_w16_p2', 'guard_rf_thr0.40', 'previous_candidate_pool_oracle', 'oracle_b2', 'oracle_online']:
        if base_name in by:
            gains[f'expanded_candidate_pool_oracle_vs_{base_name}'] = {
                'AJ_RD_256': round(float(by['expanded_candidate_pool_oracle']['AJ_RD_256']) - float(by[base_name]['AJ_RD_256']), 6),
                'AJ_256': round(float(by['expanded_candidate_pool_oracle']['AJ_256']) - float(by[base_name]['AJ_256']), 6),
                'OA_256': round(float(by['expanded_candidate_pool_oracle']['OA_256']) - float(by[base_name]['OA_256']), 6),
            }

    summary = {
        'setting': 'rgb_fresh20_49_natural',
        'protocol': 'Expanded per-query oracle over original candidates plus coordinate/visibility hybrid candidates. For each eligible re-entry query, choose the candidate with highest per-query AJ_RD_256. Non-reentry queries stay offline. Upper bound only.',
        'candidate_paths': {k: str(v) for k, v in candidate_paths.items()},
        'oracle_cache': str(oracle_path),
        'methods': rows,
        'gains': gains,
        'selection': selection,
    }
    out_json = OUTDIR / 'rgb_fresh20_49_natural_expanded_candidate_pool_oracle_summary.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = {
        'out_json': str(out_json),
        'oracle_cache': str(oracle_path),
        'methods': [{k: r[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']} for r in rows],
        'gains': gains,
        'selection': {
            'eligible_reentry_queries': selection['eligible_reentry_queries'],
            'candidate_counts': selection['candidate_counts'],
            'candidate_rates': selection['candidate_rates'],
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
