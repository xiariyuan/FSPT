#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_b2_w16_false_cost_guards_dev import standard_metrics

OUT = Path('outputs/paper_discovery_2026-06-27/b2wv_oracle_dev')
ROWS_DIR = Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev')
DATASETS = {
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
        'b2_w16': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_base.pt'),
        'b2_p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt'),
        'rows': ROWS_DIR / 'davis_window_rows.jsonl',
    },
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
        'b2_w16': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_base.pt'),
        'b2_p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_persist2.pt'),
        'rows': ROWS_DIR / 'rgb_dev10_window_rows.jsonl',
    },
}
LAMBDAS = ['lambda_0.25', 'lambda_0.5', 'lambda_1', 'lambda_2']
ACTION_W = {'reject': None, 'W4': 4, 'W8': 8, 'W16': 16}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load_cache(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def eval_ajrd(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def eval_payload(cache_path: Path, payload: Dict[str, Any], out_dir: Path, name: str) -> Dict[str, Any]:
    aj = eval_ajrd(cache_path, out_dir / f'{name}_ajrd.json')
    std = standard_metrics(payload['records'])
    return {
        'cache': str(cache_path),
        'AJ_RD_256': aj.get('true_AJ_RD_256'),
        'AJ_RD': aj.get('true_AJ_RD'),
        'first_reentry_frame_proxy': aj.get('first_reentry_frame_proxy'),
        'AJ_256': std['AJ_256_pct'],
        'OA_256': std['OA_256_pct'],
        'delta_avg_256': std['delta_avg_256_pct'],
    }


def build_oracle_payload(dataset: str, lam_key: str, base: Dict[str, Any], over: Dict[str, Any], rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows_by_vid_q: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_vid_q.setdefault((str(r['video_id']), int(r['query_idx'])), []).append(r)
    for v in rows_by_vid_q.values():
        v.sort(key=lambda x: int(x['trigger_t']))

    action_counts = {a: 0 for a in ACTION_W}
    tracks_with_any_candidate = 0
    tracks_with_any_accept = 0
    records = []
    for b, o in zip(base['records'], over['records']):
        vid = str(b['video_id'])
        pred_tr = npy(b['pred_tracks'], np.float32).copy()
        pred_v = npy(b['pred_visibility'], bool).copy()
        over_tr = npy(o['pred_tracks'], np.float32)
        over_v = npy(o['pred_visibility'], bool)
        qpts = npy(b['query_points'], np.float32)
        n, T = pred_v.shape
        for qi in range(n):
            key = (vid, int(qi))
            cands = rows_by_vid_q.get(key, [])
            if not cands:
                continue
            tracks_with_any_candidate += 1
            accepted = False
            for r in cands:
                action = str(r['best_action'][lam_key])
                action_counts[action] += 1
                W = ACTION_W[action]
                if W is None:
                    continue
                t = int(r['trigger_t'])
                lo = max(0, t - 1)
                hi = min(T, t + int(W) + 1)
                pred_tr[qi, lo:hi] = over_tr[qi, lo:hi]
                pred_v[qi, lo:hi] = over_v[qi, lo:hi]
                accepted = True
            if accepted:
                tracks_with_any_accept += 1
        rr = dict(b)
        rr['pred_tracks'] = pred_tr.astype(np.float32)
        rr['pred_visibility'] = pred_v.astype(bool)
        rr['model_name'] = f'b2wv_oracle_{lam_key}_{dataset}'
        rr['b2wv_oracle_lambda'] = lam_key
        records.append(rr)
    payload = dict(base)
    payload['model_name'] = f'b2wv_oracle_{lam_key}_{dataset}'
    payload['b2wv_oracle_lambda'] = lam_key
    payload['b2wv_oracle_note'] = 'GT counterfactual best-action oracle over candidate windows; diagnostic upper bound, not deployable.'
    payload['records'] = records
    stats = {
        'dataset': dataset,
        'lambda': lam_key,
        'action_counts': action_counts,
        'total_candidate_windows': int(sum(action_counts.values())),
        'tracks_with_any_candidate': int(tracks_with_any_candidate),
        'tracks_with_any_accept': int(tracks_with_any_accept),
        'accept_rate_windows': round((action_counts['W4'] + action_counts['W8'] + action_counts['W16']) / max(sum(action_counts.values()), 1), 6),
    }
    return payload, stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        'protocol': 'B2-WV GT counterfactual best-action oracle cache. Development only. Diagnostic upper bound; not deployable.',
        'datasets': {},
    }
    for dataset, cfg in DATASETS.items():
        d_out = OUT / dataset
        d_out.mkdir(parents=True, exist_ok=True)
        base = load_cache(cfg['base'])
        over = load_cache(cfg['override'])
        rows = load_rows(cfg['rows'])
        dsum = {'baselines': {}, 'oracles': {}, 'rows': str(cfg['rows'])}
        for bname, path in [('base', cfg['base']), ('b2_w16', cfg['b2_w16']), ('b2_p2', cfg['b2_p2'])]:
            payload = load_cache(path)
            dsum['baselines'][bname] = eval_payload(path, payload, d_out, bname)
        for lam in LAMBDAS:
            payload, stats = build_oracle_payload(dataset, lam, base, over, rows)
            cp = d_out / f'b2wv_oracle_{lam}_{dataset}.pt'
            torch.save(payload, cp)
            met = eval_payload(cp, payload, d_out, f'b2wv_oracle_{lam}_{dataset}')
            p2 = dsum['baselines']['b2_p2']
            base_met = dsum['baselines']['base']
            met['stats'] = stats
            met['delta_vs_p2'] = {
                'AJ_RD_256': round(float(met['AJ_RD_256']) - float(p2['AJ_RD_256']), 6),
                'AJ_256': round(float(met['AJ_256']) - float(p2['AJ_256']), 6),
            }
            met['delta_vs_base'] = {
                'AJ_RD_256': round(float(met['AJ_RD_256']) - float(base_met['AJ_RD_256']), 6),
                'AJ_256': round(float(met['AJ_256']) - float(base_met['AJ_256']), 6),
            }
            dsum['oracles'][lam] = met
        summary['datasets'][dataset] = dsum
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
