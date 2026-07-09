#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sweep_b2_w16_false_cost_guards_dev import standard_metrics
from scripts.train_b2_rv_dev_cv import FEATURES, POLICIES, make_model, matrix, labels, select_threshold, eval_policy

IN_ROWS = Path('outputs/paper_discovery_2026-06-27/b2_rv_feature_audit_dev/trigger_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/b2_rv_cache_dev_cv')
DATASETS = {
    'davis': {
        'base': Path('outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt'),
        'w16': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_base.pt'),
        'p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/davis/w16_persist2.pt'),
    },
    'rgb_dev10': {
        'base': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt'),
        'override': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt'),
        'w16': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_base.pt'),
        'p2': Path('outputs/paper_discovery_2026-06-27/b2_w16_false_cost_dev/rgb_dev10/w16_persist2.pt'),
    },
}
# Evaluate two representative policies only: aggressive utility and conservative harm-only.
TARGET_POLICIES = [
    ('score_safe90', 'score', 0.90),
    ('harm_safe90', 'anti_harm', 0.90),
]


def load_rows() -> List[Dict[str, Any]]:
    out = []
    with IN_ROWS.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_cache(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location='cpu', weights_only=False)


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def eval_ajrd(cache_path: Path, out_json: Path) -> Dict[str, Any]:
    subprocess.run([sys.executable, 'scripts/eval_aj_rd_from_cache.py', '--cache-path', str(cache_path), '--output-json', str(out_json)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.load(open(out_json))


def eval_payload(cache_path: Path, payload: Dict[str, Any], out_dir: Path, name: str) -> Dict[str, Any]:
    ajrd = eval_ajrd(cache_path, out_dir / f'{name}_ajrd.json')
    std = standard_metrics(payload['records'])
    return {
        'cache': str(cache_path),
        'AJ_RD_256': ajrd.get('true_AJ_RD_256'),
        'AJ_RD': ajrd.get('true_AJ_RD'),
        'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
        'AJ_256': std['AJ_256_pct'],
        'OA_256': std['OA_256_pct'],
        'delta_avg_256': std['delta_avg_256_pct'],
    }


def fit_oof_policy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    X = matrix(rows)
    y_harm = labels(rows, 'harmful_full')
    y_help = labels(rows, 'helpful_post')
    groups = np.asarray([f"{r['dataset']}::{r['video_id']}" for r in rows])
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    accept_by_policy = {name: np.zeros(len(rows), dtype=bool) for name, _, _ in TARGET_POLICIES}
    score_dump: List[Dict[str, Any]] = []
    folds: List[Dict[str, Any]] = []

    for fold_id, (tr, te) in enumerate(gkf.split(X, y_harm, groups), start=1):
        harm_model = make_model()
        help_model = make_model()
        harm_model.fit(X[tr], y_harm[tr].astype(int))
        help_model.fit(X[tr], y_help[tr].astype(int))
        p_harm_train = harm_model.predict_proba(X[tr])[:, 1]
        p_help_train = help_model.predict_proba(X[tr])[:, 1]
        p_harm_test = harm_model.predict_proba(X[te])[:, 1]
        p_help_test = help_model.predict_proba(X[te])[:, 1]
        score_train = p_help_train - p_harm_train
        score_test = p_help_test - p_harm_test
        anti_harm_train = -p_harm_train
        anti_harm_test = -p_harm_test
        fold_info = {
            'fold': fold_id,
            'train_n': int(len(tr)),
            'test_n': int(len(te)),
            'test_groups': sorted(set(groups[te].tolist())),
            'policies': {},
        }
        for pol_name, score_kind, post_ret in TARGET_POLICIES:
            train_score = score_train if score_kind == 'score' else anti_harm_train
            test_score = score_test if score_kind == 'score' else anti_harm_test
            thr, train_metric = select_threshold([rows[i] for i in tr], train_score, post_ret)
            accept = test_score >= thr
            accept_by_policy[pol_name][te] = accept
            fold_info['policies'][pol_name] = {
                'score_kind': score_kind,
                'post_retention_min_train': post_ret,
                'threshold': round(float(thr), 6),
                'train_metric': train_metric,
                'test_metric_trigger_row': eval_policy([rows[i] for i in te], accept),
            }
        for local_j, idx in enumerate(te):
            rd = {
                'row_index': int(idx),
                'fold': fold_id,
                'dataset': rows[idx]['dataset'],
                'video_id': rows[idx]['video_id'],
                'query_idx': int(rows[idx]['query_idx']),
                'class': rows[idx]['class'],
                'delta_full': rows[idx]['delta_full'],
                'delta_post': rows[idx]['delta_post'],
                'p_harm': round(float(p_harm_test[local_j]), 6),
                'p_help': round(float(p_help_test[local_j]), 6),
                'score': round(float(score_test[local_j]), 6),
                'anti_harm': round(float(anti_harm_test[local_j]), 6),
            }
            for pol_name in accept_by_policy:
                rd[f'accept_{pol_name}'] = bool(accept_by_policy[pol_name][idx])
            score_dump.append(rd)
        folds.append(fold_info)
    return {'accept_by_policy': accept_by_policy, 'folds': folds, 'score_dump': score_dump}


def build_oof_cache(dataset: str, policy: str, accept_map: Dict[Tuple[str, str, int], bool], payloads: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = payloads['base']
    p2 = payloads['p2']
    out_records = []
    stats = {
        'dataset': dataset,
        'policy': policy,
        'total_queries': 0,
        'queries_with_trigger_row': 0,
        'accepted_trigger_queries': 0,
        'rejected_trigger_queries': 0,
    }
    for br, pr in zip(base['records'], p2['records']):
        vid = str(br['video_id'])
        pred_tracks = npy(br['pred_tracks'], np.float32).copy()
        pred_vis = npy(br['pred_visibility'], bool).copy()
        p2_tracks = npy(pr['pred_tracks'], np.float32)
        p2_vis = npy(pr['pred_visibility'], bool)
        n = pred_vis.shape[0]
        for qi in range(n):
            stats['total_queries'] += 1
            key = (dataset, vid, int(qi))
            if key not in accept_map:
                continue
            stats['queries_with_trigger_row'] += 1
            if accept_map[key]:
                stats['accepted_trigger_queries'] += 1
                pred_tracks[qi] = p2_tracks[qi]
                pred_vis[qi] = p2_vis[qi]
            else:
                stats['rejected_trigger_queries'] += 1
        rr = dict(br)
        rr['pred_tracks'] = pred_tracks.astype(np.float32)
        rr['pred_visibility'] = pred_vis.astype(bool)
        rr['model_name'] = f'b2_rv_{policy}_{dataset}'
        out_records.append(rr)
    payload = dict(base)
    payload['model_name'] = f'b2_rv_{policy}_{dataset}'
    payload['b2_rv_policy'] = policy
    payload['b2_rv_definition'] = 'track-level OOF verifier: accept first-trigger track => use B2-W16-P2 track, reject => use base track'
    payload['records'] = out_records
    stats['accept_rate_trigger_queries'] = round(stats['accepted_trigger_queries'] / max(stats['queries_with_trigger_row'], 1), 6)
    return payload, stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fit = fit_oof_policy(rows)
    with (OUT / 'oof_scores.jsonl').open('w') as f:
        for r in fit['score_dump']:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    accept_maps: Dict[str, Dict[Tuple[str, str, int], bool]] = {p: {} for p, _, _ in TARGET_POLICIES}
    for i, r in enumerate(rows):
        key = (r['dataset'], r['video_id'], int(r['query_idx']))
        for pol_name, _, _ in TARGET_POLICIES:
            accept_maps[pol_name][key] = bool(fit['accept_by_policy'][pol_name][i])

    # Load caches and evaluate baselines once.
    summary = {
        'protocol': 'Development-only cache-level grouped CV for B2-RV. Uses DAVIS + RGB dev first-10. No RGB fresh20-49.',
        'definition': 'track-level OOF verifier: if first trigger accepted, use B2-W16-P2 output for the whole query track; otherwise keep base output.',
        'target_policies': [p[0] for p in TARGET_POLICIES],
        'folds': fit['folds'],
        'datasets': {},
        'aggregate_by_policy': {},
    }

    for dataset, cfg in DATASETS.items():
        d_out = OUT / dataset
        d_out.mkdir(parents=True, exist_ok=True)
        payloads = {k: load_cache(v) for k, v in cfg.items()}
        dsum = {
            'caches': {k: str(v) for k, v in cfg.items()},
            'baselines': {},
            'rv': {},
        }
        for name in ['base', 'w16', 'p2']:
            dsum['baselines'][name] = eval_payload(cfg[name], payloads[name], d_out, f'{name}_{dataset}')
        for pol_name, _, _ in TARGET_POLICIES:
            payload, st = build_oof_cache(dataset, pol_name, accept_maps[pol_name], payloads)
            cp = d_out / f'b2_rv_{pol_name}_{dataset}.pt'
            torch.save(payload, cp)
            met = eval_payload(cp, payload, d_out, f'b2_rv_{pol_name}_{dataset}')
            base_met = dsum['baselines']['base']
            p2_met = dsum['baselines']['p2']
            met['stats'] = st
            met['delta_vs_base'] = {
                'AJ_RD_256': round(float(met['AJ_RD_256']) - float(base_met['AJ_RD_256']), 6),
                'AJ_256': round(float(met['AJ_256']) - float(base_met['AJ_256']), 6),
            }
            met['delta_vs_p2'] = {
                'AJ_RD_256': round(float(met['AJ_RD_256']) - float(p2_met['AJ_RD_256']), 6),
                'AJ_256': round(float(met['AJ_256']) - float(p2_met['AJ_256']), 6),
            }
            dsum['rv'][pol_name] = met
        summary['datasets'][dataset] = dsum

    # Aggregate by query counts across datasets as a rough dev aggregate.
    for pol_name, _, _ in TARGET_POLICIES:
        # Query-weighted by number of queries for standard-ish metrics and re-entry by not exactly comparable, still useful dev summary.
        agg = {'policy': pol_name}
        for metric in ['AJ_RD_256', 'AJ_256', 'OA_256', 'delta_avg_256']:
            vals = []
            weights = []
            for dataset, ds in summary['datasets'].items():
                vals.append(float(ds['rv'][pol_name][metric]))
                weights.append(float(ds['rv'][pol_name]['stats']['total_queries']))
            agg[metric] = round(float(np.average(vals, weights=weights)), 6)
        for metric in ['AJ_RD_256', 'AJ_256']:
            vals_base = []
            vals_p2 = []
            vals_rv = []
            weights = []
            for dataset, ds in summary['datasets'].items():
                vals_base.append(float(ds['baselines']['base'][metric]))
                vals_p2.append(float(ds['baselines']['p2'][metric]))
                vals_rv.append(float(ds['rv'][pol_name][metric]))
                weights.append(float(ds['rv'][pol_name]['stats']['total_queries']))
            agg[f'delta_vs_base_{metric}'] = round(float(np.average(vals_rv, weights=weights) - np.average(vals_base, weights=weights)), 6)
            agg[f'delta_vs_p2_{metric}'] = round(float(np.average(vals_rv, weights=weights) - np.average(vals_p2, weights=weights)), 6)
        summary['aggregate_by_policy'][pol_name] = agg

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({
        'protocol': summary['protocol'],
        'definition': summary['definition'],
        'aggregate_by_policy': summary['aggregate_by_policy'],
        'datasets': {
            d: {
                'baselines': summary['datasets'][d]['baselines'],
                'rv': summary['datasets'][d]['rv'],
            } for d in summary['datasets']
        },
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
