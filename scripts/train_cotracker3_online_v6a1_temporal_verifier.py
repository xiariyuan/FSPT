#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a1_temporal_verifier/cotracker3_v6a1_temporal_event_dataset.npz'
DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a1_temporal_verifier'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_dataset(path: Path, label: str):
    z = np.load(path, allow_pickle=True)
    X = np.asarray(z['X'], np.float32)
    y = np.asarray(z[label], np.float32).astype(bool)
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    feature_names = [str(f) for f in z['feature_names'].tolist()]
    groups = np.asarray([m['video_id'] for m in metas], object)
    return z, X, y, metas, feature_names, groups


def make_models(seed: int):
    return {
        'logreg_balanced': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=seed)),
        'extra_trees_balanced': ExtraTreesClassifier(n_estimators=400, max_depth=7, min_samples_leaf=3, class_weight='balanced', random_state=seed),
        'random_forest_balanced': RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=3, class_weight='balanced', random_state=seed),
    }


def model_prob(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return model.decision_function(X).astype(np.float32)


def eval_records(records):
    vals = []
    n = 0
    for r in records:
        pred = torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt = torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv = torch.from_numpy(npy(r['pred_visibility'], bool))
        gv = torch.from_numpy(npy(r['gt_visibility'], bool))
        q = torch.from_numpy(npy(r['query_points'], np.float32))
        n += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
        vals.append({
            'AJ': float(m['AJ']) * 100,
            'OA': float(m['OA']) * 100,
            'delta_avg': float(m['average_pts_within_thresh']) * 100,
            'delta_4px': float(m['pts_within_4']) * 100,
        })
    return {k: float(np.mean([v[k] for v in vals])) for k in ['AJ', 'OA', 'delta_avg', 'delta_4px']} | {'n_records': len(vals), 'n_queries': n}


def run_ajrd(cache: Path, out_json: Path):
    subprocess.run(
        [sys.executable, str(ROOT / 'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(cache), '--output-json', str(out_json)],
        cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return json.loads(out_json.read_text())


def infer_offsets(X: np.ndarray, feature_names: list[str], metas: list[dict], strategy: str, post: int = 4) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    offs = []
    for i, m in enumerate(metas):
        if strategy == 'open_t':
            offs.append(0)
            continue
        scores = np.asarray([X[i, name_to_idx[f'score_dt{dt}']] for dt in range(0, post + 1)], np.float32)
        vis = np.asarray([X[i, name_to_idx[f'vis_dt{dt}']] for dt in range(0, post + 1)], np.float32)
        if strategy == 'first_recovery':
            cand = np.where((scores >= 0.60) | (vis >= 0.5))[0]
            offs.append(int(cand[0]) if cand.size else int(np.argmax(scores)))
        elif strategy == 'scoremax':
            offs.append(int(np.argmax(scores)))
        elif strategy == 'oracle_best_safe16':
            offs.append(max(0, int(m.get('best_safe16_offset', -1))))
        else:
            raise ValueError(strategy)
    return np.asarray(offs, np.int64)


def build_open_records(native_payload: dict, metas: list[dict], probs: np.ndarray, threshold: float, offsets: np.ndarray):
    records = []
    for r in native_payload['records']:
        rr = dict(r)
        rr['pred_tracks'] = npy(r['pred_tracks'], np.float32).copy()
        rr['pred_visibility'] = npy(r['pred_visibility'], bool).copy()
        records.append(rr)
    vid_to_idx = {str(r['video_id']): i for i, r in enumerate(native_payload['records'])}
    # one opening per unique event; choose max probability if duplicates exist
    event_to_best = {}
    for i, m in enumerate(metas):
        key = (m['video_id'], int(m['query_idx']), int(m['frame_t']))
        if key not in event_to_best or probs[i] > probs[event_to_best[key]]:
            event_to_best[key] = i
    opened = []
    for key, i in event_to_best.items():
        if probs[i] < threshold:
            continue
        m = metas[i]
        vi = vid_to_idx[m['video_id']]
        q = int(m['query_idx'])
        t = int(m['frame_t']) + int(offsets[i])
        if 0 <= t < records[vi]['pred_visibility'].shape[1]:
            records[vi]['pred_visibility'][q, t] = True
            opened.append(i)
    return records, opened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(DEFAULT_NPZ))
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--label', choices=['y_future_safe16', 'y_future_safe8', 'y_future_gt_visible', 'y_open_t_safe16'], default='y_future_safe16')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-metric-thresholds', type=int, default=24)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z, X, y, metas, feature_names, groups = load_dataset(Path(args.npz), args.label)
    forbidden = ['gt', 'err', 'safe', 'label']
    leakage = [f for f in feature_names if any(s in f.lower() for s in forbidden)]
    if leakage:
        raise RuntimeError(f'Potential leakage in features: {leakage}')

    native = torch.load(args.native_cache, map_location='cpu', weights_only=False)
    native_cache = Path(args.native_cache)
    native_std = eval_records(native['records'])
    native_aj = run_ajrd(native_cache, outdir / f'v6a1_native_ref_{args.label}_ajrd.json')
    native_row = {**native_std, 'AJ_RD': native_aj['true_AJ_RD'], 'AJ_RD_256': native_aj['true_AJ_RD_256']}

    logo = LeaveOneGroupOut()
    results = {}
    threshold_grid = None
    for model_name, model in make_models(args.seed).items():
        oof = np.zeros(len(y), np.float32)
        folds = []
        for tr, te in logo.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                oof[te] = 0.0
                continue
            model.fit(X[tr], y[tr])
            p = model_prob(model, X[te])
            oof[te] = p
            for thr in [0.5, 0.7, 0.85]:
                pred = p >= thr
                pr, rc, f1, _ = precision_recall_fscore_support(y[te], pred, average='binary', zero_division=0)
                folds.append({
                    'heldout_video': str(groups[te][0]), 'threshold': thr,
                    'n_test': int(len(te)), 'n_pos': int(y[te].sum()),
                    'precision': float(pr), 'recall': float(rc), 'f1': float(f1), 'p_mean': float(p.mean())
                })
        ap_score = float(average_precision_score(y, oof)) if int(y.sum()) else 0.0
        auc_score = float(roc_auc_score(y, oof)) if len(np.unique(y)) == 2 else None
        np.save(outdir / f'v6a1_{model_name}_{args.label}_oof.npy', oof)
        # Compact threshold grid: quantiles + fixed thresholds
        thresholds = np.unique(np.concatenate([
            np.linspace(0.1, 0.95, 18),
            np.quantile(oof, np.linspace(0.55, 0.98, 12)),
        ])).astype(float)
        # Downsample if too many
        if len(thresholds) > args.max_metric_thresholds:
            idx = np.linspace(0, len(thresholds) - 1, args.max_metric_thresholds).round().astype(int)
            thresholds = thresholds[idx]
        strategies = ['open_t', 'first_recovery', 'scoremax']
        sweeps = []
        for strategy in strategies:
            offsets = infer_offsets(X, feature_names, metas, strategy)
            for thr in thresholds:
                pred = oof >= thr
                accept_count = int(pred.sum())
                precision = float((pred & y).sum() / max(accept_count, 1))
                recall = float((pred & y).sum() / max(int(y.sum()), 1))
                metric = None
                if accept_count > 0:
                    records, opened = build_open_records(native, metas, oof, float(thr), offsets)
                    payload = dict(native)
                    payload['records'] = records
                    payload['model_name'] = f'v6a1_{model_name}_{args.label}_{strategy}_thr{thr:.3f}'
                    cache = outdir / f'v6a1_{model_name}_{args.label}_{strategy}_thr{thr:.3f}.pt'
                    torch.save(payload, cache)
                    std = eval_records(records)
                    aj = run_ajrd(cache, outdir / f'v6a1_{model_name}_{args.label}_{strategy}_thr{thr:.3f}_ajrd.json')
                    metric = {**std, 'AJ_RD': aj['true_AJ_RD'], 'AJ_RD_256': aj['true_AJ_RD_256'], 'cache': str(cache)}
                    metric['delta_vs_native'] = {k: (metric[k] - native_row[k]) for k in ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']}
                sweeps.append({
                    'strategy': strategy, 'threshold': float(thr), 'accept_count': accept_count,
                    'precision': precision, 'recall': recall, 'metric': metric
                })
        metric_sweeps = [s for s in sweeps if s['metric']]
        # Report top under two objectives: constrained and raw AJ_RD_256.
        constrained = [s for s in metric_sweeps if s['metric']['delta_vs_native']['AJ'] >= -0.10 and s['metric']['delta_vs_native']['OA'] >= -0.10]
        best_constrained = sorted(constrained, key=lambda s: (s['metric']['delta_vs_native']['AJ_RD_256'], s['metric']['delta_vs_native']['AJ_RD'], s['metric']['delta_vs_native']['AJ']), reverse=True)[:10]
        best_raw = sorted(metric_sweeps, key=lambda s: (s['metric']['delta_vs_native']['AJ_RD_256'], s['metric']['delta_vs_native']['AJ_RD']), reverse=True)[:10]
        results[model_name] = {
            'ap': ap_score, 'auc': auc_score, 'folds': folds,
            'oof_path': str(outdir / f'v6a1_{model_name}_{args.label}_oof.npy'),
            'sweeps': sweeps,
            'best_constrained': best_constrained,
            'best_raw': best_raw,
        }
    summary = {
        'script': 'scripts/train_cotracker3_online_v6a1_temporal_verifier.py',
        'npz': str(args.npz), 'label': args.label, 'native_cache': str(args.native_cache),
        'n_samples': int(len(y)), 'n_pos': int(y.sum()), 'pos_rate': float(y.mean()),
        'feature_names': feature_names, 'leakage_audit_passed': True,
        'videos': sorted(set(map(str, groups.tolist()))), 'native_row': native_row,
        'models': results,
    }
    out = outdir / f'v6a1_temporal_verifier_{args.label}_loov_summary.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    brief = {m: {'ap': v['ap'], 'auc': v['auc'], 'best_constrained': v['best_constrained'][:3], 'best_raw': v['best_raw'][:3]} for m, v in results.items()}
    print(json.dumps({'out': str(out), 'n_samples': summary['n_samples'], 'n_pos': summary['n_pos'], 'pos_rate': summary['pos_rate'], 'native_row': native_row, 'model_brief': brief}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
