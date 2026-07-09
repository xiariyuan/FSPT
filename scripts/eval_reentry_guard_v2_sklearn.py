#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.eval_reentry_guard_v1 as v1

OUTDIR = Path('outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn')
OUTDIR.mkdir(parents=True, exist_ok=True)
SRC = Path('outputs/paper_discovery_2026-06-27/reentry_guard_v1')


def load_npz(name: str):
    p = SRC / f'{name}_features_labels.npz'
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return d['X'].astype(np.float32), d['y'].astype(np.int64), d['meta']


def best_f1_threshold(prob: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    best = {'thr': 0.5, 'f1': -1.0, 'precision': 0.0, 'recall': 0.0}
    for thr in np.linspace(0.05, 0.95, 91):
        pred = prob >= thr
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best['f1']:
            best = {
                'thr': float(thr),
                'f1': float(f1),
                'precision': float(precision_score(y, pred, zero_division=0)),
                'recall': float(recall_score(y, pred, zero_division=0)),
            }
    return best


def fit_models(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    models['logreg_sklearn'] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, solver='lbfgs'))
    models['random_forest'] = RandomForestClassifier(n_estimators=300, max_depth=9, min_samples_leaf=8, class_weight='balanced_subsample', random_state=20260701, n_jobs=-1)
    models['extra_trees'] = ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=6, class_weight='balanced', random_state=20260701, n_jobs=-1)
    models['hist_gbdt'] = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=31, learning_rate=0.04, l2_regularization=0.05, random_state=20260701)
    for m in models.values():
        m.fit(X, y)
    return models


def predict_prob(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    # fallback for hist gradient has predict_proba too normally
    p = model.predict(X)
    return p.astype(np.float32)


def apply_gate_with_prob(setting: str, paths: Dict[str, Path], meta: np.ndarray, prob: np.ndarray, thr: float, out_path: Path, model_name: str) -> Dict[str, Any]:
    accept = prob >= float(thr)
    acc_map = {}
    for row, a in zip(meta, accept):
        vi, qi, vid = row
        acc_map[(str(vid), int(qi))] = bool(a)
    off = torch.load(paths['offline'], map_location='cpu', weights_only=False)
    b2 = torch.load(paths['b2'], map_location='cpu', weights_only=False)
    records = []
    selected = 0
    for br, b2r in zip(off['records'], b2['records']):
        vid = str(br['video_id'])
        pred_p = v1.npy(br['pred_tracks'], np.float32).copy()
        pred_v = v1.npy(br['pred_visibility'], bool).copy()
        cp = v1.npy(b2r['pred_tracks'], np.float32)
        cv = v1.npy(b2r['pred_visibility'], bool)
        qn = pred_v.shape[0]
        use = np.zeros(qn, dtype=bool)
        for qi in range(qn):
            if acc_map.get((vid, qi), False):
                use[qi] = True
        selected += int(np.sum(use))
        pred_p[use] = cp[use]
        pred_v[use] = cv[use]
        nr = dict(br)
        nr['pred_tracks'] = pred_p.astype(np.float32)
        nr['pred_visibility'] = pred_v.astype(bool)
        nr['model_name'] = f'{model_name}_{setting}_thr{thr:.2f}'
        records.append(nr)
    payload = dict(off)
    payload['records'] = records
    payload['model_name'] = f'{model_name}_{setting}_thr{thr:.2f}'
    payload['reentry_guard_v2'] = {'model': model_name, 'threshold': float(thr), 'selected_queries': int(selected), 'eligible_reentry_queries': int(len(acc_map)), 'selection_rate': round(selected / max(len(acc_map), 1), 6)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload['reentry_guard_v2']


def evaluate_model_on_setting(model_name: str, model: Any, setting: str, paths: Dict[str, Path], thresholds: List[float]) -> List[Dict[str, Any]]:
    X, y, meta = load_npz(setting)
    prob = predict_prob(model, X)
    rows = []
    for thr in thresholds:
        outp = OUTDIR / setting / model_name / f'{model_name}_thr{thr:.2f}.pt'
        sel = apply_gate_with_prob(setting, paths, meta, prob, thr, outp, model_name)
        erow = v1.eval_cache(f'{model_name}_thr{thr:.2f}', outp)
        erow['selection'] = sel
        rows.append(erow)
    return rows


def main() -> None:
    X1, y1, _ = load_npz('dev_translate_L16')
    X2, y2, _ = load_npz('dev_occluder_L16')
    X_train = np.vstack([X1, X2])
    y_train = np.concatenate([y1, y2])
    models = fit_models(X_train, y_train)
    model_info = {}
    for name, model in models.items():
        prob = predict_prob(model, X_train)
        best = best_f1_threshold(prob, y_train)
        model_info[name] = best
        with open(OUTDIR / f'{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    eval_settings = ['rgb_fresh20_49_natural', 'fresh20_49_translate_L16', 'fresh20_49_occluder_L16']
    results = []
    for setting in eval_settings:
        paths = v1.SETTINGS[setting]
        base_rows = [v1.eval_cache('offline', paths['offline']), v1.eval_cache('online_global', paths['online']), v1.eval_cache('b2_w16_p2', paths['b2'])]
        b2 = next(r for r in base_rows if r['name'] == 'b2_w16_p2')
        all_rows = list(base_rows)
        for name, model in models.items():
            tset = sorted(set([0.3, 0.4, 0.5, 0.6, float(model_info[name]['thr'])]))
            all_rows.extend(evaluate_model_on_setting(name, model, setting, paths, tset))
        by = {r['name']: r for r in all_rows}
        gains = {}
        for r in all_rows:
            if r['name'] == 'offline':
                continue
            gains[f'{r["name"]}_vs_offline'] = {
                'AJ_RD_256': round(float(r['AJ_RD_256']) - float(by['offline']['AJ_RD_256']), 6),
                'AJ_256': round(float(r['AJ_256']) - float(by['offline']['AJ_256']), 6),
                'OA_256': round(float(r['OA_256']) - float(by['offline']['OA_256']), 6),
            }
            gains[f'{r["name"]}_vs_b2'] = {
                'AJ_RD_256': round(float(r['AJ_RD_256']) - float(b2['AJ_RD_256']), 6),
                'AJ_256': round(float(r['AJ_256']) - float(b2['AJ_256']), 6),
                'OA_256': round(float(r['OA_256']) - float(b2['OA_256']), 6),
            }
        # best objective: maximize AJ_RD then AJ, but require AJ >= B2 - 0.2 to avoid over-aggressive gate.
        guards = [r for r in all_rows if r['name'] not in ['offline','online_global','b2_w16_p2']]
        feasible = [r for r in guards if float(r['AJ_256']) >= float(b2['AJ_256']) - 0.2]
        if feasible:
            best = max(feasible, key=lambda r: (float(r['AJ_RD_256']), float(r['AJ_256'])))
        else:
            best = max(guards, key=lambda r: (float(r['AJ_RD_256']), float(r['AJ_256'])))
        results.append({'setting': setting, 'methods': all_rows, 'gains': gains, 'best_guard': best['name']})
    summary = {
        'train': {'settings': ['dev_translate_L16','dev_occluder_L16'], 'n': int(len(y_train)), 'pos': int(np.sum(y_train==1)), 'neg': int(np.sum(y_train==0)), 'pos_rate': round(float(np.mean(y_train)),6)},
        'model_info': model_info,
        'results': results,
    }
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    compact = []
    for r in results:
        b2 = next(m for m in r['methods'] if m['name'] == 'b2_w16_p2')
        best = next(m for m in r['methods'] if m['name'] == r['best_guard'])
        compact.append({'setting': r['setting'], 'b2': {k:b2[k] for k in ['AJ_RD_256','AJ_256','OA_256']}, 'best': {k:best[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']}, 'selection': best.get('selection'), 'gain_vs_b2': r['gains'][f'{best["name"]}_vs_b2']})
    print(json.dumps({'summary_path': str(OUTDIR / 'summary.json'), 'train': summary['train'], 'model_info': model_info, 'compact': compact}, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
