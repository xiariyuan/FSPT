#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/gemini/code/FSPT')
sys.path.insert(0, str(ROOT))
from datasets.metrics import compute_tapvid_metrics

DEFAULT_FEATURES = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30_event_features.npz'
DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_full30.pt'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7b1_damage_aware_raw_visconf'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def err_px(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a, np.float32) - np.asarray(b, np.float32)) * 255.0))


def build_damage_labels(native_payload: dict, metas: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    rec_by_vid = {str(r['video_id']): r for r in native_payload['records']}
    y_damage, y_safe = [], []
    stats = {'n_gt_occ': 0, 'n_err_bad_visible': 0, 'n_safe16': 0, 'n_total': len(metas)}
    for m in metas:
        r = rec_by_vid[m['video_id']]
        q = int(m['query_idx']); t = int(m['frame_t'])
        pred = npy(r['pred_tracks'], np.float32)
        gt = npy(r['gt_tracks'], np.float32)
        gv = npy(r['gt_visibility'], bool)
        if not bool(gv[q, t]):
            damage = True
            safe = False
            stats['n_gt_occ'] += 1
        else:
            e = err_px(pred[q, t], gt[q, t])
            damage = e > 16.0
            safe = e <= 16.0
            if damage:
                stats['n_err_bad_visible'] += 1
            if safe:
                stats['n_safe16'] += 1
        y_damage.append(damage); y_safe.append(safe)
    y_damage = np.asarray(y_damage, bool)
    y_safe = np.asarray(y_safe, bool)
    stats['damage_rate'] = float(y_damage.mean())
    stats['safe16_rate'] = float(y_safe.mean())
    return y_damage, y_safe, stats


def component_indices(names: list[str]) -> list[int]:
    component_terms = (
        'raw_vis', 'raw_conf', 'vis_prob', 'conf_prob', 'min_prob', 'max_prob',
        'prob_gap', 'raw_gap', 'vis_margin', 'conf_margin', 'limiting',
        'prob_log_ratio', 'product_vs_min', 'component'
    )
    return [i for i, n in enumerate(names) if any(t in n for t in component_terms)]


def product_indices(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith('raw::') and ('score' in n or 'native_visible' in n)]


def top_univariate_indices(X: np.ndarray, y: np.ndarray, candidate_idx: list[int], k: int) -> list[int]:
    rows = []
    for j in candidate_idx:
        v = X[:, j]
        if len(np.unique(v)) < 2:
            continue
        ap = max(average_precision_score(y, v), average_precision_score(y, -v))
        rows.append((float(ap), j))
    return [j for _, j in sorted(rows, reverse=True)[:k]]


def fit_oof_models(X: np.ndarray, y_benefit: np.ndarray, y_damage: np.ndarray, groups: np.ndarray, benefit_idx: list[int], damage_candidates: list[int], damage_top_k: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    benefit_oof = np.zeros(len(y_benefit), np.float32)
    damage_oof = np.zeros(len(y_damage), np.float32)
    folds = []
    for tr, te in LeaveOneGroupOut().split(X, y_benefit, groups):
        video = str(groups[te][0])
        if len(np.unique(y_benefit[tr])) < 2 or len(np.unique(y_damage[tr])) < 2:
            folds.append({'video': video, 'skipped': True})
            continue
        benefit_clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear')
        )
        benefit_clf.fit(X[tr][:, benefit_idx], y_benefit[tr])
        benefit_oof[te] = benefit_clf.predict_proba(X[te][:, benefit_idx])[:, 1].astype(np.float32)

        dmg_idx = top_univariate_indices(X[tr], y_damage[tr], damage_candidates, damage_top_k)
        damage_clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
        )
        damage_clf.fit(X[tr][:, dmg_idx], y_damage[tr])
        damage_oof[te] = damage_clf.predict_proba(X[te][:, dmg_idx])[:, 1].astype(np.float32)
        folds.append({
            'video': video,
            'n_test': int(len(te)),
            'benefit_pos_test': int(y_benefit[te].sum()),
            'damage_pos_test': int(y_damage[te].sum()),
            'damage_features': int(len(dmg_idx)),
            'benefit_p_mean': float(benefit_oof[te].mean()),
            'damage_p_mean': float(damage_oof[te].mean()),
        })
    return benefit_oof, damage_oof, folds


def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'], np.float32))
        gt=torch.from_numpy(npy(r['gt_tracks'], np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'], bool))
        gv=torch.from_numpy(npy(r['gt_visibility'], bool))
        q=torch.from_numpy(npy(r['query_points'], np.float32))
        n += int(q.shape[0])
        m=compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}


def run_ajrd(cache: Path, out_json: Path):
    subprocess.run(
        [sys.executable, str(ROOT/'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(cache), '--output-json', str(out_json)],
        cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return json.loads(out_json.read_text())


def build_open_records(native_payload: dict, metas: list[dict], score: np.ndarray, threshold: float):
    records=[]
    for r in native_payload['records']:
        rr=dict(r)
        rr['pred_visibility']=npy(r['pred_visibility'], bool).copy()
        records.append(rr)
    vid_to_idx={str(r['video_id']):i for i,r in enumerate(records)}
    opened=[]
    for i,m in enumerate(metas):
        if score[i] < threshold:
            continue
        vi=vid_to_idx.get(m['video_id'])
        if vi is None:
            continue
        q=int(m['query_idx']); t=int(m['frame_t'])
        if 0 <= t < records[vi]['pred_visibility'].shape[1]:
            records[vi]['pred_visibility'][q,t]=True
            opened.append(i)
    return records, opened


def metric_for_score(native_payload: dict, metas: list[dict], score: np.ndarray, threshold: float, cache: Path, ajrd_json: Path):
    records, opened = build_open_records(native_payload, metas, score, threshold)
    payload = dict(native_payload)
    payload['records'] = records
    payload['model_name'] = f'v7b1_damage_aware_thr{threshold:.4f}'
    torch.save(payload, cache)
    std = eval_records(records)
    aj = run_ajrd(cache, ajrd_json)
    return {**std, 'AJ_RD': aj['true_AJ_RD'], 'AJ_RD_256': aj['true_AJ_RD_256'], 'opened': len(opened), 'cache': str(cache)}, opened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(DEFAULT_FEATURES))
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--damage-top-k', type=int, default=60)
    ap.add_argument('--lambdas', default='0,0.25,0.5,0.75,1.0,1.25,1.5,2.0')
    ap.add_argument('--max-thresholds-per-lambda', type=int, default=14)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.features, allow_pickle=True)
    X = np.asarray(z['X_aug'], np.float32)
    names = [str(n) for n in z['feature_names_aug'].tolist()]
    metas = [json.loads(str(m)) for m in z['meta_json'].tolist()]
    groups = np.asarray([m['video_id'] for m in metas], object)
    y_benefit = np.asarray(z['y_reentry_early4'], np.float32).astype(bool)
    native = torch.load(args.native_cache, map_location='cpu', weights_only=False)
    y_damage, y_safe, damage_stats = build_damage_labels(native, metas)

    benefit_idx = component_indices(names)
    damage_candidates = sorted(set(component_indices(names) + product_indices(names)))
    benefit_oof, damage_oof, fold_info = fit_oof_models(X, y_benefit, y_damage, groups, benefit_idx, damage_candidates, args.damage_top_k)
    np.save(outdir/'benefit_oof_early4_all_components.npy', benefit_oof)
    np.save(outdir/'damage_oof_bad_topk.npy', damage_oof)

    native_std = eval_records(native['records'])
    native_aj = run_ajrd(Path(args.native_cache), outdir/'native_ref_ajrd.json')
    native_row = {**native_std, 'AJ_RD': native_aj['true_AJ_RD'], 'AJ_RD_256': native_aj['true_AJ_RD_256']}

    lambdas = [float(x) for x in args.lambdas.split(',') if x.strip()]
    all_sweeps = []
    for lam in lambdas:
        decision = benefit_oof - lam * damage_oof
        qs = np.linspace(0.50, 0.995, args.max_thresholds_per_lambda)
        thresholds = np.unique(np.quantile(decision, qs)).astype(float)
        # Also include zero for benefit-minus-damage score when relevant.
        thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.0], dtype=float)]))
        for thr in thresholds:
            accept = decision >= thr
            if int(accept.sum()) <= 0:
                continue
            cache = outdir / f'v7b1_lam{lam:.2f}_thr{thr:.5f}.pt'
            ajrd_json = outdir / f'v7b1_lam{lam:.2f}_thr{thr:.5f}_ajrd.json'
            metric, opened = metric_for_score(native, metas, decision, float(thr), cache, ajrd_json)
            delta = {k: metric[k] - native_row[k] for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
            pred = accept
            pr, rc, f1, _ = precision_recall_fscore_support(y_benefit, pred, average='binary', zero_division=0)
            dmg_rate = float(y_damage[pred].mean()) if pred.sum() else 0.0
            safe_rate = float(y_safe[pred].mean()) if pred.sum() else 0.0
            all_sweeps.append({
                'lambda': lam,
                'threshold': float(thr),
                'accept_count': int(pred.sum()),
                'benefit_precision': float(pr),
                'benefit_recall': float(rc),
                'benefit_f1': float(f1),
                'accepted_damage_rate': dmg_rate,
                'accepted_safe16_rate': safe_rate,
                'metric': metric,
                'delta_vs_native': delta,
            })

    constrained = [s for s in all_sweeps if s['delta_vs_native']['AJ'] >= -0.10 and s['delta_vs_native']['OA'] >= -0.10]
    success = [s for s in constrained if s['delta_vs_native']['AJ_RD'] >= 0.0015 and s['delta_vs_native']['AJ_RD_256'] >= 0.004]
    best_constrained = sorted(constrained, key=lambda s: (s['delta_vs_native']['AJ_RD_256'], s['delta_vs_native']['AJ_RD'], s['delta_vs_native']['AJ']), reverse=True)[:20]
    best_raw = sorted(all_sweeps, key=lambda s: (s['delta_vs_native']['AJ_RD_256'], s['delta_vs_native']['AJ_RD']), reverse=True)[:20]

    summary = {
        'script': 'scripts/train_cotracker3_online_v7b1_damage_aware_raw_visconf.py',
        'features': str(args.features),
        'native_cache': str(args.native_cache),
        'n_samples': int(len(y_benefit)),
        'benefit_pos': int(y_benefit.sum()),
        'benefit_rate': float(y_benefit.mean()),
        'damage_stats': damage_stats,
        'benefit_oof_ap': float(average_precision_score(y_benefit, benefit_oof)),
        'benefit_oof_auc': float(roc_auc_score(y_benefit, benefit_oof)),
        'damage_oof_ap': float(average_precision_score(y_damage, damage_oof)),
        'damage_oof_auc': float(roc_auc_score(y_damage, damage_oof)),
        'safe_oof_ap_from_negative_damage': float(average_precision_score(y_safe, -damage_oof)),
        'native_row': native_row,
        'fold_info': fold_info,
        'lambdas': lambdas,
        'sweeps': all_sweeps,
        'best_constrained': best_constrained,
        'best_raw': best_raw,
        'success_count': len(success),
        'success_rows': success[:20],
        'benefit_feature_count': len(benefit_idx),
        'damage_candidate_count': len(damage_candidates),
        'damage_top_k': args.damage_top_k,
    }
    out = outdir / 'v7b1_damage_aware_raw_visconf_summary.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({
        'out': str(out),
        'benefit_oof_ap': summary['benefit_oof_ap'],
        'benefit_oof_auc': summary['benefit_oof_auc'],
        'damage_oof_ap': summary['damage_oof_ap'],
        'damage_oof_auc': summary['damage_oof_auc'],
        'success_count': summary['success_count'],
        'best_constrained': best_constrained[:8],
        'best_raw': best_raw[:5],
    }, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
